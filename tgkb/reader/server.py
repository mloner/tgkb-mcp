"""Reader MCP stdio server — read-only query surface over the shared store (Story 4.3).

The reader process AD-1: it shares ONLY the SQLite file with the writer and
imports from ``tgkb.common`` alone (never ``tgkb.writer``). It is launched by
Claude Code (or another MCP client) over the stdio transport, registered via the
MCP-config snippet documented in the README.

What this module does (Story 4.3):

- Opens the shared DB **read-only** at startup via the ``common/`` seam
  :func:`tgkb.common.schema.open_reader_with_fts` (AD-2 two-layer read-only +
  ``ru_lemma`` registered), inside a FastMCP ``lifespan`` async context manager,
  and closes it on shutdown.
- Evaluates the tokenizer fingerprint ONCE at startup
  (:func:`tgkb.common.schema.fingerprint_matches`) and stashes the resulting
  ``search_ok`` bool. On drift it REFUSES to serve search (it cannot repair —
  that is the writer's job, AD-2/AD-11); this story surfaces ``search_ok`` and
  warns on stderr. The real ``search_messages`` gate arrives in Story 4.4.
- Keeps **stdout** clean for the JSON-RPC protocol frames: every human-facing
  diagnostic goes to **stderr** via the ``tgkb`` logger (AC#6). No bare
  ``print`` anywhere in ``reader/``.
- Exposes ONE minimal read-only ``server_status`` health tool proving the wiring
  (real ``SELECT count(*)`` wrapped in :func:`tgkb.common.db.reading`), replacing
  the old ``ping`` stub. No ``search_messages`` (4.4) / ``read_query`` (4.5).

Uses the OFFICIAL ``mcp`` SDK (``mcp.server.fastmcp.FastMCP``), NOT the
third-party ``fastmcp`` package. No AI/LLM/embedding dependency or key anywhere
(NFR-2 / AD-7): the reader imports only ``mcp`` + stdlib + ``tgkb.common``.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
import apsw
from mcp.server.fastmcp import Context, FastMCP

from tgkb.common import VERSION, db, schema

logger = logging.getLogger("tgkb")

# --- Concurrency template (Task 1 — resolves the two deferred story-4.3 findings) ---
# 4.3 handed the reader a SINGLE shared read-only connection (``AppContext.conn``)
# and a *sync* ``@mcp.tool`` shape. Two deferred concurrency findings were gated to
# be resolved HERE, before the first real query tool (deferred-work.md#story-4.3):
#
#   (a) event-loop blocking — the ``mcp`` SDK dispatches a sync ``@mcp.tool`` with a
#       bare ``return fn(...)`` on the asyncio event-loop thread, so a heavy query
#       written sync would stall the server's I/O for the whole query; and
#   (b) shared-connection ``BEGIN`` race — two overlapping ``db.reading`` ``BEGIN``s
#       on the one APSW connection raise "cannot start a transaction within a
#       transaction". Latent under the v1 single-consumer stdio envelope, but the
#       moment a tool offloads to a thread OR a second request arrives, it breaks.
#
# The safe minimal template (the one Story 4.5 copies): the ``@mcp.tool`` body is
# ``async`` and offloads the blocking DB read to a worker thread via
# ``anyio.to_thread.run_sync`` (fixes (a) — the event loop stays free), and ALL DB
# access is serialized under this module-level lock held across the whole
# ``db.reading`` block (fixes (b) — no two ``BEGIN``s overlap on the one connection).
# We do NOT introduce a connection pool / per-request connections in this story:
# that would require re-verifying the read-only + ``ru_lemma`` + drift-gate wiring
# per connection (a larger change); the lock is the smaller, verifiable step. The
# reads stay provably short (an index-backed ``MATCH`` + a bounded citation
# projection), so AD-3's short-read-transaction rule is not weakened — the lock is
# never held across queries, only for the duration of one query's ``db.reading``.
_DB_LOCK = threading.Lock()

# Citation context depth (AD-9 / PRD OQ#3): the number of neighbor messages on
# EACH side of a hit, within the same chat, ordered by (chat_id, message_id). This
# is a tunable, NOT an invariant (ARCHITECTURE-SPINE#Deferred) — a small sensible
# default, overridable per call via the tool's ``context`` parameter.
_DEFAULT_CONTEXT_DEPTH = 2

# Default result cap (NFR-4): bound top-N so a broad query never floods the caller
# at target scale. Overridable per call via the tool's ``limit`` parameter.
_DEFAULT_LIMIT = 20

# Hard upper bounds clamped at the tool boundary so a caller-supplied ``limit`` /
# ``context`` cannot bypass NFR-4's result cap or degenerate the neighbor window
# into a whole-chat scan while the query holds ``_DB_LOCK`` (the "reads stay
# provably short" premise the lock relies on). A request above the cap is clamped
# down, never rejected.
_MAX_LIMIT = 200
_MAX_CONTEXT_DEPTH = 25


class ReadQueryRejected(RuntimeError):
    """Raised when ``read_query`` refuses a query at the Layer-1 tool boundary (AC#2).

    A REFUSAL raised in code BEFORE the SQL ever touches the connection (FR-12 /
    NFR-3 / AD-2: "rejected at the tool boundary, not merely by convention"). It is
    deliberately distinct from an empty-match return (``[]``, AC#5): ``[]`` means "a
    valid read-only query matched nothing", whereas this means "the input is not a
    single provably-read-only statement and was never executed". The message names
    WHY (a write/DDL keyword, a multi-statement string, an un-strippable comment)
    without leaking a secret or a raw ``apsw`` traceback. Mirrors the
    :class:`SearchIndexStaleError` error-class + ``reader/__init__`` export pattern.
    """


class SearchIndexStaleError(RuntimeError):
    """Raised when ``search_messages`` is called against a stale FTS index (AC#6).

    A REFUSAL, deliberately distinct from the empty-match return (an empty list,
    AC#5). ``search_ok=False`` means the persisted tokenizer fingerprint disagrees
    with the running ``ru_lemma`` fingerprint (AD-11 drift): index lemmas and query
    lemmas no longer match, so recall may have silently collapsed. Returning an
    empty list here would be a LIE ("no matches") when the truth is "I cannot look
    correctly" — so the tool raises this instead. The reader NEVER repairs (repair
    is a write → writer-only, AD-2/AD-11); the message names the fix (run the
    writer).
    """


def _ensure_stderr_logging() -> None:
    """Attach a stderr handler to the ``tgkb`` logger if it has none (AC#6).

    Re-typed (NOT imported — AD-1 forbids ``reader`` → ``writer``) from the
    writer's ``cli._ensure_console_logging``. For the reader the stderr
    discipline is stricter: on the stdio transport **stdout carries the MCP
    JSON-RPC frames**, so any human-facing byte on stdout corrupts the protocol.
    All diagnostics therefore go to stderr. Idempotent: does nothing if a handler
    is already present, so tests that install ``caplog`` are unaffected.
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        # Stop records propagating to the root logger: if a root handler is
        # pointed at stdout (or is added later), a propagated `tgkb` record would
        # land on stdout and corrupt the MCP JSON-RPC frame (AC#6). Our own
        # stderr handler above is the sole sink for reader diagnostics.
        logger.propagate = False


@dataclass
class AppContext:
    """Startup resources shared with every request via the FastMCP lifespan.

    ``conn`` is the SINGLE read-only connection held open for the process
    lifetime (AD-2/AD-3: one connection; each query wraps its read in the SHORT
    :func:`tgkb.common.db.reading` transaction, never one perpetual open txn).
    ``search_ok`` is the tokenizer-drift verdict evaluated ONCE at startup: when
    ``False`` the index is stale and search must be REFUSED (the reader cannot
    repair — AD-11); Story 4.4's ``search_messages`` reads this gate.
    """

    conn: apsw.Connection
    search_ok: bool


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Open the read-only DB, evaluate drift, yield the context, close on shutdown.

    Startup (stdio runs the lifespan on startup):

    1. resolve the DB path via :func:`tgkb.common.db.default_db_path` — the SAME
       path the writer uses (the shared-file channel of AD-1), single-sourced;
    2. open read-only via :func:`tgkb.common.schema.open_reader_with_fts` (NOT
       ``db.open_reader`` directly — the seam also registers ``ru_lemma``, needed
       for any later ``MATCH``). The reader MUST NOT create the DB: an absent DB
       raises here, which we translate to a clear secret-free stderr message and
       let propagate (fail startup cleanly) rather than dumping a raw traceback;
    3. evaluate :func:`tgkb.common.schema.fingerprint_matches` ONCE → ``search_ok``.
       On drift, the server still STARTS and stays up (Claude Code launched it; a
       hard crash gives no diagnosable signal), but logs a one-line stderr warning
       and gates search off until the writer rebuilds.

    Shutdown closes the connection.
    """
    _ensure_stderr_logging()
    db_path: Path = db.default_db_path()

    try:
        conn = schema.open_reader_with_fts(db_path)
    except apsw.CantOpenError:
        # A reader must never bring a DB into existence (that is the writer's
        # job). Surface a clear, secret-free instruction and fail startup — do
        # NOT fall back to a writable open. The path is local state, not a
        # secret (NFR-1: no api_hash / session / passphrase / message text).
        logger.error(
            "no knowledge base found at %s; run `tgkb export` / `tgkb sync` "
            "first to build it",
            db_path,
        )
        raise
    except (apsw.Error, ValueError):
        # Any OTHER open failure — a corrupt/partial file (apsw.NotADBError),
        # a lock outlasting busy_timeout, a permission denial, or a non-local
        # path rejected by db._reject_non_local (ValueError) — must also fail
        # startup with a clear, secret-free message naming only the path, never
        # a raw traceback (NFR-1: the path is local state, not a secret).
        logger.error(
            "could not open the knowledge base at %s; it may be missing, "
            "unreadable, or corrupt — run `tgkb export` / `tgkb sync` to "
            "(re)build it",
            db_path,
        )
        raise

    try:
        search_ok = schema.fingerprint_matches(conn)
        if not search_ok:
            logger.warning(
                "search index is stale (tokenizer drift); run the writer "
                "(`tgkb sync` / `tgkb export`) to rebuild before searching"
            )
        yield AppContext(conn=conn, search_ok=search_ok)
    finally:
        conn.close()


mcp = FastMCP("tgkb-reader", lifespan=app_lifespan)


def _server_status(ctx: AppContext) -> dict[str, object]:
    """Compute the honest read-only status dict (the tool's testable core).

    Split out so tests can assert the REAL behavior (a real ``SELECT count(*)``
    over a real connection, wrapped in the SHORT :func:`tgkb.common.db.reading`
    transaction) without standing up a full MCP request context. The
    ``@mcp.tool`` wrapper below only reaches into the lifespan context and
    delegates here.

    Serialized under ``_DB_LOCK`` (Task 1): this probe shares the single
    ``AppContext.conn`` with ``search_messages``, and since ``search_messages`` now
    offloads its read to a worker thread, a ``server_status`` call on the event
    loop could otherwise open a second overlapping ``BEGIN`` on the one APSW
    connection → ``SQLError: cannot start a transaction within a transaction``. The
    lock makes ALL readers on the shared connection mutually exclusive, not just
    search-vs-search. The read is a trivial ``count(*)``, so it stays short.
    """
    with _DB_LOCK:
        with db.reading(ctx.conn) as conn:
            message_count = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    return {
        "server": "tgkb-reader",
        "version": VERSION,
        "message_count": message_count,
        "search_ok": ctx.search_ok,
        "read_only": True,
    }


@mcp.tool()
def server_status(ctx: Context) -> dict[str, object]:
    """Report read-only health: message count, search availability, read-only flag.

    A minimal introspection/health probe (NOT a query surface) proving the server
    is wired to the real DB read-only. Reaches the lifespan connection via the
    injected FastMCP ``Context`` and returns a small honest status dict. The real
    query tools (``search_messages`` 4.4, ``read_query`` 4.5) are out of scope.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    return _server_status(app_ctx)


# --- search_messages (Story 4.4) --------------------------------------------
#
# The AD-9 citation projection. We recover the full message row + its chat by
# ``messages.rowid`` (the pinned rowid ↔ (chat_id, message_id) mapping, AD-12 —
# the same rowid ``search_message_rowids`` returns) and join ``chats`` for the
# citation's ``chat`` title and the ``t.me`` link basis (``is_public`` + ``username``).
# Tombstones are excluded HERE (``m.deleted = 0``, AC#3) — ``search_message_rowids``
# does NOT filter them. One rowid per parametrized ``?`` and BM25 rank order is
# preserved by the caller iterating the recipe's rowids in order and issuing one
# ``fetchone()`` each (NO ``IN`` clause, so nothing to re-sort). The read stays
# index-backed on the PK/rowid (NFR-4). We do NOT re-implement the MATCH+bm25 SQL
# — that is ``common/``'s.
#
# ``LEFT JOIN`` (not inner): a live hit whose ``chat`` row is missing (no FK on
# ``messages.chat_id`` in v1, so an ingest gap is possible) must still surface as
# a REAL hit with a null ``chat``/``link`` — NOT silently vanish as if tombstoned.
# The ``m.deleted = 0`` filter is what excludes tombstones; a missing chat row is
# an integrity gap, not a deletion, and hiding it would lose a real message.
_CITATION_BY_ROWID_SQL = """
SELECT m.rowid, m.chat_id, m.message_id, m.sender_name, m.text, m.date_utc,
       c.title, c.username, c.is_public
FROM messages m
LEFT JOIN chats c ON c.chat_id = m.chat_id
WHERE m.rowid = ? AND m.deleted = 0
"""

# Neighbor (context) lookup: messages from the SAME chat within a message_id
# window around the hit, tombstones excluded (AC#3 — never surface deleted
# content as context), ordered by (chat_id, message_id) (AD-9). Index-backed by
# the composite PRIMARY KEY (chat_id, message_id) auto-index — no new index
# needed (and the reader could not create one anyway, AD-2). The hit itself is
# excluded from its own context (message_id <> ?).
_CONTEXT_NEIGHBORS_SQL = """
SELECT message_id, sender_name, text, date_utc
FROM messages
WHERE chat_id = ?
  AND message_id BETWEEN ? AND ?
  AND message_id <> ?
  AND deleted = 0
ORDER BY chat_id, message_id
"""


def _build_link(
    username: str | None, is_public: int | None, message_id: int
) -> str | None:
    """The AD-9 ``t.me`` deep-link, or ``None`` for a private / unlinkable chat.

    A chat is linkable ONLY when it has a public username: the writer sets
    ``is_public = bool(username)`` ([export.py:277]). For a public chat the
    message deep-link is ``https://t.me/<username>/<message_id>``. When the chat is
    private / has no username the link is ``None`` (NULL) — NEVER a fabricated URL
    (AC#2). Both signals are required (defensive against an inconsistent row, and
    against a ``LEFT JOIN`` null when the ``chat`` row is missing entirely — both
    ``is_public`` and ``username`` are ``None`` then, so no link is fabricated).
    """
    if is_public and username:
        return f"https://t.me/{username}/{message_id}"
    return None


def _context_for(
    conn: apsw.Connection, chat_id: int, message_id: int, depth: int
) -> list[dict[str, Any]]:
    """Fetch up to ``depth`` non-tombstoned neighbors on each side of a hit (AD-9).

    Neighbors are from the SAME ``chat_id``, within ``[message_id-depth,
    message_id+depth]`` (the hit excluded), ordered by ``(chat_id, message_id)``.
    ``depth <= 0`` means "no context" → an empty list (no query). The window is a
    superset of ``depth`` neighbors per side (message_ids are not necessarily
    contiguous), so it never returns MORE than ``2*depth`` but may return fewer —
    the bound is honoured. Runs on the caller's already-open read transaction.
    """
    if depth <= 0:
        return []
    rows = conn.execute(
        _CONTEXT_NEIGHBORS_SQL,
        (chat_id, message_id - depth, message_id + depth, message_id),
    ).fetchall()
    return [
        {"message_id": mid, "sender": sender, "text": text, "date": date_utc}
        for (mid, sender, text, date_utc) in rows
    ]


def _normalize_query(query: str) -> str:
    """Collapse surrounding/internal whitespace to a single-spaced phrase (Task 4).

    Query hygiene at the tool boundary (AD-9): ``search_message_rowids`` passes the
    string RAW to FTS5, so the tool normalizes here. Returns ``""`` for an
    empty/whitespace-only input (the caller maps that to the explicit empty list,
    AC#5 — a no-match, never an ``apsw.SQLError``).
    """
    return " ".join(query.split())


def _fts_terms(query: str) -> str:
    """Quote each whitespace token as its own FTS5 string literal (Task 4).

    FTS5 query metacharacters (``"`` ``*`` ``:`` and the bare keywords
    ``AND``/``OR``/``NOT``/``NEAR``) would otherwise be interpreted as query
    syntax — or raise ``apsw.SQLError`` — turning a user phrase into an operator
    expression or a crash. Each token is wrapped in FTS5 double-quotes (a literal
    ``"`` inside a token is escaped by doubling it, the FTS5 string-literal
    convention), so every token matches literally and no metacharacter is honoured.

    Tokens are joined by spaces, which FTS5 reads as an implicit **AND** — so a
    multi-word query "бюджет совещание" matches messages that contain BOTH lemmas
    (term-AND), NOT only messages where they appear adjacent and in order. Wrapping
    the WHOLE query as one ``"…"`` phrase (the earlier shape) silently required
    adjacency and returned nothing for ordinary multi-word queries — the recall
    regression this fixes. Each token is still lemmatized by ``ru_lemma`` at
    tokenization, so morphology (AC#4) is unaffected.

    The caller guarantees ``query`` is already whitespace-normalized and non-empty
    (:func:`_normalize_query`), so ``query.split()`` yields at least one token.
    """
    return " ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in query.split())


def _search_messages(
    app_ctx: AppContext,
    query: str,
    limit: int = _DEFAULT_LIMIT,
    context: int = _DEFAULT_CONTEXT_DEPTH,
) -> list[dict[str, Any]]:
    """Citation-complete full-text search — the tool's testable, blocking core.

    Split from the thin ``@mcp.tool`` wrapper (mirroring 4.3's ``_server_status`` /
    ``server_status`` split) so tests assert the REAL behavior against a REAL
    connection without a full MCP transport. Steps:

    1. **Stale-index refusal gate FIRST (AC#6).** If ``app_ctx.search_ok`` is False
       (tokenizer drift, evaluated once at 4.3 startup), RAISE
       :class:`SearchIndexStaleError` — a refusal, NOT an empty list, NOT a repair.
    2. **Query hygiene + bound clamping (AC#5, Task 4).** Normalize whitespace; an
       empty/whitespace query short-circuits to ``[]`` (a no-match, never a crash).
       Each token is FTS5-quoted (term-AND) so metacharacters cannot raise or change
       the query semantics. ``limit``/``context`` are clamped to hard maxima.
    3. **Ranked MATCH (AC#1).** Reuse ``schema.search_message_rowids`` for
       BM25-ranked ``messages.rowid``s (index-backed, never a full scan). We do NOT
       re-derive its SQL. Because tombstones are filtered AFTER the MATCH (below),
       we OVER-FETCH rowids and stop once ``limit`` LIVE hits are collected — a
       plain SQL ``LIMIT`` here would let tombstones in the top-N silently shrink
       the live result below ``limit`` even when more live matches exist.
    4. **Tombstone exclusion + AD-9 projection (AC#2, #3).** For each rowid in rank
       order, project the citation (LEFT JOIN ``chats``), skipping tombstoned rows,
       building the ``t.me`` link and the ordered context, until ``limit`` reached.
    5. **Empty-set correct (AC#5).** No surviving hit → ``[]``.

    Read-only + short-transaction (AC#7, AD-3): the whole DB interaction runs in ONE
    short :func:`tgkb.common.db.reading` transaction, serialized under ``_DB_LOCK``
    (Task 1). No write, no DDL, no checkpoint anywhere.
    """
    # 1. Refuse on a stale index — a REFUSAL, distinct from the empty-match return.
    if not app_ctx.search_ok:
        raise SearchIndexStaleError(
            "search index is stale (tokenizer drift); cannot serve search — "
            "run the writer (`tgkb sync` / `tgkb export`) to rebuild the index"
        )

    # 2. Hygiene + clamp. Empty/whitespace-only query is an honest no-match, not a
    # crash. A non-positive limit means "no rows" (mirrors the seam's guard); any
    # positive limit/context is clamped down to the hard cap so a huge value cannot
    # flood the caller or hold the lock on a whole-chat scan.
    if limit <= 0:
        return []
    limit = min(limit, _MAX_LIMIT)
    context = min(max(context, 0), _MAX_CONTEXT_DEPTH)
    normalized = _normalize_query(query)
    if not normalized:
        return []
    match_expr = _fts_terms(normalized)

    # 3–5, all under ONE short read transaction serialized by the module lock so
    # overlapping BEGINs cannot occur on the single shared connection (Task 1).
    with _DB_LOCK:
        with db.reading(app_ctx.conn) as conn:
            try:
                # Over-fetch: tombstones are filtered below, so a SQL LIMIT here
                # would under-return. Fetch ALL matching rowids (BM25-ranked) and
                # stop the projection once `limit` LIVE hits are collected. The
                # MATCH itself is index-backed (NFR-4); the projection is bounded
                # by `limit` regardless of how many rowids the MATCH returns.
                rowids = schema.search_message_rowids(conn, match_expr, limit=None)
            except apsw.SQLError as exc:
                # Only a MALFORMED-QUERY FTS5 error is mapped to a no-match (Task 4:
                # never leak a raw traceback for bad query syntax). Any OTHER
                # SQLError (a corrupt/locked index, a real DB fault) is re-raised —
                # masking it as `[]` would report an honest failure as "no matches"
                # and violate the AC#6 honesty contract. FTS5 syntax errors carry
                # "fts5:" / "malformed MATCH" in their message.
                text = str(exc).lower()
                if "fts5" in text or "malformed match" in text or "syntax error" in text:
                    return []
                raise

            results: list[dict[str, Any]] = []
            for rowid in rowids:
                if len(results) >= limit:
                    break
                row = conn.execute(_CITATION_BY_ROWID_SQL, (rowid,)).fetchone()
                if row is None:
                    # Tombstoned (deleted=1): the `WHERE m.deleted = 0` filter drops
                    # it, so the hit is excluded (AC#3). A missing chat row no longer
                    # produces None (LEFT JOIN) — a live orphan-chat hit still
                    # surfaces, with a null chat/link, rather than vanishing.
                    continue
                (
                    _rowid,
                    chat_id,
                    message_id,
                    sender_name,
                    text,
                    date_utc,
                    title,
                    username,
                    is_public,
                ) = row
                results.append(
                    {
                        "text": text,
                        "sender": sender_name,
                        "date": date_utc,
                        "chat": title,
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "link": _build_link(username, is_public, message_id),
                        "context": _context_for(conn, chat_id, message_id, context),
                    }
                )
            return results


@mcp.tool()
async def search_messages(
    ctx: Context,
    query: str,
    limit: int = _DEFAULT_LIMIT,
    context: int = _DEFAULT_CONTEXT_DEPTH,
) -> list[dict[str, Any]]:
    """Ranked full-text search over stored messages, returning cited results (AD-9).

    Runs an FTS5 ``MATCH`` via the shared ``ru_lemma`` tokenizer (so an inflected
    Russian query like "договор" finds "договоры/договора/договором" — FR-7) and
    returns matching messages BM25-ranked, most-relevant first. Each result is a
    citation-complete dict: ``text``, ``sender``, ``date``, ``chat``, ``chat_id``,
    ``message_id``, a ``t.me`` ``link`` (or ``null`` for a private chat), and a
    small list of surrounding ``context`` messages ordered by ``(chat_id,
    message_id)``. Tombstoned (deleted) messages are excluded from hits and context.

    A query matching nothing returns an explicit empty list ``[]`` (never an error,
    never a placeholder row). If the search index is stale (tokenizer drift) the
    tool REFUSES to serve (raises a clear error naming the fix) rather than
    returning an empty list — a stale index cannot honestly answer "no matches".

    ``limit`` caps the number of hits (default ``20``); ``context`` sets how many
    neighbor messages to attach on each side of a hit (default ``2``).

    Answer ONLY from the returned messages. Every returned row is a real stored
    message with a citation — ground each claim in a returned message and cite it
    (chat, sender, date, and the ``t.me`` link when present). Do NOT answer from
    prior knowledge and do NOT invent messages. If the returned list is empty,
    there are no matching messages — say so plainly rather than fabricating an
    answer.

    The tool is READ-ONLY: it performs a single short read transaction and writes
    nothing. The blocking DB read is offloaded to a worker thread so the server's
    event loop stays responsive (Task 1).
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    return await anyio.to_thread.run_sync(
        _search_messages, app_ctx, query, limit, context
    )


# --- read_query (Story 4.5) -------------------------------------------------
#
# The guarded structured-lookup surface (FR-12, NFR-3, AD-2). Unlike
# ``search_messages`` (a FIXED full-text tool), ``read_query`` accepts an
# arbitrary caller SQL string and executes it ONLY if a Layer-1 statement
# classifier proves it is a single, provably-read-only SELECT-class statement —
# otherwise it is REJECTED at the tool boundary, in code, before it touches the
# connection (``ReadQueryRejected``). The read-only connection (Story 1.2's
# ``open_reader``: ``SQLITE_OPEN_READONLY`` + ``query_only=ON``) stands as the
# INDEPENDENT Layer-2 backstop (AC#3) — neither layer alone is the guarantee.
#
# The classifier is ALLOWLIST-based and FAIL-CLOSED (AC#2, #4): after stripping
# SQL comments (string-literal aware), the input must be a SINGLE statement whose
# leading keyword is one of {SELECT, WITH, EXPLAIN}; anything else — a write, DDL,
# a multi-statement string, a data-modifying CTE, or anything we cannot prove safe
# — is rejected. We never enumerate a blocklist of "bad" keywords (bypassable);
# we allow only a proven-safe shape and reject the rest.

# Leading keywords we can prove are read-only. SELECT and EXPLAIN cannot mutate;
# WITH is allowed only after we further prove the CTE body contains no writing
# statement (see ``_classify_read_only``). Everything else → reject (fail closed).
_ALLOWED_LEADING_KEYWORDS = frozenset({"SELECT", "WITH", "EXPLAIN"})

# Data-modifying keywords that make a WITH…(INSERT/UPDATE/DELETE) CTE a WRITE even
# though it starts with the allowlisted ``WITH`` token. A CTE like
# ``WITH x AS (DELETE FROM messages RETURNING *) SELECT * FROM x`` is legal SQL and
# WRITES — so if the comment-stripped statement contains any of these as a word we
# reject the whole thing (fail closed; we do not attempt to prove the write is
# unreachable). SELECT-only CTEs never contain these tokens.
_CTE_WRITE_KEYWORDS = frozenset(
    {"INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE"}
)


def _strip_sql_comments(sql: str) -> str:
    """Remove ``--``-to-EOL and ``/* … */`` comments, string-literal aware (AC#4).

    A naive strip would corrupt a legitimate ``SELECT '--'`` or ``SELECT '/*'`` (a
    comment marker INSIDE a string literal is data, not a comment) — and, worse,
    could expose or hide a real leading keyword. We walk the string once, tracking
    whether we are inside a ``'…'`` / ``"…"`` literal (SQLite escapes an embedded
    quote by DOUBLING it), and only treat ``--`` / ``/* */`` as comments when
    OUTSIDE a literal. If we cannot parse cleanly (an unterminated string or block
    comment), we raise :class:`ReadQueryRejected` — fail closed, never guess.

    Comments are replaced with a single space so they cannot glue two tokens (or
    two statements) together after removal.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        # String / quoted-identifier literal: copy verbatim until the matching
        # (un-doubled) closing quote. Doubled quote = escaped quote = stays inside.
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        out.append(quote * 2)
                        i += 2
                        continue
                    out.append(quote)
                    i += 1
                    break
                out.append(sql[i])
                i += 1
            else:
                raise ReadQueryRejected(
                    "read_query rejected: unterminated string literal in the query"
                )
            continue
        # Line comment: -- … to end of line.
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
            continue
        # Block comment: /* … */ (SQLite block comments do NOT nest).
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            i += 2
            closed = False
            while i + 1 < n:
                if sql[i] == "*" and sql[i + 1] == "/":
                    i += 2
                    closed = True
                    break
                i += 1
            if not closed:
                raise ReadQueryRejected(
                    "read_query rejected: unterminated block comment in the query"
                )
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_single_statement(sql_no_comments: str) -> bool:
    """True iff the comment-stripped SQL is exactly ONE statement (AC#4).

    We reject any input carrying a SECOND statement so a benign leading
    ``SELECT 1`` cannot smuggle a trailing ``; DROP TABLE messages`` past the
    guard. APSW's ``Connection.execute`` would happily run every ``;``-separated
    statement — so we must never hand it a multi-statement string. Rule: a ``;`` is
    allowed ONLY as trailing terminator (optionally followed by whitespace); any
    ``;`` with non-whitespace after it introduces a second statement → not single.
    Comments are already stripped but string literals are preserved, so we scan
    literal-aware here too — a ``;`` inside ``'…'`` / ``"…"`` is data, not a
    separator.
    """
    i = 0
    n = len(sql_no_comments)
    while i < n:
        ch = sql_no_comments[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n:
                if sql_no_comments[i] == quote:
                    if i + 1 < n and sql_no_comments[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == ";":
            # Anything other than trailing whitespace after a ';' is a 2nd statement.
            if sql_no_comments[i + 1 :].strip() != "":
                return False
        i += 1
    return True


def _classify_read_only(sql: str) -> str:
    """Prove ``sql`` is a single read-only statement, else raise (AC#1, #2, #4).

    The Layer-1 statement guard — the security invariant of Story 4.5. Runs BEFORE
    any ``conn.execute``. Steps (all fail-closed):

    1. Strip comments (string-literal aware); reject an un-strippable input.
    2. Reject an empty/whitespace-only statement (nothing to run).
    3. Reject a multi-statement string (only ONE statement per call).
    4. The leading keyword MUST be in the allowlist {SELECT, WITH, EXPLAIN}.
    5. For ``WITH`` (a CTE), reject if the body contains any data-modifying
       keyword (a ``WITH … DELETE … RETURNING`` CTE writes) — fail closed.

    Returns the comment-stripped SQL to execute (so the executed text is exactly
    what was classified — no gap between what we proved and what we run). Raises
    :class:`ReadQueryRejected` with a clear, secret-free reason on any failure.
    """
    stripped = _strip_sql_comments(sql).strip()
    if not stripped:
        raise ReadQueryRejected(
            "read_query rejected: empty query (a single read-only SELECT-class "
            "statement is required)"
        )
    if not _is_single_statement(stripped):
        raise ReadQueryRejected(
            "read_query rejected: multiple statements are not allowed — submit a "
            "single read-only SELECT-class statement per call"
        )

    # Leading keyword (word chars only) — the allowlist gate.
    match = re.match(r"[A-Za-z_]+", stripped)
    leading = (match.group(0) if match else "").upper()
    if leading not in _ALLOWED_LEADING_KEYWORDS:
        raise ReadQueryRejected(
            f"read_query accepts a single read-only SELECT-class statement "
            f"(SELECT / WITH / EXPLAIN); '{leading or stripped[:16]}' is not "
            f"read-only and was rejected"
        )

    # A WITH-CTE that carries a data-modifying statement writes — reject it.
    if leading == "WITH":
        words = set(re.findall(r"[A-Za-z_]+", stripped.upper()))
        writers = words & _CTE_WRITE_KEYWORDS
        if writers:
            raise ReadQueryRejected(
                "read_query rejected: the CTE contains a data-modifying statement "
                f"({', '.join(sorted(writers))}); only a read-only WITH … SELECT "
                "is allowed"
            )

    return stripped


def _read_query(
    app_ctx: AppContext,
    sql: str,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Guarded read-only SQL lookup — the tool's testable, blocking core (AC#1–#7).

    Split from the thin ``@mcp.tool`` wrapper (mirroring the ``_search_messages`` /
    ``search_messages`` split) so tests assert the REAL behavior against a REAL
    connection without a full MCP transport. Steps:

    1. **Layer-1 classify FIRST (AC#1, #2, #4).** :func:`_classify_read_only`
       proves ``sql`` is a single provably-read-only statement BEFORE any
       ``conn.execute``; on failure it raises :class:`ReadQueryRejected` — a clear
       tool error, never a raw traceback, never a silent no-op. This is the whole
       point of the story (rejection at the tool boundary, in code — not merely by
       the read-only connection).
    2. **``limit`` clamp (AC#7).** ``limit <= 0`` → ``[]``; a positive limit is
       clamped down to ``_MAX_LIMIT`` so a broad ``SELECT * FROM messages`` cannot
       flood the caller or hold ``_DB_LOCK`` on a full materialization (NFR-4).
    3. **Execute + project (AC#1, #5).** Run the classified-safe SQL on the
       read-only ``AppContext.conn`` inside ONE short :func:`tgkb.common.db.reading`
       transaction serialized under ``_DB_LOCK`` (the Story 4.4 template). Column
       names come from ``cursor.getdescription()``; each row becomes a
       column-keyed dict. Fetch at MOST ``limit`` rows (iterate + stop) rather than
       rewriting the caller's SQL. No match → ``[]``.

    Layer 2 (AC#3) is the connection itself: if the classifier were ever wrong, a
    write reaching ``app_ctx.conn`` raises ``apsw.ReadOnlyError``. This core never
    re-opens the connection, clears ``query_only``, checkpoints, or writes.
    """
    # 0. Defensive input typing. The @mcp.tool schema declares sql: str / limit: int,
    #    but the testable core is public and a client could still send null/other —
    #    turn that into a clear ReadQueryRejected rather than a raw TypeError
    #    (len(None) / None <= 0) escaping the "never a raw traceback" contract. A
    #    None limit falls back to the default; a non-int would fail the clamp below.
    if not isinstance(sql, str):
        raise ReadQueryRejected(
            "read_query rejected: 'sql' must be a string (a single read-only "
            "SELECT-class statement)"
        )
    if limit is None:
        limit = _DEFAULT_LIMIT

    # 1. Layer-1 guard — reject anything not provably a single read-only statement.
    #    Raises ReadQueryRejected BEFORE any DB touch.
    safe_sql = _classify_read_only(sql)

    # 2. Bound clamp (mirror search_messages): non-positive → no rows; huge → cap.
    if limit <= 0:
        return []
    limit = min(limit, _MAX_LIMIT)

    # 3. Execute under ONE short read transaction serialized by the module lock so
    #    overlapping BEGINs cannot occur on the single shared connection (Task 1).
    with _DB_LOCK:
        with db.reading(app_ctx.conn) as conn:
            try:
                cursor = conn.execute(safe_sql)
                # getdescription() is only valid once the statement has produced a
                # result-column shape; a statement that returns rows populates it.
                # We read column names lazily from the first fetch loop below.
                results: list[dict[str, Any]] = []
                for row in cursor:
                    if len(results) >= limit:
                        break
                    columns = [desc[0] for desc in cursor.getdescription()]
                    results.append(dict(zip(columns, row)))
            except ReadQueryRejected:
                raise
            except apsw.Error as exc:
                # ANY apsw failure while executing the classified-safe SQL — a
                # malformed but read-only query (a typo'd column / bad function →
                # apsw.SQLError), a lock outlasting busy_timeout (apsw.BusyError),
                # or — if the classifier were ever wrong — the Layer-2 read-only
                # rejection (apsw.ReadOnlyError) — is wrapped in a clear, secret-free
                # ReadQueryRejected rather than escaping as a raw apsw traceback (the
                # tool's stated contract). We catch apsw.Error (the common base), NOT
                # only apsw.SQLError: ReadOnlyError and BusyError do NOT derive from
                # SQLError, so a narrower catch would let them leak. This is NOT a
                # write rejection (raised before execution) and NOT a no-match ([]);
                # a genuine fault is still reported, never masked as "no rows".
                raise ReadQueryRejected(
                    f"read_query could not execute the query: {exc}"
                ) from exc
            return results


@mcp.tool()
async def read_query(
    ctx: Context,
    sql: str,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Run a guarded read-only SQL query and return the rows (structured lookup).

    For precise structured lookups Claude Code composes as SQL — e.g.
    ``SELECT count(*) FROM messages`` or ``SELECT chat_id, message_id, text FROM
    messages WHERE chat_id = 1 ORDER BY message_id DESC``. Each result row is a
    column-keyed dict (keyed by the query's own SELECTed column names); a query
    matching nothing returns an explicit empty list ``[]``.

    The tool is GUARDED and READ-ONLY. It accepts only a SINGLE, provably
    read-only ``SELECT`` / ``WITH … SELECT`` / ``EXPLAIN`` statement. Any write or
    DDL (``INSERT``/``UPDATE``/``DELETE``/``CREATE``/``DROP``/``ALTER``/…), a
    multi-statement string, or a comment-obfuscated write is REJECTED at the tool
    boundary with a clear error and NEVER executed. Underneath, the connection is
    itself opened read-only, so a mutation could not succeed even if the guard were
    bypassed.

    ``limit`` caps the number of returned rows (default ``20``, hard maximum
    ``200``); the result is truncated to ``limit`` rows — a caller-visible cap, not
    a silent drop of "the rest".

    Answer ONLY from the returned rows. Every returned row is a real record from
    the store — ground each claim in a returned row and do NOT answer from prior
    knowledge or invent data. An empty list means nothing matched: say there are no
    matching rows rather than fabricating an answer.

    The blocking DB read is offloaded to a worker thread so the server's event loop
    stays responsive; it runs a single short read transaction and writes nothing.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    return await anyio.to_thread.run_sync(_read_query, app_ctx, sql, limit)


def main() -> None:
    """Run the reader MCP server over the default stdio transport.

    The ``mcp`` SDK owns the stdout JSON-RPC framing; keep ``main`` minimal and
    never write to stdout here (AC#6).
    """
    _ensure_stderr_logging()
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
