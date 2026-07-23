"""Schema DDL + idempotent ingest — the single owner of table definitions (Story 2.2).

This module is the DDL layer that :mod:`tgkb.common.db` explicitly deferred to
Story 2.2 (its docstring: *"It ships NO schema/DDL — creating tables is Story
2.2 / 3.2"*). It owns the ``chats`` / ``messages`` / ``sync_state`` table
definitions (exactly the Structural Seed shape) plus the AD-4 idempotent ingest
primitive.

Architecture invariants enforced here:

- **AD-4**: message identity is ``(chat_id, message_id)`` (a composite PRIMARY
  KEY). Ingest is ``INSERT … ON CONFLICT (chat_id, message_id) DO UPDATE`` gated
  on ``edit_date`` change — re-ingesting an unchanged message is a no-op; an
  edited message updates ``text`` + ``edit_date`` only; identity never changes.
- **AD-12 (composability)**: ``ingest_*`` NEVER opens its own transaction — the
  caller wraps it in :func:`tgkb.common.db.writing`. This is non-negotiable:
  Story 3.2 must add the ``messages_fts`` write to the SAME transaction, and
  Story 4.1 must advance ``sync_state.last_id`` in the SAME scope (AD-5).
- **AD-5**: ``sync_state.last_id`` is a monotonic high-water mark — the optional
  :func:`advance_last_id` helper encodes the never-regress rule in SQL.
- **AD-1**: this module lives under ``common/`` and imports ONLY stdlib + ``apsw``
  (type hints) + :mod:`tgkb.common.db` + (Story 3.2) :mod:`tgkb.common.ru_lemma`
  (an allowed sibling ``common`` module — the shared FTS5 tokenizer). It MUST
  NEVER import ``tgkb.writer``, ``tgkb.reader``, or ``tgkb.common.config``.
- **AD-12 (FTS in the write transaction)**: Story 3.2 adds the ``messages_fts``
  external-content FTS5 table + its ``AFTER INSERT/UPDATE/DELETE`` triggers on
  ``messages``. The triggers fire synchronously inside whatever statement mutates
  ``messages`` — always :func:`tgkb.common.db.writing` (``BEGIN IMMEDIATE``) — so
  the FTS write lands in the SAME transaction as the message write BY
  CONSTRUCTION, never a separate pass. ``ingest_message`` needs NO body change.
- **AD-6 / AD-11 (single tokenizer, single-sourced spec)**: the FTS table's
  ``tokenize=`` clause is built from :data:`tgkb.common.ru_lemma.TOKENIZE_SPEC`
  (never a hand-typed literal), and :func:`tgkb.common.ru_lemma.register` is
  called inside :func:`ensure_schema` so the tokenizer is guaranteed present
  whenever the ``CREATE VIRTUAL TABLE`` runs (registration is per-connection and
  NOT persisted in the DB file). Every reader connection that runs ``MATCH`` must
  register too — see :func:`open_reader_with_fts` (the seam Stories 4.3 / 4.4
  reuse).

**Timestamp contract (AD-consistency / AC#4).** ``date_utc`` and ``edit_date``
are stored VERBATIM as the caller passes them; this layer is timezone-transparent
and does NO conversion. The pinned format is **ISO-8601 UTC**
(e.g. ``2026-07-19T14:03:00+00:00``). Converting Telethon's ``datetime`` to that
string is **Story 2.4's** job — it must pass
``dt.astimezone(timezone.utc).isoformat()``. Never store naive local time.

**Built here (Story 3.3 — persistence + the drift guard, AD-11):**

- A tiny ``meta(key, value)`` KV table (:data:`_CREATE_META`) that persists the
  tokenizer fingerprint across process runs (a KV table so future single-value
  metadata has a home without another "first schema change" — migration tooling
  is deferred, ARCHITECTURE-SPINE#Deferred). :func:`read_fingerprint` /
  :func:`write_fingerprint` are the KV accessors; the fingerprint VALUE is
  computed by :func:`tgkb.common.ru_lemma.fingerprint` (single-sourced there,
  with the tokenizer it fingerprints).
- :func:`check_and_repair_fingerprint` — the WRITER startup seam: on a mismatch
  (or a missing fingerprint over a non-empty index) it forces
  ``INSERT INTO messages_fts(messages_fts) VALUES('rebuild')`` AND re-stamps the
  current fingerprint, both inside ONE :func:`tgkb.common.db.writing` transaction
  so the rebuild and the stamp commit atomically (a crash mid-rebuild must not
  leave a stale index marked current — the exact failure AD-11 prevents).
- :func:`fingerprint_matches` — the READER startup seam (pure read, no repair):
  the reader is read-only at the SQL level (AD-2) and CANNOT run ``'rebuild'``, so
  it detects drift and Story 4.3 makes the server REFUSE to serve on ``False``.
  The split is WRITER = detect-and-repair, READER = detect-and-refuse.
- ``ensure_schema`` now stamps the current fingerprint on a fresh DB (when the
  ``meta`` row is absent), and :func:`open_writer_with_schema` calls
  :func:`check_and_repair_fingerprint` after ``ensure_schema`` so every writer run
  repairs drift before ingesting.

**Not built here (deliberate seams):**

- **The ``search_messages`` MCP tool, Citations, the read-only tool boundary,
  and tombstone filtering** — Story 4.4 (AD-9). :func:`search_message_rowids`
  here is a MINIMAL internal ``MATCH``+``bm25`` proof/recipe, NOT a reader tool,
  and does NOT filter tombstoned rows (4.4 owns that at query time).
- **The ``deleted`` tombstone↔FTS interaction** — ``messages`` rows are never
  hard-deleted in v1 (AD-13: deletions are a ``deleted`` flag flip, Story 4.2).
  A future ``UPDATE messages SET deleted=1`` fires the ``AFTER UPDATE`` trigger,
  which re-inserts the (unchanged) text — leaving the row searchable. That is
  fine here because Story 4.4 filters tombstones at query time. **Story 4.2 / 4.4
  own the tombstone↔FTS interaction** — this story only keeps the external-content
  contract honest (the ``AFTER DELETE`` trigger prevents an orphaned index if a
  hard delete ever happened).
- **Wiring the reader (``reader/server.py``)** — Story 4.3. This story adds only
  the reusable ``common/`` seam :func:`open_reader_with_fts`; it does NOT wire it
  into the reader server.
- **No ``db.py`` change** — registration + DDL are ``schema.py`` / FTS-lifecycle
  concerns; ``db.py`` stays the pure connection/PRAGMA substrate.
- ``sync_state.last_id`` advancement in a real sync loop — Stories 2.4 / 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import apsw

from tgkb.common import db, ru_lemma

# --- DDL --------------------------------------------------------------------
# Table names live here ONCE (single-source, mirroring db.DB_FILENAME etc.). The
# full v1 column set is defined up front — including `deleted` (Story 4.2) and
# `edit_date` (used here) — because migration tooling is deferred (ARCHITECTURE
# -SPINE#Deferred): adding a column later would force "the first schema change".

_CREATE_CHATS = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id   INTEGER PRIMARY KEY,
    title     TEXT,
    username  TEXT,
    is_public INTEGER NOT NULL DEFAULT 0
)
"""

# The core table. Identity is the composite PRIMARY KEY (chat_id, message_id) —
# this is the AD-4 identity AND the ON CONFLICT target for ingest. It is a normal
# rowid table (NOT `WITHOUT ROWID`): Story 3.2's external-content FTS5 needs the
# implicit rowid for its rowid ↔ (chat_id, message_id) mapping (AD-12).
_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id         INTEGER NOT NULL,
    message_id      INTEGER NOT NULL,
    sender_id       INTEGER,
    sender_name     TEXT,
    text            TEXT,
    date_utc        TEXT NOT NULL,
    reply_to_msg_id INTEGER,
    edit_date       TEXT,
    deleted         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (chat_id, message_id)
)
"""

# Defined here; its atomic advancement (advance_last_id) is wired by Stories
# 2.4 / 4.1 in the SAME transaction as the messages it tracks (AD-5).
_CREATE_SYNC_STATE = """
CREATE TABLE IF NOT EXISTS sync_state (
    chat_id    INTEGER PRIMARY KEY,
    last_id    INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""

# --- Metadata KV table (Story 3.3, AD-11) -----------------------------------
# A tiny generic key/value table for single-value DB metadata. It exists so the
# tokenizer fingerprint (and any future single-value metadata) has a home WITHOUT
# forcing a new column / "first schema change" each time (migration tooling is
# deferred — ARCHITECTURE-SPINE#Deferred; deferred-work.md#story-2.2). The
# fingerprint is stored under _FINGERPRINT_KEY (single-sourced below, mirroring
# the TOKENIZER_NAME "never a literal typed twice" discipline).
_CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

# The fixed `meta` key under which the tokenizer fingerprint lives. Single-sourced
# so read_fingerprint / write_fingerprint / the stamp path never disagree.
_FINGERPRINT_KEY = "tokenizer_fingerprint"

# Upsert so a re-stamp (fresh-DB stamp, then a post-rebuild re-stamp) updates the
# SAME row rather than raising on the PRIMARY KEY conflict.
_WRITE_FINGERPRINT_SQL = """
INSERT INTO meta (key, value)
VALUES (?, ?)
ON CONFLICT (key) DO UPDATE SET value = excluded.value
"""

_READ_FINGERPRINT_SQL = "SELECT value FROM meta WHERE key = ?"

# FTS5's blessed full-index rebuild (Story 3.3's drift-recovery action). It drops
# and re-derives the ENTIRE messages_fts index from `messages`, re-tokenizing
# every text through the CURRENTLY-registered ru_lemma. `ru_lemma` MUST be
# registered on the connection first (the registration-ordering trap, carried
# from 3.2) — check_and_repair_fingerprint registers defensively before running it.
_REBUILD_FTS_SQL = "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"

# --- FTS5 index (Story 3.2, AD-12 / AD-6 / AD-11) ---------------------------
# `messages_fts` is an EXTERNAL-CONTENT FTS5 table over `messages`: FTS5 stores
# only the inverted index; the column data lives in `messages`. This pins the
# `rowid ↔ (chat_id, message_id)` mapping (AC#1) — `content_rowid='rowid'` ties
# the FTS rowid to `messages`'s implicit rowid (which is why Story 2.2 kept
# `messages` a rowid table), and `(chat_id, message_id)` is recovered by joining
# `messages_fts.rowid = messages.rowid`. Only `text` is indexed (FR-7 is over
# message text; sender/date/etc. are citation columns fetched from `messages` in
# Story 4.4, not FTS-indexed).
#
# The `tokenize=` clause is built from ru_lemma.TOKENIZE_SPEC (never a literal),
# so the "same logic both ends" contract (AD-11) is single-sourced and the table
# can never disagree with what `register()` registers.
_CREATE_MESSAGES_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    content='messages',
    content_rowid='rowid',
    tokenize="{ru_lemma.TOKENIZER_NAME} {ru_lemma.BASE_TOKENIZER}"
)
"""

# External-content FTS5 does NOT auto-update — these three triggers ARE the sync
# mechanism (the canonical SQLite external-content pattern). They fire
# SYNCHRONOUSLY inside the statement mutating `messages`, which is always inside
# `db.writing` (BEGIN IMMEDIATE), so the FTS write is in the SAME transaction as
# the message write BY CONSTRUCTION (AD-12) — no separate pass, and a rollback
# drops both.
#
# The 'delete' command (first column = the table's own name = FTS5's special
# command channel) MUST be given the OLD indexed `text` so FTS5 can re-tokenize
# it and remove exactly the right index entries — supplying wrong/absent old text
# orphans entries and a later MATCH returns stale/wrong hits (the exact drift
# AD-12 exists to prevent). The UPDATE trigger is delete-OLD-then-insert-NEW for
# the same reason: an edit-upsert (changed edit_date, Story 2.2) removes the old
# text's entries and indexes the new text, atomically.
_CREATE_FTS_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, text) VALUES (new.rowid, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES('delete', old.rowid, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES('delete', old.rowid, old.text);
        INSERT INTO messages_fts(rowid, text) VALUES (new.rowid, new.text);
    END
    """,
)

# The index-backed MATCH recipe (AC#4). `WHERE messages_fts MATCH ?` guarantees
# index-backed search, never a full scan (NFR-4); `ORDER BY bm25(messages_fts)`
# is BM25 most-relevant-first (FTS5's bm25() returns LOWER = more relevant, so
# plain ASC is correct — no DESC). The join recovers `messages` rows by rowid.
# This is the proven recipe Story 4.4 reuses to build `search_messages`.
_SEARCH_ROWIDS_SQL = """
SELECT m.rowid
FROM messages_fts f
JOIN messages m ON m.rowid = f.rowid
WHERE messages_fts MATCH ?
ORDER BY bm25(messages_fts)
"""

# The AD-4 ingest statement — one statement expressing BOTH AC#2 and AC#3.
#
#   * No conflict (new message)         → plain INSERT.
#   * Conflict, edit_date unchanged     → WHERE is false → no UPDATE → DO NOTHING
#                                         (AC#2: no duplicate, text untouched).
#   * Conflict, edit_date changed       → UPDATE text + edit_date only (AC#3);
#                                         identity/authorship/original-date kept.
#
# `IS NOT` (null-safe inequality), NOT `<>`: a first edit is NULL → timestamp, and
# `NULL <> 'x'` is NULL (falsey) which would WRONGLY skip the update. `NULL IS NOT
# 'x'` is true. This is the one real correctness trap in the ingest SQL.
#
# Only `text` and `edit_date` are updated on conflict (AC#3 names exactly those
# two): sender_*, date_utc, reply_to_msg_id, and the PK columns are never changed
# on edit.
_INGEST_SQL = """
INSERT INTO messages
    (chat_id, message_id, sender_id, sender_name, text, date_utc, reply_to_msg_id, edit_date)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (chat_id, message_id) DO UPDATE SET
    text = excluded.text,
    edit_date = excluded.edit_date
WHERE excluded.edit_date IS NOT messages.edit_date
"""

_UPSERT_CHAT_SQL = """
INSERT INTO chats (chat_id, title, username, is_public)
VALUES (?, ?, ?, ?)
ON CONFLICT (chat_id) DO UPDATE SET
    title = excluded.title,
    username = excluded.username,
    is_public = excluded.is_public
"""

# The MAX(...) guard is the AD-5 monotonic rule in SQL: an edit (a lower id) MUST
# NOT regress the high-water mark, but updated_at always reflects the last touch.
_ADVANCE_LAST_ID_SQL = """
INSERT INTO sync_state (chat_id, last_id, updated_at)
VALUES (?, ?, ?)
ON CONFLICT (chat_id) DO UPDATE SET
    last_id = MAX(sync_state.last_id, excluded.last_id),
    updated_at = excluded.updated_at
"""

# The READ side of the AD-5 high-water mark (Story 4.1's catch-up seeds its page
# cursor from this). Single-sourced here alongside its write twin above.
_READ_LAST_ID_SQL = "SELECT last_id FROM sync_state WHERE chat_id = ?"

# The AD-13 soft-delete statement (Story 4.2). A deletion NEVER hard-deletes a
# row — it flips the `deleted` tombstone flag. This UPDATE fires the `messages_au`
# AFTER UPDATE trigger, which refreshes the row's FTS entry in the SAME
# transaction (the v1 tombstone↔FTS design: the text STAYS in messages_fts and
# exclusion happens at QUERY time via a `deleted=0` filter, which Story 4.4 owns).
# Single-sourced here mirroring _INGEST_SQL / _ADVANCE_LAST_ID_SQL. A no-match
# (unknown/already-live-or-deleted row) is a silent no-op, never an error.
_MARK_DELETED_SQL = "UPDATE messages SET deleted = 1 WHERE chat_id = ? AND message_id = ?"


@dataclass(frozen=True)
class Message:
    """A single message row to ingest (frozen data holder, mirrors ``TrackedChat``).

    ``date_utc`` / ``edit_date`` are ISO-8601 UTC strings — the caller (Story 2.4)
    normalizes Telethon's ``datetime`` to UTC; this layer stores them verbatim.

    ``deleted`` is intentionally absent: ingest always inserts LIVE rows; the
    ``deleted`` column defaults to 0 and is flipped only by Story 4.2's tombstone
    path (AD-13).
    """

    chat_id: int
    message_id: int
    sender_id: int | None
    sender_name: str | None
    text: str | None
    date_utc: str
    reply_to_msg_id: int | None
    edit_date: str | None


def ensure_schema(conn: apsw.Connection) -> None:
    """Create the full schema — tables, the FTS5 index, and its sync triggers (idempotent).

    Runs ``CREATE … IF NOT EXISTS`` for ``chats`` / ``messages`` / ``sync_state``,
    then the ``messages_fts`` external-content FTS5 table and its three
    ``AFTER INSERT/UPDATE/DELETE`` triggers, all inside ONE
    :func:`tgkb.common.db.writing` transaction — a fresh DB lands the whole schema
    atomically. Safe to call on every writer start; a second call is a no-op
    (AC#1). Story 2.4 calls this at its DB-open step (see
    :func:`open_writer_with_schema`).

    **Registers the ``ru_lemma`` tokenizer on ``conn`` FIRST (Story 3.2, AD-6/11).**
    ``CREATE VIRTUAL TABLE … tokenize="ru_lemma …"`` resolves the tokenizer name
    at creation time, so the tokenizer MUST already be registered on this
    connection. Registration is per-connection and NOT persisted in the DB file,
    so it is co-located here: "the FTS table's tokenizer always exists when the
    table is created" is a local invariant, and every caller
    (:func:`open_writer_with_schema`, tests) gets it for free. ``register`` is
    idempotent per connection, so calling ``ensure_schema`` twice is harmless.
    """
    ru_lemma.register(conn)
    with db.writing(conn):
        conn.execute(_CREATE_CHATS)
        conn.execute(_CREATE_MESSAGES)
        conn.execute(_CREATE_SYNC_STATE)
        conn.execute(_CREATE_META)
        conn.execute(_CREATE_MESSAGES_FTS)
        for trigger_sql in _CREATE_FTS_TRIGGERS:
            conn.execute(trigger_sql)
        # Story 3.3 (AC#1): stamp the current fingerprint on a FRESH DB (no `meta`
        # row yet). A freshly-created index is built by the CURRENTLY-running
        # tokenizer, so it is trivially consistent with it — recording that makes
        # "when the index is built, a fingerprint is persisted" literally true for
        # the create path, and keeps the check-path uniform (a fresh DB and a
        # just-rebuilt DB both end fingerprinted). We stamp ONLY when absent so an
        # existing (possibly stale) fingerprint is never silently overwritten here
        # — a real drift must go through check_and_repair_fingerprint's rebuild.
        if read_fingerprint(conn) is None:
            write_fingerprint(conn, ru_lemma.fingerprint())


# --- Fingerprint persistence + the drift guard (Story 3.3, AD-11) -----------


def read_fingerprint(conn: apsw.Connection) -> str | None:
    """Return the persisted tokenizer fingerprint, or ``None`` on a fresh DB (AC#1).

    A plain read (no transaction needed). ``None`` means the ``meta`` row is absent
    — a fresh DB before ``ensure_schema`` stamped it, or a legacy DB predating the
    fingerprint. Callers treat ``None`` over a NON-empty index as drift (the index
    was built by an unknown tokenizer) — see :func:`check_and_repair_fingerprint`.
    """
    row = conn.execute(_READ_FINGERPRINT_SQL, (_FINGERPRINT_KEY,)).fetchone()
    return None if row is None else row[0]


def write_fingerprint(conn: apsw.Connection, fp: str) -> None:
    """Upsert the persisted tokenizer fingerprint (AC#4).

    **Call inside** :func:`tgkb.common.db.writing` — this function does NOT open its
    own transaction (AD-12 composability, the SAME contract as ``ingest_*`` /
    ``advance_last_id``). That is what lets :func:`check_and_repair_fingerprint`
    stamp the fingerprint in the SAME transaction as the ``'rebuild'`` (so they
    commit atomically — AC#4), and :func:`ensure_schema` stamp it in the same
    transaction as the ``CREATE`` statements. The upsert (``INSERT … ON CONFLICT``)
    updates the existing row on a re-stamp rather than raising on the PK conflict.
    """
    conn.execute(_WRITE_FINGERPRINT_SQL, (_FINGERPRINT_KEY, fp))


def check_and_repair_fingerprint(conn: apsw.Connection) -> bool:
    """Detect tokenizer drift and, on mismatch, force a rebuild (AC#2/#3/#4).

    The WRITER-side heart of AD-11. Compares the persisted fingerprint against the
    running tokenizer's :func:`tgkb.common.ru_lemma.fingerprint`:

    - **Match** (or a ``None`` fingerprint over an EMPTY index — a fresh/unindexed
      DB nothing could be stale about): do nothing, return ``False`` — startup
      proceeds, serving/indexing allowed (AC#2).
    - **Mismatch** (dict/engine/policy changed since the index was built), OR a
      missing fingerprint over a NON-empty index (the index was built by an
      unknown tokenizer): force a full FTS rebuild and re-stamp the current
      fingerprint, return ``True`` (AC#3 + AC#4).

    The rebuild (``INSERT INTO messages_fts(messages_fts) VALUES('rebuild')``) and
    the re-stamp run inside ONE :func:`tgkb.common.db.writing` transaction, so they
    commit atomically — a crash between them must not leave a fresh index marked
    with the OLD fingerprint (merely wasteful) NOR a stale index marked current
    (the silent recall collapse AD-11 prevents).

    Registers ``ru_lemma`` on ``conn`` defensively (idempotent) FIRST: ``'rebuild'``
    re-tokenizes every ``messages.text`` through the tokenizer named in the
    ``messages_fts`` DDL, so it MUST be registered on this connection regardless of
    how ``conn`` was opened (the registration-ordering trap carried from 3.2).

    Returns ``True`` iff a rebuild was performed — the caller (the writer CLI) uses
    this to surface a one-line "rebuilding search index…" message (AD-10 "no
    silent hang" spirit) while keeping this ``common/`` seam UI-agnostic.

    **Writer-only** — it issues a write. The reader uses :func:`fingerprint_matches`
    (read-only, detect-and-refuse); repair is the single writer's job (AD-2).
    """
    ru_lemma.register(conn)  # registration-ordering trap: needed for 'rebuild'
    current = ru_lemma.fingerprint()
    stored = read_fingerprint(conn)

    if stored == current:
        return False  # AC#2: matches → nothing to do

    if stored is None and _fts_is_empty(conn):
        # Fresh/unindexed DB with no fingerprint yet: nothing indexed can be stale.
        # Stamp the current fingerprint so a later restart matches (uniform state)
        # without a pointless rebuild of an empty index.
        with db.writing(conn):
            write_fingerprint(conn, current)
        return False

    # AC#3 + AC#4: mismatch (or missing fingerprint over a populated index) →
    # rebuild the whole index from `messages` and re-stamp, atomically.
    with db.writing(conn):
        conn.execute(_REBUILD_FTS_SQL)
        write_fingerprint(conn, current)
    return True


def _fts_is_empty(conn: apsw.Connection) -> bool:
    """True if ``messages_fts`` has no indexed rows (nothing could be stale).

    Uses the underlying ``messages`` content table (external-content FTS5 stores no
    directly countable rows of its own): if there are no messages, the index is
    trivially empty and a missing fingerprint is not evidence of drift.
    """
    return conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def fingerprint_matches(conn: apsw.Connection) -> bool:
    """Read-only drift check for the reader (AC#3, reader clause).

    Returns ``True`` iff the persisted fingerprint equals the running tokenizer's
    :func:`tgkb.common.ru_lemma.fingerprint`. A pure read — it NEVER writes, so it
    is safe on the reader's read-only (``query_only`` / ``READONLY``) connection.

    The reader cannot repair drift (a ``'rebuild'`` is a write; it fails on a
    read-only connection — AD-2). So the contract is: **Story 4.3 MUST call this at
    startup and REFUSE to serve search results when it returns ``False``** — a
    stale index would quietly collapse recall (index lemmas and query lemmas no
    longer agree), which is exactly the failure AD-11 forbids. Repair is the
    writer's job (:func:`check_and_repair_fingerprint`); the reader only refuses.

    A ``None`` stored fingerprint (fresh/legacy DB) is treated as a mismatch
    (``False``): the reader has no evidence the index agrees with the running
    tokenizer, so it must not serve until the writer stamps/repairs it.
    """
    return read_fingerprint(conn) == ru_lemma.fingerprint()


def ingest_message(conn: apsw.Connection, msg: Message) -> None:
    """Idempotently ingest one message (AD-4).

    **Call inside** :func:`tgkb.common.db.writing`; this function does NOT manage
    its own transaction, so Story 3.2 can add the FTS write to the SAME
    ``writing()`` scope (AD-12) and Story 4.1 can advance ``sync_state.last_id``
    atomically alongside it (AD-5).

    Semantics (AC#2 + AC#3, one statement):

    - a brand-new ``(chat_id, message_id)`` is inserted;
    - a re-ingest with an UNCHANGED ``edit_date`` is a no-op (no duplicate, text
      untouched);
    - a re-ingest with a CHANGED ``edit_date`` updates ``text`` + ``edit_date``
      only — identity, authorship, and original date are never touched.

    ``messages_fts`` maintenance is hooked in transparently by the
    ``AFTER INSERT/UPDATE/DELETE`` triggers installed in :func:`ensure_schema`
    (Story 3.2): they fire synchronously inside the caller's ``writing()`` scope,
    so the FTS row is written in the SAME transaction as the message write
    (AD-12), with no change to this function's body and no separate pass.
    """
    conn.execute(
        _INGEST_SQL,
        (
            msg.chat_id,
            msg.message_id,
            msg.sender_id,
            msg.sender_name,
            msg.text,
            msg.date_utc,
            msg.reply_to_msg_id,
            msg.edit_date,
        ),
    )


def ingest_messages(conn: apsw.Connection, msgs: Iterable[Message]) -> None:
    """Idempotently ingest many messages (batch form for Story 2.4's bulk export).

    **Call inside** :func:`tgkb.common.db.writing` — same transaction contract as
    :func:`ingest_message` (AD-12). Uses ``executemany`` for the single AD-4
    statement, so the whole batch shares one prepared statement and the caller's
    one transaction.
    """
    conn.executemany(
        _INGEST_SQL,
        [
            (
                m.chat_id,
                m.message_id,
                m.sender_id,
                m.sender_name,
                m.text,
                m.date_utc,
                m.reply_to_msg_id,
                m.edit_date,
            )
            for m in msgs
        ],
    )


def upsert_chat(
    conn: apsw.Connection,
    chat_id: int,
    title: str | None,
    username: str | None,
    is_public: bool,
) -> None:
    """Register/refresh a chat's metadata (convenience seam used by Story 2.4).

    **Call inside** :func:`tgkb.common.db.writing`. INSERT-or-UPDATE keyed on
    ``chat_id`` so Story 2.4 can record a Tracked Chat's title/username before
    ingesting its messages. There is no FK enforced on ``messages.chat_id`` in v1.
    ``is_public`` is stored as INTEGER 0/1 (SQLite has no native bool).
    """
    conn.execute(_UPSERT_CHAT_SQL, (chat_id, title, username, 1 if is_public else 0))


def advance_last_id(
    conn: apsw.Connection, chat_id: int, last_id: int, updated_at: str
) -> None:
    """Advance a chat's sync high-water mark monotonically (AD-5; defined for 2.4/4.1).

    **Call inside** :func:`tgkb.common.db.writing`. The ``MAX(...)`` guard enforces
    the AD-5 rule — ``last_id`` never regresses (an edit re-touches an older id but
    must not lower the mark) — while ``updated_at`` always advances to the last
    touch. Defined here but wired by Stories 2.4 / 4.1 in the same transaction as
    the messages it tracks.
    """
    conn.execute(_ADVANCE_LAST_ID_SQL, (chat_id, last_id, updated_at))


def read_last_id(conn: apsw.Connection, chat_id: int) -> int:
    """Return a chat's persisted ``sync_state.last_id`` high-water mark (AD-5 read side).

    The READ twin of :func:`advance_last_id`, added for Story 4.1's catch-up: it
    seeds each chat's page cursor from the stored mark so ``sync`` fetches only what
    arrived while the tool was offline (``min_id = last_id``, exclusive).

    A plain read — NO transaction wrapper needed (mirroring :func:`read_fingerprint`).
    Returns ``0`` when the chat has no ``sync_state`` row yet (never exported/synced):
    catch-up from ``0`` correctly means "fetch everything", identical to a first bulk
    export. Never regresses the mark and never writes.
    """
    row = conn.execute(_READ_LAST_ID_SQL, (chat_id,)).fetchone()
    return 0 if row is None else row[0]


def mark_deleted(conn: apsw.Connection, chat_id: int, message_id: int) -> None:
    """Soft-delete a message: flip its ``deleted`` tombstone flag (AD-13; Story 4.2).

    **Call inside** :func:`tgkb.common.db.writing` — this function does NOT open its
    own transaction (the SAME AD-12 composability contract as :func:`ingest_messages`
    / :func:`advance_last_id`), so the caller (the realtime listener) wraps it in a
    ``writing()`` scope, and the FTS refresh happens atomically with the flag flip.

    A row is NEVER hard-deleted in v1 (AD-13): the ``UPDATE messages SET deleted=1``
    fires the ``messages_au`` ``AFTER UPDATE`` trigger, which re-tokenizes the row's
    text into ``messages_fts`` in the SAME transaction. Per the v1 tombstone↔FTS
    design (see the module docstring), the tombstoned text STAYS indexed and
    exclusion happens at QUERY time via a ``deleted=0`` filter that **Story 4.4
    owns** — so this function adds NO FTS-removal path.

    Idempotent + total: marking an already-deleted row is a no-op (``deleted``
    stays 1); marking a ``(chat_id, message_id)`` that is not stored matches no rows
    and silently does nothing (never raises) — a delete event for a message we never
    ingested (e.g. a media-only message we skipped) is simply absorbed.
    """
    conn.execute(_MARK_DELETED_SQL, (chat_id, message_id))


def open_writer_with_schema(db_path: db.PathLike) -> apsw.Connection:
    """Open the writer connection, ensure the schema, and repair tokenizer drift.

    A thin ``common`` seam: :func:`tgkb.common.db.open_writer` (schema-free per its
    own contract) → :func:`ensure_schema` → :func:`check_and_repair_fingerprint`.
    This is the ONE writer DB-open point (the export CLI calls it), so hooking the
    drift guard here means EVERY writer run (``export``, later ``sync``) ensures the
    schema (stamping a fresh fingerprint) then repairs any drift BEFORE it ingests
    or serves — with no change to ``export.py`` / ``cli.py`` bodies (mirroring how
    Story 3.2 made FTS maintenance transparent to the ingest call sites).

    The rebuild-happened flag is intentionally NOT surfaced here (this ``common/``
    seam stays UI-agnostic) — the repair ALWAYS happens on this one writer path, so
    ``export`` and later ``sync`` both get the guard for free. Console visibility is
    the writer CLI's job: BEFORE calling this seam it peeks the stored fingerprint
    on a read-only connection (:func:`fingerprint_matches`) and, on a mismatch, logs
    a one-line "rebuilding the full-text index…" notice (AD-10 "no silent hang"
    spirit) — then this seam performs the actual rebuild. The peek is advisory (the
    seam is the safety net that guarantees repair regardless); it is only for the
    log line, so ``common/`` never prints.

    ``db.open_writer`` is intentionally NOT taught about schema — the ``db.py`` ↔
    ``schema.py`` boundary stays crisp.
    """
    conn = db.open_writer(db_path)
    try:
        ensure_schema(conn)
        check_and_repair_fingerprint(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def open_reader_with_fts(db_path: db.PathLike) -> apsw.Connection:
    """Open a read-only connection with ``ru_lemma`` registered for ``MATCH`` (Story 3.2 seam).

    The reader mirror of :func:`open_writer_with_schema`:
    :func:`tgkb.common.db.open_reader` (which stays schema-free) followed by
    :func:`tgkb.common.ru_lemma.register`. A ``MATCH 'договор'`` lemmatizes the
    QUERY through ``ru_lemma``; if the query is not lemmatized the SAME way as the
    index, recall collapses (the AD-6/AD-11 drift), so EVERY reader connection
    that runs ``MATCH`` must register the tokenizer first.

    Defined here as the single ``common/`` reader-registration seam so Stories 4.3
    / 4.4 register identically and never re-type it (mirroring how ``schema.py``
    pre-built ``advance_last_id`` / ``upsert_chat`` before their consumers
    existed). **This story does NOT wire it into ``reader/server.py``** — that is
    Story 4.3. It creates NO schema (the reader is read-only; the writer owns DDL)
    — it only registers the tokenizer on an existing DB.
    """
    conn = db.open_reader(db_path)
    try:
        ru_lemma.register(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def search_message_rowids(
    conn: apsw.Connection, query: str, limit: int | None = None
) -> list[int]:
    """Return ``messages.rowid`` values matching ``query``, BM25-ranked (Story 3.2).

    A MINIMAL, internal index-backed search helper — NOT the ``search_messages``
    MCP tool. Its job is twofold: make THIS story's searchability directly
    assertable, and hand Story 4.4 a proven ``MATCH``+``bm25`` recipe. Story 4.4's
    ``reader/`` tool owns Citations (text/sender/date/chat/t.me link/context,
    AD-9), tombstone filtering (AD-9/AD-13), the read-only tool boundary, and the
    ``(chat_id, message_id)`` projection — none of which live here.

    - ``WHERE messages_fts MATCH ?`` is index-backed, never a full scan (NFR-4).
    - ``ORDER BY bm25(messages_fts)`` is BM25 most-relevant-first (AC#4): FTS5's
      ``bm25()`` scores a more relevant row LOWER, so plain ASC is correct.
    - Results are the surviving rows only — tombstoned (``deleted=1``) rows are
      NOT excluded here (Story 4.4 filters them at query time).

    Registers ``ru_lemma`` on ``conn`` (idempotent) so the query is lemmatized the
    SAME way as the index even if the caller opened a bare connection — the
    AD-6/AD-11 "same logic both ends" guarantee holds regardless of how ``conn``
    was obtained.

    ``limit`` caps the number of rows (``None`` = no cap). A ``limit <= 0`` returns
    an empty list — SQLite would otherwise treat a *negative* ``LIMIT`` as
    "unlimited", so ``limit=-1`` returning ALL rows would be a silent footgun; a
    non-positive cap unambiguously means "no rows".

    **Query is a raw FTS5 ``MATCH`` expression.** This minimal internal helper does
    NOT sanitize or phrase-quote ``query`` — a caller passing FTS5 query-syntax
    metacharacters (``"``, ``*``, ``:``, ``AND``/``OR``/``NEAR`` …) or an empty
    string gets an ``apsw.SQLError`` from FTS5. Query-string hygiene at the tool
    boundary is Story 4.4's ``search_messages`` (AD-9); tests here pass clean lemma
    queries.
    """
    if limit is not None and limit <= 0:
        return []
    ru_lemma.register(conn)
    sql = _SEARCH_ROWIDS_SQL
    params: tuple[object, ...] = (query,)
    if limit is not None:
        sql = f"{sql}\nLIMIT ?"
        params = (query, limit)
    return [row[0] for row in conn.execute(sql, params).fetchall()]
