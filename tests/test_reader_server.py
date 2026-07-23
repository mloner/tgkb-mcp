"""Tests for the read-only MCP stdio reader server (Story 4.3).

Behavior-first (the review lesson carried 1.1 → 4.2): every test asserts a REAL
guarantee against a REAL tmp-file APSW DB seeded by the writer seam
``open_writer_with_schema`` — a write on the connection RAISES (not a read-only
flag), the health tool returns the REAL ``count(*)``, ``search_ok`` flips on a
REAL fingerprint mismatch, stdout is REALLY empty, a missing DB REALLY fails
clean. No network, no real Claude Code, no blocking ``mcp.run()``.

The async ``app_lifespan`` is driven with ``asyncio.run`` (the 4.1/4.2
mechanism) — NO ``pytest-asyncio`` dependency. The ``server_status`` tool is
exercised both through its testable core (``_server_status``) and through the
real ``@mcp.tool`` wrapper via a hand-built FastMCP ``Context``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import apsw
import pytest
from mcp.server.fastmcp import Context
from mcp.shared.context import RequestContext

from tgkb.common import db, schema
from tgkb.common.schema import Message
from tgkb.reader import server


# --- helpers ----------------------------------------------------------------


def _make(
    *,
    chat_id: int = 1,
    message_id: int = 100,
    sender_id: int | None = 42,
    sender_name: str | None = "Alice",
    text: str | None = "подписали договоры вчера",
    date_utc: str = "2026-07-20T14:03:00+00:00",
    reply_to_msg_id: int | None = None,
    edit_date: str | None = None,
) -> Message:
    return Message(
        chat_id=chat_id,
        message_id=message_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        date_utc=date_utc,
        reply_to_msg_id=reply_to_msg_id,
        edit_date=edit_date,
    )


def _seed_db(path: Path, *, messages: int = 0) -> None:
    """Create a real DB via the writer seam (stamps the current fingerprint)."""
    conn = schema.open_writer_with_schema(path)
    try:
        if messages:
            with db.writing(conn):
                schema.ingest_messages(
                    conn,
                    [_make(message_id=100 + i) for i in range(messages)],
                )
    finally:
        conn.close()


def _run_lifespan(path: Path, monkeypatch, body):
    """Point the reader at ``path``, run the lifespan, and call ``body(app_ctx)``.

    ``body`` runs INSIDE the lifespan (connection open); its return value is
    returned. Uses the real ``app_lifespan`` so startup drift evaluation, stderr
    logging, and shutdown ``conn.close()`` all execute for real.
    """
    monkeypatch.setattr(db, "default_db_path", lambda: path)

    async def _drive():
        async with server.app_lifespan(server.mcp) as app_ctx:
            return body(app_ctx)

    return asyncio.run(_drive())


# --- AC#2 / AC#4: SQL-level read-only backstop ------------------------------


def test_server_opens_readonly_connection(tmp_path: Path, monkeypatch) -> None:
    """After startup the stashed connection is read-only AND rejects a write."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path)

    def check(app_ctx: server.AppContext) -> None:
        assert app_ctx.conn.readonly("main") is True
        with pytest.raises(apsw.ReadOnlyError):
            app_ctx.conn.execute(
                "INSERT INTO messages(chat_id, message_id, date_utc) "
                "VALUES (1, 1, '2026-01-01T00:00:00+00:00')"
            )

    _run_lifespan(path, monkeypatch, check)


def test_write_and_ddl_rejected_at_connection(tmp_path: Path, monkeypatch) -> None:
    """The lifespan connection rejects INSERT/UPDATE/DELETE/CREATE/DROP (Layer 2)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=1)

    statements = [
        "INSERT INTO messages(chat_id, message_id, date_utc) "
        "VALUES (9, 9, '2026-01-01T00:00:00+00:00')",
        "UPDATE messages SET text = 'x'",
        "DELETE FROM messages",
        "CREATE TABLE evil(x)",
        "DROP TABLE messages",
    ]

    def check(app_ctx: server.AppContext) -> None:
        for stmt in statements:
            with pytest.raises(apsw.ReadOnlyError):
                app_ctx.conn.execute(stmt)

    _run_lifespan(path, monkeypatch, check)


# --- AC#2 / Task 3: the health tool reports the REAL count ------------------


def test_health_tool_reports_real_message_count(tmp_path: Path, monkeypatch) -> None:
    """The health tool returns message_count == N from a real SELECT count(*)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=3)

    def check(app_ctx: server.AppContext) -> dict:
        return server._server_status(app_ctx)

    status = _run_lifespan(path, monkeypatch, check)
    assert status["message_count"] == 3
    assert status["read_only"] is True
    assert status["server"] == "tgkb-reader"
    assert status["search_ok"] is True


def test_health_tool_via_mcp_context(tmp_path: Path, monkeypatch) -> None:
    """The real @mcp.tool wrapper reaches the lifespan context and returns status."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=2)

    def check(app_ctx: server.AppContext) -> dict:
        rc = RequestContext(
            request_id=1, meta=None, session=None, lifespan_context=app_ctx
        )
        ctx = Context(request_context=rc)
        return server.server_status(ctx)

    status = _run_lifespan(path, monkeypatch, check)
    assert status["message_count"] == 2
    assert status["read_only"] is True


# --- AC#5: drift gate -------------------------------------------------------


def test_search_ok_true_when_fingerprint_matches(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=1)

    def check(app_ctx: server.AppContext) -> bool:
        return app_ctx.search_ok

    assert _run_lifespan(path, monkeypatch, check) is True


def test_empty_db_reports_zero_and_search_ok(tmp_path: Path, monkeypatch) -> None:
    """The post-`export`/pre-ingest state: schema stamped, no messages.

    Review patch (boundary coverage): ``open_writer_with_schema`` stamps the
    fingerprint even on an empty DB, so a freshly-initialized store with zero
    messages must report ``search_ok=True`` and ``message_count=0`` — an honest
    "index agrees, nothing indexed yet", NOT drift.
    """
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=0)

    status = _run_lifespan(
        path, monkeypatch, lambda app_ctx: server._server_status(app_ctx)
    )
    assert status["message_count"] == 0
    assert status["search_ok"] is True
    assert status["read_only"] is True


def test_search_refused_on_fingerprint_drift(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A stale fingerprint → search_ok False + stderr warning, no rebuild, no crash."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=1)

    # Corrupt the persisted fingerprint via a writer connection (the reader
    # cannot write). This is the REAL drift the reader must refuse to serve.
    writer = db.open_writer(path)
    try:
        with db.writing(writer):
            schema.write_fingerprint(writer, "stale-bogus-fingerprint")
    finally:
        writer.close()

    def check(app_ctx: server.AppContext) -> bool:
        return app_ctx.search_ok

    with caplog.at_level("WARNING", logger="tgkb"):
        search_ok = _run_lifespan(path, monkeypatch, check)

    assert search_ok is False
    assert any("stale" in r.getMessage() for r in caplog.records)

    # The reader did NOT rebuild — the fingerprint is unchanged after startup.
    reopened = db.open_reader(path)
    try:
        assert schema.read_fingerprint(reopened) == "stale-bogus-fingerprint"
    finally:
        reopened.close()


# --- Task 1: missing DB fails cleanly, never created ------------------------


def test_missing_db_fails_cleanly(tmp_path: Path, monkeypatch, caplog) -> None:
    path = tmp_path / "does-not-exist.sqlite3"

    def check(app_ctx: server.AppContext) -> None:  # pragma: no cover - never reached
        raise AssertionError("lifespan should not yield when the DB is missing")

    with caplog.at_level("ERROR", logger="tgkb"):
        with pytest.raises(apsw.CantOpenError):
            _run_lifespan(path, monkeypatch, check)

    # Clear, secret-free message naming the path + suggesting the writer.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert str(path) in msgs
    assert "export" in msgs or "sync" in msgs
    # The reader did NOT create the file.
    assert not path.exists()


def test_corrupt_db_fails_cleanly(tmp_path: Path, monkeypatch, caplog) -> None:
    """A garbage/corrupt file fails startup with a clear message, not a raw traceback.

    Review patch: the startup translation now covers non-``CantOpenError`` open
    failures (corrupt/partial DB, locked, permission, non-local path) too — it
    emits a secret-free message naming only the path and re-raises.
    """
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"definitely not a sqlite database" * 8)

    def check(app_ctx: server.AppContext) -> None:  # pragma: no cover - never reached
        raise AssertionError("lifespan should not yield when the DB is corrupt")

    with caplog.at_level("ERROR", logger="tgkb"):
        with pytest.raises(apsw.Error):
            _run_lifespan(path, monkeypatch, check)

    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert str(path) in msgs
    assert "export" in msgs or "sync" in msgs
    # No secret leaked in the diagnostic.
    for s in ("api_hash", "passphrase", "session"):
        assert s not in msgs


# --- AC#6: stdout hygiene ---------------------------------------------------


def test_no_stdout_writes(tmp_path: Path, monkeypatch, capsys) -> None:
    """Startup + a tool call write NOTHING to stdout (all diagnostics to stderr)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=1)

    def check(app_ctx: server.AppContext) -> dict:
        return server._server_status(app_ctx)

    _run_lifespan(path, monkeypatch, check)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_no_stdout_on_drift_warning(tmp_path: Path, monkeypatch, capsys, caplog) -> None:
    """The drift warning goes to the tgkb logger, and NOTHING leaks to stdout."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=1)
    writer = db.open_writer(path)
    try:
        with db.writing(writer):
            schema.write_fingerprint(writer, "stale-bogus")
    finally:
        writer.close()

    with caplog.at_level("WARNING", logger="tgkb"):
        _run_lifespan(path, monkeypatch, lambda app_ctx: None)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert any("stale" in r.getMessage() for r in caplog.records)


# --- AC#3: no AI key required -----------------------------------------------


def test_no_ai_key_required(tmp_path: Path, monkeypatch) -> None:
    """The server constructs and the health tool runs with all AI-key env unset."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "API_KEY"):
        monkeypatch.delenv(var, raising=False)

    path = tmp_path / "tgkb.sqlite3"
    _seed_db(path, messages=1)

    status = _run_lifespan(path, monkeypatch, lambda app_ctx: server._server_status(app_ctx))
    assert status["message_count"] == 1


def test_reader_imports_no_ai_client() -> None:
    """No reader/ CODE PATH imports an AI/embedding client or reads an AI key.

    Structural, not prose: parse every reader/ module and assert (1) no import
    names an AI/embedding client, and (2) no string literal reaches for an
    AI-key env var (``os.environ["OPENAI_API_KEY"]`` etc.). The module DOCSTRING
    naming these tokens in prose (e.g. "no AI/embedding key") is fine — the
    guarantee is about code, not comments (AC#3, NFR-2, AD-7).
    """
    import ast

    reader_dir = Path(server.__file__).parent
    banned_import_roots = {"openai", "anthropic", "cohere", "google", "mistralai"}
    banned_key_literals = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "API_KEY"}

    for py in reader_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned_import_roots, f"{py.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in banned_import_roots, f"{py.name} imports from {node.module}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in banned_key_literals, (
                    f"{py.name} references AI-key env var {node.value!r}"
                )


# --- NFR-1: secret safety ---------------------------------------------------


def test_secret_safety(tmp_path: Path, monkeypatch, capsys, caplog) -> None:
    """No startup/tool path emits api_hash / session / passphrase / message text."""
    path = tmp_path / "tgkb.sqlite3"
    secret_text = "SUPER-SECRET-MESSAGE-BODY"
    conn = schema.open_writer_with_schema(path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text=secret_text))
    finally:
        conn.close()

    with caplog.at_level("DEBUG", logger="tgkb"):
        status = _run_lifespan(
            path, monkeypatch, lambda app_ctx: server._server_status(app_ctx)
        )

    captured = capsys.readouterr()
    sentinels = (secret_text, "api_hash", "passphrase", "session")
    blob = captured.out + captured.err + " ".join(r.getMessage() for r in caplog.records)
    for s in sentinels:
        assert s not in blob
    # The health tool returns a COUNT, not text.
    assert status["message_count"] == 1
    assert secret_text not in json.dumps(status)


# --- AC#1: documented MCP registration ---------------------------------------


def test_mcp_registration_documented_in_readme() -> None:
    """The README documents how to register the reader: launches `tgkb-mcp`,
    under the `tgkb` server name, and carries no AI/LLM API key in the snippet.

    The repo intentionally ships NO `.mcp.json` (users merge the snippet into
    their own MCP config), so the registration contract lives in the README
    rather than a committed file. Guard the substance of that snippet.
    """
    root = Path(server.__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")

    # The registration snippet names the server and its launch command.
    assert '"tgkb"' in readme
    assert "tgkb-mcp" in readme
    # No AI-key env var is ever suggested in the docs (AC#3: key-free reader).
    assert "API_KEY" not in readme.upper()


# ===========================================================================
# Story 4.4 — search_messages tool with complete citations (AC#1–#7)
# ===========================================================================


def _seed_search_db(
    path: Path,
    messages: list[Message],
    *,
    chats: list[tuple[int, str | None, str | None, bool]] | None = None,
) -> None:
    """Seed a REAL DB with chats + messages via the writer seam (AD-9 citation source).

    ``chats`` is a list of ``(chat_id, title, username, is_public)`` tuples; the
    ``t.me`` link derives from ``username`` + ``is_public`` (the writer's
    ``is_public = bool(username)`` invariant). Messages are ingested through the
    real ingest path so the FTS index + triggers fire for real.
    """
    conn = schema.open_writer_with_schema(path)
    try:
        with db.writing(conn):
            for (chat_id, title, username, is_public) in chats or []:
                schema.upsert_chat(conn, chat_id, title, username, is_public)
            schema.ingest_messages(conn, messages)
    finally:
        conn.close()


def _search(path: Path, monkeypatch, **kwargs):
    """Run the search core inside the real lifespan; return its result."""
    return _run_lifespan(
        path, monkeypatch, lambda app_ctx: server._search_messages(app_ctx, **kwargs)
    )


# --- AC#1: BM25-ranked, index-backed hits -----------------------------------


def test_search_returns_bm25_ranked_hits(tmp_path: Path, monkeypatch) -> None:
    """A query returns non-empty hits, most-relevant first (BM25), index-backed."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(message_id=100, text="совещание про бюджет и планы"),
            _make(message_id=101, text="договор договор договор подписан"),
            _make(message_id=102, text="один договор в тексте"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="договор")
    ids = [h["message_id"] for h in hits]
    assert set(ids) == {101, 102}  # the budget message does not match
    # Relative rank guarantee (AC#1 "most-relevant first"), not brittle exact
    # positions: the message mentioning "договор" three times must rank strictly
    # above the single-mention one. Asserting the ORDER (not fixed indices) keeps
    # the test robust to FTS5/SQLite BM25 normalization internals across versions.
    assert ids.index(101) < ids.index(102)


# --- AC#2: full citation shape ----------------------------------------------


def test_search_citation_shape_complete(tmp_path: Path, monkeypatch) -> None:
    """Each hit carries every AD-9 field, populated from the REAL row + chats join."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, sender_name="Alice", text="подписали договоры вчера")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="договор")
    assert len(hits) == 1
    hit = hits[0]
    assert set(hit) == {
        "text", "sender", "date", "chat", "chat_id", "message_id", "link", "context",
    }
    assert hit["text"] == "подписали договоры вчера"
    assert hit["sender"] == "Alice"
    assert hit["date"] == "2026-07-20T14:03:00+00:00"
    assert hit["chat"] == "Team Chat"
    assert hit["chat_id"] == 1
    assert hit["message_id"] == 100
    assert hit["link"] == "https://t.me/teamchat/100"
    assert isinstance(hit["context"], list)


def test_public_chat_link_built_private_null(tmp_path: Path, monkeypatch) -> None:
    """Public+username chat → t.me link; private chat → link is None (never faked)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(chat_id=1, message_id=100, text="договор в публичном чате"),
            _make(chat_id=2, message_id=200, text="договор в приватном чате"),
        ],
        chats=[
            (1, "Public Chat", "pubchat", True),
            (2, "Private Chat", None, False),
        ],
    )

    hits = _search(path, monkeypatch, query="договор")
    by_chat = {h["chat_id"]: h for h in hits}
    assert by_chat[1]["link"] == "https://t.me/pubchat/100"
    assert by_chat[2]["link"] is None


# --- AC#3: tombstones excluded ----------------------------------------------


def test_tombstoned_hit_excluded(tmp_path: Path, monkeypatch) -> None:
    """A message that matches but is tombstoned is NOT returned as a hit."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(message_id=100, text="живой договор"),
            _make(message_id=101, text="удалённый договор"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    # Tombstone message 101 via a writer connection (the reader cannot write).
    # Use the writer seam so ru_lemma is registered — mark_deleted fires the FTS
    # AFTER UPDATE trigger, which re-tokenizes through ru_lemma.
    writer = schema.open_writer_with_schema(path)
    try:
        with db.writing(writer):
            schema.mark_deleted(writer, 1, 101)
    finally:
        writer.close()

    hits = _search(path, monkeypatch, query="договор")
    ids = {h["message_id"] for h in hits}
    assert ids == {100}
    # And it never appears as a context neighbor of the surviving hit.
    for h in hits:
        assert all(n["message_id"] != 101 for n in h["context"])


# --- AC#4: Russian morphology end-to-end ------------------------------------


def test_russian_morphology_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Query 'договор' matches messages storing inflected forms (ru_lemma path)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(message_id=100, text="подписали договоры"),
            _make(message_id=101, text="условия договора важны"),
            _make(message_id=102, text="работаем над договором"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="договор")
    assert {h["message_id"] for h in hits} == {100, 101, 102}


# --- AC#5: empty-set convention ---------------------------------------------


def test_no_match_returns_empty_list(tmp_path: Path, monkeypatch) -> None:
    """A query matching nothing returns [] exactly — not an error, not a placeholder."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="совещание про бюджет")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="договор")
    assert hits == []


def test_empty_or_whitespace_query_returns_empty(tmp_path: Path, monkeypatch) -> None:
    """'' / '   ' → [] (no apsw.SQLError escapes the tool boundary)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    assert _search(path, monkeypatch, query="") == []
    assert _search(path, monkeypatch, query="   ") == []


def test_fts_metacharacters_do_not_crash(tmp_path: Path, monkeypatch) -> None:
    """FTS5 metacharacters in the query are quoted, not interpreted → no traceback."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    # A raw '"' / 'AND' / '*' would raise apsw.SQLError if passed unquoted.
    for q in ['договор"', "договор AND", "договор*", 'foo" OR "bar']:
        result = _search(path, monkeypatch, query=q)
        assert isinstance(result, list)


# --- AC#6: stale index refuses (≠ empty) ------------------------------------


def test_stale_index_refuses_not_empty(tmp_path: Path, monkeypatch) -> None:
    """search_ok=False → REFUSAL (raises), not [], and the reader does not repair."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    # Corrupt the persisted fingerprint so 4.3's startup gate sets search_ok False.
    writer = db.open_writer(path)
    try:
        with db.writing(writer):
            schema.write_fingerprint(writer, "stale-bogus-fingerprint")
    finally:
        writer.close()

    def check(app_ctx: server.AppContext):
        assert app_ctx.search_ok is False
        with pytest.raises(server.SearchIndexStaleError) as exc:
            server._search_messages(app_ctx, query="договор")
        # The refusal names the staleness and the fix (run the writer).
        msg = str(exc.value).lower()
        assert "stale" in msg
        assert "export" in msg or "sync" in msg

    _run_lifespan(path, monkeypatch, check)

    # The reader did NOT repair — the (bogus) fingerprint is unchanged.
    reopened = db.open_reader(path)
    try:
        assert schema.read_fingerprint(reopened) == "stale-bogus-fingerprint"
    finally:
        reopened.close()


# --- AC#7: read-only, stdout-clean, secret-safe -----------------------------


def test_search_is_read_only(tmp_path: Path, monkeypatch) -> None:
    """The tool runs over the read-only connection; a write still raises after a search."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        server._search_messages(app_ctx, query="договор")
        # Layer-2 backstop still holds after the search: a write raises.
        with pytest.raises(apsw.ReadOnlyError):
            app_ctx.conn.execute(
                "INSERT INTO messages(chat_id, message_id, date_utc) "
                "VALUES (9, 9, '2026-01-01T00:00:00+00:00')"
            )

    _run_lifespan(path, monkeypatch, check)


def test_no_stdout_writes_on_search(tmp_path: Path, monkeypatch, capsys) -> None:
    """A search + a stale-refusal write NOTHING to stdout (diagnostics → stderr)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    _search(path, monkeypatch, query="договор")

    # Now a stale refusal path too.
    writer = db.open_writer(path)
    try:
        with db.writing(writer):
            schema.write_fingerprint(writer, "stale-bogus")
    finally:
        writer.close()

    def check(app_ctx: server.AppContext) -> None:
        with pytest.raises(server.SearchIndexStaleError):
            server._search_messages(app_ctx, query="договор")

    _run_lifespan(path, monkeypatch, check)
    assert capsys.readouterr().out == ""


def test_secret_safety_search(tmp_path: Path, monkeypatch, capsys, caplog) -> None:
    """No api_hash / session / passphrase leaks on any search path (text is legit)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    with caplog.at_level("DEBUG", logger="tgkb"):
        _search(path, monkeypatch, query="договор")
        _search(path, monkeypatch, query="нетсовпадений")

    captured = capsys.readouterr()
    blob = captured.out + captured.err + " ".join(
        r.getMessage() for r in caplog.records
    )
    for s in ("api_hash", "passphrase", "session"):
        assert s not in blob


# --- AC#2: context neighbors ordered, bounded, tombstone-free ---------------


def test_context_neighbors_ordered_and_bounded(tmp_path: Path, monkeypatch) -> None:
    """Context comes from the same chat, ordered by (chat_id, message_id), bounded."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(message_id=97, text="сообщение раньше три"),
            _make(message_id=98, text="сообщение раньше два"),
            _make(message_id=99, text="сообщение раньше один"),
            _make(message_id=100, text="ключевой договор здесь"),
            _make(message_id=101, text="сообщение позже один"),
            _make(message_id=102, text="сообщение позже два"),
            _make(message_id=103, text="сообщение позже три"),
            # A different chat — must NOT leak into chat 1's context.
            _make(chat_id=2, message_id=100, text="чужой договор чат два"),
        ],
        chats=[
            (1, "Team Chat", "teamchat", True),
            (2, "Other Chat", "otherchat", True),
        ],
    )

    hits = _search(path, monkeypatch, query="договор", context=2)
    hit = next(h for h in hits if h["chat_id"] == 1 and h["message_id"] == 100)
    ctx_ids = [n["message_id"] for n in hit["context"]]
    # ±2 neighbors, hit excluded, same chat, ascending order.
    assert ctx_ids == [98, 99, 101, 102]
    assert all("чат два" not in n["text"] for n in hit["context"])


def test_context_depth_zero_returns_no_neighbors(tmp_path: Path, monkeypatch) -> None:
    """context=0 → each hit carries an empty context list."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(message_id=99, text="сосед раньше"),
            _make(message_id=100, text="ключевой договор"),
            _make(message_id=101, text="сосед позже"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="договор", context=0)
    assert hits and all(h["context"] == [] for h in hits)


# --- Task 4: limit cap ------------------------------------------------------


def test_limit_caps_results(tmp_path: Path, monkeypatch) -> None:
    """limit bounds the number of hits; limit<=0 → [] (the seam's footgun guard)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100 + i, text=f"договор номер {i}") for i in range(5)],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    assert len(_search(path, monkeypatch, query="договор", limit=3)) == 3
    assert _search(path, monkeypatch, query="договор", limit=0) == []


# --- Tool wrapper (async) routes through the core ---------------------------


def test_search_tool_wrapper_async(tmp_path: Path, monkeypatch) -> None:
    """The real async @mcp.tool wrapper reaches the lifespan ctx and returns hits."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    monkeypatch.setattr(db, "default_db_path", lambda: path)

    async def drive():
        async with server.app_lifespan(server.mcp) as app_ctx:
            rc = RequestContext(
                request_id=1, meta=None, session=None, lifespan_context=app_ctx
            )
            ctx = Context(request_context=rc)
            return await server.search_messages(ctx, query="договор")

    hits = asyncio.run(drive())
    assert len(hits) == 1
    assert hits[0]["message_id"] == 100
    assert hits[0]["link"] == "https://t.me/teamchat/100"


# ===========================================================================
# Story 4.4 — code-review fix regressions (2026-07-22)
# ===========================================================================


def test_multiword_query_is_term_and_not_phrase(tmp_path: Path, monkeypatch) -> None:
    """[Review F7] A multi-word query matches messages containing BOTH lemmas.

    The earlier whole-query phrase-quoting required the lemmas to be ADJACENT and
    in order, so an ordinary two-word query returned nothing. The fix quotes each
    token individually (implicit FTS5 AND), so both messages that contain both
    lemmas — in either arrangement — are hits.
    """
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(message_id=100, text="обсудили бюджет и потом было совещание"),
            _make(message_id=101, text="совещание про бюджет"),
            _make(message_id=102, text="только бюджет без второго слова"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="бюджет совещание")
    ids = {h["message_id"] for h in hits}
    # Both messages with BOTH lemmas match (order/adjacency irrelevant); the
    # single-lemma message does NOT (term-AND requires both).
    assert ids == {100, 101}


def test_limit_not_undercut_by_tombstones(tmp_path: Path, monkeypatch) -> None:
    """[Review F1] Tombstones in the top-N do not shrink the live-hit count.

    With limit=2 and a tombstoned message ranked among the matches, the tool must
    still return 2 LIVE hits (over-fetch then slice), not 2-minus-tombstones.
    """
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100 + i, text=f"договор номер {i}") for i in range(5)],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    # Tombstone one matching message via the writer seam (ru_lemma registered).
    writer = schema.open_writer_with_schema(path)
    try:
        with db.writing(writer):
            schema.mark_deleted(writer, 1, 100)
    finally:
        writer.close()

    hits = _search(path, monkeypatch, query="договор", limit=2)
    assert len(hits) == 2  # full limit of LIVE hits, tombstone not counted
    assert all(h["message_id"] != 100 for h in hits)


def test_real_sql_error_propagates_not_masked(tmp_path: Path, monkeypatch) -> None:
    """[Review F5] A genuine DB error is NOT masked as an empty no-match.

    A malformed-FTS query maps to [] (Task 4), but any OTHER apsw.SQLError must
    propagate — reporting an honest failure as "no matches" would violate AC#6's
    honesty contract. Simulate a non-FTS SQLError from the search seam.
    """
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        def boom(*a, **k):
            raise apsw.SQLError("database disk image is malformed")

        monkeypatch.setattr(server.schema, "search_message_rowids", boom)
        with pytest.raises(apsw.SQLError):
            server._search_messages(app_ctx, query="договор")

    _run_lifespan(path, monkeypatch, check)


def test_malformed_fts_query_still_maps_to_empty(tmp_path: Path, monkeypatch) -> None:
    """[Review F5] A real FTS5 syntax error is still swallowed to [] (Task 4)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        def raise_fts(*a, **k):
            raise apsw.SQLError("fts5: syntax error near \"AND\"")

        monkeypatch.setattr(server.schema, "search_message_rowids", raise_fts)
        assert server._search_messages(app_ctx, query="договор") == []

    _run_lifespan(path, monkeypatch, check)


def test_orphan_chat_hit_surfaces_with_null_link(tmp_path: Path, monkeypatch) -> None:
    """[Review F6] A live message whose chat row is missing still surfaces (LEFT JOIN).

    It must NOT vanish like a tombstone; it appears with a null chat title and a
    null link, honestly, rather than being silently dropped.
    """
    path = tmp_path / "tgkb.sqlite3"
    # Ingest a message but NO matching chats row (no upsert_chat for chat_id 7).
    _seed_search_db(
        path,
        [_make(chat_id=7, message_id=100, text="осиротевший договор")],
        chats=[],
    )

    hits = _search(path, monkeypatch, query="договор")
    assert len(hits) == 1
    assert hits[0]["chat_id"] == 7
    assert hits[0]["chat"] is None
    assert hits[0]["link"] is None
    assert hits[0]["text"] == "осиротевший договор"


def test_limit_and_context_are_clamped(tmp_path: Path, monkeypatch) -> None:
    """[Review F8/F9] Huge limit/context are clamped, not passed through unbounded."""
    path = tmp_path / "tgkb.sqlite3"
    # Seed more messages than the max clamp so the cap is observable.
    msgs = [_make(message_id=100 + i, text=f"договор {i}") for i in range(server._MAX_LIMIT + 20)]
    _seed_search_db(path, msgs, chats=[(1, "Team Chat", "teamchat", True)])

    hits = _search(path, monkeypatch, query="договор", limit=10**9, context=10**9)
    # limit clamped to _MAX_LIMIT.
    assert len(hits) == server._MAX_LIMIT
    # context clamped: no hit carries more than 2*_MAX_CONTEXT_DEPTH neighbors.
    for h in hits:
        assert len(h["context"]) <= 2 * server._MAX_CONTEXT_DEPTH


def test_server_status_serialized_under_lock(tmp_path: Path, monkeypatch) -> None:
    """[Review F2] server_status acquires _DB_LOCK before opening its read.

    Behavioral proof that the shared-connection BEGIN race is closed: while the
    lock is held, a server_status call must block rather than open a second
    overlapping BEGIN on the one connection. We assert it cannot acquire the lock
    concurrently (non-blocking acquire fails while held).
    """
    import threading

    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="подписали договоры")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        # Hold the DB lock, then confirm _server_status would have to wait for it
        # (it does not run a lock-free read on the shared connection anymore).
        acquired_without_lock = {"value": None}

        def probe():
            got = server._DB_LOCK.acquire(blocking=False)
            acquired_without_lock["value"] = got
            if got:
                server._DB_LOCK.release()

        with server._DB_LOCK:
            t = threading.Thread(target=probe)
            t.start()
            t.join()
        # While held, the lock could not be acquired — proving server_status's read
        # is gated by the same lock (its body acquires _DB_LOCK).
        assert acquired_without_lock["value"] is False
        # And after release, server_status runs fine.
        status = server._server_status(app_ctx)
        assert status["message_count"] == 1

    _run_lifespan(path, monkeypatch, check)


# ===========================================================================
# Story 4.5 — guarded read_query tool (AC#1–#7)
# ===========================================================================


def _read(path: Path, monkeypatch, **kwargs):
    """Run the read_query core inside the real lifespan; return its result."""
    return _run_lifespan(
        path, monkeypatch, lambda app_ctx: server._read_query(app_ctx, **kwargs)
    )


# --- AC#1: a SELECT runs and returns column-keyed dict rows -----------------


def test_read_query_select_returns_rows(tmp_path: Path, monkeypatch) -> None:
    """A read-only SELECT returns real rows as column-keyed dicts from the real DB."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [
            _make(chat_id=1, message_id=100, sender_name="Alice", text="раз"),
            _make(chat_id=1, message_id=101, sender_name="Bob", text="два"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    rows = _read(
        path,
        monkeypatch,
        sql="SELECT chat_id, message_id, text FROM messages WHERE chat_id = 1 "
        "ORDER BY message_id",
    )
    assert rows == [
        {"chat_id": 1, "message_id": 100, "text": "раз"},
        {"chat_id": 1, "message_id": 101, "text": "два"},
    ]
    # Every value is a real stored value, keyed by the SELECTed column names.
    assert list(rows[0].keys()) == ["chat_id", "message_id", "text"]


def test_read_query_count_and_expressions(tmp_path: Path, monkeypatch) -> None:
    """count(*) / computed columns are keyed by the query's own column names."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100 + i, text=f"msg {i}") for i in range(3)],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    rows = _read(path, monkeypatch, sql="SELECT count(*) AS n FROM messages")
    assert rows == [{"n": 3}]


def test_read_query_explain_allowed(tmp_path: Path, monkeypatch) -> None:
    """EXPLAIN QUERY PLAN is read-only utility and is allowed (AC#1)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    rows = _read(
        path, monkeypatch, sql="EXPLAIN QUERY PLAN SELECT * FROM messages"
    )
    assert isinstance(rows, list) and len(rows) >= 1


def test_read_query_with_cte_select_allowed(tmp_path: Path, monkeypatch) -> None:
    """A read-only WITH … SELECT CTE is allowed (AC#1)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100 + i, text=f"msg {i}") for i in range(4)],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    rows = _read(
        path,
        monkeypatch,
        sql="WITH recent AS (SELECT message_id FROM messages) "
        "SELECT count(*) AS n FROM recent",
    )
    assert rows == [{"n": 4}]


# --- AC#5: empty result is an explicit empty list ---------------------------


def test_read_query_empty_result_is_empty_list(tmp_path: Path, monkeypatch) -> None:
    """A valid SELECT matching nothing returns [] exactly — not an error/placeholder."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    assert _read(path, monkeypatch, sql="SELECT * FROM messages WHERE 1 = 0") == []


# --- AC#2: writes and DDL rejected at the boundary, DB unchanged ------------


def _row_count(path: Path) -> int:
    conn = db.open_reader(path)
    try:
        return conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def test_read_query_rejects_write(tmp_path: Path, monkeypatch) -> None:
    """INSERT/UPDATE/DELETE/REPLACE → rejected at the boundary; DB unchanged (AC#2)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    before = _row_count(path)

    writes = [
        "INSERT INTO messages(chat_id, message_id, date_utc) "
        "VALUES (9, 9, '2026-01-01T00:00:00+00:00')",
        "UPDATE messages SET text = 'x'",
        "DELETE FROM messages",
        "REPLACE INTO messages(chat_id, message_id, date_utc) "
        "VALUES (1, 100, '2026-01-01T00:00:00+00:00')",
    ]

    def check(app_ctx: server.AppContext) -> None:
        for stmt in writes:
            with pytest.raises(server.ReadQueryRejected):
                server._read_query(app_ctx, sql=stmt)

    _run_lifespan(path, monkeypatch, check)
    assert _row_count(path) == before  # proves nothing executed


def test_read_query_rejects_ddl(tmp_path: Path, monkeypatch) -> None:
    """CREATE/DROP/ALTER/REINDEX/VACUUM/ATTACH → rejected; DB unchanged (AC#2)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    before = _row_count(path)

    ddl = [
        "CREATE TABLE evil(x)",
        "DROP TABLE messages",
        "ALTER TABLE messages ADD COLUMN evil TEXT",
        "REINDEX messages",
        "VACUUM",
        "ATTACH DATABASE 'x.db' AS other",
        "PRAGMA query_only = OFF",
    ]

    def check(app_ctx: server.AppContext) -> None:
        for stmt in ddl:
            with pytest.raises(server.ReadQueryRejected):
                server._read_query(app_ctx, sql=stmt)

    _run_lifespan(path, monkeypatch, check)
    assert _row_count(path) == before
    # Table still exists and is intact.
    conn = db.open_reader(path)
    try:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == before
    finally:
        conn.close()


# --- AC#4: multi-statement and comment-obfuscated writes rejected -----------


def test_read_query_rejects_multi_statement(tmp_path: Path, monkeypatch) -> None:
    """`SELECT 1; DROP TABLE messages` → rejected; messages table intact (AC#4)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    before = _row_count(path)

    def check(app_ctx: server.AppContext) -> None:
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(app_ctx, sql="SELECT 1; DROP TABLE messages")

    _run_lifespan(path, monkeypatch, check)
    # The trailing DROP never ran — the table exists and is intact.
    assert _row_count(path) == before


def test_read_query_trailing_semicolon_allowed(tmp_path: Path, monkeypatch) -> None:
    """A single statement with a trailing ';' is one statement, not multi (AC#4)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    assert _read(path, monkeypatch, sql="SELECT count(*) AS n FROM messages;") == [
        {"n": 1}
    ]


def test_read_query_rejects_comment_obfuscated_write(
    tmp_path: Path, monkeypatch
) -> None:
    """Comment-hidden leading token cannot smuggle a write past the guard (AC#4)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    before = _row_count(path)

    obfuscated = [
        "/* hi */ DELETE FROM messages",
        "-- c\nUPDATE messages SET text = 'x'",
        "/* leading */ INSERT INTO messages(chat_id, message_id, date_utc) "
        "VALUES (9, 9, '2026-01-01T00:00:00+00:00')",
    ]

    def check(app_ctx: server.AppContext) -> None:
        for stmt in obfuscated:
            with pytest.raises(server.ReadQueryRejected):
                server._read_query(app_ctx, sql=stmt)

    _run_lifespan(path, monkeypatch, check)
    assert _row_count(path) == before


def test_read_query_comment_stripping_preserves_string_literals(
    tmp_path: Path, monkeypatch
) -> None:
    """A comment marker INSIDE a string literal is data, not a comment (AC#4).

    `SELECT '--' AS c` must run and return the literal '--', proving the comment
    stripper is string-literal aware and does not corrupt legitimate SQL.
    """
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    rows = _read(path, monkeypatch, sql="SELECT '--' AS c, '/*' AS d")
    assert rows == [{"c": "--", "d": "/*"}]


def test_read_query_rejects_writing_cte(tmp_path: Path, monkeypatch) -> None:
    """A data-modifying WITH…(DELETE) CTE is rejected even though it starts WITH."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )
    before = _row_count(path)

    def check(app_ctx: server.AppContext) -> None:
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(
                app_ctx,
                sql="WITH gone AS (DELETE FROM messages RETURNING message_id) "
                "SELECT * FROM gone",
            )

    _run_lifespan(path, monkeypatch, check)
    assert _row_count(path) == before


def test_read_query_rejects_empty_and_unterminated(
    tmp_path: Path, monkeypatch
) -> None:
    """Empty query and un-strippable comment/string → rejected (fail closed)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        for stmt in ("", "   ", "/* never closed", "SELECT 'unterminated"):
            with pytest.raises(server.ReadQueryRejected):
                server._read_query(app_ctx, sql=stmt)

    _run_lifespan(path, monkeypatch, check)


# --- AC#3: read-only connection backstop (Layer 2, independent) -------------


def test_read_query_readonly_connection_backstop(
    tmp_path: Path, monkeypatch
) -> None:
    """A direct write on AppContext.conn raises ReadOnlyError even without the guard."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        # Bypass Layer 1 entirely: hit the connection directly. Layer 2 rejects it.
        for stmt in (
            "INSERT INTO messages(chat_id, message_id, date_utc) "
            "VALUES (9, 9, '2026-01-01T00:00:00+00:00')",
            "UPDATE messages SET text = 'x'",
        ):
            with pytest.raises(apsw.ReadOnlyError):
                app_ctx.conn.execute(stmt)

    _run_lifespan(path, monkeypatch, check)


# --- AC#7: limit bounds the result ------------------------------------------


def test_read_query_limit_caps_rows(tmp_path: Path, monkeypatch) -> None:
    """limit caps rows; huge limit clamped to _MAX_LIMIT; limit<=0 → [] (AC#7)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100 + i, text=f"msg {i}") for i in range(server._MAX_LIMIT + 20)],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    assert len(_read(path, monkeypatch, sql="SELECT * FROM messages", limit=5)) == 5
    # Huge limit clamped to the hard cap.
    assert (
        len(_read(path, monkeypatch, sql="SELECT * FROM messages", limit=10**9))
        == server._MAX_LIMIT
    )
    # Non-positive limit → no rows.
    assert _read(path, monkeypatch, sql="SELECT * FROM messages", limit=0) == []
    assert _read(path, monkeypatch, sql="SELECT * FROM messages", limit=-3) == []


# --- AC#6: read-only, short-transaction, stdout-clean, secret-safe ----------


def test_read_query_is_read_only_no_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """After a read_query the backstop still holds and a subsequent read succeeds (AC#6)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        server._read_query(app_ctx, sql="SELECT count(*) FROM messages")
        # Layer-2 backstop still holds — a write raises.
        with pytest.raises(apsw.ReadOnlyError):
            app_ctx.conn.execute("UPDATE messages SET text = 'x'")
        # No lingering open transaction: a fresh db.reading succeeds.
        with db.reading(app_ctx.conn) as conn:
            assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
        # And a subsequent read_query works — the lock was released.
        again = server._read_query(app_ctx, sql="SELECT count(*) AS n FROM messages")
        assert again == [{"n": 1}]

    _run_lifespan(path, monkeypatch, check)


def test_read_query_no_stdout_writes(tmp_path: Path, monkeypatch, capsys) -> None:
    """An allowed query AND a rejected query write NOTHING to stdout (AC#6)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        server._read_query(app_ctx, sql="SELECT count(*) FROM messages")
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(app_ctx, sql="DELETE FROM messages")

    _run_lifespan(path, monkeypatch, check)
    assert capsys.readouterr().out == ""


def test_read_query_secret_safety(tmp_path: Path, monkeypatch, capsys, caplog) -> None:
    """No api_hash/session/passphrase leaks on any read_query path (NFR-1)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        server._read_query(app_ctx, sql="SELECT count(*) FROM messages")
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(app_ctx, sql="DROP TABLE messages")

    with caplog.at_level("DEBUG", logger="tgkb"):
        _run_lifespan(path, monkeypatch, check)

    captured = capsys.readouterr()
    blob = captured.out + captured.err + " ".join(
        r.getMessage() for r in caplog.records
    )
    for s in ("api_hash", "passphrase", "session"):
        assert s not in blob


def test_read_query_bad_sql_is_clear_error_not_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    """A malformed read-only query surfaces a ReadQueryRejected, not a raw traceback."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(app_ctx, sql="SELECT no_such_column FROM messages")

    _run_lifespan(path, monkeypatch, check)


# --- Tool wrapper (async) routes through the core ---------------------------


def test_read_query_non_sqlerror_apsw_wrapped(tmp_path: Path, monkeypatch) -> None:
    """[Review] A non-SQLError apsw failure (e.g. BusyError) becomes a clear tool error.

    apsw.BusyError / apsw.ReadOnlyError do NOT derive from apsw.SQLError, so a
    narrow `except apsw.SQLError` would let them escape as a raw traceback,
    breaking the tool's "never a raw apsw traceback" contract. The broadened
    `except apsw.Error` wraps them in ReadQueryRejected. Simulate a BusyError from
    the connection's execute during a classified-safe read.
    """
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    class _BusyProxy:
        """Delegates BEGIN/COMMIT to the real conn but raises BusyError on the query.

        apsw.Connection.execute is read-only (not monkeypatchable), so we wrap the
        connection in a thin proxy for this one test. db.reading issues BEGIN/COMMIT
        which pass through; the guarded SELECT raises the non-SQLError apsw error.
        """

        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a, **k):
            if sql.strip().upper().startswith("SELECT"):
                raise apsw.BusyError("database is locked")
            return self._real.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def check(app_ctx: server.AppContext) -> None:
        proxied = server.AppContext(
            conn=_BusyProxy(app_ctx.conn), search_ok=app_ctx.search_ok
        )
        with pytest.raises(server.ReadQueryRejected) as exc:
            server._read_query(proxied, sql="SELECT * FROM messages")
        assert "could not execute" in str(exc.value)

    _run_lifespan(path, monkeypatch, check)


def test_read_query_non_string_sql_rejected(tmp_path: Path, monkeypatch) -> None:
    """[Review] A non-string sql (e.g. None) → ReadQueryRejected, not a raw TypeError."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        for bad in (None, 123, ["SELECT 1"]):
            with pytest.raises(server.ReadQueryRejected):
                server._read_query(app_ctx, sql=bad)  # type: ignore[arg-type]
        # A None limit falls back to the default and runs normally.
        assert server._read_query(
            app_ctx, sql="SELECT count(*) AS n FROM messages", limit=None  # type: ignore[arg-type]
        ) == [{"n": 1}]

    _run_lifespan(path, monkeypatch, check)


def test_read_query_async_wrapper(tmp_path: Path, monkeypatch) -> None:
    """The real async @mcp.tool wrapper reaches the lifespan ctx and returns rows."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="раз"), _make(message_id=101, text="два")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    monkeypatch.setattr(db, "default_db_path", lambda: path)

    async def drive():
        async with server.app_lifespan(server.mcp) as app_ctx:
            rc = RequestContext(
                request_id=1, meta=None, session=None, lifespan_context=app_ctx
            )
            ctx = Context(request_context=rc)
            return await server.read_query(ctx, sql="SELECT count(*) AS n FROM messages")

    rows = asyncio.run(drive())
    assert rows == [{"n": 2}]


# ===========================================================================
# Story 4.6 — Honest retrieval contract and tool descriptions (AC#1–#4)
#
# A CONTRACT / DOCUMENTATION story: 4.4 (search_messages) and 4.5 (read_query)
# already SHIP the honest-retrieval mechanics (real-rows-only, [] on no-match,
# refusal != empty). This story adds the caller-facing PROSE contract into the
# tool descriptions (AD-9 final clause) and LOCKS the honesty invariants as a
# permanent, cross-tool contract so a future change cannot silently erode FR-13.
# ===========================================================================


# Stable honesty-contract phrases (Task 1): the tool descriptions MUST carry
# these so the MCP client (Claude Code) is instructed to ground answers in the
# returned rows. Asserted below so a future docstring edit that drops the
# contract fails a test. Matched case-insensitively on the description text.
_ANSWER_ONLY_PHRASE = "answer only from the returned"
_EMPTY_HONESTY_PHRASE = "no matching messages"


def _tool_descriptions() -> dict[str, str]:
    """Return {tool_name: description} as the MCP client would receive them.

    Drives the REAL FastMCP ``mcp.list_tools()`` (async, so via ``asyncio.run`` —
    no ``pytest-asyncio``) so the assertion is on the client-visible tool
    metadata, not merely on ``fn.__doc__``. FastMCP sources each tool's
    ``description`` from its ``@mcp.tool`` function docstring.
    """
    tools = asyncio.run(server.mcp.list_tools())
    return {t.name: (t.description or "") for t in tools}


# --- AC#3: tool descriptions ship the honest-retrieval contract -------------


def test_search_messages_description_ships_honest_contract() -> None:
    """search_messages' registered description instructs answer-only-from-returned."""
    desc = _tool_descriptions()["search_messages"].lower()
    assert _ANSWER_ONLY_PHRASE in desc
    assert _EMPTY_HONESTY_PHRASE in desc


def test_read_query_description_ships_honest_contract() -> None:
    """read_query's registered description instructs answer-only-from-returned."""
    desc = _tool_descriptions()["read_query"].lower()
    assert _ANSWER_ONLY_PHRASE in desc
    # read_query returns structured rows; its empty-set instruction speaks of rows.
    # Assert the honesty-specific phrase from the 4.6 contract paragraph — NOT the
    # bare word "empty", which already appears in the pre-4.6 docstring ("matching
    # nothing returns an explicit empty list") and would pass even if the whole 4.6
    # honesty paragraph were deleted.
    assert "nothing matched" in desc


def test_both_tools_ship_answer_only_from_returned() -> None:
    """AD-9: the server ships tool descriptions telling the caller to answer only
    from returned rows — verified as ONE cross-tool contract on both query tools."""
    descs = _tool_descriptions()
    for name in ("search_messages", "read_query"):
        assert _ANSWER_ONLY_PHRASE in descs[name].lower(), name


# --- AC#1: every returned row is a REAL row (no synthesis) -------------------


def test_search_returns_only_real_ingested_rows(tmp_path: Path, monkeypatch) -> None:
    """Every returned hit's identity corresponds to an actually-ingested message —
    a fabricated/defaulted row would fail this (FR-13 real-rows-only, locked)."""
    path = tmp_path / "tgkb.sqlite3"
    seeded = {(1, 100), (1, 101), (1, 102)}
    _seed_search_db(
        path,
        [
            _make(message_id=100, text="договор один"),
            _make(message_id=101, text="договор два"),
            _make(message_id=102, text="ничего про это"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    hits = _search(path, monkeypatch, query="договор")
    assert hits  # matches exist
    for hit in hits:
        assert (hit["chat_id"], hit["message_id"]) in seeded


def test_read_query_returns_only_real_ingested_values(
    tmp_path: Path, monkeypatch
) -> None:
    """read_query rows are real stored values, not synthesized (FR-13, locked)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="реальный текст сообщения")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        rows = server._read_query(
            app_ctx, sql="SELECT message_id, text FROM messages"
        )
        assert rows == [{"message_id": 100, "text": "реальный текст сообщения"}]

    _run_lifespan(path, monkeypatch, check)


# --- AC#2: empty = [] on genuine no-match, for BOTH tools --------------------


def test_both_tools_return_empty_list_on_genuine_no_match(
    tmp_path: Path, monkeypatch
) -> None:
    """Consistency Convention: absence is an empty set (not error, not placeholder)
    — verified as ONE cross-tool invariant on a GENUINE no-match (not an error)."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="совещание про бюджет")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        # search_messages: a valid query that matches nothing → exactly [].
        assert server._search_messages(app_ctx, query="абсолютнонесуществующееслово") == []
        # read_query: a valid read-only query that matches nothing → exactly [].
        assert (
            server._read_query(
                app_ctx, sql="SELECT message_id FROM messages WHERE 1 = 0"
            )
            == []
        )

    _run_lifespan(path, monkeypatch, check)


# --- AC#2: refusal != empty, for BOTH tools ---------------------------------


def test_refusal_is_distinct_from_empty_for_both_tools(
    tmp_path: Path, monkeypatch
) -> None:
    """A refusal (stale index / rejected SQL) is RAISED and distinct from [].

    The crux of honesty: an empty set means "looked correctly, found nothing";
    a refusal means "cannot/will not answer" — neither may masquerade as the
    other. search_messages refuses a stale index (SearchIndexStaleError);
    read_query refuses a non-read-only statement (ReadQueryRejected). Both are
    exported refusal classes, both RAISE (never return [])."""
    path = tmp_path / "tgkb.sqlite3"
    _seed_search_db(
        path,
        [_make(message_id=100, text="договор подписан")],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        # search_messages REFUSES on a stale index (search_ok False) — a raise,
        # NOT an empty list.
        stale_ctx = server.AppContext(conn=app_ctx.conn, search_ok=False)
        with pytest.raises(server.SearchIndexStaleError):
            server._search_messages(stale_ctx, query="договор")
        # ... and a healthy index still returns real hits (proves refusal is the
        # stale path, not the query).
        assert server._search_messages(app_ctx, query="договор")

        # read_query REFUSES a write/DDL statement — a raise, NOT an empty list.
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(app_ctx, sql="DELETE FROM messages")

    _run_lifespan(path, monkeypatch, check)

    # The two refusal classes are distinct exported types (conceptually paired,
    # never the same as the empty-list return).
    assert issubclass(server.SearchIndexStaleError, RuntimeError)
    assert issubclass(server.ReadQueryRejected, RuntimeError)
    assert server.SearchIndexStaleError is not server.ReadQueryRejected


# --- AC#4: SM-1 tool-boundary honesty slice ---------------------------------


def test_sm1_tool_boundary_honesty_slice(tmp_path: Path, monkeypatch) -> None:
    """The tool-boundary half of the SM-1 thin slice: a matching query returns
    real cited rows (each id is a seeded id — no invented row); a non-matching
    query returns exactly []; a refusal path REFUSES. The model-side "answer only
    from returned rows" is out of code scope (the tool ships the instruction)."""
    path = tmp_path / "tgkb.sqlite3"
    seeded_ids = {100, 101}
    _seed_search_db(
        path,
        [
            _make(message_id=100, text="обсудили договор с поставщиком"),
            _make(message_id=101, text="прислали договоры на подпись"),
        ],
        chats=[(1, "Team Chat", "teamchat", True)],
    )

    def check(app_ctx: server.AppContext) -> None:
        # (a) a matching query → real cited rows, every id genuinely seeded.
        hits = server._search_messages(app_ctx, query="договор")
        assert hits
        for hit in hits:
            assert hit["message_id"] in seeded_ids
            assert hit["text"]  # a real, non-empty stored message
        # (b) a non-matching query → an honest empty set.
        assert server._search_messages(app_ctx, query="квантоваяхромодинамика") == []
        # (c) a refusal path (rejected SQL) → RAISES, never a silent empty.
        with pytest.raises(server.ReadQueryRejected):
            server._read_query(app_ctx, sql="DROP TABLE messages")

    _run_lifespan(path, monkeypatch, check)


def test_no_stdout_on_list_tools(capsys) -> None:
    """Listing tools (reading their descriptions) writes nothing to stdout — the
    honesty contract lives in metadata, never on the protocol stdout stream."""
    _tool_descriptions()
    captured = capsys.readouterr()
    assert captured.out == ""
