"""Tests for the schema + idempotent ingest layer (Story 2.2, ``tgkb/common/schema.py``).

No network — pure SQLite. Every test uses ``tmp_path`` + ``db.open_writer`` for a
real APSW file (schema behavior needs a real DB, unlike config's pure-stdlib
tests). Tests exercise *behavior* (real table/column/PK shape, count-and-content
after re-ingest, edited-vs-preserved columns), not flags — per the review lesson
carried 1.1 → 2.1.
"""

from __future__ import annotations

from pathlib import Path

import apsw
import pytest

from tgkb.common import db, schema
from tgkb.common.schema import Message


def _open(tmp_path: Path) -> apsw.Connection:
    """Open a real writer against a fresh tmp DB and create the schema."""
    conn = db.open_writer(tmp_path / "tgkb.sqlite3")
    schema.ensure_schema(conn)
    return conn


def _table_names(conn: apsw.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _ingest(conn: apsw.Connection, msg: Message) -> None:
    with db.writing(conn):
        schema.ingest_message(conn, msg)


def _make(
    *,
    chat_id: int = 1,
    message_id: int = 100,
    sender_id: int | None = 42,
    sender_name: str | None = "Alice",
    text: str | None = "hello",
    date_utc: str = "2026-07-19T14:03:00+00:00",
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


# --- AC#1: schema shape -----------------------------------------------------


def test_ensure_schema_creates_three_tables(tmp_path: Path) -> None:
    """AC#1: ensure_schema creates chats/messages/sync_state with the right shape."""
    conn = _open(tmp_path)
    try:
        names = _table_names(conn)
        assert {"chats", "messages", "sync_state"} <= names
        # messages_fts is created by Story 3.2 (external-content FTS5 over
        # messages) — behavior tested in tests/test_fts.py.
        assert "messages_fts" in names

        # chats columns.
        chats_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()
        }
        assert chats_cols == {"chat_id", "title", "username", "is_public"}

        # messages columns — the exact set.
        msg_info = conn.execute("PRAGMA table_info(messages)").fetchall()
        msg_cols = {r[1] for r in msg_info}
        assert msg_cols == {
            "chat_id",
            "message_id",
            "sender_id",
            "sender_name",
            "text",
            "date_utc",
            "reply_to_msg_id",
            "edit_date",
            "deleted",
        }

        # sync_state columns.
        ss_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(sync_state)").fetchall()
        }
        assert ss_cols == {"chat_id", "last_id", "updated_at"}
    finally:
        conn.close()


def test_messages_has_composite_pk(tmp_path: Path) -> None:
    """AC#1/AC#5: messages PK is composite (chat_id, message_id)."""
    conn = _open(tmp_path)
    try:
        # PRAGMA table_info: the `pk` column (index 5) is 0 for non-PK, and the
        # 1-based position within the PK for PK columns. Both chat_id and
        # message_id must have pk > 0 → composite PK.
        info = conn.execute("PRAGMA table_info(messages)").fetchall()
        pk = {r[1]: r[5] for r in info}
        assert pk["chat_id"] > 0
        assert pk["message_id"] > 0
        # No other column is part of the PK.
        assert all(v == 0 for k, v in pk.items() if k not in ("chat_id", "message_id"))

        # Not WITHOUT ROWID — the implicit rowid must survive for Story 3.2's FTS
        # (external-content FTS5 pins a rowid ↔ (chat_id, message_id) mapping,
        # AD-12). Assert the rowid is actually selectable: a WITHOUT ROWID table
        # would raise "no such column: rowid" here. (An index_list check is NOT
        # sufficient — a WITHOUT ROWID composite-PK table also reports a unique
        # pk auto-index, so it would pass for the wrong schema.)
        with db.writing(conn):
            schema.ingest_message(conn, _make())
        assert (
            conn.execute(
                "SELECT rowid FROM messages WHERE chat_id=1 AND message_id=100"
            ).fetchone()[0]
            is not None
        )
    finally:
        conn.close()


def test_ensure_schema_idempotent(tmp_path: Path) -> None:
    """AC#1: calling ensure_schema twice is a no-op (no raise, one set of tables)."""
    conn = db.open_writer(tmp_path / "tgkb.sqlite3")
    try:
        schema.ensure_schema(conn)
        schema.ensure_schema(conn)  # must not raise
        # Still exactly one of each table.
        for t in ("chats", "messages", "sync_state"):
            n = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                (t,),
            ).fetchone()[0]
            assert n == 1
    finally:
        conn.close()


# --- AC#2: idempotent insert (no duplicate, text untouched) -----------------


def test_insert_then_reinsert_same_message_no_duplicate(tmp_path: Path) -> None:
    """AC#2: re-ingesting the identical message never duplicates or clobbers text."""
    conn = _open(tmp_path)
    try:
        msg = _make(text="original")
        _ingest(conn, msg)
        _ingest(conn, msg)  # identical (same edit_date=None)

        rows = conn.execute(
            "SELECT text FROM messages WHERE chat_id=1 AND message_id=100"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "original"
    finally:
        conn.close()


# --- AC#3: edit updates text + edit_date only, identity preserved -----------


def test_edit_updates_text_and_edit_date(tmp_path: Path) -> None:
    """AC#3: a changed edit_date updates text + edit_date, preserves everything else.

    Covers the NULL → value edit_date transition (the `IS NOT` null-safety trap).
    """
    conn = _open(tmp_path)
    try:
        _ingest(
            conn,
            _make(
                text="v1",
                edit_date=None,
                sender_id=42,
                date_utc="2026-07-19T14:03:00+00:00",
                reply_to_msg_id=7,
            ),
        )
        _ingest(
            conn,
            _make(
                text="v2-edited",
                edit_date="2026-07-19T15:00:00+00:00",
                # These would-be changes MUST be ignored on edit:
                sender_id=999,
                date_utc="2000-01-01T00:00:00+00:00",
                reply_to_msg_id=999,
            ),
        )

        row = conn.execute(
            "SELECT text, edit_date, sender_id, date_utc, reply_to_msg_id "
            "FROM messages WHERE chat_id=1 AND message_id=100"
        ).fetchall()
        assert len(row) == 1
        text, edit_date, sender_id, date_utc, reply_to = row[0]
        # Edited columns changed.
        assert text == "v2-edited"
        assert edit_date == "2026-07-19T15:00:00+00:00"
        # Non-edited columns preserved from the FIRST ingest (AC#3 identity).
        assert sender_id == 42
        assert date_utc == "2026-07-19T14:03:00+00:00"
        assert reply_to == 7
    finally:
        conn.close()


def test_edit_value_to_value_transition(tmp_path: Path) -> None:
    """AC#3: a value → different-value edit_date change also updates text."""
    conn = _open(tmp_path)
    try:
        _ingest(conn, _make(text="v1", edit_date="2026-07-19T15:00:00+00:00"))
        _ingest(conn, _make(text="v2", edit_date="2026-07-19T16:00:00+00:00"))
        row = conn.execute(
            "SELECT text, edit_date FROM messages WHERE chat_id=1 AND message_id=100"
        ).fetchone()
        assert row == ("v2", "2026-07-19T16:00:00+00:00")
    finally:
        conn.close()


def test_unchanged_edit_date_is_noop(tmp_path: Path) -> None:
    """AC#2/#3 boundary: same non-null edit_date but different text → NOT updated.

    Proves the `WHERE excluded.edit_date IS NOT messages.edit_date` gate: an edit
    is detected by edit_date changing, so a re-ingest with the same edit_date must
    leave the stored text untouched.
    """
    conn = _open(tmp_path)
    try:
        _ingest(conn, _make(text="first", edit_date="2026-07-19T15:00:00+00:00"))
        # Same edit_date, different text — the gate must reject the update.
        _ingest(conn, _make(text="sneaky-change", edit_date="2026-07-19T15:00:00+00:00"))
        row = conn.execute(
            "SELECT text FROM messages WHERE chat_id=1 AND message_id=100"
        ).fetchone()
        assert row[0] == "first"
    finally:
        conn.close()


# --- AC#4: UTC storage is timezone-transparent ------------------------------


def test_utc_timestamps_stored_verbatim(tmp_path: Path) -> None:
    """AC#4: the storage layer stores the UTC string it is given, unmangled."""
    conn = _open(tmp_path)
    try:
        stamp = "2026-07-19T14:03:00+00:00"
        _ingest(conn, _make(date_utc=stamp, edit_date=None))
        got = conn.execute(
            "SELECT date_utc FROM messages WHERE chat_id=1 AND message_id=100"
        ).fetchone()[0]
        assert got == stamp
    finally:
        conn.close()


# --- AC#5: message ids are per-chat, not global -----------------------------


def test_same_message_id_different_chats_distinct_rows(tmp_path: Path) -> None:
    """AC#5: same message_id in different chats → two independent rows."""
    conn = _open(tmp_path)
    try:
        _ingest(conn, _make(chat_id=1, message_id=100, text="in-chat-1"))
        _ingest(conn, _make(chat_id=2, message_id=100, text="in-chat-2"))

        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
        assert (
            conn.execute(
                "SELECT text FROM messages WHERE chat_id=1 AND message_id=100"
            ).fetchone()[0]
            == "in-chat-1"
        )
        assert (
            conn.execute(
                "SELECT text FROM messages WHERE chat_id=2 AND message_id=100"
            ).fetchone()[0]
            == "in-chat-2"
        )

        # Editing chat 1's row must not touch chat 2's.
        _ingest(
            conn,
            _make(
                chat_id=1,
                message_id=100,
                text="chat-1-edited",
                edit_date="2026-07-19T15:00:00+00:00",
            ),
        )
        assert (
            conn.execute(
                "SELECT text FROM messages WHERE chat_id=2 AND message_id=100"
            ).fetchone()[0]
            == "in-chat-2"
        )
    finally:
        conn.close()


# --- transaction composability + rollback -----------------------------------


def test_ingest_commits_and_rolls_back(tmp_path: Path) -> None:
    """Contract: ingest runs inside db.writing(); commit persists, rollback discards.

    Modeled on tests/test_db.py::test_writer_begin_immediate_commits_and_rolls_back.
    This proves ingest is transaction-composable (AD-12) — it does NOT open its
    own transaction; the caller's writing() scope governs atomicity.
    """
    conn = _open(tmp_path)
    try:
        # Clean exit → committed.
        with db.writing(conn):
            schema.ingest_message(conn, _make(message_id=1, text="kept"))
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1

        # Exception inside the block → rolled back.
        with pytest.raises(RuntimeError):
            with db.writing(conn):
                schema.ingest_message(conn, _make(message_id=2, text="discarded"))
                raise RuntimeError("boom")
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_open_writer_with_schema(tmp_path: Path) -> None:
    """The Story 2.4 DB-open seam opens the writer AND creates the schema in one step."""
    path = tmp_path / "tgkb.sqlite3"
    conn = schema.open_writer_with_schema(path)
    try:
        # Schema is present without a separate ensure_schema call.
        assert {"chats", "messages", "sync_state"} <= _table_names(conn)
        # And it is a live writer connection.
        with db.writing(conn):
            schema.ingest_message(conn, _make())
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_open_writer_with_schema_closes_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ensure_schema fails, the just-opened connection is closed (no leak).

    Exercises the close-on-failure guard — the one piece of real error handling
    the seam adds. We force ensure_schema to raise and assert the connection the
    helper opened was closed (a closed APSW connection raises on use).
    """
    boom = RuntimeError("schema creation failed")
    opened: list[apsw.Connection] = []
    real_open_writer = db.open_writer

    def _spy_open_writer(p: db.PathLike) -> apsw.Connection:
        conn = real_open_writer(p)
        opened.append(conn)
        return conn

    monkeypatch.setattr(schema.db, "open_writer", _spy_open_writer)
    monkeypatch.setattr(schema, "ensure_schema", lambda conn: (_ for _ in ()).throw(boom))

    with pytest.raises(RuntimeError, match="schema creation failed"):
        schema.open_writer_with_schema(tmp_path / "tgkb.sqlite3")

    assert len(opened) == 1
    # The opened connection must have been closed by the guard — using a closed
    # APSW connection raises.
    with pytest.raises(apsw.ConnectionClosedError):
        opened[0].execute("SELECT 1")


def test_ingest_messages_batch(tmp_path: Path) -> None:
    """Batch ingest inserts many rows in one writing() scope (used by Story 2.4)."""
    conn = _open(tmp_path)
    try:
        msgs = [_make(message_id=i, text=f"m{i}") for i in range(1, 6)]
        with db.writing(conn):
            schema.ingest_messages(conn, msgs)
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 5
    finally:
        conn.close()


# --- helpers: upsert_chat + advance_last_id ---------------------------------


def test_upsert_chat(tmp_path: Path) -> None:
    """upsert_chat inserts then updates in place (used by Story 2.4)."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.upsert_chat(conn, 1, "Old Title", "oldname", True)
        with db.writing(conn):
            schema.upsert_chat(conn, 1, "New Title", "newname", False)

        rows = conn.execute(
            "SELECT chat_id, title, username, is_public FROM chats"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == (1, "New Title", "newname", 0)
    finally:
        conn.close()


def test_upsert_chat_nullable_username(tmp_path: Path) -> None:
    """upsert_chat accepts a NULL username (a private chat has no @handle)."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.upsert_chat(conn, 5, "Private", None, False)
        row = conn.execute(
            "SELECT username, is_public FROM chats WHERE chat_id=5"
        ).fetchone()
        assert row == (None, 0)
    finally:
        conn.close()


def test_advance_last_id_monotonic(tmp_path: Path) -> None:
    """AD-5: last_id is a monotonic high-water mark — it never regresses."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.advance_last_id(conn, 1, 100, "2026-07-19T15:00:00+00:00")
        # A lower value must NOT regress the high-water mark.
        with db.writing(conn):
            schema.advance_last_id(conn, 1, 50, "2026-07-19T16:00:00+00:00")
        row = conn.execute(
            "SELECT last_id, updated_at FROM sync_state WHERE chat_id=1"
        ).fetchone()
        assert row[0] == 100  # stayed at the max
        assert row[1] == "2026-07-19T16:00:00+00:00"  # updated_at still advances

        # A higher value advances it.
        with db.writing(conn):
            schema.advance_last_id(conn, 1, 200, "2026-07-19T17:00:00+00:00")
        assert (
            conn.execute(
                "SELECT last_id FROM sync_state WHERE chat_id=1"
            ).fetchone()[0]
            == 200
        )
    finally:
        conn.close()
