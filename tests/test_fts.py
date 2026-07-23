"""Tests for the FTS5 index maintained in the write transaction (Story 3.2).

The real ``messages_fts`` external-content index + its ``AFTER INSERT/UPDATE/
DELETE`` triggers, over the REAL schema (Story 3.1 proved the tokenizer over an
in-memory table; here it is the actual index). No network — a real APSW file via
``tmp_path`` + ``db.open_writer`` + ``schema.ensure_schema`` so the real DDL path
and writer PRAGMAs are exercised.

Behavior-first (per the review lesson carried 1.1 → 3.1): every test asserts a
real ``MATCH`` count / row identity / BM25 order / rollback-drops-both — never a
"table exists" or "trigger created" flag. Expected lemmas are pinned to what
``pymorphy3==2.0.6`` + ``pymorphy3-dicts-ru==2.4.417150.4580142`` actually return
(the ``договоры``/``договора``/``договор`` family shares ``договор``), verified
against the pinned dict, not from memory.
"""

from __future__ import annotations

from pathlib import Path

import apsw
import pytest

from tgkb.common import db, ru_lemma, schema
from tgkb.common.schema import Message


def _open(tmp_path: Path) -> apsw.Connection:
    """Open a real writer against a fresh tmp DB and create the full schema.

    ``ensure_schema`` registers ``ru_lemma`` and creates ``messages_fts`` + its
    triggers, so the returned connection can ingest (fires triggers) and ``MATCH``.
    """
    conn = db.open_writer(tmp_path / "tgkb.sqlite3")
    schema.ensure_schema(conn)
    return conn


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


def _match_count(conn: apsw.Connection, query: str) -> int:
    """Rows found by an index-backed MATCH for ``query`` (query is lemmatized)."""
    return len(schema.search_message_rowids(conn, query))


# --- AC#1: messages_fts is an external-content index over messages ----------


def test_messages_fts_is_external_content_over_messages(tmp_path: Path) -> None:
    """AC#1: messages_fts exists as an FTS5 table; the external-content mapping is
    behaviorally proven by inserting a messages row and finding it via the join on
    rowid (not just string-grepping sqlite_master)."""
    conn = _open(tmp_path)
    try:
        # Behavioral proof of content='messages'/content_rowid='rowid': a plain
        # INSERT into `messages` (via ingest) is found through messages_fts by the
        # rowid join in search_message_rowids.
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="привет договор"))
        rowids = schema.search_message_rowids(conn, "договор")
        assert len(rowids) == 1
        # The FTS rowid is the messages rowid (external-content mapping).
        msg_rowid = conn.execute(
            "SELECT rowid FROM messages WHERE chat_id=1 AND message_id=100"
        ).fetchone()[0]
        assert rowids[0] == msg_rowid

        # Additionally (allowed): the DDL declares external content + single-sourced
        # tokenizer spec.
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages_fts'"
        ).fetchone()[0]
        assert "content='messages'" in create_sql
        assert "content_rowid='rowid'" in create_sql
        assert f"{ru_lemma.TOKENIZER_NAME} {ru_lemma.BASE_TOKENIZER}" in create_sql
    finally:
        conn.close()


def test_tokenize_spec_single_sourced(tmp_path: Path) -> None:
    """AC#1: the CREATE VIRTUAL TABLE tokenize= clause is built from
    ru_lemma.TOKENIZE_SPEC / the named constants, not a hand-typed literal.

    Mirrors test_ru_lemma.py::test_tokenizer_name_is_single_sourced and
    test_throttle.py::test_default_rate_is_single_sourced.
    """
    conn = _open(tmp_path)
    try:
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages_fts'"
        ).fetchone()[0]
        # The exact pairing the single source yields must appear verbatim.
        assert " ".join(ru_lemma.TOKENIZE_SPEC) in create_sql
        # And the DDL constant itself is derived from the named constants.
        assert (
            f'tokenize="{ru_lemma.TOKENIZER_NAME} {ru_lemma.BASE_TOKENIZER}"'
            in schema._CREATE_MESSAGES_FTS
        )
    finally:
        conn.close()


# --- AC#2: index maintained in the SAME transaction as the message write ----


def test_ingest_indexes_in_same_transaction(tmp_path: Path) -> None:
    """AC#2: an INSERT ingest makes the row MATCH-able after commit — the INSERT
    trigger fired inside the ingest transaction (no separate indexing pass)."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="подписали договоры вчера"))
        assert _match_count(conn, "договор") == 1
    finally:
        conn.close()


def test_ingest_rollback_leaves_fts_empty(tmp_path: Path) -> None:
    """AC#2 (atomicity, the strongest AD-12 proof): a rollback drops the row from
    BOTH messages AND messages_fts — the FTS write is truly in the same
    transaction, so it rolls back together."""
    conn = _open(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with db.writing(conn):
                schema.ingest_message(conn, _make(text="секретный договор"))
                raise RuntimeError("boom")
        # Rolled back: absent from messages...
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        # ...and absent from the FTS index (a MATCH returns 0).
        assert _match_count(conn, "договор") == 0
    finally:
        conn.close()


def test_edit_upsert_updates_fts(tmp_path: Path) -> None:
    """AC#2: an edit-upsert (changed edit_date) re-indexes the new text AND drops
    the old text's entries — the UPDATE trigger's delete-OLD-then-insert-NEW, no
    orphan. Exercises the 'delete' OLD.text correctness."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="старый договор", edit_date=None))
        assert _match_count(conn, "договор") == 1

        # Re-ingest same (chat_id, message_id) with a CHANGED edit_date + new text.
        with db.writing(conn):
            schema.ingest_message(
                conn,
                _make(text="новый контракт", edit_date="2026-07-19T15:00:00+00:00"),
            )
        # New text is findable...
        assert _match_count(conn, "контракт") == 1
        # ...and a lemma unique to the OLD text is gone (no orphaned entries).
        assert _match_count(conn, "договор") == 0
    finally:
        conn.close()


def test_reingest_unchanged_no_fts_change(tmp_path: Path) -> None:
    """AC#2: re-ingesting an identical message (same edit_date) fires NO UPDATE
    (the WHERE gate is false) → no trigger → no FTS churn, no error, count stable."""
    conn = _open(tmp_path)
    try:
        msg = _make(text="договор один", edit_date="2026-07-19T15:00:00+00:00")
        with db.writing(conn):
            schema.ingest_message(conn, msg)
        before = schema.search_message_rowids(conn, "договор")
        assert len(before) == 1

        # Identical re-ingest: same edit_date → DO NOTHING → no trigger.
        with db.writing(conn):
            schema.ingest_message(conn, msg)
        after = schema.search_message_rowids(conn, "договор")
        assert after == before  # same single rowid, no duplicate
    finally:
        conn.close()


# --- AC#3: Russian morphology ------------------------------------------------


def test_search_matches_russian_inflections(tmp_path: Path) -> None:
    """AC#3 (FR-7 payoff over the REAL schema): a corpus of inflected forms
    ('договоры', 'договора') is all returned by a MATCH for 'договор'."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(message_id=1, text="есть договоры"))
            schema.ingest_message(conn, _make(message_id=2, text="нет договора"))
        assert _match_count(conn, "договор") == 2
        # And querying with yet another inflection finds both too.
        assert _match_count(conn, "договорами") == 2
    finally:
        conn.close()


# --- AC#4: BM25 ranking, index-backed ---------------------------------------


def test_search_is_bm25_ranked(tmp_path: Path) -> None:
    """AC#4: the more-relevant row sorts first under ORDER BY bm25(messages_fts).

    A row where the term dominates (short text, term repeated) is more relevant
    than a long row where it appears once — BM25 ranks the former first.
    """
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            # message_id=1: term buried in a long, mostly-unrelated message.
            schema.ingest_message(
                conn,
                _make(
                    message_id=1,
                    text="сегодня погода отличная и мы гуляли долго в парке договор",
                ),
            )
            # message_id=2: short, term-dense → more relevant.
            schema.ingest_message(
                conn, _make(message_id=2, text="договор договор")
            )
        rowids = schema.search_message_rowids(conn, "договор")
        assert len(rowids) == 2
        # Map rowids back to message_id; the dense short one (id=2) ranks first.
        first_msg_id = conn.execute(
            "SELECT message_id FROM messages WHERE rowid=?", (rowids[0],)
        ).fetchone()[0]
        assert first_msg_id == 2
    finally:
        conn.close()


def test_fts_uses_index_not_full_scan(tmp_path: Path) -> None:
    """NFR-4: the MATCH query is index-backed — EXPLAIN QUERY PLAN uses the FTS5
    virtual table, never a SCAN of messages."""
    conn = _open(tmp_path)
    try:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN " + schema._SEARCH_ROWIDS_SQL, ("договор",)
        ).fetchall()
        detail = " ".join(str(row[-1]) for row in plan)
        # The FTS5 virtual table drives the query (the join alias `f` is
        # messages_fts) — an index-backed MATCH, not a linear scan.
        assert "VIRTUAL TABLE INDEX" in detail
        # `messages` is reached only by its rowid (the external-content lookup),
        # never a full table scan.
        assert "SCAN messages" not in detail
        assert "SCAN m" not in detail
    finally:
        conn.close()


# --- AC#5: incremental searchability, no reindex ----------------------------


def test_newly_ingested_searchable_no_reindex(tmp_path: Path) -> None:
    """AC#5: a message ingested in a LATER separate transaction is searchable with
    no rebuild call, and the earlier message is still found — incremental index."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(message_id=1, text="первый договор"))
        assert _match_count(conn, "договор") == 1

        # Separate later transaction, no rebuild.
        with db.writing(conn):
            schema.ingest_message(conn, _make(message_id=2, text="второй контракт"))
        assert _match_count(conn, "контракт") == 1  # new one searchable
        assert _match_count(conn, "договор") == 1  # old one still found
    finally:
        conn.close()


# --- the reader-registration seam (open_reader_with_fts) --------------------


def test_reader_can_match_through_fts_seam(tmp_path: Path) -> None:
    """The Story 3.2 reader seam: write rows via the writer, then open a read-only
    connection via open_reader_with_fts and MATCH them — proving the seam registers
    ru_lemma so a read-only connection can query the shared index (recall holds)."""
    path = tmp_path / "tgkb.sqlite3"
    writer = schema.open_writer_with_schema(path)
    try:
        with db.writing(writer):
            schema.ingest_message(writer, _make(text="общий договор"))
        db.checkpoint(writer)
    finally:
        writer.close()

    reader = schema.open_reader_with_fts(path)
    try:
        # Query-side lemmatization agrees with index-side (registered by the seam).
        assert _match_count(reader, "договор") == 1
        # And the connection is genuinely read-only (writes rejected).
        with pytest.raises(apsw.ReadOnlyError):
            reader.execute("INSERT INTO messages_fts(rowid, text) VALUES (99, 'x')")
    finally:
        reader.close()


# --- search helper boundaries (review hardening) ----------------------------


def test_search_limit_caps_and_guards_non_positive(tmp_path: Path) -> None:
    """search_message_rowids: a positive limit caps results; limit<=0 returns [].

    Guards the negative-LIMIT footgun — SQLite treats a negative LIMIT as
    "unlimited", so without the guard limit=-1 would return ALL rows instead of
    none. limit=0 and limit=-1 must both mean "no rows".
    """
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            for i in range(1, 4):
                schema.ingest_message(conn, _make(message_id=i, text="договор"))
        assert len(schema.search_message_rowids(conn, "договор")) == 3  # no cap
        assert len(schema.search_message_rowids(conn, "договор", 2)) == 2  # positive cap
        assert schema.search_message_rowids(conn, "договор", 0) == []  # zero → none
        assert schema.search_message_rowids(conn, "договор", -1) == []  # neg → none, NOT all
    finally:
        conn.close()


def test_null_text_message_is_indexable_and_no_orphan(tmp_path: Path) -> None:
    """A media/service message with text=None flows through the triggers safely.

    `messages.text` is nullable (Message.text is str | None). Ingesting a NULL-text
    row, then a text row, must leave search working; editing the NULL row to carry
    text must make it findable; and an FTS5 'integrity-check' must pass (no orphaned
    index entries from tokenizing NULL on insert vs delete).
    """
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(message_id=1, text=None))
            schema.ingest_message(conn, _make(message_id=2, text="обычный договор"))
        # NULL-text row indexes nothing; the real row is still findable.
        assert _match_count(conn, "договор") == 1

        # Edit the NULL-text row to carry text (edit-upsert) → now findable.
        with db.writing(conn):
            schema.ingest_message(
                conn,
                _make(
                    message_id=1,
                    text="теперь есть текст",
                    edit_date="2026-07-19T15:00:00+00:00",
                ),
            )
        assert _match_count(conn, "текст") == 1

        # FTS5's own integrity check passes — no orphaned/corrupt entries from the
        # NULL-text insert path (the external-content 'delete' contract held).
        conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
    finally:
        conn.close()


# --- delete trigger keeps the external-content contract honest --------------


def test_hard_delete_removes_fts_row(tmp_path: Path) -> None:
    """The AFTER DELETE trigger is external-content-contract completeness: if a
    messages row is hard-deleted, its FTS entries are removed (no orphan) so a
    later MATCH cannot return a row that no longer exists.

    (v1 never hard-deletes — deletions are tombstones, Story 4.2 — this is
    defensive correctness for the external-content contract only.)
    """
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="удаляемый договор"))
        assert _match_count(conn, "договор") == 1

        with db.writing(conn):
            conn.execute("DELETE FROM messages WHERE chat_id=1 AND message_id=100")
        assert _match_count(conn, "договор") == 0  # no orphaned index entry
    finally:
        conn.close()
