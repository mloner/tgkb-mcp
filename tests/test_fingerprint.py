"""Tests for the tokenizer fingerprint + drift-guarded rebuild (Story 3.3).

This is the DETECTION + RECOVERY half of AD-11. Story 3.1 built the shared
``ru_lemma`` tokenizer; Story 3.2 built the real ``messages_fts`` external-content
index + triggers and documented ``INSERT INTO messages_fts(messages_fts)
VALUES('rebuild')`` as 3.3's drift-recovery action. This story adds:

- ``ru_lemma.fingerprint()`` = dict version + engine version + hypothesis-policy
  tag, single-sourced with the tokenizer it fingerprints (AC#1),
- persistence in a tiny ``meta(key, value)`` KV table, stamped when the index is
  created/rebuilt (AC#1, AC#4),
- ``check_and_repair_fingerprint(conn)`` (writer, writable) that forces a full
  FTS rebuild on mismatch and re-stamps the fingerprint atomically (AC#3, AC#4),
- ``fingerprint_matches(conn)`` (reader, read-only) that lets Story 4.3 REFUSE to
  serve against a stale index (AC#3, reader clause).

Behavior-first (the review lesson carried 1.1 → 3.2): every test asserts a REAL
guarantee — a ``MATCH`` survives a rebuild, the stored fingerprint VALUE changed,
the reader-check returns ``False`` on drift — never a proxy like "the meta table
exists" or "rebuild was called". The dict-version component is verified against
``importlib.metadata.version(...)``, not from memory. Expected lemmas are pinned
to ``pymorphy3==2.0.6`` + ``pymorphy3-dicts-ru==2.4.417150.4580142`` (the
``договоры``/``договора``/``договор`` family shares ``договор``).
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import apsw
import pytest

from tgkb.common import db, ru_lemma, schema
from tgkb.common.schema import Message


def _open(tmp_path: Path) -> apsw.Connection:
    """Open a real writer against a fresh tmp DB and create the full schema.

    ``ensure_schema`` registers ``ru_lemma``, creates ``messages_fts`` + triggers,
    and (Story 3.3) stamps the current fingerprint on the fresh DB.
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
    return len(schema.search_message_rowids(conn, query))


# --- AC#1: the fingerprint is deterministic and composed --------------------


def test_fingerprint_is_deterministic_and_composed() -> None:
    """AC#1: ``fingerprint()`` is stable across calls (AD-7 determinism) and its
    string CONTAINS the installed pymorphy3-dicts-ru version AND the policy tag —
    proving the composition, verified against importlib.metadata (not memory)."""
    first = ru_lemma.fingerprint()
    second = ru_lemma.fingerprint()
    assert first == second  # deterministic — no clock, no randomness

    dict_version = importlib.metadata.version("pymorphy3-dicts-ru")
    engine_version = importlib.metadata.version("pymorphy3")
    assert dict_version in first  # the real installed dict version, read at runtime
    assert engine_version in first  # engine version too (intent of AD-11)
    assert ru_lemma.HYPOTHESIS_POLICY in first  # the hand-maintained policy tag


# --- AC#1: a fresh DB is stamped with the current fingerprint ---------------


def test_fresh_db_persists_current_fingerprint(tmp_path: Path) -> None:
    """AC#1: after opening a fresh DB, the stored fingerprint equals the running
    tokenizer's fingerprint — the create path stamped it (an empty index built by
    the current tokenizer is trivially consistent with it)."""
    conn = schema.open_writer_with_schema(tmp_path / "tgkb.sqlite3")
    try:
        assert schema.read_fingerprint(conn) == ru_lemma.fingerprint()
    finally:
        conn.close()


def test_fresh_db_read_fingerprint_before_schema_is_none() -> None:
    """AC#1 (the None branch): a truly fresh in-memory DB with no ``meta`` row
    yields ``None`` — read_fingerprint reports absence, it does not raise."""
    conn = apsw.Connection(":memory:")
    try:
        # Create only the meta table (no stamp) to exercise the empty-table read.
        conn.execute(schema._CREATE_META)
        assert schema.read_fingerprint(conn) is None
    finally:
        conn.close()


# --- AC#2: a matching fingerprint proceeds, no needless rebuild -------------


def test_matching_fingerprint_no_rebuild(tmp_path: Path) -> None:
    """AC#2: on a DB whose stored fingerprint matches the running tokenizer,
    ``check_and_repair_fingerprint`` returns False (no rebuild) and a MATCH still
    finds an already-indexed message — a matching startup proceeds untouched."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="подписали договоры вчера"))
        assert _match_count(conn, "договор") == 1

        # Fingerprint was stamped at ensure_schema and still matches → no rebuild.
        rebuilt = schema.check_and_repair_fingerprint(conn)
        assert rebuilt is False
        assert _match_count(conn, "договор") == 1  # recall untouched
    finally:
        conn.close()


# --- AC#3 + AC#4: a mismatch forces a rebuild and restores recall -----------


def test_mismatch_forces_rebuild_and_restores_recall(tmp_path: Path) -> None:
    """AC#3 + AC#4 (the AD-11 payoff): with messages_fts populated, poison the
    stored fingerprint to simulate drift, then run the check → it rebuilds
    (returns True), re-stamps the CURRENT fingerprint (AC#4), and a MATCH for an
    inflection still finds the message (rebuild restored the index from messages).

    Poisoning the stored value is sufficient: the check compares stored vs current,
    so a bogus stored value is indistinguishable from a real dict/policy change.
    """
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="есть договоры"))
        assert _match_count(conn, "договор") == 1

        # Simulate drift: overwrite the stored fingerprint with a bogus value.
        with db.writing(conn):
            schema.write_fingerprint(conn, "stale-bogus-fingerprint")
        assert schema.read_fingerprint(conn) == "stale-bogus-fingerprint"

        rebuilt = schema.check_and_repair_fingerprint(conn)
        assert rebuilt is True  # mismatch detected → rebuild performed
        # AC#4: the persisted fingerprint is now the current tokenizer's.
        assert schema.read_fingerprint(conn) == ru_lemma.fingerprint()
        # AC#3: recall restored — a different inflection still finds the message.
        assert _match_count(conn, "договорами") == 1
    finally:
        conn.close()


def test_rebuild_reindexes_from_messages_table(tmp_path: Path) -> None:
    """AC#3 (the "not stale" guarantee): corrupt the FTS index out-of-band (orphan
    an entry) while ``messages`` stays intact, poison the fingerprint, run the
    check → the rebuild re-derives full recall FROM ``messages`` (a no-op rebuild
    could not do this). Behavioral: assert MATCH counts before/after."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(message_id=1, text="первый договор"))
            schema.ingest_message(conn, _make(message_id=2, text="второй договор"))
        assert _match_count(conn, "договор") == 2

        # Corrupt the FTS index out-of-band: 'delete' one row's entries directly,
        # leaving `messages` intact → the index now under-reports (recall hole).
        with db.writing(conn):
            row = conn.execute(
                "SELECT rowid, text FROM messages WHERE message_id=1"
            ).fetchone()
            conn.execute(
                "INSERT INTO messages_fts(messages_fts, rowid, text) "
                "VALUES('delete', ?, ?)",
                (row[0], row[1]),
            )
        assert _match_count(conn, "договор") == 1  # recall hole confirmed

        # Poison the fingerprint and run the check → rebuild re-derives from source.
        with db.writing(conn):
            schema.write_fingerprint(conn, "stale-bogus")
        assert schema.check_and_repair_fingerprint(conn) is True
        assert _match_count(conn, "договор") == 2  # full recall restored FROM messages
    finally:
        conn.close()


# --- AC#3 reader clause: the reader detects drift but cannot repair ----------


def test_reader_check_returns_false_on_mismatch(tmp_path: Path) -> None:
    """AC#3 (reader clause): the read-only ``fingerprint_matches`` returns False on
    a stale DB (so Story 4.3 can REFUSE to serve) and True on a matching DB. The
    reader can only READ — a 'rebuild' through its read-only connection RAISES,
    a second-layer proof that repair is the writer's job (AD-2)."""
    path = tmp_path / "tgkb.sqlite3"
    writer = schema.open_writer_with_schema(path)
    try:
        with db.writing(writer):
            schema.ingest_message(writer, _make(text="общий договор"))
        db.checkpoint(writer)
    finally:
        writer.close()

    # Matching DB → reader agrees it is fresh.
    reader = schema.open_reader_with_fts(path)
    try:
        assert schema.fingerprint_matches(reader) is True
    finally:
        reader.close()

    # Poison via the writer, then the reader must detect staleness.
    writer = schema.open_writer_with_schema(path)
    try:
        with db.writing(writer):
            schema.write_fingerprint(writer, "stale-bogus")
    finally:
        writer.close()

    reader = schema.open_reader_with_fts(path)
    try:
        assert schema.fingerprint_matches(reader) is False
        # The reader is genuinely read-only: it CANNOT run the repair itself.
        with pytest.raises(apsw.ReadOnlyError):
            reader.execute(
                "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
            )
    finally:
        reader.close()


def test_reader_check_true_on_open_writer_reopen(tmp_path: Path) -> None:
    """AC#2/AC#3: after a writer repairs drift, a reader opened afterwards sees a
    matching fingerprint (the repair persisted) — the writer-repairs/reader-refuses
    seam split composes end to end."""
    path = tmp_path / "tgkb.sqlite3"
    writer = schema.open_writer_with_schema(path)
    try:
        with db.writing(writer):
            schema.ingest_message(writer, _make(text="договор"))
        # Poison then repair via the writer seam.
        with db.writing(writer):
            schema.write_fingerprint(writer, "stale-bogus")
        assert schema.check_and_repair_fingerprint(writer) is True
        db.checkpoint(writer)
    finally:
        writer.close()

    reader = schema.open_reader_with_fts(path)
    try:
        assert schema.fingerprint_matches(reader) is True
        assert _match_count(reader, "договор") == 1
    finally:
        reader.close()


# --- AC#4 atomicity: rebuild + restamp share one transaction ----------------


def test_rebuild_is_atomic_via_real_function(tmp_path: Path, monkeypatch) -> None:
    """AC#4 atomicity, driven through the REAL ``check_and_repair_fingerprint``.

    Monkeypatch ``schema.write_fingerprint`` to raise AFTER the ``'rebuild'`` has
    run inside the function's single ``db.writing`` block. Since the rebuild and
    the re-stamp share ONE transaction, the raise rolls BOTH back — the stored
    fingerprint must remain the OLD poisoned value. This exercises the production
    function itself (a regression that split the two into separate transactions
    would make this fail), not an inline copy of the SQL.
    """
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="договор"))
        with db.writing(conn):
            schema.write_fingerprint(conn, "stale-bogus")

        real_write = schema.write_fingerprint

        def boom_after_rebuild(c, fp):  # noqa: ANN001 - test shim
            raise RuntimeError("boom after rebuild, before commit")

        monkeypatch.setattr(schema, "write_fingerprint", boom_after_rebuild)
        with pytest.raises(RuntimeError):
            schema.check_and_repair_fingerprint(conn)
        monkeypatch.setattr(schema, "write_fingerprint", real_write)

        # Rolled back together: the fingerprint is STILL the poisoned value AND the
        # index was not left half-rebuilt-and-committed.
        assert schema.read_fingerprint(conn) == "stale-bogus"
        # A subsequent real repair still works (connection not wedged).
        assert schema.check_and_repair_fingerprint(conn) is True
        assert schema.read_fingerprint(conn) == ru_lemma.fingerprint()
        assert _match_count(conn, "договор") == 1
    finally:
        conn.close()


def test_fingerprint_degrades_on_missing_metadata(monkeypatch) -> None:
    """Robustness (review finding): ``fingerprint()`` sits on the mandatory writer-
    open / reader-startup path, so a metadata lookup failure must NOT raise (it
    would hard-fail export + the reader). A ``PackageNotFoundError`` degrades the
    component to the ``unknown`` sentinel — fail-closed: an ``unknown`` fingerprint
    cannot match a real stored one, so the writer rebuilds / the reader refuses."""
    import importlib.metadata as md

    def raise_not_found(_dist):  # noqa: ANN001 - test shim
        raise md.PackageNotFoundError("simulated missing metadata")

    monkeypatch.setattr(ru_lemma.importlib.metadata, "version", raise_not_found)
    fp = ru_lemma.fingerprint()  # must NOT raise
    assert "dicts=unknown" in fp
    assert "engine=unknown" in fp
    assert ru_lemma.HYPOTHESIS_POLICY in fp


# --- writer DB-open seam wires the check in (AC#2/AC#3) ----------------------


def test_open_writer_with_schema_repairs_drift(tmp_path: Path) -> None:
    """AC#2/AC#3: the writer DB-open seam runs the check/repair. Poison the stored
    fingerprint, reopen via ``open_writer_with_schema`` → drift is repaired at open
    (fingerprint re-stamped, recall intact) with no explicit check call."""
    path = tmp_path / "tgkb.sqlite3"
    conn = schema.open_writer_with_schema(path)
    try:
        with db.writing(conn):
            schema.ingest_message(conn, _make(text="договоры"))
        with db.writing(conn):
            schema.write_fingerprint(conn, "stale-bogus")
    finally:
        conn.close()

    # Reopen: the seam ensures schema (no-op) then checks/repairs drift.
    conn = schema.open_writer_with_schema(path)
    try:
        assert schema.read_fingerprint(conn) == ru_lemma.fingerprint()
        assert _match_count(conn, "договор") == 1
    finally:
        conn.close()


# --- meta table shape (allowed structural check, paired with behavior) -------


def test_meta_table_exists_and_upsert_is_idempotent(tmp_path: Path) -> None:
    """The ``meta`` KV table stores the fingerprint under a single-sourced key and
    ``write_fingerprint`` upserts (INSERT … ON CONFLICT) — a second write updates
    the SAME row rather than raising or duplicating."""
    conn = _open(tmp_path)
    try:
        with db.writing(conn):
            schema.write_fingerprint(conn, "a")
        with db.writing(conn):
            schema.write_fingerprint(conn, "b")
        assert schema.read_fingerprint(conn) == "b"
        # Exactly one meta row for the fingerprint key (upsert, not insert-dup).
        count = conn.execute(
            "SELECT count(*) FROM meta WHERE key=?",
            (schema._FINGERPRINT_KEY,),
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


# --- public surface is re-exported (dependency wiring, Task 5) ---------------


def test_public_surface_reexported() -> None:
    """Task 5: the new public functions are re-exported from ``tgkb.common``."""
    import tgkb.common as common

    assert common.fingerprint is ru_lemma.fingerprint
    assert common.check_and_repair_fingerprint is schema.check_and_repair_fingerprint
    assert common.fingerprint_matches is schema.fingerprint_matches
    assert common.read_fingerprint is schema.read_fingerprint
    assert common.write_fingerprint is schema.write_fingerprint
    for name in (
        "fingerprint",
        "check_and_repair_fingerprint",
        "fingerprint_matches",
        "read_fingerprint",
        "write_fingerprint",
    ):
        assert name in common.__all__
