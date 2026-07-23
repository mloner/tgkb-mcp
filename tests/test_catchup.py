"""Tests for the catch-up backfill engine (Story 4.1, ``export.run_catchup``).

Catch-up is the TWIN of bulk export (Story 2.4): same throttle, same idempotent
per-page ingest, same atomic ``advance_last_id``. The single NEW mechanic is
seeding the page cursor from the persisted ``sync_state.last_id`` (``min_id =
last_id``) and paging via the **bare client** (a regular session — NOT a takeout).
These tests assert the REAL guarantee (first-page ``min_id``, rows in the DB + a
real MATCH, no-duplicate count on overlap, monotonic ``last_id``, takeout NEVER
entered, full-N slept on FLOOD_WAIT) — never a proxy flag (the 1.1→3.3 lesson).

No network, no real sleep, no ``pytest-asyncio`` (AD-7): coroutines run via
``asyncio.run(...)`` in the test body. The DB is a REAL tmp-file APSW connection
via :func:`schema.open_writer_with_schema`, so ``sync_state`` / ingest / the FTS
trigger path are exercised for real. The clock+sleep fakes and the message fakes
are reused from :mod:`tests.test_export`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from telethon.errors import ChannelPrivateError, FloodWaitError

from tgkb.common import db, schema
from tgkb.common.config import TrackedChat
from tgkb.writer import export
from tgkb.writer.throttle import RequestThrottle

# Reuse the export test fakes verbatim (the story's instruction: import them, do
# not re-type). FakeMessage/FakeEntity/FakeClock/SleepSpy/_make_throttle are all
# source-neutral; only the paging SOURCE differs (bare client, not a takeout).
from tests.test_export import (
    FakeClock,
    FakeEntity,
    FakeMessage,
    SleepSpy,
    _fixed_now,
    _make_throttle,
)


# --- A catch-up client: pages via the BARE client's get_messages (no takeout) --


class CatchupFakeClient:
    """A fake connected client for catch-up: ``get_entity`` + ``get_messages``.

    Unlike the export ``FakeClient`` (whose history comes from a ``takeout(...)``
    proxy), catch-up pages the BARE client, so ``get_messages`` lives here. It
    records every ``min_id`` it was asked for (to prove cursor-seeding) and honors
    ``min_id`` (exclusive) + ``limit`` + ``reverse`` exactly like Telethon.

    ``takeout(...)`` is present but flags entry so a test can assert catch-up NEVER
    opens a takeout (the bulk-vs-live distinction).
    """

    def __init__(
        self,
        entities: dict[Any, Any],
        pages_by_entity: dict[int, list[FakeMessage]],
    ) -> None:
        self._entities = entities
        self._by_entity = pages_by_entity
        self.get_entity_calls: list[Any] = []
        self.get_messages_min_ids: list[int] = []
        self.get_messages_calls = 0
        self.takeout_entered = False
        self.flood_sleep_threshold: int | None = None

    async def get_entity(self, selector: Any) -> FakeEntity:
        self.get_entity_calls.append(selector)
        result = self._entities[selector]
        if isinstance(result, BaseException):
            raise result
        return result

    async def get_messages(
        self, entity: Any, *, limit: int, reverse: bool, min_id: int
    ) -> list[FakeMessage]:
        self.get_messages_calls += 1
        self.get_messages_min_ids.append(min_id)
        assert reverse is True, "catch-up must page oldest-first (reverse=True)"
        msgs = sorted(self._by_entity.get(entity.id, []), key=lambda m: m.id)
        return [m for m in msgs if m.id > min_id][:limit]

    def takeout(self, **_kwargs: Any) -> Any:  # pragma: no cover - must NOT be called
        self.takeout_entered = True
        raise AssertionError("catch-up must NOT open a takeout session")


def _open_db(tmp_path) -> Any:
    return schema.open_writer_with_schema(tmp_path / "tgkb.sqlite3")


def _seed_last_id(conn: Any, chat_id: int, last_id: int) -> None:
    """Advance a chat's high-water mark to simulate a prior run (in a write txn)."""
    with db.writing(conn):
        schema.advance_last_id(conn, chat_id, last_id, updated_at=_fixed_now())


def _run_catchup(client, chats, throttle, conn, **kw):
    return asyncio.run(
        export.run_catchup(
            client, chats, throttle, conn,
            progress_cb=kw.pop("progress_cb", lambda _m: None),
            now_iso=kw.pop("now_iso", _fixed_now),
            **kw,
        )
    )


# --- AC#1: the cursor is seeded from last_id (the one behavior that is 4.1) ----


def test_catchup_fetches_from_last_id(tmp_path) -> None:
    """AC#1: the FIRST page request uses min_id == last_id (NOT 0)."""
    conn = _open_db(tmp_path)
    try:
        _seed_last_id(conn, 1001, 100)
        client = CatchupFakeClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [FakeMessage(i, text=f"m{i}") for i in (101, 102, 103)]},
        )
        throttle, _ = _make_throttle()
        _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
    finally:
        conn.close()
    # The distinguishing behavior vs 2.4: the cursor started at the stored mark.
    assert client.get_messages_min_ids[0] == 100


# --- AC#2: missed messages are present AND searchable --------------------------


def test_catchup_ingests_missed_messages_and_they_are_searchable(tmp_path) -> None:
    """AC#2: messages that arrived offline land in the DB and match via FTS (Cyrillic)."""
    conn = _open_db(tmp_path)
    try:
        _seed_last_id(conn, 1001, 100)
        client = CatchupFakeClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [
                FakeMessage(101, text="Подписали договор аренды"),
                FakeMessage(102, text="Встреча перенесена"),
                FakeMessage(103, text="Ещё один договор"),
            ]},
        )
        throttle, _ = _make_throttle()
        summary = _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
        ids = [r[0] for r in conn.execute("SELECT message_id FROM messages ORDER BY message_id")]
        # A real MATCH on a lemma from the text exercises the 3.2 FTS trigger path.
        rowids = schema.search_message_rowids(conn, "договор")
    finally:
        conn.close()
    assert ids == [101, 102, 103]
    assert summary.exported == {1001: 3}
    assert len(rowids) == 2  # both "договор" messages are findable


# --- AC#1 / Consistency: catch-up uses NO takeout ------------------------------


def test_catchup_does_not_use_takeout(tmp_path) -> None:
    """Consistency: catch-up pages the BARE client and never opens a takeout."""
    conn = _open_db(tmp_path)
    try:
        _seed_last_id(conn, 1001, 0)
        client = CatchupFakeClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [FakeMessage(1, text="a"), FakeMessage(2, text="b")]},
        )
        throttle, _ = _make_throttle()
        _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
    finally:
        conn.close()
    assert client.takeout_entered is False  # takeout NEVER entered
    assert client.get_messages_calls >= 1  # history paged via the bare client


# --- AC#3: idempotent on an overlapping range ----------------------------------


def test_catchup_idempotent_on_overlap(tmp_path) -> None:
    """AC#3: a range that re-surfaces already-ingested ids creates no duplicates."""
    conn = _open_db(tmp_path)
    try:
        # Pre-ingest ids 98,99,100 and mark last_id=99, so a min_id=99 page
        # re-surfaces id 100 (an overlap the ON CONFLICT clause must absorb).
        with db.writing(conn):
            schema.ingest_messages(conn, [
                schema.Message(1001, mid, None, None, f"m{mid}", "2026-07-20T00:00:00+00:00", None, None)
                for mid in (98, 99, 100)
            ])
            schema.advance_last_id(conn, 1001, 99, updated_at=_fixed_now())
        client = CatchupFakeClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [FakeMessage(100, text="m100"), FakeMessage(101, text="m101")]},
        )
        throttle, _ = _make_throttle()
        _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
        count = list(conn.execute("SELECT COUNT(*) FROM messages"))[0][0]
        ids = [r[0] for r in conn.execute("SELECT message_id FROM messages ORDER BY message_id")]
    finally:
        conn.close()
    assert count == 4  # 98,99,100,101 — id 100 not duplicated
    assert ids == [98, 99, 100, 101]


# --- AC#4: last_id advances atomically and never regresses ---------------------


def test_catchup_advances_last_id_atomically_and_monotonically(tmp_path) -> None:
    """AC#4: last_id → the max ingested id; an edited OLD message never regresses it."""
    conn = _open_db(tmp_path)
    try:
        _seed_last_id(conn, 1001, 100)
        # Serve 101..105 first; assert last_id becomes 105.
        client = CatchupFakeClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [FakeMessage(i, text=f"m{i}") for i in range(101, 106)]},
        )
        throttle, _ = _make_throttle()
        _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
        assert schema.read_last_id(conn, 1001) == 105

        # Now an EDIT of an OLD message (id 50, edited) re-appears in a page. A
        # min_id=105 page would exclude it, so serve it via a direct advance to
        # simulate the mark-guard: advancing with a lower id must NOT regress 105.
        with db.writing(conn):
            schema.advance_last_id(conn, 1001, 50, updated_at="2026-07-21T00:00:00+00:00")
        assert schema.read_last_id(conn, 1001) == 105  # MAX guard held
    finally:
        conn.close()


def test_catchup_last_id_atomic_on_crash(tmp_path) -> None:
    """AC#4/AD-5: a crash mid-catch-up leaves last_id consistent with committed rows."""
    conn = _open_db(tmp_path)
    boom = RuntimeError("simulated crash on page 2")

    class CrashingClient(CatchupFakeClient):
        async def get_messages(self, entity, *, limit, reverse, min_id):  # type: ignore[override]
            page = await super().get_messages(entity, limit=limit, reverse=reverse, min_id=min_id)
            if min_id >= 2:  # after the first page (ids 1,2) committed, blow up.
                raise boom
            return page

    try:
        _seed_last_id(conn, 1001, 0)
        client = CrashingClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [FakeMessage(i, text=f"m{i}") for i in range(1, 5)]},
        )
        throttle, _ = _make_throttle()
        import tgkb.writer.export as exp_mod
        orig = exp_mod.PAGE_SIZE
        exp_mod.PAGE_SIZE = 2
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
        finally:
            exp_mod.PAGE_SIZE = orig
        ids = [r[0] for r in conn.execute("SELECT message_id FROM messages ORDER BY message_id")]
        last_id = list(conn.execute("SELECT last_id FROM sync_state WHERE chat_id=1001"))[0][0]
    finally:
        conn.close()
    assert ids == [1, 2]  # only the first page committed
    assert last_id == 2   # mark consistent with committed rows, never ahead


# --- FR-3: a bad chat is skipped, the run continues; flood re-raised -----------


def test_catchup_bad_chat_skipped_run_continues(tmp_path) -> None:
    """FR-3: a chat that fails resolution is skipped; the OTHER chat still backfills."""
    conn = _open_db(tmp_path)
    msgs: list[str] = []
    try:
        _seed_last_id(conn, 1001, 0)
        client = CatchupFakeClient(
            {"bad": ChannelPrivateError(request=None), "good": FakeEntity(1001, username="good")},
            {1001: [FakeMessage(1, text="ok")]},
        )
        throttle, _ = _make_throttle()
        summary = _run_catchup(
            client, (TrackedChat(username="bad"), TrackedChat(username="good")),
            throttle, conn, progress_cb=msgs.append,
        )
        count = list(conn.execute("SELECT COUNT(*) FROM messages"))[0][0]
    finally:
        conn.close()
    assert summary.skipped_chats == [("@bad", "ChannelPrivateError")]
    assert summary.exported == {1001: 1}
    assert count == 1
    assert any("@bad" in m and "ChannelPrivateError" in m for m in msgs)


def test_catchup_flood_on_get_entity_reraised(tmp_path) -> None:
    """FR-3/AD-10: a FloodWaitError surviving the throttle on get_entity is RE-RAISED."""
    conn = _open_db(tmp_path)
    try:
        _seed_last_id(conn, 1001, 0)
        err = FloodWaitError(request=None, capture=30)
        client = CatchupFakeClient({"pub": err}, {})
        clock = FakeClock()
        spy = SleepSpy(clock)
        throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy, max_flood_retries=0)
        with pytest.raises(FloodWaitError):
            _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
    finally:
        conn.close()


# --- Task 1: read_last_id defaults + read-back ---------------------------------


def test_read_last_id_defaults_zero_for_unknown_chat(tmp_path) -> None:
    """Task 1: read_last_id → 0 on a never-synced chat, and the stored value after advance."""
    conn = _open_db(tmp_path)
    try:
        assert schema.read_last_id(conn, 999) == 0  # never synced → fetch-all
        with db.writing(conn):
            schema.advance_last_id(conn, 999, 42, updated_at=_fixed_now())
        assert schema.read_last_id(conn, 999) == 42
    finally:
        conn.close()


# --- AC#5-spirit: the throttle is really in the catch-up path ------------------


def test_catchup_flood_wait_visible(tmp_path, caplog) -> None:
    """A FLOOD_WAIT on get_messages is slept the full N by the throttle and surfaced."""
    conn = _open_db(tmp_path)

    class FloodOnceClient(CatchupFakeClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._flooded = False

        async def get_messages(self, entity, *, limit, reverse, min_id):  # type: ignore[override]
            if not self._flooded:
                self._flooded = True
                raise FloodWaitError(request=None, capture=42)
            return await super().get_messages(entity, limit=limit, reverse=reverse, min_id=min_id)

    try:
        _seed_last_id(conn, 1001, 0)
        client = FloodOnceClient(
            {"pub": FakeEntity(1001, username="pub")},
            {1001: [FakeMessage(1, text="a")]},
        )
        throttle, spy = _make_throttle()
        with caplog.at_level(logging.INFO, logger="tgkb.writer.throttle"):
            summary = _run_catchup(client, (TrackedChat(username="pub"),), throttle, conn)
        count = list(conn.execute("SELECT COUNT(*) FROM messages"))[0][0]
    finally:
        conn.close()
    assert 42 in spy.durations  # slept the full N (throttle really in the path)
    assert "42" in "\n".join(r.getMessage() for r in caplog.records)
    assert count == 1
    assert summary.exported == {1001: 1}
