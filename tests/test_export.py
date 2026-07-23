"""Tests for the bulk-export engine (Story 2.4, ``tgkb/writer/export.py``).

No network, **no real sleep**, no real Telegram. Coroutines are driven with
``asyncio.run(...)`` in the test body — matching ``tests/test_auth.py`` /
``tests/test_throttle.py``; ``pytest-asyncio`` is deliberately NOT in the Stack
(AD-7 / minimal deps).

The Telegram client is FAKED (a ``FakeClient`` whose ``takeout(...)`` returns a
fake takeout proxy serving canned pages of fake ``Message``-like objects); the
throttle is a REAL :class:`RequestThrottle` with an injected ``FakeClock`` +
``SleepSpy`` (no wall-clock time passes); the DB is a REAL tmp-file APSW
connection via :func:`schema.open_writer_with_schema`, so ingest + ``sync_state``
are exercised for real. Tests assert the REAL guarantee (rows in the DB, the
``last_id`` value, no-duplicate count, run-continues-after-a-bad-chat, the full N
slept on FLOOD_WAIT) — never a proxy flag (the 1.1→2.3 review lesson).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from telethon.errors import ChannelPrivateError, FloodWaitError

from tgkb.common import schema
from tgkb.common.config import TrackedChat
from tgkb.writer import export
from tgkb.writer.throttle import RequestThrottle


# --- Fakes --------------------------------------------------------------------


class FakeClock:
    """A non-decreasing fake monotonic clock the sleep-spy advances."""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class SleepSpy:
    """Records requested sleep durations WITHOUT sleeping; advances the clock."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.durations: list[float] = []
        self._clock = clock

    async def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)


def _make_throttle(*, rate: float | None = None) -> tuple[RequestThrottle, SleepSpy]:
    """A REAL throttle with an injected clock+sleep-spy (no wall-clock time)."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    return RequestThrottle(rate, monotonic=clock.now, sleep=spy), spy


_BASE_DATE = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


def _fake_sender(sender_id: int) -> Any:
    """A real Telethon ``User`` entity so ``get_display_name`` yields a name.

    ``get_display_name`` uses ``isinstance`` against the real User/Chat/Channel
    types, so a duck-typed stub returns ``''`` — construct an actual ``User``.
    """
    from telethon.tl.types import User

    return User(id=sender_id, first_name=f"Sender {sender_id}")


class FakeMessage:
    """A stand-in for a Telethon ``Message`` (only the attrs export reads)."""

    def __init__(
        self,
        msg_id: int,
        *,
        text: str | None = None,
        sender_id: int | None = 555,
        reply_to: int | None = None,
        edited: bool = False,
        date: datetime | None = None,
    ) -> None:
        self.id = msg_id
        self.message = text
        self.sender_id = sender_id
        # Telethon exposes ``sender`` (a User/Chat entity) — export passes it to
        # ``get_display_name``, which reads ``first_name``/``title``/``username``.
        # A plain str is NOT display-name-able, so use a small entity-shaped fake.
        self.sender = _fake_sender(sender_id) if sender_id is not None else None
        self.reply_to_msg_id = reply_to
        self.date = date if date is not None else (_BASE_DATE + timedelta(minutes=msg_id))
        self.edit_date = (self.date + timedelta(minutes=1)) if edited else None
        self.media = None


class FakeEntity:
    """A stand-in for a resolved Telethon entity (Channel/Chat/User)."""

    def __init__(self, entity_id: int, *, title: str | None = None, username: str | None = None) -> None:
        self.id = entity_id
        self.title = title
        self.username = username
        self.broadcast = True


class FakeTakeout:
    """A fake takeout proxy: serves canned pages via ``get_messages``.

    ``pages_by_entity`` maps ``entity.id`` → a flat list of FakeMessage the proxy
    returns, honoring ``min_id`` (exclusive) + ``limit`` + ``reverse`` exactly as
    Telethon does, so the engine's manual paging is exercised for real.
    """

    def __init__(self, pages_by_entity: dict[int, list[FakeMessage]]) -> None:
        self._by_entity = pages_by_entity
        self.get_messages_calls = 0
        self.entered = False

    async def get_messages(self, entity: Any, *, limit: int, reverse: bool, min_id: int) -> list[FakeMessage]:
        self.get_messages_calls += 1
        assert reverse is True, "export must page oldest-first (reverse=True)"
        msgs = sorted(self._by_entity.get(entity.id, []), key=lambda m: m.id)
        page = [m for m in msgs if m.id > min_id][:limit]
        return page


class FakeTakeoutCtx:
    """Async-context wrapper returned by ``client.takeout(...)``."""

    def __init__(self, proxy: FakeTakeout, *, init_raises: BaseException | None = None) -> None:
        self._proxy = proxy
        self._init_raises = init_raises

    async def __aenter__(self) -> FakeTakeout:
        if self._init_raises is not None:
            raise self._init_raises
        self._proxy.entered = True
        return self._proxy

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class FakeClient:
    """A fake connected client: ``takeout(...)`` + ``get_entity(...)``.

    ``entities`` maps a selector (username str or id int) → FakeEntity (or an
    Exception instance to raise). ``takeout(...)`` is SYNCHRONOUS and returns the
    async-context proxy, exactly like Telethon's ``client.takeout``.
    """

    def __init__(
        self,
        entities: dict[Any, Any],
        pages_by_entity: dict[int, list[FakeMessage]],
        *,
        takeout_init_raises: BaseException | None = None,
    ) -> None:
        self._entities = entities
        self.proxy = FakeTakeout(pages_by_entity)
        self._takeout_init_raises = takeout_init_raises
        self.get_entity_calls: list[Any] = []
        self.takeout_kwargs: dict[str, Any] | None = None
        self.flood_sleep_threshold: int | None = None

    def takeout(self, **kwargs: Any) -> FakeTakeoutCtx:
        self.takeout_kwargs = kwargs
        return FakeTakeoutCtx(self.proxy, init_raises=self._takeout_init_raises)

    async def get_entity(self, selector: Any) -> FakeEntity:
        self.get_entity_calls.append(selector)
        result = self._entities[selector]
        if isinstance(result, BaseException):
            raise result
        return result


def _open_db(tmp_path) -> Any:
    """A real writer connection with the schema ensured (tmp file)."""
    return schema.open_writer_with_schema(tmp_path / "tgkb.sqlite3")


def _fixed_now() -> str:
    return "2026-07-20T00:00:00+00:00"


def _query_all(conn: Any) -> list[tuple]:
    return list(conn.execute("SELECT chat_id, message_id, text, sender_id, sender_name, date_utc, reply_to_msg_id FROM messages ORDER BY message_id"))


# --- AC#2: text messages ingested with the right columns ----------------------


def test_export_ingests_text_messages(tmp_path) -> None:
    """AC#2: rows land with correct (chat_id, message_id), text, sender, UTC-ISO date, reply linkage."""
    entity = FakeEntity(1001, title="Chat", username="pub")
    client = FakeClient(
        {"pub": entity},
        {1001: [FakeMessage(1, text="hello"), FakeMessage(2, text="reply", reply_to=1)]},
    )
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        summary = asyncio.run(
            export.run_export(
                client, (TrackedChat(username="pub"),), throttle, conn,
                progress_cb=lambda _m: None, now_iso=_fixed_now,
            )
        )
        rows = _query_all(conn)
    finally:
        conn.close()

    assert summary.exported == {1001: 2}
    assert rows[0][0] == 1001  # chat_id == entity.id
    assert rows[0][2] == "hello"
    assert rows[0][3] == 555  # sender_id
    assert rows[0][4] == "Sender 555"  # sender_name via get_display_name
    assert rows[0][5].endswith("+00:00")  # UTC-ISO, NOT naive
    assert rows[1][6] == 1  # reply_to_msg_id linkage


def test_export_records_chat_metadata(tmp_path) -> None:
    """AC#2: upsert_chat records title/username/is_public (public ⇔ has a username)."""
    client = FakeClient(
        {"pub": FakeEntity(1001, title="Public", username="pub")},
        {1001: [FakeMessage(1, text="x")]},
    )
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        chat = list(conn.execute("SELECT chat_id, username, is_public FROM chats"))[0]
    finally:
        conn.close()
    assert chat == (1001, "pub", 1)


def test_export_private_chat_is_not_public(tmp_path) -> None:
    """A chat with no username is stored is_public=0 (no t.me link later)."""
    client = FakeClient({-500: FakeEntity(-500, title="Private", username=None)}, {-500: [FakeMessage(1, text="x")]})
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        asyncio.run(export.run_export(client, (TrackedChat(id=-500),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        chat = list(conn.execute("SELECT is_public FROM chats"))[0]
    finally:
        conn.close()
    assert chat == (0,)


# --- AC#1: takeout + throttle (single choke-point) ----------------------------


def test_export_uses_takeout_and_throttle(tmp_path) -> None:
    """AC#1: the takeout proxy is entered, and every Telegram call goes through throttle.run."""
    client = FakeClient(
        {"pub": FakeEntity(1001, username="pub")},
        {1001: [FakeMessage(1, text="a"), FakeMessage(2, text="b")]},
    )
    # A throttle spy: count every run(...) invocation.
    throttle, _ = _make_throttle()
    run_calls = {"n": 0}
    orig_run = throttle.run

    async def _counting_run(factory):
        run_calls["n"] += 1
        return await orig_run(factory)

    throttle.run = _counting_run  # type: ignore[method-assign]

    conn = _open_db(tmp_path)
    try:
        asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
    finally:
        conn.close()

    assert client.proxy.entered is True  # takeout was entered (async with)
    # get_entity(1) + get_messages(page)(1) + get_messages(empty tail? no — short page ends it)
    # 1 entity resolve + 1 page (2 msgs < PAGE_SIZE → stop) = 2 throttled calls.
    assert run_calls["n"] == 2
    assert client.takeout_kwargs == {"channels": True, "chats": True, "megagroups": True, "users": True}


def test_export_no_media_flag_on_takeout(tmp_path) -> None:
    """AC#4: takeout is opened WITHOUT files=True (text-only, no media download)."""
    client = FakeClient({"pub": FakeEntity(1001, username="pub")}, {1001: []})
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
    finally:
        conn.close()
    assert "files" not in client.takeout_kwargs


# --- AC#4: skip media/service without aborting --------------------------------


def test_export_skips_media_only(tmp_path) -> None:
    """AC#4: media-only/service messages (falsy .message) are skipped; export completes."""
    client = FakeClient(
        {"pub": FakeEntity(1001, username="pub")},
        {1001: [
            FakeMessage(1, text="keep"),
            FakeMessage(2, text=None),   # service
            FakeMessage(3, text=""),     # media-only
            FakeMessage(4, text="keep2"),
        ]},
    )
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        summary = asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        rows = _query_all(conn)
    finally:
        conn.close()
    assert summary.exported == {1001: 2}
    assert [r[1] for r in rows] == [1, 4]  # only the text messages ingested


# --- AC#3: idempotent on re-run -----------------------------------------------


def test_export_idempotent_on_rerun(tmp_path) -> None:
    """AC#3: running export twice over the same pages creates no duplicate rows."""
    pages = {1001: [FakeMessage(1, text="a"), FakeMessage(2, text="b"), FakeMessage(3, text="c")]}
    conn = _open_db(tmp_path)
    try:
        for _ in range(2):
            # Fresh client each run (min_id cursor starts at 0 each run — proving
            # the overlapping-range idempotency, not just a no-op second pass).
            client = FakeClient({"pub": FakeEntity(1001, username="pub")}, pages)
            throttle, _ = _make_throttle()
            asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        count = list(conn.execute("SELECT COUNT(*) FROM messages"))[0][0]
    finally:
        conn.close()
    assert count == 3  # no duplicates via (chat_id, message_id)


# --- AC#6: last_id advances to the max ingested id, atomically -----------------


def test_export_advances_last_id(tmp_path) -> None:
    """AC#6: sync_state.last_id == MAX message_id after export."""
    client = FakeClient({"pub": FakeEntity(1001, username="pub")}, {1001: [FakeMessage(5, text="a"), FakeMessage(9, text="b"), FakeMessage(7, text="c")]})
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        last_id = list(conn.execute("SELECT last_id FROM sync_state WHERE chat_id=1001"))[0][0]
    finally:
        conn.close()
    assert last_id == 9


def test_export_last_id_atomic_on_crash(tmp_path) -> None:
    """AC#6/AD-5: a crash mid-export leaves last_id consistent with committed rows — never ahead.

    The first page commits (rows + last_id together); the second page raises
    AFTER the first committed. The DB must show the first page's rows AND its
    last_id — never the second page's, and never rows without the matching mark.
    """
    boom = RuntimeError("simulated crash on page 2")

    class CrashingTakeout(FakeTakeout):
        async def get_messages(self, entity, *, limit, reverse, min_id):  # type: ignore[override]
            page = await super().get_messages(entity, limit=limit, reverse=reverse, min_id=min_id)
            if min_id >= 2:  # after the first page (ids 1,2) committed, blow up.
                raise boom
            return page

    client = FakeClient({"pub": FakeEntity(1001, username="pub")}, {})
    # Two full pages so the loop makes a second get_messages call.
    client.proxy = CrashingTakeout({1001: [FakeMessage(i, text=f"m{i}") for i in range(1, 5)]})
    # Force page size to 2 so the first page is [1,2] and the second call raises.
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        import tgkb.writer.export as exp_mod
        orig = exp_mod.PAGE_SIZE
        exp_mod.PAGE_SIZE = 2
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        finally:
            exp_mod.PAGE_SIZE = orig
        rows = [r[1] for r in _query_all(conn)]
        last_id = list(conn.execute("SELECT last_id FROM sync_state WHERE chat_id=1001"))[0][0]
    finally:
        conn.close()
    # First page committed: rows 1,2 and last_id == 2. Nothing from the crashed page.
    assert rows == [1, 2]
    assert last_id == 2


# --- FR-3: a bad/unreachable chat is skipped, the run continues ----------------


def test_export_bad_chat_skipped_run_continues(tmp_path, capsys) -> None:
    """FR-3: get_entity ValueError on one chat → it is skipped, the OTHER chat still exports."""
    client = FakeClient(
        {"bad": ValueError("No user has 'bad'"), "good": FakeEntity(1001, username="good")},
        {1001: [FakeMessage(1, text="ok")]},
    )
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    msgs: list[str] = []
    try:
        summary = asyncio.run(
            export.run_export(
                client,
                (TrackedChat(username="bad"), TrackedChat(username="good")),
                throttle, conn, progress_cb=msgs.append, now_iso=_fixed_now,
            )
        )
        count = list(conn.execute("SELECT COUNT(*) FROM messages"))[0][0]
    finally:
        conn.close()
    assert summary.skipped_chats == [("@bad", "ValueError")]
    assert summary.exported == {1001: 1}
    assert count == 1  # the good chat exported despite the bad one
    assert any("@bad" in m and "ValueError" in m for m in msgs)


def test_export_flood_on_get_entity_not_skipped(tmp_path) -> None:
    """FLOOD_WAIT surviving the throttle on get_entity is re-raised, NOT misreported as unreachable.

    FloodWaitError subclasses RPCError, so without an explicit re-raise the
    reachability handler would silently skip the chat labeling an account-safety
    flood as a "bad config entry" (AD-10 signal lost). A tiny retry budget makes
    the flood exhaust immediately.
    """
    err = FloodWaitError(request=None, capture=30)
    client = FakeClient({"pub": err}, {})
    # Throttle with a retry budget of 0 → the first flood exhausts and re-raises.
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy, max_flood_retries=0)
    conn = _open_db(tmp_path)
    try:
        with pytest.raises(FloodWaitError):
            asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
    finally:
        conn.close()


def test_export_private_chat_skipped(tmp_path) -> None:
    """FR-3: ChannelPrivateError on resolution is also a skip-and-continue."""
    err = ChannelPrivateError(request=None)
    client = FakeClient(
        {123: err, "good": FakeEntity(1001, username="good")},
        {1001: [FakeMessage(1, text="ok")]},
    )
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        summary = asyncio.run(export.run_export(client, (TrackedChat(id=123), TrackedChat(username="good")), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
    finally:
        conn.close()
    assert summary.skipped_chats == [("id=123", "ChannelPrivateError")]
    assert summary.exported == {1001: 1}


# --- AC#5: FLOOD_WAIT is visible (the throttle is really in the path) ----------


def test_export_flood_wait_visible(tmp_path, caplog) -> None:
    """AC#5: a FLOOD_WAIT on get_messages is slept the full N by the throttle and surfaced; export completes."""

    class FloodOnceTakeout(FakeTakeout):
        def __init__(self, pages_by_entity):
            super().__init__(pages_by_entity)
            self._flooded = False

        async def get_messages(self, entity, *, limit, reverse, min_id):  # type: ignore[override]
            if not self._flooded:
                self._flooded = True
                raise FloodWaitError(request=None, capture=42)
            return await super().get_messages(entity, limit=limit, reverse=reverse, min_id=min_id)

    client = FakeClient({"pub": FakeEntity(1001, username="pub")}, {})
    client.proxy = FloodOnceTakeout({1001: [FakeMessage(1, text="a")]})
    throttle, spy = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        with caplog.at_level(logging.INFO, logger="tgkb.writer.throttle"):
            summary = asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        count = list(conn.execute("SELECT COUNT(*) FROM messages"))[0][0]
    finally:
        conn.close()
    assert 42 in spy.durations  # slept the full N (throttle really in the path)
    assert "42" in "\n".join(r.getMessage() for r in caplog.records)  # surfaced
    assert count == 1  # export recovered and completed
    assert summary.exported == {1001: 1}


# --- Timestamp + edit mapping -------------------------------------------------


def test_export_edit_date_mapped_utc(tmp_path) -> None:
    """An edited message stores a UTC-ISO edit_date; an unedited one stores NULL."""
    client = FakeClient({"pub": FakeEntity(1001, username="pub")}, {1001: [FakeMessage(1, text="a"), FakeMessage(2, text="b", edited=True)]})
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        rows = list(conn.execute("SELECT message_id, edit_date FROM messages ORDER BY message_id"))
    finally:
        conn.close()
    assert rows[0] == (1, None)
    assert rows[1][0] == 2 and rows[1][1].endswith("+00:00")


# --- Multi-page paging --------------------------------------------------------


def test_export_pages_through_full_history(tmp_path) -> None:
    """Manual paging walks the whole history across multiple pages (min_id cursor)."""
    msgs = [FakeMessage(i, text=f"m{i}") for i in range(1, 6)]  # 5 messages
    client = FakeClient({"pub": FakeEntity(1001, username="pub")}, {1001: msgs})
    throttle, _ = _make_throttle()
    conn = _open_db(tmp_path)
    try:
        import tgkb.writer.export as exp_mod
        orig = exp_mod.PAGE_SIZE
        exp_mod.PAGE_SIZE = 2  # force 3 pages: [1,2],[3,4],[5]
        try:
            summary = asyncio.run(export.run_export(client, (TrackedChat(username="pub"),), throttle, conn, progress_cb=lambda _m: None, now_iso=_fixed_now))
        finally:
            exp_mod.PAGE_SIZE = orig
        ids = [r[1] for r in _query_all(conn)]
    finally:
        conn.close()
    assert ids == [1, 2, 3, 4, 5]
    assert summary.exported == {1001: 5}
