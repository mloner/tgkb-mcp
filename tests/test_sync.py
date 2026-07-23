"""Tests for the realtime sync listener + reconnect loop (Story 4.2, ``sync.run_sync``).

Story 4.2 turns ``sync`` into a long-running listener: after catch-up it attaches
``events.NewMessage`` / ``events.MessageEdited`` / ``events.MessageDeleted`` handlers
(filtered to the Tracked Chats) and ``run_until_disconnected()``, re-running
catch-up on a drop (AC#3). Almost everything is REUSE — the genuinely new mechanics
are the three handlers, the reconnect loop, ``schema.mark_deleted``, and the WAL
checkpoint cadence.

These tests assert the REAL guarantee, never a proxy flag (the 1.1→4.1 lesson):
a new-message row is present + searchable via a real Cyrillic MATCH + ``last_id``
advanced; an edit updates text + FTS with no ``last_id`` regress; a deletion sets
``deleted=1`` (row still present, NOT hard-deleted); reconnect re-enters catch-up;
overlap after reconnect creates no duplicates; an untracked-chat event is not
ingested; the checkpoint fires on cadence BETWEEN transactions.

No network, no real sleep, no ``pytest-asyncio`` (AD-7): coroutines run via
``asyncio.run(...)``. No real ``run_until_disconnected`` — a fake client records the
registered handlers, invokes them with canned ``FakeEvent``s, then returns
(simulating a drop). The DB is a REAL tmp-file APSW connection via
:func:`schema.open_writer_with_schema`, so ingest / FTS triggers / ``sync_state`` /
``deleted`` are exercised for real. The message/clock fakes are reused from
:mod:`tests.test_export`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from telethon import events
from telethon.tl import types
from telethon.utils import get_peer_id

from tgkb.common import db, schema
from tgkb.common.config import TrackedChat
from tgkb.writer import sync
from tgkb.writer.throttle import RequestThrottle

from tests.test_export import (
    FakeClock,
    FakeEntity,
    FakeMessage,
    SleepSpy,
    _fixed_now,
    _make_throttle,
)


# --- Real channel entities so get_peer_id behaves EXACTLY as in production ------
#
# The listener bridges two id spaces: it keys `tracked_ids` by the MARKED peer id
# `get_peer_id(entity)` (what `event.chat_id` carries — the `-100…` form for
# channels) and ingests under the RAW `entity.id`. `get_peer_id` only accepts real
# Telethon types, so the tests use an actual `Channel` (a duck-typed fake raises
# "cannot cast to int"). This makes the id-space bridge — the exact thing the
# channel live-ingest bug was about — exercised for real, not against a shim.


def _channel_entity(raw_id: int, *, username: str | None = "pub") -> types.Channel:
    """A real Telethon ``Channel`` with raw id ``raw_id`` (so ``get_peer_id`` works)."""
    return types.Channel(
        id=raw_id, title="chan", photo=None, date=None, creator=True, left=False,
        broadcast=True, verified=False, megagroup=False, restricted=False,
        signatures=False, min=False, scam=False, has_link=False, has_geo=False,
        slowmode_enabled=False, access_hash=0, username=username,
        restriction_reason=[], admin_rights=None, banned_rights=None,
        default_banned_rights=None, participants_count=None,
    )


def _marked(raw_id: int) -> int:
    """The marked peer id a channel's ``event.chat_id`` carries (the ``-100…`` form)."""
    return get_peer_id(_channel_entity(raw_id))


# --- Canned events ------------------------------------------------------------


class FakeEvent:
    """A stand-in for a Telethon NewMessage / MessageEdited event.

    Carries the ``chat_id`` the handler checks against the tracked set and the
    ``.message`` (a :class:`FakeMessage`) the handler maps + ingests.
    """

    def __init__(self, chat_id: int, message: FakeMessage) -> None:
        self.chat_id = chat_id
        self.message = message


class FakeDeleteEvent:
    """A stand-in for a Telethon MessageDeleted event (``chat_id`` + ``deleted_ids``).

    ``chat_id=None`` models the real Telethon edge where a deletion cannot be
    attributed to a peer (ordinary users/small groups) — the handler must log-and-skip.
    """

    def __init__(self, chat_id: int | None, deleted_ids: list[int]) -> None:
        self.chat_id = chat_id
        self.deleted_ids = deleted_ids


# --- A listener fake: records handlers + drives them on run_until_disconnected --


class ListenerFakeClient:
    """A fake connected client for the listener: catch-up paging + event delivery.

    ``get_entity`` / ``get_messages`` drive the catch-up warm-up (reused from the
    catch-up engine). ``add_event_handler`` records each ``(callback, event_filter)``
    so a test can inspect the registered handlers and their ``chats=`` filter;
    ``remove_event_handler`` drops them (proving the reconnect loop detaches).

    ``run_until_disconnected`` simulates a live session: on each call it pops the
    next batch of canned events from ``event_batches`` and dispatches each to the
    matching registered handler (by the event filter's type), then returns —
    modelling a connection drop. When ``event_batches`` is exhausted it returns
    immediately with no events (a bare drop).
    """

    def __init__(
        self,
        entities: dict[Any, Any],
        pages_by_entity: dict[int, list[FakeMessage]] | None = None,
        event_batches: list[list[Any]] | None = None,
    ) -> None:
        self._entities = entities
        self._by_entity = pages_by_entity or {}
        self._event_batches = event_batches or []
        self.handlers: list[tuple[Any, Any]] = []
        self.get_entity_calls: list[Any] = []
        self.get_messages_calls = 0
        self.run_until_disconnected_calls = 0
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
        msgs = sorted(self._by_entity.get(entity.id, []), key=lambda m: m.id)
        return [m for m in msgs if m.id > min_id][:limit]

    def add_event_handler(self, callback: Any, event: Any) -> None:
        self.handlers.append((callback, event))

    def remove_event_handler(self, callback: Any, event: Any = None) -> None:
        self.handlers = [(cb, ev) for (cb, ev) in self.handlers if cb is not callback]

    async def run_until_disconnected(self) -> None:
        self.run_until_disconnected_calls += 1
        if not self._event_batches:
            return  # bare drop, no events this cycle
        batch = self._event_batches.pop(0)
        for ev_obj in batch:
            await self._deliver(ev_obj)
        # return → simulate the connection dropping after delivering this batch.

    async def _deliver(self, ev_obj: Any) -> None:
        """Await the handler registered for this event's filter type."""
        if isinstance(ev_obj, FakeDeleteEvent):
            want = events.MessageDeleted
        elif getattr(ev_obj.message, "edit_date", None) is not None:
            want = events.MessageEdited
        else:
            want = events.NewMessage
        for cb, filt in self.handlers:
            if isinstance(filt, want):
                await cb(ev_obj)
                return


def _open_db(tmp_path) -> Any:
    return schema.open_writer_with_schema(tmp_path / "tgkb.sqlite3")


def _seed(conn: Any, chat_id: int, last_id: int, msgs: list[schema.Message] | None = None) -> None:
    with db.writing(conn):
        if msgs:
            schema.ingest_messages(conn, msgs)
        schema.advance_last_id(conn, chat_id, last_id, updated_at=_fixed_now())


def _run_sync(client, chats, throttle, conn, **kw):
    return asyncio.run(
        sync.run_sync(
            client, chats, throttle, conn,
            progress_cb=kw.pop("progress_cb", lambda _m: None),
            now_iso=kw.pop("now_iso", _fixed_now),
            # No real wall-clock wait in the reconnect backoff unless a test opts in.
            reconnect_backoff=kw.pop("reconnect_backoff", 0),
            **kw,
        )
    )


def _stop_after(n: int):
    """A should_continue predicate that allows exactly ``n`` loop iterations."""
    calls = {"n": 0}

    def _pred() -> bool:
        calls["n"] += 1
        return calls["n"] <= n

    return _pred


# --- Task 1: mark_deleted unit behavior ---------------------------------------


def test_mark_deleted_sets_flag_and_is_idempotent(tmp_path) -> None:
    """Task 1: mark_deleted sets deleted=1, is idempotent, and no-ops on a missing row."""
    conn = _open_db(tmp_path)
    try:
        with db.writing(conn):
            schema.ingest_messages(conn, [
                schema.Message(1001, 60, None, None, "живой", "2026-07-20T00:00:00+00:00", None, None),
            ])
        with db.writing(conn):
            schema.mark_deleted(conn, 1001, 60)
        assert conn.execute("SELECT deleted FROM messages WHERE message_id=60").fetchone()[0] == 1
        # Idempotent: a second mark stays 1.
        with db.writing(conn):
            schema.mark_deleted(conn, 1001, 60)
        assert conn.execute("SELECT deleted FROM messages WHERE message_id=60").fetchone()[0] == 1
        # Non-existent row: no-op, does not raise, creates nothing.
        with db.writing(conn):
            schema.mark_deleted(conn, 1001, 9999)
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE message_id=9999").fetchone()[0] == 0
    finally:
        conn.close()


# --- AC#1: a new message is ingested, searchable, and advances last_id atomically ---


def test_new_message_ingested_and_searchable(tmp_path) -> None:
    """AC#1 (core): a NewMessage event lands in the DB and is findable via a real MATCH."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[
                FakeEvent(_marked(1001), FakeMessage(101, text="Подписали договор аренды")),
            ]],
        )
        throttle, _ = _make_throttle()
        _run_sync(
            client, (TrackedChat(username="pub"),), throttle, conn,
            should_continue=_stop_after(1),
        )
        row = conn.execute("SELECT text FROM messages WHERE message_id=101").fetchone()
        rowids = schema.search_message_rowids(conn, "договор")
        last_id = schema.read_last_id(conn, 1001)
    finally:
        conn.close()
    assert row is not None and row[0] == "Подписали договор аренды"
    assert len(rowids) == 1  # searchable — exercised the messages_ai FTS trigger
    assert last_id == 101  # last_id advanced atomically with the row


def test_new_message_advances_last_id_atomically(tmp_path) -> None:
    """AC#1: after a NewMessage id=X ingest, read_last_id == X (row + mark land together)."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 200)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[FakeEvent(_marked(1001), FakeMessage(250, text="новое сообщение"))]],
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        assert schema.read_last_id(conn, 1001) == 250
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE message_id=250").fetchone()[0] == 1
    finally:
        conn.close()


# --- AC#2: an edit updates text + FTS and never regresses last_id ---------------


def test_edit_updates_text_and_fts_no_regress(tmp_path) -> None:
    """AC#2: an edit of an OLD message updates text+FTS but does NOT regress last_id."""
    conn = _open_db(tmp_path)
    try:
        # Seed id=50 with text A, mark last_id=100 (a later message already seen).
        _seed(conn, 1001, 100, [
            schema.Message(1001, 50, None, None, "старый договор", "2026-07-20T00:00:00+00:00", None, None),
        ])
        # Edit id=50 → text B (a NEW edit_date, as real Telegram always sets).
        edited = FakeMessage(50, text="новая встреча", edited=True)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[FakeEvent(_marked(1001), edited)]],
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        text = conn.execute("SELECT text FROM messages WHERE message_id=50").fetchone()[0]
        hit_new = schema.search_message_rowids(conn, "встреча")
        hit_old = schema.search_message_rowids(conn, "договор")
        last_id = schema.read_last_id(conn, 1001)
    finally:
        conn.close()
    assert text == "новая встреча"  # text updated via ON CONFLICT DO UPDATE
    assert len(hit_new) == 1  # FTS refreshed via messages_au → new lemma findable
    assert len(hit_old) == 0  # old lemma no longer indexed
    assert last_id == 100  # MAX guard held — editing id=50 did NOT regress 100


# --- AC#4: a deletion soft-deletes (row present, deleted=1), never hard-deletes --


def test_deletion_soft_deletes_and_excluded(tmp_path) -> None:
    """AC#4: a MessageDeleted event flips deleted=1; the row STILL EXISTS (not hard-deleted)."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100, [
            schema.Message(1001, 60, None, None, "удаляемое сообщение", "2026-07-20T00:00:00+00:00", None, None),
        ])
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[FakeDeleteEvent(_marked(1001), [60])]],
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        row = conn.execute("SELECT deleted FROM messages WHERE message_id=60").fetchone()
    finally:
        conn.close()
    assert row is not None  # NOT hard-deleted — the row is still present
    assert row[0] == 1  # tombstone flag set (query-time exclusion is Story 4.4)


def test_deletion_unattributable_chat_skipped(tmp_path) -> None:
    """AC#4 edge: a deletion with no resolvable chat_id is skipped, not misattributed."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100, [
            schema.Message(1001, 60, None, None, "жив", "2026-07-20T00:00:00+00:00", None, None),
        ])
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[FakeDeleteEvent(None, [60])]],  # unattributable
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        deleted = conn.execute("SELECT deleted FROM messages WHERE message_id=60").fetchone()[0]
    finally:
        conn.close()
    assert deleted == 0  # NOT touched — we never guess a chat_id


# --- AC#3: reconnect re-runs catch-up, and overlap creates no duplicates --------


def test_reconnect_reruns_catchup(tmp_path) -> None:
    """AC#3: the reconnect loop re-enters catch-up on a drop (run_until_disconnected twice)."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 0)
        # No events; each run_until_disconnected returns (a bare drop). Two loop
        # iterations → catch-up (get_messages sweep) entered on BOTH.
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            {1001: [FakeMessage(1, text="a")]},
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(2))
    finally:
        conn.close()
    assert client.run_until_disconnected_calls == 2  # re-listened after the drop
    # catch-up (a get_messages sweep) ran on BOTH iterations, not just the first.
    assert client.get_messages_calls >= 2


def test_reconnect_no_duplicates_on_overlap(tmp_path) -> None:
    """AC#3: a message ingested before a drop, then re-served by the reconnect catch-up, is not duplicated.

    Models the real overlap: id=101 was ingested LIVE before the connection dropped,
    but the persisted mark still trails at 100 (the pre-drop window). On reconnect,
    catch-up seeds ``min_id = read_last_id = 100`` (EXCLUSIVE), so its page GENUINELY
    re-serves id=101 — forcing the ``(chat_id, message_id)`` ON CONFLICT to fire.
    The assertion proves the collision was absorbed (row count stays 1), not that a
    ``min_id`` filter trivially excluded it (the prior version's weakness).
    """
    conn = _open_db(tmp_path)
    try:
        # id=101 already stored (as if ingested live pre-drop), but the mark trails
        # at 100 — so catch-up's min_id=100 page WILL re-serve 101 (a true overlap).
        _seed(conn, 1001, 100, [
            schema.Message(1001, 101, None, None, "договор", "2026-07-20T00:00:00+00:00", None, None),
        ])
        # The reconnect catch-up page re-serves id=101 (min_id=100 includes it) plus
        # a genuinely-new id=102. Only one loop iteration is needed to exercise the
        # recover catch-up over the overlap.
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            {1001: [FakeMessage(101, text="договор"), FakeMessage(102, text="новое")]},
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        count_101 = conn.execute("SELECT COUNT(*) FROM messages WHERE message_id=101").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        last_id = schema.read_last_id(conn, 1001)
    finally:
        conn.close()
    assert count_101 == 1  # the re-served id=101 was absorbed by ON CONFLICT — no duplicate
    assert total == 2  # exactly {101 (overlapped), 102 (new)}
    assert last_id == 102  # mark advanced monotonically over the overlap


# --- AC#1: only Tracked Chats are ingested -------------------------------------


def test_listener_only_ingests_tracked_chats(tmp_path) -> None:
    """AC#1 'in a Tracked Chat': an event for an UNTRACKED chat is not ingested."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[
                FakeEvent(_marked(2002), FakeMessage(500, text="чужое сообщение")),  # untracked
            ]],
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        untracked = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id=2002").fetchone()[0]
    finally:
        conn.close()
    assert untracked == 0  # the untracked-chat event was rejected by the chat_id guard


def test_channel_event_marked_id_ingests_under_raw_id(tmp_path) -> None:
    """Regression: a channel event whose chat_id is the MARKED peer id ingests under the RAW id.

    The bug: `event.chat_id` for a channel is the marked `-100…` peer id
    (`get_peer_id(entity)`), but the row's canonical `chat_id` — and what catch-up
    and every stored row use — is the raw `entity.id`. Before the fix, the handler
    compared the marked `event.chat_id` against a tracked set keyed by the raw id,
    so it NEVER matched for a channel and every live event was silently dropped.
    Here: raw id 1001 → marked id `_marked(1001)` (e.g. -1000000001001); a NewMessage
    with that marked chat_id must (a) NOT be skipped and (b) land under chat_id=1001.
    """
    conn = _open_db(tmp_path)
    marked = _marked(1001)
    assert marked != 1001  # sanity: the two id spaces genuinely differ for a channel
    try:
        _seed(conn, 1001, 100)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[FakeEvent(marked, FakeMessage(101, text="сообщение канала"))]],
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        under_raw = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id=1001 AND message_id=101").fetchone()[0]
        under_marked = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (marked,)).fetchone()[0]
        last_id = schema.read_last_id(conn, 1001)
    finally:
        conn.close()
    assert under_raw == 1  # ingested under the RAW chat_id (matches catch-up + stored rows)
    assert under_marked == 0  # never stored under the marked -100… id
    assert last_id == 101  # and the raw chat's high-water mark advanced


def test_handlers_registered_with_tracked_entities(tmp_path) -> None:
    """AC#1: the three handlers are registered, each filtered to the resolved tracked entity."""
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100)
        entity = _channel_entity(1001)
        # Capture the handler set at the moment of listening (before the loop detaches
        # on the drop) by stopping after one iteration and inspecting via a spy filter.
        registered: list[Any] = []

        class CapturingClient(ListenerFakeClient):
            def add_event_handler(self, callback: Any, event: Any) -> None:  # type: ignore[override]
                registered.append(event)
                super().add_event_handler(callback, event)

        client = CapturingClient({"pub": entity})
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
    finally:
        conn.close()
    types = {type(e) for e in registered}
    assert events.NewMessage in types
    assert events.MessageEdited in types
    assert events.MessageDeleted in types


# --- Task 4: the WAL checkpoint fires on cadence, between transactions ----------


def test_checkpoint_runs_on_cadence(tmp_path, monkeypatch) -> None:
    """Task 4: with a low cadence, ingesting enough events triggers db.checkpoint BETWEEN txns."""
    conn = _open_db(tmp_path)
    calls = {"n": 0, "in_txn": []}
    real_checkpoint = db.checkpoint

    def _spy_checkpoint(c: Any) -> None:
        calls["n"] += 1
        # A PRAGMA wal_checkpoint cannot run inside an open transaction — prove we
        # are BETWEEN transactions by confirming a fresh BEGIN IMMEDIATE succeeds
        # (it would raise "cannot start a transaction within a transaction" if we
        # were still inside one), then roll it back and run the real checkpoint.
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute("ROLLBACK")
            calls["in_txn"].append(False)
        except Exception:  # noqa: BLE001
            calls["in_txn"].append(True)
        real_checkpoint(c)

    monkeypatch.setattr(sync.db, "checkpoint", _spy_checkpoint)

    try:
        _seed(conn, 1001, 0)
        # cadence=2 → a checkpoint after every 2 ingested events. Three events → 1.
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[
                FakeEvent(_marked(1001), FakeMessage(1, text="один")),
                FakeEvent(_marked(1001), FakeMessage(2, text="два")),
                FakeEvent(_marked(1001), FakeMessage(3, text="три")),
            ]],
        )
        throttle, _ = _make_throttle()
        _run_sync(
            client, (TrackedChat(username="pub"),), throttle, conn,
            should_continue=_stop_after(1), checkpoint_every=2,
        )
    finally:
        conn.close()
    assert calls["n"] >= 1  # the checkpoint fired on the chosen cadence
    assert all(flag is False for flag in calls["in_txn"])  # always between transactions


# --- FR-3: a bad chat is skipped at listener startup, the listener still runs ---


def test_listener_startup_skips_bad_chat(tmp_path) -> None:
    """FR-3: an unresolvable chat is skipped at listener startup; the good chat still listens."""
    from telethon.errors import ChannelPrivateError

    conn = _open_db(tmp_path)
    msgs: list[str] = []
    try:
        _seed(conn, 1001, 100)
        client = ListenerFakeClient(
            {"bad": ChannelPrivateError(request=None), "good": _channel_entity(1001)},
            event_batches=[[FakeEvent(_marked(1001), FakeMessage(101, text="сообщение"))]],
        )
        throttle, _ = _make_throttle()
        _run_sync(
            client, (TrackedChat(username="bad"), TrackedChat(username="good")),
            throttle, conn, progress_cb=msgs.append, should_continue=_stop_after(1),
        )
        ingested = conn.execute("SELECT COUNT(*) FROM messages WHERE message_id=101").fetchone()[0]
    finally:
        conn.close()
    assert ingested == 1  # the good chat's live event was ingested
    assert any("@bad" in m and "ChannelPrivateError" in m for m in msgs)


# --- Media-only live event: not ingested, but last_id still advances -----------


def test_media_only_live_event_advances_last_id_without_ingest(tmp_path) -> None:
    """A media-only/service NewMessage (falsy text) is NOT ingested, but last_id advances.

    Mirrors the paging engine's all-media-page behavior (the seen id must still move
    the high-water mark), and covers the ``_map_message`` → None branch of the live
    handler, which was previously untested (Acceptance Auditor gap).
    """
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[FakeEvent(_marked(1001), FakeMessage(150, text=None))]],  # media-only
        )
        throttle, _ = _make_throttle()
        _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        rows = conn.execute("SELECT COUNT(*) FROM messages WHERE message_id=150").fetchone()[0]
        last_id = schema.read_last_id(conn, 1001)
    finally:
        conn.close()
    assert rows == 0  # media-only message NOT ingested (text-only rule)
    assert last_id == 150  # but the high-water mark still advanced to the seen id


# --- Task/review: the reconnect loop backs off before re-running catch-up -------


def test_reconnect_backoff_sleeps_between_iterations(tmp_path) -> None:
    """The recover loop pauses `reconnect_backoff` seconds before re-running catch-up.

    Prevents a persistent-drop storm from tight-spinning the loop. The FIRST catch-up
    must run WITHOUT a pause (no startup delay); the pause applies only between
    iterations. Uses an injected async sleep-spy so no wall-clock time passes.
    """
    conn = _open_db(tmp_path)
    slept: list[float] = []

    async def _sleep_spy(seconds: float) -> None:
        slept.append(seconds)

    try:
        _seed(conn, 1001, 0)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            {1001: [FakeMessage(1, text="a")]},
        )
        throttle, _ = _make_throttle()
        # Three iterations → two between-iteration backoffs (none before the first).
        _run_sync(
            client, (TrackedChat(username="pub"),), throttle, conn,
            should_continue=_stop_after(3), reconnect_backoff=7.5, sleep=_sleep_spy,
        )
    finally:
        conn.close()
    assert slept == [7.5, 7.5]  # backed off before iterations 2 and 3, not before 1


# --- DEBUG event tracing: caught events are logged (type + ids, never text) -----


def test_debug_logs_caught_and_skipped_events(tmp_path, caplog) -> None:
    """At DEBUG, the listener traces each ingested/skipped event — type + chat_id + message_id, never text.

    This is the diagnostic the operator uses when a message does not appear to
    ingest: DEBUG shows whether the event arrived and under which chat_id, so a
    chat_id mismatch (the resolved entity id vs. the incoming event) is visible.
    Message text must NEVER appear in the logs (NFR-1).
    """
    conn = _open_db(tmp_path)
    try:
        _seed(conn, 1001, 100)
        client = ListenerFakeClient(
            {"pub": _channel_entity(1001)},
            event_batches=[[
                FakeEvent(_marked(1001), FakeMessage(101, text="секретный текст договора")),  # tracked
                FakeEvent(_marked(2002), FakeMessage(500, text="чужое")),  # untracked → skip-logged
            ]],
        )
        throttle, _ = _make_throttle()
        with caplog.at_level(logging.DEBUG, logger="tgkb.writer.sync"):
            _run_sync(client, (TrackedChat(username="pub"),), throttle, conn, should_continue=_stop_after(1))
        log_text = "\n".join(r.getMessage() for r in caplog.records)
    finally:
        conn.close()
    assert "NewMessage ingest" in log_text and "message_id=101" in log_text
    # The untracked event is traced by its MARKED chat_id (what event.chat_id carries).
    assert "skipped" in log_text and str(_marked(2002)) in log_text
    # NFR-1: no message text ever reaches the logs.
    assert "секретный текст договора" not in log_text
    assert "чужое" not in log_text
