"""Realtime sync: catch-up + a long-running live listener (Story 4.2).

This module turns ``sync`` from Story 4.1's **catch-up-then-exit** pass into a
**catch-up-then-listen-live** loop (FR-9, AD-5, AD-13). :func:`run_sync` first
runs :func:`tgkb.writer.export.run_catchup` (the warm-up: backfill the offline
gap from each chat's ``sync_state.last_id``), then attaches three Telethon event
handlers — ``events.NewMessage`` / ``events.MessageEdited`` / ``events.MessageDeleted``,
each filtered to the Tracked Chats — and ``run_until_disconnected()`` so the
process stays up and ingests live. On a connection drop it **re-runs catch-up**
before listening again, so a disconnected window loses nothing (AC#3).

**Almost everything here is REUSE.** Every live event goes through the SAME
ingest primitives Epic 2 built: :func:`export._map_message` → ``ingest_messages``
+ ``advance_last_id`` in ONE :func:`tgkb.common.db.writing` scope, and the FTS
row is maintained by the existing ``messages_ai`` / ``messages_au`` triggers in
that same transaction (AD-12). The catch-up engine (:func:`export.run_catchup`)
is imported and reused VERBATIM as both the warm-up and the recover-on-drop step
— never re-derived. The genuinely NEW mechanics this module owns:

1. Wiring the three Telethon event handlers to the existing ingest path.
2. A reconnect loop that re-runs catch-up on drop (AC#3).
3. WAL checkpoint cadence for the now-long-running process (deferred-work.md
   #story-1.2 assigned it HERE — a listener that stays up for days would grow the
   WAL unboundedly without periodic ``PRAGMA wal_checkpoint``).

**Where the listener lives (a decision Story 4.1 deferred to here).** Story 4.1's
Dev Notes said the realtime listener "can live in a new ``sync.py`` when it
arrives." It lives here — a NEW module — because catch-up (bounded, exit-when-done)
and the live listener (long-running, event-driven, reconnect loop) are distinct
concerns: ``export.py`` stays the "bulk + catch-up FETCH" module and ``sync.py``
is the "live LISTEN + recover" module, while the shared fetch/ingest engine stays
single-sourced in ``export.py`` (imported, never re-typed).

**Reconnect strategy — option (A), the explicit re-catch-up loop (chosen).** Of
the two clean shapes the Dev Notes describe, this module uses (A): a
``while not stopping:`` loop that, each iteration, runs catch-up, attaches the
handlers, awaits ``run_until_disconnected()``, and — on a return that is NOT a
deliberate stop — detaches the handlers and loops back to re-run catch-up. This
literally satisfies AC#3 ("re-running the catch-up path from ``last_id``") and is
the testable shape (a fake ``run_until_disconnected`` that returns simulates a
drop; a stop flag ends the loop). Handlers are DETACHED before re-attaching so a
reconnect never double-registers (double ingestion) — idempotent ingest would
absorb a duplicate anyway, but detaching keeps the handler set clean.

**Account safety (AD-10 / NFR-3).** The AD-10 throttle governs OUTBOUND requests
WE initiate (``get_entity`` at startup, ``get_messages`` during catch-up).
INBOUND events pushed by Telegram are NOT our requests, so event ingest is NOT
throttled. The listener is a read-only OBSERVER: it never calls ``mark_read``,
replies, or issues any outbound action — no traffic the throttle did not sanction.

**Dependency direction (AD-1).** This module imports only stdlib + telethon +
:mod:`tgkb.common` (+ :mod:`tgkb.writer.export` / :mod:`tgkb.writer.throttle`). It
MUST NEVER import ``tgkb.reader``.

**Secret-safe (NFR-1).** Nothing here logs ``api_hash``, the session token, or
message text. Event handlers store ``text`` in the DB (that IS the product) but
never print or log it; error/skip lines name identities (ids/@usernames — not
secrets) and exception TYPES only.
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import Any, Awaitable, Callable, Protocol

from telethon import events
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    RPCError,
)
from telethon.utils import get_display_name, get_peer_id

from tgkb.common import db, schema
from tgkb.common.config import TrackedChat
from tgkb.writer import export
from tgkb.writer.throttle import RequestThrottle

_log = logging.getLogger(__name__)

# WAL checkpoint cadence (Task 4, deferred-work.md#story-1.2 / AD-3). The
# long-running listener checkpoints every N ingested events — NOT per-message
# (that would defeat WAL's batching benefit) and NOT never (the WAL would grow
# unbounded over a multi-day run). A simple write-counter is enough for v1 (the
# Dev Notes explicitly forbid a sophisticated scheduler). 500 is a coarse,
# defensible default; single-sourced here and injectable by tests.
CHECKPOINT_EVERY = 500

# A bounded pause before the reconnect loop re-runs catch-up after a drop. Without
# it, a persistent-connectivity failure (`run_until_disconnected` returning
# immediately, over and over) would tight-spin the recover loop — re-resolving
# entities + re-paging catch-up with zero delay, burning CPU and API quota even
# though the AD-10 throttle caps the outbound *rate*. A small fixed pause keeps the
# recover loop calm during an outage (v1: a simple constant, not exponential
# backoff — matching the "keep it simple" discipline of the checkpoint cadence).
# Single-sourced here and injectable by tests (which pass 0 to avoid a real wait).
RECONNECT_BACKOFF_SECONDS = 5.0


class SyncClient(Protocol):
    """The minimal shape :func:`run_sync` needs from a connected Telethon client.

    A superset of :class:`export.CatchupClient` (catch-up pages off ``get_messages``)
    plus the realtime seams: ``add_event_handler`` / ``remove_event_handler`` to
    (de)register the three handlers, and ``run_until_disconnected`` to block while
    ingesting live. Declared as a ``Protocol`` (structural typing) so tests pass a
    lightweight fake; the real :class:`telethon.TelegramClient` satisfies it.
    """

    async def get_entity(self, entity: object) -> object: ...

    async def get_messages(self, entity: object, **kwargs: object) -> object: ...

    def add_event_handler(self, callback: object, event: object) -> None: ...

    def remove_event_handler(self, callback: object, event: object = ...) -> None: ...

    async def run_until_disconnected(self) -> object: ...


async def _resolve_tracked_entities(
    client: SyncClient,
    chats: tuple[TrackedChat, ...],
    throttle: RequestThrottle,
    progress_cb: export.ProgressCallback,
) -> tuple[list[object], dict[int, int]]:
    """Resolve each Tracked Chat's entity ONCE (through the throttle) for the filter.

    The ``events.*`` handlers must only fire for the CONFIGURED Tracked Chats, not
    every chat the account is in (AC#1 "in a Tracked Chat"). We resolve each chat's
    entity here — reusing the EXACT ``chat.username if chat.username is not None
    else chat.id`` selector + the FR-3 ``FloodWaitError``-before-``RPCError``
    skip-and-continue ordering from :func:`export._resolve_and_page_one_chat`.

    Returns (a) the resolved entities to build ``chats=`` from and (b) a
    ``marked-peer-id → raw entity.id`` map. **The two id spaces MUST be bridged
    here** (the fix for the channel/supergroup live-ingest bug): Telethon's
    ``event.chat_id`` is the *marked* peer id (channels/supergroups carry the
    ``-100…`` prefix, e.g. ``-1004308125346``), computed via
    :func:`telethon.utils.get_peer_id`, whereas ``entity.id`` — the id catch-up and
    ingest use as the canonical ``chat_id`` — is the *raw* id (``4308125346``). If
    the handler compared ``event.chat_id`` directly against ``entity.id`` they would
    NEVER match for a channel, so every live event was silently dropped by the
    defence-in-depth guard while ``chats=`` had already (correctly) admitted it.
    So the KEY is ``get_peer_id(entity)`` (matches ``event.chat_id``) and the VALUE
    is ``entity.id`` (the raw chat_id we ingest under — agrees with catch-up and the
    already-stored rows). The handler looks up by the marked key and ingests under
    the raw value.

    A chat that cannot be resolved is reported and skipped; the listener still runs
    for the rest (FR-3). A ``FloodWaitError`` that survived the throttle is an
    account-safety signal and is RE-RAISED (it must precede the reachability
    handler — ``FloodWaitError`` subclasses ``RPCError``).
    """
    resolved: list[object] = []
    tracked_ids: dict[int, int] = {}
    for chat in chats:
        identity = export._chat_identity(chat)
        selector = chat.username if chat.username is not None else chat.id
        try:
            entity = await throttle.run(
                lambda s=selector: client.get_entity(s)  # type: ignore[attr-defined]
            )
        except FloodWaitError:
            # Account-safety signal — re-raise (MUST precede the RPCError handler,
            # which FloodWaitError subclasses). Same ordering as export.
            raise
        except (ValueError, ChannelPrivateError, RPCError) as exc:
            reason = type(exc).__name__
            progress_cb(
                f"Skipping {identity}: could not resolve/reach it ({reason}). "
                "Check the config entry; the listener continues for the other chats."
            )
            continue
        raw_id = entity.id  # type: ignore[attr-defined]  # canonical chat_id (ingest under this)
        marked_id = get_peer_id(entity)  # matches event.chat_id (the -100… form for channels)
        resolved.append(entity)
        tracked_ids[marked_id] = raw_id
    return resolved, tracked_ids


def _make_handlers(
    conn: object,
    tracked_ids: dict[int, int],
    now_iso: Callable[[], str],
    on_write: Callable[[], None],
    progress_cb: export.ProgressCallback,
) -> tuple[Callable[[Any], Awaitable[None]], ...]:
    """Build the three async event handlers (new / edited / deleted).

    Each handler does the ONE new I/O this story adds — a DB write in a single
    :func:`tgkb.common.db.writing` scope — and NOTHING else (no outbound Telegram
    action, no throttle: inbound events are not our requests, AD-10). ``on_write``
    is invoked AFTER each ``writing()`` scope closes so the caller can run the WAL
    checkpoint BETWEEN transactions (Task 4 — a ``PRAGMA wal_checkpoint`` cannot run
    inside an open transaction).

    New + edited share the SAME path (``_map_message`` → ``ingest_messages`` +
    ``advance_last_id``): the ``_INGEST_SQL`` ``ON CONFLICT DO UPDATE`` upserts an
    edit's text+edit_date (AC#2) and the ``MAX`` guard in ``advance_last_id`` means
    editing an OLD message never regresses ``last_id`` (AC#2). Deletion flips the
    ``deleted`` tombstone via :func:`schema.mark_deleted` (AC#4).
    """

    def _ingest_one(chat_id: int, message: object) -> None:
        row = export._map_message(chat_id, message)
        if row is None:
            # Service / media-only message (falsy text) — skipped, not ingested
            # (the AC#4-of-2.4 text-only rule). Still advance last_id to the seen
            # id so the high-water mark reflects it (matching the paging engine).
            with db.writing(conn):  # type: ignore[arg-type]
                schema.advance_last_id(
                    conn, chat_id, message.id, updated_at=now_iso()  # type: ignore[attr-defined]
                )
        else:
            # AC#1/#2: ingest (or edit-upsert) the row AND advance last_id in ONE
            # transaction (AD-5 atomic advance). The messages_ai / messages_au
            # trigger maintains FTS in this SAME scope → immediately searchable.
            with db.writing(conn):  # type: ignore[arg-type]
                schema.ingest_messages(conn, [row])  # type: ignore[arg-type]
                schema.advance_last_id(
                    conn, chat_id, message.id, updated_at=now_iso()  # type: ignore[attr-defined]
                )
        on_write()

    async def on_new_message(event: Any) -> None:
        # event.chat_id is the MARKED peer id (channels carry the -100… prefix);
        # tracked_ids is keyed by that marked id. The VALUE is the raw entity.id we
        # ingest under (agrees with catch-up + already-stored rows).
        event_chat_id = event.chat_id
        raw_chat_id = tracked_ids.get(event_chat_id)
        msg_id = getattr(getattr(event, "message", None), "id", None)
        if raw_chat_id is None:
            # DEBUG trace only — never the text (NFR-1). Explains "nothing ingested"
            # when an event arrives for a chat not in the tracked (marked-id) set.
            _log.debug(
                "NewMessage skipped: chat_id=%s not in tracked %s (message_id=%s)",
                event_chat_id, sorted(tracked_ids), msg_id,
            )
            return
        _log.debug(
            "NewMessage ingest: chat_id=%s (raw=%s) message_id=%s",
            event_chat_id, raw_chat_id, msg_id,
        )
        _ingest_one(raw_chat_id, event.message)

    async def on_message_edited(event: Any) -> None:
        event_chat_id = event.chat_id
        raw_chat_id = tracked_ids.get(event_chat_id)
        msg_id = getattr(getattr(event, "message", None), "id", None)
        if raw_chat_id is None:
            _log.debug(
                "MessageEdited skipped: chat_id=%s not in tracked %s (message_id=%s)",
                event_chat_id, sorted(tracked_ids), msg_id,
            )
            return
        _log.debug(
            "MessageEdited ingest: chat_id=%s (raw=%s) message_id=%s",
            event_chat_id, raw_chat_id, msg_id,
        )
        _ingest_one(raw_chat_id, event.message)

    async def on_message_deleted(event: Any) -> None:
        # MessageDeleted carries `deleted_ids` (a list); its `chat_id` is reliably
        # populated only for channels/supergroups. For ordinary users/small groups
        # Telegram's delete update may not identify the peer → log-and-skip (do NOT
        # guess a chat_id and tombstone the wrong chat's message). Active
        # deletion-mirroring is a v1 non-goal (AC#4), so a missed attribution is a
        # known, documented limitation, not a correctness bug.
        # event.chat_id is the MARKED peer id; map it to the raw chat_id we store
        # under. `None`/unknown → log-and-skip (do NOT guess a chat_id).
        event_chat_id = getattr(event, "chat_id", None)
        raw_chat_id = tracked_ids.get(event_chat_id) if event_chat_id is not None else None
        deleted_ids = getattr(event, "deleted_ids", None) or []
        if raw_chat_id is None:
            progress_cb(
                "Deletion event not attributable to a Tracked Chat "
                f"(chat_id={event_chat_id}); skipping (v1 does not mirror deletions)."
            )
            return
        if not deleted_ids:
            return
        _log.debug(
            "MessageDeleted: chat_id=%s (raw=%s) deleted_ids=%s",
            event_chat_id, raw_chat_id, list(deleted_ids),
        )
        # AC#4: soft-delete each id (never hard-delete) in ONE transaction. The
        # messages_au trigger refreshes FTS in the same scope; query-time exclusion
        # is Story 4.4's job.
        with db.writing(conn):  # type: ignore[arg-type]
            for msg_id in deleted_ids:
                schema.mark_deleted(conn, raw_chat_id, msg_id)  # type: ignore[arg-type]
        on_write()

    return on_new_message, on_message_edited, on_message_deleted


async def run_sync(
    client: SyncClient,
    chats: tuple[TrackedChat, ...],
    throttle: RequestThrottle,
    conn: object,
    *,
    progress_cb: export.ProgressCallback = export._default_progress,
    now_iso: Callable[[], str] | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
    should_continue: Callable[[], bool] | None = None,
    reconnect_backoff: float = RECONNECT_BACKOFF_SECONDS,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Run catch-up, then listen live, re-running catch-up on every reconnect (AC#1–#4).

    The long-running entry point ``sync`` calls (via the CLI ``_run_sync_flow``).
    Structured as the AC#3 explicit re-catch-up loop:

    1. ``await export.run_catchup(...)`` — backfill the gap since ``last_id`` (first
       iteration: the offline gap; later iterations: whatever arrived while
       disconnected).
    2. Resolve the Tracked Chats' entities (once) and attach the three event
       handlers filtered to them.
    3. ``await client.run_until_disconnected()`` — block, ingesting live, until the
       connection drops or the process is signalled.
    4. Detach the handlers; if this was a drop (not a deliberate stop), loop back to
       step 1 to sweep the disconnected-window gap and re-attach.

    Re-running catch-up after a drop overlaps whatever was ingested live before it;
    the ``(chat_id, message_id)`` ``ON CONFLICT`` absorbs it (zero duplicates) and
    the ``MAX`` guard keeps ``last_id`` monotonic — safe BY CONSTRUCTION, no dedup
    logic needed (AC#3, inherited from 2.2/4.1).

    :param client: an already-connected, authorized client with
        ``flood_sleep_threshold=0`` set by the caller (Story 2.3's #1 seam).
    :param chats: the valid Tracked Chats (Story 2.1 guarantee).
    :param throttle: the single AD-10 throttle every OUTBOUND call routes through.
    :param conn: an open writer ``apsw.Connection`` with the schema ensured.
    :param progress_cb: sink for secret-free progress lines.
    :param now_iso: injectable UTC-now-as-ISO-string source (tests pin it).
    :param checkpoint_every: WAL checkpoint cadence (writes between checkpoints);
        injectable so tests can drive the cadence deterministically.
    :param should_continue: a stop predicate — the loop re-enters catch-up only
        while it returns ``True``. Default: run forever (a real drop returns from
        ``run_until_disconnected``; a Ctrl-C raises ``KeyboardInterrupt`` and the
        CLI's ``finally`` disconnects). Tests inject a flag to end the loop after a
        simulated drop.
    :param reconnect_backoff: seconds to pause before re-running catch-up after a
        drop (NOT before the first catch-up). Keeps a persistent-drop storm from
        tight-spinning the recover loop. Tests pass ``0`` to avoid a real wait.
    :param sleep: injectable async sleep (default :func:`asyncio.sleep`) — the
        seam the backoff uses, so tests drive it without wall-clock time.
    """
    if now_iso is None:
        from datetime import datetime

        def now_iso() -> str:  # noqa: E306 - tiny local default
            return datetime.now(timezone.utc).isoformat()

    if should_continue is None:
        def should_continue() -> bool:  # noqa: E306 - default: loop forever
            return True

    if sleep is None:
        import asyncio

        sleep = asyncio.sleep

    first_iteration = True

    # Write-counter for the WAL checkpoint cadence (Task 4). Closed over by the
    # `on_write` hook the handlers call after each transaction; the checkpoint runs
    # BETWEEN transactions (never inside a `db.writing` scope — the PRAGMA can't).
    write_count = 0

    def on_write() -> None:
        nonlocal write_count
        write_count += 1
        if checkpoint_every > 0 and write_count % checkpoint_every == 0:
            # Between transactions here (each handler closed its writing() scope
            # before calling on_write). checkpoint() manages its own PRAGMA.
            db.checkpoint(conn)  # type: ignore[arg-type]

    while should_continue():
        if not first_iteration:
            # Reached only when the loop WILL re-run after a drop, so this notice is
            # accurate (unlike an unconditional message after run_until_disconnected).
            progress_cb("Re-running catch-up before re-listening…")
            if reconnect_backoff > 0:
                # A bounded pause before recovering, so a persistent-drop storm does
                # not tight-spin this loop (see RECONNECT_BACKOFF_SECONDS). Only
                # between iterations — the FIRST catch-up runs immediately.
                await sleep(reconnect_backoff)
        first_iteration = False

        # Step 1 — warm-up / recover: sweep the (offline or disconnected-window)
        # gap. Reused VERBATIM as both the first backfill and the reconnect recover
        # step (AC#3). Idempotent overlap makes the re-run safe.
        await export.run_catchup(
            client, chats, throttle, conn, progress_cb=progress_cb, now_iso=now_iso
        )

        # Step 2 — resolve the tracked entities (once per iteration, through the
        # throttle) and attach the three handlers filtered to them (AC#1/#2/#4).
        entities, tracked_ids = await _resolve_tracked_entities(
            client, chats, throttle, progress_cb
        )
        if not tracked_ids:
            progress_cb(
                "No Tracked Chats could be resolved for the live listener; "
                "nothing to listen to."
            )
            return

        handlers = _make_handlers(conn, tracked_ids, now_iso, on_write, progress_cb)
        on_new, on_edit, on_delete = handlers
        # The chats= filter narrows each event to the resolved tracked entities so
        # the account's other chats are never ingested (AC#1).
        client.add_event_handler(on_new, events.NewMessage(chats=entities))
        client.add_event_handler(on_edit, events.MessageEdited(chats=entities))
        client.add_event_handler(on_delete, events.MessageDeleted(chats=entities))

        # Surface WHICH chats the listener is subscribed to (INFO, so it shows
        # without --log-level DEBUG). Shows the raw chat_id (what rows are stored
        # under, no `-100…` prefix) alongside the marked peer id that incoming
        # `event.chat_id` values carry — so an id-space mismatch is diagnosable at a
        # glance. Ids are not secrets.
        progress_cb(
            "Listening on chat_id(s) [raw←event]: "
            + ", ".join(f"{raw}←{marked}" for marked, raw in sorted(tracked_ids.items()))
        )
        progress_cb("Listening for new messages (Ctrl-C to stop)…")
        try:
            # Step 3 — block here, ingesting live, until a drop or a deliberate stop.
            await client.run_until_disconnected()
        finally:
            # Step 4 — detach on every exit so a reconnect never double-registers
            # (and a clean stop leaves no dangling handlers).
            client.remove_event_handler(on_new)
            client.remove_event_handler(on_edit)
            client.remove_event_handler(on_delete)

        # run_until_disconnected returned = the connection dropped (a deliberate
        # Ctrl-C raises KeyboardInterrupt, which propagates past this loop to the
        # CLI's finally, so control only reaches here on a drop). The next
        # `while should_continue()` decides whether to loop back; the reconnect
        # notice is emitted at the TOP of the next iteration (see below) so it never
        # prints when the loop is actually about to stop — the previous
        # unconditional message here was misleading on the final iteration.
        progress_cb("Connection dropped.")


__all__ = [
    "CHECKPOINT_EVERY",
    "RECONNECT_BACKOFF_SECONDS",
    "SyncClient",
    "run_sync",
]
