"""Bulk history export via a Telethon takeout session (Story 2.4).

This module is the **first real Telegram fetch loop**: it opens a Telethon
**takeout** (export-mode) session and pulls each Tracked Chat's text-message
history **through the Story 2.3 throttle**, ingesting it **idempotently** via the
Story 2.2 schema and advancing each chat's ``sync_state.last_id`` so Epic 4
catch-up can resume from the high-water mark. It is the assembly point for
everything Epic 2 built; it invents nothing that already exists.

Architecture invariants enforced here:

- **AD-10 (account safety — the heart of AC#1):** bulk export uses a Telethon
  ``takeout`` session (``invokeWithTakeout``, lower flood limits) AND every
  outbound Telegram request — entity resolution *and* every history page — is
  routed through :meth:`tgkb.writer.throttle.RequestThrottle.run`. There is NO
  code path that calls Telegram outside the throttle. History is paged
  **manually** with ``get_messages`` (one page per ``throttle.run``) rather than
  letting ``iter_messages`` page internally, because the throttle wraps ONE
  rebuildable awaitable and an async-iterator's internal ``GetHistoryRequest``
  pages would bypass it.
- **AD-4 (idempotent identity):** rows carry ``(chat_id, message_id)``; ingest is
  ``INSERT … ON CONFLICT DO NOTHING``/edit-upsert (Story 2.2), so a re-run or an
  overlapping range creates no duplicates (AC#3). No dedup logic lives here.
- **AD-5 (monotonic high-water, crash-safe):** ``ingest_messages`` AND
  ``advance_last_id`` run in the SAME :func:`tgkb.common.db.writing`
  (``BEGIN IMMEDIATE``) scope per page, so messages and their ``last_id`` land
  together — never messages-without-mark or mark-ahead-of-messages. Export is
  oldest-first (``reverse=True``) so ``last_id`` grows page by page and a crash
  leaves a truthful mark (AC#6).
- **AD-12 (composability):** the schema ingest primitives never open their own
  transaction — this module wraps them — so Story 3.2 can later slot the
  ``messages_fts`` write into the SAME per-page scope.
- **AD-1 (dependency direction):** ``writer/export.py`` imports only stdlib +
  telethon + :mod:`tgkb.common` (+ :mod:`tgkb.writer.throttle`). It MUST NEVER
  import ``tgkb.reader``.

Text-only MVP (AC#4): media-only and service messages (falsy ``msg.message``)
are skipped — not ingested, and the export does NOT abort. Media download is a
PRD non-goal; takeout is opened without ``files``.

Reachability skip-and-continue (FR-3, the Story 2.1 extension): Story 2.1
validated config OFFLINE and explicitly deferred *reachability* here with a
binding instruction — extend the SAME skip-and-continue model. One
bad/unreachable Tracked Chat reports a clear, secret-free error and the run
continues for the rest.

Secret-safe by construction (NFR-1): nothing here logs ``api_hash``, the session
token, or message text. ``msg.message`` is stored in the DB (that IS the
product) but is never printed or logged; error paths name the exception TYPE,
never its value.

Testability: :func:`run_export` takes an already-connected client + the chats +
the throttle + an open writer connection as params — never secrets, never a
passphrase — so it is unit-testable with a fake client and a tmp-file DB (the
CLI owns credential/session sourcing; the mechanism receives concrete values,
mirroring :func:`tgkb.writer.auth.authenticate`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import Awaitable, Callable, Protocol

from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    RPCError,
    TakeoutInitDelayError,
)
from telethon.utils import get_display_name

from tgkb.common import db, schema
from tgkb.common.config import TrackedChat
from tgkb.writer.throttle import RequestThrottle

# ``tgkb``-namespaced logger (matching throttle.py / auth.py) so the CLI can
# route/level export output; the throttle's FLOOD_WAIT warnings share the
# ``tgkb.writer.*`` namespace and are NOT suppressed by the telethon-only cap.
_log = logging.getLogger(__name__)

# How many messages to pull per throttled ``get_messages`` page. One page = one
# ``throttle.run`` = one ``writing()`` transaction, so this bounds both the
# request size and the write-lock hold time / WAL growth per commit (a single
# multi-hundred-thousand-row transaction would hold the write lock and bloat the
# WAL for the whole export). Single-sourced here (the project's
# single-source-constant convention). 200 is Telegram's typical history page.
PAGE_SIZE = 200


class TakeoutClient(Protocol):
    """The minimal shape :func:`run_export` needs from a connected Telethon client.

    Declared as a ``Protocol`` (structural typing) so tests can pass a lightweight
    fake without subclassing ``TelegramClient``. The real
    :class:`telethon.TelegramClient` satisfies it.
    """

    def takeout(self, **kwargs: object) -> object:  # returns an async-context proxy
        ...

    async def get_entity(self, entity: object) -> object: ...


class CatchupClient(Protocol):
    """The minimal shape :func:`run_catchup` needs from a connected Telethon client.

    Unlike :class:`TakeoutClient`, catch-up pages history directly off the client
    (a regular session — NO takeout), so this Protocol needs ``get_messages`` on the
    client itself. Structural typing so tests can pass a lightweight fake.
    """

    async def get_entity(self, entity: object) -> object: ...

    async def get_messages(self, entity: object, **kwargs: object) -> object: ...


@dataclass
class ExportSummary:
    """The outcome of a :func:`run_export` run (returned for the CLI/tests).

    - ``exported`` maps ``chat_id`` → count of message rows ingested (text only).
    - ``skipped_chats`` lists ``(identity, reason_type)`` for each Tracked Chat
      that could not be resolved/reached (FR-3 skip-and-continue); ``identity`` is
      the config ``@username``/``id`` (never a secret) and ``reason_type`` is the
      exception TYPE name (never its value — NFR-1).
    - ``total_messages`` is the sum across chats, for a final one-line summary.
    """

    exported: dict[int, int] = field(default_factory=dict)
    skipped_chats: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_messages(self) -> int:
        return sum(self.exported.values())


# A progress callback receives a single human-readable, secret-free line. The
# default prints to stderr (mirroring how auth.py surfaces decision points), so a
# long export never looks hung; tests inject a recorder to assert progress
# without capturing stdout timing.
ProgressCallback = Callable[[str], None]


def _default_progress(message: str) -> None:
    """Default progress sink: log at INFO on the ``tgkb.writer.export`` logger.

    Using the logger (not bare ``print``) keeps progress on the same
    ``tgkb``-namespaced channel as the throttle's FLOOD_WAIT warnings, so a single
    logging config surfaces both to the user.
    """
    _log.info("%s", message)


def _chat_identity(chat: TrackedChat) -> str:
    """A short, secret-free label for a Tracked Chat (``@name`` or ``id=…``).

    Used in progress + skip messages so the report identifies THAT chat. Ids are
    not secrets; api_hash/session/text never appear here.
    """
    return f"@{chat.username}" if chat.username is not None else f"id={chat.id}"


def _to_iso_utc(dt: object) -> str | None:
    """Convert a Telethon tz-aware ``datetime`` to the pinned ISO-8601 UTC string.

    Story 2.2's schema stores ``date_utc``/``edit_date`` VERBATIM and names 2.4 as
    the converter (its docstring: pass ``dt.astimezone(timezone.utc).isoformat()``).
    ``None`` in → ``None`` out (``edit_date`` is optional). Never stores naive/local
    time.
    """
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()  # type: ignore[attr-defined]


def _map_message(entity_id: int, msg: object) -> schema.Message | None:
    """Map one Telethon ``Message`` → :class:`schema.Message`, or ``None`` to skip.

    AC#4 skip rule: a falsy ``msg.message`` means a service message (``None``) or
    a media-only message (empty text) — return ``None`` so the caller skips it
    (not ingested in v1; the export does NOT abort). Otherwise build the row using
    ``entity_id`` (the canonical ``chat_id``, so it agrees with ``upsert_chat``),
    the RAW text ``msg.message`` (not ``msg.text``, which re-applies parse-mode
    markup), ``msg.sender_id`` (do NOT force-resolve ``msg.sender`` — an extra
    round-trip), ``msg.reply_to_msg_id`` (the reply/thread linkage, ``None`` when
    not a reply), and UTC-ISO timestamps.
    """
    text = getattr(msg, "message", None)
    if not text:
        return None  # AC#4: service / media-only → skip, do not ingest, do not abort.
    return schema.Message(
        chat_id=entity_id,
        message_id=msg.id,
        sender_id=getattr(msg, "sender_id", None),
        sender_name=get_display_name(getattr(msg, "sender", None)) or None,
        text=text,
        date_utc=_to_iso_utc(msg.date),  # type: ignore[arg-type]
        reply_to_msg_id=getattr(msg, "reply_to_msg_id", None),
        edit_date=_to_iso_utc(getattr(msg, "edit_date", None)),
    )


async def _resolve_and_page_one_chat(
    client: object,
    source: object,
    chat: TrackedChat,
    throttle: RequestThrottle,
    conn: object,
    summary: "ExportSummary",
    progress_cb: ProgressCallback,
    now_iso: Callable[[], str],
    *,
    verb: str,
    start_min_id_for: Callable[[int], int],
) -> None:
    """Resolve one Tracked Chat through the throttle, record it, and page it.

    The single-sourced per-chat outer step shared by bulk export (Story 2.4) and
    catch-up (Story 4.1): entity resolution THROUGH the throttle with the FR-3
    reachability skip-and-continue (the ``FloodWaitError``-before-``RPCError``
    ordering is load-bearing — ``FloodWaitError`` subclasses ``RPCError``), then
    ``upsert_chat``, then :func:`_page_chat_history`. Callers differ ONLY in:

    - ``source`` — the paging source (the takeout proxy for bulk, the bare
      ``client`` for catch-up); entity resolution ALWAYS uses ``client``.
    - ``verb`` — the progress wording ("Exporting"/"Catching up").
    - ``start_min_id_for`` — maps ``entity_id`` → the start cursor: ``0`` for bulk
      (re-page all), ``schema.read_last_id(conn, entity_id)`` for catch-up (only the
      offline gap).

    Records the per-chat count in ``summary.exported`` or the skip in
    ``summary.skipped_chats``; never aborts the whole run for one bad chat.
    """
    identity = _chat_identity(chat)
    # Resolve the entity THROUGH the throttle (AC#1 — entity resolution is a
    # network round-trip too). A TrackedChat has exactly one of username/id
    # (Story 2.1 guarantee).
    selector = chat.username if chat.username is not None else chat.id
    try:
        entity = await throttle.run(
            lambda s=selector: client.get_entity(s)  # type: ignore[attr-defined]
        )
    except FloodWaitError:
        # A FLOOD_WAIT that survived the throttle's bounded retries is an
        # account-safety signal, NOT a "bad config entry" — and since
        # FloodWaitError subclasses RPCError it would otherwise be caught by the
        # reachability handler below and silently misreported as "unreachable,
        # check the config". Re-raise so the run stops and the CLI surfaces the
        # flood loudly (AD-10). MUST precede the (ValueError, ChannelPrivateError,
        # RPCError) handler.
        raise
    except (ValueError, ChannelPrivateError, RPCError) as exc:
        # FR-3 reachability skip-and-continue: report + move on, do NOT abort the
        # whole run for the other chats. Name the exception TYPE (never its value —
        # NFR-1).
        reason = type(exc).__name__
        progress_cb(
            f"Skipping {identity}: could not resolve/reach it ({reason}). "
            "Check the config entry; continuing with the other chats."
        )
        summary.skipped_chats.append((identity, reason))
        return

    entity_id = entity.id  # canonical chat_id — agrees with every row.
    progress_cb(f"{verb} {identity} (chat_id={entity_id})…")

    # Record chat metadata (AC#2 + AD-9's later t.me link needs is_public).
    # is_public ⇔ has a public username (linkable via t.me/<username>).
    with db.writing(conn):  # type: ignore[arg-type]
        schema.upsert_chat(  # type: ignore[arg-type]
            conn,
            entity_id,
            title=get_display_name(entity) or getattr(entity, "title", None),
            username=getattr(entity, "username", None),
            is_public=bool(getattr(entity, "username", None)),
        )

    count = await _page_chat_history(
        source, entity, throttle, conn, identity, progress_cb, now_iso,
        start_min_id=start_min_id_for(entity_id),
    )
    summary.exported[entity_id] = count


async def _page_chat_history(
    source: object,
    entity: object,
    throttle: RequestThrottle,
    conn: object,
    identity: str,
    progress: ProgressCallback,
    now_iso: Callable[[], str],
    *,
    start_min_id: int,
) -> int:
    """Page one resolved chat's history through the throttle and ingest it.

    The single-sourced paging engine shared by BOTH bulk export (Story 2.4) and
    catch-up backfill (Story 4.1). The two callers differ ONLY in what they pass:

    - ``source`` — the object exposing ``get_messages``: the **takeout** proxy for
      bulk export, the **bare client** for catch-up (a regular session — takeout is
      bulk-only per ARCHITECTURE-SPINE#Consistency).
    - ``start_min_id`` — where the cursor begins: ``0`` for bulk (re-page full
      history), the chat's persisted ``sync_state.last_id`` for catch-up (fetch only
      what arrived while offline). ``min_id`` is the EXCLUSIVE lower bound.

    Pages **manually** with ``source.get_messages(..., reverse=True, min_id=cursor)``
    — every page is ONE ``throttle.run`` (AC#1: no un-throttled page), oldest-first
    so ``last_id`` grows monotonically and a crash leaves a truthful mark (AC#4/#6).
    Each page is ingested + its max id advanced in ONE ``db.writing`` scope (AD-5),
    so overlap is dedup-safe (AC#3) and the advance is atomic (AC#4) BY CONSTRUCTION.
    Returns the count of text rows ingested for this chat.
    """
    entity_id = entity.id  # type: ignore[attr-defined]
    ingested = 0
    # ``reverse=True`` → oldest-first; ``min_id`` is the EXCLUSIVE lower bound, so
    # advancing it to the last id of each page walks forward without overlap. The
    # start cursor is the caller's high-water mark (0 for bulk, last_id for catch-up).
    cursor = start_min_id
    while True:
        page = await throttle.run(
            lambda c=cursor: source.get_messages(  # type: ignore[attr-defined]
                entity, limit=PAGE_SIZE, reverse=True, min_id=c
            )
        )
        if not page:
            break

        batch: list[schema.Message] = []
        max_id = cursor
        for msg in page:
            if msg.id > max_id:
                max_id = msg.id
            row = _map_message(entity_id, msg)
            if row is not None:
                batch.append(row)

        # Ingest the page's text rows AND advance the high-water mark to the page's
        # max id in the SAME transaction (AD-5): a crash never leaves messages
        # stored with the mark behind, nor the mark ahead of the messages. We
        # advance to ``max_id`` (the highest id SEEN, incl. skipped media) so the
        # cursor always makes progress even on an all-media page — otherwise the
        # loop would re-fetch the same page forever.
        with db.writing(conn):  # type: ignore[arg-type]
            if batch:
                schema.ingest_messages(conn, batch)  # type: ignore[arg-type]
            schema.advance_last_id(conn, entity_id, max_id, updated_at=now_iso())  # type: ignore[arg-type]

        ingested += len(batch)
        cursor = max_id
        progress(f"Exporting {identity}: {ingested} message(s) so far…")

        # A short final page (fewer than a full page) means we have reached the
        # head — stop rather than issue one more empty request.
        if len(page) < PAGE_SIZE:
            break

    progress(f"Finished {identity}: {ingested} message(s) exported.")
    return ingested


async def run_export(
    client: TakeoutClient,
    chats: tuple[TrackedChat, ...],
    throttle: RequestThrottle,
    conn: object,
    *,
    progress_cb: ProgressCallback = _default_progress,
    now_iso: Callable[[], str] | None = None,
) -> ExportSummary:
    """Run the bulk history export over ``chats`` and return a summary.

    Opens a single Telethon **takeout** session (``async with client.takeout(...)``)
    with ``channels``/``chats``/``megagroups``/``users`` enabled and NO media
    (text-only, AC#4), then for each Tracked Chat: resolves the entity **through
    the throttle**, records its metadata via ``upsert_chat``, and pages its
    history **through the throttle**, ingesting each page idempotently and
    advancing ``sync_state.last_id`` atomically (AC#1, #2, #3, #6).

    Reachability skip-and-continue (FR-3): a chat that cannot be resolved/reached
    (``ValueError`` from ``get_entity`` on a bad username/id; ``ChannelPrivateError``
    /``RPCError`` on a private/inaccessible peer) is reported via ``progress_cb`` and
    recorded in :attr:`ExportSummary.skipped_chats`; the run CONTINUES for the
    other chats and does NOT abort.

    :param client: an already-connected, authorized client with
        ``flood_sleep_threshold=0`` set by the caller (so FLOOD_WAIT surfaces to
        the throttle, not silently slept by Telethon — Story 2.3's #1 seam).
    :param chats: the valid Tracked Chats (each with exactly one of
        ``username``/``id``, the Story 2.1 guarantee).
    :param throttle: the single AD-10 throttle every call routes through.
    :param conn: an open writer ``apsw.Connection`` with the schema ensured
        (via :func:`tgkb.common.schema.open_writer_with_schema`).
    :param progress_cb: sink for secret-free progress lines (default: the
        ``tgkb.writer.export`` logger). AC#5 — a long export never looks hung.
    :param now_iso: injectable UTC-now-as-ISO-string source for ``updated_at``
        (tests pin it; default is the real clock).
    :returns: an :class:`ExportSummary` (per-chat counts + skipped chats).
    :raises TakeoutInitDelayError: propagated from the takeout INIT when Telegram
        requires the user to wait/approve before a takeout may start — the CLI
        maps it to an actionable message (Task 4). It is raised on entering the
        ``async with`` block (NOT a per-request call the throttle wraps), so it is
        deliberately not caught here.
    """
    if now_iso is None:
        from datetime import datetime

        def now_iso() -> str:  # noqa: E306 - tiny local default
            return datetime.now(timezone.utc).isoformat()

    summary = ExportSummary()

    # Opening the takeout can raise TakeoutInitDelayError — that is caught by the
    # CLI (Task 4), NOT here (it is not a per-request call the throttle wraps).
    # Default finalize=True: a clean exit commits the takeout as success; an
    # exception preserves takeout_id for resume (AD-5's per-chat last_id already
    # gives re-run resume). No media (files omitted) — text-only MVP (AC#4).
    async with client.takeout(  # type: ignore[attr-defined]
        channels=True, chats=True, megagroups=True, users=True
    ) as takeout:
        for chat in chats:
            # Bulk export pages through the TAKEOUT proxy and re-pages full history
            # from the start (start_min_id=0). Entity resolution still uses the bare
            # client (get_entity is not a takeout call).
            await _resolve_and_page_one_chat(
                client, takeout, chat, throttle, conn, summary, progress_cb, now_iso,
                verb="Exporting", start_min_id_for=lambda _entity_id: 0,
            )

    progress_cb(
        f"Export complete: {summary.total_messages} message(s) across "
        f"{len(summary.exported)} chat(s); {len(summary.skipped_chats)} skipped."
    )
    return summary


async def run_catchup(
    client: "CatchupClient",
    chats: tuple[TrackedChat, ...],
    throttle: RequestThrottle,
    conn: object,
    *,
    progress_cb: ProgressCallback = _default_progress,
    now_iso: Callable[[], str] | None = None,
) -> ExportSummary:
    """Backfill everything missed since the last run, then STOP at the listener seam (Story 4.1).

    The catch-up half of Epic 4's realtime sync and the **twin of** :func:`run_export`:
    same throttle, same idempotent per-page ingest, same atomic ``advance_last_id``.
    The two differences, both deliberate:

    1. **No takeout — a regular session.** ARCHITECTURE-SPINE#Consistency pins bulk
       export to a Telethon **takeout** session and the realtime path (of which
       catch-up is the warm-up) to a regular session + ``events.NewMessage``. So
       catch-up pages history via the **bare** ``client.get_messages`` (still through
       the ONE AD-10 throttle), NEVER ``client.takeout(...)``. Re-initing a takeout
       on every ``sync`` start could hit the takeout init-delay gate; a regular
       session avoids it entirely.
    2. **Seeds the cursor from ``sync_state.last_id`` (AC#1).** Per chat, the start
       cursor is ``schema.read_last_id(conn, entity_id)`` — ``min_id`` is EXCLUSIVE,
       so catch-up fetches strictly messages NEWER than the high-water mark (the
       offline gap). A never-synced chat reads ``0`` → fetch-all (correct).

    Idempotency on overlapping ranges (AC#3) and the atomic, monotonic ``last_id``
    advance (AC#4) are INHERITED from the shared :func:`_page_chat_history` loop —
    catch-up adds no new dedup/atomicity logic. Reachability skip-and-continue (FR-3)
    is inherited from :func:`_resolve_and_page_one_chat` (same FloodWaitError-first
    ordering as export).

    **This function STOPS after catch-up — it does NOT attach the realtime listener.**
    AC#1 explicitly sequences catch-up BEFORE the listener; Story 4.2 attaches
    ``client.on(events.NewMessage(...))`` and ``run_until_disconnected()``. The seam
    is marked below.

    :param client: an already-connected, authorized client with
        ``flood_sleep_threshold=0`` set by the caller (so FLOOD_WAIT surfaces to the
        throttle, not silently slept by Telethon — Story 2.3's #1 seam). Catch-up
        pages directly off this client (no takeout proxy).
    :param chats: the valid Tracked Chats (Story 2.1 guarantee).
    :param throttle: the single AD-10 throttle every call routes through.
    :param conn: an open writer ``apsw.Connection`` with the schema ensured.
    :param progress_cb: sink for secret-free progress lines (default: the
        ``tgkb.writer.export`` logger). FR-8 observability — never a silent hang.
    :param now_iso: injectable UTC-now-as-ISO-string source (tests pin it).
    :returns: an :class:`ExportSummary` (per-chat counts + skipped chats).
    """
    if now_iso is None:
        from datetime import datetime

        def now_iso() -> str:  # noqa: E306 - tiny local default
            return datetime.now(timezone.utc).isoformat()

    summary = ExportSummary()

    for chat in chats:
        # Catch-up pages through the BARE client (regular session — NO takeout) and
        # seeds the cursor from each chat's persisted high-water mark (AC#1).
        await _resolve_and_page_one_chat(
            client, client, chat, throttle, conn, summary, progress_cb, now_iso,
            verb="Catching up",
            start_min_id_for=lambda entity_id: schema.read_last_id(conn, entity_id),  # type: ignore[arg-type]
        )

    progress_cb(
        f"Catch-up complete: {summary.total_messages} new message(s) across "
        f"{len(summary.exported)} chat(s); {len(summary.skipped_chats)} skipped."
    )
    # Story 4.2 lives in `sync.py`: `run_sync` CALLS this function as its warm-up /
    # recover-on-drop step, then attaches the `events.NewMessage`/`MessageEdited`/
    # `MessageDeleted` listener and `run_until_disconnected()`. This function stays
    # BOUNDED (exit-when-done) — the `sync` CLI, `test_catchup.py`, and the reconnect
    # loop all depend on it returning; the "listening now" message lives in run_sync.
    progress_cb("Catch-up done.")
    return summary


__all__ = [
    "CatchupClient",
    "ExportSummary",
    "FloodWaitError",
    "PAGE_SIZE",
    "ProgressCallback",
    "TakeoutClient",
    "TakeoutInitDelayError",
    "run_catchup",
    "run_export",
]
