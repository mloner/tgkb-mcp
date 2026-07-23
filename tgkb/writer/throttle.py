"""The single proactive throttle for every outbound Telegram request (Story 2.3).

This module owns the account-safety choke-point mandated by **AD-10**: "All
Telegram fetches pass through one proactive throttle in ``writer/`` with a
configurable request-rate ceiling (proactive) *in addition to* honoring
``FLOOD_WAIT_N`` fully (reactive). No code path calls Telegram outside this
throttle." Getting the account flagged or banned (SM-C1 / FR-5) is the highest
product risk; this object is the mechanism that prevents it.

It lives in ``writer/`` — NOT ``common/`` — because throttling guards *outbound
Telegram traffic*, which is exclusively a writer capability: the reader (Epic 4)
never calls Telegram, it only reads SQLite. Putting it in ``common/`` would
wrongly imply the reader participates (AD-1 dependency direction:
``reader/ --> common``, only ``writer/ --> telethon``).

AD-10 has TWO obligations, satisfied here in ONE object:

- **Proactive (the ceiling):** a minimum-interval gate spaces consecutive
  request *starts* by at least ``1/rate`` seconds, so however fast callers
  enqueue work the effective outbound rate stays at or below the configured
  ceiling (AC#2). Reactive FLOOD_WAIT handling alone cannot provide this.
- **Reactive (FLOOD_WAIT):** when Telegram returns ``FLOOD_WAIT_N`` the throttle
  sleeps the FULL ``N`` seconds and then retries, never retry-hammering before
  ``N`` elapses (AC#3). Every such wait is surfaced with its duration so the
  tool never silently hangs (AC#4).

**Single choke-point (AC#1):** :meth:`RequestThrottle.run` is the ONE method
every outbound Telegram request routes through — Story 2.4's ``export`` and Story
4.x's ``sync`` wrap ``client.get_messages`` / ``iter_messages`` / takeout calls
through it verbatim. A second, un-throttled call path silently violates AD-10
even if it looks polite. Note the throttle serializes and spaces request
*starts* (the rate-limiting gate), then releases the lock and awaits the call —
it caps the outbound *rate*, it does NOT force requests to run strictly one at a
time end-to-end. That is the correct account-safety property (rate is what
Telegram flags on), not a full mutex over in-flight requests.

**Cross-story seam — the single most important note (do NOT skip when wiring
2.4/4.x):** Telethon's ``TelegramClient`` has ``flood_sleep_threshold`` (default
**60s**): it AUTO-SLEEPS on any ``FloodWaitError`` whose ``seconds`` is *below*
the threshold and only RAISES above it. With the default, a common ``FLOOD_WAIT``
(e.g. 30s) is silently slept by Telethon and THIS throttle never sees it — so
AC#3/#4 (own the sleep, surface the wait) would be bypassed for the common case.
Therefore Story 2.4 / 4.x MUST construct the export/sync client with
``flood_sleep_threshold=0`` so floods surface as ``FloodWaitError`` to
:meth:`run` instead of being silently slept inside Telethon. This module does not
build that client (2.4/4.x do); it only mandates the setting here so the throttle
is the real FLOOD_WAIT owner.

**Config-agnostic mechanism.** The throttle takes ``rate_per_second`` as a
constructor argument — it never reads ``config.json`` itself (that would couple
``writer/throttle`` → ``common/config`` and break unit isolation). The CLI /
export path reads config and constructs the throttle, mirroring how
:func:`tgkb.writer.auth.authenticate` takes credentials as params rather than
reading env itself.

Secret-safe by construction (NFR-1): the throttle logs only wait *durations*
(seconds are not a secret — :mod:`tgkb.writer.auth` already treats
``FloodWaitError.seconds`` as non-secret); it never logs any request payload,
credential, session, or message content.

Telethon 1.44.0 is async, so :meth:`run` is a coroutine (the writer's
export/sync loops are Telethon coroutines). The clock and the sleep are injected
so tests are deterministic and fast (assert computed delays, never wall-clock).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, TypeVar

from telethon.errors import FloodWaitError

# Conservative built-in ceiling used when config supplies none. Single-sourced
# here (the throttle is the natural owner of its own default) so config imports
# THIS constant rather than hardcoding the number a second time — mirroring the
# project's single-source-constant convention (DB_FILENAME, SESSION_FILENAME,
# CONFIG_FILENAME). A conservative default protects the account even if the user
# never sets a rate.
DEFAULT_REQUEST_RATE_PER_SECOND = 1.0

# Bounded FLOOD_WAIT retries so a pathological repeated flood cannot loop
# forever silently — mirrors auth.py's ``_MAX_ATTEMPTS`` discipline. The full N
# is still slept on EVERY attempt (AC#3); after the cap the FloodWaitError is
# re-raised for the caller to handle.
_MAX_FLOOD_RETRIES = 5

# ``tgkb``-namespaced logger so callers can route/level throttle output (matching
# how cli.py caps the telethon logger). Wait durations are surfaced through this.
_log = logging.getLogger(__name__)

T = TypeVar("T")


class RequestThrottle:
    """The single proactive throttle every outbound Telegram request routes through.

    Enforces AD-10: a proactive minimum-interval rate ceiling PLUS reactive
    full-N ``FLOOD_WAIT`` compliance, in one object with one entry point
    (:meth:`run`).

    :param rate_per_second: the request-rate ceiling. Consecutive request starts
        are spaced by at least ``1/rate_per_second`` seconds. ``None`` or a value
        ``<= 0`` means **no proactive limit** (pass-through) — but FLOOD_WAIT is
        still honored. Defaults to :data:`DEFAULT_REQUEST_RATE_PER_SECOND`.
    :param monotonic: monotonic time source (injected for tests). MUST be
        non-decreasing — never ``time.time()`` (wall-clock can jump backward on
        NTP/DST and corrupt the spacing math). Defaults to :func:`time.monotonic`.
    :param sleep: async sleep (injected for tests). Defaults to
        :func:`asyncio.sleep`.
    :param max_flood_retries: how many times a repeated ``FLOOD_WAIT`` is slept
        + retried before the error is re-raised. Defaults to
        :data:`_MAX_FLOOD_RETRIES`.
    """

    def __init__(
        self,
        rate_per_second: float | None = DEFAULT_REQUEST_RATE_PER_SECOND,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_flood_retries: int = _MAX_FLOOD_RETRIES,
    ) -> None:
        # Normalize "no limit": rate <= 0 or None → min_interval 0 (pass-through).
        if rate_per_second is None or rate_per_second <= 0:
            self._min_interval = 0.0
        else:
            self._min_interval = 1.0 / rate_per_second
        self._monotonic = monotonic
        self._sleep = sleep
        self._max_flood_retries = max_flood_retries
        # Serializes the rate GATE so two concurrent callers cannot both slip
        # through one interval (AC#1/#2). Held across the proactive sleep (in
        # _acquire) so the spacing is computed and applied atomically; released
        # before the actual Telegram call is awaited (the throttle caps the
        # outbound rate, not concurrency of in-flight calls).
        self._lock = asyncio.Lock()
        # Monotonic time the last request was released. ``None`` until the first
        # release, so the first request is never delayed.
        self._last_release: float | None = None

    async def run(self, coro_factory: Callable[[], Awaitable[T]]) -> T:
        """Route ONE Telegram request through the throttle. AD-10.

        Every outbound Telegram request in export/sync MUST go through this — it
        is the single choke-point (AC#1). It gates the request *start* to the
        rate ceiling, it does not hold a lock over the in-flight call (see the
        module docstring). Callers MUST build the client with
        ``flood_sleep_threshold=0`` (module docstring) so floods reach the
        ``FloodWaitError`` handling below. The sequence is: (1) ``await`` the
        proactive rate gate (AC#2), (2) ``await`` the Telegram coroutine, (3) on
        :class:`~telethon.errors.FloodWaitError` surface the wait (AC#4), sleep
        the FULL ``exc.seconds`` via the injected sleep (AC#3), then retry —
        bounded by ``max_flood_retries``, sleeping the full N on every attempt.

        :param coro_factory: a **zero-arg callable that returns a FRESH
            awaitable** each call — e.g.
            ``lambda: client.get_messages(chat, min_id=last_id)``. It is
            re-invoked on a FLOOD_WAIT retry; a pre-created coroutine object
            CANNOT be awaited twice (``RuntimeError: cannot reuse already awaited
            coroutine``), so passing one would break the retry.
        :returns: whatever the awaited Telegram coroutine returns.
        :raises FloodWaitError: if ``FLOOD_WAIT`` recurs past
            ``max_flood_retries`` (the full N was still slept on each attempt).
        """
        # ``+ 1`` so the count is "the initial attempt PLUS max_flood_retries
        # flood retries". Any non-flood exception propagates immediately.
        for attempt in range(self._max_flood_retries + 1):
            await self._acquire()
            try:
                return await coro_factory()
            except FloodWaitError as exc:
                if attempt >= self._max_flood_retries:
                    # Cap reached: sleep the full N once more is NOT done here —
                    # we have already slept on every prior attempt; re-raise so
                    # the caller decides. (Honoring the full N each time is AC#3;
                    # not looping forever is the safety cap.)
                    _log.warning(
                        "FLOOD_WAIT persisted after %d retries (last wait %ds); "
                        "giving up.",
                        self._max_flood_retries,
                        exc.seconds,
                    )
                    raise
                # AC#4: surface the wait + its duration (no silent hang). AC#3:
                # sleep the FULL N, then loop to retry — never retry before N.
                _log.warning(
                    "FLOOD_WAIT: Telegram asked to wait %ds before retrying; "
                    "sleeping the full duration.",
                    exc.seconds,
                )
                await self._sleep(exc.seconds)
        # Unreachable: the loop either returns, re-raises, or continues.
        raise AssertionError("throttle retry loop exited unexpectedly")

    async def _acquire(self) -> None:
        """Block until the proactive rate ceiling permits the next request start.

        Serialized by :attr:`_lock` so concurrent callers cannot both pass the
        gate in one interval. With no limit (``_min_interval == 0``) this is a
        near-no-op that only stamps the release time. Uses the injected monotonic
        clock so the spacing math is deterministic and testable.
        """
        async with self._lock:
            if self._min_interval > 0 and self._last_release is not None:
                elapsed = self._monotonic() - self._last_release
                delay = self._min_interval - elapsed
                if delay > 0:
                    await self._sleep(delay)
            # Stamp the release AFTER any wait, so the next request is spaced from
            # THIS request's actual start, not from when it was enqueued.
            self._last_release = self._monotonic()
