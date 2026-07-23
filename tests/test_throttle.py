"""Tests for the proactive request throttle (Story 2.3, ``tgkb/writer/throttle.py``).

No network, **no real sleep** — the clock and the sleep are injected. A
sleep-spy records requested durations WITHOUT actually sleeping, and a fake
monotonic clock is advanced by the spy so the spacing math is exercised
deterministically. Tests assert on the *computed* delays, never on wall-clock
elapsed time (that would be flaky — the 1.1→2.2 review lesson).

Coroutines are driven with ``asyncio.run(...)`` in the test body — matching
``tests/test_auth.py``'s mechanism; ``pytest-asyncio`` is deliberately NOT in the
Stack (AD-7 / minimal deps).

Behavior, not flags (the review lesson): AC#2 asserts real inter-request
spacing; AC#3 asserts the FULL N was slept AND the no-hammer ordering
(observe → sleep N → retry); AC#4 asserts the duration is IN the surfaced
message; the bounded-retry exhaustion and the ``coro_factory``-rebuilt-on-retry
contract are covered explicitly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

import pytest
from telethon.errors import FloodWaitError

from tgkb.writer.throttle import (
    DEFAULT_REQUEST_RATE_PER_SECOND,
    RequestThrottle,
    _MAX_FLOOD_RETRIES,
)


class FakeClock:
    """A fake monotonic clock the sleep-spy advances so spacing math is exercised.

    ``now()`` is the injected ``monotonic``; ``advance(dt)`` moves it forward. It
    is non-decreasing (as a real monotonic clock is).
    """

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class SleepSpy:
    """Records every requested sleep duration WITHOUT sleeping in real time.

    Optionally advances a :class:`FakeClock` by each requested duration so the
    throttle's next spacing computation sees the (simulated) passage of time —
    exactly as a real ``asyncio.sleep`` would. Used as the injected ``sleep``.
    """

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.durations: list[float] = []
        self._clock = clock

    async def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)


def _flood_wait(seconds: int) -> FloodWaitError:
    """Construct a real Telethon FloodWaitError (it needs a ``request`` arg)."""
    return FloodWaitError(request=None, capture=seconds)


# --- AC#2: proactive rate ceiling ---------------------------------------------


def test_rate_ceiling_spaces_requests() -> None:
    """AC#2: with a 2 req/s ceiling (0.5s spacing) N back-to-back requests are
    spaced by >= 1/rate — the first is immediate, each subsequent one waits."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(2.0, monotonic=clock.now, sleep=spy)

    async def _call() -> str:
        return "ok"

    async def _drive() -> None:
        for _ in range(4):
            # The requests are enqueued back-to-back with no wall-clock time
            # passing between them (the clock only advances via the sleep-spy).
            assert await throttle.run(lambda: _call()) == "ok"

    asyncio.run(_drive())

    # First request: no wait (no prior release). Each of the next 3: must wait
    # the FULL 0.5s interval, since the fake clock only advances by the sleeps.
    assert spy.durations == [0.5, 0.5, 0.5]


def test_no_ceiling_is_passthrough() -> None:
    """AC#2 boundary: rate None/0/negative → zero proactive sleeps (pass-through)."""

    async def _call() -> str:
        return "ok"

    for rate in (None, 0, -1.0):
        clock = FakeClock()
        spy = SleepSpy(clock)
        throttle = RequestThrottle(rate, monotonic=clock.now, sleep=spy)

        async def _drive() -> None:
            for _ in range(3):
                assert await throttle.run(lambda: _call()) == "ok"

        asyncio.run(_drive())
        assert spy.durations == [], f"rate={rate!r} should issue no proactive sleeps"


# --- AC#3: FLOOD_WAIT full-N sleep + no retry-hammer + bounded ----------------


def test_flood_wait_sleeps_full_n() -> None:
    """AC#3: a FloodWaitError(N) is caught, the FULL N is slept, then the retry
    runs and returns the success value."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    # No proactive limit so the only recorded sleep is the flood sleep.
    throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy)

    calls: list[int] = []

    async def _call() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise _flood_wait(30)
        return "recovered"

    result = asyncio.run(throttle.run(lambda: _call()))

    assert result == "recovered"
    assert spy.durations == [30]  # slept EXACTLY N
    assert len(calls) == 2  # initial attempt + one retry


def test_flood_wait_no_retry_hammer() -> None:
    """AC#3 core: the retry happens only AFTER the full N is slept — never
    before. Assert the ordering (observe flood) → (sleep N) → (retry)."""
    clock = FakeClock()
    events: list[str] = []

    async def _sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    throttle = RequestThrottle(None, monotonic=clock.now, sleep=_sleep)

    attempts: list[int] = []

    async def _call() -> str:
        attempts.append(1)
        events.append(f"call:{len(attempts)}")
        if len(attempts) == 1:
            raise _flood_wait(45)
        return "ok"

    result = asyncio.run(throttle.run(lambda: _call()))

    assert result == "ok"
    # The 2nd call must come strictly AFTER the sleep(45) — no hammer.
    assert events == ["call:1", "sleep:45", "call:2"]


def test_flood_wait_surfaced(caplog: pytest.LogCaptureFixture) -> None:
    """AC#4: the wait + its duration is surfaced (logged), no silent hang, and
    the message carries the duration but NO secret."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy)

    calls: list[int] = []

    async def _call() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise _flood_wait(77)
        return "ok"

    with caplog.at_level(logging.INFO, logger="tgkb.writer.throttle"):
        asyncio.run(throttle.run(lambda: _call()))

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "77" in log_text  # the duration is surfaced (AC#4)
    # No secret-shaped token in the surfaced message.
    for banned in ("api_hash", "api_id", "session", "password"):
        assert banned not in log_text.lower()


def test_flood_wait_retry_bounded() -> None:
    """AC#3 safety: a call that floods FOREVER eventually re-raises after the cap,
    having slept the full N on every attempt (never loops silently)."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy)

    calls: list[int] = []

    async def _call() -> str:
        calls.append(1)
        raise _flood_wait(10)

    with pytest.raises(FloodWaitError):
        asyncio.run(throttle.run(lambda: _call()))

    # Slept the full N on each of the bounded flood retries.
    assert spy.durations == [10] * _MAX_FLOOD_RETRIES
    # initial attempt + one call per flood retry (each of which re-raised).
    assert len(calls) == _MAX_FLOOD_RETRIES + 1


def test_flood_wait_retry_under_active_ceiling() -> None:
    """AC#2 + AC#3 together: with a REAL ceiling set, a flood recovery sleeps the
    full N and the retry is STILL re-routed through the proactive gate — proving
    the two halves compose. Here N (20s) already exceeds the 0.5s interval, so
    the elapsed flood-sleep satisfies the spacing and the gate adds NO extra
    sleep (never a wait SHORTER than the ceiling requires, never a double-wait
    longer than N)."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(2.0, monotonic=clock.now, sleep=spy)  # 0.5s spacing

    calls: list[int] = []

    async def _call() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise _flood_wait(20)
        return "ok"

    result = asyncio.run(throttle.run(lambda: _call()))

    assert result == "ok"
    assert len(calls) == 2
    # First attempt: no prior release → no proactive sleep; then flood → sleep
    # the FULL 20. On retry the gate sees 20s already elapsed (>= 0.5 interval),
    # so it adds nothing. Total recovery wait is exactly N, not N + interval.
    assert spy.durations == [20]


def test_flood_wait_short_n_still_re_gated() -> None:
    """AC#2 + AC#3 boundary: when the flood N (0.1s) is SHORTER than the ceiling
    interval (0.5s), the retry is additionally spaced by the remaining interval —
    the proactive ceiling is honored even across a short flood recovery."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(2.0, monotonic=clock.now, sleep=spy)  # 0.5s spacing

    calls: list[int] = []

    async def _call() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise _flood_wait(0)  # a 0s flood (advances clock by 0)
        return "ok"

    # Force a tiny non-zero flood by advancing manually is unnecessary: with N=0
    # the retry sees zero elapsed and must wait the full 0.5s interval.
    result = asyncio.run(throttle.run(lambda: _call()))

    assert result == "ok"
    # Slept the full N (0), then the retry is re-gated by the full 0.5s interval.
    assert spy.durations == [0, 0.5]


# --- Contract: coro_factory rebuilt on each retry -----------------------------


def test_coro_factory_rebuilt_on_retry() -> None:
    """The factory is CALLED AGAIN on a flood retry (a fresh awaitable per
    attempt) — proving the "factory, not a live coroutine" contract that
    prevents 'cannot reuse already awaited coroutine'."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy)

    factory_calls: list[int] = []
    attempts: list[int] = []

    async def _call() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise _flood_wait(5)
        return "ok"

    def _factory() -> Awaitable[str]:
        factory_calls.append(1)
        return _call()

    result = asyncio.run(throttle.run(_factory))

    assert result == "ok"
    assert len(factory_calls) == 2  # rebuilt a fresh awaitable for the retry


# --- AC#1/#2: serialize concurrent callers ------------------------------------


def test_throttle_serializes_concurrent_callers() -> None:
    """AC#1/#2: two concurrent run(...) calls under a ceiling do not both pass the
    gate within one interval — the asyncio.Lock + spacing serializes them."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(2.0, monotonic=clock.now, sleep=spy)  # 0.5s spacing

    order: list[str] = []

    def _make(tag: str) -> Callable[[], Awaitable[str]]:
        async def _call() -> str:
            order.append(tag)
            return tag

        return lambda: _call()

    async def _drive() -> None:
        await asyncio.gather(
            throttle.run(_make("a")),
            throttle.run(_make("b")),
        )

    asyncio.run(_drive())

    # Both ran, and exactly ONE proactive spacing sleep of 0.5s was issued (the
    # second caller was gated behind the first — they did not both pass free).
    assert sorted(order) == ["a", "b"]
    assert spy.durations == [0.5]


# --- Non-flood errors propagate immediately -----------------------------------


def test_non_flood_error_propagates_without_retry() -> None:
    """A non-FloodWaitError is NOT retried — it propagates on the first attempt."""
    clock = FakeClock()
    spy = SleepSpy(clock)
    throttle = RequestThrottle(None, monotonic=clock.now, sleep=spy)

    calls: list[int] = []

    async def _call() -> str:
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(throttle.run(lambda: _call()))

    assert len(calls) == 1  # no retry on a non-flood error
    assert spy.durations == []  # and no flood sleep


# --- Default is single-sourced ------------------------------------------------


def test_default_rate_is_single_sourced() -> None:
    """The default ceiling is exactly the module constant (single source)."""
    # A conservative, positive default protects the account when unset.
    assert DEFAULT_REQUEST_RATE_PER_SECOND > 0
    clock = FakeClock()
    spy = SleepSpy(clock)
    # Constructed with NO rate arg → uses DEFAULT_REQUEST_RATE_PER_SECOND.
    throttle = RequestThrottle(monotonic=clock.now, sleep=spy)

    async def _call() -> str:
        return "ok"

    async def _drive() -> None:
        await throttle.run(lambda: _call())
        await throttle.run(lambda: _call())

    asyncio.run(_drive())

    expected_interval = 1.0 / DEFAULT_REQUEST_RATE_PER_SECOND
    assert spy.durations == [expected_interval]
