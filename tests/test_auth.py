"""Tests for the interactive Telegram auth flow (Story 1.3, ``tgkb/writer/auth.py``).

Every test mocks Telethon entirely — **no test hits the real Telegram network**
and no real credentials are read. The client is a hand-rolled async fake whose
methods are scripted per test (this keeps the retry/2FA branching explicit and
avoids depending on ``pytest-asyncio``, which is deliberately NOT in the Stack).
Coroutines are driven with ``asyncio.run(...)`` in the test body.

Tests exercise *behavior*, not flags (the 1.1/1.2 review lesson): AC#3 tests
actually raise ``PhoneCodeInvalidError``/``PasswordHashInvalidError`` and assert
the retry happened; AC#4 captures real stdout/stderr AND log records and asserts
the secret sentinels are absent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import pytest
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    SessionPasswordNeededError,
    UnauthorizedError,
)

from tgkb.writer import auth
from tgkb.writer.auth import AuthError, authenticate, load_authorized_client

# Distinctive sentinels so the no-secret-leak assertion is unambiguous (AC#4).
API_HASH = "HASH_SENTINEL_xyz"
CODE = "CODE_SENTINEL"
PW = "PW_SENTINEL"
API_ID = 123456
PHONE = "+15551234567"


class _Me:
    """Stand-in for the object ``get_me()`` returns (name is not a secret)."""

    first_name = "Nikolai"
    username = "nikolai"


class FakeClient:
    """Scriptable async fake of ``telethon.TelegramClient``.

    Each network method is async and records that it was called. Per-test
    behavior is scripted via the ``*_effects`` lists (consumed in order): an
    entry that is a ``BaseException`` is raised, anything else is returned.
    """

    def __init__(self, api_id: int, api_hash: str) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.connected = False
        self.disconnected = False
        self.authorized = False
        self.send_code_calls: list[str] = []
        self.signin_code_calls: list[Optional[str]] = []
        self.signin_password_calls: list[Optional[str]] = []
        self.get_me_called = False
        # Scripted effects (default: everything succeeds).
        self.send_code_effects: list[Any] = [None]
        self.signin_code_effects: list[Any] = [None]
        self.signin_password_effects: list[Any] = [None]

    async def connect(self) -> None:
        self.connected = True

    async def is_user_authorized(self) -> bool:
        return self.authorized

    async def send_code_request(self, phone: str) -> Any:
        self.send_code_calls.append(phone)
        return _next(self.send_code_effects)

    async def sign_in(
        self,
        phone: Optional[str] = None,
        code: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        if password is not None:
            self.signin_password_calls.append(password)
            return _next(self.signin_password_effects)
        self.signin_code_calls.append(code)
        return _next(self.signin_code_effects)

    async def get_me(self) -> _Me:
        self.get_me_called = True
        return _Me()

    async def disconnect(self) -> None:
        self.disconnected = True


def _next(effects: list[Any]) -> Any:
    """Pop the next scripted effect; raise it if it is an exception."""
    effect = effects.pop(0) if effects else None
    if isinstance(effect, BaseException):
        raise effect
    return effect


# --- Telethon error factories (these errors require a `request` arg) ----------


def _code_invalid() -> PhoneCodeInvalidError:
    return PhoneCodeInvalidError(request=None)


def _code_expired() -> PhoneCodeExpiredError:
    return PhoneCodeExpiredError(request=None)


def _pw_invalid() -> PasswordHashInvalidError:
    return PasswordHashInvalidError(request=None)


def _phone_invalid() -> PhoneNumberInvalidError:
    return PhoneNumberInvalidError(request=None)


def _pw_needed() -> SessionPasswordNeededError:
    return SessionPasswordNeededError(request=None)


def _flood_wait(seconds: int = 42) -> FloodWaitError:
    return FloodWaitError(request=None, capture=seconds)


# --- Fixtures / helpers -------------------------------------------------------


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeClient]:
    """Patch ``auth.TelegramClient`` to build a pre-scripted :class:`FakeClient`.

    Usage: call the returned factory with the desired scripted effects *before*
    running ``authenticate``; it patches the constructor and returns the fake so
    the test can both script it and later inspect its call records.
    """

    def _make(
        *,
        authorized: bool = False,
        send_code_effects: Optional[list[Any]] = None,
        signin_code_effects: Optional[list[Any]] = None,
        signin_password_effects: Optional[list[Any]] = None,
    ) -> FakeClient:
        client = FakeClient(API_ID, API_HASH)
        client.authorized = authorized
        if send_code_effects is not None:
            client.send_code_effects = send_code_effects
        if signin_code_effects is not None:
            client.signin_code_effects = signin_code_effects
        if signin_password_effects is not None:
            client.signin_password_effects = signin_password_effects

        def _factory(_session: Any, _api_id: int, _api_hash: str) -> FakeClient:
            return client

        monkeypatch.setattr(auth, "TelegramClient", _factory)
        return client

    return _make


def _counting_cb(returns: str) -> tuple[Callable[[], str], list[int]]:
    """A prompt callback returning ``returns``; the list counts invocations."""
    calls: list[int] = []

    def cb() -> str:
        calls.append(1)
        return returns

    return cb, calls


# --- Tests --------------------------------------------------------------------


def test_auth_happy_path_no_2fa(make_client: Callable[..., FakeClient]) -> None:
    """AC#1: code sign-in succeeds, no 2FA; client returned, password_cb unused."""
    client = make_client()
    code_cb, code_calls = _counting_cb(CODE)
    password_cb, pw_calls = _counting_cb(PW)

    result = asyncio.run(
        authenticate(API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb)
    )

    assert result is client
    assert client.connected is True
    assert client.send_code_calls == [PHONE]
    assert client.signin_code_calls == [CODE]  # signed in once with the code
    assert client.signin_password_calls == []  # no 2FA path
    assert code_calls == [1]  # code_cb invoked exactly once
    assert pw_calls == []  # password_cb NOT invoked
    assert client.disconnected is False  # success => connection stays open


def test_auth_already_authorized_short_circuits(
    make_client: Callable[..., FakeClient],
) -> None:
    """An already-authorized client returns immediately (the Story 1.4 guard)."""
    client = make_client(authorized=True)
    code_cb, code_calls = _counting_cb(CODE)
    password_cb, pw_calls = _counting_cb(PW)

    result = asyncio.run(
        authenticate(API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb)
    )

    assert result is client
    assert client.send_code_calls == []  # no code requested
    assert code_calls == [] and pw_calls == []  # no prompts


def test_auth_2fa_path(make_client: Callable[..., FakeClient]) -> None:
    """AC#2: SessionPasswordNeededError drives the 2FA prompt + password sign-in."""
    client = make_client(
        signin_code_effects=[_pw_needed()],  # code step signals 2FA required
        signin_password_effects=[None],  # password step succeeds
    )
    code_cb, code_calls = _counting_cb(CODE)
    password_cb, pw_calls = _counting_cb(PW)

    result = asyncio.run(
        authenticate(API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb)
    )

    assert result is client
    assert client.signin_code_calls == [CODE]
    assert client.signin_password_calls == [PW]  # signed in with the password
    assert pw_calls == [1]  # password_cb invoked (AC#2)


def test_auth_bad_code_retries_then_succeeds(
    make_client: Callable[..., FakeClient],
) -> None:
    """AC#3: a wrong code re-prompts and the retry succeeds — no crash."""
    client = make_client(signin_code_effects=[_code_invalid(), None])
    code_cb, code_calls = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)

    result = asyncio.run(
        authenticate(API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb)
    )

    assert result is client
    assert len(code_calls) == 2  # code_cb invoked twice (retry)
    assert len(client.signin_code_calls) == 2


def test_auth_bad_code_exhausts_retries(
    make_client: Callable[..., FakeClient],
) -> None:
    """AC#3: every code attempt is invalid => AuthError after bounded retries."""
    client = make_client(
        signin_code_effects=[_code_invalid(), _code_invalid(), _code_invalid()]
    )
    code_cb, code_calls = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)

    with pytest.raises(AuthError):
        asyncio.run(
            authenticate(
                API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb
            )
        )

    assert len(code_calls) == 3  # exactly the bounded number of attempts
    assert client.disconnected is True  # failure closed the connection


def test_auth_expired_code_raises_autherror(
    make_client: Callable[..., FakeClient],
) -> None:
    """AC#3: an expired code raises AuthError (re-run required), not a loop."""
    client = make_client(signin_code_effects=[_code_expired()])
    code_cb, code_calls = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)

    with pytest.raises(AuthError):
        asyncio.run(
            authenticate(
                API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb
            )
        )

    assert len(code_calls) == 1  # no retry loop on an expired code
    assert client.disconnected is True


def test_auth_bad_2fa_password_retries(
    make_client: Callable[..., FakeClient],
) -> None:
    """AC#3: a wrong 2FA password re-prompts and the retry succeeds."""
    client = make_client(
        signin_code_effects=[_pw_needed()],
        signin_password_effects=[_pw_invalid(), None],
    )
    code_cb, _ = _counting_cb(CODE)
    password_cb, pw_calls = _counting_cb(PW)

    result = asyncio.run(
        authenticate(API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb)
    )

    assert result is client
    assert len(pw_calls) == 2  # password_cb invoked twice (retry)
    assert len(client.signin_password_calls) == 2


def test_disconnect_on_failure(make_client: Callable[..., FakeClient]) -> None:
    """A mid-flow failure closes the connection (no leaked MTProto connection)."""
    boom = RuntimeError("network down")
    client = make_client(send_code_effects=[boom])
    code_cb, _ = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)

    with pytest.raises(RuntimeError):
        asyncio.run(
            authenticate(
                API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb
            )
        )

    assert client.disconnected is True  # disconnect() awaited on the failure path


def test_auth_invalid_phone_maps_to_autherror(
    make_client: Callable[..., FakeClient],
) -> None:
    """AC#3: a phone Telegram rejects (raised by send_code_request) → AuthError.

    The handler used to sit around sign_in, where PhoneNumberInvalidError can
    never fire; the fix moves it to the real raise site (send_code_request), so
    a bad phone yields an actionable AuthError, not a raw Telethon traceback.
    """
    client = make_client(send_code_effects=[_phone_invalid()])
    code_cb, _ = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)

    with pytest.raises(AuthError):
        asyncio.run(
            authenticate(
                API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb
            )
        )

    assert client.disconnected is True  # connection closed even on this path


def test_auth_flood_wait_maps_to_autherror(
    make_client: Callable[..., FakeClient],
) -> None:
    """A FloodWaitError (rate limit) becomes an AuthError, not a raw traceback."""
    client = make_client(send_code_effects=[_flood_wait(42)])
    code_cb, _ = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)

    with pytest.raises(AuthError) as excinfo:
        asyncio.run(
            authenticate(
                API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb
            )
        )

    assert "42s" in str(excinfo.value)  # the wait seconds surface (not a secret)
    assert client.disconnected is True


def test_no_secret_in_output(
    make_client: Callable[..., FakeClient],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC#4: api_hash / code / 2FA password never reach stdout, stderr, or logs.

    Drives a full 2FA happy path AND a bad-code+bad-password retry path (so the
    retry messages are exercised too), capturing stdout/stderr and all log
    records, then asserts none of the secret sentinels appear anywhere.
    """
    caplog.set_level(0)  # capture at every level, including telethon DEBUG

    # Path 1: 2FA success with one bad code and one bad password first (exercises
    # the "please re-enter" messages, which must name the field, not the value).
    make_client(
        signin_code_effects=[_code_invalid(), _pw_needed()],
        signin_password_effects=[_pw_invalid(), None],
    )
    code_cb, _ = _counting_cb(CODE)
    password_cb, _ = _counting_cb(PW)
    asyncio.run(
        authenticate(API_ID, API_HASH, PHONE, code_cb=code_cb, password_cb=password_cb)
    )

    # Path 2: exhaust code retries so AuthError's message is emitted too.
    make_client(signin_code_effects=[_code_invalid()] * 3)
    code_cb2, _ = _counting_cb(CODE)
    password_cb2, _ = _counting_cb(PW)
    with pytest.raises(AuthError) as excinfo:
        asyncio.run(
            authenticate(
                API_ID, API_HASH, PHONE, code_cb=code_cb2, password_cb=password_cb2
            )
        )

    out, err = capsys.readouterr()
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    haystacks = [out, err, log_text, str(excinfo.value)]

    for secret in (API_HASH, CODE, PW):
        for hay in haystacks:
            assert secret not in hay, f"secret {secret!r} leaked into output/logs"


# --- Story 1.4: session reuse via load_authorized_client ----------------------

SESSION_STR = "SESSION_SENTINEL_reuse"


class ReuseClient:
    """Async fake for the reconstructed-from-session client (Story 1.4 reuse path).

    Scripts ``connect`` and ``is_user_authorized`` so a test can drive the three
    reuse outcomes: valid session, revoked session (False), or a Telegram
    session-invalid RPC error. Records whether ``send_code_request`` was ever
    called — the reuse path must NEVER request a new login code (AC#2).
    """

    def __init__(
        self,
        *,
        authorized: bool = True,
        connect_effect: BaseException | None = None,
        authorized_effect: BaseException | None = None,
    ) -> None:
        self._authorized = authorized
        self._connect_effect = connect_effect
        self._authorized_effect = authorized_effect
        self.connected = False
        self.disconnected = False
        self.send_code_calls: list[str] = []

    async def connect(self) -> None:
        self.connected = True
        if self._connect_effect is not None:
            raise self._connect_effect

    async def is_user_authorized(self) -> bool:
        if self._authorized_effect is not None:
            raise self._authorized_effect
        return self._authorized

    async def send_code_request(self, phone: str) -> None:  # pragma: no cover
        # Present so an accidental re-auth on the reuse path would be detectable;
        # the assertions below verify it is NEVER reached.
        self.send_code_calls.append(phone)

    async def disconnect(self) -> None:
        self.disconnected = True


def _patch_reuse_client(
    monkeypatch: pytest.MonkeyPatch, client: ReuseClient
) -> None:
    """Patch ``auth.TelegramClient`` (and ``StringSession``) so reconstruction
    yields ``client`` from an arbitrary sentinel session string.

    ``StringSession(<real token>)`` validates its input, so patch it to a
    pass-through — the fake factory ignores the session object anyway.
    """

    def _factory(_session: Any, _api_id: int, _api_hash: str) -> ReuseClient:
        return client

    monkeypatch.setattr(auth, "TelegramClient", _factory)
    monkeypatch.setattr(auth, "StringSession", lambda s=None: s)


def test_load_authorized_client_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC#2: a still-authorized session returns the connected client, NO re-auth."""
    client = ReuseClient(authorized=True)
    _patch_reuse_client(monkeypatch, client)

    result = asyncio.run(
        load_authorized_client(SESSION_STR, API_ID, API_HASH, PHONE)
    )

    assert result is client
    assert client.connected is True
    assert client.disconnected is False  # live client stays open for the caller
    assert client.send_code_calls == []  # NEVER requested a new code (AC#2)


def test_load_authorized_client_invalid_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC#3: an unauthorized (revoked) session returns None and disconnects."""
    client = ReuseClient(authorized=False)
    _patch_reuse_client(monkeypatch, client)

    result = asyncio.run(
        load_authorized_client(SESSION_STR, API_ID, API_HASH, PHONE)
    )

    assert result is None
    assert client.disconnected is True  # cleaned up before falling back
    assert client.send_code_calls == []


def test_load_authorized_client_revoked_session_authkey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC#3: an AuthKeyUnregisteredError on the check → None, no crash."""
    client = ReuseClient(
        authorized_effect=AuthKeyUnregisteredError(request=None)
    )
    _patch_reuse_client(monkeypatch, client)

    result = asyncio.run(
        load_authorized_client(SESSION_STR, API_ID, API_HASH, PHONE)
    )

    assert result is None
    assert client.disconnected is True


def test_load_authorized_client_unauthorized_on_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC#3 variant: a base UnauthorizedError raised on connect → None, no crash."""
    client = ReuseClient(
        connect_effect=UnauthorizedError(request=None, message="unauthorized")
    )
    _patch_reuse_client(monkeypatch, client)

    result = asyncio.run(
        load_authorized_client(SESSION_STR, API_ID, API_HASH, PHONE)
    )

    assert result is None
    assert client.disconnected is True
