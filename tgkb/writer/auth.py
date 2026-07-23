"""Interactive first-run Telegram authentication (Story 1.3).

This module owns the single interactive authentication flow: connect over
MTProto, request a login code, sign in with it, and — when the account has 2FA
— prompt for and submit the 2FA password. It returns a **connected, authorized**
Telethon client.

Scope boundaries (do NOT widen here):

- **No session persistence.** The flow uses an in-memory ``StringSession()``;
  nothing lands on disk. Encrypting + reusing the session at rest is Story 1.4
  (AD-8). [Source: epics.md#Story 1.4]
- **No config file / app-data store.** ``api_id``/``api_hash``/``phone`` are
  supplied by the caller (the CLI reads them from env vars with a prompt
  fallback). Config schema is Story 2.1 / 1.4. [Source: ARCHITECTURE-SPINE#Deferred]
- **No SQLite.** This story touches no DB, so this module imports neither
  ``apsw`` nor ``tgkb.common.db`` (AD-1: auth is a ``writer/`` capability; it may
  import ``tgkb.common`` and third-party/stdlib, never ``tgkb.reader``).

Secret-safe by construction (AC#4 / NFR-1): ``api_hash``, the login code, and
the 2FA password are never passed to ``print``/``logging``/exception messages.
Errors name the *field* ("invalid 2FA password"), never the *value*. The
callbacks are injected so the flow is unit-testable without a TTY and so this
module stays free of IO/echo concerns.

Telethon 1.44.0 is async; ``authenticate`` is a coroutine. The CLI bridges it
from the synchronous entry point via ``asyncio.run(...)``.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Union

from telethon import TelegramClient
from telethon.errors import (
    AuthKeyDuplicatedError,
    AuthKeyUnregisteredError,
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    RPCError,
    SessionPasswordNeededError,
    SessionExpiredError,
    SessionRevokedError,
    UnauthorizedError,
)
from telethon.sessions import StringSession

# Telegram-side signals that a stored session is dead and full re-auth is
# required (Story 1.4, AC#3) — these must trigger a clean re-auth, never a crash.
# ``UnauthorizedError`` is the base of the AuthKey*/Session* family; listing it
# last catches any sibling not enumerated above. [Source: research 2026-07-18]
_SESSION_INVALID_ERRORS = (
    AuthKeyUnregisteredError,
    AuthKeyDuplicatedError,
    SessionRevokedError,
    SessionExpiredError,
    UnauthorizedError,
)

# Bounded retries for a wrong code / wrong 2FA password (AC#3): re-prompt a few
# times with a clear message, then give up with a typed error — never loop
# forever, never surface a raw Telethon traceback.
_MAX_ATTEMPTS = 3

# A prompt callback returns the entered value. It may be sync (``input``/
# ``getpass``) or async (a coroutine) — we ``await`` the result only if needed,
# so tests can inject either shape.
PromptCallback = Callable[[], Union[str, Awaitable[str]]]


class AuthError(Exception):
    """Raised when authentication fails after exhausting retries / unrecoverably.

    Carries an actionable, secret-free message. The CLI (Task 4) catches this
    and maps it to a non-zero exit code without printing a traceback.
    """


async def _call(cb: PromptCallback) -> str:
    """Invoke a prompt callback, awaiting it if it returned a coroutine.

    Lets callers inject a plain ``lambda: input(...)`` OR an async callback; the
    flow stays agnostic. The returned value is a secret in the general case
    (2FA password) and is NEVER logged here.
    """
    result = cb()
    if hasattr(result, "__await__"):
        return await result  # type: ignore[return-value]
    return result  # type: ignore[return-value]


async def _sign_in_with_code(
    client: TelegramClient, phone: str, code_cb: PromptCallback
) -> None:
    """Prompt for the login code and sign in, retrying on a wrong code (AC#3).

    Raises :class:`AuthError` after ``_MAX_ATTEMPTS`` wrong codes, on an expired
    code (a fresh ``send_code_request`` is required — the user must re-run
    ``auth``), or on an invalid phone number. Lets
    :class:`SessionPasswordNeededError` propagate so the caller drives the 2FA
    branch.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        code = await _call(code_cb)
        try:
            await client.sign_in(phone=phone, code=code)
            return
        except PhoneCodeInvalidError:
            if attempt >= _MAX_ATTEMPTS:
                raise AuthError(
                    "Invalid login code entered too many times. "
                    "Please re-run `tgkb auth` to request a new code."
                ) from None
            # Field name only — never echo the entered code (AC#4).
            print("Invalid login code — please re-enter.")
        except PhoneCodeExpiredError:
            raise AuthError(
                "The login code has expired. Please re-run `tgkb auth` to "
                "request a fresh code."
            ) from None


async def _sign_in_with_password(
    client: TelegramClient, password_cb: PromptCallback
) -> None:
    """Prompt for the 2FA password and sign in, retrying on a wrong one (AC#2/#3).

    Raises :class:`AuthError` after ``_MAX_ATTEMPTS`` wrong passwords. The
    password value is never logged or echoed (AC#4).
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        password = await _call(password_cb)
        try:
            await client.sign_in(password=password)
            return
        except PasswordHashInvalidError:
            if attempt >= _MAX_ATTEMPTS:
                raise AuthError(
                    "Invalid 2FA password entered too many times. "
                    "Please re-run `tgkb auth`."
                ) from None
            # Field name only — never echo the entered password (AC#4).
            print("Invalid 2FA password — please re-enter.")


async def authenticate(
    api_id: int,
    api_hash: str,
    phone: str,
    *,
    code_cb: PromptCallback,
    password_cb: PromptCallback,
) -> TelegramClient:
    """Drive the interactive Telegram auth flow; return a connected client.

    Constructs ONE ``TelegramClient`` over an in-memory ``StringSession`` (no
    disk write) and holds it for the entire flow — the ``phone_code_hash`` from
    ``send_code_request`` and the auth-key negotiation both live on that single
    client, so re-instantiating mid-flow would raise ``PhoneCodeExpiredError``
    (Telethon issue #4509). Uses ``connect()`` (not ``start()``) to keep control
    of the code/password prompts and the retry loop.

    :param api_id: Telegram API id — MUST be an ``int`` (Telethon requirement).
    :param api_hash: Telegram API hash — a SECRET; never logged.
    :param phone: the account phone number in international format.
    :param code_cb: called to obtain the login code (sync or async).
    :param password_cb: called to obtain the 2FA password (sync or async).
    :returns: the connected, authorized :class:`TelegramClient`.
    :raises AuthError: on any recoverable-by-user failure — bad code/password
        after retries, expired code, invalid phone, bad api_id/api_hash, a
        rate-limit (``FloodWaitError``), or any other Telegram-side
        ``RPCError``. The message is actionable and secret-free, and the
        connection is always closed before this propagates. Non-Telegram
        failures (e.g. a local network error) still propagate as-is after the
        connection is closed.
    """
    # One client / one session for the whole flow (see docstring — issue #4509).
    client = TelegramClient(StringSession(), api_id, api_hash)
    try:
        await _drive_auth(client, phone, code_cb, password_cb)
        return client
    except BaseException:
        # Never leak an open MTProto connection on ANY failure path (including
        # the AuthError we raise ourselves). disconnect() may be async in
        # Telethon; await it if so. The error (already an actionable, secret-free
        # AuthError for every Telegram-side failure) propagates after close.
        await _disconnect(client)
        raise


async def load_authorized_client(
    session_str: str, api_id: int, api_hash: str, phone: str
) -> TelegramClient | None:
    """Reconstruct a client from a stored session; return it only if still valid.

    Story 1.4 reuse path (AC#2/#3): rebuild ``TelegramClient`` from the decrypted
    ``StringSession``, connect, and check validity. Returns the CONNECTED client
    when the session is still authorized (the caller proceeds with **no Telegram
    re-auth** — AC#2). Returns ``None`` when the session is invalid/revoked so the
    caller cleanly falls back to the full :func:`authenticate` flow (AC#3):

    - ``is_user_authorized()`` is ``False`` → session no longer valid; or
    - a Telegram session-invalidation RPC error fires on connect / the check
      (:data:`_SESSION_INVALID_ERRORS`, e.g. ``AuthKeyUnregisteredError``).

    Either way the (possibly half-open) connection is closed before ``None`` is
    returned — a dead stored session must never crash the writer.

    ``phone`` is accepted for signature symmetry with :func:`authenticate` (and so
    a future caller can re-request a code on the SAME identity) — the reuse check
    itself needs only the session. The session string is a SECRET and is never
    logged here.
    """
    try:
        # StringSession(...) VALIDATES the token and raises ValueError on a
        # corrupt/garbage session string. Construct it inside the try so a bad
        # stored session degrades to the AC#3 "invalid session → re-auth" path
        # instead of crashing the reuse flow. (A correctly-passphrased file whose
        # session field is nonetheless malformed is exactly this case.)
        client: TelegramClient | None = TelegramClient(
            StringSession(session_str), api_id, api_hash
        )
    except ValueError:
        # AC#3: unusable stored session token → fall back to full re-auth.
        return None
    try:
        await client.connect()
        if await client.is_user_authorized():
            return client  # AC#2: still valid — hand the live client to the caller.
        # AC#3: connected but not authorized (revoked/expired) → fall back.
        await _disconnect(client)
        return None
    except _SESSION_INVALID_ERRORS:
        # AC#3: Telegram says the session is dead — re-auth, don't crash.
        await _disconnect(client)
        return None
    except BaseException:
        # Any other failure (e.g. a local network fault): never leak the socket.
        await _disconnect(client)
        raise


async def _disconnect(client: TelegramClient) -> None:
    """Close the client, awaiting ``disconnect()`` if Telethon made it async."""
    result = client.disconnect()
    if hasattr(result, "__await__"):
        await result


async def _drive_auth(
    client: TelegramClient,
    phone: str,
    code_cb: PromptCallback,
    password_cb: PromptCallback,
) -> None:
    """Run connect → code → (optional) 2FA, mapping Telegram errors to AuthError.

    Split out from :func:`authenticate` so the single ``except BaseException`` in
    the caller closes the connection on every failure — including the
    :class:`AuthError` cases mapped here — with no error-handling duplication.
    """
    await client.connect()

    # Fresh in-memory flow won't be pre-authorized, but keep the guard —
    # Story 1.4 reuses this coroutine with a decrypted, possibly-authorized
    # session and relies on this early return.
    if await client.is_user_authorized():
        return

    try:
        # send_code_request validates the phone and (with bad credentials) is
        # where PhoneNumberInvalidError / ApiIdInvalidError surface — map them to
        # a friendly AuthError here, at their real raise site.
        await client.send_code_request(phone)  # Telegram sends the code (AC#1).
    except PhoneNumberInvalidError:
        raise AuthError(
            "The phone number was rejected by Telegram. Check TG_PHONE "
            "(international format, e.g. +15551234567) and re-run `tgkb auth`."
        ) from None
    except FloodWaitError as exc:
        raise _flood_wait_error(exc) from None
    except RPCError as exc:
        raise _rpc_error(exc) from None

    try:
        await _sign_in_with_code(client, phone, code_cb)
    except SessionPasswordNeededError:
        # Account has 2FA enabled (AC#2): the code succeeded, now the password
        # is required on the SAME client.
        await _sign_in_with_password(client, password_cb)
    except FloodWaitError as exc:
        raise _flood_wait_error(exc) from None
    except RPCError as exc:
        raise _rpc_error(exc) from None


def _flood_wait_error(exc: FloodWaitError) -> AuthError:
    """Map a rate-limit to an actionable AuthError (``seconds`` is not a secret).

    The AD-10 proactive throttle is Story 2.3; here we only report cleanly
    instead of crashing.
    """
    return AuthError(
        f"Telegram rate-limited the login; wait {exc.seconds}s and re-run "
        "`tgkb auth`."
    )


def _rpc_error(exc: RPCError) -> AuthError:
    """Map any other Telegram-side RPCError to an actionable, secret-free AuthError.

    Covers bad api_id/api_hash, banned/unoccupied phone, empty code/password,
    server errors, etc. Names the error *type*, never its value, so no raw
    Telethon traceback (which could echo request contents) reaches the user
    (AC#4).
    """
    return AuthError(
        f"Telegram rejected the login ({type(exc).__name__}). Check your "
        "TG_API_ID / TG_API_HASH / TG_PHONE and re-run `tgkb auth`."
    )
