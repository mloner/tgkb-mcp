"""Writer CLI entry point.

Command status:

- ``auth``   — real interactive Telegram authentication with **encrypted session
  persistence + reuse** (Stories 1.3 + 1.4).
- ``export`` — real bulk history export via a takeout session (Story 2.4).
- ``sync``   — **catch-up backfill** of messages missed while offline (Story 4.1)
  **followed by a realtime listener** (Story 4.2): it stays up ingesting live
  messages/edits/deletions and re-runs catch-up on a connection drop, until Ctrl-C.

Uses stdlib ``argparse`` only (no CLI framework is in the pinned Stack).

``auth`` follows a load-or-authenticate decision tree (Story 1.4, AD-8):

1. **A persisted encrypted session exists** → prompt for the passphrase, decrypt
   the ``.tgkb-data/session.enc`` credential bundle, reconstruct the client, and
   — if Telegram still considers it authorized — proceed with **no re-auth**
   (AC#2), reading ``api_id``/``api_hash``/``phone`` from the decrypted payload
   (no env vars needed). A revoked/expired session cleanly falls through to full
   auth (AC#3); a wrong passphrase exits 1 without wiping the file.
2. **No session (or fell through)** → the Story 1.3 flow: resolve
   ``TG_API_ID``/``TG_API_HASH``/``TG_PHONE`` (env + prompt fallback),
   authenticate, then persist the session + ``api_hash`` encrypted under a NEW
   passphrase (double-entry confirmed) for reuse next time (AC#1/#4).

Secret-safe (AC#4 / NFR-1): ``api_hash``, the 2FA password, and the passphrase
are read via ``getpass`` (no echo) and never logged; only ciphertext reaches
disk; the ``telethon`` logger is capped at ``WARNING`` so no verbose MTProto
frame can surface a secret.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import sys

from tgkb.common import VERSION, config
from tgkb.common import db, schema
from tgkb.writer.throttle import DEFAULT_REQUEST_RATE_PER_SECOND, RequestThrottle
from tgkb.common import secrets_store
from tgkb.writer import export, sync
from tgkb.writer.auth import AuthError, authenticate, load_authorized_client

try:
    # Local-testing convenience: auto-load a gitignored .env so `tgkb auth`
    # picks up TG_* creds without manual exports. It is a DEV-only dependency,
    # so guard the import — the CLI must still run when it is absent (e.g. a
    # clean prod install). The real at-rest credential store is Story 1.4.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without the dev extra
    load_dotenv = None

_NOT_IMPLEMENTED = "tgkb {cmd}: not yet implemented (scaffold stub)."


def _stub(command: str) -> int:
    """Print a not-implemented notice and exit cleanly."""
    print(_NOT_IMPLEMENTED.format(cmd=command))
    return 0


def _require_credential(env_var: str, prompt: str, *, secret: bool) -> str:
    """Return a credential from ``env_var`` or an interactive fallback prompt.

    Secrets (``api_hash``, 2FA password) use ``getpass.getpass`` (no echo);
    non-secrets use ``input``. The value itself is never logged or echoed.

    An empty or whitespace-only env var is treated as *unset* (a common CI
    misconfiguration, e.g. ``TG_API_ID=""``) so we fall back to the prompt
    rather than passing a blank value downstream. When there is no interactive
    TTY (non-TTY / piped / CI), the prompt would raise ``EOFError``; convert
    that to an :class:`AuthError` so the CLI exits cleanly instead of dumping a
    traceback.
    """
    value = os.environ.get(env_var)
    if value and value.strip():
        return value
    try:
        return getpass.getpass(prompt) if secret else input(prompt)
    except EOFError:
        raise AuthError(
            f"{env_var} is not set and no interactive terminal is available. "
            f"Set {env_var} in the environment and re-run `tgkb auth`."
        ) from None


def _display_name(me: object) -> str:
    """Extract a non-secret display name from a ``get_me()`` result."""
    return (
        getattr(me, "first_name", None)
        or getattr(me, "username", None)
        or "user"
    )


def _passphrase_cb(*, confirm: bool) -> str:
    """Prompt for the local-session passphrase via ``getpass`` (no echo, NFR-1).

    ``confirm=False`` (UNLOCK): prompt once. ``confirm=True`` (first-run ENCRYPT):
    reject an empty/whitespace-only passphrase (an empty one would encrypt the
    ``api_hash`` + session under a key anyone can reproduce by pressing Enter —
    defeating the AD-8 at-rest protection), then prompt again to confirm and
    require a match so a typo cannot lock the user out of a freshly stored session.
    A rejection/mismatch raises :class:`AuthError` (→ clean exit 1, no traceback).
    The passphrase is a secret and is never logged (NFR-1).
    """
    prompt = "Passphrase (to encrypt/unlock the local session): "
    passphrase = getpass.getpass(prompt)
    if confirm:
        if not passphrase.strip():
            raise AuthError(
                "The passphrase cannot be empty — it protects your session and "
                "api_hash at rest. Please re-run `tgkb auth`."
            )
        again = getpass.getpass("Confirm passphrase: ")
        if passphrase != again:
            raise AuthError("Passphrases did not match. Please re-run `tgkb auth`.")
    return passphrase


async def _reuse_stored_session(session_path, passphrase: str) -> str | None:
    """Try the persisted-session reuse path; return a display name or ``None``.

    Decrypts the credential bundle, reconstructs the client via
    :func:`load_authorized_client`, and — if still authorized — reads the display
    name and disconnects, all inside the caller's single event loop (AC#2).
    Returns ``None`` when the stored session is invalid/revoked so the caller
    falls through to full re-auth (AC#3). A wrong passphrase raises
    :class:`~tgkb.common.secrets_store.SecretDecryptError` (handled by the caller
    → exit 1, file NOT wiped).
    """
    payload = secrets_store.loads_payload(
        secrets_store.read_encrypted(session_path, passphrase)
    )
    client = await load_authorized_client(
        payload["session"], payload["api_id"], payload["api_hash"], payload["phone"]
    )
    if client is None:
        return None  # AC#3: revoked/expired → caller re-authenticates.
    try:
        return _display_name(await client.get_me())
    finally:
        await client.disconnect()


async def _authenticate_and_persist(
    api_id: int,
    api_hash: str,
    phone: str,
    session_path,
    passphrase: str,
    *,
    code_cb,
    password_cb,
) -> str:
    """Run the full 1.3 auth flow, then persist the session encrypted (AC#1/#4).

    All of ``authenticate`` → ``session.save()`` → ``get_me`` → ``disconnect``
    run inside the caller's ONE event loop (the client is bound to it). The saved
    session string + ``api_hash`` + ``api_id`` + ``phone`` are bundled and written
    as ciphertext under ``session_path``; nothing secret is logged.
    """
    client = await authenticate(
        api_id, api_hash, phone, code_cb=code_cb, password_cb=password_cb
    )
    try:
        # session.save() returns the StringSession token — it contains the auth
        # key (a SECRET). It goes straight into the encrypted store, never logged.
        session_str = client.session.save()
        name = _display_name(await client.get_me())
    finally:
        await client.disconnect()

    payload = secrets_store.dumps_payload(
        session=session_str, api_hash=api_hash, api_id=api_id, phone=phone
    )
    secrets_store.write_encrypted(session_path, passphrase, payload)
    return name


def _cmd_auth(_args: argparse.Namespace) -> int:
    """Authenticate with Telegram, reusing an encrypted session when present.

    Load-or-authenticate decision tree (Stories 1.3 + 1.4): reuse a valid stored
    session with no re-auth (AC#2), cleanly re-auth an invalid one (AC#3), or run
    the full first-run flow and persist the session encrypted (AC#1/#4). Returns
    0 on success, 1 on a handled auth/credential/passphrase error.
    """
    # Load a local .env (if present and python-dotenv is installed) so TG_*
    # creds are picked up without manual exports on FIRST run. Does NOT override
    # variables already set in the environment (dotenv default). Once a session
    # is persisted, later runs read creds from the decrypted payload, not env.
    if load_dotenv is not None:
        load_dotenv()

    # Cap Telethon's logger so no verbose MTProto DEBUG frame (which can carry
    # request contents) surfaces a secret (AC#4 defensive backstop).
    logging.getLogger("telethon").setLevel(logging.WARNING)

    def code_cb() -> str:
        return input("Login code: ")

    def password_cb() -> str:
        return getpass.getpass("2FA password: ")

    session_path = secrets_store.default_session_path()

    try:
        name = _load_or_authenticate(
            session_path, code_cb=code_cb, password_cb=password_cb
        )
    except secrets_store.SecretDecryptError as exc:
        # Wrong passphrase (or a corrupt file). Report the actionable message and
        # exit 1 — do NOT silently wipe the stored session (out of scope; a
        # re-key flow is a future story).
        print(str(exc), file=sys.stderr)
        return 1
    except AuthError as exc:
        # Actionable, secret-free message from the flow — no traceback.
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Ctrl-C at a prompt is a normal abort — exit cleanly, no traceback.
        print("Authentication aborted.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — last-resort guard, see below
        # Defensive backstop: anything not converted to a handled error (e.g. a
        # local network fault) must still not dump a traceback that could echo
        # request contents (AC#4). Name the error type, not its value.
        print(f"Authentication failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print(f"Authenticated as {name}.")
    return 0


def _load_or_authenticate(session_path, *, code_cb, password_cb) -> str:
    """Reuse a stored session, else authenticate + persist; return a display name.

    This is the single "get an authorized client, reusing the stored session"
    entry point that ``export``/``sync`` (Stories 2.4 / 4.x) will reuse rather
    than re-implementing session loading. It owns the credential-source decision
    (persisted payload vs. env/prompt) and drives ONE ``asyncio.run`` per branch
    (a connected Telethon client is bound to the loop that created it — never open
    a second loop).

    Raises :class:`~tgkb.common.secrets_store.SecretDecryptError` on a wrong
    passphrase and :class:`~tgkb.writer.auth.AuthError` on an auth/credential
    failure; the caller maps both to exit 1.
    """
    if secrets_store.session_exists(session_path):
        # --- Reuse path: unlock the stored session (AC#2), single passphrase. ---
        passphrase = _passphrase_cb(confirm=False)
        name = asyncio.run(_reuse_stored_session(session_path, passphrase))
        if name is not None:
            return name  # AC#2: valid session, no Telegram re-auth.
        # AC#3: stored session is no longer valid — re-authenticate below, reusing
        # the SAME passphrase to re-encrypt the fresh session (no second prompt).
        print("Stored session is no longer valid; re-authenticating.", file=sys.stderr)
        api_id, api_hash, phone = _resolve_credentials()
        return asyncio.run(
            _authenticate_and_persist(
                api_id, api_hash, phone, session_path, passphrase,
                code_cb=code_cb, password_cb=password_cb,
            )
        )

    # --- First-run path: env/prompt creds, then persist under a NEW passphrase. ---
    api_id, api_hash, phone = _resolve_credentials()
    passphrase = _passphrase_cb(confirm=True)
    return asyncio.run(
        _authenticate_and_persist(
            api_id, api_hash, phone, session_path, passphrase,
            code_cb=code_cb, password_cb=password_cb,
        )
    )


async def _acquire_connected_client(session_path, *, code_cb, password_cb):
    """Return a LIVE, connected, authorized ``TelegramClient`` for export/sync.

    This is the ONE seam the existing auth code does not cover. ``_load_or_authenticate``
    (and its ``_reuse_stored_session`` / ``_authenticate_and_persist`` helpers) all
    **disconnect** the client and return a display name — perfect for ``auth``
    (which only proves login works), useless for ``export`` (which must keep issuing
    takeout calls on a live client). This helper mirrors ``_reuse_stored_session``'s
    decrypt path but does NOT disconnect on the reuse hit —
    :func:`~tgkb.writer.auth.load_authorized_client` already returns the LIVE
    connected client (Story 1.4). On a revoked/expired stored session (``None``) it
    reuses the SAME fall-through as ``_load_or_authenticate`` (re-auth + re-persist,
    reusing the one passphrase) and returns THAT connected client.

    The caller MUST run this inside ONE ``asyncio.run(...)`` and keep using the
    returned client in that SAME loop — a Telethon client is bound to the loop that
    created it; opening a second loop would break it. The caller is responsible for
    disconnecting the returned client when done.

    Raises :class:`~tgkb.common.secrets_store.SecretDecryptError` on a wrong
    passphrase and :class:`~tgkb.writer.auth.AuthError` on an auth/credential
    failure (the caller maps both to exit 1), exactly like ``_load_or_authenticate``.
    """
    if secrets_store.session_exists(session_path):
        # --- Reuse path: unlock the stored session, single passphrase. ---
        passphrase = _passphrase_cb(confirm=False)
        payload = secrets_store.loads_payload(
            secrets_store.read_encrypted(session_path, passphrase)
        )
        client = await load_authorized_client(
            payload["session"], payload["api_id"], payload["api_hash"], payload["phone"]
        )
        if client is not None:
            return client  # Valid stored session — hand back the LIVE client (no disconnect).
        # Stored session no longer valid — re-authenticate below, reusing the SAME
        # passphrase to re-encrypt the fresh session (no second prompt).
        print("Stored session is no longer valid; re-authenticating.", file=sys.stderr)
        api_id, api_hash, phone = _resolve_credentials()
        return await _authenticate_and_persist_live(
            api_id, api_hash, phone, session_path, passphrase,
            code_cb=code_cb, password_cb=password_cb,
        )

    # --- First-run path: env/prompt creds, then persist under a NEW passphrase. ---
    api_id, api_hash, phone = _resolve_credentials()
    passphrase = _passphrase_cb(confirm=True)
    return await _authenticate_and_persist_live(
        api_id, api_hash, phone, session_path, passphrase,
        code_cb=code_cb, password_cb=password_cb,
    )


async def _authenticate_and_persist_live(
    api_id: int,
    api_hash: str,
    phone: str,
    session_path,
    passphrase: str,
    *,
    code_cb,
    password_cb,
):
    """Run the full 1.3 auth flow, persist the session encrypted, return the LIVE client.

    A sibling of :func:`_authenticate_and_persist` that does NOT disconnect: ``export``
    needs the connected client back. Persists the same encrypted bundle (session +
    api_hash + api_id + phone) so a later run reuses it with no env vars, then returns
    the still-connected client. Nothing secret is logged.
    """
    client = await authenticate(
        api_id, api_hash, phone, code_cb=code_cb, password_cb=password_cb
    )
    # session.save() returns the StringSession token (a SECRET) — it goes straight
    # into the encrypted store, never logged. We do NOT disconnect: the caller keeps
    # issuing calls on this client.
    session_str = client.session.save()
    payload = secrets_store.dumps_payload(
        session=session_str, api_hash=api_hash, api_id=api_id, phone=phone
    )
    secrets_store.write_encrypted(session_path, passphrase, payload)
    return client


def _resolve_credentials() -> tuple[int, str, str]:
    """Resolve ``(api_id, api_hash, phone)`` from env vars with a prompt fallback.

    First-run credential source (Story 1.3, unchanged): env + interactive
    fallback, empty/whitespace env treated as unset, non-int ``TG_API_ID`` →
    :class:`AuthError`, no-TTY → :class:`AuthError`. ``api_hash`` is read via
    ``getpass`` (no echo) and never logged.
    """
    raw_api_id = _require_credential("TG_API_ID", "Telegram api_id: ", secret=False)
    try:
        api_id = int(raw_api_id.strip())
    except ValueError:
        raise AuthError("TG_API_ID must be an integer.") from None
    api_hash = _require_credential("TG_API_HASH", "Telegram api_hash: ", secret=True)
    phone = _require_credential("TG_PHONE", "Phone (e.g. +15551234567): ", secret=False)
    return api_id, api_hash, phone


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Run the offline self-check (see :mod:`tgkb.writer.doctor`)."""
    from tgkb.writer.doctor import run

    return run()


def _log_pending_fts_rebuild(db_path) -> None:
    """Log a one-line notice if opening the writer will force an FTS rebuild.

    Story 3.3 (AD-11 / AD-10): a tokenizer-fingerprint mismatch triggers a full
    ``messages_fts`` rebuild inside :func:`schema.open_writer_with_schema`, which
    over a large corpus takes real time. We peek at the stored fingerprint on a
    plain read-only connection FIRST — the seam then repairs it, after which
    nothing is observable — so the user sees a "rebuilding…" line instead of a
    silent hang. Keeping the peek here (not in ``common/``) keeps the seam
    UI-agnostic and the message where the console is owned.

    Best-effort and fully guarded: a fresh/missing DB, or any read hiccup, simply
    logs nothing and lets the export proceed — this is visibility, not a gate.
    """
    try:
        from pathlib import Path

        if not Path(db_path).exists():
            return  # fresh DB — open will stamp, no rebuild to announce
        reader = db.open_reader(db_path)
        try:
            if not schema.fingerprint_matches(reader):
                logging.getLogger("tgkb.writer").info(
                    "Tokenizer fingerprint changed since the search index was "
                    "built — rebuilding the full-text index before export "
                    "(this can take a while on a large history)…"
                )
        finally:
            reader.close()
    except Exception as exc:  # noqa: BLE001 — visibility only; never block the export
        # Broad by design (a peek failure must never block the export), but log at
        # DEBUG so a real regression in the visibility path is diagnosable rather
        # than silently swallowed. The actual repair still happens in the seam.
        logging.getLogger("tgkb.writer").debug(
            "Pending-rebuild peek failed (non-fatal): %r", exc
        )
        return


async def _run_export_flow(session_path, chats, rate_per_second) -> export.ExportSummary:
    """Acquire a LIVE client, run the export, and always disconnect — ONE loop.

    Runs entirely inside a single ``asyncio.run`` (the caller's) because a Telethon
    client is bound to the loop that created it. Sets ``flood_sleep_threshold=0`` on
    the export client — the single most important cross-story seam (Story 2.3's #1
    note): Telethon otherwise silently sleeps sub-60s floods and OUR throttle never
    sees them, bypassing AC#5's "wait is visible" and AD-10's reactive half. Setting
    the public attribute on the live client keeps the setting where it is USED and
    does not touch the 1.3/1.4 auth contract or its tests.

    Opens the writer DB (schema ensured), runs ``export.run_export`` through ONE
    :class:`RequestThrottle`, and disconnects + closes in a ``finally``.
    """
    def code_cb() -> str:
        return input("Login code: ")

    def password_cb() -> str:
        return getpass.getpass("2FA password: ")

    client = await _acquire_connected_client(
        session_path, code_cb=code_cb, password_cb=password_cb
    )
    # Everything after the client is acquired runs under a try/finally so the LIVE
    # MTProto connection is ALWAYS disconnected — including if opening the writer
    # DB raises (a non-local path ValueError, the journal_mode!=wal RuntimeError,
    # or any disk/permission error during CREATE TABLE). Otherwise an authorized
    # session would leak on a DB-open failure.
    conn = None
    try:
        # Story 2.3's #1 seam: floods must surface to OUR throttle, not be slept by
        # Telethon. Public attribute on the live client — see the docstring.
        client.flood_sleep_threshold = 0
        throttle = RequestThrottle(rate_per_second=rate_per_second)
        # Story 3.3 (AD-11 / AD-10 "no silent hang"): a tokenizer-fingerprint
        # mismatch forces a full FTS rebuild, which over a large corpus is not
        # instant. Detect it BEFORE opening (open_writer_with_schema repairs it,
        # after which the fingerprint matches and nothing is observable) so we can
        # surface a one-line before/after message; the seam is the safety net that
        # guarantees the actual repair. common/ stays UI-agnostic — the writer
        # prints. Best-effort: any read hiccup here must not block the export.
        _log_pending_fts_rebuild(db.default_db_path())
        conn = schema.open_writer_with_schema(db.default_db_path())
        return await export.run_export(client, chats, throttle, conn)
    finally:
        if conn is not None:
            conn.close()
        await _disconnect_client(client)


async def _disconnect_client(client) -> None:
    """Disconnect a client, awaiting ``disconnect()`` if Telethon made it async.

    Runs in a ``finally``, so it must never raise and mask the primary exception
    (or a clean result): a disconnect on an already-broken connection is
    swallowed — the socket is being torn down regardless.
    """
    try:
        result = client.disconnect()
        if hasattr(result, "__await__"):
            await result
    except Exception:  # noqa: BLE001 — best-effort cleanup in a finally
        pass


def _cmd_export(_args: argparse.Namespace) -> int:
    """Bulk history export via a takeout session (Story 2.4).

    Loads the Tracked Chats + the throttle rate (reporting per-entry problems to
    stderr — the SAME skip-and-continue as ``config-check``), acquires a LIVE
    authorized client (reusing the encrypted session), and runs the export through
    ONE proactive throttle, ingesting idempotently and advancing each chat's
    ``sync_state.last_id`` (AC#1/#2/#3/#6). Returns 0 on a clean finish, 1 on a
    handled config/auth/takeout/interrupt error — mirroring ``_cmd_auth``'s
    exit-code + secret-free-error contract (never a traceback that could echo
    request contents, NFR-1).
    """
    # First-run export with no stored session needs TG_* creds; a reused session
    # reads creds from the decrypted payload (no env). Mirrors _cmd_auth.
    if load_dotenv is not None:
        load_dotenv()
    # Secret backstop: cap only the telethon logger (leave tgkb.* — including the
    # throttle's FLOOD_WAIT warnings and export progress — visible to the user).
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("tgkb.writer").setLevel(logging.INFO)
    # Route tgkb.* logs to the console if the app has no handler configured, so
    # progress + FLOOD_WAIT waits reach the user (AC#5 — never a silent hang).
    _ensure_console_logging()

    session_path = secrets_store.default_session_path()

    try:
        # Load config ONCE (chats + rate + per-entry errors), mirroring
        # _cmd_config_check — do NOT re-read the file twice.
        parsed = config.load_config()
    except config.ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for err in parsed.errors:
        print(f"skipping chat entry #{err.index}: {err.message}", file=sys.stderr)

    chats = parsed.chats
    if not chats:
        print("No valid tracked chats in the config; nothing to export.", file=sys.stderr)
        return 1

    try:
        asyncio.run(
            _run_export_flow(session_path, chats, parsed.rate_per_second)
        )
    except secrets_store.SecretDecryptError as exc:
        # Wrong passphrase / corrupt session file — actionable, no wipe.
        print(str(exc), file=sys.stderr)
        return 1
    except AuthError as exc:
        # Actionable, secret-free message from the auth flow — no traceback.
        print(str(exc), file=sys.stderr)
        return 1
    except export.TakeoutInitDelayError as exc:
        # Takeout-specific gate: Telegram requires waiting (and/or approving the
        # data export in an official client) before a takeout may start. The wait
        # can be hours/days, so exit-and-instruct rather than sleep silently
        # (matching auth.py's "wait Ns and re-run" for a login FloodWait).
        print(
            f"Telegram requires waiting {exc.seconds}s before a takeout can start. "
            "Approve the data export in an official Telegram client if prompted, "
            "then re-run `tgkb export`.",
            file=sys.stderr,
        )
        return 1
    except export.FloodWaitError as exc:
        # A FLOOD_WAIT that survived the throttle's bounded retries (during entity
        # resolution or history paging) is a loud account-safety signal — surface
        # it actionably rather than as a generic "Export failed". MUST come AFTER
        # the TakeoutInitDelayError handler above (that is a FloodWaitError
        # subclass and needs its own, more specific message).
        print(
            f"Telegram rate-limited the export (FLOOD_WAIT {exc.seconds}s) beyond "
            "the retry budget. Wait for the cooldown, then re-run `tgkb export` "
            "(already-exported messages are skipped on the re-run).",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Export aborted.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — last-resort guard (see _cmd_auth)
        # Never dump a traceback that could echo request contents (NFR-1). Name
        # the error TYPE, not its value.
        print(f"Export failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    return 0


def _ensure_console_logging() -> None:
    """Attach a stderr handler to the ``tgkb`` logger if it has none.

    The library never configures logging (that is the application's job), but the
    ``export`` command IS the application entry point — without a handler, the
    throttle's FLOOD_WAIT warnings and export progress would be silently dropped
    (violating AC#5). Idempotent: does nothing if a handler is already present, so
    tests that install ``caplog`` / their own handlers are unaffected.
    """
    logger = logging.getLogger("tgkb")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


async def _run_sync_flow(session_path, chats, rate_per_second) -> None:
    """Acquire a LIVE client, run catch-up THEN the live listener, and always disconnect.

    The near-twin of :func:`_run_export_flow`, differing ONLY in the engine call
    (:func:`sync.run_sync` vs ``run_export``). Everything else is IDENTICAL and reused
    verbatim: :func:`_acquire_connected_client` (session reuse/re-auth/persist),
    ``flood_sleep_threshold=0`` (Story 2.3's #1 seam — a sub-60s flood during catch-up
    must surface to OUR throttle, not be silently slept by Telethon), the
    :func:`_log_pending_fts_rebuild` peek, :func:`schema.open_writer_with_schema`, the
    single :class:`RequestThrottle`, and the :func:`_disconnect_client` ``finally``.

    :func:`sync.run_sync` runs catch-up (the Story 4.1 warm-up) and THEN attaches the
    realtime listener + reconnect loop (Story 4.2) — it BLOCKS in
    ``run_until_disconnected`` until the connection drops or the user hits Ctrl-C (a
    ``KeyboardInterrupt``, now the NORMAL stop path). The ``finally`` disconnects the
    long-running client cleanly on either exit.

    Runs entirely inside a single ``asyncio.run`` (the caller's) — a Telethon client
    is bound to the loop that created it. The listener uses the bare client (NO
    takeout, a regular session — ARCHITECTURE-SPINE#Consistency).
    """
    def code_cb() -> str:
        return input("Login code: ")

    def password_cb() -> str:
        return getpass.getpass("2FA password: ")

    client = await _acquire_connected_client(
        session_path, code_cb=code_cb, password_cb=password_cb
    )
    # As in _run_export_flow: everything after acquiring the client is under a
    # try/finally so the LIVE MTProto connection is ALWAYS disconnected — including
    # if opening the writer DB raises, and now also on the Ctrl-C/drop that ends the
    # listener — so an authorized session never leaks.
    conn = None
    try:
        # Story 2.3's #1 seam: floods must surface to OUR throttle (see docstring).
        client.flood_sleep_threshold = 0
        throttle = RequestThrottle(rate_per_second=rate_per_second)
        # Story 3.3 (AD-11 / AD-10 "no silent hang"): announce a pending FTS rebuild
        # before opening (best-effort), then the seam performs the actual repair.
        _log_pending_fts_rebuild(db.default_db_path())
        conn = schema.open_writer_with_schema(db.default_db_path())
        await sync.run_sync(client, chats, throttle, conn)
    finally:
        if conn is not None:
            conn.close()
        await _disconnect_client(client)


def _cmd_sync(_args: argparse.Namespace) -> int:
    """Catch-up backfill THEN a live realtime listener (Stories 4.1 + 4.2).

    The near-TWIN of :func:`_cmd_export`: loads the Tracked Chats + throttle rate
    (reporting per-entry problems to stderr — the SAME skip-and-continue), acquires a
    LIVE authorized client (reusing the encrypted session), pages each chat from its
    persisted ``sync_state.last_id`` through ONE proactive throttle (idempotent
    ingest, atomic high-water advance — AC#1/#2/#3/#4), and then ATTACHES the realtime
    listener (Story 4.2): it stays up ingesting live messages/edits/deletions, and
    re-runs catch-up on a connection drop so nothing is lost.

    Because the listener is long-running, ``KeyboardInterrupt`` (Ctrl-C) is now the
    NORMAL stop path — it is caught below as a clean user stop ("Sync aborted.",
    return 1). In normal operation ``sync.run_sync`` loops until Ctrl-C (its
    ``should_continue`` predicate is "run forever"), so the return-0 path is reached
    only if the loop exits on its own — e.g. no Tracked Chat resolves for the
    listener. Returns 1 on a handled config/auth/flood/interrupt error — the SAME
    exit-code + secret-free-error contract as ``_cmd_export``, MINUS the
    ``TakeoutInitDelayError`` gate (the sync path uses NO takeout, so it cannot fire).
    Never dumps a traceback that could echo request contents (NFR-1).
    """
    # First-run with no stored session needs TG_* creds; a reused session reads
    # creds from the decrypted payload (no env). Mirrors _cmd_export.
    if load_dotenv is not None:
        load_dotenv()
    # Secret backstop: cap only the telethon logger; keep tgkb.* (throttle
    # FLOOD_WAIT warnings + catch-up progress) visible to the user. The tgkb.writer
    # level is user-selectable via --log-level (default INFO); DEBUG additionally
    # traces every live event the listener catches (NFR-1 safe: type + chat_id +
    # message_id only, never message text). `getattr` tolerates a bare Namespace
    # (the CLI tests construct one without the flag).
    log_level = getattr(_args, "log_level", "INFO") or "INFO"
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("tgkb.writer").setLevel(getattr(logging, log_level, logging.INFO))
    _ensure_console_logging()

    session_path = secrets_store.default_session_path()

    try:
        parsed = config.load_config()
    except config.ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for err in parsed.errors:
        print(f"skipping chat entry #{err.index}: {err.message}", file=sys.stderr)

    chats = parsed.chats
    if not chats:
        print("No valid tracked chats in the config; nothing to sync.", file=sys.stderr)
        return 1

    try:
        asyncio.run(
            _run_sync_flow(session_path, chats, parsed.rate_per_second)
        )
    except secrets_store.SecretDecryptError as exc:
        # Wrong passphrase / corrupt session file — actionable, no wipe.
        print(str(exc), file=sys.stderr)
        return 1
    except AuthError as exc:
        # Actionable, secret-free message from the auth flow — no traceback.
        print(str(exc), file=sys.stderr)
        return 1
    except export.FloodWaitError as exc:
        # A FLOOD_WAIT that survived the throttle's bounded retries (entity
        # resolution or history paging) is a loud account-safety signal — surface it
        # actionably. NOTE: no TakeoutInitDelayError handler here — catch-up uses no
        # takeout, so that gate cannot fire (unlike _cmd_export).
        print(
            f"Telegram rate-limited the sync (FLOOD_WAIT {exc.seconds}s) beyond the "
            "retry budget. Wait for the cooldown, then re-run `tgkb sync` "
            "(already-ingested messages are skipped on the re-run).",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Sync aborted.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — last-resort guard (see _cmd_auth)
        # Never dump a traceback that could echo request contents (NFR-1). Name the
        # error TYPE, not its value.
        print(f"Sync failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    return 0


def _load_tracked_chats() -> tuple[config.TrackedChat, ...]:
    """Load the Tracked Chats, reporting skipped entries on stderr; return the valid ones.

    This is the single "get the valid Tracked Chats, having already reported any
    skipped entries" entry point. Story 2.4's ``export`` reuses it verbatim as its
    first step rather than re-implementing config loading.

    Behavior:

    - A per-entry problem (AC#2) is printed to ``stderr`` as "skipping chat entry
      #N: …" — VISIBLE but NON-fatal — and the entry is dropped. The valid chats
      are still returned.
    - A file-level :class:`~tgkb.common.config.ConfigError` (missing/malformed
      file, no ``chats`` array — AC#3) is printed to ``stderr`` and re-raised to the
      caller, which maps it to a clean exit 1 with no traceback (mirroring the
      ``_cmd_auth`` error-handling contract).
    """
    parsed = config.load_config()  # may raise ConfigError (file-level, AC#3)
    for err in parsed.errors:
        print(f"skipping chat entry #{err.index}: {err.message}", file=sys.stderr)
    return parsed.chats


def _cmd_config_check(_args: argparse.Namespace) -> int:
    """Load + report the Tracked Chats config so AC#2/#3 behavior is observable now.

    Prints the resolved Tracked Chats to ``stdout`` and any skipped per-entry
    problems to ``stderr``. Returns 0 when the file itself is valid — even if some
    entries were skipped (a partially-valid config is a WARNING, not a failure) —
    and 1 on a file-level :class:`~tgkb.common.config.ConfigError` (missing/malformed
    file), printing its actionable, secret-free message with no traceback.

    Does NOT touch Telegram, SQLite, or export — reachability of each chat is left
    to Story 2.4.
    """
    try:
        parsed = config.load_config()
    except config.ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # Mirror _load_tracked_chats: skipped per-entry problems go to stderr (AC#2).
    for err in parsed.errors:
        print(f"skipping chat entry #{err.index}: {err.message}", file=sys.stderr)

    chats = parsed.chats
    if chats:
        print(f"{len(chats)} tracked chat(s):")
        for chat in chats:
            identity = (
                f"@{chat.username}" if chat.username is not None else f"id={chat.id}"
            )
            title = f"  # {chat.title}" if chat.title else ""
            print(f"  - {identity}{title}")
    else:
        print("No valid tracked chats in the config.")

    # Story 2.3: make the resolved proactive-throttle ceiling observable now
    # (before `export` exists). An absent `[throttle]` (None) resolves to the
    # throttle's own default; an explicit 0/negative means no proactive limit.
    rate = parsed.rate_per_second
    if rate is None:
        print(
            f"Request-rate ceiling: {DEFAULT_REQUEST_RATE_PER_SECOND} req/s "
            "(default; set [throttle] rate_per_second to change)."
        )
    elif rate <= 0:
        print("Request-rate ceiling: disabled (no proactive limit).")
    else:
        print(f"Request-rate ceiling: {rate} req/s.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the writer CLI parser exposing auth / export / sync."""
    parser = argparse.ArgumentParser(
        prog="tgkb",
        description="Telegram knowledge base writer (auth, export, sync).",
    )
    parser.add_argument("--version", action="version", version=f"tgkb {VERSION}")

    subparsers = parser.add_subparsers(
        dest="command", metavar="{auth,export,sync,config-check,doctor}"
    )

    p_auth = subparsers.add_parser(
        "auth", help="Authenticate with Telegram (login code + optional 2FA)."
    )
    p_auth.set_defaults(func=_cmd_auth)

    p_export = subparsers.add_parser("export", help="Bulk history export via a takeout session.")
    p_export.set_defaults(func=_cmd_export)

    p_sync = subparsers.add_parser(
        "sync",
        help="Catch-up backfill + realtime listener (runs until stopped).",
    )
    p_sync.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help=(
            "Console log level for tgkb output (default: INFO). Use DEBUG to trace "
            "every live event the listener catches (type + chat_id + message_id, "
            "never message text) — useful when a message does not appear to ingest."
        ),
    )
    p_sync.set_defaults(func=_cmd_sync)

    p_config_check = subparsers.add_parser(
        "config-check",
        help="Load and report the Tracked Chats config (no network).",
    )
    p_config_check.set_defaults(func=_cmd_config_check)

    p_doctor = subparsers.add_parser(
        "doctor", help="Run offline self-checks (no network)."
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "func", None) is None:
        # No subcommand given: show help and exit cleanly.
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
