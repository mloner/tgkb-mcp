"""Tests for the ``auth`` CLI integration layer (Stories 1.3 + 1.4, ``tgkb/writer/cli.py``).

`tests/test_auth.py` covers ``authenticate`` / ``load_authorized_client`` directly;
this file covers the *CLI wiring* those unit tests never reach — the
load-or-authenticate decision tree, credential resolution, the passphrase prompt,
and the exit-code contract.

Story 1.3 contract still enforced here (first-run path):

- credential resolution: env var vs. prompt fallback, empty/whitespace env var
  treated as unset, non-int ``TG_API_ID`` → exit 1, no-TTY prompt → exit 1;
- exit codes: success → 0, ``AuthError`` → 1, unexpected exception → 1,
  ``KeyboardInterrupt`` → 1;
- AC#4 backstop: the ``telethon`` logger is capped at ``WARNING`` and no
  secret/traceback reaches stdout/stderr.

Story 1.4 additions:

- reuse a valid stored session → exit 0, NO ``send_code_request``, no env read (AC#2);
- an invalid stored session → clean fall-through to full re-auth + re-persist (AC#3);
- a wrong passphrase → exit 1, actionable message, file NOT deleted;
- first run with no session → a ``session.enc`` is written whose bytes contain no
  plaintext secret (AC#1/#4).

Everything is mocked — **no network, no real ``.tgkb-data/``**. ``authenticate``
and ``load_authorized_client`` are replaced with async fakes; ``getpass.getpass``
is monkeypatched to supply the passphrase / 2FA password; the session path is
redirected under ``tmp_path``.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Callable

import pytest

from tgkb.common import secrets_store
from tgkb.writer import cli
from tgkb.writer.auth import AuthError

# Sentinels so a leak assertion is unambiguous (AC#4 at the CLI layer).
API_HASH = "HASH_SENTINEL_cli"
API_ID_STR = "424242"
SESSION_SENTINEL = "SESSION_SENTINEL_cli"
PASSPHRASE = "cli-test-passphrase"


class _Me:
    first_name = "Nikolai"
    username = "nikolai"


class _FakeSession:
    """Stand-in for ``client.session`` whose ``save()`` returns the token string."""

    def __init__(self, token: str = SESSION_SENTINEL) -> None:
        self._token = token

    def save(self) -> str:
        return self._token


def _set_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    """Set/clear the TG_* env vars for a test."""
    for key in ("TG_API_ID", "TG_API_HASH", "TG_PHONE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _redirect_session_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Any:
    """Point ``default_session_path`` at a file under ``tmp_path`` (never real)."""
    path = tmp_path / ".tgkb-data" / secrets_store.SESSION_FILENAME
    monkeypatch.setattr(cli.secrets_store, "default_session_path", lambda: path)
    return path


def _feed_getpass(monkeypatch: pytest.MonkeyPatch, *values: str) -> None:
    """Make ``getpass.getpass`` return ``values`` in order (passphrase, 2FA, ...)."""
    seq = list(values)

    def _fake_getpass(_prompt: str = "") -> str:
        return seq.pop(0) if seq else ""

    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)


def _fake_authenticate(
    *, me: Any = _Me(), raises: BaseException | None = None
) -> Callable[..., Any]:
    """Build an async stand-in for ``authenticate`` (records args on ``.seen``)."""

    class _FakeClient:
        def __init__(self) -> None:
            self.session = _FakeSession()

        async def get_me(self) -> Any:
            return me

        async def disconnect(self) -> None:
            self.disconnected = True

    async def _auth(api_id: int, api_hash: str, phone: str, **_kw: Any) -> Any:
        _auth.seen = {"api_id": api_id, "api_hash": api_hash, "phone": phone}
        if raises is not None:
            raise raises
        return _FakeClient()

    _auth.seen = None  # type: ignore[attr-defined]
    return _auth


def _fake_load(*, valid: bool) -> Callable[..., Any]:
    """Build an async stand-in for ``load_authorized_client``.

    ``valid=True`` returns a connected client (reuse succeeds, AC#2); ``valid=False``
    returns ``None`` (stored session invalid → fall through, AC#3). Records the
    args it saw on ``.seen``.
    """

    class _FakeClient:
        async def get_me(self) -> Any:
            return _Me()

        async def disconnect(self) -> None:
            self.disconnected = True

    async def _load(session: str, api_id: int, api_hash: str, phone: str) -> Any:
        _load.seen = {"session": session, "api_id": api_id, "api_hash": api_hash, "phone": phone}
        return _FakeClient() if valid else None

    _load.seen = None  # type: ignore[attr-defined]
    return _load


def _run_auth(monkeypatch: pytest.MonkeyPatch, fake: Callable[..., Any]) -> int:
    """Patch ``cli.authenticate`` with ``fake`` and invoke ``_cmd_auth``."""
    monkeypatch.setattr(cli, "authenticate", fake)
    return cli._cmd_auth(argparse.Namespace())


# --- Story 1.3 first-run contract (now persists — session path + passphrase) ---


def test_auth_success_from_env_returns_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Env creds + a successful first-run flow → exit 0, display name, session persisted."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)  # first-run: confirm twice
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+15551234567")
    fake = _fake_authenticate()

    rc = _run_auth(monkeypatch, fake)

    assert rc == 0
    assert fake.seen == {"api_id": 424242, "api_hash": API_HASH, "phone": "+15551234567"}
    out, _ = capsys.readouterr()
    assert "Authenticated as Nikolai." in out


def test_auth_non_int_api_id_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-integer TG_API_ID is rejected before any network work (exit 1)."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID="not-an-int", TG_API_HASH=API_HASH, TG_PHONE="+1")
    fake = _fake_authenticate()

    rc = _run_auth(monkeypatch, fake)

    assert rc == 1
    assert fake.seen is None  # authenticate never called
    _, err = capsys.readouterr()
    assert "TG_API_ID must be an integer." in err


def test_auth_empty_env_var_no_tty_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty TG_API_ID (CI misconfig) + no TTY → clean exit 1, not EOFError."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID="", TG_API_HASH=API_HASH, TG_PHONE="+1")
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: (_ for _ in ()).throw(EOFError()))
    fake = _fake_authenticate()

    rc = _run_auth(monkeypatch, fake)

    assert rc == 1
    assert fake.seen is None
    _, err = capsys.readouterr()
    assert "TG_API_ID" in err and "no interactive terminal" in err


def test_auth_error_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An AuthError from the flow → exit 1 with the actionable message."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+1")
    fake = _fake_authenticate(raises=AuthError("Invalid login code, re-run auth."))

    rc = _run_auth(monkeypatch, fake)

    assert rc == 1
    _, err = capsys.readouterr()
    assert "Invalid login code, re-run auth." in err


def test_auth_unexpected_exception_returns_1_no_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-AuthError failure → exit 1, error TYPE only (no value/traceback, AC#4)."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+1")
    fake = _fake_authenticate(raises=RuntimeError("secret-ish detail xyz"))

    rc = _run_auth(monkeypatch, fake)

    assert rc == 1
    _, err = capsys.readouterr()
    assert "Authentication failed (RuntimeError)." in err
    assert "secret-ish detail xyz" not in err  # value never surfaced


def test_auth_keyboard_interrupt_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C at a prompt aborts cleanly (exit 1), no traceback."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+1")
    fake = _fake_authenticate(raises=KeyboardInterrupt())

    rc = _run_auth(monkeypatch, fake)

    assert rc == 1
    _, err = capsys.readouterr()
    assert "aborted" in err.lower()


def test_auth_caps_telethon_logger_to_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """AC#4 backstop: _cmd_auth sets the telethon logger to WARNING (behavior)."""
    _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    logging.getLogger("telethon").setLevel(logging.DEBUG)
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+1")

    _run_auth(monkeypatch, _fake_authenticate())

    assert logging.getLogger("telethon").level == logging.WARNING


# --- Story 1.4: persistence, reuse, invalid-session re-auth, wrong passphrase ---


def test_cmd_auth_first_run_persists(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """AC#1/#4: first run (no session file) writes a session.enc with no plaintext secret."""
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+15551234567")

    assert not session_path.exists()
    rc = _run_auth(monkeypatch, _fake_authenticate())

    assert rc == 0
    assert session_path.is_file()  # session persisted
    raw = session_path.read_bytes()
    # AC#4: on-disk ciphertext contains no plaintext secret sentinel.
    assert API_HASH.encode() not in raw
    assert SESSION_SENTINEL.encode() not in raw
    assert PASSPHRASE.encode() not in raw
    # And it really round-trips back under the passphrase.
    payload = secrets_store.loads_payload(
        secrets_store.read_encrypted(session_path, PASSPHRASE)
    )
    assert payload == {
        "session": SESSION_SENTINEL, "api_hash": API_HASH,
        "api_id": 424242, "phone": "+15551234567",
    }


def test_cmd_auth_reuses_stored_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC#2: a valid stored session → exit 0, NO re-auth, no TG_* env read."""
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    # Pre-write a stored session bundle.
    payload = secrets_store.dumps_payload(
        session=SESSION_SENTINEL, api_hash=API_HASH, api_id=424242, phone="+15551234567"
    )
    secrets_store.write_encrypted(session_path, PASSPHRASE, payload)
    _feed_getpass(monkeypatch, PASSPHRASE)  # unlock: single prompt
    _set_env(monkeypatch)  # NO env vars set at all

    load = _fake_load(valid=True)
    monkeypatch.setattr(cli, "load_authorized_client", load)
    # Prove the reuse path NEVER resolves credentials (AC#2: "no env vars needed"):
    # if _resolve_credentials is reached, fail loudly rather than pass by accident.
    def _boom() -> tuple[int, str, str]:
        raise AssertionError("reuse path must not resolve TG_* credentials (AC#2)")

    monkeypatch.setattr(cli, "_resolve_credentials", _boom)
    # If authenticate is ever called on the reuse path, fail loudly.
    auth_fake = _fake_authenticate()
    rc = _run_auth(monkeypatch, auth_fake)

    assert rc == 0
    assert auth_fake.seen is None  # authenticate NEVER called (no re-auth, AC#2)
    # load got the creds from the decrypted payload, not env.
    assert load.seen == {
        "session": SESSION_SENTINEL, "api_id": 424242,
        "api_hash": API_HASH, "phone": "+15551234567",
    }
    out, _ = capsys.readouterr()
    assert "Authenticated as Nikolai." in out


def test_cmd_auth_invalid_session_reauths(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC#3: an invalid stored session falls through to full auth and re-persists."""
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    payload = secrets_store.dumps_payload(
        session="OLD_DEAD_SESSION", api_hash="OLD_HASH", api_id=1, phone="+1"
    )
    secrets_store.write_encrypted(session_path, PASSPHRASE, payload)
    _feed_getpass(monkeypatch, PASSPHRASE)  # unlock once; re-persist reuses it
    # Fresh creds come from env on the re-auth fall-through.
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+15551234567")

    monkeypatch.setattr(cli, "load_authorized_client", _fake_load(valid=False))
    auth_fake = _fake_authenticate()
    rc = _run_auth(monkeypatch, auth_fake)

    assert rc == 0
    # Fell through to a real authenticate with the ENV creds.
    assert auth_fake.seen == {"api_id": 424242, "api_hash": API_HASH, "phone": "+15551234567"}
    _, err = capsys.readouterr()
    assert "no longer valid" in err
    # The freshly authenticated session was re-persisted (new payload on disk).
    new = secrets_store.loads_payload(
        secrets_store.read_encrypted(session_path, PASSPHRASE)
    )
    assert new["session"] == SESSION_SENTINEL and new["api_hash"] == API_HASH


def test_cmd_auth_wrong_passphrase_exit_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC#3-adjacent: stored session + wrong passphrase → exit 1, file NOT deleted."""
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    payload = secrets_store.dumps_payload(
        session=SESSION_SENTINEL, api_hash=API_HASH, api_id=1, phone="+1"
    )
    secrets_store.write_encrypted(session_path, "the-right-one", payload)
    before = session_path.read_bytes()
    _feed_getpass(monkeypatch, "the-WRONG-one")  # unlock with the wrong passphrase

    # authenticate/load must never be reached on a decrypt failure.
    monkeypatch.setattr(cli, "load_authorized_client", _fake_load(valid=True))
    auth_fake = _fake_authenticate()
    rc = _run_auth(monkeypatch, auth_fake)

    assert rc == 1
    assert auth_fake.seen is None
    _, err = capsys.readouterr()
    assert "passphrase" in err.lower() or "corrupted" in err.lower()
    # File NOT wiped (no auto-reset flow).
    assert session_path.is_file()
    assert session_path.read_bytes() == before


def test_cmd_auth_empty_passphrase_first_run_exit_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """First-run empty/whitespace passphrase is rejected → exit 1, nothing persisted."""
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, "   ")  # whitespace-only, rejected before confirm
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+1")

    rc = _run_auth(monkeypatch, _fake_authenticate())

    assert rc == 1
    _, err = capsys.readouterr()
    assert "cannot be empty" in err
    assert not session_path.exists()  # nothing persisted


def test_cmd_auth_passphrase_mismatch_first_run_exit_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """First-run double-entry mismatch → clean exit 1, no session written."""
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, "one", "two")  # confirm mismatch
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+1")

    rc = _run_auth(monkeypatch, _fake_authenticate())

    assert rc == 1
    _, err = capsys.readouterr()
    assert "did not match" in err
    assert not session_path.exists()  # nothing persisted on mismatch


# --- AC#4: no secret (incl. the passphrase) in CLI output/logs ----------------


def test_auth_no_secret_in_cli_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC#4 at the CLI layer: api_hash / session / PASSPHRASE never reach output or logs."""
    caplog.set_level(0)
    session_path = _redirect_session_path(monkeypatch, tmp_path)
    _feed_getpass(monkeypatch, PASSPHRASE, PASSPHRASE, PASSPHRASE, PASSPHRASE)
    _set_env(monkeypatch, TG_API_ID=API_ID_STR, TG_API_HASH=API_HASH, TG_PHONE="+15551234567")

    # Once success, once a failure, so both output paths are covered. Both are
    # first-run paths: remove the persisted file between runs so the second run
    # does not take the reuse branch (keeping the two output paths independent).
    _run_auth(monkeypatch, _fake_authenticate())
    session_path.unlink(missing_ok=True)
    _run_auth(monkeypatch, _fake_authenticate(raises=AuthError("bad code")))

    out, err = capsys.readouterr()
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    for hay in (out, err, log_text):
        assert API_HASH not in hay
        assert SESSION_SENTINEL not in hay
        assert PASSPHRASE not in hay
