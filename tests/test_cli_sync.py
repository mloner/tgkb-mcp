"""Tests for the ``sync`` CLI wiring (Stories 4.1 + 4.2, ``tgkb/writer/cli.py``).

`tests/test_sync.py` covers the ``sync.run_sync`` listener/reconnect engine and
`tests/test_catchup.py` covers ``run_catchup`` directly; this file covers the CLI
wiring — config loading + per-entry error reporting, the live-client acquire seam
(reuse a stored session), the ``flood_sleep_threshold=0`` seam, and the exit-code
contract (mirroring ``_cmd_export``, MINUS the ``TakeoutInitDelayError`` gate —
the sync path uses no takeout). Story 4.2 turned catch-up-then-exit into
catch-up-then-listen, so ``_run_sync_flow`` now calls ``sync.run_sync`` (patched
here) and ``KeyboardInterrupt`` — a Ctrl-C stop of the long-running listener — is
the normal stop path.

Everything is mocked — no network, no real ``.tgkb-data/``, no real
``run_until_disconnected``. ``sync.run_sync`` and ``_acquire_connected_client``
are patched; the config + session + DB paths are redirected under ``tmp_path``.
This mirrors ``tests/test_cli_export.py`` so the commands share the SAME contract.
"""

from __future__ import annotations

import argparse
from typing import Any

from tgkb.common import config, secrets_store
from tgkb.writer import cli, sync
from tgkb.writer.auth import AuthError

API_HASH = "HASH_SENTINEL_sync"
SESSION_SENTINEL = "SESSION_SENTINEL_sync"
PASSPHRASE = "sync-test-passphrase"


def _write_config(tmp_path, body: str) -> Any:
    path = tmp_path / "config.json"
    path.write_text(body, encoding="utf-8")
    return path


def _redirect_config(monkeypatch, path) -> None:
    _orig = config.load_config
    monkeypatch.setattr(cli.config, "load_config", lambda: _orig(path))


def _redirect_session_path(monkeypatch, tmp_path) -> Any:
    path = tmp_path / ".tgkb-data" / secrets_store.SESSION_FILENAME
    monkeypatch.setattr(cli.secrets_store, "default_session_path", lambda: path)
    return path


def _redirect_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli.db, "default_db_path", lambda: tmp_path / "tgkb.sqlite3")


class _FakeClient:
    """A minimal client the acquire-seam hands back; only needs disconnect + attr."""

    def __init__(self) -> None:
        self.flood_sleep_threshold: int | None = None
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


def _fake_acquire(client: _FakeClient):
    async def _acquire(session_path, *, code_cb, password_cb):
        return client
    return _acquire


def _fake_run_sync(*, raises=None, capture_client=None):
    """Fake for ``sync.run_sync``: records the client + threshold, optionally raises.

    ``run_sync`` returns ``None`` (it is the long-running listener entry point, not a
    bounded pass), so the fake returns ``None`` on the happy path — the CLI's exit
    code comes from a clean return, not a summary value.
    """
    async def _run(client, chats, throttle, conn, **_kw):
        if capture_client is not None:
            capture_client["client"] = client
            capture_client["threshold"] = client.flood_sleep_threshold
        if raises is not None:
            raise raises
        return None
    return _run


VALID_CONFIG = '{"chats": [{"username": "pub"}]}'


# --- Config-level exit contract -----------------------------------------------


def test_sync_missing_config_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """A missing config file → exit 1, actionable stderr, no traceback."""
    _redirect_config(monkeypatch, tmp_path / "does-not-exist.json")
    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "Config file not found" in err


def test_sync_empty_chats_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """A file-level 'no chats' error → exit 1 (ConfigError path)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, '{"other": true}'))
    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "chats" in err


def test_sync_all_invalid_entries_nothing_to_sync(monkeypatch, tmp_path, capsys) -> None:
    """A file with only invalid entries → exit 1 'nothing to sync', per-entry stderr."""
    body = '{"chats": [{"title": "bad"}]}'
    _redirect_config(monkeypatch, _write_config(tmp_path, body))
    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "skipping chat entry #1" in err
    assert "nothing to sync" in err


def test_sync_reports_skipped_entries(monkeypatch, tmp_path, capsys) -> None:
    """Per-entry problems are reported to stderr (skip-and-continue); sync still runs."""
    body = '{"chats": [{"username": "pub"}, {"title": "bad"}]}'
    _redirect_config(monkeypatch, _write_config(tmp_path, body))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync())

    rc = cli._cmd_sync(argparse.Namespace())

    assert rc == 0
    _, err = capsys.readouterr()
    assert "skipping chat entry #2" in err


# --- Happy path + the flood_sleep_threshold seam ------------------------------


def test_sync_success_exit_0_sets_flood_threshold(monkeypatch, tmp_path) -> None:
    """A valid config + faked client → exit 0, and flood_sleep_threshold set to 0."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(client))
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync(capture_client=captured))

    rc = cli._cmd_sync(argparse.Namespace())

    assert rc == 0
    # The Story 2.3 #1 seam: the client must have flood_sleep_threshold==0 by the
    # time run_catchup runs (a sub-60s flood surfaces to OUR throttle).
    assert captured["threshold"] == 0
    assert client.disconnected is True  # disconnected in the finally


# --- --log-level flag ---------------------------------------------------------


def test_sync_log_level_debug_sets_writer_logger(monkeypatch, tmp_path) -> None:
    """`--log-level DEBUG` raises the tgkb.writer logger to DEBUG (event tracing on)."""
    import logging

    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync())

    rc = cli._cmd_sync(argparse.Namespace(log_level="DEBUG"))

    assert rc == 0
    assert logging.getLogger("tgkb.writer").level == logging.DEBUG


def test_sync_bare_namespace_defaults_info(monkeypatch, tmp_path) -> None:
    """A bare Namespace (no --log-level) defaults tgkb.writer to INFO, not an error."""
    import logging

    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync())

    rc = cli._cmd_sync(argparse.Namespace())  # no log_level attr

    assert rc == 0
    assert logging.getLogger("tgkb.writer").level == logging.INFO


# --- Error mapping (mirror _cmd_export, minus the takeout gate) ----------------


def test_sync_auth_error_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """An AuthError from the acquire path → exit 1, actionable message, no traceback."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)

    async def _boom(*_a, **_k):
        raise AuthError("Invalid login code, re-run auth.")

    monkeypatch.setattr(cli, "_acquire_connected_client", _boom)
    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "Invalid login code, re-run auth." in err


def test_sync_wrong_passphrase_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """A SecretDecryptError (wrong passphrase) → exit 1, actionable message."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)

    async def _boom(*_a, **_k):
        raise secrets_store.SecretDecryptError("Wrong passphrase or corrupted file.")

    monkeypatch.setattr(cli, "_acquire_connected_client", _boom)
    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "passphrase" in err.lower() or "corrupted" in err.lower()


def test_sync_flood_wait_exit_1_actionable(monkeypatch, tmp_path, capsys) -> None:
    """A retry-exhausted FloodWaitError → actionable message with N, not generic 'Sync failed'."""
    from telethon.errors import FloodWaitError

    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync(raises=FloodWaitError(request=None, capture=120)))

    rc = cli._cmd_sync(argparse.Namespace())

    assert rc == 1
    _, err = capsys.readouterr()
    assert "120" in err
    assert "FLOOD_WAIT" in err
    assert "Sync failed" not in err  # NOT the generic last-resort message


def test_sync_keyboard_interrupt_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """Ctrl-C mid-sync → clean exit 1, no traceback."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync(raises=KeyboardInterrupt()))

    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "aborted" in err.lower()


def test_sync_unexpected_exception_exit_1_no_value(monkeypatch, tmp_path, capsys) -> None:
    """A non-handled failure → exit 1, error TYPE only (never its value, NFR-1)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync(raises=RuntimeError("secret-ish xyz")))

    rc = cli._cmd_sync(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "Sync failed (RuntimeError)." in err
    assert "secret-ish xyz" not in err


def test_sync_client_disconnected_on_db_open_failure(monkeypatch, tmp_path) -> None:
    """The acquired client is disconnected even if opening the writer DB raises (no leak)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(client))
    monkeypatch.setattr(cli.schema, "open_writer_with_schema", lambda _p: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    monkeypatch.setattr(cli.db, "default_db_path", lambda: tmp_path / "tgkb.sqlite3")

    rc = cli._cmd_sync(argparse.Namespace())

    assert rc == 1
    assert client.disconnected is True  # the live client was NOT leaked


# --- Secret safety ------------------------------------------------------------


def test_sync_no_secret_in_output(monkeypatch, tmp_path, capsys, caplog) -> None:
    """No api_hash / session token / passphrase reaches stdout/stderr/logs on success."""
    caplog.set_level(0)
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.sync, "run_sync", _fake_run_sync())

    cli._cmd_sync(argparse.Namespace())

    out, err = capsys.readouterr()
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    for hay in (out, err, log_text):
        assert API_HASH not in hay
        assert SESSION_SENTINEL not in hay
        assert PASSPHRASE not in hay
