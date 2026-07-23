"""Tests for the ``export`` CLI wiring (Story 2.4, ``tgkb/writer/cli.py``).

`tests/test_export.py` covers the ``run_export`` engine directly; this file
covers the CLI wiring those unit tests never reach — config loading + per-entry
error reporting, the live-client acquire seam (reuse a stored session), the
exit-code contract (mirroring ``_cmd_auth``), the ``TakeoutInitDelayError`` gate,
and the ``flood_sleep_threshold=0`` seam.

Everything is mocked — **no network, no real ``.tgkb-data/``**. ``run_export`` and
``_acquire_connected_client`` are patched; the config + session paths are
redirected under ``tmp_path``. Coroutines that must run are driven by the real
``_cmd_export`` (which calls ``asyncio.run`` itself).
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from tgkb.common import config, secrets_store
from tgkb.writer import cli, export
from tgkb.writer.auth import AuthError

API_HASH = "HASH_SENTINEL_export"
SESSION_SENTINEL = "SESSION_SENTINEL_export"
PASSPHRASE = "export-test-passphrase"


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


def _fake_run_export(*, summary=None, raises=None, capture_client=None):
    async def _run(client, chats, throttle, conn, **_kw):
        if capture_client is not None:
            capture_client["client"] = client
            capture_client["threshold"] = client.flood_sleep_threshold
        if raises is not None:
            raise raises
        return summary if summary is not None else export.ExportSummary(exported={1: 3})
    return _run


VALID_CONFIG = '{"chats": [{"username": "pub"}]}'


# --- Config-level exit contract -----------------------------------------------


def test_export_missing_config_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """A missing config file → exit 1, actionable stderr, no traceback."""
    _redirect_config(monkeypatch, tmp_path / "does-not-exist.json")
    rc = cli._cmd_export(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "Config file not found" in err


def test_export_empty_chats_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """A file-level 'no chats' error → exit 1 (ConfigError path)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, '{"other": true}'))
    rc = cli._cmd_export(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "chats" in err


def test_export_reports_skipped_entries(monkeypatch, tmp_path, capsys) -> None:
    """Per-entry problems are reported to stderr (skip-and-continue), export still runs."""
    body = '{"chats": [{"username": "pub"}, {"title": "bad"}]}'
    _redirect_config(monkeypatch, _write_config(tmp_path, body))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export())

    rc = cli._cmd_export(argparse.Namespace())

    assert rc == 0
    _, err = capsys.readouterr()
    assert "skipping chat entry #2" in err


# --- Happy path + the flood_sleep_threshold seam ------------------------------


def test_export_success_exit_0_sets_flood_threshold(monkeypatch, tmp_path) -> None:
    """A valid config + faked client → exit 0, and flood_sleep_threshold set to 0 (Story 2.3 seam)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(client))
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export(capture_client=captured))

    rc = cli._cmd_export(argparse.Namespace())

    assert rc == 0
    # The single most important cross-story seam: the export client must have
    # flood_sleep_threshold==0 BY THE TIME run_export runs.
    assert captured["threshold"] == 0
    assert client.disconnected is True  # disconnected in the finally


# --- Error mapping (mirror _cmd_auth) -----------------------------------------


def test_export_auth_error_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """An AuthError from the acquire path → exit 1, actionable message, no traceback."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)

    async def _boom(*_a, **_k):
        raise AuthError("Invalid login code, re-run auth.")

    monkeypatch.setattr(cli, "_acquire_connected_client", _boom)
    rc = cli._cmd_export(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "Invalid login code, re-run auth." in err


def test_export_wrong_passphrase_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """A SecretDecryptError (wrong passphrase) → exit 1, actionable message."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)

    async def _boom(*_a, **_k):
        raise secrets_store.SecretDecryptError("Wrong passphrase or corrupted file.")

    monkeypatch.setattr(cli, "_acquire_connected_client", _boom)
    rc = cli._cmd_export(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "passphrase" in err.lower() or "corrupted" in err.lower()


def test_export_takeout_init_delay_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """Task 4: TakeoutInitDelayError → clean, actionable message with N, no traceback, exit 1."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    err_obj = export.TakeoutInitDelayError(request=None, capture=3600)
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export(raises=err_obj))

    rc = cli._cmd_export(argparse.Namespace())

    assert rc == 1
    _, err = capsys.readouterr()
    assert "3600" in err
    assert "official Telegram client" in err


def test_export_flood_wait_exit_1_actionable(monkeypatch, tmp_path, capsys) -> None:
    """A retry-exhausted FloodWaitError → actionable message with N, not a generic 'Export failed'."""
    from telethon.errors import FloodWaitError

    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export(raises=FloodWaitError(request=None, capture=120)))

    rc = cli._cmd_export(argparse.Namespace())

    assert rc == 1
    _, err = capsys.readouterr()
    assert "120" in err
    assert "FLOOD_WAIT" in err
    assert "Export failed" not in err  # NOT the generic last-resort message


def test_export_client_disconnected_on_db_open_failure(monkeypatch, tmp_path) -> None:
    """The acquired client is disconnected even if opening the writer DB raises (no leak)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(client))
    # Make the DB open blow up AFTER the client is acquired.
    monkeypatch.setattr(cli.schema, "open_writer_with_schema", lambda _p: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    monkeypatch.setattr(cli.db, "default_db_path", lambda: tmp_path / "tgkb.sqlite3")

    rc = cli._cmd_export(argparse.Namespace())

    assert rc == 1  # handled by the last-resort guard
    assert client.disconnected is True  # the live client was NOT leaked


def test_export_keyboard_interrupt_exit_1(monkeypatch, tmp_path, capsys) -> None:
    """Ctrl-C mid-export → clean exit 1, no traceback."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export(raises=KeyboardInterrupt()))

    rc = cli._cmd_export(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "aborted" in err.lower()


def test_export_unexpected_exception_exit_1_no_value(monkeypatch, tmp_path, capsys) -> None:
    """A non-handled failure → exit 1, error TYPE only (never its value, NFR-1)."""
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export(raises=RuntimeError("secret-ish xyz")))

    rc = cli._cmd_export(argparse.Namespace())
    assert rc == 1
    _, err = capsys.readouterr()
    assert "Export failed (RuntimeError)." in err
    assert "secret-ish xyz" not in err


# --- Secret safety ------------------------------------------------------------


def test_export_no_secret_in_output(monkeypatch, tmp_path, capsys, caplog) -> None:
    """No api_hash / session token reaches stdout/stderr/logs on the success path."""
    caplog.set_level(0)
    _redirect_config(monkeypatch, _write_config(tmp_path, VALID_CONFIG))
    _redirect_session_path(monkeypatch, tmp_path)
    _redirect_db(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_acquire_connected_client", _fake_acquire(_FakeClient()))
    monkeypatch.setattr(cli.export, "run_export", _fake_run_export())

    cli._cmd_export(argparse.Namespace())

    out, err = capsys.readouterr()
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    for hay in (out, err, log_text):
        assert API_HASH not in hay
        assert SESSION_SENTINEL not in hay
        assert PASSPHRASE not in hay
