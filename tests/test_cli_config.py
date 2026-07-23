"""Tests for the ``config-check`` CLI wiring (Story 2.1, ``tgkb/writer/cli.py``).

`tests/test_config.py` covers the parser directly; this file covers the CLI seam:
the exit-code contract and the visible-but-non-fatal reporting of skipped entries.

Drives ``cli.main(["config-check"])`` with ``config.default_config_path``
monkeypatched under ``tmp_path`` so no test touches the real ``.tgkb-data/``.
No network, no SQLite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tgkb.common import config
from tgkb.writer import cli


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the default config path under tmp_path; return the (not-yet-written) path."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "default_config_path", lambda: path)
    return path


def test_config_check_valid_returns_0(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path.write_text(
        '{"chats": [{"username": "ok_channel"}, {"id": -1001234567890}]}',
        encoding="utf-8",
    )
    rc = cli.main(["config-check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ok_channel" in out
    assert "-1001234567890" in out


def test_config_check_reports_skipped_entries(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Mixed file: one good entry, one bad (both id + username on the second).
    config_path.write_text(
        '{"chats": [{"username": "good_one"}, {"username": "x", "id": 5}]}',
        encoding="utf-8",
    )
    rc = cli.main(["config-check"])
    captured = capsys.readouterr()

    # A partially-valid config is a WARNING, not a failure → exit 0.
    assert rc == 0
    # The valid chat is still listed on stdout.
    assert "good_one" in captured.out
    # The skipped entry is reported on stderr (AC#2 — visible, non-fatal).
    assert "skipping chat entry #2" in captured.err


def test_config_check_missing_file_exit_1(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # config_path is never written → file-level ConfigError (AC#3).
    rc = cli.main(["config-check"])
    captured = capsys.readouterr()

    assert rc == 1
    # Actionable message on stderr naming the expected path; no traceback.
    assert str(config_path) in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_config_check_malformed_file_exit_1(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path.write_text("{this is not valid json", encoding="utf-8")
    rc = cli.main(["config-check"])
    captured = capsys.readouterr()
    assert rc == 1
    # The message names the offending file (AC#3) and never dumps a traceback.
    assert str(config_path) in captured.err
    assert "Traceback" not in captured.err


def test_load_tracked_chats_helper_returns_valid(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The reusable helper (Story 2.4 entry point) drops bad entries, returns valid.
    config_path.write_text(
        '{"chats": [{"username": "keep"}, {"title": "drop, no identity"}]}',
        encoding="utf-8",
    )
    chats = cli._load_tracked_chats()
    assert chats == (config.TrackedChat(username="keep"),)
    assert "skipping chat entry #2" in capsys.readouterr().err
