"""Tests for the offline self-check (``tgkb doctor``, ``tgkb/writer/doctor.py``).

The doctor is itself a testing convenience, so its tests are light: assert that
on a healthy tree every check passes and ``run()`` exits 0, that the individual
checks are behavioral (the DB check really opens a WAL file and a read-only
reader), and that the CLI routes the ``doctor`` subcommand to it.
"""

from __future__ import annotations

from tgkb.writer import cli, doctor


def test_doctor_run_all_pass(capsys) -> None:
    """On a healthy checkout, run() prints all-OK and returns 0."""
    rc = doctor.run()
    out = capsys.readouterr().out
    assert rc == 0
    assert "All 4 checks passed" in out
    assert "[FAIL]" not in out


def test_doctor_individual_checks_pass() -> None:
    """Each check returns True (behavioral: the DB check opens a real WAL file)."""
    assert doctor._check_version() is True
    assert doctor._check_database() is True
    assert doctor._check_fts() is True
    assert doctor._check_reader_process() is True


def test_doctor_reports_failure_nonzero(monkeypatch, capsys) -> None:
    """A failing check makes run() return 1 and print a partial-pass summary."""
    monkeypatch.setattr(doctor, "_check_database", lambda: False)
    rc = doctor.run()
    out = capsys.readouterr().out
    assert rc == 1
    assert "3/4 checks passed" in out


def test_cli_routes_doctor_subcommand(monkeypatch, capsys) -> None:
    """`tgkb doctor` dispatches to the doctor and propagates its exit code."""
    called = {}

    def _fake_run() -> int:
        called["ran"] = True
        return 0

    monkeypatch.setattr("tgkb.writer.doctor.run", _fake_run)
    rc = cli.main(["doctor"])
    assert rc == 0
    assert called.get("ran") is True
