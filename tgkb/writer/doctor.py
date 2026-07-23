"""Offline self-check for tgkb (``tgkb doctor``).

Runs the quick, network-free smoke checks a human would otherwise paste as
multi-line one-liners, and prints a tidy pass/fail list. It exercises the parts
of the current build that can be verified without talking to Telegram:

- the package version is importable (Story 1.1),
- a real WAL database can be opened, written, and read, and a reader connection
  is genuinely read-only (Story 1.2, AD-2/AD-3),
- the reader (MCP) process module imports and answers in a SEPARATE process —
  checked via a subprocess, never a direct import, so the AD-1 rule "writer
  must not import reader" stays intact.

This is a developer/testing convenience, not a product feature: it introduces
no new dependency and touches no network. For the full behavioral guarantees
(auth retry, no-secret-in-logs, concurrency) run ``uv run pytest``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from tgkb.common import VERSION
from tgkb.common import db, schema


def _ok(label: str) -> None:
    print(f"  [OK]   {label}")


def _fail(label: str, detail: str) -> None:
    print(f"  [FAIL] {label}: {detail}")


def _check_version() -> bool:
    """Story 1.1: the package exposes a version string."""
    try:
        assert VERSION and isinstance(VERSION, str)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the doctor
        _fail("package version", repr(exc))
        return False
    _ok(f"package version importable (tgkb {VERSION})")
    return True


def _check_database() -> bool:
    """Story 1.2: WAL writer round-trips, reader is read-only (AD-2/AD-3)."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doctor.sqlite3"
            writer = db.open_writer(path)
            try:
                journal = writer.execute("PRAGMA journal_mode").fetchone()[0]
                assert journal.lower() == "wal", f"journal_mode={journal!r}"
                with db.writing(writer):
                    writer.execute("CREATE TABLE t(x)")
                    writer.execute("INSERT INTO t VALUES (42)")
            finally:
                writer.close()

            reader = db.open_reader(path)
            try:
                value = reader.execute("SELECT x FROM t").fetchone()[0]
                assert value == 42, f"read back {value!r}"
                # The reader MUST reject a write (read-only guarantee).
                try:
                    reader.execute("INSERT INTO t VALUES (99)")
                except Exception:
                    pass  # expected — reader is read-only
                else:
                    _fail("database", "reader was able to WRITE (read-only broken)")
                    return False
            finally:
                reader.close()
    except Exception as exc:  # noqa: BLE001
        _fail("database (WAL + read-only)", repr(exc))
        return False
    _ok("database: WAL writer round-trip + reader is read-only")
    return True


def _check_fts() -> bool:
    """Story 3.2: the FTS5 index round-trips a Russian inflection (AD-6/AD-12).

    Opens a real writer through ``open_writer_with_schema`` (which registers
    ``ru_lemma`` and creates ``messages_fts`` + triggers), ingests one Russian
    message, and confirms a MATCH for a DIFFERENT inflection of the same word
    finds it — proving index-side + query-side lemmatization agree and the
    triggers maintained the index in the ingest transaction.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doctor-fts.sqlite3"
            conn = schema.open_writer_with_schema(path)
            try:
                with db.writing(conn):
                    schema.ingest_message(
                        conn,
                        schema.Message(
                            chat_id=1,
                            message_id=1,
                            sender_id=None,
                            sender_name=None,
                            text="подписали договоры вчера",
                            date_utc="2026-07-19T14:03:00+00:00",
                            reply_to_msg_id=None,
                            edit_date=None,
                        ),
                    )
                # Query a different inflection than was indexed.
                hits = schema.search_message_rowids(conn, "договор")
                assert len(hits) == 1, f"MATCH returned {hits!r}"
            finally:
                conn.close()
    except Exception as exc:  # noqa: BLE001
        _fail("FTS5 index (Russian MATCH round-trip)", repr(exc))
        return False
    _ok("FTS5 index: Russian inflection indexed + found via MATCH (bm25)")
    return True


def _check_reader_process() -> bool:
    """Story 1.1: the reader (MCP) module imports and serves — in a subprocess.

    Checked out-of-process (never ``import tgkb.reader`` here) so the AD-1
    "writer must not import reader" invariant is not violated by the doctor.
    """
    code = (
        "import tgkb.reader.server as s; "
        "assert s.mcp.name == 'tgkb-reader'; "
        "assert callable(s.server_status); "
        "print('ok')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        _fail("reader (MCP) process", repr(exc))
        return False
    if result.returncode != 0 or result.stdout.strip() != "ok":
        _fail("reader (MCP) process", (result.stderr or result.stdout).strip()[:200])
        return False
    _ok("reader (MCP) module imports and serves (separate process)")
    return True


def run() -> int:
    """Run every offline check; return 0 if all pass, 1 otherwise."""
    print("tgkb doctor - offline self-check\n")
    results = [
        _check_version(),
        _check_database(),
        _check_fts(),
        _check_reader_process(),
    ]
    passed, total = sum(results), len(results)
    print()
    if all(results):
        print(f"All {total} checks passed. For full behavioral tests run: uv run pytest")
        return 0
    print(f"{passed}/{total} checks passed — see [FAIL] lines above.")
    return 1
