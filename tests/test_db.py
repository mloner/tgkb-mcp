"""Tests for the SQLite access layer (Story 1.2, ``tgkb/common/db.py``).

Every test uses pytest's ``tmp_path`` fixture so no test ever touches the real
``.tgkb-data/`` directory. Tests exercise *behavior*, not flags: the read-only
test actually attempts a write and asserts the raise; the concurrency test
actually opens two connections against one file (per the Story 1.1 review
lesson recorded in Dev Notes).
"""

from __future__ import annotations

from pathlib import Path

import apsw
import pytest

from tgkb.common import db
from tgkb.common.models import APP_DATA_DIR_NAME


def test_default_db_path_under_app_data_dir() -> None:
    """The DB path lives under the app-data dir with the fixed filename."""
    p = db.default_db_path()
    assert p.parent.name == APP_DATA_DIR_NAME
    assert p.name == db.DB_FILENAME == "tgkb.sqlite3"


def test_writer_pragmas(tmp_path: Path) -> None:
    """AC#1: a writer connection is WAL, busy_timeout=5000, synchronous=NORMAL."""
    conn = db.open_writer(tmp_path / "tgkb.sqlite3")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        # synchronous=NORMAL is reported as the integer 1.
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        conn.close()


def test_writer_creates_parent_dir(tmp_path: Path) -> None:
    """AC#1: open_writer creates a missing app-data parent dir."""
    nested = tmp_path / "made" / "by" / "writer" / "tgkb.sqlite3"
    assert not nested.parent.exists()
    conn = db.open_writer(nested)
    try:
        assert nested.parent.is_dir()
        assert nested.exists()
    finally:
        conn.close()


def test_writer_begin_immediate_commits_and_rolls_back(tmp_path: Path) -> None:
    """AC#1: writing() commits on clean exit and rolls back on exception."""
    conn = db.open_writer(tmp_path / "tgkb.sqlite3")
    try:
        conn.execute("CREATE TABLE t(x)")

        # Clean exit -> committed.
        with db.writing(conn):
            conn.execute("INSERT INTO t VALUES (1)")
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1

        # Exception inside the block -> rolled back.
        with pytest.raises(RuntimeError):
            with db.writing(conn):
                conn.execute("INSERT INTO t VALUES (2)")
                raise RuntimeError("boom")
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1
    finally:
        conn.close()


def test_writing_takes_write_lock_upfront(tmp_path: Path) -> None:
    """AC#1/AD-3: writing() uses BEGIN IMMEDIATE (write lock taken at enter).

    With the write lock already held, a second connection's BEGIN IMMEDIATE
    must fail fast (busy_timeout=0 on the contender) — proving the lock was
    acquired upfront, not lazily.
    """
    path = tmp_path / "tgkb.sqlite3"
    a = db.open_writer(path)
    b = db.open_writer(path)
    try:
        with db.writing(a):
            a.execute("CREATE TABLE t(x)")
            b.set_busy_timeout(0)  # do not wait, so contention surfaces immediately
            with pytest.raises(apsw.BusyError):
                b.execute("BEGIN IMMEDIATE")
    finally:
        a.close()
        b.close()


def test_reader_is_readonly(tmp_path: Path) -> None:
    """AC#2: a reader rejects writes/DDL at the SQL level but allows SELECT."""
    path = tmp_path / "tgkb.sqlite3"
    w = db.open_writer(path)
    w.execute("CREATE TABLE t(x)")
    with db.writing(w):
        w.execute("INSERT INTO t VALUES (42)")
    w.close()

    r = db.open_reader(path)
    try:
        # SELECT works.
        assert r.execute("SELECT x FROM t").fetchone()[0] == 42
        # Writes and DDL are rejected at the SQL level.
        for stmt in ("INSERT INTO t VALUES (1)", "UPDATE t SET x=0", "CREATE TABLE u(y)"):
            with pytest.raises(apsw.ReadOnlyError):
                r.execute(stmt)
    finally:
        r.close()


def test_reader_does_not_create_db(tmp_path: Path) -> None:
    """AC#2/AC#5: open_reader on a missing path raises and creates no file."""
    missing = tmp_path / "does-not-exist.sqlite3"
    with pytest.raises(apsw.Error):
        db.open_reader(missing)
    assert not missing.exists()
    # No stray WAL/SHM sidecars either.
    assert list(tmp_path.iterdir()) == []


def test_concurrent_writer_commit_reader_read(tmp_path: Path) -> None:
    """AC#3: a writer commit and a short reader read do not raise SQLITE_BUSY.

    A single-process, sequential open of both connections against one file is
    sufficient to prove the PRAGMAs (WAL + busy_timeout) are wired; a full
    multi-thread race is not required.
    """
    path = tmp_path / "tgkb.sqlite3"
    w = db.open_writer(path)
    r = db.open_reader(path)
    try:
        w.execute("CREATE TABLE t(x)")
        # Writer commits a short transaction.
        with db.writing(w):
            w.execute("INSERT INTO t VALUES (7)")
        # Reader reads inside a short read transaction — no lock error, row visible.
        with db.reading(r):
            assert r.execute("SELECT x FROM t").fetchone()[0] == 7
    finally:
        w.close()
        r.close()


def test_reading_is_short_transaction(tmp_path: Path) -> None:
    """AC#3: reading() opens and closes a transaction (no perpetual open txn)."""
    path = tmp_path / "tgkb.sqlite3"
    w = db.open_writer(path)
    w.execute("CREATE TABLE t(x)")
    w.close()

    r = db.open_reader(path)
    try:
        assert r.in_transaction is False
        with db.reading(r):
            assert r.in_transaction is True
        assert r.in_transaction is False
    finally:
        r.close()


def test_checkpoint_writer_only(tmp_path: Path) -> None:
    """AC#4: checkpoint() succeeds on a writer and fails on a reader."""
    path = tmp_path / "tgkb.sqlite3"
    w = db.open_writer(path)
    w.execute("CREATE TABLE t(x)")
    with db.writing(w):
        w.execute("INSERT INTO t VALUES (1)")

    # Writer can checkpoint.
    db.checkpoint(w)

    r = db.open_reader(path)
    try:
        with pytest.raises(apsw.Error):
            db.checkpoint(r)
    finally:
        w.close()
        r.close()


def test_rejects_non_local_path() -> None:
    """AC#5: a UNC (network) path is rejected with ValueError on both openers."""
    unc = r"\\server\share\db.sqlite3"
    with pytest.raises(ValueError):
        db.open_writer(unc)
    with pytest.raises(ValueError):
        db.open_reader(unc)


def test_rejects_non_file_uri() -> None:
    """AC#5: a non-file URI scheme is rejected with ValueError."""
    with pytest.raises(ValueError):
        db.open_writer("https://example.com/db.sqlite3")
    with pytest.raises(ValueError):
        db.open_reader("http://example.com/db.sqlite3")


def test_reader_opens_path_with_special_chars(tmp_path: Path) -> None:
    """Review regression: a path containing '#' (a legal filename char) must
    still open through the reader. Hand-rolled URI string concat parsed '#' as a
    fragment; open_reader now uses Path.as_uri() which percent-encodes it.
    """
    special_dir = tmp_path / "a#b%c"
    special_dir.mkdir()
    path = special_dir / "tgkb.sqlite3"

    w = db.open_writer(path)
    w.execute("CREATE TABLE t(x)")
    with db.writing(w):
        w.execute("INSERT INTO t VALUES (99)")
    w.close()

    r = db.open_reader(path)
    try:
        assert r.execute("SELECT x FROM t").fetchone()[0] == 99
    finally:
        r.close()


def test_writing_rolls_back_when_commit_fails(tmp_path: Path) -> None:
    """Review regression: if COMMIT itself raises, writing() must roll back so
    the connection is not left mid-transaction (which would wedge the next
    BEGIN IMMEDIATE with 'cannot start a transaction within a transaction').

    A deferred FOREIGN KEY violation is the driver-faithful way to make COMMIT
    (not the INSERT) raise: the check is deferred to commit time, and APSW
    leaves the transaction OPEN after the failed COMMIT — exactly the wedge the
    guard prevents.
    """
    conn = db.open_writer(tmp_path / "tgkb.sqlite3")
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE child(pid INTEGER REFERENCES parent(id) "
            "DEFERRABLE INITIALLY DEFERRED)"
        )

        # The INSERT succeeds (deferred), but COMMIT fails on the FK check.
        with pytest.raises(apsw.ConstraintError):
            with db.writing(conn):
                conn.execute("INSERT INTO child(pid) VALUES (999)")

        # The connection must NOT be left in a transaction — a fresh writing()
        # block must succeed rather than raising a nested-BEGIN error.
        assert conn.in_transaction is False
        conn.execute("INSERT INTO parent(id) VALUES (1)")
        with db.writing(conn):
            conn.execute("INSERT INTO child(pid) VALUES (1)")  # valid now
        # The rolled-back row (999) is gone; only the committed valid row remains.
        assert conn.execute("SELECT count(*) FROM child").fetchone()[0] == 1
    finally:
        conn.close()
