"""SQLite access layer — the SOLE place SQLite connections are opened (Story 1.2).

This module implements the connection/concurrency substrate every later
DB-touching story stands on (schema 2.2, FTS 3.x, sync 4.x, MCP reader
4.3-4.5). It ships NO schema/DDL — creating tables is Story 2.2 / 3.2.

Architecture invariants enforced here:

- **AD-1**: only ``common`` touches SQLite. ``writer`` and ``reader`` obtain
  connections via :func:`open_writer` / :func:`open_reader`; they must never
  construct an ``apsw.Connection`` themselves.
- **AD-2**: the reader is read-only at the *SQL level* (not file-level
  ``O_RDONLY``, which would break WAL), enforced by TWO layers —
  ``SQLITE_OPEN_READONLY`` + URI ``?mode=ro`` (Layer 1) and
  ``PRAGMA query_only=ON`` (Layer 2).
- **AD-3**: WAL mode; ``busy_timeout≈5000ms`` on every connection in both
  processes; ``synchronous=NORMAL``; writes take the write lock upfront via
  ``BEGIN IMMEDIATE``; readers use short read transactions; only the writer
  checkpoints; the DB is host-local.

The pinned driver is **APSW** (bundles SQLite 3.53.x + FTS5). The stdlib
``sqlite3`` module is banned project-wide — APSW is required later for the
custom FTS5 tokenizer (Story 3.1) and mixing drivers would fork connection
semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import apsw

from tgkb.common.models import APP_DATA_DIR_NAME

# Single fixed DB filename so Stories 1.4 / 2.2 reuse it verbatim.
DB_FILENAME = "tgkb.sqlite3"

# busy_timeout applied to EVERY connection in BOTH processes (AD-3): turns a
# transient lock into a short wait instead of an immediate SQLITE_BUSY.
_BUSY_TIMEOUT_MS = 5000

# Accepts either a filesystem path or a raw string path (so tests can pass a
# host-independent UNC/URI string without a real network).
PathLike = Union[str, Path]


def default_db_path() -> Path:
    """Return the DB path under the repo-relative app-data dir.

    ``.tgkb-data/tgkb.sqlite3``. The app-data dir name comes from
    :data:`tgkb.common.models.APP_DATA_DIR_NAME` (never hardcoded here) so the
    Story 1.1 convention stays single-sourced.
    """
    return Path(APP_DATA_DIR_NAME) / DB_FILENAME


def _reject_non_local(db_path: PathLike) -> None:
    """Guard AC#5: reject obvious network locations (raises ``ValueError``).

    This is a lightweight, explicit check — NOT exhaustive network-mount
    detection (out of scope; a mapped drive letter or a bind-mounted network
    share is not caught). It rejects a Windows UNC path (``\\\\host\\share``)
    and any URI with a non-``file`` scheme. WAL requires a host-local file, so a
    non-local path would silently break concurrency later.
    """
    raw = str(db_path)

    # UNC path: \\host\share\... (both slash directions).
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise ValueError(
            f"database path must be host-local, not a network (UNC) location: {raw!r}"
        )

    # URI with a non-file scheme (e.g. http://, smb://). A bare Windows drive
    # path like C:\... is NOT a URI scheme, so guard on '://' being present.
    if "://" in raw and not raw.startswith("file://"):
        scheme = raw.split("://", 1)[0]
        raise ValueError(
            f"database path must be a local filesystem path, not a {scheme!r} URI: {raw!r}"
        )


def _apply_common_pragmas(conn: apsw.Connection) -> None:
    """Set busy_timeout on any connection (AD-3: every connection, both procs)."""
    conn.set_busy_timeout(_BUSY_TIMEOUT_MS)


def open_writer(db_path: PathLike) -> apsw.Connection:
    """Open the single read-write connection with the project's WAL PRAGMAs.

    Creates the parent app-data dir if missing, opens a read-write APSW
    connection, and applies the writer PRAGMAs (Task 2). This is the only
    connection allowed to write, checkpoint, or own ``journal_mode``.
    """
    _reject_non_local(db_path)
    path = Path(db_path)
    # A writer may bring the DB (and its dir) into existence.
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = apsw.Connection(str(path))
    try:
        _apply_common_pragmas(conn)  # busy_timeout BEFORE any BEGIN IMMEDIATE (AC#3).

        # journal_mode is a PERSISTENT DB property — set once it sticks — but we
        # set it on every writer open (idempotent) so a fresh file always lands
        # in WAL.
        journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if journal_mode.lower() != "wal":
            # A silent fallback to delete/memory (e.g. a non-local file) is a
            # latent concurrency bug — surface it immediately (Task 2).
            raise RuntimeError(
                f"expected journal_mode=wal but SQLite reports {journal_mode!r}; "
                "the database file may not be host-local"
            )

        conn.execute("PRAGMA synchronous=NORMAL")
    except BaseException:
        # Do not leak the open connection if any post-open setup fails.
        conn.close()
        raise
    return conn


def open_reader(db_path: PathLike) -> apsw.Connection:
    """Open a read-only connection with both AD-2 defense layers.

    A reader MUST NOT create the parent dir or the DB file — a reader must never
    bring a DB into existence. Opening a missing DB read-only raises.

    Layer 1 (connection flag backstop): ``SQLITE_OPEN_READONLY | SQLITE_OPEN_URI``
    with a ``file:...?mode=ro`` URI. ``mode=ro`` keeps WAL working, unlike a
    file-level ``O_RDONLY`` open (which AC#2 forbids).
    Layer 2 (statement guard): ``PRAGMA query_only=ON``.
    """
    _reject_non_local(db_path)
    path = Path(db_path)

    # Build the SQLite URI via Path.as_uri() so path components are correctly
    # percent-encoded — a filename containing '?', '#', or '%' (all legal) would
    # otherwise be misparsed as a URI query/fragment/escape and open the wrong
    # file (or fail). as_uri() yields e.g. file:///C:/dir/tgkb.sqlite3 on Windows
    # and file:///a/b on POSIX; we append the mode=ro query parameter.
    uri = path.resolve().as_uri() + "?mode=ro"

    # Layer 1: connection-level read-only + URI.
    conn = apsw.Connection(
        uri, flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI
    )
    try:
        _apply_common_pragmas(conn)  # AD-3: reader also waits out transient locks.
        # Layer 2: statement-time read-only guard.
        conn.execute("PRAGMA query_only=ON")
    except BaseException:
        # Do not leak the open connection if post-open setup fails.
        conn.close()
        raise
    return conn


@contextmanager
def writing(conn: apsw.Connection) -> Iterator[apsw.Connection]:
    """The ONLY sanctioned write-transaction entry point (AD-3).

    Issues ``BEGIN IMMEDIATE`` on enter (write lock taken UPFRONT, avoiding the
    DEFERRED lock-upgrade -> SQLITE_BUSY failure), ``COMMIT`` on clean exit, and
    ``ROLLBACK`` on exception. Stories 2.2 / 2.4 / 4.x wrap every write in this
    so the AD-4/AD-5/AD-12 "same transaction" guarantees hold.

    Do NOT use APSW's implicit ``with conn:`` for writes — that begins a
    DEFERRED transaction (write lock acquired lazily).
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        try:
            conn.execute("COMMIT")
        except BaseException:
            # If COMMIT itself fails (e.g. a transient SQLITE_BUSY), roll back so
            # the connection is not left mid-transaction — otherwise the next
            # BEGIN IMMEDIATE would raise "cannot start a transaction within a
            # transaction" and wedge the writer.
            conn.execute("ROLLBACK")
            raise


@contextmanager
def reading(conn: apsw.Connection) -> Iterator[apsw.Connection]:
    """A SHORT read transaction for point-in-time-consistent reads (AD-3).

    ``BEGIN`` (deferred) on enter, ``COMMIT``/``ROLLBACK`` on exit. Reader code
    (Story 4.x tools) opens one of these per query and closes it — it must never
    hold a transaction open across queries. A long-open read transaction in WAL
    pins the WAL and prevents the writer's checkpoint from reclaiming space.
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        try:
            conn.execute("COMMIT")
        except BaseException:
            # If COMMIT fails, roll back so the connection is not left with an
            # open read transaction (which would pin the WAL and block the
            # writer's checkpoint).
            conn.execute("ROLLBACK")
            raise


def checkpoint(conn: apsw.Connection) -> None:
    """Run a WAL checkpoint — a WRITER-ONLY maintenance primitive (AC#4).

    Uses ``wal_checkpoint(TRUNCATE)`` so the WAL file is reset to zero after a
    full checkpoint, keeping it from growing unbounded on a long-lived ``sync``
    process. TRUNCATE can be a no-op if a reader holds an old snapshot — that's
    fine, the next cycle retries (this is a tunable, not an invariant; PASSIVE
    is an equally defensible choice).

    This does NOT run automatically on commit — Epic 4's ``sync`` owns *when* to
    checkpoint. Only the writer may checkpoint: calling this on a read-only
    (``query_only`` / ``READONLY``) connection raises rather than silently
    no-opping.
    """
    # SQLite does NOT error on `wal_checkpoint` against a read-only connection —
    # it silently returns a no-op row. AC#4 requires the reader to be *unable*
    # to checkpoint, so guard explicitly: `readonly('main')` is True for a
    # SQLITE_OPEN_READONLY connection (our reader) and False for the writer.
    if conn.readonly("main"):
        raise apsw.ReadOnlyError(
            "checkpoint() is a writer-only maintenance primitive; "
            "a read-only connection must not checkpoint (AC#4)"
        )
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
