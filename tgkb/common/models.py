"""Shared constants and models used across both processes.

This module intentionally holds concrete values so ``writer`` and ``reader``
each have something real to import from ``common`` — proving the AD-1
dependency direction is wired from day one. It will grow to hold the real
data models (Story 2.2) and config schema (Story 2.1).
"""

# Package version — sourced from installed distribution metadata so there is a
# single source of truth (pyproject.toml). Falls back to a static string when
# the package is not installed (e.g. running from a raw checkout).
from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    VERSION = _dist_version("tgkb")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    VERSION = "0.1.0"

# App-data dir convention (deferred decision resolved at scaffolding, see Dev Notes).
# Config, encrypted session, and the SQLite database (+ WAL/SHM sidecars) all live
# under ONE gitignored, repo-relative, host-local directory (AD-8 / NFR-1).
# Stories 1.4 / 2.1 / 2.2 inherit this convention.
APP_DATA_DIR_NAME = ".tgkb-data"
