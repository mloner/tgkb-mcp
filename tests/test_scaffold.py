"""Smoke tests for the project scaffold (Story 1.1).

Assert the three packages import and that the AD-1 dependency direction holds:
``writer`` and ``reader`` both reference ``common`` and neither references the
sibling process package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import tgkb
import tgkb.common
import tgkb.reader
import tgkb.writer

# Package root = the directory containing the tgkb/ package.
_PKG_ROOT = Path(tgkb.__file__).parent  # .../tgkb/


def test_packages_import() -> None:
    """All three packages import successfully."""
    assert tgkb.common is not None
    assert tgkb.writer is not None
    assert tgkb.reader is not None


def test_reader_mcp_module_imports_and_serves() -> None:
    """The reader MCP module (heavy `mcp` import) imports and can serve a tool.

    Covers the entry-point module that the package __init__ does not import.
    """
    import tgkb.reader.server as server

    assert server.mcp.name == "tgkb-reader"
    # Story 4.3 replaced the `ping` stub with the real read-only `server_status`
    # health tool; assert the server still imports and serves a tool.
    assert callable(server.server_status)
    assert callable(server.main)


def test_writer_and_reader_reference_common() -> None:
    """writer and reader expose the value they import from common (import direction)."""
    assert tgkb.writer.VERSION == tgkb.common.VERSION
    assert tgkb.reader.VERSION == tgkb.common.VERSION


def _tgkb_subpkg(dotted: str) -> str | None:
    """Reduce a dotted module name to its ``tgkb.<sub>`` subpackage, or None."""
    parts = dotted.split(".")
    if len(parts) >= 2 and parts[0] == "tgkb":
        return f"{parts[0]}.{parts[1]}"
    return None


def _imported_subpkgs(pkg_dir: Path, own_subpkg: str) -> set[str]:
    """Collect the ``tgkb.*`` subpackages imported anywhere under ``pkg_dir``.

    Handles all import spellings so the AD-1 guard cannot be evaded:
    - ``import tgkb.reader`` / ``import tgkb.reader as r``
    - ``from tgkb.reader import x`` (absolute)
    - ``from tgkb import reader`` (submodule via ``alias.name``)
    - ``from ..reader import x`` / ``from . import reader`` (relative, via ``node.level``)
    """
    imported: set[str] = set()
    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (sub := _tgkb_subpkg(alias.name)) is not None:
                        imported.add(sub)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Relative import: resolve against this package. For a file
                    # directly under tgkb/<name>/, its package is own_subpkg
                    # ("tgkb.<name>"). level=1 -> that package; level=2 -> one up
                    # ("tgkb"); so strip (level - 1) trailing components.
                    base_parts = own_subpkg.split(".")  # ["tgkb", "<name>"]
                    keep = max(0, len(base_parts) - (node.level - 1))
                    root_parts = base_parts[:keep]
                    prefix = ".".join(root_parts + ([node.module] if node.module else []))
                    for alias in node.names:
                        candidate = f"{prefix}.{alias.name}" if prefix else alias.name
                        if (sub := _tgkb_subpkg(candidate)) is not None:
                            imported.add(sub)
                        elif (sub := _tgkb_subpkg(prefix)) is not None:
                            imported.add(sub)
                    continue
                if not node.module:
                    continue
                if (sub := _tgkb_subpkg(node.module)) is not None:
                    imported.add(sub)
                elif node.module == "tgkb":
                    # ``from tgkb import reader`` — the submodule is in alias.name.
                    for alias in node.names:
                        if (sub := _tgkb_subpkg(f"tgkb.{alias.name}")) is not None:
                            imported.add(sub)
    return imported


def test_no_cross_process_imports() -> None:
    """AD-1: writer never imports reader and reader never imports writer."""
    assert (_PKG_ROOT / "writer").is_dir(), "writer package dir must exist"
    assert (_PKG_ROOT / "reader").is_dir(), "reader package dir must exist"

    writer_imports = _imported_subpkgs(_PKG_ROOT / "writer", "tgkb.writer")
    reader_imports = _imported_subpkgs(_PKG_ROOT / "reader", "tgkb.reader")

    assert "tgkb.reader" not in writer_imports, "writer must not import reader (AD-1)"
    assert "tgkb.writer" not in reader_imports, "reader must not import writer (AD-1)"

    # And both must import common (dependency direction is actually wired).
    assert "tgkb.common" in writer_imports, "writer must import common"
    assert "tgkb.common" in reader_imports, "reader must import common"


def test_scanner_detects_all_cross_import_spellings(tmp_path: Path) -> None:
    """The AD-1 scanner must catch every import spelling, not just `import tgkb.x`.

    Guards against the evasions the reviewer flagged: `from tgkb import reader`
    and relative imports would have slipped past the naive scanner.
    """
    pkg = tmp_path / "writer"
    pkg.mkdir()

    spellings = [
        "import tgkb.reader",
        "import tgkb.reader as r",
        "from tgkb.reader import x",
        "from tgkb import reader",
        "from ..reader import x",
        "from .. import reader",
    ]
    for i, src in enumerate(spellings):
        (pkg / f"m{i}.py").write_text(src, encoding="utf-8")
        found = _imported_subpkgs(pkg, "tgkb.writer")
        assert "tgkb.reader" in found, f"scanner missed: {src!r}"
        (pkg / f"m{i}.py").unlink()
