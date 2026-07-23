"""Reader process — read-only MCP query surface over the shared store.

Exposes MCP query tools. Imports from ``tgkb.common`` only; never from
``tgkb.writer`` (AD-1).
"""

# Import from common to establish and prove the dependency direction (reader -> common).
from tgkb.common import VERSION
from tgkb.reader.server import ReadQueryRejected, SearchIndexStaleError

__all__ = ["VERSION", "ReadQueryRejected", "SearchIndexStaleError"]
