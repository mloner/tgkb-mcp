"""Writer process — the only code that writes to the store.

Owns auth, history export, and realtime sync. Imports from ``tgkb.common``
only; never from ``tgkb.reader`` (AD-1).
"""

# Import from common to establish and prove the dependency direction (writer -> common).
from tgkb.common import VERSION

__all__ = ["VERSION"]
