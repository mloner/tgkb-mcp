"""tgkb — Telegram knowledge base.

Two cooperating processes over one shared SQLite store (CQRS-lite):

- ``tgkb.writer`` — the only code that writes (auth, export, sync).
- ``tgkb.reader`` — read-only MCP query surface.
- ``tgkb.common`` — shared code (db access, schema, tokenizer, config, models).

Dependency direction (AD-1): ``writer -> common`` and ``reader -> common``,
never ``writer <-> reader``.
"""

__version__ = "0.1.0"
