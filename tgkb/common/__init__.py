"""Shared code for the writer and reader processes.

Everything both processes need lives here: database access, schema and
migrations, the lemmatizing FTS5 tokenizer, config, and models. Only
``common`` touches SQLite (relevant from Story 1.2 onward).
"""

from tgkb.common.config import (
    CONFIG_FILENAME,
    ChatConfigError,
    ConfigError,
    ParsedConfig,
    TrackedChat,
    default_config_path,
    load_config,
)
from tgkb.common.db import (
    DB_FILENAME,
    checkpoint,
    default_db_path,
    open_reader,
    open_writer,
    reading,
    writing,
)
from tgkb.common.models import APP_DATA_DIR_NAME, VERSION
from tgkb.common.ru_lemma import (
    BASE_TOKENIZER,
    HYPOTHESIS_POLICY,
    TOKENIZE_SPEC,
    TOKENIZER_NAME,
    fingerprint,
    lemmatize,
    register,
    tokenize,
)
from tgkb.common.schema import (
    Message,
    advance_last_id,
    check_and_repair_fingerprint,
    ensure_schema,
    fingerprint_matches,
    ingest_message,
    ingest_messages,
    mark_deleted,
    open_reader_with_fts,
    open_writer_with_schema,
    read_fingerprint,
    read_last_id,
    search_message_rowids,
    upsert_chat,
    write_fingerprint,
)
from tgkb.common.secrets_store import (
    SESSION_FILENAME,
    SecretDecryptError,
    default_session_path,
    read_encrypted,
    session_exists,
    write_encrypted,
)

__all__ = [
    "APP_DATA_DIR_NAME",
    "VERSION",
    "CONFIG_FILENAME",
    "ConfigError",
    "ChatConfigError",
    "ParsedConfig",
    "TrackedChat",
    "default_config_path",
    "load_config",
    "DB_FILENAME",
    "default_db_path",
    "open_writer",
    "open_reader",
    "writing",
    "reading",
    "checkpoint",
    "Message",
    "ensure_schema",
    "ingest_message",
    "ingest_messages",
    "upsert_chat",
    "advance_last_id",
    "read_last_id",
    "mark_deleted",
    "open_writer_with_schema",
    "open_reader_with_fts",
    "search_message_rowids",
    "read_fingerprint",
    "write_fingerprint",
    "check_and_repair_fingerprint",
    "fingerprint_matches",
    "SESSION_FILENAME",
    "SecretDecryptError",
    "default_session_path",
    "read_encrypted",
    "write_encrypted",
    "session_exists",
    "TOKENIZER_NAME",
    "BASE_TOKENIZER",
    "TOKENIZE_SPEC",
    "HYPOTHESIS_POLICY",
    "fingerprint",
    "lemmatize",
    "register",
    "tokenize",
]
