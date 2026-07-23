"""Config parsing for the list of Tracked Chats (Story 2.1).

This module RESOLVES the architecture's deferred config-format decision: the
config is **JSON, parsed with the stdlib** :mod:`json` (read-only). Choosing
``json`` adds **zero new dependencies** — matching the project's
minimal-dependency, deterministic-core philosophy — is a format every user and
tool already understands, and preserves the int-vs-float distinction (a numeric
peer ``id`` stays an ``int``). JSON has **no comments**, so every field is
documented in ``config.json.example`` and the README instead. There is no
serializer: we only ever READ config.

Only the **tracked-chat list** (and the optional throttle knob) lives here.
Telegram **credentials** (``api_id`` / ``api_hash`` / ``phone``) are secrets and
are deliberately NOT in this file — they come from the environment (``TG_*`` env
vars / a gitignored ``.env`` on first run, then the encrypted session). Keeping
the secret out of a config file that sits beside a committable
``config.json.example`` is the whole point (NFR-1).

It lives in ``common/`` because config knowledge is shared (per the module
layout: "common: … config, models"), beside :mod:`tgkb.common.db` and
:mod:`tgkb.common.secrets_store`. It imports ONLY stdlib + the app-data-dir name
from :mod:`tgkb.common.models`; it MUST NEVER import ``tgkb.writer``,
``tgkb.reader``, or ``apsw``/``db`` (config parsing touches no SQLite and no
process-specific code — AD-1).

**The AC#2 skip-and-continue contract (the heart of this story):**
:func:`load_config` returns :class:`ParsedConfig` ``(chats, errors)`` — the valid
:class:`TrackedChat` list PLUS a per-entry error list — instead of raising on the
first bad entry. An **entry-level** problem (neither/both of id+username, wrong
type, empty username) is *collected* into ``errors`` and the entry is skipped, so
the run continues for the good entries. Only a **file-level** problem (missing
file, un-parseable JSON, no ``chats`` array) raises a typed
:class:`ConfigError` → a clean exit 1, no traceback (AC#3).

**Offline only.** This story validates only what is checkable without a network
round-trip (shape/type). Whether a ``username`` actually resolves to a peer, or
an ``id`` is accessible, is a Telegram round-trip that belongs to **Story 2.4's**
export path — which must extend this SAME ``(chats, errors)`` skip-and-continue
shape to *reachability* errors rather than inventing a second result shape.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from tgkb.common.models import APP_DATA_DIR_NAME

# Single fixed filename so later commands (Story 2.4 export) reuse it verbatim.
# Sits beside the DB + encrypted session under the one gitignored app-data dir
# (mirror of db.DB_FILENAME / secrets_store.SESSION_FILENAME).
CONFIG_FILENAME = "config.json"

# One-line format hint reused across file-level error messages so the user always
# sees "what a valid config looks like" alongside "what went wrong".
_FORMAT_HINT = (
    'Create it as a JSON object with a "chats" array, each entry having exactly '
    'one of "username" (public @name, without the @) or "id" (numeric peer id).'
)


class ConfigError(Exception):
    """Raised on a FILE-level config failure (AC#3).

    Missing file, un-parseable JSON, or a top-level ``chats`` that is not an
    array of objects. Carries an actionable, secret-free message naming the path
    and the expected format. The CLI maps it to a clean exit 1 with no traceback.

    A single malformed *entry* does NOT raise this — it is collected into
    :attr:`ParsedConfig.errors` so the run continues (AC#2).
    """


@dataclass(frozen=True)
class TrackedChat:
    """A single Tracked Chat identity parsed from a ``chats`` entry.

    A well-formed entry sets **exactly one** of ``username`` / ``id`` (both-or-
    neither is a per-entry validation error). This is a plain data holder — it is
    NOT resolved against Telegram here (resolution is Story 2.4). ``title`` is
    free-text for the user's own readability and is ignored by any resolution
    logic.
    """

    username: str | None = None
    id: int | None = None
    title: str | None = None


@dataclass(frozen=True)
class ChatConfigError:
    """A per-entry validation problem (AC#2) — collected, not raised.

    Carries the 1-based ``index`` of the offending ``chats`` entry and a
    specific, secret-free ``message`` so the CLI can print
    "skipping chat entry #N: …". When available, the entry's ``title``/``username``
    is woven into the message so the report identifies THAT entry.
    """

    index: int
    message: str


@dataclass(frozen=True)
class ParsedConfig:
    """The result of :func:`load_config`: valid chats AND per-entry errors.

    ``chats`` holds every well-formed :class:`TrackedChat`; ``errors`` holds a
    :class:`ChatConfigError` for each skipped entry. A partially-valid config is a
    WARNING (non-empty ``errors`` with a non-empty ``chats``), not a failure — the
    file itself parsed fine. Story 2.4 extends this exact shape to reachability
    errors so there is ONE skip-and-continue model.

    ``rate_per_second`` carries the optional Story 2.3 proactive-throttle ceiling
    (the ``throttle.rate_per_second`` object member): the parsed float when the
    user set it, ``0.0`` when they explicitly disabled the limit, and ``None``
    when the ``throttle`` object is **absent** — in which case the CLI/export path
    applies the throttle's OWN default
    (``tgkb.writer.throttle.DEFAULT_REQUEST_RATE_PER_SECOND``). ``config.py``
    deliberately does NOT import that default (AD-1: ``common`` never imports
    ``writer``) and does NOT duplicate the number — it only reports "absent" as
    ``None`` and lets the throttle single-source its default.
    """

    chats: tuple[TrackedChat, ...]
    errors: tuple[ChatConfigError, ...]
    rate_per_second: float | None = None


def default_config_path() -> Path:
    """Return the config path under the repo-relative app-data dir.

    ``.tgkb-data/config.json``. The app-data dir name comes from
    :data:`tgkb.common.models.APP_DATA_DIR_NAME` (never hardcoded here) so the
    Story 1.1 convention stays single-sourced — mirroring
    :func:`tgkb.common.db.default_db_path` and
    :func:`tgkb.common.secrets_store.default_session_path`.
    """
    return Path(APP_DATA_DIR_NAME) / CONFIG_FILENAME


def _entry_label(index: int, entry: dict) -> str:
    """Build a specific "chat entry #N" label, weaving in title/username if present.

    Keeps AC#2 error messages identifying THAT entry. Never includes an ``id`` as
    a "secret" — ids are not secrets, but we keep messages minimal and readable.
    """
    hint = None
    title = entry.get("title")
    username = entry.get("username")
    if isinstance(title, str) and title.strip():
        hint = f"title={title!r}"
    elif isinstance(username, str) and username.strip():
        hint = f"username={username!r}"
    return f"chat entry #{index}" + (f" ({hint})" if hint else "")


def _validate_entry(index: int, entry: object) -> TrackedChat | ChatConfigError:
    """Validate one ``chats`` entry offline; return a chat OR a collected error.

    Rules (all checkable without a network round-trip):

    - The entry must be an object (dict). A stray scalar in the array is an error.
    - Exactly one of ``username`` / ``id`` is present and non-empty (neither →
      error; both → error, "use one").
    - ``id``, if present, must be a real ``int`` (a quoted ``"123"`` is a ``str``
      in JSON → error; ``json`` keeps the int-vs-float distinction, so a real
      integer literal stays an ``int``). Note ``bool`` is an ``int`` subclass in
      Python, so ``"id": true`` is rejected explicitly.
    - ``username``, if present, must be a non-empty ``str``; a leading ``@`` is
      stripped defensively (users paste ``@name``); whitespace-only is rejected.
    """
    if not isinstance(entry, dict):
        return ChatConfigError(index, f"chat entry #{index} is not an object.")

    label = _entry_label(index, entry)
    has_username = "username" in entry
    has_id = "id" in entry

    if has_username and has_id:
        return ChatConfigError(
            index, f"{label} sets both `username` and `id`; use exactly one."
        )
    if not has_username and not has_id:
        return ChatConfigError(
            index, f"{label} has neither `username` nor `id`; set exactly one."
        )

    if has_id:
        raw_id = entry["id"]
        # bool is a subclass of int — reject `"id": true` explicitly. A float
        # (e.g. `"id": 1.5`, or `123.0` which JSON parses as a float) is not a
        # valid peer id either.
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            return ChatConfigError(
                index,
                f"{label} has a non-integer `id`; use an unquoted whole number "
                f"(e.g. \"id\": -1001234567890).",
            )
        title = entry.get("title")
        return TrackedChat(
            id=raw_id, title=title if isinstance(title, str) else None
        )

    # has_username
    raw_username = entry["username"]
    if not isinstance(raw_username, str):
        return ChatConfigError(index, f"{label} has a non-string `username`.")
    # Strip surrounding whitespace and ALL leading '@' (users paste "@name" or
    # occasionally "@@name" from a copy) — then re-strip, since "@ name" leaves
    # a leading space after the '@' is removed.
    username = raw_username.strip().lstrip("@").strip()
    if not username:
        return ChatConfigError(index, f"{label} has an empty `username`.")
    if any(c.isspace() for c in username):
        # A Telegram @handle never contains whitespace; catch it offline rather
        # than deferring an obvious mistake to a network round-trip in 2.4.
        return ChatConfigError(
            index, f"{label} has a `username` containing whitespace."
        )
    title = entry.get("title")
    return TrackedChat(
        username=username, title=title if isinstance(title, str) else None
    )


def _parse_rate(data: dict, path: Path) -> float | None:
    """Parse the optional ``throttle.rate_per_second`` global knob (Story 2.3).

    Returns the parsed rate, or ``None`` when the ``throttle`` object is absent
    (the caller applies the throttle's own default — see :class:`ParsedConfig`).
    A ceiling of ``0`` (or any non-positive number) is a VALID explicit
    "no proactive limit" and is returned as-is (the throttle still honors
    FLOOD_WAIT).

    Unlike a bad ``chats`` entry (skip-and-continue, AC#2), a malformed global
    rate is a **file-level** :class:`ConfigError` — it is a single global knob,
    not one of many entries, so "skip and continue" does not apply. Raised cases:
    ``throttle`` is not an object; ``rate_per_second`` is missing from a present
    ``throttle``; or it is not a real number (``bool`` is rejected explicitly,
    since it is an ``int`` subclass in Python).
    """
    if "throttle" not in data:
        return None
    throttle = data["throttle"]
    if not isinstance(throttle, dict):
        raise ConfigError(
            f"Config file {path} has a `throttle` that is not an object; use "
            '`"throttle": {"rate_per_second": <number>}`.'
        )
    if "rate_per_second" not in throttle:
        raise ConfigError(
            f"Config file {path} has a `throttle` object without "
            "`rate_per_second`; set `rate_per_second` (requests per second, "
            "e.g. 1.0) or remove the `throttle` object to use the default."
        )
    raw = throttle["rate_per_second"]
    # bool is a subclass of int — reject `"rate_per_second": true` explicitly.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(
            f"Config file {path} has a non-numeric `rate_per_second`; use a "
            "number (requests per second, e.g. 1.0)."
        )
    rate = float(raw)
    # Python's json parser accepts `NaN`/`Infinity`/`-Infinity` as legitimate
    # floats that pass the isinstance check above, but neither expresses a usable
    # rate: `inf` (and huge finite rates) makes the min-interval 0 → pass-through,
    # and `nan` makes every spacing comparison False → the gate never waits.
    # Either SILENTLY disables the account-safety ceiling (the exact SM-C1/FR-5
    # ban risk AD-10 exists to prevent), so a non-finite rate is a file-level
    # error, not a valid config.
    if not math.isfinite(rate):
        raise ConfigError(
            f"Config file {path} has a non-finite `rate_per_second` "
            f"({raw}); use a finite number (requests per second, e.g. 1.0), "
            "or 0 to disable the proactive limit."
        )
    return rate


def load_config(path: Path | None = None) -> ParsedConfig:
    """Parse + validate the JSON config; return valid chats AND per-entry errors.

    Defaults to :func:`default_config_path` (``.tgkb-data/config.json``).

    **File-level failures raise a typed** :class:`ConfigError` **(AC#3)** with an
    actionable, secret-free message — never a raw ``FileNotFoundError`` or
    ``json.JSONDecodeError``:

    - The file does not exist → names the expected path + a format hint.
    - Malformed JSON → the underlying parse message (which names line/column) +
      the path.
    - No top-level ``chats`` array → "must contain a `chats` array with one or
      more entries".

    **Per-entry problems are collected, not raised (AC#2):** each malformed entry
    becomes a :class:`ChatConfigError` (naming that entry) and is skipped; each
    well-formed entry becomes a :class:`TrackedChat`. The run continues for the
    good entries.

    This validates only OFFLINE shape/type. Reachability (does the ``username``
    resolve? is the ``id`` accessible?) is a Telegram round-trip left to Story 2.4.

    The optional ``throttle.rate_per_second`` global knob (Story 2.3) is parsed
    into :attr:`ParsedConfig.rate_per_second` (``None`` when absent → the caller
    applies the throttle's default). A malformed rate is a **file-level**
    :class:`ConfigError`, not a per-entry skip.
    """
    path = path if path is not None else default_config_path()

    if not path.is_file():
        # Covers both a truly missing path and a path that resolves to a
        # directory (``is_file()`` is False for a dir) — either way the user has
        # no readable config file here.
        raise ConfigError(f"Config file not found: {path}\n{_FORMAT_HINT}")

    try:
        # json.loads on the decoded text. Malformed JSON raises
        # json.JSONDecodeError (a ValueError subclass) which names line/column.
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Never let JSONDecodeError escape raw — the CLI has one ConfigError
        # handler. The parse message already names the line/column.
        raise ConfigError(f"Could not parse config file {path}: {exc}") from None
    except OSError as exc:
        # A file that exists but cannot be read (permissions, a race that
        # removed/replaced it, a bad symlink) must ALSO surface as the typed
        # ConfigError — not a raw traceback — so AC#3's "no stack-trace dump"
        # holds for every file-level failure, not just malformed JSON.
        raise ConfigError(f"Could not read config file {path}: {exc}") from None
    except UnicodeDecodeError as exc:
        # A non-UTF-8 file is a file-level problem too — surface it typed.
        raise ConfigError(
            f"Could not decode config file {path} as UTF-8: {exc}"
        ) from None

    # The top level must be a JSON object (dict). A bare array/scalar at the root
    # is a file-level error — we need a `chats` member (and optionally `throttle`).
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} must be a JSON object with a `chats` array.\n"
            f"{_FORMAT_HINT}"
        )

    raw_chats = data.get("chats")
    if not isinstance(raw_chats, list) or not raw_chats:
        # Missing key, a non-array `chats`, OR an empty `chats` array all mean
        # "no chats declared" — a file-level error (AC#3). An empty array must
        # NOT report success: the message promises "one or more" entries, so a
        # chat-less config is a clear, actionable failure, not a silent exit-0
        # no-op.
        raise ConfigError(
            f"Config file {path} must contain a `chats` array with one or more "
            f"entries.\n{_FORMAT_HINT}"
        )

    # The optional global throttle ceiling (Story 2.3). A malformed value is a
    # file-level ConfigError (raised here), consistent with the other file-level
    # failures above — NOT a per-entry skip.
    rate_per_second = _parse_rate(data, path)

    chats: list[TrackedChat] = []
    errors: list[ChatConfigError] = []
    for i, entry in enumerate(raw_chats, start=1):
        result = _validate_entry(i, entry)
        if isinstance(result, TrackedChat):
            chats.append(result)
        else:
            errors.append(result)

    return ParsedConfig(
        chats=tuple(chats),
        errors=tuple(errors),
        rate_per_second=rate_per_second,
    )
