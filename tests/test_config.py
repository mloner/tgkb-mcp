"""Tests for the Tracked Chats config parser (Story 2.1, ``tgkb/common/config.py``).

No network, no SQLite — config parsing is pure stdlib + files. Every test writes
its config under ``tmp_path`` so nothing touches the real ``.tgkb-data/``.
Synchronous throughout (no Telethon, no coroutines → no ``pytest-asyncio``).

Coverage maps to the ACs:

- AC#1: exact parsed values + types, by ``username`` and by ``id``; ``@`` stripped.
- AC#2: a mixed valid/invalid file → the survivors AND the ONE specific error;
  ``load_config`` does NOT raise on a bad entry.
- AC#3: missing / malformed / no-``chats`` file → typed ``ConfigError`` (not the
  raw ``FileNotFoundError`` / ``JSONDecodeError``), message names the path, no traceback.
- Single-source path constant + no-secret discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgkb.common import config
from tgkb.common.config import (
    ChatConfigError,
    ConfigError,
    ParsedConfig,
    TrackedChat,
    default_config_path,
    load_config,
)
from tgkb.common.models import APP_DATA_DIR_NAME


def _write(tmp_path: Path, obj: object) -> Path:
    """Write ``obj`` as JSON to a config file under ``tmp_path``."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _write_raw(tmp_path: Path, body: str) -> Path:
    """Write a raw (possibly malformed) config body verbatim."""
    path = tmp_path / "config.json"
    path.write_text(body, encoding="utf-8")
    return path


# --- AC#1: valid parse, by username and by id ---------------------------------


def test_loads_valid_chats_by_username_and_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "chats": [
                {"username": "some_public_channel"},
                {"id": -1001234567890, "title": "My private group"},
            ]
        },
    )
    result = load_config(path)

    assert isinstance(result, ParsedConfig)
    assert result.errors == ()
    assert result.chats == (
        TrackedChat(username="some_public_channel"),
        TrackedChat(id=-1001234567890, title="My private group"),
    )
    # Assert exact types — JSON gives a real int, username stays a str.
    by_username, by_id = result.chats
    assert by_username.username == "some_public_channel" and by_username.id is None
    assert isinstance(by_id.id, int) and by_id.username is None
    # The "only listed chats" guarantee is structural: nothing unlisted can appear.
    assert len(result.chats) == 2


def test_leading_at_stripped(tmp_path: Path) -> None:
    path = _write(tmp_path, {"chats": [{"username": "@durov"}]})
    result = load_config(path)
    assert result.errors == ()
    assert result.chats == (TrackedChat(username="durov"),)


def test_multiple_leading_at_stripped(tmp_path: Path) -> None:
    # A carelessly pasted "@@name" (or "@ name") strips to the bare handle.
    for raw in ("@@durov", "@ durov"):
        path = _write(tmp_path, {"chats": [{"username": raw}]})
        result = load_config(path)
        assert result.errors == (), raw
        assert result.chats == (TrackedChat(username="durov"),), raw


def test_username_with_whitespace_is_error(tmp_path: Path) -> None:
    # A handle with internal whitespace is never valid — caught offline.
    path = _write(tmp_path, {"chats": [{"username": "foo bar"}]})
    result = load_config(path)
    assert result.chats == ()
    assert len(result.errors) == 1
    assert "whitespace" in result.errors[0].message.lower()


# --- AC#2: skip-and-continue (collect, do not raise) --------------------------


def test_invalid_entry_collected_not_raised(tmp_path: Path) -> None:
    # Three entries; the MIDDLE one is bad (neither id nor username).
    path = _write(
        tmp_path,
        {
            "chats": [
                {"username": "first_ok"},
                {"title": "bad entry, no identity"},
                {"id": 777},
            ]
        },
    )
    result = load_config(path)  # must NOT raise — the whole point of AC#2.

    assert result.chats == (
        TrackedChat(username="first_ok"),
        TrackedChat(id=777),
    )
    assert len(result.errors) == 1
    (err,) = result.errors
    assert isinstance(err, ChatConfigError)
    assert err.index == 2  # names the specific offending entry.
    assert "neither" in err.message.lower()
    # The message identifies THAT entry (its title woven in).
    assert "bad entry, no identity" in err.message


def test_id_must_be_int(tmp_path: Path) -> None:
    # Quoted → a str in JSON → that entry is an error; the other still parses.
    path = _write(
        tmp_path,
        {"chats": [{"id": "123"}, {"username": "ok"}]},
    )
    result = load_config(path)
    assert result.chats == (TrackedChat(username="ok"),)
    assert len(result.errors) == 1
    assert result.errors[0].index == 1
    assert "id" in result.errors[0].message.lower()


def test_float_id_is_error(tmp_path: Path) -> None:
    # JSON has no separate int type in syntax, but the parser distinguishes
    # `123.0` (float) from `123` (int). A float id is not a valid peer id.
    path = _write(tmp_path, {"chats": [{"id": 123.0}, {"username": "ok"}]})
    result = load_config(path)
    assert result.chats == (TrackedChat(username="ok"),)
    assert len(result.errors) == 1
    assert "id" in result.errors[0].message.lower()


def test_both_id_and_username_is_error(tmp_path: Path) -> None:
    path = _write(tmp_path, {"chats": [{"username": "x", "id": 5}]})
    result = load_config(path)
    assert result.chats == ()
    assert len(result.errors) == 1
    assert "both" in result.errors[0].message.lower()


def test_empty_username_is_error(tmp_path: Path) -> None:
    # A whitespace-only / bare-"@" username is rejected as an entry error.
    path = _write(tmp_path, {"chats": [{"username": "@"}]})
    result = load_config(path)
    assert result.chats == ()
    assert len(result.errors) == 1
    assert "username" in result.errors[0].message.lower()


# --- AC#3: file-level failures raise a typed ConfigError ----------------------


def test_missing_file_raises_configerror(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(ConfigError) as excinfo:
        load_config(missing)
    # Typed ConfigError, NOT the raw FileNotFoundError.
    assert not isinstance(excinfo.value, FileNotFoundError)
    assert str(missing) in str(excinfo.value)
    assert "chats" in str(excinfo.value)  # format hint present.


def test_malformed_json_raises_configerror(tmp_path: Path) -> None:
    path = _write_raw(tmp_path, "{this is not json,,,}")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    # NOT the raw json.JSONDecodeError.
    assert not isinstance(excinfo.value, json.JSONDecodeError)
    assert str(path) in str(excinfo.value)


def test_root_not_object_raises_configerror(tmp_path: Path) -> None:
    # Valid JSON, but the root is an array, not an object with `chats`.
    path = _write(tmp_path, [{"username": "x"}])
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "chats" in str(excinfo.value)


def test_no_chats_key_raises_configerror(tmp_path: Path) -> None:
    # Valid JSON object, but no `chats` array.
    path = _write(tmp_path, {"title": "not a chats array"})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "chats" in str(excinfo.value)


def test_chats_not_an_array_raises_configerror(tmp_path: Path) -> None:
    # `chats` present but a scalar, not an array of objects.
    path = _write(tmp_path, {"chats": 5})
    with pytest.raises(ConfigError):
        load_config(path)


def test_empty_chats_array_raises_configerror(tmp_path: Path) -> None:
    # An empty `chats` declares zero chats → a file-level error, NOT a silent
    # exit-0 no-op. The message promises "one or more" entries.
    path = _write(tmp_path, {"chats": []})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "chats" in str(excinfo.value)


# --- Story 2.3: optional throttle.rate_per_second ceiling ---------------------


def _with_throttle(rate: object) -> dict:
    return {"chats": [{"username": "ok"}], "throttle": {"rate_per_second": rate}}


def test_rate_absent_is_none(tmp_path: Path) -> None:
    # No `throttle` object → rate_per_second is None; the CLI applies the
    # throttle's own default (config does not duplicate the number).
    path = _write(tmp_path, {"chats": [{"username": "ok"}]})
    result = load_config(path)
    assert result.rate_per_second is None
    assert result.chats == (TrackedChat(username="ok"),)


def test_rate_present_valid_parsed(tmp_path: Path) -> None:
    path = _write(tmp_path, _with_throttle(2.5))
    result = load_config(path)
    assert result.rate_per_second == 2.5


def test_rate_integer_coerced_to_float(tmp_path: Path) -> None:
    # A bare integer is a valid rate; it is returned as a float.
    path = _write(tmp_path, _with_throttle(3))
    result = load_config(path)
    assert result.rate_per_second == 3.0
    assert isinstance(result.rate_per_second, float)


def test_rate_zero_is_valid_no_limit(tmp_path: Path) -> None:
    # An explicit 0 means "no proactive limit" — VALID, not an error.
    path = _write(tmp_path, _with_throttle(0))
    result = load_config(path)
    assert result.rate_per_second == 0.0


def test_rate_non_numeric_raises_configerror(tmp_path: Path) -> None:
    # A malformed global rate is a FILE-level ConfigError (not a per-entry skip).
    path = _write(tmp_path, _with_throttle("fast"))
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "rate_per_second" in str(excinfo.value)


def test_rate_bool_rejected(tmp_path: Path) -> None:
    # bool is an int subclass in Python — reject `"rate_per_second": true`.
    path = _write(tmp_path, _with_throttle(True))
    with pytest.raises(ConfigError):
        load_config(path)


def test_throttle_not_object_raises(tmp_path: Path) -> None:
    # A `throttle` that is not an object is a file-level error.
    path = _write(tmp_path, {"chats": [{"username": "ok"}], "throttle": 1.0})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "throttle" in str(excinfo.value)


def test_throttle_object_without_rate_raises(tmp_path: Path) -> None:
    # A present `throttle` object must declare rate_per_second.
    path = _write(
        tmp_path, {"chats": [{"username": "ok"}], "throttle": {"other": "x"}}
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "rate_per_second" in str(excinfo.value)


def test_rate_non_finite_raises_configerror(tmp_path: Path) -> None:
    # Python's json parser accepts NaN/Infinity/-Infinity as real floats that
    # pass the isinstance check, but neither expresses a usable rate — both would
    # SILENTLY disable the account-safety ceiling. They must be rejected as a
    # file-level error. (json.dumps emits these bare tokens by default.)
    for bad in (float("nan"), float("inf"), float("-inf")):
        path = _write(tmp_path, _with_throttle(bad))
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        assert "finite" in str(excinfo.value).lower(), bad


# --- Path single-sourcing + no-secret discipline ------------------------------


def test_default_config_path() -> None:
    assert default_config_path() == Path(APP_DATA_DIR_NAME) / "config.json"
    assert config.CONFIG_FILENAME == "config.json"


def test_error_messages_are_secret_free(tmp_path: Path) -> None:
    # Config holds chat ids, not secrets — but keep the no-secret-in-output
    # discipline: no message should ever look like an api_hash/session token.
    path = _write(
        tmp_path,
        {"chats": [{"id": "not-an-int"}, {"username": "x", "id": 5}]},
    )
    result = load_config(path)
    messages = [e.message for e in result.errors]
    # Also exercise a file-level message.
    try:
        load_config(tmp_path / "missing.json")
    except ConfigError as exc:
        messages.append(str(exc))
    joined = " ".join(messages).lower()
    # Guard against real secret-like tokens leaking into messages. ("secret"
    # itself is excluded as a keyword — it appears in tmp path names here and is
    # not a secret value.)
    for banned in ("api_hash", "api_id", "passphrase", "session.enc"):
        assert banned not in joined
