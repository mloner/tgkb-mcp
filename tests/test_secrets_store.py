"""Tests for the encrypt/decrypt envelope + payload bundle (Story 1.4).

Covers ``tgkb/common/secrets_store.py``. **No network, no real ``.tgkb-data/``**:
every file test uses ``tmp_path``. These tests exercise *behavior*, not flags
(the 1.1→1.3 review lesson): AC#4's on-disk-ciphertext guarantee is verified by
reading the RAW bytes written to disk and asserting the secret sentinels are
absent — never by merely asserting "encrypt was called".
"""

from __future__ import annotations

import base64
import json

import pytest

from tgkb.common import secrets_store as store
from tgkb.common.secrets_store import (
    SecretDecryptError,
    SESSION_FILENAME,
    decrypt,
    default_session_path,
    encrypt,
    read_encrypted,
    session_exists,
    write_encrypted,
)

# Distinctive sentinels so a leak assertion is unambiguous (AC#4).
PASSPHRASE = "correct horse battery staple"
SESSION_SENTINEL = "SESSION_SENTINEL_abc"
HASH_SENTINEL = "HASH_SENTINEL_xyz"


def test_encrypt_decrypt_roundtrip() -> None:
    """decrypt(pw, encrypt(pw, x)) == x — the core round-trip."""
    plaintext = b"a secret blob"
    assert decrypt(PASSPHRASE, encrypt(PASSPHRASE, plaintext)) == plaintext


def test_wrong_passphrase_raises() -> None:
    """A wrong passphrase raises the typed SecretDecryptError (never raw InvalidToken).

    And the error message leaks neither the passphrase nor the plaintext (NFR-1).
    """
    blob = encrypt("right-passphrase", b"the-plaintext-secret")
    with pytest.raises(SecretDecryptError) as excinfo:
        decrypt("wrong-passphrase", blob)
    msg = str(excinfo.value)
    assert "right-passphrase" not in msg
    assert "wrong-passphrase" not in msg
    assert "the-plaintext-secret" not in msg


def test_tampered_ciphertext_raises() -> None:
    """Flipping a byte in the ciphertext → SecretDecryptError (Fernet authenticates)."""
    blob = encrypt(PASSPHRASE, b"data")
    env = json.loads(blob)
    # Corrupt one character of the ciphertext token.
    token = env["ciphertext"]
    flipped = ("Z" if token[10] != "Z" else "Y") + token[11:]
    env["ciphertext"] = token[:10] + flipped
    tampered = json.dumps(env).encode("utf-8")
    with pytest.raises(SecretDecryptError):
        decrypt(PASSPHRASE, tampered)


def test_ciphertext_contains_no_plaintext(tmp_path) -> None:
    """AC#4: the raw on-disk bytes contain NO plaintext secret substring."""
    payload = store.dumps_payload(
        session=SESSION_SENTINEL, api_hash=HASH_SENTINEL, api_id=424242, phone="+1"
    )
    path = tmp_path / SESSION_FILENAME
    write_encrypted(path, PASSPHRASE, payload)

    raw = path.read_bytes()
    assert SESSION_SENTINEL.encode() not in raw
    assert HASH_SENTINEL.encode() not in raw
    # The passphrase must never land on disk either.
    assert PASSPHRASE.encode() not in raw


@pytest.mark.parametrize(
    "corrupt",
    [
        b"",  # empty / truncated-to-nothing file
        b"not json at all",  # non-JSON garbage
        b"{}",  # valid JSON, but missing every envelope key
        json.dumps({"v": 1, "kdf": "scrypt", "salt": "@@notb64@@",
                    "kdf_params": {"n": 131072, "r": 8, "p": 1, "length": 32},
                    "ciphertext": "x"}).encode(),  # bad base64 salt
        json.dumps({"v": 999, "kdf": "scrypt", "salt": "AAAA",
                    "kdf_params": {"n": 131072, "r": 8, "p": 1, "length": 32},
                    "ciphertext": "x"}).encode(),  # unsupported version
    ],
)
def test_corrupt_envelope_raises_secret_decrypt_error(corrupt: bytes) -> None:
    """A truncated/garbage/missing-key/bad-version envelope → SecretDecryptError.

    Regression for the review finding: the envelope parse used to run outside the
    try, so these cases escaped as raw JSONDecodeError/KeyError/binascii.Error.
    """
    with pytest.raises(SecretDecryptError):
        decrypt(PASSPHRASE, corrupt)


def test_loads_payload_rejects_schema_drift() -> None:
    """A correctly-decrypted-but-wrong-shape payload → SecretDecryptError."""
    # Missing keys, then wrong type for api_id.
    with pytest.raises(SecretDecryptError):
        store.loads_payload(json.dumps({"session": "s"}).encode())
    with pytest.raises(SecretDecryptError):
        store.loads_payload(
            json.dumps(
                {"session": "s", "api_hash": "h", "api_id": "not-int", "phone": "+1"}
            ).encode()
        )


def test_write_encrypted_is_atomic_leaves_no_temp(tmp_path) -> None:
    """After a successful write only the target exists — no leftover .tmp file."""
    path = tmp_path / SESSION_FILENAME
    write_encrypted(path, PASSPHRASE, b"data")
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()


def test_salt_is_random_per_encrypt() -> None:
    """A fresh random salt is used every encryption (envelopes' salts differ)."""
    a = json.loads(encrypt(PASSPHRASE, b"x"))
    b = json.loads(encrypt(PASSPHRASE, b"x"))
    assert a["salt"] != b["salt"]
    # Sanity: salt is 16 bytes once b64-decoded.
    assert len(base64.b64decode(a["salt"])) == 16


def test_envelope_shape_is_self_describing() -> None:
    """The envelope stores the KDF name + params so future files stay decryptable."""
    env = json.loads(encrypt(PASSPHRASE, b"x"))
    assert env["v"] == 1
    assert env["kdf"] == "scrypt"
    assert env["kdf_params"] == {"n": 131072, "r": 8, "p": 1, "length": 32}
    assert "salt" in env and "ciphertext" in env


def test_write_read_encrypted_roundtrip(tmp_path) -> None:
    """write_encrypted then read_encrypted returns the original bytes.

    ``session_exists`` is False before the write and True after.
    """
    path = tmp_path / "sub" / SESSION_FILENAME  # parent dir does not exist yet
    plaintext = b"round-trip-through-disk"

    assert session_exists(path) is False
    write_encrypted(path, PASSPHRASE, plaintext)
    assert session_exists(path) is True
    assert read_encrypted(path, PASSPHRASE) == plaintext


def test_read_encrypted_wrong_passphrase_raises(tmp_path) -> None:
    """read_encrypted surfaces SecretDecryptError on a wrong passphrase."""
    path = tmp_path / SESSION_FILENAME
    write_encrypted(path, "right", b"data")
    with pytest.raises(SecretDecryptError):
        read_encrypted(path, "wrong")


def test_payload_bundle_roundtrip() -> None:
    """Task 3: the {session, api_hash, api_id, phone} payload round-trips."""
    blob = store.dumps_payload(
        session="SESS", api_hash="HASH", api_id=999, phone="+15551234567"
    )
    restored = store.loads_payload(decrypt(PASSPHRASE, encrypt(PASSPHRASE, blob)))
    assert restored == {
        "session": "SESS",
        "api_hash": "HASH",
        "api_id": 999,
        "phone": "+15551234567",
    }


def test_default_session_path_uses_app_data_dir() -> None:
    """default_session_path() composes APP_DATA_DIR_NAME + SESSION_FILENAME."""
    from tgkb.common.models import APP_DATA_DIR_NAME

    path = default_session_path()
    assert path.name == SESSION_FILENAME == "session.enc"
    assert path.parent.name == APP_DATA_DIR_NAME
