"""Encrypt-at-rest envelope for the Telegram session + ``api_hash`` (Story 1.4).

This is the SINGLE module that turns a secret blob into ciphertext on disk and
back. It lives in ``common/`` because the app-data-dir convention and the
on-disk envelope format are shared knowledge (like :mod:`tgkb.common.db`) — even
though, per **AD-8**, only the *writer* process ever decrypts ("decryption
happens only in the writer process at start"). The reader holds no Telegram
secret and MUST NOT import this module.

Design (AD-8, verified 2026-07-18):

- **KDF: Scrypt** (``cryptography.hazmat.primitives.kdf.scrypt``) with the
  OWASP-2026 interactive minimum ``n=2**17, r=8, p=1, length=32`` and a fresh
  16-byte random salt per encryption. Scrypt is chosen over Argon2id because it
  carries no OpenSSL-3.2+ ``UnsupportedAlgorithm`` portability caveat; the
  envelope records the KDF name + params so a later swap stays a drop-in.
- **Cipher: Fernet** (AES-128-CBC + HMAC-SHA256, encrypt-then-MAC) — authenticated,
  so a wrong passphrase or any tampering is detected *before* any plaintext is
  returned (raised as :class:`cryptography.fernet.InvalidToken`, re-raised here
  as the typed, secret-free :class:`SecretDecryptError`).

Envelope written to disk (JSON, self-describing):

    {"v": 1, "kdf": "scrypt",
     "kdf_params": {"n": 131072, "r": 8, "p": 1, "length": 32},
     "salt": <b64(16 random bytes)>, "ciphertext": <fernet token>}

**Secret-safe (NFR-1/AD-8):** this module never ``print``s/``log``s/interpolates
the passphrase, the derived key, the plaintext, the session string, or the
``api_hash``. Error messages name the field, never the value. Only ciphertext is
ever written to disk (AC#4).

Imports: ``cryptography`` + stdlib + :data:`tgkb.common.models.APP_DATA_DIR_NAME`
only — never ``tgkb.reader``, never ``tgkb.writer``, never ``apsw``/``db``.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from tgkb.common.models import APP_DATA_DIR_NAME

# Single fixed filename so later commands (2.4 / 4.x) reuse it verbatim. Sits
# beside the DB under the one gitignored app-data dir (mirror of db.DB_FILENAME).
SESSION_FILENAME = "session.enc"

# Scrypt params — OWASP-2026 interactive minimum. ``2**17`` is written literally
# for readability; the envelope stores the RESOLVED int (131072) so decrypt reads
# params from the file, never a hardcoded constant.
_SCRYPT_PARAMS = {"n": 2**17, "r": 8, "p": 1, "length": 32}

# Envelope format version — bump if the on-disk shape ever changes.
_ENVELOPE_VERSION = 1


class SecretDecryptError(Exception):
    """Raised when decryption fails (wrong passphrase or corrupt/tampered file).

    Carries an actionable, secret-free message. Wraps the low-level
    :class:`cryptography.fernet.InvalidToken` so callers never see the raw crypto
    exception and no secret is echoed.
    """


def default_session_path() -> Path:
    """Return the encrypted-session path under the repo-relative app-data dir.

    ``.tgkb-data/session.enc``. The app-data dir name comes from
    :data:`tgkb.common.models.APP_DATA_DIR_NAME` (never hardcoded here) so the
    Story 1.1 convention stays single-sourced — mirroring
    :func:`tgkb.common.db.default_db_path`.
    """
    return Path(APP_DATA_DIR_NAME) / SESSION_FILENAME


def _derive_key(passphrase: str, salt: bytes, params: dict) -> bytes:
    """Derive a Fernet key from the passphrase via Scrypt with the given params.

    Returns a url-safe-base64-encoded 32-byte key (the shape Fernet requires).
    ``params`` is read from the caller (the envelope on decrypt), never a global,
    so a file written with older params stays decryptable.
    """
    derived = Scrypt(
        salt=salt,
        length=params["length"],
        n=params["n"],
        r=params["r"],
        p=params["p"],
    ).derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(derived)


def encrypt(passphrase: str, plaintext: bytes) -> bytes:
    """Encrypt ``plaintext`` under a passphrase-derived key; return the envelope.

    A fresh random 16-byte salt is generated per call (never reused; not secret —
    stored in the envelope). Returns the self-describing JSON envelope as bytes.
    """
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt, _SCRYPT_PARAMS)
    token = Fernet(key).encrypt(plaintext)
    envelope = {
        "v": _ENVELOPE_VERSION,
        "kdf": "scrypt",
        # Store the RESOLVED ints (not 2**17) so decrypt reads them verbatim.
        "kdf_params": {
            "n": _SCRYPT_PARAMS["n"],
            "r": _SCRYPT_PARAMS["r"],
            "p": _SCRYPT_PARAMS["p"],
            "length": _SCRYPT_PARAMS["length"],
        },
        "salt": base64.b64encode(salt).decode("ascii"),
        "ciphertext": token.decode("ascii"),
    }
    return json.dumps(envelope).encode("utf-8")


def decrypt(passphrase: str, blob: bytes) -> bytes:
    """Decrypt an envelope produced by :func:`encrypt`; return the plaintext.

    Re-derives the key using the salt + KDF params STORED in the envelope (so a
    file written with different params stays readable). On a wrong passphrase or
    tampered/corrupt ciphertext, Fernet raises :class:`InvalidToken`; this is
    caught and re-raised as :class:`SecretDecryptError` with a secret-free
    message — the raw exception never escapes.

    The envelope is untrusted on-disk data: a truncated/partially-written file,
    a mangled field, or an unsupported future version must ALSO surface as a
    :class:`SecretDecryptError` (never a raw ``JSONDecodeError``/``KeyError``/
    ``binascii.Error``) so the CLI's single ``SecretDecryptError`` handler gives
    the actionable "wrong passphrase or corrupted" message instead of an opaque
    backstop error.
    """
    try:
        env = json.loads(blob)
        if env.get("v") != _ENVELOPE_VERSION or env.get("kdf") != "scrypt":
            raise SecretDecryptError(
                "Unsupported or corrupted session file (unexpected format version)."
            )
        salt = base64.b64decode(env["salt"])
        key = _derive_key(passphrase, salt, env["kdf_params"])
        ciphertext = env["ciphertext"].encode("ascii")
    except SecretDecryptError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError):
        # ValueError covers json.JSONDecodeError + binascii.Error (bad base64);
        # KeyError a missing envelope/param field; TypeError/AttributeError a
        # field of the wrong shape. All mean the file is corrupt, not that the
        # passphrase is wrong — but the secret-free message covers both.
        raise SecretDecryptError(
            "Wrong passphrase or the session file is corrupted."
        ) from None
    try:
        return Fernet(key).decrypt(ciphertext)
    except InvalidToken:
        raise SecretDecryptError(
            "Wrong passphrase or the session file is corrupted."
        ) from None


def write_encrypted(path: Path, passphrase: str, plaintext: bytes) -> None:
    """Encrypt ``plaintext`` and atomically write the ciphertext envelope to ``path``.

    Creates the parent app-data dir if missing, writes to a sibling temp file
    opened owner-only (``0o600`` on POSIX; a no-op on Windows), then
    :func:`os.replace` swaps it into place — an ATOMIC rename so an interrupted
    write (crash / power loss / disk-full) can never leave a half-written
    ``session.enc`` that would then fail to decrypt. Opening with ``0o600`` from
    the start closes the create-world-readable-then-narrow window. Only ciphertext
    ever reaches disk (AC#4).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = encrypt(passphrase, plaintext)
    tmp = path.with_name(path.name + ".tmp")
    # os.open with 0o600 sets the mode at creation (no perms window). O_TRUNC so a
    # leftover temp from a prior crash is overwritten, not appended to.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(envelope)
            fh.flush()
            os.fsync(fh.fileno())  # durability before the atomic swap
    except BaseException:
        # Never leave the temp file behind on a failed write.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)  # atomic on POSIX and Windows


def read_encrypted(path: Path, passphrase: str) -> bytes:
    """Read the ciphertext envelope at ``path`` and return the decrypted plaintext.

    Raises :class:`SecretDecryptError` on a wrong passphrase or corrupt file.
    """
    return decrypt(passphrase, path.read_bytes())


def session_exists(path: Path) -> bool:
    """Return whether an encrypted session file already exists at ``path``."""
    return path.is_file()


# --- Session payload bundle (Task 3) ------------------------------------------
#
# The plaintext encrypted above bundles BOTH secrets (session string + api_hash)
# plus api_id/phone, so ONE passphrase-protected file is the whole credential
# store: a reused session then needs no env vars at all on later runs (api_id +
# api_hash are required to reconstruct TelegramClient). AD-8 requires api_hash
# encrypted at rest, not just the session — bundling satisfies both with one
# passphrase prompt.


def dumps_payload(*, session: str, api_hash: str, api_id: int, phone: str) -> bytes:
    """Serialize the credential bundle to JSON bytes ready for :func:`encrypt`.

    All four fields are needed to reconstruct ``TelegramClient(session, api_id,
    api_hash)`` on a later run without any env vars. ``session`` and ``api_hash``
    are SECRETS — this returns their PLAINTEXT bytes, so the caller must feed the
    result straight into :func:`encrypt`/:func:`write_encrypted` and never log it.
    """
    return json.dumps(
        {"session": session, "api_hash": api_hash, "api_id": api_id, "phone": phone}
    ).encode("utf-8")


def loads_payload(blob: bytes) -> dict:
    """Parse + validate the credential bundle produced by :func:`dumps_payload`.

    Returns ``{"session", "api_hash", "api_id", "phone"}``. The returned dict
    holds SECRETS (session, api_hash) — never log it.

    The plaintext has already passed Fernet authentication by the time this runs,
    so a shape mismatch here means a schema-drifted bundle (an older/newer writer),
    not tampering. Validate the required keys/types and raise the typed
    :class:`SecretDecryptError` so the reuse path degrades to a clean, actionable
    error instead of a raw ``KeyError`` deep in the CLI.
    """
    try:
        payload = json.loads(blob)
        session = payload["session"]
        api_hash = payload["api_hash"]
        api_id = payload["api_id"]
        phone = payload["phone"]
    except (ValueError, KeyError, TypeError):
        raise SecretDecryptError(
            "The stored session payload is corrupted or has an unexpected format."
        ) from None
    if not (
        isinstance(session, str)
        and isinstance(api_hash, str)
        and isinstance(api_id, int)
        and isinstance(phone, str)
    ):
        raise SecretDecryptError(
            "The stored session payload is corrupted or has an unexpected format."
        )
    return {"session": session, "api_hash": api_hash, "api_id": api_id, "phone": phone}
