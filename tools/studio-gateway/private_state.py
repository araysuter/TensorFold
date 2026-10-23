"""Credential checks and ephemeral, tenant-scoped native snapshot storage."""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
import time
import uuid
from pathlib import Path

from config import KeyFile

from errors import GatewayError


@dataclasses.dataclass(frozen=True)
class Principal:
    identity: str
    label: str


class KeyStore:
    """Read current key files for every check, including checks after queuing.

    Identity is a digest of the actual credential. Reusing a display label or
    rotating a key cannot grant access to another key's parked contexts.
    """
    def __init__(self, key_files: tuple[KeyFile, ...]):
        self.key_files = key_files

    @staticmethod
    def _values(headers, name: str) -> list[str]:
        if hasattr(headers, "get_all"):
            return headers.get_all(name, [])
        return [value for key, value in headers.items() if key.lower() == name.lower()]

    @staticmethod
    def _read(key_file: KeyFile) -> list[tuple[str, str]]:
        fd = os.open(key_file.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
                    or info.st_uid != os.getuid() or info.st_size > 1024 * 1024):
                raise ValueError("unsafe key file")
            raw = os.read(fd, 1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("oversized key file")
        finally:
            os.close(fd)
        keys = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            label, token = key_file.label, line
            if ":" in line:
                sublabel, token = (value.strip() for value in line.split(":", 1))
                if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", sublabel):
                    raise ValueError("invalid key label")
                label += ":" + sublabel
            if not token or any(not 33 <= ord(char) <= 126 for char in token):
                raise ValueError("invalid key")
            keys.append((label, hashlib.sha256(token.encode()).hexdigest()))
        if not keys:
            raise ValueError("empty key file")
        return keys

    def authenticate(self, headers) -> Principal:
        auth, alternate = self._values(headers, "Authorization"), self._values(headers, "x-api-key")
        if len(auth) > 1 or len(alternate) > 1:
            raise GatewayError(401, "Invalid API key", "invalid_api_key")
        candidates = []
        if auth:
            parts = auth[0].split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise GatewayError(401, "Invalid API key", "invalid_api_key")
            candidates.append(parts[1])
        if alternate:
            candidates.append(alternate[0])
        if not candidates or len(set(candidates)) != 1:
            raise GatewayError(401, "Invalid API key", "invalid_api_key")
        token = candidates[0]
        if not token or any(not 33 <= ord(char) <= 126 for char in token):
            raise GatewayError(401, "Invalid API key", "invalid_api_key")
        wanted = hashlib.sha256(token.encode()).hexdigest()
        match = None
        for key_file in self.key_files:
            try:
                rows = self._read(key_file)
            except FileNotFoundError:
                if not key_file.required:
                    continue
                raise GatewayError(401, "API key configuration unavailable", "invalid_api_key") from None
            except (OSError, UnicodeError, ValueError):
                raise GatewayError(401, "API key configuration unavailable", "invalid_api_key") from None
            for label, digest in rows:
                if hmac.compare_digest(wanted, digest):
                    match = Principal(digest, label)
        if match is None:
            raise GatewayError(401, "Invalid API key", "invalid_api_key")
        return match


class Workspace:
    """An exclusive private run directory; stale runs are cleared after crashes."""
    MARKER = b"tensorfold-studio-gateway-ephemeral-v1\n"

    def __init__(self, root: Path):
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("cache_root must be an owned real directory with mode 700")
        self.lock_fd = os.open(root / ".gateway.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            raise ValueError("another gateway owns this cache_root") from None
        self.path = root / ("run-" + uuid.uuid4().hex)
        try:
            for old in root.glob("run-*"):
                if (re.fullmatch(r"run-[0-9a-f]{32}", old.name) and old.is_dir()
                        and not old.is_symlink() and (old / ".gateway-run").is_file()
                        and not (old / ".gateway-run").is_symlink()
                        and (old / ".gateway-run").read_bytes() == self.MARKER):
                    shutil.rmtree(old)
            self.path.mkdir(mode=0o700)
            (self.path / ".gateway-run").write_bytes(self.MARKER)
            self.contexts = self.path / "contexts"
            self.contexts.mkdir(mode=0o700)
            self.backend_key = secrets.token_urlsafe(48)
            self.backend_key_file = self.path / "backend-key"
            fd = os.open(self.backend_key_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write("gateway:" + self.backend_key + "\n")
            self.closed = False
        except BaseException:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            shutil.rmtree(self.path)
        finally:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)


@dataclasses.dataclass
class Conversation:
    model: str
    owner: str
    id: str
    path: Path
    prefixes: tuple[str, ...] = ()
    touched: float = dataclasses.field(default_factory=time.monotonic)
    saved: bool = False
    dirty: bool = False
    tokens: int = 0
    bytes: int = 0
    fingerprint: str | None = None


class ContextStore:
    """No transcripts: only opaque affinity hashes and complete native snapshots.

    There are at most four parked conversations per model, plus its most recent
    active conversation (which can also be on SSD while the model is idle).
    """
    def __init__(self, root: Path, max_parked: int = 4):
        self.root, self.max_parked = root, max_parked
        self.salt = secrets.token_bytes(32)
        self.rows: dict[tuple[str, str, str], Conversation] = {}

    def _hash(self, value: object) -> str:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hmac.new(self.salt, raw, hashlib.sha256).hexdigest()

    def _prefixes(self, payload: dict) -> tuple[str, ...]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            # Responses and raw completions may use explicit context IDs; exact
            # request input affinity is a safe fallback for these endpoints.
            return (self._hash(payload.get("input", payload.get("prompt", ""))),)
        digest = hmac.new(self.salt, digestmod=hashlib.sha256)
        result = []
        for message in messages:
            digest.update(json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
            digest.update(b"\0")
            # A system prompt alone cannot associate unrelated conversations.
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
                result.append(digest.hexdigest())
        return tuple(result)

    def select(self, model: str, owner: str, payload: dict, context_id: str | None = None) -> Conversation:
        prefixes = self._prefixes(payload)
        selected = None
        if context_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", context_id):
                raise GatewayError(400, "X-TensorFold-Context needs 1-128 letters, digits, dots, underscores or hyphens", "invalid_context")
            identity = self._hash([model, owner, context_id])
            selected = self.rows.get((model, owner, identity))
        else:
            candidates = []
            for row in self.rows.values():
                if row.owner != owner or row.model != model:
                    continue
                common = 0
                for before, after in zip(row.prefixes, prefixes):
                    if before != after:
                        break
                    common += 1
                if common:
                    candidates.append((common, row.touched, row))
            if candidates:
                selected = max(candidates, key=lambda value: value[:2])[2]
            identity = selected.id if selected else uuid.uuid4().hex
        if selected is None:
            name = self._hash([model, owner, identity]) + ".tfs"
            selected = Conversation(model, owner, identity, self.root / name)
            self.rows[(model, owner, identity)] = selected
        selected.prefixes = prefixes
        selected.touched = time.monotonic()
        return selected

    def prune(self, model: str, protected: Conversation | None = None):
        candidates = sorted((row for row in self.rows.values() if row.model == model), key=lambda row: row.touched)
        while len(candidates) > self.max_parked + 1:
            row = next((entry for entry in candidates if entry is not protected), None)
            if row is None:
                break
            candidates.remove(row)
            self.rows.pop((row.model, row.owner, row.id), None)
            row.path.unlink(missing_ok=True)

    def discard_unused(self, row: Conversation, active: Conversation | None):
        if row is not active and not row.saved and not row.dirty:
            self.rows.pop((row.model, row.owner, row.id), None)
            row.path.unlink(missing_ok=True)

