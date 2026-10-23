"""Serialized native model lifecycle, supervision, and conversation switching."""
from __future__ import annotations

import collections
import contextlib
import hashlib
import http.client
import http.server
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from config import GatewayConfig, ModelConfig, validate

from errors import GatewayError
from private_state import Workspace, ContextStore, Conversation


_BUILD_HASHES: dict[tuple[str, int, int], str] = {}


def build_identity(executable: str) -> str:
    path = Path(executable).resolve(strict=True)
    before = path.stat()
    key = (str(path), before.st_size, before.st_mtime_ns)
    if key not in _BUILD_HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(part)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise GatewayError(503, "Model executable changed during startup", "backend_start_failed")
        _BUILD_HASHES[key] = digest.hexdigest()
    return _BUILD_HASHES[key]



class NativeBackend:
    def __init__(self, config: GatewayConfig, model: ModelConfig, workspace: Workspace):
        self.config, self.model, self.workspace = config, model, workspace
        self.process: subprocess.Popen | None = None
        self.parent_fd: int | None = None
        self.stop_lock = threading.Lock()
        self.context_status = "cold"

    def alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def _connection(self, timeout: float | None = None) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.config.backend_port,
                                          timeout=timeout or self.config.request_timeout)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer " + self.workspace.backend_key, "Content-Type": "application/json",
                "Connection": "close"}

    def start(self, cancel: Callable[[], bool]):
        # Never take over or authenticate to an unrelated process on this port.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.config.backend_port)) == 0:
                raise GatewayError(503, "Private backend port is already in use", "backend_port_busy")
        substitutions = {"port": str(self.config.backend_port), "backend_port": str(self.config.backend_port),
                         "backend_key_file": str(self.workspace.backend_key_file),
                         "snapshot_dir": str(self.workspace.contexts)}
        argv = list(self.model.argv)
        for index, arg in enumerate(argv):
            for key, value in substitutions.items():
                arg = arg.replace("{" + key + "}", value)
            argv[index] = str(Path(arg).expanduser()) if arg.startswith("~/") else arg
        allowed_env = {"PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"}
        env = {key: value for key, value in os.environ.items() if key in allowed_env}
        env.update(self.model.env)
        env["TF_SESSION_ROOT"] = str(self.workspace.contexts)
        env["TF_SESSION_BUILD_ID"] = build_identity(argv[0])
        child_fd, self.parent_fd = os.pipe()
        os.set_inheritable(child_fd, True)
        log_fd = os.open(self.workspace.path / "engine.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        command = [sys.executable, str(Path(__file__).with_name("gateway.py").resolve()), "--supervise", str(child_fd), "--", *argv]
        try:
            self.process = process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd,
                                                      pass_fds=(child_fd,), start_new_session=True, env=env,
                                                      cwd=str(Path(argv[0]).parent))
        except BaseException:
            os.close(self.parent_fd)
            self.parent_fd = None
            raise
        finally:
            os.close(child_fd)
            os.close(log_fd)
        deadline = time.monotonic() + self.model.readiness_timeout
        try:
            while time.monotonic() < deadline:
                if cancel():
                    raise GatewayError(499, "Request cancelled", "request_cancelled")
                if process.poll() is not None:
                    raise GatewayError(503, "Model process exited during startup; inspect the private runtime log", "backend_start_failed")
                conn = self._connection(0.5)
                try:
                    conn.request("GET", "/health", headers=self._headers())
                    response = conn.getresponse()
                    response.read(1024 * 1024)
                    if response.status == 200:
                        # A successful models request proves the newly generated
                        # private credential is recognized by this exact engine.
                        conn.close()
                        conn = self._connection(0.5)
                        conn.request("GET", "/v1/models", headers=self._headers())
                        authenticated = conn.getresponse()
                        authenticated.read(1024 * 1024)
                        if authenticated.status == 200:
                            return
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    conn.close()
                time.sleep(0.1)
            raise GatewayError(503, "Model startup timed out", "backend_start_timeout")
        except BaseException:
            self.stop()
            raise

    def control(self, action: str, path: Path | None = None) -> dict:
        conn = self._connection()
        try:
            body = json.dumps({"path": str(path)} if path is not None else {}).encode()
            conn.request("POST", "/v1/internal/session/" + action, body, self._headers())
            response = conn.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise GatewayError(503, "Native conversation state operation failed", "snapshot_error")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("invalid control response")
            return result
        except (OSError, ValueError, http.client.HTTPException):
            raise GatewayError(503, "Native conversation state operation failed", "snapshot_error") from None
        finally:
            conn.close()

    def can_resume(self, payload: dict) -> bool:
        # This is the native renderer's own token-prefix check, before inference.
        # Message-level affinity alone cannot distinguish template changes,
        # branches, or an unchanged prompt shorter than its saved response.
        with self.request("POST", "/v1/imprint/resume", payload) as response:
            raw = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise GatewayError(503, "Native conversation preflight failed", "snapshot_error")
            try:
                result = json.loads(raw)
                tokens = result["ram_tokens"]
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                    raise ValueError("invalid resume receipt")
            except (ValueError, KeyError, TypeError):
                raise GatewayError(503, "Native conversation preflight failed", "snapshot_error") from None
            return tokens > 0

    @contextlib.contextmanager
    def request(self, method: str, path: str, payload: dict | None = None):
        conn = self._connection()
        try:
            body = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            conn.request(method, path, body=body, headers=self._headers())
            yield conn.getresponse()
        finally:
            conn.close()

    def stop(self):
        with self.stop_lock:
            self._stop()

    def _stop(self):
        if self.parent_fd is not None:
            os.close(self.parent_fd)
            self.parent_fd = None
        process = self.process
        if process is None:
            return
        pgid = process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=self.config.stop_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        # A launcher may have forked descendants. Close their process group too;
        # no next engine may start while the previous generation can run.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", self.config.backend_port)) != 0:
                    break
            time.sleep(0.05)
        else:
            raise GatewayError(503, "Old backend has not released its port", "backend_stop_failed")
        self.process = None
        time.sleep(self.config.release_delay)


class ModelManager:
    def __init__(self, config: GatewayConfig, backend_factory=NativeBackend):
        self.config = validate(config)
        self.workspace = Workspace(config.cache_root)
        self.contexts = ContextStore(self.workspace.contexts, config.max_parked)
        self.backend_factory = backend_factory
        self.models = {name: model for model in config.models for name in (model.id, *model.aliases)}
        self.backend: NativeBackend | None = None
        self.active: Conversation | None = None
        self.last_used = time.monotonic()
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.busy = False
        self.closing = False
        self.closed = False
        self.close_lock = threading.Lock()
        self.idle_thread = threading.Thread(target=self._idle_loop, name="gateway-idle", daemon=True)
        self.idle_thread.start()

    def model(self, name: str | None) -> ModelConfig:
        model = self.models.get(name or self.config.default_model)
        if model is None:
            raise GatewayError(404, "Model is not available", "model_not_found")
        return model

    @contextlib.contextmanager
    def _exclusive(self, cancel: Callable[[], bool], recheck: Callable[[], None] | None = None):
        token = object()
        deadline = time.monotonic() + self.config.queue_timeout
        with self.condition:
            if self.closing:
                raise GatewayError(503, "Gateway is shutting down", "gateway_closing")
            if len(self.queue) >= self.config.max_pending:
                raise GatewayError(429, "Model request queue is full", "queue_full")
            self.queue.append(token)
            try:
                while self.busy or self.queue[0] is not token:
                    if self.closing or cancel():
                        raise GatewayError(499, "Request cancelled", "request_cancelled")
                    if time.monotonic() >= deadline:
                        raise GatewayError(503, "Model request queue timed out", "queue_timeout")
                    self.condition.wait(timeout=0.1)
                if self.closing:
                    raise GatewayError(503, "Gateway is shutting down", "gateway_closing")
                if cancel():
                    raise GatewayError(499, "Request cancelled", "request_cancelled")
                if recheck:
                    recheck()
                self.busy = True
            finally:
                self.queue.remove(token)
                self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.last_used = time.monotonic()
                self.condition.notify_all()

    def _save(self):
        if not self.backend or not self.active or not self.active.dirty:
            return
        result = self.backend.control("save", self.active.path)
        tokens, size = result.get("tokens"), result.get("bytes")
        if (not isinstance(tokens, int) or tokens < 0 or not isinstance(size, int) or size < 0
                or not isinstance(result.get("fingerprint"), str)):
            raise GatewayError(503, "Native snapshot receipt is invalid", "snapshot_error")
        if tokens and size:
            try:
                info = self.active.path.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o077 or info.st_size != size):
                    raise ValueError("unsafe snapshot")
            except (OSError, ValueError):
                raise GatewayError(503, "Native snapshot was not saved safely", "snapshot_error") from None
            self.active.saved = True
            self.active.tokens, self.active.bytes = tokens, size
            self.active.fingerprint = result["fingerprint"]
        else:
            # A request rejected by the native engine can leave no state. It is
            # allowed to proceed cold next time, and is never claimed resumable.
            self.active.saved = False
        self.active.dirty = False

    def _unload(self, save: bool = True):
        backend = self.backend
        if backend:
            if save:
                self._save()
            backend.stop()
            if self.backend is backend:
                self.backend = None
        self.active = None

    @contextlib.contextmanager
    def lease(self, model_id: str, owner: str, payload: dict | None = None,
              context_id: str | None = None, cancel: Callable[[], bool] = lambda: False,
              recheck: Callable[[], None] | None = None,
              request_path: str = "/v1/chat/completions"):
        model = self.model(model_id)
        with self._exclusive(cancel, recheck):
            if recheck:
                recheck()
            wanted = self.contexts.select(model.id, owner, payload or {}, context_id)
            try:
                self._activate(model, wanted, payload or {}, request_path, cancel)
            except BaseException:
                self.contexts.discard_unused(wanted, self.active)
                self.contexts.prune(model.id, self.active)
                raise
            self.contexts.prune(model.id, self.active)
            yield self.backend

    def _activate(self, model: ModelConfig, wanted: Conversation, payload: dict,
                  request_path: str, cancel: Callable[[], bool]):
        if self.backend and not self.backend.alive():
            # A crashed engine cannot export its most recent RAM-only state.
            # The last verified SSD snapshot stays intact; a client carrying its
            # complete history can safely resume that prefix or start cold.
            if self.active:
                self.active.dirty = False
            self._unload(save=False)
        if self.backend and self.backend.model.id != model.id:
            self._unload(save=True)
        if self.backend is None:
            if cancel() or self.closing:
                raise GatewayError(499, "Request cancelled", "request_cancelled")
            self.backend = self.backend_factory(self.config, model, self.workspace)
            try:
                self.backend.start(lambda: self.closing or cancel())
            except BaseException:
                # Keep a failed, unreaped child owned here. Any later switch
                # must stop/reap it before another engine can be launched.
                if self.backend.process is None:
                    self.backend = None
                raise
        self.backend.context_status = "resident" if self.active is wanted else "cold"
        already_active = self.active is wanted
        if self.active is not wanted:
            self._save()
            try:
                # Import requires an empty cache, and an explicit clear
                # also guarantees isolation between API key principals.
                self.backend.control("clear")
                if wanted.saved and request_path == "/v1/chat/completions":
                    receipt = self.backend.control("load", wanted.path)
                    if (receipt.get("fingerprint") != wanted.fingerprint
                            or receipt.get("tokens") != wanted.tokens
                            or receipt.get("bytes") != wanted.bytes):
                        raise GatewayError(503, "Conversation snapshot identity mismatch", "snapshot_error")
                    if self.backend.can_resume(payload):
                        self.backend.context_status = "resumed"
                    else:
                        # An intentional branch/shorter retry begins cold. The
                        # old full snapshot was valid; its token prefix simply
                        # does not belong to this new prompt. No generation has
                        # happened, so clearing does not repeat any attempt.
                        self.backend.control("clear")
                self.active = wanted
            except BaseException:
                # A failed import must not leave the previous tenant's state
                # accidentally active. Releasing the process is definitive.
                self._unload(save=False)
                raise
        elif already_active and wanted.dirty and request_path == "/v1/chat/completions":
            # Explicit IDs can be reused for a branch as well as continuations.
            # Keep the old checkpoint before replacing it with a new cold branch.
            if not self.backend.can_resume(payload):
                self._save()
                # The canonical warm pass can still be pending when its HTTP
                # reply finishes. Save waits for that native barrier; only a
                # second miss now proves this is a genuinely different prefix.
                if not self.backend.can_resume(payload):
                    self.backend.control("clear")
                    self.backend.context_status = "cold"

    def mark_used(self):
        if self.active:
            self.active.dirty = True
            self.active.touched = time.monotonic()

    def cancel_active(self):
        # Cancellation may leave native generation in flight after HTTP closes.
        # Stop it before releasing the global lease. Its earlier SSD checkpoint
        # remains available; an incomplete response is never advertised as saved.
        self._unload(save=False)

    def status(self) -> dict:
        with self.condition:
            return {"status": "stopping" if self.closing else "ok", "busy": self.busy,
                    "model": self.backend.model.id if self.backend else None,
                    "loaded_models": 1 if self.backend else 0, "queued": len(self.queue),
                    "context_window": 131072, "idle_unload_seconds": self.config.idle_seconds}

    def _idle_loop(self):
        interval = min(5.0, max(0.05, self.config.idle_seconds / 4))
        while True:
            with self.condition:
                if self.closing:
                    return
                due = not self.busy and not self.queue and self.backend is not None and time.monotonic() - self.last_used >= self.config.idle_seconds
                if not due:
                    self.condition.wait(timeout=interval)
                    continue
                self.busy = True
            try:
                self._unload(save=True)
            except Exception:
                # Keep the active model if its state cannot be saved. No prompt
                # or exception details (which could contain paths) reach logs.
                print("gateway: idle unload could not save or stop its backend", file=sys.stderr, flush=True)
            finally:
                with self.condition:
                    self.last_used = time.monotonic()
                    self.busy = False
                    self.condition.notify_all()

    def close(self):
        with self.close_lock:
            self._close()

    def _close(self):
        with self.condition:
            if self.closed:
                return
            self.closing = True
            self.condition.notify_all()
        # Shutdown deliberately discards all contexts. Terminating the backend
        # also interrupts a currently running request instead of waiting hours.
        try:
            backend = self.backend
            if backend:
                backend.stop()
            deadline = time.monotonic() + self.config.stop_timeout + 10
            with self.condition:
                while self.busy and time.monotonic() < deadline:
                    self.condition.wait(timeout=0.05)
            self._unload(save=False)
        finally:
            self.idle_thread.join(timeout=self.config.stop_timeout + 2)
            self.workspace.close()
            self.closed = True

