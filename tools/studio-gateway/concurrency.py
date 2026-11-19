"""RAM-reserved stateless request leases sharing the existing model lifecycle."""
from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import threading
import time

from errors import GatewayError
import memory
from runtime import ModelManager


class RequestBackend:
    """A cancellation handle for one HTTP request, never the shared model process."""

    context_status = "cold"

    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self.connection = None
        self.cancelled = False

    @contextlib.contextmanager
    def request(self, method, path, payload=None):
        connection = self.backend._connection()
        with self.lock:
            if self.cancelled:
                raise GatewayError(499, "Request cancelled", "request_cancelled")
            self.connection = connection
        try:
            body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
            connection.request(method, path, body=body, headers=self.backend._headers())
            with self.lock:
                cancelled = self.cancelled
            if cancelled:
                self.stop()
                raise GatewayError(499, "Request cancelled", "request_cancelled")
            yield connection.getresponse()
        finally:
            with self.lock:
                self.connection = None
            connection.close()

    def stop(self):
        with self.lock:
            self.cancelled = True
            connection = self.connection
            if connection and connection.sock:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if connection:
                connection.close()


class ConcurrentManager(ModelManager):
    def __init__(self, *args, **kwargs):
        self.users = 0
        self.group_model = None
        self.loading = False
        self.base_footprint = 0
        super().__init__(*args, **kwargs)

    def _reserve(self, model):
        return max(self.config.memory_reserve_gib, model.request_reserve_gib)

    def _check_slot(self, model, count):
        try:
            available, pressure = memory.snapshot()
            footprint = memory.group_footprint(self.backend.process.pid)
            after, after_pressure = memory.snapshot()
            available = min(available, after)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            raise GatewayError(503, "Memory status unavailable; retry later", "memory_status_unavailable") from None
        self.base_footprint = min(self.base_footprint, footprint)
        allocated = max(0, footprint - self.base_footprint)
        growth = max(0, int(count * model.memory_per_request_gib * memory.GIB) - allocated)
        required = int(self._reserve(model) * memory.GIB) + growth
        if pressure != 1 or after_pressure != 1 or available < required:
            raise GatewayError(503, "Insufficient RAM for another full-context request", "insufficient_memory")

    def _start_stateless(self, model, cancel):
        if self.backend and not self.backend.alive():
            self._unload(save=False)
        if self.backend and self.backend.model.id != model.id:
            memory.check_memory(model, self._reserve(model), floor_only=True)
            self._unload(save=True)
        if self.backend is None:
            memory.check_memory(model, self._reserve(model))
            if self.closing or cancel():
                raise GatewayError(499, "Request cancelled", "request_cancelled")
            self.backend = self.backend_factory(self.config, model, self.workspace)
            try:
                self.backend.start(lambda: self.closing or cancel())
                memory.check_memory(model, self._reserve(model), floor_only=True)
            except BaseException:
                self._unload(save=False)
                raise
        self.active = None
        try:
            self.base_footprint = memory.group_footprint(self.backend.process.pid)
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            raise GatewayError(503, "Memory status unavailable; retry later", "memory_status_unavailable") from None

    @contextlib.contextmanager
    def lease(self, model_id, owner, payload=None, context_id=None, cancel=lambda: False,
              recheck=None, request_path="/v1/chat/completions"):
        model = self.model(model_id)
        if not model.stateless:
            with super().lease(model_id, owner, payload, context_id, cancel, recheck, request_path) as backend:
                yield backend
            return
        token = object()
        deadline = time.monotonic() + self.config.queue_timeout
        leader = False
        with self.condition:
            if len(self.queue) >= self.config.max_pending:
                raise GatewayError(429, "Model request queue is full", "queue_full")
            self.queue.append(token)
            try:
                while True:
                    if self.closing or cancel():
                        raise GatewayError(499, "Request cancelled", "request_cancelled")
                    if time.monotonic() >= deadline:
                        raise GatewayError(503, "Model request queue timed out", "queue_timeout")
                    if self.queue[0] is token:
                        if not self.busy:
                            self.busy = True
                            self.users = 1
                            self.group_model = model.id
                            self.loading = True
                            leader = True
                            break
                        if self.group_model == model.id and not self.loading:
                            if self.users >= model.max_concurrent:
                                raise GatewayError(429, "All model request slots are in use", "concurrency_limit")
                            if not self.backend or not self.backend.alive():
                                raise GatewayError(503, "Model process exited", "backend_start_failed")
                            if recheck:
                                recheck()
                            self._check_slot(model, self.users + 1)
                            self.users += 1
                            break
                    self.condition.wait(timeout=0.1)
            finally:
                self.queue.remove(token)
                self.condition.notify_all()
        try:
            if leader:
                try:
                    if recheck:
                        recheck()
                    self._start_stateless(model, cancel)
                    with self.condition:
                        self._check_slot(model, 1)
                        self.loading = False
                        self.condition.notify_all()
                except BaseException:
                    if self.backend and self.backend.model.id == model.id:
                        self._unload(save=False)
                    raise
            if cancel() or self.closing:
                raise GatewayError(499, "Request cancelled", "request_cancelled")
            if recheck:
                recheck()
            yield RequestBackend(self.backend)
        finally:
            with self.condition:
                self.users -= 1
                if not self.users:
                    self.group_model = None
                    self.loading = False
                    self.busy = False
                    self.last_used = time.monotonic()
                self.condition.notify_all()

    def cancel_active(self, backend=None):
        if isinstance(backend, RequestBackend):
            backend.stop()
        else:
            super().cancel_active()

    def status(self):
        with self.condition:
            result = super().status()
            model = self.backend.model if self.backend else self.model(None)
            result.update(context_window=model.context_window,
                          active_requests=self.users if model.stateless else int(self.busy),
                          max_concurrent=model.max_concurrent)
            return result
