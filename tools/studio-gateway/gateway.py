#!/usr/bin/env python3
"""One authenticated API, one native model process, private SSD conversations.

Only the gateway owns the public loopback port. Native engines receive an
ephemeral internal credential and can export/import complete execution state.
No prompt, response, API key, or conversation identifier is logged here.
"""
from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

from config import GatewayConfig, KeyFile, ModelConfig, load_config, validate

from errors import GatewayError
from private_state import Principal, KeyStore, Workspace, Conversation, ContextStore
from runtime import NativeBackend, ModelManager
from stats import read_stats


def client_gone(connection: socket.socket) -> bool:
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        return bool(readable) and connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except (OSError, ValueError):
        return True


class GatewayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TensorFoldGateway/1"
    sys_version = ""

    def log_message(self, *_args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(15)
        self.response_started = False

    def _json(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.response_started = True
        self.close_connection = True
        self.wfile.write(raw)

    def _error(self, exc: GatewayError):
        if not self.response_started:
            self._json(exc.status if exc.status != 499 else 408,
                       {"error": {"message": exc.message, "type": "invalid_request_error" if exc.status < 500 else "server_error",
                                  "code": exc.code}})

    def _host(self):
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            raise GatewayError(400, "A single Host header is required", "invalid_host")
        host = hosts[0].lower()
        allowed = {"localhost", "127.0.0.1", *[entry.lower() for entry in self.server.config.allowed_hosts]}
        base = host.rsplit(":", 1)[0] if ":" in host else host
        if base not in allowed:
            raise GatewayError(403, "Host is not allowed", "invalid_host")

    def _authorize(self):
        return self.server.keys.authenticate(self.headers)

    def handle_expect_100(self):
        try:
            self._host()
            self._authorize()
            self._length()
        except GatewayError as exc:
            self._error(exc)
            return False
        self.send_response_only(100)
        self.end_headers()
        return True

    def _length(self) -> int:
        if self.headers.get_all("Transfer-Encoding", []):
            raise GatewayError(400, "Transfer-Encoding is unsupported; send Content-Length", "invalid_framing")
        values = self.headers.get_all("Content-Length", [])
        if len(values) != 1 or not re.fullmatch(r"[0-9]+", values[0]):
            raise GatewayError(411, "A single Content-Length header is required", "invalid_framing")
        if len(values[0]) > 12:
            raise GatewayError(413, "Request body is too large", "body_too_large")
        size = int(values[0])
        if size > self.server.config.max_body_bytes:
            raise GatewayError(413, "Request body is too large", "body_too_large")
        return size

    def _path(self) -> str:
        split = urlsplit(self.path)
        if split.scheme or split.netloc or split.query or split.fragment or "%" in split.path:
            raise GatewayError(404, "Route not found", "not_found")
        return split.path.rstrip("/") or "/"

    def do_GET(self):
        try:
            self._host()
            path = self._path()
            if path in {"/health", "/v1/health"}:
                self._json(200, self.server.manager.status())
                return
            principal = self._authorize()
            if path in {"/v1/models", "/models"}:
                self._json(200, {"object": "list", "data": [
                    {"id": model.id, "object": "model", "owned_by": "local", "created": 0}
                    for model in self.server.config.models]})
            elif path in {"/stats", "/v1/stats"}:
                # Collector traffic must not pin the model in RAM or load one.
                # The native stats endpoint is read-only and safe during work.
                backend = self.server.manager.backend
                if backend is None:
                    self._json(200, {"gateway": self.server.manager.status(), "models": []})
                else:
                    result = read_stats(backend)
                    result["gateway"] = self.server.manager.status()
                    self._json(200, result)
            else:
                raise GatewayError(404, "Route not found", "not_found")
        except GatewayError as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except TimeoutError:
            self._error(GatewayError(504, "Backend request timed out", "backend_timeout"))

    def _payload(self, path: str) -> dict:
        size = self._length()
        data = self.rfile.read(size)
        if len(data) != size:
            raise GatewayError(400, "Incomplete request body", "invalid_body")
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeError, RecursionError):
            raise GatewayError(400, "Body must be valid JSON", "invalid_body") from None
        if not isinstance(payload, dict):
            raise GatewayError(400, "Body must be a JSON object", "invalid_body")
        if "model" in payload and not isinstance(payload["model"], str):
            raise GatewayError(400, "Model must be a string", "invalid_model")
        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = payload.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 131072):
                raise GatewayError(400, "Output token limit must fit the 128K context window", "invalid_token_limit")
        if "stream" in payload and not isinstance(payload["stream"], bool):
            raise GatewayError(400, "Stream must be a boolean", "invalid_stream")
        if payload.get("n", 1) != 1:
            raise GatewayError(400, "Only one completion per request is supported", "invalid_n")
        if path == "/v1/responses":
            # Native response IDs live in process memory and cannot survive a
            # model swap. Require clients to send their complete input history.
            if payload.get("previous_response_id") or payload.get("background"):
                raise GatewayError(400, "Send full input history; stored response chains and background jobs are unsupported", "unsupported_response_state")
            payload["store"] = False
        return payload

    def do_POST(self):
        try:
            self._host()
            principal = self._authorize()  # before body read, queuing, or model loading
            path = self._path()
            if path not in {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages"}:
                raise GatewayError(404, "Route not found", "not_found")
            payload = self._payload(path)
            model = self.server.manager.model(payload.get("model"))
            payload["model"] = model.id
            context_values = self.headers.get_all("X-TensorFold-Context", [])
            if len(context_values) > 1:
                raise GatewayError(400, "A single X-TensorFold-Context header is allowed", "invalid_context")
            context_id = context_values[0] if context_values else None

            def recheck():
                current = self._authorize()
                if current.identity != principal.identity:
                    raise GatewayError(401, "API key changed while queued", "invalid_api_key")

            self.connection.settimeout(self.server.config.request_timeout)
            with self.server.manager.lease(model.id, principal.identity, payload, context_id,
                                           cancel=lambda: client_gone(self.connection), recheck=recheck,
                                           request_path=path) as backend:
                self._proxy(backend, path, payload)
        except GatewayError as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except TimeoutError:
            self._error(GatewayError(504, "Backend request timed out", "backend_timeout"))
        except (OSError, http.client.HTTPException, ValueError):
            self._error(GatewayError(502, "Backend connection failed", "backend_error"))

    def _proxy(self, backend: NativeBackend, path: str, payload: dict):
        finished = False
        # getresponse()/read() may block during a long reasoning-only response.
        # A companion watcher closes the backend and terminates generation when
        # the public client disappears, even before the first output byte.
        done = threading.Event()
        disconnected = threading.Event()

        def watch():
            while not done.wait(0.1):
                if client_gone(self.connection) or self.server.manager.closing:
                    disconnected.set()
                    try:
                        backend.stop()
                    except Exception:
                        pass
                    return

        watcher = threading.Thread(target=watch, name="gateway-disconnect", daemon=True)
        watcher.start()
        try:
            with backend.request("POST", path, payload) as response:
                if not 200 <= response.status < 300:
                    # Native error bodies can contain internal paths or, in a
                    # faulty backend, credentials. Never expose them publicly.
                    response.read(1024 * 1024)
                    finished = True
                    raise GatewayError(response.status if 400 <= response.status <= 599 else 502,
                                       "The native backend rejected this request", "backend_rejected_request")
                is_stream = bool(payload.get("stream")) and 200 <= response.status < 300
                content_type = response.getheader("Content-Type", "application/json")
                if not is_stream:
                    raw = response.read(32 * 1024 * 1024 + 1)
                    if len(raw) > 32 * 1024 * 1024:
                        raise GatewayError(502, "Backend response is too large", "backend_error")
                    self.send_response(response.status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.send_header("X-TensorFold-Context-Status", backend.context_status)
                    self.end_headers()
                    self.response_started = True
                    self.close_connection = True
                    self.wfile.write(raw)
                    self.wfile.flush()
                else:
                    self.send_response(response.status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Connection", "close")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("X-TensorFold-Context-Status", backend.context_status)
                    self.end_headers()
                    self.response_started = True
                    self.close_connection = True
                    while True:
                        chunk = response.read1(16384)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                finished = True
                if 200 <= response.status < 300:
                    self.server.manager.mark_used()
        finally:
            done.set()
            watcher.join(timeout=self.server.config.stop_timeout + 2)
            if not finished or disconnected.is_set():
                self.server.manager.cancel_active()


class GatewayServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, config: GatewayConfig):
        self.config = validate(config)
        self.keys = KeyStore(config.key_files)
        self.manager = ModelManager(config)
        self.connections = threading.BoundedSemaphore(config.max_pending + 8)
        try:
            super().__init__((config.host, config.port), GatewayHandler)
        except BaseException:
            self.manager.close()
            raise

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def handle_error(self, *_args):
        # The default traceback can include headers/request payloads via third
        # party exceptions. Keep private runtime errors out of public terminals.
        print("gateway: request handler failed", file=sys.stderr, flush=True)

    def server_close(self):
        try:
            self.manager.close()
        finally:
            super().server_close()


def create_server(config: GatewayConfig) -> GatewayServer:
    return GatewayServer(config)


def supervise(parent_fd: int, command: list[str]) -> int:
    """Kill the whole engine group when its owning gateway exits, even SIGKILL.

    This small supervisor holds the read end of a pipe whose sole writer is the
    gateway. Native children never inherit that pipe, so EOF is reliable.
    """
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda *_: stopping.set())
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True)
    try:
        while process.poll() is None and not stopping.is_set():
            ready, _, _ = select.select([parent_fd], [], [], 0.1)
            if ready and os.read(parent_fd, 1) == b"":
                stopping.set()
        if stopping.is_set():
            # Ignore our own broadcast, while every engine descendant receives
            # TERM. A subsequent KILL guarantees the group cannot outlive us.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.killpg(os.getpgrp(), signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgrp(), signal.SIGKILL)
        return process.wait()
    finally:
        os.close(parent_fd)


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--supervise":
        command = sys.argv[3:]
        if command and command[0] == "--":
            command = command[1:]
        return supervise(int(sys.argv[2]), command)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.check_config:
            # Validate credential ownership/mode without revealing their values.
            for key_file in config.key_files:
                try:
                    KeyStore._read(key_file)
                except FileNotFoundError:
                    if key_file.required:
                        raise
            print("Gateway configuration and key files are valid")
            return 0
        os.umask(0o077)
        server = create_server(config)
    except (OSError, ValueError, KeyError, TypeError):
        print("gateway: invalid configuration, credential file, or private state directory", file=sys.stderr)
        return 2

    stopping = threading.Event()

    def stop(*_args):
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    print(f"gateway: listening on 127.0.0.1:{server.server_address[1]}; {len(config.models)} models available on demand", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
