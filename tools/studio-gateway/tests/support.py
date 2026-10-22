"""CPU-only integration harness for the Studio gateway."""
from __future__ import annotations

from contextlib import contextmanager
import http.client
import importlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))

OWNER_KEY = "tf-test-owner-9a685259"
FRIEND_KEY = "tf-test-friend-589ee41e"


def available_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def private_file(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o600)
    return path


def eventually(assertion, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            assertion()
            return
        except AssertionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


class Harness:
    def __init__(self, root: Path, *, model_options=None, readiness_timeout=3, **overrides):
        config = importlib.import_module("config")
        gateway = importlib.import_module("gateway")
        self.root = root
        self.events = root / "events.jsonl"
        self.owner_key_file = private_file(root / "owner.key", OWNER_KEY + "\n")
        self.friend_key_file = private_file(root / "friend.key", FRIEND_KEY + "\n")
        self.state = root / "state"
        self.backend_port = available_port()
        model_options = model_options or {}
        models = tuple(config.ModelConfig(
            id=model, aliases=(model + "-alias",),
            argv=(sys.executable, str(DIRECTORY / "fake_backend.py"),
                  "--port", str(self.backend_port), "--label", model,
                  "--events", str(self.events), "--api-key-file", "{backend_key_file}",
                  *model_options.get(model, ())),
            readiness_timeout=(readiness_timeout.get(model, 3)
                               if isinstance(readiness_timeout, dict) else readiness_timeout),
        ) for model in ("swift", "ornith"))
        values = dict(host="127.0.0.1", port=0, backend_port=self.backend_port,
                      key_files=(config.KeyFile("owner", self.owner_key_file),
                                 config.KeyFile("friend", self.friend_key_file, required=False)),
                      cache_root=self.state, models=models, default_model="swift",
                      idle_seconds=1800, release_delay=0, stop_timeout=2)
        values.update(overrides)
        self.config = config.GatewayConfig(**values)
        self.server = gateway.create_server(self.config)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.address = self.server.server_address

    def request(self, path="/v1/chat/completions", *, method="POST", key=OWNER_KEY,
                body=None, headers=None, timeout=5):
        connection = http.client.HTTPConnection(*self.address, timeout=timeout)
        request_headers = {"Connection": "close"}
        if key is not None:
            request_headers["Authorization"] = "Bearer " + key
        if headers:
            request_headers.update(headers)
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            request_headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def completion(self, model="swift", *, _key=OWNER_KEY, _headers=None, **fields):
        return self.request(key=_key, headers=_headers, body={"model": model, "messages": [
            {"role": "user", "content": "Public synthetic test prompt."}], **fields})

    def rows(self):
        if not self.events.exists():
            return []
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def close(self):
        self.server.shutdown()
        self.server.manager.close()
        self.server.server_close()
        self.thread.join(timeout=3)


@contextmanager
def gateway(**overrides):
    with tempfile.TemporaryDirectory(prefix="tensorfold-gateway-tests-") as directory:
        harness = Harness(Path(directory), **overrides)
        try:
            yield harness
        finally:
            harness.close()
