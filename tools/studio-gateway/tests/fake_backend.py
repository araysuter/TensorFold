"""Small, CPU-only native-backend fixture for gateway lifecycle/HTTP tests.

Events contain no request bodies or credentials.  This runs in a real child
process so lifecycle assertions exercise subprocess handling, not a mock of it.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.server
import json
import os
from pathlib import Path
import signal
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--ready-delay", type=float, default=0)
    parser.add_argument("--exit-immediately", action="store_true")
    args, _ = parser.parse_known_args()

    def event(kind: str, **values: object) -> None:
        row = {"event": kind, "label": args.label, "pid": os.getpid(),
               "at": time.monotonic(), **values}
        with args.events.open("a") as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(json.dumps(row) + "\n")
            output.flush()

    def terminate(_number: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    # SIGKILL cannot run a finally block. Check previous fixture processes when
    # the next starts, rather than treating a missing stop event as an overlap.
    predecessors = []
    if args.events.exists():
        for line in args.events.read_text().splitlines():
            row = json.loads(line)
            if row["event"] != "start":
                continue
            try:
                os.kill(row["pid"], 0)
            except ProcessLookupError:
                continue
            predecessors.append(row["pid"])
    event("start", live_predecessors=predecessors)
    active_context: dict[str, object] = {}
    fingerprint = hashlib.sha256(args.label.encode()).hexdigest()[:16]

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *arguments: object) -> None:
            pass

        def answer(self, status: int, value: object) -> None:
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path != "/health" and not self.authorized():
                return
            if self.path == "/health":
                self.answer(200, {"status": "ok", "model": args.label})
            elif self.path == "/v1/models":
                self.answer(200, {"object": "list", "data": [
                    {"id": args.label, "object": "model"}]})
            elif self.path == "/metrics":
                raw = b"\n".join([
                    b"# TYPE tensorfold:requests_running gauge",
                    b"tensorfold:requests_running 1", b"tensorfold:requests_waiting 2",
                    b"tensorfold:generation_tokens_total 6000", b"tensorfold:prompt_tokens_total 12000",
                    b"tensorfold:prompt_tokens_cached_total 2000",
                    b"tensorfold:request_decode_seconds_sum 60", b"tensorfold:request_prefill_seconds_sum 20",
                    b"tensorfold:process_footprint_bytes 1000", b"tensorfold:device_memory_peak_bytes 2000",
                    b'tensorfold:requests_total{key="never-publish-this-label",status="200"} 12',
                    b"tensorfold:pinned_memory_bytes NaN", b"tensorfold:unknown_private_value 12345",
                ])
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/memory":
                self.answer(200, {"schema_version": 1,
                    "process": {"physical_footprint_bytes": 1100,
                                "lifetime_peak_physical_footprint_bytes": 2100,
                                "private_path": "/never/publish/this"},
                    "device": {"active_bytes": 900, "cache_bytes": 100, "peak_bytes": 1900},
                    "prompt_cache_plan": {"budget_bytes": 500, "ram_bytes": 64000, "cap_bytes": 48000},
                    "unknown_private_value": "never-publish-this-value"})
            else:
                self.answer(404, {"error": "unknown fixture endpoint"})

        def authorized(self) -> bool:
            if args.api_key_file is not None:
                expected = "Bearer " + args.api_key_file.read_text().strip().split(":", 1)[-1]
                if self.headers.get("Authorization") != expected:
                    event("wrong_backend_auth")
                    self.answer(401, {"error": "private fixture credential required"})
                    return False
            return True

        def do_POST(self) -> None:
            if not self.authorized():
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if self.path == "/v1/imprint/resume":
                self.answer(200, {"object": "imprint.resume",
                    "ram_tokens": 4 if active_context and not request.get("fixture_prefix_miss") else 0})
                event("preflight")
                return
            if self.path == "/v1/internal/session/save":
                if active_context.get("fail_save"):
                    event("save_failed")
                    self.answer(500, {"error": "fixture snapshot failure"})
                    return
                snapshot = Path(request["path"])
                if active_context:
                    snapshot.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    snapshot.write_text(json.dumps(active_context))
                    snapshot.chmod(0o600)
                event("save")
                self.answer(200, {"object": "session.save", "fingerprint": fingerprint,
                                  "tokens": 4 if active_context else 0,
                                  "bytes": snapshot.stat().st_size if active_context else 0})
                return
            if self.path == "/v1/internal/session/load":
                snapshot = Path(request["path"])
                try:
                    previous = json.loads(snapshot.read_text())
                    active_context.clear()
                    active_context.update(previous)
                except (OSError, ValueError):
                    self.answer(409, {"error": "invalid fixture context"})
                    return
                event("load")
                self.answer(200, {"object": "session.load", "fingerprint": fingerprint,
                                  "tokens": 4, "bytes": snapshot.stat().st_size})
                return
            if self.path == "/v1/internal/session/clear":
                active_context.clear()
                event("clear")
                self.answer(200, {"object": "session.clear", "tokens": 0,
                                  "bytes": 0, "fingerprint": fingerprint})
                return
            if self.path != "/v1/chat/completions":
                self.answer(404, {"error": "unknown fixture endpoint"})
                return
            event("request")
            if request.get("fixture_backend_error"):
                self.answer(500, {"error": {"message": args.api_key_file.read_text().strip().split(":", 1)[-1]}})
                return
            delay = float(request.get("fixture_delay", 0))
            previous_context = active_context.get("marker")
            if "fixture_marker" in request:
                active_context["marker"] = request["fixture_marker"]
            if "fixture_save_error" in request:
                active_context["fail_save"] = request["fixture_save_error"]
            active_context["has_cache"] = True
            if not request.get("stream"):
                time.sleep(delay)
                self.answer(200, {"id": "fixture", "model": args.label,
                    "fixture_previous_context": previous_context,
                    "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": args.label}}]})
                event("request_end")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for index in range(int(request.get("fixture_chunks", 3))):
                    data = {"id": "fixture", "model": args.label,
                            "choices": [{"index": 0, "delta": {
                                "content": f"{args.label}:{index}"}}]}
                    self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(delay)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                event("request_end")
            except (BrokenPipeError, ConnectionResetError):
                event("request_disconnected")
            finally:
                self.close_connection = True

    try:
        if args.exit_immediately:
            raise SystemExit(23)
        time.sleep(args.ready_delay)
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            event("ready")
            server.serve_forever(poll_interval=0.02)
    finally:
        event("stop")


if __name__ == "__main__":
    main()
