#!/usr/bin/env python3
"""Opt-in real-GPU checks; run against an isolated, session-enabled gateway.

Example: python3 studio_smoke.py --run --key-file ~/.config/tensorfold/api-key
No model response, conversation text, credential, or local path is emitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid


BASELINE_PROMPTS = (
    "What is 17 plus 25? Reply with just the integer.",
    "Write a Python function that returns the nth Fibonacci number, using iteration.",
)


class SmokeFailure(Exception):
    """Only fixed, non-sensitive messages may be constructed here."""


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_key(path, label=None):
    fd = os.open(path.expanduser(), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and not info.st_mode & 0o077 and info.st_size <= 1024 * 1024,
                "Credential file ownership or permissions are unsafe")
        raw = os.read(fd, 1024 * 1024 + 1).decode()
    finally:
        os.close(fd)
    candidates = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        found, token = None, line
        if ":" in line:
            found, token = (part.strip() for part in line.split(":", 1))
        if label is None or found == label:
            candidates.append(token)
    require(len(candidates) == 1 and candidates[0]
            and all(33 <= ord(char) <= 126 for char in candidates[0]),
            "Choose exactly one valid credential with --key-label if necessary")
    return candidates[0]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, base, key, timeout):
        self.base, self.key, self.timeout = base, key, timeout
        self.samples = []

    def open(self, method, path, body=None, context=None, authenticated=True):
        headers = {"Content-Type": "application/json", "Connection": "close"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self.key
        if context:
            headers["X-TensorFold-Context"] = context
        wire = None if body is None else json.dumps(body).encode()
        request = Request(self.base + path, data=wire, headers=headers, method=method)
        return build_opener(NoRedirect()).open(request, timeout=self.timeout)

    def status(self, method, path, body=None, authenticated=True):
        try:
            with self.open(method, path, body, authenticated=authenticated) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            # Never include server error bodies in a public receipt.
            error.close()
            return error.code, None

    def complete(self, stage, body, context):
        start = time.monotonic()
        with self.open("POST", "/v1/chat/completions", body, context) as response:
            result = json.load(response)
            state = response.headers.get("X-TensorFold-Context-Status")
        return self.record(stage, body, result, state, time.monotonic() - start)

    def stream(self, stage, body, context, on_done):
        start = time.monotonic()
        result = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        terminal = False
        with self.open("POST", "/v1/chat/completions", body, context) as response:
            state = response.headers.get("X-TensorFold-Context-Status")
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                value = line[5:].strip()
                if value == b"[DONE]":
                    require(not terminal, "Duplicate stream terminal marker")
                    terminal = True
                    # Start the other-model request before draining this stream's
                    # final EOF; the gateway must keep its commit/queue barrier.
                    on_done()
                    continue
                require(not terminal, "Stream contained data after its terminal marker")
                event = json.loads(value)
                require("error" not in event, "Native stream reported an error")
                for name in ("usage", "tensorfold", "token_ids"):
                    if name in event:
                        result[name] = event[name]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if isinstance(delta.get("content"), str):
                        result["choices"][0]["message"]["content"] += delta["content"]
                    if choice.get("finish_reason"):
                        result["choices"][0]["finish_reason"] = choice["finish_reason"]
        require(terminal, "Stream ended without its terminal marker")
        return self.record(stage, body, result, state, time.monotonic() - start)

    def record(self, stage, body, result, state, elapsed):
        runtime, usage = result.get("tensorfold", {}), result.get("usage", {})
        token_sha = runtime.get("token_sha256")
        require(isinstance(token_sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", token_sha),
                "Native full token SHA256 is missing")
        choice = result.get("choices", [{}])[0]
        finish = choice.get("finish_reason")
        require(finish in {"stop", "length"}, "Synthetic answer did not finish normally")
        message = choice.get("message", {})
        require(isinstance(message.get("content"), str), "Assistant content is missing")
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        require(isinstance(cached, int) and cached >= 0, "Native cached-token count is missing")
        require(state in {"cold", "resident", "resumed"}, "Gateway context status is missing")
        for name in ("prompt_tokens", "completion_tokens"):
            require(type(usage.get(name)) is int and usage[name] >= 0, "Native token usage is invalid")
        require(isinstance(runtime.get("drafts"), bool), "Native draft-mode receipt is missing")
        ids = result.get("token_ids", runtime.get("token_ids"))
        row = dict(stage=stage, model=body["model"], request_sha256=digest(body),
                   semantic_request_sha256=digest({key: value for key, value in body.items() if key != "draft"}),
                   token_sha256=token_sha.lower(), context_status=state, cached_tokens=cached,
                   wall_seconds=round(elapsed, 4), max_tokens=body["max_tokens"],
                   temperature=body["temperature"], thinking=False,
                   finish_reason=finish,
                   draft_requested=body.get("draft", "default"), drafts_used=runtime.get("drafts"),
                   prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
        for name in ("time_to_first_token", "prefill_seconds", "tokens_per_second", "seconds"):
            number = runtime.get(name)
            if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number):
                row[name] = number
        if isinstance(ids, list) and all(isinstance(item, int) for item in ids):
            row["token_ids_sha256"], row["token_ids_count"] = digest(ids), len(ids)
        self.samples.append(row)
        return {"receipt": row, "message": message, "ids": ids}


def body(model, messages, *, baseline=False, stream=False, draft=True):
    value = dict(model=model, messages=messages, max_tokens=128 if baseline else 64,
                 temperature=0, stream=stream, return_token_ids=True,
                 chat_template_kwargs={"enable_thinking": False})
    if not baseline:
        value["draft"] = draft
    if stream:
        value["stream_options"] = {"include_usage": True}
    return value


def user(text):
    return {"role": "user", "content": text}


def continuation(messages, answer, text):
    require(answer["receipt"]["finish_reason"] == "stop", "Conversation seed hit its output limit")
    message = answer["message"]
    assistant = {"role": "assistant", "content": message["content"]}
    if isinstance(message.get("reasoning_content"), str):
        assistant["reasoning_content"] = message["reasoning_content"]
    return [*messages, assistant, user(text)]


def resumed(answer):
    require(answer["receipt"]["context_status"] == "resumed", "SSD restore was not reported as resumed")
    require(answer["receipt"]["cached_tokens"] > 0, "SSD restore did not reuse cached tokens")


def equivalent(restored, fresh):
    left, right = restored["receipt"], fresh["receipt"]
    require(left["semantic_request_sha256"] == right["semantic_request_sha256"],
            "Equivalence requests differed beyond drafting mode")
    require(left["drafts_used"] is True, "Restored request did not use its drafter")
    require(right["drafts_used"] is False, "Fresh comparison request unexpectedly used drafting")
    require(right["cached_tokens"] == 0, "Fresh explicit context reused an old cache")
    require(left["token_sha256"] == right["token_sha256"], "Restored and fresh token hashes differed")
    require(restored["message"]["content"] == fresh["message"]["content"],
            "Restored and fresh decoded content differed")
    if isinstance(restored["ids"], list) and isinstance(fresh["ids"], list):
        require(restored["ids"] == fresh["ids"], "Restored and fresh token IDs differed")


def exercise(args, receipt):
    client = Client(args.base_url, read_key(args.key_file, args.key_label), args.timeout)
    receipt["samples"] = client.samples
    prefix = "smoke-" + uuid.uuid4().hex[:16]
    checks = receipt["checks"]
    status, models = client.status("GET", "/v1/models")
    require(status == 200 and {args.swift_model, args.ornith_model}.issubset(
        {item["id"] for item in models.get("data", [])}), "Both models were not listed")
    require(client.status("GET", "/v1/models", authenticated=False)[0] == 401,
            "Unauthenticated model listing was accepted")
    for action in ("save", "load", "clear"):
        require(client.status("POST", "/v1/internal/session/" + action, {})[0] == 404,
                "Private native session endpoint was exposed")
    require(client.status("GET", "/stats")[0] == 200, "Collector stats endpoint failed")
    checks.append({"name": "metadata_auth_and_internal_routes", "passed": True})

    reference = json.loads(args.baseline_file.expanduser().read_text()) if args.baseline_file else []
    for index, prompt in enumerate(BASELINE_PROMPTS):
        request = body(args.ornith_model, [user(prompt)], baseline=True)
        answer = client.complete("ornith_baseline_" + str(index), request, prefix + "-baseline-" + str(index))
        if reference:
            match = next((row for row in reference if digest(row["request"]) == digest(request)), None)
            require(match is not None, "A matching native baseline request was not found")
            expected = match["response"].get("tensorfold", {}).get("token_sha256", "").lower()
            require(expected == answer["receipt"]["token_sha256"], "Gateway baseline token hash changed")
        checks.append({"name": "ornith_baseline_" + str(index), "passed": True,
                       "reference_compared": bool(reference)})

    ornith_context, swift_context = prefix + "-shared", prefix + "-swift"
    a1 = [user("Remember the public test label silver-pine. Reply only READY.")]
    first = client.complete("ornith_a1", body(args.ornith_model, a1), ornith_context)
    a2 = continuation(a1, first, BASELINE_PROMPTS[0])
    second = client.complete("ornith_a2", body(args.ornith_model, a2), ornith_context)
    a3 = continuation(a2, second, "What label did I ask you to remember? Reply with only that label.")
    b1 = [user("Remember the public test label amber-brook. Reply only READY.")]
    switched = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        streamed = client.stream("swift_b1_stream", body(args.swift_model, b1, stream=True), swift_context,
            lambda: switched.append(pool.submit(client.complete, "ornith_a3_restored",
                                                body(args.ornith_model, a3), ornith_context)))
        require(len(switched) == 1, "Stream switch barrier was not exercised")
        restored = switched[0].result()
    resumed(restored)
    fresh = client.complete("ornith_a3_fresh", body(args.ornith_model, a3, draft=False), prefix + "-ornith-fresh")
    equivalent(restored, fresh)
    checks.append({"name": "ornith_ssd_restore_and_fresh_equivalence", "passed": True})
    checks.append({"name": "stream_terminal_immediate_model_switch_barrier", "passed": True})

    b2 = continuation(b1, streamed, "What label did I ask you to remember? Reply with only that label.")
    swift_restore = client.complete("swift_b2_restored", body(args.swift_model, b2), swift_context)
    resumed(swift_restore)
    swift_fresh = client.complete("swift_b2_fresh", body(args.swift_model, b2, draft=False), prefix + "-swift-fresh")
    equivalent(swift_restore, swift_fresh)
    checks.append({"name": "swift_ssd_restore_and_fresh_equivalence", "passed": True})
    if args.friend_key_file:
        friend = Client(args.base_url, read_key(args.friend_key_file, args.friend_key_label), args.timeout)
        require(friend.key != client.key, "Friend isolation needs a different credential")
        receipt["friend_samples"] = friend.samples
        guest = friend.complete("friend_same_context_id", body(args.ornith_model, a3), ornith_context)
        require(guest["receipt"]["context_status"] == "cold" and guest["receipt"]["cached_tokens"] == 0,
                "Friend request reused another principal's context")
        a4 = continuation(a3, restored, "Repeat that same label, with no explanation.")
        owner = client.complete("owner_context_after_friend", body(args.ornith_model, a4), ornith_context)
        resumed(owner)
        checks.append({"name": "owner_friend_context_isolation", "passed": True})
    require(client.status("GET", "/stats")[0] == 200, "Collector stats endpoint failed after swaps")
    receipt["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly permit real GPU inference and model swaps")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--key-label")
    parser.add_argument("--friend-key-file", type=Path)
    parser.add_argument("--friend-key-label")
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--swift-model", default="swift-1.5")
    parser.add_argument("--ornith-model", default="ornith-1.5-122b-mixed-2bit")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required to permit real GPU inference")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be a finite positive number")
    split = urlsplit(args.base_url)
    if split.scheme not in {"http", "https"} or not split.netloc or split.username or split.password or split.query or split.fragment:
        parser.error("base URL must be HTTP(S) without credentials, a query, or a fragment")
    args.base_url = args.base_url.rstrip("/")
    if args.base_url.endswith("/v1"):
        args.base_url = args.base_url[:-3]
    receipt = {"schema_version": 1, "suite": "studio-gateway-real-gpu-smoke", "status": "running", "checks": []}
    try:
        exercise(args, receipt)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__,
            "message": str(error) if isinstance(error, SmokeFailure) else "Smoke test failed; inspect private runtime diagnostics"}
    raw = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.output:
        fd = os.open(args.output.expanduser(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(raw)
    print(raw, end="", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
