#!/usr/bin/env python3
"""Opt-in single-trial context capacity checks on an isolated local gateway.

This uses public synthetic filler. It measures capacity and native SSD restore,
not quality or representative prompt-processing performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from studio_smoke import Client, SmokeFailure, body, continuation, read_key, require, user


def native_call(args, path, value=None):
    runs = list(args.cache_root.expanduser().glob("run-*"))
    require(len(runs) == 1, "An isolated gateway workspace is required")
    key = read_key(runs[0] / "backend-key")
    data = None if value is None else json.dumps(value).encode()
    req = Request(args.native_url + path, data=data,
                  headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urlopen(req, timeout=args.timeout) as response:
        return json.load(response)


def memory(args):
    value = native_call(args, "/memory")
    return {
        "context_window": value.get("context_window"),
        "process": {k: v for k, v in (value.get("process") or {}).items()
                    if k in {"physical_footprint_bytes", "lifetime_peak_physical_footprint_bytes"}},
        "device": {k: v for k, v in (value.get("device") or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)},
    }


def prompt(repetitions):
    return ("The following words are synthetic capacity-test filler.\n" +
            " maple" * repetitions + "\nThe filler has ended. Reply with just OK.")


def sized_request(args, model):
    repetitions = args.target_tokens
    for _ in range(4):
        request = body(model, [user(prompt(repetitions))])
        request["max_tokens"] = 8
        count = native_call(args, "/v1/tokenize", request)["count"]
        if count == args.target_tokens:
            return request
        repetitions += args.target_tokens - count
        require(repetitions > 0, "Synthetic prompt sizing failed")
    raise SmokeFailure("Synthetic prompt did not reach the requested exact token count")


def exercise(args, report):
    client = Client(args.base_url, read_key(args.key_file), args.timeout)
    report["samples"] = client.samples
    prefix = "capacity-" + uuid.uuid4().hex[:16]
    models = ("swift-1.5", "ornith-1.5-122b-mixed-2bit")
    for index, model in enumerate(models):
        print(json.dumps({"stage": "loading", "model": model}), flush=True)
        warm = body(model, [user("Reply with just OK.")])
        client.complete("capacity_bootstrap", warm, prefix + "-boot-" + str(index))
        before = memory(args)
        require(before["context_window"] == 131072, "The engine reduced its 128K context window")
        request = sized_request(args, model)
        context = prefix + "-long-" + str(index)
        print(json.dumps({"stage": "prefilling", "model": model,
                          "prompt_tokens": args.target_tokens}), flush=True)
        first = client.complete("near_128k_cold", request, context)
        require(first["receipt"]["prompt_tokens"] == args.target_tokens,
                "The generated prompt length differed from its tokenization")
        require(first["receipt"]["cached_tokens"] == 0, "Capacity prompt was not cold")
        require(first["receipt"]["drafts_used"] is True, "Capacity check did not use its drafter")
        runs = list(args.cache_root.expanduser().glob("run-*"))
        path = runs[0] / "contexts" / (prefix + "-observe-" + str(index) + ".tfs")
        try:
            started = time.monotonic()
            saved = native_call(args, "/v1/internal/session/save", {"path": str(path)})
            save_seconds = time.monotonic() - started
            require(saved["tokens"] >= args.target_tokens - 16
                    and saved["bytes"] == path.stat().st_size,
                    "The near-full state was not completely saved")
            at_capacity = memory(args)
        finally:
            path.unlink(missing_ok=True)
        # A different model forces the gateway to save and destroy this engine.
        other = models[1 - index]
        client.complete("capacity_swap_away", body(other, [user("Reply with just OK.")]),
                        prefix + "-away-" + str(index))
        follow = body(model, continuation(request["messages"], first, "Again reply with just OK."))
        follow["max_tokens"] = 8
        second = client.complete("near_128k_ssd_resumed", follow, context)
        require(second["receipt"]["context_status"] == "resumed"
                and second["receipt"]["cached_tokens"] >= args.target_tokens - 16,
                "Near-full conversation was not restored from SSD")
        require(second["receipt"]["drafts_used"] is True, "Restored context disabled its drafter")
        row = {"model": model, "status": "passed", "target_prompt_tokens": args.target_tokens,
               "synthetic_prompt_sha256": hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest(),
               "saved_tokens": saved["tokens"], "snapshot_bytes": saved["bytes"],
               "save_seconds": save_seconds, "memory_before": before,
               "memory_at_capacity": at_capacity, "memory_after_restore": memory(args)}
        # Hash the API request without exporting its full prompt or token array.
        row["request_sha256"] = first["receipt"]["request_sha256"]
        report["models"].append(row)
        print(json.dumps({"stage": "capacity_passed", "model": model,
                          "cached_tokens": second["receipt"]["cached_tokens"]}), flush=True)
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--native-url", default="http://127.0.0.1:18082")
    parser.add_argument("--cache-root", type=Path, default=Path.home() / "Library/Caches/TensorFold/studio-gateway")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=130944)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if not args.run or not 4096 <= args.target_tokens <= 130944:
        parser.error("--run and a target between 4096 and 130944 tokens are required")
    for address in (args.base_url, args.native_url):
        parsed = urlsplit(address)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            parser.error("capacity checks require isolated HTTP loopback endpoints")
    args.base_url, args.native_url = args.base_url.rstrip("/"), args.native_url.rstrip("/")
    report = {"schema_version": 1, "suite": "single-trial-synthetic-context-capacity",
              "status": "running", "models": [], "quality_benchmark": False}
    try:
        exercise(args, report)
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error) if isinstance(error, SmokeFailure)
                              else "Capacity check failed; inspect private diagnostics"}
    fd = os.open(args.output.expanduser(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "models_completed": len(report["models"])}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
