#!/usr/bin/env python3
"""Opt-in real-GPU idle unload/SSD resume check for a short-idle test gateway.

Configure an isolated gateway with idle_seconds=2 before running this script.
Only sanitized counts, timings, modes and token hashes enter its JSON receipt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import uuid

from studio_smoke import Client, SmokeFailure, body, continuation, read_key, require, resumed, user


def exercise(args, receipt):
    key = read_key(args.key_file, args.key_label)
    client = Client(args.base_url, key, args.timeout)
    receipt["samples"] = client.samples
    context = "idle-" + uuid.uuid4().hex[:20]
    messages = [user("Remember the public test label cedar-river. Reply only READY.")]
    seed = client.complete("idle_seed", body(args.model, messages), context)
    require(seed["receipt"]["drafts_used"] is True, "Seed request did not use its drafter")
    start, deadline, polls = time.monotonic(), time.monotonic() + args.idle_wait_seconds, 0
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Gateway did not release its model before the idle deadline")
        probe = Client(args.base_url, key, min(2, remaining))
        status, health = probe.status("GET", "/health", authenticated=False)
        polls += 1
        require(status == 200 and isinstance(health, dict), "Unauthenticated health probe failed")
        require("model" in health and "busy" in health and "loaded_models" in health,
                "Health probe did not expose model lifecycle counters")
        if health["model"] is None and health["busy"] is False and health["loaded_models"] == 0:
            break
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    idle_elapsed = time.monotonic() - start
    configured = health.get("idle_unload_seconds")
    receipt["idle"] = {"model_unloaded": True, "busy": False, "loaded_models": 0,
                       "seconds_after_seed": round(idle_elapsed, 4), "health_polls": polls,
                       "probe_authenticated": False}
    if type(configured) in (int, float) and math.isfinite(configured):
        receipt["idle"]["configured_idle_seconds"] = configured
    status, models = client.status("GET", "/v1/models")
    require(status == 200 and {args.swift_model, args.ornith_model}.issubset(
        {row["id"] for row in models.get("data", [])}), "Model metadata changed after idle unload")
    # Metadata traffic must neither reload the engine nor reset the idle state.
    status, health = client.status("GET", "/health", authenticated=False)
    require(status == 200 and health.get("model") is None and health.get("busy") is False,
            "Model listing unexpectedly loaded an engine")
    following = continuation(messages, seed,
        "What label did I ask you to remember? Reply with only that label.")
    answer = client.complete("idle_restored_followup", body(args.model, following), context)
    resumed(answer)
    require(answer["receipt"]["drafts_used"] is True, "Restored request did not use its drafter")
    receipt["checks"] = [{"name": "idle_unload_without_metadata_pinning", "passed": True},
                         {"name": "both_models_available_while_unloaded", "passed": True},
                         {"name": "idle_full_state_resume_with_drafter", "passed": True}]
    receipt["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Permit real GPU inference and idle unload")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--key-label")
    parser.add_argument("--model", default="swift-1.5")
    parser.add_argument("--swift-model", default="swift-1.5")
    parser.add_argument("--ornith-model", default="ornith-1.5-122b-mixed-2bit")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--idle-wait-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required to permit real GPU inference")
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.idle_wait_seconds)):
        parser.error("timeouts must be finite positive numbers")
    split = urlsplit(args.base_url)
    if split.scheme not in {"http", "https"} or not split.netloc or split.username or split.password or split.query or split.fragment:
        parser.error("base URL must be HTTP(S) without credentials, a query, or a fragment")
    args.base_url = args.base_url.rstrip("/")
    if args.base_url.endswith("/v1"):
        args.base_url = args.base_url[:-3]
    receipt = {"schema_version": 1, "suite": "studio-gateway-real-idle-resume", "status": "running"}
    try:
        exercise(args, receipt)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__,
            "message": str(error) if isinstance(error, SmokeFailure) else "Idle test failed; inspect private runtime diagnostics"}
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
