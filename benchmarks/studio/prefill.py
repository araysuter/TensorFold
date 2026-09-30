"""Uncached Studio prompt-processing trials, with output hashes and local GPU/swap readings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.request


def gpu_utilization() -> int | None:
    try:
        output = subprocess.run(["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
                                capture_output=True, text=True, timeout=3).stdout
        hit = re.search(r'"Device Utilization %"\s*=\s*(\d+)', output)
        return int(hit[1]) if hit else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def swap_usage() -> str | None:
    try:
        return subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True,
                              timeout=3).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="swift-1.5")
    parser.add_argument("--tokenizer", default=str(Path.home() / ".cache/huggingface/hub/"
                        "models--ukisai--Swift-1.5-4bit-MLX/snapshots/82276731e47e1db4ac502f24c63cfaf886639be9"))
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/tensorfold/api-key")
    parser.add_argument("--contexts", nargs="+", type=int, default=[20000, 60000, 128000])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or min(args.contexts) < 128 or args.max_tokens < 1:
        parser.error("reps and max-tokens must be positive; contexts must be at least 128")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    key = args.key_file.read_text().strip()
    deadline = time.monotonic() + 60
    while True:
        try:
            request = urllib.request.Request(args.url.rstrip("/") + "/health",
                                            headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(request, timeout=3) as response:
                health = json.load(response)
            if health.get("status") == "ok" and health.get("model") == args.model and not health.get("warming"):
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"No ready {args.model} TensorFold server at {args.url}")
        time.sleep(1)
    args.out.mkdir(parents=True, exist_ok=False)
    block = tokenizer.encode("\n".join(f'def transform_{i}(rows):\n'
                             '    return sorted(rows, key=lambda row: row["id"])\n'
                             for i in range(90)), add_special_tokens=False)
    suffix = tokenizer.encode("\nExplain the correctness risks and propose fixes.\nAnswer:\n",
                              add_special_tokens=False)
    records = []
    (args.out / "config.json").write_text(json.dumps({"label": args.label, "model": args.model,
                "tokenizer": args.tokenizer, "contexts": args.contexts, "reps": args.reps,
                "max_tokens": args.max_tokens, "temperature": 0, "seed": 17, "draft": True}, indent=2) + "\n")
    for size in args.contexts:
        for trial in range(args.reps):
            # Different first token on every request prevents RAM and disk prefix reuse.
            # Every server variant receives exactly the same IDs and sampling seed.
            first = 1000 + args.contexts.index(size) * args.reps + trial
            prefix = [first] + tokenizer.encode(f"Run {trial}: inspect this synthetic Python repository.\n",
                                                add_special_tokens=False)
            room = size - len(prefix) - len(suffix)
            ids = prefix + (block * (room // len(block) + 1))[:room] + suffix
            body = {"model": args.model, "prompt": ids, "max_tokens": args.max_tokens, "temperature": 0,
                    "seed": 17, "reasoning_effort": "none", "draft": True}
            readings = []
            stop = threading.Event()

            def monitor() -> None:
                while not stop.is_set():
                    value = gpu_utilization()
                    if value is not None:
                        readings.append(value)
                    stop.wait(1)

            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            before = swap_usage()
            began = time.perf_counter()
            try:
                request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                          "Authorization": "Bearer " + key})
                with urllib.request.urlopen(request, timeout=1800) as response:
                    reply = json.load(response)
            finally:
                stop.set()
                watcher.join(timeout=4)
            elapsed = time.perf_counter() - began
            health_request = urllib.request.Request(args.url.rstrip("/") + "/health",
                                                    headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(health_request, timeout=30) as response:
                memory = json.load(response).get("memory", {})
            usage, metrics = reply["usage"], reply["tensorfold"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            if usage["prompt_tokens"] != size or cached:
                raise RuntimeError(f"invalid cold trial: prompt={usage['prompt_tokens']}, cached={cached}")
            record = {"label": args.label, "tokens": size, "trial": trial, "cached_tokens": cached,
                      "prefill_seconds": metrics["prefill_seconds"], "ttft_seconds": metrics["time_to_first_token"],
                      "prompt_tps": size / metrics["prefill_seconds"], "wall_seconds": elapsed,
                      "completion_tokens": usage["completion_tokens"], "token_sha": metrics["token_sha"],
                      "decode_tps": metrics.get("tokens_per_second"), "memory_after": memory,
                      "text": reply["choices"][0]["text"], "gpu_utilization": readings,
                      "swap_before": before, "swap_after": swap_usage()}
            records.append(record)
            (args.out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k not in ("text", "gpu_utilization")}), flush=True)
    summary = [{"tokens": size, "samples": args.reps,
                "median_prefill_seconds": statistics.median(r["prefill_seconds"] for r in records if r["tokens"] == size),
                "median_prompt_tps": statistics.median(r["prompt_tps"] for r in records if r["tokens"] == size)}
               for size in args.contexts]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
