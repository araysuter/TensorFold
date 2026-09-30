"""Measure a short prompt submitted while a long prompt is being processed."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must name a new result file")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path.home() / ".cache/huggingface/hub/models--ukisai--Swift-1.5-4bit-MLX/"
        "snapshots/82276731e47e1db4ac502f24c63cfaf886639be9", local_files_only=True)
    key = (Path.home() / ".config/tensorfold/api-key").read_text().strip()
    block = tokenizer.encode('def transform(rows):\n'
                             '    return sorted(rows, key=lambda row: row["id"])\n' * 40,
                             add_special_tokens=False)
    suffix = tokenizer.encode("\nExplain the correctness risks.\nAnswer:\n", add_special_tokens=False)

    def send(size: int) -> dict:
        room = size - 1 - len(suffix)
        ids = [2200 + (size == 1024)] + (block * (room // len(block) + 1))[:room] + suffix
        body = {"model": "swift-1.5", "prompt": ids, "max_tokens": 32, "temperature": 0,
                "seed": 17, "draft": True, "reasoning_effort": "none"}
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                  "Authorization": "Bearer " + key})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            reply = json.load(response)
        record = {"label": args.label, "tokens": size, "wall_seconds": time.perf_counter() - started,
                  "usage": reply["usage"], "metrics": reply["tensorfold"],
                  "text": reply["choices"][0]["text"]}
        assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        print(json.dumps(record), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        long = pool.submit(send, 20000)
        time.sleep(1)
        short = pool.submit(send, 1024)
        results = [long.result(), short.result()]
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
