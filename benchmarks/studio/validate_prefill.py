"""Check long-prompt draft equality and cached chat resumption on a running Studio server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import threading
import urllib.request
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/tensorfold/api-key")
    parser.add_argument("--tokenizer", default=str(Path.home() / ".cache/huggingface/hub/"
                        "models--ukisai--Swift-1.5-4bit-MLX/snapshots/82276731e47e1db4ac502f24c63cfaf886639be9"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix-tag", default="Studio prefill resumption " + uuid.uuid4().hex)
    parser.add_argument("--chat-only", action="store_true", help="check only cold and resumed chat")
    args = parser.parse_args()
    if args.tokens < 8192:
        parser.error("tokens must be at least 8192 to exercise long prompt attention")
    if args.out.exists():
        parser.error("out must name a new result file")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    from prefill import swap_counters
    key = args.key_file.read_text().strip()
    records = []

    def send(endpoint: str, body: dict) -> dict:
        before = swap_counters()
        request = urllib.request.Request(args.url.rstrip("/") + endpoint,
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                  "Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=900) as response:
            reply = json.load(response)
        reply["_swap_counts"] = {"before": before, "after": swap_counters()}
        return reply

    def record(label: str, reply: dict) -> dict:
        metrics = reply["tensorfold"]
        request = urllib.request.Request(args.url.rstrip("/") + "/health",
                                         headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=30) as response:
            memory = json.load(response).get("memory", {})
        item = {"label": label, "usage": reply["usage"], "token_sha": metrics["token_sha"],
                "prefill_seconds": metrics["prefill_seconds"], "speculative": reply.get("speculative"),
                "choice": reply["choices"][0], "memory_after": memory,
                "swap_counts": reply["_swap_counts"],
                "swap_after": subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True,
                                              text=True, timeout=3).stdout.strip()}
        records.append(item)
        args.out.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({k: v for k, v in item.items() if k != "choice"}), flush=True)
        return item

    block = tokenizer.encode('def transform(rows):\n'
                             '    return sorted(rows, key=lambda row: row["id"])\n' * 40,
                             add_special_tokens=False)
    suffix = tokenizer.encode("\nImplement an asynchronous work queue with retries and cancellation.\nAnswer:\n",
                              add_special_tokens=False)
    batches = []
    for draft in (() if args.chat_only else (False, True)):
        barrier = threading.Barrier(2)

        def worker(user: int) -> dict:
            room = args.tokens - 1 - len(suffix)
            ids = [1700 + user] + (block * (room // len(block) + 1))[:room] + suffix
            body = {"model": "swift-1.5", "prompt": ids, "max_tokens": 128, "temperature": .7,
                    "top_k": 20, "top_p": .95, "seed": 17, "draft": draft, "reasoning_effort": "none"}
            barrier.wait()
            return send("/v1/completions", body)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            batch = list(pool.map(worker, range(2)))
        for user, reply in enumerate(batch):
            record(f"concurrent-draft-{draft}-user-{user}", reply)
            assert reply["usage"]["prompt_tokens"] == args.tokens
            assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        batches.append(batch)
    if batches:
        for plain, drafted in zip(*batches):
            assert plain["tensorfold"]["token_sha"] == drafted["tensorfold"]["token_sha"]
            assert plain["choices"][0]["text"] == drafted["choices"][0]["text"]
        print("long concurrent draft/undrafted equality: passed", flush=True)

    system = args.prefix_tag + "\n" + tokenizer.decode(
        (block * (args.tokens // len(block) + 1))[:args.tokens])
    chat = {"model": "swift-1.5", "messages": [
              {"role": "system", "content": system},
              {"role": "user", "content": "Identify the main correctness risk in this Python module."}],
            "max_tokens": 128, "temperature": 0, "seed": 17, "draft": True, "reasoning_effort": "none"}
    cold = record("chat-cold", send("/v1/chat/completions", chat))
    warm = record("chat-resumed", send("/v1/chat/completions", chat))
    assert cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] >= 8192
    assert cold["token_sha"] == warm["token_sha"]
    assert cold["choice"]["message"] == warm["choice"]["message"]
    print("long cached chat resumption equality: passed", flush=True)


if __name__ == "__main__":
    main()
