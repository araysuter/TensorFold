"""Cold token-array concurrency matrix with solo/plain token hashes and client-observed timing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request

from bench_auth import authorize, read_api_key, redact


def request(base, model, key, length, stream, tokens, seed, draft, barrier=None):
    body = dict(model=model, prompt=[1000 + (i * 79 + stream * 997) % 30000 for i in range(length)],
                max_tokens=tokens, temperature=0, seed=seed, ignore_eos=True, draft=draft,
                stream=True, stream_options={"include_usage": True})
    query = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json",
                                            "User-Agent": "TensorFold-Concurrency-Benchmark/1.0"})
    authorize(query, key)
    if barrier:
        barrier.wait(timeout=30)
    row = dict(stream=stream, input_tokens=length, sent=time.monotonic(), status=None)
    try:
        first = last = None
        usage, runtime = {}, {}
        complete = False
        with urllib.request.urlopen(query, timeout=1800) as response:
            row["status"] = response.status
            for raw in response:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    complete = True
                    break
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    raise ValueError("stream error")
                usage = chunk.get("usage") or usage
                runtime = chunk.get("tensorfold") or runtime
                if any(c.get("text") or c.get("delta", {}).get("content")
                       or c.get("delta", {}).get("reasoning_content") for c in chunk.get("choices", [])):
                    last = time.monotonic()
                    first = first or last
        if not complete or usage.get("completion_tokens") != tokens or first is None:
            raise ValueError("incomplete output")
        if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
            raise ValueError("cold cache not established")
        if not runtime.get("token_sha256"):
            raise ValueError("token hash unavailable")
        row.update(first=first, last=last, usage=usage, runtime=runtime,
                   prompt_tps=usage["prompt_tokens"] / runtime["prefill_seconds"],
                   decode_tps=runtime["tokens_per_second"], token_sha256=runtime["token_sha256"])
    except urllib.error.HTTPError as exc:
        row.update(status=exc.code, error="request_rejected")
    except urllib.error.URLError as exc:
        row.update(error="URLError", transport_error=type(exc.reason).__name__)
        if isinstance(exc.reason, OSError):
            row["transport_errno"] = exc.reason.errno
    except Exception as exc:
        row["error"] = type(exc).__name__
    row["completed"] = time.monotonic()
    return row


def summarize(length, users, rows):
    successful = [r for r in rows if "error" not in r]
    cell = dict(input_tokens=length, users=users, admitted=len(successful), rejected=users - len(successful), rows=rows)
    if successful:
        start = min(r["sent"] for r in successful)
        first = min(r["first"] for r in successful)
        last = max(r["last"] for r in successful)
        end = max(r["completed"] for r in successful)
        cell.update(median_prompt_tps=statistics.median(r["prompt_tps"] for r in successful),
                    median_decode_tps=statistics.median(r["decode_tps"] for r in successful),
                    median_ttft=statistics.median(r["first"] - r["sent"] for r in successful),
                    aggregate_prompt_tps=sum(r["usage"]["prompt_tokens"] for r in successful)
                    / (max(r["first"] for r in successful) - start),
                    aggregate_decode_tps=sum(r["usage"]["completion_tokens"] - 1 for r in successful) / (last - first),
                    end_to_end_output_tps=sum(r["usage"]["completion_tokens"] for r in successful) / (end - start))
        events = sorted([(r["first"], 1) for r in successful] + [(r["last"], -1) for r in successful])
        active = overlap = 0
        for _, delta in events:
            active += delta
            overlap = max(overlap, active)
        cell["decode_overlap"] = overlap
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="Root API URL without /v1")
    parser.add_argument("model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--lengths", default="4000,8000,32000")
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--context", type=int, default=96000)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = read_api_key(parser, args.api_key_env)
    try:
        lengths, levels = ([int(v) for v in value.split(",")] for value in (args.lengths, args.levels))
        if (not lengths or not levels or not 1 <= args.tokens <= args.context
                or any(n < 1 or n + args.tokens > args.context for n in lengths)
                or any(n < 1 or n > 8 for n in levels)):
            raise ValueError()
    except ValueError:
        parser.error("positive prompt lengths plus output must fit context; concurrency must be 1 through 8")
    report = dict(status="running", model=args.model, context=args.context, output_tokens=args.tokens,
                  seed=args.seed, repetitions=1, cache_gib=0, oracles=[], cells=[])

    def save():
        args.output.write_text(json.dumps(redact(report, key), indent=2) + "\n")

    try:
        for length in lengths:
            oracle = {}
            for stream in range(max(levels)):
                row = request(args.base, args.model, key, length, stream, args.tokens, args.seed, False)
                report["oracles"].append(row)
                save()
                if "error" in row:
                    raise ValueError("solo/plain oracle failed")
                oracle[stream] = row["token_sha256"]
            for users in levels:
                barrier = threading.Barrier(users)
                with ThreadPoolExecutor(max_workers=users) as pool:
                    rows = list(pool.map(lambda i: request(args.base, args.model, key, length, i,
                                                         args.tokens, args.seed, True, barrier), range(users)))
                cell = summarize(length, users, rows)
                cell["exact"] = all(r.get("token_sha256") == oracle[r["stream"]]
                                    for r in rows if "error" not in r)
                report["cells"].append(cell)
                save()
                print(json.dumps({k: v for k, v in cell.items() if k != "rows"}), flush=True)
                if not cell["exact"]:
                    raise ValueError("concurrent output differs from solo/plain")
        report["status"] = "passed" if all(c["admitted"] == c["users"] for c in report["cells"]) else "capacity_limited"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        save()
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
