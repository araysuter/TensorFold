#!/usr/bin/env python3
"""TensorFold context/concurrency benchmark. Run in the TensorFold venv.
Uses transformers only for the pinned model tokenizer; HTTP, CSV and SVG use stdlib.
Synthetic coding workload, not a correctness evaluation. No server configuration changes.
"""
import argparse
import concurrent.futures
import csv
import getpass
import html
import json
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid


def request(base, key, body, timeout):
    req = urllib.request.Request(base.rstrip('/') + '/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'AstraStudioBenchmark/1.1', 'Accept': 'application/json',
        **({'Authorization': 'Bearer ' + key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if not body.get('stream'):
                return json.load(response)
            result = {}
            done = False
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue  # SSE comments keep the proxy alive, not generated tokens.
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                for field in ('usage', 'tensorfold', 'speculative'):
                    if field in event:
                        result[field] = event[field]
                if event.get('choices'):
                    result['choices'] = event['choices']
            if not done or not result.get('usage'):
                raise RuntimeError('Stream ended without DONE or final token usage')
            return result
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode(errors='replace').replace(key, '[redacted]') if key else exc.read(2048).decode(errors='replace')
        raise RuntimeError(f'HTTP {exc.code}; server={exc.headers.get("Server", "unknown")}; {detail}') from None


def prompt_ids(tokenizer, count):
    # Unique text at the beginning prevents substantial prefix reuse across trials.
    prefix = tokenizer.encode('Benchmark ' + uuid.uuid4().hex + '\nReview this synthetic Python repository:\n', add_special_tokens=False)
    code = '\n'.join(f'def transform_{i}(records):\n    return sorted({{r["id"]: r for r in records}}.values(), key=lambda r: r["id"])\n' for i in range(200))
    block = tokenizer.encode(code, add_special_tokens=False)
    suffix = tokenizer.encode('\nFind correctness and concurrency risks. Write a detailed replacement implementation, then extensive tests and explain the design. Continue with edge cases until the output budget is exhausted.\nAnswer:\n', add_special_tokens=False)
    room = count - len(prefix) - len(suffix)
    if room < 0:
        raise ValueError('Context target too short')
    return prefix + (block * (room // len(block) + 1))[:room] + suffix


def run_one(args, key, ids, barrier, concurrency, rep, user):
    row = dict(context_tokens=len(ids), concurrency=concurrency, repetition=rep,
               user=user, status='error', error='')
    body = dict(model=args.model, prompt=ids, max_tokens=args.output_tokens,
                temperature=0, reasoning_effort='none', draft=not args.no_draft,
                stream=True)
    barrier.wait()
    start = time.perf_counter()
    try:
        reply = request(args.base_url, key, body, args.timeout)
        if 'error' in reply:
            raise RuntimeError(str(reply['error']))
        usage = reply.get('usage', {})
        metrics = reply.get('tensorfold', {})
        n = usage.get('prompt_tokens')
        row.update(actual_prompt_tokens=n, output_tokens=usage.get('completion_tokens'),
                   cached_tokens=usage.get('prompt_tokens_details', {}).get('cached_tokens', 0),
                   decode_tps=metrics.get('tokens_per_second'),
                   ttft_seconds=metrics.get('time_to_first_token'),
                   queue_prefill_seconds=metrics.get('prefill_seconds'),
                   finish_reason=reply.get('choices', [{}])[0].get('finish_reason'),
                   acceptance_rate=reply.get('speculative', {}).get('acceptance_rate'),
                   server_metrics=metrics)
        row['status'] = 'ok' if n == len(ids) and row['output_tokens'] else 'invalid'
        if row['status'] == 'invalid':
            row['error'] = 'Missing output or server prompt count differs from requested count'
        prefill = row['queue_prefill_seconds']
        row['effective_input_tps'] = (n - row['cached_tokens']) / prefill if n and prefill else None
    except Exception as exc:
        row['error'] = str(exc)
    row['elapsed_seconds'] = time.perf_counter() - start
    return row


def plot(groups, key, title, filename):
    width, height = 800, 400
    points = [g for g in groups if g.get(key) is not None]
    max_y = max([g[key] for g in points] or [1]) or 1
    max_x = max([g['context_tokens'] for g in groups] or [128000])
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="800" height="400" fill="white"/><g font-family="sans-serif" font-size="13" fill="#222">', f'<text x="65" y="25" font-size="18">{html.escape(title)}</text>']
    for i in range(6):
        y = 330-i*54
        svg += [f'<path d="M65 {y}H770" stroke="#ddd"/>', f'<text x="5" y="{y+4}">{max_y*i/5:.1f}</text>']
    for c, color in [(1, '#2563eb'), (2, '#e05220')]:
        series = sorted([g for g in points if g['concurrency'] == c], key=lambda g:g['context_tokens'])
        # Individual markers deliberately avoid interpolating across failed cells.
        for g in series:
            x, y = 65+g['context_tokens']/max_x*705, 330-g[key]/max_y*270
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
        svg.append(f'<text x="{100+(c-1)*270}" y="385" fill="{color}">{c} concurrent request(s)</text>')
    for n in sorted(set(g['context_tokens'] for g in groups)):
        svg.append(f'<text x="{65+n/max_x*705}" y="350" text-anchor="end">{n//1000}k</text>')
    svg.append('<text x="300" y="369">Input context tokens</text></g></svg>')
    filename.write_text(''.join(svg))


def save(args, rows, batches):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(dict(config=vars(args), requests=rows, batches=batches), indent=2))
    flat = [{k:v for k,v in r.items() if k != 'server_metrics'} for r in rows]
    if flat:
        fields = list(dict.fromkeys(k for r in flat for k in r))
        with (out/'requests.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(flat)
    groups = []
    for n in args.contexts:
        for c in (1, 2):
            subset = [r for r in rows if r['context_tokens']==n and r['concurrency']==c and r['status']=='ok']
            group = dict(context_tokens=n, concurrency=c, successful_requests=len(subset))
            for metric in ['decode_tps', 'ttft_seconds', 'queue_prefill_seconds', 'effective_input_tps']:
                values = [r[metric] for r in subset if r.get(metric) is not None]
                group[metric] = statistics.median(values) if values else None
            values = [b['aggregate_tps'] for b in batches if b['context_tokens']==n and b['concurrency']==c and b['aggregate_tps'] is not None]
            group['aggregate_tps'] = statistics.median(values) if values else None
            groups.append(group)
    (out/'summary.json').write_text(json.dumps(groups, indent=2))
    charts = [('decode_tps', 'Per-request generation speed (server-reported tok/s)'),
              ('ttft_seconds', 'Time to first token (seconds)'),
              ('queue_prefill_seconds', 'Queue + prompt processing time (seconds)'),
              ('aggregate_tps', 'Combined end-to-end output throughput (tok/s)')]
    for key, title in charts:
        plot(groups, key, title, out/(key+'.svg'))
    failures = sum(r['status']!='ok' for r in rows)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Studio benchmark</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#f4f5f7}img{width:100%;margin:12px 0}</style><h1>TensorFold context benchmark</h1><p>Medians across repetitions. Synthetic cold-prefix coding prompts; drafting '+('off' if args.no_draft else 'on')+'. Concurrency means simultaneous submissions; the server may queue them. Queue + prefill is not pure GPU prefill time. Combined throughput includes queueing and prefill. Missing points indicate failures or missing metrics.</p><p>'+str(len(rows))+' requests; '+str(failures)+' failed/invalid. See requests.csv for actual lengths, cache hits and errors.</p>'+''.join(f'<img src="{key}.svg" alt="{html.escape(title)}">' for key,title in charts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='https://ai-api.ashersuter.com/v1')
    p.add_argument('--model', default='swift-1.5')
    p.add_argument('--api-key-file', help='Optional key file; otherwise securely prompt for the API token')
    p.add_argument('--tokenizer', default='ukisai/Swift-1.5-4bit-MLX')
    p.add_argument('--revision', default='82276731e47e1db4ac502f24c63cfaf886639be9')
    p.add_argument('--contexts', nargs='+', type=int, default=[20000,40000,60000,80000,128000])
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--context-limit', type=int, default=131072)
    p.add_argument('--reps', type=int, default=1, help='Trials per context/concurrency pair (default: 1; use 3 for more reliable medians)')
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--no-draft', action='store_true')
    p.add_argument('--resume-from', help='Previous results directory; retain successful groups and rerun failed/missing groups')
    p.add_argument('--out', default='studio-benchmark-results')
    args = p.parse_args()
    if args.reps < 1 or args.output_tokens < 1 or any(n < 256 or n+args.output_tokens > args.context_limit for n in args.contexts):
        p.error('Require positive repetitions/output, input >=256, and input + output <= context limit')
    if (pathlib.Path(args.out)/'results.json').exists():
        p.error('Output already exists; choose a new --out directory')
    key = pathlib.Path(args.api_key_file).expanduser().read_text().strip() if args.api_key_file else getpass.getpass('API token (hidden): ').strip()
    if not key:
        p.error('API token is required')
    previous = None
    if args.resume_from:
        previous = json.loads((pathlib.Path(args.resume_from)/'results.json').read_text())
        for field in ('model', 'tokenizer', 'revision', 'output_tokens', 'no_draft', 'context_limit'):
            if previous['config'].get(field) != getattr(args, field):
                p.error(f'Cannot resume with changed {field}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    print('Checking authentication and a tiny generation before the load test...', flush=True)
    try:
        probe = request(args.base_url, key, dict(model=args.model, prompt='Say hello.', max_tokens=4,
                        temperature=0, reasoning_effort='none', draft=False, stream=True), min(args.timeout, 120))
        if 'error' in probe or not probe.get('choices'):
            raise RuntimeError(str(probe.get('error', 'No completion returned')))
    except Exception as exc:
        p.exit(1, f'Preflight failed; no benchmark started: {exc}\n')
    rows, batches = [], []
    completed = set()
    if previous:
        for batch in previous.get('batches', []):
            n, c, rep = (batch[k] for k in ('context_tokens', 'concurrency', 'repetition'))
            group = [r for r in previous['requests'] if
                     (r['context_tokens'], r['concurrency'], r['repetition']) == (n, c, rep)]
            if len(group) == c and all(r['status'] == 'ok' for r in group) and batch.get('aggregate_tps') is not None:
                rows.extend(group)
                batches.append(batch)
                completed.add((n, c))
        save(args, rows, batches)
        print('Keeping completed groups: ' + ', '.join(f'{n:,} / {c} concurrent' for n,c in sorted(completed)), flush=True)
    print('Running sequential test groups; up to two concurrent requests. Results saved after each group.', flush=True)
    for n in args.contexts:
        for c in (1,2):
            if (n, c) in completed:
                continue
            for rep in range(1,args.reps+1):
                prompts = [prompt_ids(tokenizer,n) for _ in range(c)]
                barrier = threading.Barrier(c)
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    futures = [pool.submit(run_one,args,key,ids,barrier,c,rep,i+1) for i,ids in enumerate(prompts)]
                    current = [f.result() for f in futures]
                elapsed = time.perf_counter()-start
                rows.extend(current)
                batches.append(dict(context_tokens=n,concurrency=c,repetition=rep,elapsed_seconds=elapsed,
                    aggregate_tps=sum(r['output_tokens'] for r in current)/elapsed if all(r['status']=='ok' for r in current) else None))
                save(args,rows,batches)
                print(f'{n:,} tokens / {c} requests / repeat {rep}: '+', '.join(f"{r['status']} {r.get('decode_tps')} tok/s; {r['elapsed_seconds']:.1f}s" for r in current), flush=True)
    print('Open '+str(pathlib.Path(args.out)/'report.html'))


if __name__ == '__main__':
    main()
