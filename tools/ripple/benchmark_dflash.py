#!/usr/bin/env python3
"""Measure bounded, fresh-KV Ripple plain/drafted token identity and prompt/decode throughput."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import tempfile
import threading
import time


PROMPTS = {
    'prose': 'Explain how a hash table handles collisions. Use three short paragraphs and a concrete example.',
    'code': 'Write a Python function merge_intervals(intervals) and explain its time complexity. Include edge cases.',
}
SOURCE_REVISIONS = {'target': '4f308b9954b73cd200812095916c7baf68cf27f4',
                    'drafter': '35609b3be3dd08a822b3ba1257c1ea5dd6f87632'}


def token_hash(tokens):
    return hashlib.sha256(struct.pack('<' + 'I' * len(tokens), *tokens)).hexdigest()


def numbers(value, cast=int):
    values = [cast(v) for v in value.split(',')]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('Use a nonempty comma-separated list without duplicates')
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', type=Path, required=True)
    result.add_argument('--drafter', type=Path)
    result.add_argument('--exact-verify', action='store_true')
    result.add_argument('--mlx-build-receipt', type=Path)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--phase', choices=['decode', 'prefill'], default='decode')
    result.add_argument('--modes', default='plain,draft')
    result.add_argument('--prefill-sizes', type=numbers, default=[2048])
    result.add_argument('--prefill-chunks', type=numbers, default=[512])
    result.add_argument('--prefill-clear-intervals', type=numbers, default=[0],
                        help='Draft-only clear policies in prompt tokens; 0 keeps old every-chunk clearing')
    result.add_argument('--draft-block-widths', type=numbers, default=[4])
    result.add_argument('--adaptive-blocks', action='store_true',
                        help='Add an adaptive draft case starting at width4 beside the requested fixed widths')
    result.add_argument('--temperatures', type=lambda v: numbers(v, float), default=[0.0, 0.7])
    result.add_argument('--seeds', type=numbers, default=[17])
    result.add_argument('--reps', type=int, default=1)
    result.add_argument('--warmups', type=int, default=0)
    result.add_argument('--prime-tokens', type=int, choices=[0, 16, 32], default=0)
    result.add_argument('--max-tokens', type=int, default=256)
    result.add_argument('--context', type=int, default=131072)
    result.add_argument('--max-cases', type=int, default=32)
    result.add_argument('--max-seconds', type=float, default=1800)
    result.add_argument('--cache-states', default='fresh')
    result.add_argument('--cache-limit-mib', type=int, choices=[128, 512, 1024], default=128)
    result.add_argument('--label', default='ripple-mlx-dflash2')
    return result


def plan(args):
    modes, states = args.modes.split(','), args.cache_states.split(',')
    if (not modes or len(set(modes)) != len(modes) or set(modes) - {'plain', 'draft'}
            or not states or len(set(states)) != len(states) or set(states) - {'fresh', 'warm', 'restored'}):
        raise ValueError('Unsupported or duplicate mode/cache state')
    if 'draft' in modes and args.drafter is None:
        raise ValueError('--drafter is required for draft mode')
    if (not 1 <= args.reps <= 10 or not 0 <= args.warmups <= 3 or not 1 <= args.max_tokens <= 256
            or not 512 <= args.context <= 131072 or not 0 < args.max_seconds <= 14400
            or not 1 <= args.max_cases <= 512):
        raise ValueError('Invalid bounded run budget')
    if any(v not in (128, 256, 512, 1024, 2048) for v in args.prefill_chunks):
        raise ValueError('Prefill chunks must be 128, 256, 512, 1024 or 2048')
    if any(v not in (0, 4096, 8192) for v in args.prefill_clear_intervals):
        raise ValueError('Prefill clear intervals must be 0, 4096 or 8192')
    if any(v not in (4, 8, 16) for v in args.draft_block_widths):
        raise ValueError('Draft block widths must be 4, 8 or 16')
    if any(not math.isfinite(t) or t < 0 or t > 2 for t in args.temperatures):
        raise ValueError('Temperatures must be finite and between 0 and 2')
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError('Seeds must be uint32')
    if any(not 128 <= size <= args.context for size in args.prefill_sizes):
        raise ValueError('Prefill sizes must fit the configured context')
    prompts = list(PROMPTS) if args.phase == 'decode' else args.prefill_sizes
    temperatures = args.temperatures if args.phase == 'decode' else [0.0]
    cases = []
    for rep in range(args.reps + args.warmups):
        for prompt in prompts:
            for chunk in args.prefill_chunks:
                for temp in temperatures:
                    for seed in args.seeds if temp else args.seeds[:1]:
                        for state in states:
                            ordered = modes if rep % 2 == 0 else modes[::-1]
                            for mode in ordered:
                                variants = draft_variants(args) if mode == 'draft' else [dict(
                                    block_width=None, adaptive=False, prefill_clear_interval=0)]
                                for variant in variants if rep % 2 == 0 else variants[::-1]:
                                    cases.append(dict(prompt=prompt, chunk=chunk, temperature=temp, seed=seed,
                                                      state=state, mode=mode, **variant,
                                                      rep=rep - args.warmups, warmup=rep < args.warmups))
    if args.prime_tokens:
        priming = []
        for mode in modes:
            variant = draft_variants(args)[0] if mode == 'draft' else dict(
                block_width=None, adaptive=False, prefill_clear_interval=0)
            priming.append(dict(prompt='prose', chunk=512, temperature=0.0, seed=args.seeds[0],
                                state='fresh', mode=mode, **variant, rep=-2, warmup=True,
                                priming=True, output_budget=args.prime_tokens))
        cases = priming + cases
    if len(cases) > args.max_cases:
        raise ValueError(f'Matrix needs {len(cases)} cases; explicitly raise --max-cases to admit it')
    return cases


def draft_variants(args):
    policies = [(width, False) for width in args.draft_block_widths]
    if args.adaptive_blocks:
        policies.append((4, True))
    return [dict(block_width=width, adaptive=adaptive, prefill_clear_interval=interval)
            for interval in args.prefill_clear_intervals for width, adaptive in policies]


def public_tokens(tokenizer, prompt, maximum, context):
    if isinstance(prompt, str):
        content = PROMPTS[prompt]
        result = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                               tokenize=True, add_generation_prompt=True)
        result = result.tolist() if hasattr(result, 'tolist') else result
    else:
        count = min(prompt, context - maximum)
        prefix = tokenizer.encode('Read these synthetic public software test records.\n', add_special_tokens=False)
        filler = tokenizer.encode('A deterministic test checks a dictionary, a list and an integer counter.\n',
                                  add_special_tokens=False)
        suffix = tokenizer.encode('\nExplain what the records test.\n', add_special_tokens=False)
        room = count - len(prefix) - len(suffix)
        if room < 0 or not filler:
            raise ValueError('Tokenizer cannot construct the requested synthetic prompt')
        result = prefix + (filler * (room // len(filler) + 1))[:room] + suffix
        if len(result) != count:
            raise ValueError('Synthetic prompt length mismatch')
    if not result or len(result) + maximum > context:
        raise ValueError('Public prompt plus output budget exceeds context')
    return list(result)


def comparison_key(case):
    return tuple(case[k] for k in ['prompt', 'chunk', 'temperature', 'seed', 'state', 'rep', 'warmup'])


def compare_cases(cases):
    grouped = {}
    for case in cases:
        group = grouped.setdefault(comparison_key(case), {'draft': []})
        if case['mode'] == 'plain':
            group['plain'] = case
        else:
            group['draft'].append(case)
    comparisons = []
    for group in grouped.values():
        if 'plain' not in group:
            continue
        for draft in group['draft']:
            plain = group['plain']
            equal = plain['output_tokens'] == draft['output_tokens']
            mismatch = first_mismatch(plain['output_tokens'], draft['output_tokens'])
            comparisons.append({'plain_case': plain['case_id'], 'draft_case': draft['case_id'], 'token_identity': equal,
                                'draft_block_width': draft.get('block_width', 16), 'first_mismatch': mismatch,
                                'adaptive': draft.get('adaptive', False),
                                'prefill_clear_interval': draft.get('prefill_clear_interval', 0),
                                'request_speedup': plain['elapsed_seconds'] / draft['elapsed_seconds'],
                                'steady_decode_speedup': rate_ratio(draft, plain)})
    return comparisons


def first_mismatch(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)) if left != right else None)


def chunk_comparisons(cases):
    grouped, result = {}, []
    for case in cases:
        key = tuple(case.get(k) for k in ['prompt', 'temperature', 'seed', 'state', 'rep', 'warmup', 'mode',
                                         'block_width', 'adaptive', 'prefill_clear_interval'])
        reference = grouped.setdefault(key, case)
        if reference['chunk'] == case['chunk']:
            continue
        result.append({'reference_case': reference['case_id'], 'candidate_case': case['case_id'],
                       'reference_chunk': reference['chunk'], 'candidate_chunk': case['chunk'],
                       'token_identity': reference['output_tokens'] == case['output_tokens'],
                       'first_mismatch': first_mismatch(reference['output_tokens'], case['output_tokens'])})
    return result


def policy_comparisons(cases):
    grouped, result = {}, []
    for case in cases:
        key = tuple(case.get(k) for k in ['prompt', 'chunk', 'temperature', 'seed', 'state', 'rep', 'warmup', 'mode'])
        reference = grouped.setdefault(key, case)
        if reference is case:
            continue
        result.append({'reference_case': reference['case_id'], 'candidate_case': case['case_id'],
                       'token_identity': reference['output_tokens'] == case['output_tokens'],
                       'first_mismatch': first_mismatch(reference['output_tokens'], case['output_tokens'])})
    return result


def rate_ratio(candidate, baseline):
    a, b = candidate['steady_decode_tps'], baseline['steady_decode_tps']
    return a / b if a is not None and b else None


def offsets(cache):
    return [int(entry.offset) for entry in cache]


def metadata(args):
    files = ['benchmark_dflash.py', 'dflash_generate.py', 'dflash_model.py', 'dflash_target.py',
             'dflash_exact.py', 'dflash_runtime.py',
             'modeling_ricursive_mlx.py']
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in files if Path(__file__).with_name(name).exists()}
    configs, provenance = {}, {}
    for name, path in [('target', args.model), ('drafter', args.drafter)]:
        if path:
            configs[name] = {file: hashlib.sha256((path / file).read_bytes()).hexdigest()
                             for file in ['config.json', 'conversion-receipt.json'] if (path / file).exists()}
            receipt = json.loads((path / 'conversion-receipt.json').read_text())
            provenance[name] = {key: receipt[key] for key in ['source_repo', 'source_revision', 'method', 'mlp_bits',
                                                              'weight_bytes', 'format'] if key in receipt}
            if receipt.get('source_revision') != SOURCE_REVISIONS[name]:
                raise ValueError('Checkpoint receipt does not match the selected source revision')
    return {'schema': 'ripple-dflash-benchmark-v1', 'label': args.label, 'phase': args.phase,
            'source_revisions_expected': SOURCE_REVISIONS, 'checkpoint_provenance': provenance,
            'config_sha256': configs, 'source_sha256': hashes,
            'packages': {p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'transformers']},
            'platform': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
            'runtime': 'temporary-mlx', 'native_tensorfold': False, 'prefix_retention': False,
            'plain_baseline': 'stock-generation-target-only-cache-with-inert-hidden-tap-wrappers',
            'limits': {'mlx_bytes': 54 * 1024**3, 'physical_bytes': 56 * 1024**3,
                       'seconds': args.max_seconds, 'cases': args.max_cases, 'context': args.context,
                       'cache_pool_bytes': args.cache_limit_mib * 1024**2},
            'settings': {'max_output_tokens': args.max_tokens, 'reps': args.reps, 'warmups': args.warmups,
                         'prime_tokens': args.prime_tokens,
                         'draft_block_widths': args.draft_block_widths, 'adaptive_blocks': args.adaptive_blocks,
                         'prefill_clear_intervals': args.prefill_clear_intervals},
            'public_prompts': PROMPTS, 'prompt_arrays': {}, 'cases': [], 'comparisons': [],
            'chunk_comparisons': [], 'policy_comparisons': [], 'status': 'loading'}


class Guard:
    def __init__(self, row, save, footprint, maximum):
        self.row, self.save, self.footprint, self.maximum = row, save, footprint, maximum
        self.started, self.peak = time.monotonic(), 0
        self.case_peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def monitor(self):
        while not self.done.wait(0.1):
            current = self.footprint()
            self.peak = max(self.peak, current)
            self.case_peak = max(self.case_peak, current)
            if self.peak > 56 * 1024**3 or time.monotonic() - self.started > self.maximum:
                self.row['status'] = 'memory_guard_stopped' if self.peak > 56 * 1024**3 else 'time_guard_stopped'
                self.row['peak_physical_footprint_bytes'] = self.peak
                self.save()
                os._exit(75)


def run_case(args, spec, ids, runtime, guard):
    mx, tokenizer, plain, drafted, stock_generate, draft_generate, make_sampler = runtime
    model = plain if spec['mode'] == 'plain' else drafted
    generator = stock_generate if spec['mode'] == 'plain' else draft_generate
    if spec['mode'] == 'draft':
        if spec['block_width'] != model.draft.args.block_size and not hasattr(model, 'draft_block_size'):
            raise ValueError('Decoder does not expose the draft_block_size qualification control')
        if hasattr(model, 'draft_block_size'):
            model.draft_block_size = spec['block_width']
        for name, value in [('adaptive_blocks', spec['adaptive']),
                            ('prefill_clear_interval', spec['prefill_clear_interval'])]:
            if value and not hasattr(model, name):
                raise ValueError('Decoder does not expose the ' + name + ' qualification control')
            if hasattr(model, name):
                setattr(model, name, value)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    guard.case_peak = guard.footprint()
    cache = model.make_cache()
    prefix = 0 if spec['state'] == 'fresh' else len(ids) // 2
    snapshot = None
    last_clear = 0
    if prefix:
        for start in range(0, prefix, spec['chunk']):
            model(mx.array([ids[start:min(start + spec['chunk'], prefix)]]), cache=cache)
            mx.eval([entry.state for entry in cache])
            done = min(start + spec['chunk'], prefix)
            if not spec['prefill_clear_interval'] or done - last_clear >= spec['prefill_clear_interval']:
                mx.clear_cache()
                last_clear = done
        if spec['state'] == 'restored':
            from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
            with tempfile.TemporaryDirectory(prefix='ripple-bench-') as root:
                path = Path(root) / 'cache.safetensors'
                save_prompt_cache(str(path), cache)
                snapshot = {'bytes': path.stat().st_size, 'offsets_saved': offsets(cache),
                            'scope': 'standalone-mlx-cache-not-worker-snapshot'}
                del cache
                gc.collect()
                mx.clear_cache()
                cache = load_prompt_cache(str(path))
                mx.eval([entry.state for entry in cache])
                snapshot['offsets_restored'] = offsets(cache)
    before = offsets(cache)
    baseline = dict(drafted.counters) if drafted is not None else {}
    mx.random.seed(spec['seed'])
    started, first, last, output, last_response = time.monotonic(), None, None, [], None
    progress = []

    def callback(done, total):
        progress.append({'tokens': done, 'total': total, 'seconds': time.monotonic() - started})

    maximum = spec.get('output_budget', args.max_tokens)
    iterator = generator(model, tokenizer, ids[prefix:], max_tokens=maximum,
                         sampler=make_sampler(temp=spec['temperature']), prompt_cache=cache,
                         prefill_step_size=spec['chunk'], prompt_progress_callback=callback)
    try:
        for response in iterator:
            now = time.monotonic()
            if first is None:
                first = now
            last, last_response = now, response
            output.append(int(response.token))
    finally:
        iterator.close()
    mx.eval([entry.state for entry in cache])
    elapsed = time.monotonic() - started
    if last_response is None:
        raise ValueError('Generator returned no tokens')
    after = offsets(cache)
    if len(set(after)) != 1:
        raise ValueError('Benchmark ended with inconsistent target/draft cache offsets')
    counters = {key: value - baseline.get(key, 0) for key, value in drafted.counters.items()} if drafted else {}
    proposed = counters.get('draft_proposed_tokens_total', 0)
    accepted = counters.get('draft_accepted_tokens_total', 0)
    verifies = counters.get('draft_verify_calls_total', 0)
    steady = (len(output) - 1) / (last - first) if len(output) > 1 and last > first else None
    return dict(spec, prompt_tokens=len(ids), prompt_sha256=token_hash(ids), cached_prefix_tokens=prefix,
                output_tokens=output, output_sha256=token_hash(output), generation_tokens=len(output),
                max_output_tokens=maximum, elapsed_seconds=elapsed, ttft_seconds=first - started,
                steady_decode_tps=steady, request_output_tps=len(output) / elapsed,
                library_prompt_tps=float(last_response.prompt_tps),
                library_prompt_seconds=(len(ids) - prefix) / max(float(last_response.prompt_tps), 1e-9),
                library_generation_tps=float(last_response.generation_tps),
                finish_reason=getattr(last_response, 'finish_reason', None),
                prefill_progress=progress, offsets_before=before, offsets_after=after, snapshot=snapshot,
                peak_mlx_bytes=mx.get_peak_memory(), physical_footprint_bytes=guard.footprint(),
                sampled_peak_physical_footprint_bytes=guard.case_peak,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                final_block_width=getattr(model, 'draft_block_size', None) if spec['mode'] == 'draft' else None,
                tokens_per_verify_call=(len(output) - 1) / verifies if verifies else None,
                counters=counters, acceptance=accepted / proposed if proposed else None)


def main(args):
    specs = plan(args)
    if args.output.exists():
        raise ValueError('Use a new receipt to preserve previous attempts')
    if (args.model / 'CONVERSION_INCOMPLETE').exists():
        raise ValueError('Incomplete target conversion')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from probe import footprint, paging
    mx.set_memory_limit(54 * 1024**3)
    mx.set_cache_limit(args.cache_limit_mib * 1024**2)
    row, lock = metadata(args), threading.Lock()
    row['before_paging'] = paging()
    row['physical_ram_bytes'] = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
    row['hardware_model'] = subprocess.check_output(['sysctl', '-n', 'hw.model']).decode().strip()
    row['device'] = mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info()

    def save():
        with lock:
            temporary = args.output.with_suffix(args.output.suffix + '.tmp')
            temporary.write_text(json.dumps(row, indent=2) + '\n')
            os.replace(temporary, args.output)

    save()
    guard = Guard(row, save, footprint, args.max_seconds)
    guard.thread.start()
    try:
        started = time.monotonic()
        plain, tokenizer = load(str(args.model), trust_remote_code=True)
        drafted, draft_generate = None, None
        if 'draft' in args.modes.split(','):
            from dflash_model import load_draft
            from dflash_target import DraftedTarget
            from dflash_generate import stream_generate as draft_generate
            if args.exact_verify:
                if args.mlx_build_receipt is None:
                    raise ValueError('--exact-verify requires an isolated MLX build receipt')
                from dflash_runtime import verify_runtime
                row['isolated_runtime_fingerprint'] = verify_runtime(args.mlx_build_receipt)
            drafted = DraftedTarget(plain, load_draft(args.drafter), exact_verification=args.exact_verify)
        row.update(load_seconds=time.monotonic() - started, status='running')
        runtime = mx, tokenizer, plain, drafted, stream_generate, draft_generate, make_sampler
        for number, spec in enumerate(specs):
            ids = public_tokens(tokenizer, spec['prompt'], args.max_tokens, args.context)
            digest = token_hash(ids)
            row['prompt_arrays'][digest] = ids
            row['active_case'] = dict(spec, case_id=number)
            save()
            case = run_case(args, spec, ids, runtime, guard)
            case['case_id'] = number
            row['cases'].append(case)
            row['comparisons'] = compare_cases(row['cases'])
            row['chunk_comparisons'] = chunk_comparisons(row['cases'])
            row['policy_comparisons'] = policy_comparisons(row['cases'])
            save()
            print(json.dumps({'event': 'case_complete', 'case_id': number, 'mode': spec['mode'],
                              'prompt_tokens': len(ids), 'output_sha256': case['output_sha256'],
                              'steady_decode_tps': case['steady_decode_tps']}), flush=True)
            checks = row['comparisons'] + row['chunk_comparisons'] + row['policy_comparisons']
            if any(not item['token_identity'] for item in checks):
                raise ValueError('Plain/drafted, cross-chunk or policy token identity failed; receipt preserved')
            if drafted is not None:
                drafted.taps.values.clear()
        row.update(status='complete', after_paging=paging(), peak_physical_footprint_bytes=guard.peak)
        row.pop('active_case', None)
        save()
    except BaseException as exc:
        row.update(status='failed', failure_type=type(exc).__name__, peak_physical_footprint_bytes=guard.peak)
        save()
        raise
    finally:
        guard.done.set()
        guard.thread.join(timeout=1)
    return row


if __name__ == '__main__':
    try:
        main(parser().parse_args())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
