#!/usr/bin/env python3
"""Build public, per-cell Ripple comparisons from private benchmark receipts without pooling workloads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from benchmark_dflash import PROMPTS, SOURCE_REVISIONS


HASH = re.compile(r'[0-9a-f]{64}')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_. +()-]{0,79}')
VERSION = re.compile(r'[A-Za-z0-9][A-Za-z0-9.+_-]{0,99}')
PAIR_FIELDS = ['prompt', 'prompt_sha256', 'prompt_tokens', 'chunk', 'temperature', 'seed', 'state', 'rep',
               'max_output_tokens', 'cached_prefix_tokens']
CASE_FIELDS = PAIR_FIELDS + ['case_id', 'mode', 'block_width', 'adaptive', 'prefill_clear_interval',
                             'generation_tokens', 'output_sha256', 'finish_reason', 'final_block_width']
METRICS = ['elapsed_seconds', 'ttft_seconds', 'steady_decode_tps', 'request_output_tps', 'library_prompt_tps',
           'library_prompt_seconds', 'acceptance', 'tokens_per_verify_call']
MEMORY = ['peak_mlx_bytes', 'sampled_peak_physical_footprint_bytes', 'physical_footprint_bytes',
          'process_peak_rss_bytes']
STATUSES = {'complete', 'failed', 'running', 'loading', 'memory_guard_stopped', 'time_guard_stopped'}


def token_hash(tokens):
    if not isinstance(tokens, list) or any(type(v) is not int or not 0 <= v < 2**32 for v in tokens):
        raise ValueError('Invalid token array')
    return hashlib.sha256(struct.pack('<' + 'I' * len(tokens), *tokens)).hexdigest()


def digest(value):
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ValueError('Missing or invalid SHA256 identity')
    return value


def number(value, optional=False):
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid numeric benchmark field')
    return value


def version_map(values):
    if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                           or not VERSION.fullmatch(k) or not VERSION.fullmatch(v)
                                           for k, v in values.items()):
        raise ValueError('Invalid package version metadata')
    return values


def hash_map(values):
    if not isinstance(values, dict):
        raise ValueError('Invalid source identity metadata')
    result = {}
    for name, value in values.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', name):
            raise ValueError('Source identity contains a path')
        result[name] = digest(value)
    return result


def cell(raw, arrays):
    prompt = raw.get('prompt')
    if prompt not in PROMPTS and (type(prompt) is not int or not 128 <= prompt <= 131072):
        raise ValueError('Expected a known public prompt ID or bounded synthetic prompt size')
    prompt_sha = digest(raw.get('prompt_sha256'))
    output_sha = digest(raw.get('output_sha256'))
    if token_hash(arrays.get(prompt_sha)) != prompt_sha or token_hash(raw.get('output_tokens')) != output_sha:
        raise ValueError('Token array hash mismatch')
    if len(arrays[prompt_sha]) != raw.get('prompt_tokens') or len(raw['output_tokens']) != raw.get('generation_tokens'):
        raise ValueError('Token array length mismatch')
    if raw.get('mode') not in ('plain', 'draft') or raw.get('state') not in ('fresh', 'warm', 'restored'):
        raise ValueError('Unsupported benchmark mode or cache state')
    if raw.get('finish_reason') not in (None, 'stop', 'length'):
        raise ValueError('Invalid finish reason')
    result = {key: raw.get(key) for key in CASE_FIELDS}
    for key in ['case_id', 'prompt_tokens', 'chunk', 'temperature', 'seed', 'rep', 'max_output_tokens',
                'cached_prefix_tokens', 'generation_tokens']:
        number(result[key])
    if not 1 <= result['generation_tokens'] <= result['max_output_tokens'] <= 256:
        raise ValueError('Output budget does not match the benchmark contract')
    if result['chunk'] not in (128, 256, 512, 1024, 2048):
        raise ValueError('Unbounded prefill chunk')
    result['adaptive'] = raw.get('adaptive', False)
    result['prefill_clear_interval'] = raw.get('prefill_clear_interval', 0)
    if type(result['adaptive']) is not bool or result['prefill_clear_interval'] not in (0, 4096, 8192):
        raise ValueError('Invalid draft policy metadata')
    if result['block_width'] not in (None, 1, 4, 8, 16) or result['final_block_width'] not in (None, 1, 4, 8, 16):
        raise ValueError('Invalid draft width metadata')
    for key in METRICS:
        result[key] = number(raw.get(key), optional=True)
    result['memory_gib'] = {key.removesuffix('_bytes'): number(raw.get(key), optional=True) / 1024**3
                            if raw.get(key) is not None else None for key in MEMORY}
    result['offsets_after'] = [number(v) for v in raw.get('offsets_after', [])]
    return result


def source_metadata(raw, label, receipt_sha):
    target = raw.get('checkpoint_provenance', {}).get('target', {})
    drafter = raw.get('checkpoint_provenance', {}).get('drafter', {})
    if target.get('source_revision') != SOURCE_REVISIONS['target']:
        raise ValueError('Receipt uses a different target source revision')
    if drafter and drafter.get('source_revision') != SOURCE_REVISIONS['drafter']:
        raise ValueError('Receipt uses a different drafter source revision')
    configs = {name: hash_map(value) for name, value in raw.get('config_sha256', {}).items()
               if name in ('target', 'drafter')}
    if not {'config.json', 'conversion-receipt.json'} <= configs.get('target', {}).keys():
        raise ValueError('Target config and conversion identities are required')
    status = raw.get('status')
    if status not in STATUSES or raw.get('phase') not in ('decode', 'prefill'):
        raise ValueError('Invalid benchmark status or phase')
    context = number(raw.get('limits', {}).get('context'))
    if not 512 <= context <= 131072:
        raise ValueError('Invalid total context')
    failure = raw.get('failure_type')
    if failure is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,99}', failure):
        raise ValueError('Invalid failure type')
    hardware = raw.get('hardware_model')
    if hardware is not None and not re.fullmatch(r'(?:Mac|VirtualMac|Apple)[A-Za-z0-9,._ -]{1,70}', hardware):
        raise ValueError('Invalid hardware model')
    runtime = raw.get('isolated_runtime_fingerprint')
    return {'label': label, 'receipt_sha256': receipt_sha, 'status': status, 'phase': raw['phase'],
            'failure_type': failure, 'context': context, 'hardware_model': hardware,
            'cache_pool_bytes': number(raw.get('limits', {}).get('cache_pool_bytes'), optional=True),
            'platform': version_map(raw.get('platform', {})),
            'physical_ram_gib': number(raw.get('physical_ram_bytes'), optional=True) / 1024**3
            if raw.get('physical_ram_bytes') is not None else None,
            'source_versions': {'packages': version_map(raw.get('packages', {})),
                                'source_sha256': hash_map(raw.get('source_sha256', {})),
                                'config_sha256': configs, 'target_source_revision': target['source_revision'],
                                'drafter_source_revision': drafter.get('source_revision'),
                                'isolated_runtime_fingerprint': digest(runtime) if runtime is not None else None},
            'target_weight_bytes': number(target.get('weight_bytes'), optional=True),
            'drafter_weight_bytes': number(drafter.get('weight_bytes'), optional=True),
            'target_mlp_bits': number(target.get('mlp_bits'), optional=True),
            'target_method': target.get('method') if target.get('method') in ('affine', 'mse') else None,
            'excluded_priming_cases': 0, 'excluded_warmup_cases': 0, 'cases': [], 'comparisons': [],
            'priming_output_budgets': [], 'recorded_check_counts': {}, 'recorded_failed_checks': []}


def pair(cases, raw_cases):
    plains, result = {}, []
    for case in cases:
        if case['mode'] == 'plain':
            key = tuple(case[k] for k in PAIR_FIELDS)
            if key in plains:
                raise ValueError('Duplicate plain baseline for one cell')
            plains[key] = case
    for draft in [c for c in cases if c['mode'] == 'draft']:
        plain = plains.get(tuple(draft[k] for k in PAIR_FIELDS))
        identity, mismatch = None, None
        if plain:
            left, right = raw_cases[plain['case_id']]['output_tokens'], raw_cases[draft['case_id']]['output_tokens']
            identity = left == right
            mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                            min(len(left), len(right)) if not identity else None)
        result.append({'plain_case': plain['case_id'] if plain else None, 'draft_case': draft['case_id'],
                       'token_identity': identity, 'first_mismatch': mismatch,
                       'performance_qualified': identity is True,
                       'steady_decode_speedup': ratio(draft.get('steady_decode_tps'), plain.get('steady_decode_tps'))
                       if plain else None,
                       'request_speedup': ratio(plain.get('elapsed_seconds'), draft.get('elapsed_seconds'))
                       if plain else None,
                       'prompt_speedup': ratio(draft.get('library_prompt_tps'), plain.get('library_prompt_tps'))
                       if plain else None})
    return result


def ratio(top, bottom):
    return top / bottom if top is not None and bottom is not None and bottom > 0 else None


def build(inputs):
    sources, target_identity, context, prompt_identities = [], None, None, {}
    for path, label in inputs:
        if not NAME.fullmatch(label):
            raise ValueError('Use a short public label without paths or markup')
        data = path.read_bytes()
        raw = json.loads(data)
        if raw.get('schema') != 'ripple-dflash-benchmark-v1' or raw.get('public_prompts') != PROMPTS:
            raise ValueError('Expected a known public Ripple benchmark receipt')
        source = source_metadata(raw, label, hashlib.sha256(data).hexdigest())
        identity = source['source_versions']['config_sha256']['target']
        if target_identity is not None and identity != target_identity:
            raise ValueError('Target config or conversion identity differs across receipts')
        if context is not None and source['context'] != context:
            raise ValueError('Total context differs across receipts')
        target_identity, context = identity, source['context']
        raw_cases = {}
        for case in raw.get('cases', []):
            if case.get('priming'):
                source['excluded_priming_cases'] += 1
                budget = number(case.get('max_output_tokens'))
                if budget not in source['priming_output_budgets']:
                    source['priming_output_budgets'].append(budget)
                continue
            if case.get('warmup') or case.get('rep', 0) < 0:
                source['excluded_warmup_cases'] += 1
                continue
            value = cell(case, raw.get('prompt_arrays', {}))
            if value['case_id'] in raw_cases:
                raise ValueError('Duplicate benchmark case ID')
            if value['prompt_tokens'] + value['max_output_tokens'] > source['context']:
                raise ValueError('Case exceeds the total context')
            key = value['prompt'], value['prompt_tokens']
            if key in prompt_identities and prompt_identities[key] != value['prompt_sha256']:
                raise ValueError('Public prompt hash differs across equivalent cells')
            prompt_identities[key] = value['prompt_sha256']
            raw_cases[value['case_id']] = case
            source['cases'].append(value)
        source['comparisons'] = pair(source['cases'], raw_cases)
        for category in ['comparisons', 'chunk_comparisons', 'policy_comparisons']:
            source['recorded_check_counts'][category] = {'pass': 0, 'fail': 0}
            for check in raw.get(category, []):
                if type(check.get('token_identity')) is bool:
                    outcome = 'pass' if check['token_identity'] else 'fail'
                    source['recorded_check_counts'][category][outcome] += 1
                if check.get('token_identity') is False:
                    failed = {'category': category}
                    for key in ['plain_case', 'draft_case', 'reference_case', 'candidate_case', 'first_mismatch']:
                        if key in check:
                            failed[key] = number(check[key], optional=True)
                    source['recorded_failed_checks'].append(failed)
        source['timing_scope'] = ('short weight/kernel primer excluded; fresh KV per fresh-state cell'
                                  if source['excluded_priming_cases'] else 'no separate weight/kernel primer recorded')
        sources.append(source)
    drafter_ids = {json.dumps(s['source_versions']['config_sha256'].get('drafter'), sort_keys=True) for s in sources}
    return {'schema': 'ripple-dflash-public-comparison-v1', 'sources': sources,
            'notes': ['Each row is one attempt; repetitions and workloads are not pooled.',
                      'Speed ratios compare plain and drafted cells within the same receipt and settings.',
                      'Prompt rate is the generator-reported prompt metric; TTFT and request elapsed remain separate.',
                      'Short-chat prompt rates can include first-use costs; use the primed prefill phase for PP claims.',
                      'Rates from failed or unpaired identity cells are diagnostic measurements, not qualified speedups.',
                      'Sampled physical peaks use periodic process-footprint readings; RSS is a process lifetime peak.',
                      'Target identity uses config and conversion-receipt hashes; target shards were not rehashed here.',
                      'Different drafter conversion identities are separate measurements, not a quality comparison.']
            + (['The inputs include different drafter conversion identities.'] if len(drafter_ids) > 1 else [])}


def formatted(value):
    return '—' if value is None else f'{value:.2f}'


def markdown(artifact):
    lines = ['Each row is one attempt. Priming and warmups are excluded; settings are kept separate.', '',
             '| Run | Prompt | Temp / seed | Chunk | Draft policy | Tokens | Identity | Decode plain → draft tok/s |'
             ' Steady × | Request × | PP plain → draft tok/s | Draft peak MLX / physical GiB |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for source in artifact['sources']:
        cases = {case['case_id']: case for case in source['cases']}
        for comparison in source['comparisons']:
            draft, plain = cases[comparison['draft_case']], cases.get(comparison['plain_case'], {})
            identity = {True: 'PASS', False: 'FAIL', None: 'unpaired'}[comparison['token_identity']]
            width = draft['block_width'] if draft['block_width'] is not None else '?'
            policy = f"w{width} {'adaptive' if draft['adaptive'] else 'fixed'}, clear{draft['prefill_clear_interval']}"
            memory = draft['memory_gib']
            lines.append(f"| {source['label']} | {draft['prompt']} ({draft['state']}, r{draft['rep']}) | "
                         f"{draft['temperature']} / {draft['seed']} | {draft['chunk']} | {policy} | "
                         f"{draft['generation_tokens']} | {identity} | "
                         f"{formatted(plain.get('steady_decode_tps'))} → {formatted(draft['steady_decode_tps'])} | "
                         f"{formatted(comparison['steady_decode_speedup'])} | {formatted(comparison['request_speedup'])} | "
                         f"{formatted(plain.get('library_prompt_tps'))} → {formatted(draft['library_prompt_tps'])} | "
                         f"{formatted(memory.get('peak_mlx'))} / {formatted(memory.get('sampled_peak_physical_footprint'))} |")
    lines += ['', 'Sources:']
    for source in artifact['sources']:
        failures = sum(c['token_identity'] is False for c in source['comparisons'])
        pool = ratio(source['cache_pool_bytes'], 1024**2)
        chunks = source['recorded_check_counts']['chunk_comparisons']
        lines.append(f"- {source['label']}: {source['status']}; {source['timing_scope']}; "
                     f"cache pool {formatted(pool)} MiB; {failures} failed paired cells; "
                     f"cross-chunk checks {chunks['pass']} pass/{chunks['fail']} fail; "
                     f"{len(source['recorded_failed_checks'])} recorded failed checks.")
    lines += ['', *artifact['notes']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    if len(args.receipt) != len(args.label) or len(set(args.label)) != len(args.label):
        raise ValueError('Provide one unique public --label per --receipt')
    if args.output.exists() or (args.markdown and args.markdown.exists()):
        raise ValueError('Use new output files to preserve previous artifacts')
    if args.markdown and args.output.absolute() == args.markdown.absolute():
        raise ValueError('JSON and Markdown outputs must be different files')
    artifact = build(list(zip(args.receipt, args.label)))
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    if args.markdown:
        args.markdown.write_text(markdown(artifact))


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
