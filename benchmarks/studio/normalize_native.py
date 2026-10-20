"""Combine native API prefill and matching completion-log queue durations without rerunning requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace

import benchmark


def normalize(data: dict, log: str) -> dict:
    pattern = r'done (\S+) prompt=(\d+) cached=(\d+) .*? tokens=(\d+) .*? queued=([0-9.]+)'
    queues = {m[1]: (int(m[2]), int(m[3]), int(m[4]), float(m[5])) for m in re.finditer(pattern, log)}
    for row in data['requests']:
        if row['status'] != 'ok':
            continue
        receipt = queues.get(row.get('request_id'))
        expected = (row['actual_prompt_tokens'], row['cached_tokens'], row['output_tokens'])
        if receipt is None or receipt[:3] != expected:
            raise ValueError('Missing or mismatched native queue receipt')
        active, queued = row['server_metrics']['prefill_seconds'], receipt[3]
        if active is None or active < 0:
            raise ValueError('Missing native prefill duration')
        row.update(native_prefill_seconds=active, native_queued_seconds=queued,
                   queue_prefill_seconds=active + queued,
                   prompt_timing_source='API prefill plus completion-log queue (queue rounded to 1 ms)')
        row['effective_input_tps'] = (expected[0] - expected[1]) / (active + queued) if active + queued else None
    data['config']['native_queue_timing'] = 'Matching response IDs; completion-log queue rounded to 1 ms'
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('server_log', type=Path)
    args = parser.parse_args()
    data = normalize(json.loads(args.results.read_text()), args.server_log.read_text(errors='replace'))
    benchmark.save(SimpleNamespace(**data['config']), data['requests'], data['batches'])
    print(f"Normalized {len(data['requests'])} native request receipts; no model requests were sent.")


if __name__ == '__main__':
    main()
