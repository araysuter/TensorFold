#!/usr/bin/env python3
"""Run uncached MLX smoke/speed/capacity probes with a bounded physical-memory guard."""
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import re
import struct
import subprocess
import threading
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def footprint():
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    buffer = ctypes.create_string_buffer(4096)
    if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer)) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot read physical footprint')
    return struct.unpack_from('Q', buffer.raw, 72)[0]


def paging():
    raw = subprocess.check_output(['vm_stat']).decode()
    counters = {}
    for name in ['Pageouts', 'Swapouts']:
        match = re.search(r'^' + name + r':\s*(\d+)', raw, re.MULTILINE)
        if match:
            counters[name.lower()] = int(match.group(1))
    match = re.search(r'page size of (\d+) bytes', raw)
    counters['page_bytes'] = int(match.group(1)) if match else None
    return counters


def long_prompt(tokenizer, count):
    prefix = tokenizer.encode('Read the synthetic record and identify its verification code.\n',
                              add_special_tokens=False)
    filler = tokenizer.encode('This is an ordinary entry describing a local software test.\n',
                              add_special_tokens=False)
    marker = tokenizer.encode('\nThe unique verification code is JADE-4821-COMET.\n', add_special_tokens=False)
    suffix = tokenizer.encode('\nRepeat the verification code.\nThe verification code is', add_special_tokens=False)
    room = count - len(prefix) - len(marker) - len(suffix)
    values = (filler * (room // len(filler) + 1))[:room]
    middle = len(values) // 2
    result = prefix + values[:middle] + marker + values[middle:] + suffix
    assert len(result) == count
    return result


def main(args):
    if (args.model / 'CONVERSION_INCOMPLETE').exists():
        raise ValueError('Incomplete model conversion')
    if args.output.exists():
        raise ValueError('Use a new receipt to preserve previous attempts')
    mx.set_cache_limit(128 * 1024**2)
    mx.set_memory_limit(54 * 1024**3)
    row = {'model': args.model.name, 'phase': args.phase, 'runtime': 'MLX-LM custom Ripple adapter',
           'drafter': False, 'native_tensorfold': False, 'prefix_cache': False, 'kv_precision': 'BF16',
           'prefill_step_size': args.prefill_step_size, 'memory_guard_gib': 56, 'cases': [], 'status': 'loading',
           'packages': {p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'transformers']}}
    lock = threading.Lock()
    done = threading.Event()

    def save():
        with lock:
            args.output.write_text(json.dumps(row, indent=2) + '\n')

    def monitor():
        while not done.wait(0.25):
            current = footprint()
            row['peak_physical_footprint_bytes'] = max(row.get('peak_physical_footprint_bytes', 0), current)
            if current > 56 * 1024**3:
                row['status'] = 'memory_guard_stopped'
                save()
                os._exit(75)

    row['initial_physical_footprint_bytes'] = footprint()
    row['before_swap'] = subprocess.check_output(['sysctl', 'vm.swapusage']).decode().strip()
    row['before_paging'] = paging()
    save()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.monotonic()
    model, tokenizer = load(str(args.model), trust_remote_code=True)
    row['load_seconds'] = time.monotonic() - start
    row['status'] = 'running'
    save()

    def run(name, prompt, maximum):
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
        case = {'name': name, 'prompt_tokens': len(ids), 'max_output_tokens': maximum,
                'prompt_ids_sha256': hashlib.sha256(struct.pack('<' + 'I' * len(ids), *ids)).hexdigest()}
        row['active_case'] = name
        save()
        start = time.monotonic()
        first = None
        output = []
        last = None
        progress_mark = -4096

        def progress(processed, total):
            nonlocal progress_mark
            if processed - progress_mark >= 4096 or processed == total:
                progress_mark = processed
                print(json.dumps({'event': 'prefill', 'case': name, 'tokens': processed, 'total': total,
                                  'seconds': time.monotonic() - start, 'footprint_bytes': footprint()}), flush=True)

        for response in stream_generate(model, tokenizer, ids, max_tokens=maximum, sampler=make_sampler(temp=0),
                                        prompt_cache=model.make_cache(), prefill_step_size=args.prefill_step_size,
                                        prompt_progress_callback=progress):
            if first is None:
                first = time.monotonic() - start
            last = response
            output.append(response.text)
        if last is None:
            raise ValueError('No model output')
        case.update(text=''.join(output), ttft_seconds=first, elapsed_seconds=time.monotonic() - start,
                    generation_tokens=last.generation_tokens, prompt_tps=last.prompt_tps,
                    generation_tps=last.generation_tps, finish_reason=getattr(last, 'finish_reason', None),
                    peak_mlx_bytes=mx.get_peak_memory(), physical_footprint_bytes=footprint(),
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        row['cases'].append(case)
        save()
        print(json.dumps({'event': 'case_complete', **case}), flush=True)

    if args.phase == 'smoke':
        for name, content in [('math', 'What is 17 multiplied by 23? Give the final answer.'),
                              ('python', 'Write a short Python function is_prime(n) that handles n less than 2.'),
                              ('json', 'Reply with exactly this JSON object: {"ready":true}'),
                              ('summary', 'Explain sliding-window attention in two sentences.')]:
            run(name, tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                   tokenize=False, add_generation_prompt=True), 512)
        run('cold-prefill-2048', long_prompt(tokenizer, 2048), 64)
    elif args.phase == 'capacity':
        run('capacity-131008-plus64-output', long_prompt(tokenizer, 131008), 64)
    else:
        for count in [2048, 8192, 32768]:
            run(f'cold-prefill-{count}', long_prompt(tokenizer, count), 64)
    row['status'] = 'complete'
    row.pop('active_case', None)
    row['after_swap'] = subprocess.check_output(['sysctl', 'vm.swapusage']).decode().strip()
    row['after_paging'] = paging()
    save()
    done.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'capacity', 'prefill'], required=True)
    parser.add_argument('--prefill-step-size', type=int, choices=[128, 256, 512], default=256)
    main(parser.parse_args())
