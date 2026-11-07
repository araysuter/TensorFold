#!/usr/bin/env python3
"""Single-trial Arete profile qualification on an isolated loopback gateway."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import signal
import struct
import sys
import threading
import time
import zlib
from urllib.parse import urlsplit

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / 'studio-gateway/tests'))
sys.path.insert(0, str(TOOLS / 'studio-gateway'))
from studio_smoke import Client, body, continuation, read_key, require, user
from memory import snapshot, group_footprint, GIB


def image_message(color):
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value))
    pixels = (b'\x00' + bytes(color) * 128) * 128
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
    return {'role': 'user', 'content': [{'type': 'text', 'text': 'Name the dominant color in this image. Answer one word.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}]}


def main(args):
    client = Client(args.base_url, read_key(args.key_file, args.key_label), 1800)
    result = {'status': 'running', 'checks': {}, 'samples': client.samples, 'quality_benchmark': False,
              'model': args.model, 'max_peak_physical_bytes': 0}
    if args.resume:
        previous = json.loads(args.resume.read_text())
        require(previous['model'] == args.model, 'Resume model differs')
        client.samples.extend(previous['samples'])
        result.update(checks=previous['checks'], max_peak_physical_bytes=previous['max_peak_physical_bytes'])
    stop = threading.Event()
    pane_pid = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', args.pane,
                                           '#{pane_pid}'], text=True))

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    def check(name, value):
        require(value, name)
        result['checks'][name] = True
        save()

    def native(path, value=None):
        run, = args.cache_root.glob('run-*')
        backend = Client(args.native_url, read_key(run / 'backend-key'), 1800)
        with backend.open('GET' if value is None else 'POST', path, value) as response:
            return json.load(response)

    def monitor():
        while not stop.wait(1):
            try:
                available, pressure = snapshot()
                raw = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,comm='], text=True, timeout=2)
                rows = [line.split(None, 3) for line in raw.splitlines()]
                parents = {int(pid): int(parent) for pid, parent, _, _ in rows}
                groups = set()
                for pid, _, group, command in rows:
                    if not command.endswith('/tensorfold-gateway-native'):
                        continue
                    ancestor = int(pid)
                    while ancestor in parents and ancestor != pane_pid:
                        ancestor = parents.pop(ancestor)
                    if ancestor == pane_pid:
                        groups.add(int(group))
                size = max((group_footprint(g) for g in groups), default=0)
                result['max_peak_physical_bytes'] = max(size, result['max_peak_physical_bytes'])
                if pressure != 1 or available < args.reserve_gib * GIB or size > args.max_footprint_gib * GIB:
                    result['status'] = 'memory_guard_stopped'
                    result['guard_failure'] = dict(available_bytes=available, pressure=pressure, footprint_bytes=size)
                    save()
                    for group in groups:
                        os.killpg(group, signal.SIGTERM)
                    return
            except (OSError, ValueError, subprocess.SubprocessError):
                continue

    def chat(prompt, draft=True, temperature=0):
        value = body(args.model, [user(prompt)], draft=draft)
        value.update(max_tokens=256, temperature=temperature, seed=43)
        return value

    try:
        threading.Thread(target=monitor, daemon=True).start()
        prompts = ['Explain how a hash table handles collisions in one paragraph.',
                   'Write a Python function that merges two sorted lists. Include a small test.']
        if args.skip_text_bench:
            prompts = []
            result['text_benchmark_skipped'] = True
        for index, prompt in enumerate(prompts):
            for temp in (0, 0.7):
                if result['checks'].get(f'draft_identity_{index}_{temp}'):
                    continue
                plain = client.complete('plain', chat(prompt, False, temp), f'plain-{index}-{temp}')
                drafted = client.complete('drafted', chat(prompt, True, temp), f'draft-{index}-{temp}')
                check(f'draft_identity_{index}_{temp}', plain['receipt']['token_sha256'] == drafted['receipt']['token_sha256'])
        first_body = body(args.model, [user('What is 17 plus 25? Reply with just the integer.')])
        first = client.complete('warm_seed', first_body, 'warm-context')
        check('arithmetic_smoke', first['message']['content'].strip() == '42')
        follow = body(args.model, continuation(first_body['messages'], first, 'Add 8. Reply with just the integer.'))
        warm = client.complete('warm_follow', follow, 'warm-context')
        check('warm_cache', warm['receipt']['cached_tokens'] > 0)
        restored_body = body(args.model, continuation(follow['messages'], warm,
                                                      'Subtract 1. Reply with just the integer.'))
        client.complete('switch_context', body(args.model, [user('Reply with OK.')]), 'other-context')
        resumed = client.complete('restored_follow', restored_body, 'warm-context')
        fresh = client.complete('fresh_follow', restored_body, 'fresh-context')
        check('restored_matches_fresh', resumed['receipt']['token_sha256'] == fresh['receipt']['token_sha256'])
        check('ssd_resume', resumed['receipt']['context_status'] == 'resumed' and resumed['receipt']['cached_tokens'] > 0)
        request = chat('List three benefits of unit tests.', temperature=0.7)
        solo = client.complete('solo', request, 'solo')
        with ThreadPoolExecutor(2) as pool:
            pairs = list(pool.map(lambda i: client.complete('concurrent', request, f'concurrent-{i}'), range(2)))
        check('concurrent_identity', all(p['receipt']['token_sha256'] == solo['receipt']['token_sha256'] for p in pairs))
        for i in range(6):
            client.complete('global_eviction', first_body, f'eviction-{i}')
            check(f'four_snapshots_{i}', len(list(args.cache_root.rglob('*.tfs'))) <= 4)
        if args.vision:
            for color, rgb in [('red', (255, 0, 0)), ('blue', (0, 0, 255))]:
                visual = body(args.model, [image_message(rgb)], draft=False)
                plain = client.complete('vision_plain_' + color, visual, 'vision-plain-' + color)
                visual['draft'] = True
                drafted = client.complete('vision_drafted_' + color, visual, 'vision-shared')
                fresh = client.complete('vision_fresh_' + color, visual, 'vision-fresh-' + color)
                check('vision_answer_' + color, color in drafted['message']['content'].lower())
                check('vision_draft_identity_' + color,
                      plain['receipt']['token_sha256'] == drafted['receipt']['token_sha256'])
                check('vision_image_identity_' + color,
                      fresh['receipt']['token_sha256'] == drafted['receipt']['token_sha256'])
                follow_image = body(args.model, continuation(visual['messages'], drafted,
                                                              'Repeat the color. Answer one word.'))
                warm_image = client.complete('vision_restored_' + color, follow_image, 'vision-shared')
                cold_image = client.complete('vision_follow_fresh_' + color, follow_image, 'vision-follow-fresh-' + color)
                check('vision_follow_identity_' + color,
                      warm_image['receipt']['token_sha256'] == cold_image['receipt']['token_sha256'])
                result.setdefault('vision_cache_reused', {})[color] = warm_image['receipt']['cached_tokens'] > 0
        if args.full_window:
            repeats = 130944
            for _ in range(4):
                long = body(args.model, [user('Synthetic capacity filler.\n' + ' maple' * repeats + '\nReply with OK.')])
                long['max_tokens'] = 8
                count = native('/v1/tokenize', long)['count']
                if count == 130944:
                    break
                repeats += 130944 - count
            check('exact_full_window_prompt', count == 130944)
            cold = client.complete('near_128k', long, 'full-window')
            check('full_window', cold['receipt']['prompt_tokens'] == 130944 and cold['receipt']['cached_tokens'] == 0)
            result['memory_at_capacity'] = native('/memory')
            client.complete('capacity_away', first_body, 'capacity-away')
            extended = body(args.model, continuation(long['messages'], cold, 'Again reply with OK.'))
            extended['max_tokens'] = 8
            restored = client.complete('full_window_restored', extended, 'full-window')
            check('full_window_ssd', restored['receipt']['context_status'] == 'resumed'
                  and restored['receipt']['cached_tokens'] >= 130900)
        result['status'] = 'passed'
    except Exception as exc:
        if result['status'] != 'memory_guard_stopped':
            result['status'] = 'failed'
        result['error_type'] = type(exc).__name__
        raise
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--base-url', default='http://127.0.0.1:19080')
    parser.add_argument('--native-url', default='http://127.0.0.1:19081')
    parser.add_argument('--model', default='arete-27b')
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--key-label', default='owner')
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pane', required=True, help='Owned isolated gateway pane, used to identify its backend group')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--full-window', action='store_true')
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--skip-text-bench', action='store_true')
    parser.add_argument('--max-footprint-gib', type=float, default=44)
    parser.add_argument('--reserve-gib', type=float, default=6)
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error('--run requires an isolated gateway owned by this test operation')
    if not 0 < arguments.max_footprint_gib <= 54 or not 4 <= arguments.reserve_gib <= 16:
        parser.error('Use a physical cap up to 54 GiB and a host reserve from 4 through 16 GiB')
    for value in [arguments.base_url, arguments.native_url]:
        url = urlsplit(value)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.port in (8080, 18080)
                or url.username or url.password or url.query or url.fragment or url.path not in ('', '/')):
            parser.error('Use isolated loopback ports, not the production gateway')
    if arguments.output.exists():
        parser.error('Use a new receipt path')
    main(arguments)
