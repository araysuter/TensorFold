#!/usr/bin/env python3
"""Verify native complete-state disk round trips with public synthetic prompts and token hashes."""
import argparse
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--key-file', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    token = args.key_file.read_text().strip().split(":", 1)[-1]
    report = {'model': args.model, 'status': 'running', 'checks': []}
    paths = [args.root / ('native-probe-' + x + '.tfs') for x in ['start', 'corrupt']]

    def call(path, body, expected=200, auth=True):
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer ' + token
        req = urllib.request.Request(args.url.rstrip('/') + path, data=json.dumps(body).encode(), headers=headers)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                code, response = r.status, json.load(r)
        except urllib.error.HTTPError as e:
            code, response = e.code, json.loads(e.read())
        if code != expected:
            raise RuntimeError(f'{path}: expected {expected}, got {code}: {response.get("error")}')
        return response, time.monotonic() - started

    def clear():
        return call('/v1/internal/session/clear', {})[0]

    def completion(messages, draft=True):
        body = dict(model=args.model, messages=messages, max_tokens=64, temperature=0, stream=False,
                    draft=draft, return_token_ids=True, chat_template_kwargs={'enable_thinking': False})
        response, elapsed = call('/v1/chat/completions', body)
        if response.get('error') or not response.get('tensorfold', {}).get('token_sha256'):
            raise RuntimeError('generation did not return a token hash')
        return response, elapsed, body

    try:
        clear()
        one = [{'role': 'user', 'content': 'Remember that the sample project code is QUARTZ-729. Reply with just OK.'}]
        first, _, _ = completion(one)
        saved, save_seconds = call('/v1/internal/session/save', {'path': str(paths[0])})
        if not saved['tokens'] or saved['bytes'] != paths[0].stat().st_size:
            raise RuntimeError('save did not commit a complete nonempty snapshot')
        follow = one + [first['choices'][0]['message'],
                        {'role': 'user', 'content': 'What was the sample project code? Reply with just the code.'}]
        body = dict(model=args.model, messages=follow, max_tokens=64, temperature=0,
                    chat_template_kwargs={'enable_thinking': False})
        match, _ = call('/v1/imprint/resume', body)
        if match.get('ram_tokens', 0) == 0:
            raise RuntimeError('canonical continuation does not match the saved state')
        cached, cached_seconds, _ = completion(follow)
        clear()
        loaded, load_seconds = call('/v1/internal/session/load', {'path': str(paths[0])})
        if any(saved[k] != loaded[k] for k in ['tokens', 'bytes', 'fingerprint']):
            raise RuntimeError('restore receipt differs from save')
        restored, restored_seconds, _ = completion(follow)
        if restored['usage']['prompt_tokens_details']['cached_tokens'] <= 0:
            raise RuntimeError('restored generation did not actually reuse a cache')
        if cached['tensorfold']['token_sha256'] != restored['tensorfold']['token_sha256']:
            raise RuntimeError('restored output differs from the same RAM-state continuation')
        report['checks'].append({'name': 'disk_roundtrip', 'passed': True, 'saved_tokens': saved['tokens'],
                                 'snapshot_bytes': saved['bytes'], 'fingerprint': saved['fingerprint'],
                                 'save_seconds': save_seconds, 'load_seconds': load_seconds,
                                 'cached_seconds': cached_seconds, 'restored_seconds': restored_seconds,
                                 'cached_tokens': restored['usage']['prompt_tokens_details']['cached_tokens'],
                                 'token_sha256': restored['tensorfold']['token_sha256']})
        clear()
        plain, _, _ = completion(follow, draft=False)
        if plain['tensorfold']['token_sha256'] != restored['tensorfold']['token_sha256']:
            raise RuntimeError('fresh plain output differs from restored drafted output')
        report['checks'].append({'name': 'fresh_plain_equal_restored_drafted', 'passed': True})
        clear()
        shutil.copyfile(paths[0], paths[1])
        with paths[1].open('r+b') as f:
            f.seek(16)
            old = f.read(1)
            f.seek(16)
            f.write(bytes([old[0] ^ 0x40]))
        call('/v1/internal/session/load', {'path': str(paths[1])}, expected=409)
        empty, _ = call('/v1/internal/session/save', {'path': str(paths[1])})
        if empty['tokens'] or empty['bytes']:
            raise RuntimeError('failed import leaked a visible RAM state')
        report['checks'].append({'name': 'corrupt_restore_rejected_without_visible_state', 'passed': True})
        call('/v1/internal/session/save', {'path': str(args.root / '../outside.tfs')}, expected=400)
        call('/v1/internal/session/clear', {}, expected=401, auth=False)
        report['checks'].append({'name': 'path_and_auth_confinement', 'passed': True})
        report['status'] = 'passed'
    except Exception as e:
        report['status'] = 'failed'
        report['error_type'] = type(e).__name__
        report['error'] = str(e)
        raise
    finally:
        try:
            clear()
        except Exception:
            pass
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        for p in paths:
            p.unlink(missing_ok=True)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
