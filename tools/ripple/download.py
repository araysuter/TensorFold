#!/usr/bin/env python3
"""Download pinned original Ripple checkpoints and verify every large-file checksum."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(16 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    from huggingface_hub import HfApi, hf_hub_download
    manifest = json.loads(args.manifest.read_text())
    if args.expected_account and HfApi().whoami()['name'] != args.expected_account:
        raise ValueError('Unexpected Hugging Face account')
    args.root.mkdir(parents=True, exist_ok=True)
    missing = 0
    for row in manifest['models']:
        destination = args.root / (row['repo_id'].split('/')[-1].lower() + '-bf16-' + row['revision'][:8])
        for entry in row['files']:
            path = destination / entry['path']
            if not path.exists() or path.stat().st_size != entry['size']:
                missing += entry['size']
    if missing and shutil.disk_usage(args.root).free < missing + 110 * 1024**3:
        raise ValueError('Insufficient space for original checkpoints, quants and reserve')
    receipt = {'started_unix': time.time(), 'models': []}
    for row in manifest['models']:
        name = row['repo_id'].split('/')[-1].lower() + '-bf16-' + row['revision'][:8]
        destination = args.root / name
        destination.mkdir(exist_ok=True)
        (destination / 'DOWNLOAD_INCOMPLETE').write_text('Download and verification in progress.\n')
        result = {'repo_id': row['repo_id'], 'revision': row['revision'], 'files': []}
        ordered = sorted(row['files'], key=lambda f: (f['path'].endswith('.safetensors'), f['path']))
        def download(entry):
            if entry['path'] == '.gitattributes':
                return None
            path = Path(hf_hub_download(row['repo_id'], entry['path'], revision=row['revision'],
                                        local_dir=destination))
            if path.stat().st_size != entry['size']:
                raise ValueError('File size mismatch: ' + entry['path'])
            actual = sha256(path)
            if entry.get('sha256') and actual != entry['sha256']:
                raise ValueError('Checksum mismatch: ' + entry['path'])
            if not entry.get('sha256') and entry.get('blob_id'):
                data = path.read_bytes()
                blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                if blob != entry['blob_id']:
                    raise ValueError('Source Git blob mismatch: ' + entry['path'])
            record = {'path': entry['path'], 'bytes': path.stat().st_size, 'sha256': actual}
            print(json.dumps({'event': 'verified', 'repo_id': row['repo_id'], 'file': entry['path'],
                              'bytes': path.stat().st_size, 'sha256': actual}), flush=True)
            return record
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            result['files'] = [entry for entry in pool.map(download, ordered) if entry is not None]
        (destination / 'source-verification.json').write_text(json.dumps(result, indent=2) + '\n')
        (destination / 'DOWNLOAD_INCOMPLETE').unlink()
        receipt['models'].append(result)
    receipt['finished_unix'] = time.time()
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'event': 'complete', 'models': len(receipt['models'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--expected-account')
    run(parser.parse_args())
