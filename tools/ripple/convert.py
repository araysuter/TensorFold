#!/usr/bin/env python3
"""Stream verified original BF16 Ripple tensors into independent mixed MLX quantizations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import time
from quantization import quantize


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(16 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def policy(name, shape, bits):
    if len(shape) != 2 or not name.endswith('.weight'):
        return None
    module = name[:-7]
    if module.endswith('g_proj') or module in ('model.embed_tokens', 'lm_head'):
        return {'bits': 8, 'group_size': 64, 'mode': 'affine'}
    if '.mlp.' in module:
        return {'bits': bits, 'group_size': 32 if bits < 4 else 64, 'mode': 'affine'}
    return {'bits': 4, 'group_size': 64, 'mode': 'affine'}


def header(path):
    with path.open('rb') as source:
        length = struct.unpack('<Q', source.read(8))[0]
        if length > 16 * 1024**2:
            raise ValueError('Unexpected safetensors header length')
        return json.loads(source.read(length)), 8 + length


def wait_file(path, seconds):
    deadline = time.monotonic() + seconds
    announced = False
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Source file did not finish downloading: ' + path.name)
        if not announced:
            print(json.dumps({'event': 'waiting_for_source', 'file': path.name}), flush=True)
            announced = True
        time.sleep(5)


def run(args):
    import mlx.core as mx
    import numpy as np
    mx.set_cache_limit(128 * 1024**2)
    mx.set_memory_limit(24 * 1024**3)
    source = json.loads(args.manifest.read_text())['models'][0]
    source_files = {x['path']: x for x in source['files']}
    index_path = args.source / 'model.safetensors.index.json'
    wait_file(index_path, args.wait_seconds)
    source_index = json.loads(index_path.read_text())
    index_keys = set(source_index['weight_map'])
    if args.output.exists():
        raise ValueError('Output directory must be new; partial conversions cannot be served')
    args.output.mkdir(parents=True)
    (args.output / 'CONVERSION_INCOMPLETE').write_text('Conversion and final coverage checks are pending.\n')
    states = {}
    for bits in args.bits:
        path = args.output / f'mixed-{bits}bit'
        path.mkdir()
        (path / 'CONVERSION_INCOMPLETE').write_text('Not safe to load or upload.\n')
        states[bits] = {'path': path, 'pending': {}, 'pending_bytes': 0, 'shards': [],
                        'weight_map': {}, 'quantization': {'bits': 4, 'group_size': 64, 'mode': 'affine'},
                        'weight_bytes': 0, 'squared_error': 0.0, 'source_squared': 0.0}
    parameters = 0
    seen = set()
    verified = []

    def flush(state):
        if not state['pending']:
            return
        name = f'model-{len(state["shards"]) + 1:05d}.safetensors'
        path = state['path'] / name
        mx.save_safetensors(str(path), state['pending'], metadata={'format': 'mlx'})
        state['shards'].append({'file': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
        for key in state['pending']:
            if key in state['weight_map']:
                raise ValueError('Duplicate output key: ' + key)
            state['weight_map'][key] = name
        state['pending'].clear()
        state['pending_bytes'] = 0
        mx.clear_cache()

    for filename in sorted(set(source_index['weight_map'].values())):
        path = args.source / filename
        wait_file(path, args.wait_seconds)
        expected = source_files[filename]
        if path.stat().st_size != expected['size'] or sha256(path) != expected['sha256']:
            raise ValueError('Original checkpoint checksum mismatch: ' + filename)
        specs, offset = header(path)
        verified.append({'file': filename, 'sha256': expected['sha256']})
        for key, spec in sorted(specs.items()):
            if key == '__metadata__':
                continue
            if key not in index_keys or source_index['weight_map'][key] != filename or key in seen:
                raise ValueError('Source tensor index mismatch: ' + key)
            if spec['dtype'] != 'BF16':
                raise ValueError('Unexpected source tensor dtype: ' + key)
            shape = spec['shape']
            mapped = np.memmap(path, dtype='<u2', mode='r', offset=offset + spec['data_offsets'][0],
                               shape=tuple(shape))
            value = mx.array(np.array(mapped, copy=True)).view(mx.bfloat16)
            mx.eval(value)
            del mapped
            parameters += math.prod(shape)
            for bits, state in states.items():
                setting = policy(key, shape, bits)
                values = {key: value}
                if setting:
                    q, scales, biases = quantize(value, **setting, method=args.method)
                    mx.eval(q, scales, biases)
                    module = key[:-7]
                    values = {key: q, module + '.scales': scales, module + '.biases': biases}
                    state['quantization'][module] = setting
                    restored = mx.dequantize(q, scales, biases, **setting)
                    error = mx.sum((restored.astype(mx.float32) - value.astype(mx.float32))**2)
                    energy = mx.sum(value.astype(mx.float32)**2)
                    mx.eval(error, energy)
                    state['squared_error'] += error.item()
                    state['source_squared'] += energy.item()
                    del restored, error, energy
                size = sum(t.nbytes for t in values.values())
                if state['pending_bytes'] + size > args.shard_mib * 1024**2:
                    flush(state)
                state['pending'].update(values)
                state['pending_bytes'] += size
                state['weight_bytes'] += size
                del values
            seen.add(key)
            del value
            mx.clear_cache()
        for state in states.values():
            flush(state)
        print(json.dumps({'event': 'converted_source_shard', 'file': filename,
                          'source_tensors': len(seen), 'peak_mlx_bytes': mx.get_peak_memory(),
                          'output_weight_bytes': {b: s['weight_bytes'] for b, s in states.items()}}), flush=True)
    if seen != index_keys:
        raise ValueError('Incomplete source tensor coverage')
    for bits, state in states.items():
        config = json.loads((args.source / 'config.json').read_text())
        config.pop('auto_map', None)
        config['model_file'] = 'modeling_ricursive_mlx.py'
        config['quantization'] = state['quantization']
        config['quantization_config'] = state['quantization']
        path = state['path']
        (path / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        index = {'metadata': {'total_size': state['weight_bytes']}, 'weight_map': state['weight_map']}
        (path / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'chat_template.jinja', 'LICENSE']:
            shutil.copy2(args.source / name, path / name)
        shutil.copy2(Path(__file__).with_name('modeling_ricursive_mlx.py'), path / 'modeling_ricursive_mlx.py')
        receipt = {'source_repo': source['repo_id'], 'source_revision': source['revision'],
                   'source_files': verified, 'source_parameters': parameters, 'mlp_bits': bits,
                   'format': 'MLX affine, mixed precision, directly from original BF16',
                   'method': args.method, 'adapter_sha256': sha256(path / 'modeling_ricursive_mlx.py'),
                   'protected_modules': '8-bit embedding, output head and attention gates; other attention 4-bit',
                   'weight_bytes': state['weight_bytes'], 'effective_bits_per_weight':
                       state['weight_bytes'] * 8 / parameters,
                   'relative_weight_rmse': math.sqrt(state['squared_error'] / state['source_squared']),
                   'peak_conversion_mlx_bytes': mx.get_peak_memory(), 'shards': state['shards'],
                   'quality_validated': False, 'native_tensorfold_validated': False, 'context_128k_validated': False}
        (path / 'conversion-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (path / 'VALIDATION_PENDING').write_text('Conversion complete; inference, quality and capacity validation pending.\n')
        (path / 'CONVERSION_INCOMPLETE').unlink()
    (args.output / 'CONVERSION_INCOMPLETE').unlink()
    print(json.dumps({'event': 'complete', 'source_parameters': parameters, 'source_tensors': len(seen)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bits', type=int, nargs='+', choices=[1, 2, 4], default=[4, 2, 1])
    parser.add_argument('--shard-mib', type=int, default=768)
    parser.add_argument('--wait-seconds', type=int, default=21600)
    parser.add_argument('--method', choices=['affine', 'mse'], default='affine')
    run(parser.parse_args())
