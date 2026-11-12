#!/usr/bin/env python3
"""Prepare affine 4/8-bit DFlash2 tensors while preserving the original draft architecture metadata."""
import argparse
import json
import math
from pathlib import Path
import shutil


def policy(name, shape, bits=4):
    if bits not in (4, 8):
        raise ValueError('Drafter bits must be 4 or 8')
    if len(shape) != 2 or not name.endswith('.weight'):
        return None
    module = name[:-7]
    selected = 8 if module in ('fc', 'candidate_selector.hidden_projection') or 'kernel_projection' in module else bits
    return {'bits': selected, 'group_size': 64, 'mode': 'affine'}


def run(args):
    bits = getattr(args, 'bits', 4)
    if bits not in (4, 8):
        raise ValueError('Drafter bits must be 4 or 8')
    import mlx.core as mx
    import numpy as np
    from convert import header, sha256
    from probe import footprint
    manifest = json.loads(args.manifest.read_text())['models'][1]
    expected = next(f for f in manifest['files'] if f['path'] == 'model.safetensors')
    source = args.source / 'model.safetensors'
    if sha256(source) != expected['sha256'] or (args.source / 'DOWNLOAD_INCOMPLETE').exists():
        raise ValueError('Original drafter is incomplete or failed verification')
    if args.output.exists():
        raise ValueError('Output directory must be new')
    args.output.mkdir(parents=True)
    (args.output / 'CONVERSION_INCOMPLETE').write_text('Drafter conversion is not complete.\n')
    specs, offset = header(source)
    output = {}
    quant = {'bits': bits, 'group_size': 64, 'mode': 'affine'}
    params = 0
    mx.set_cache_limit(128 * 1024**2)
    mx.set_memory_limit(54 * 1024**3)
    peak_physical = 0

    def check_memory():
        nonlocal peak_physical
        peak_physical = max(peak_physical, footprint())
        if peak_physical > 56 * 1024**3:
            raise MemoryError('Drafter conversion exceeded the 56 GiB physical-memory guard')

    check_memory()
    for key, spec in sorted(specs.items()):
        if key == '__metadata__':
            continue
        if spec['dtype'] != 'BF16':
            raise ValueError('Unexpected original drafter dtype')
        check_memory()
        shape = spec['shape']
        params += math.prod(shape)
        view = np.memmap(source, dtype='<u2', mode='r', offset=offset + spec['data_offsets'][0],
                         shape=tuple(shape))
        value = mx.array(np.array(view, copy=True)).view(mx.bfloat16)
        mx.eval(value)
        setting = policy(key, shape, bits)
        if setting is not None:
            module = key[:-7]
            q, scales, biases = mx.quantize(value, **setting)
            mx.eval(q, scales, biases)
            output.update({key: q, module + '.scales': scales, module + '.biases': biases})
            quant[module] = setting
        else:
            output[key] = value
        del view, value
        check_memory()
    path = args.output / 'model.safetensors'
    mx.save_safetensors(str(path), output, metadata={'format': 'mlx'})
    check_memory()
    config = json.loads((args.source / 'config.json').read_text())
    config['quantization'] = quant
    config['quantization_config'] = quant
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.copy2(args.source / 'LICENSE', args.output / 'LICENSE')
    record = {'source_repo': manifest['repo_id'], 'source_revision': manifest['revision'],
              'source_sha256': expected['sha256'], 'source_parameters': params,
              'weight_bytes': sum(t.nbytes for t in output.values()), 'sha256': sha256(path),
              'linear_bits': bits, 'mlp_bits': bits, 'method': 'affine',
              'format': ('MLX affine 4-bit; FC, selector projection and convolution projections 8-bit'
                         if bits == 4 else 'MLX affine 8-bit for all linear projections'),
              'peak_conversion_mlx_bytes': mx.get_peak_memory(), 'sampled_peak_physical_footprint_bytes': peak_physical,
              'limits': {'mlx_bytes': 54 * 1024**3, 'physical_bytes': 56 * 1024**3},
              'physical_guard_sampling': 'Before and after each source tensor and final serialization',
              'unquantized': 'BF16 norms, selector codebooks and convolution base kernels',
              'architecture': 'DFlash2, not a standalone Qwen3 language model',
              'target_generation_equivalence_validated': False, 'native_tensorfold_validated': False}
    (args.output / 'conversion-receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    (args.output / 'VALIDATION_PENDING').write_text(
        'Requires a Ripple-compatible DFlash2 runtime and exactness checks.\n')
    (args.output / 'CONVERSION_INCOMPLETE').unlink()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bits', type=int, choices=[4, 8], default=4)
    run(parser.parse_args())
