#!/usr/bin/env python3
"""Prepare mixed 4/8-bit DFlash2 tensors while preserving the original draft architecture metadata."""
import argparse
import json
import math
from pathlib import Path
import shutil
import mlx.core as mx
import numpy as np
from convert import header, sha256


def run(args):
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
    quant = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
    params = 0
    mx.set_cache_limit(128 * 1024**2)
    for key, spec in sorted(specs.items()):
        if key == '__metadata__':
            continue
        if spec['dtype'] != 'BF16':
            raise ValueError('Unexpected original drafter dtype')
        shape = spec['shape']
        params += math.prod(shape)
        view = np.memmap(source, dtype='<u2', mode='r', offset=offset + spec['data_offsets'][0],
                         shape=tuple(shape))
        value = mx.array(np.array(view, copy=True)).view(mx.bfloat16)
        mx.eval(value)
        if len(shape) == 2 and key.endswith('.weight'):
            module = key[:-7]
            bits = 8 if module in ('fc', 'candidate_selector.hidden_projection') or 'kernel_projection' in module else 4
            setting = {'bits': bits, 'group_size': 64, 'mode': 'affine'}
            q, scales, biases = mx.quantize(value, **setting)
            mx.eval(q, scales, biases)
            output.update({key: q, module + '.scales': scales, module + '.biases': biases})
            quant[module] = setting
        else:
            output[key] = value
        del view, value
    path = args.output / 'model.safetensors'
    mx.save_safetensors(str(path), output, metadata={'format': 'mlx'})
    config = json.loads((args.source / 'config.json').read_text())
    config['quantization'] = quant
    config['quantization_config'] = quant
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.copy2(args.source / 'LICENSE', args.output / 'LICENSE')
    record = {'source_repo': manifest['repo_id'], 'source_revision': manifest['revision'],
              'source_sha256': expected['sha256'], 'source_parameters': params,
              'weight_bytes': sum(t.nbytes for t in output.values()), 'sha256': sha256(path),
              'format': 'MLX affine 4-bit; FC, selector projection and convolution projections 8-bit',
              'unquantized': 'BF16 norms, selector codebooks and convolution base kernels',
              'architecture': 'DFlash2, not a standalone Qwen3 language model',
              'target_generation_equivalence_validated': False, 'native_tensorfold_validated': False}
    (args.output / 'conversion-receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    (args.output / 'VALIDATION_PENDING').write_text('Requires a Ripple-compatible DFlash2 runtime and exactness checks.\n')
    (args.output / 'CONVERSION_INCOMPLETE').unlink()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
