#!/usr/bin/env python3
"""Reproduce Arete's MLX affine 5-bit text weights with its original BF16 vision tower."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

REVISION = '108f0afbdcfaa1ceeda4a635004304dfff6d4da2'
REPO = 'NousResearch/Arete-27B'


def main(args):
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    from mlx_lm import convert

    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing output directory')
    source = Path(snapshot_download(REPO, revision=REVISION, local_dir=args.source, max_workers=4))
    mx.set_cache_limit(64 * 1024**2)
    mx.set_memory_limit(32 * 1024**3)
    started = time.monotonic()
    convert(str(source), mlx_path=str(args.output), quantize=True, q_bits=5,
            q_group_size=64, q_mode='affine', dtype='bfloat16', trust_remote_code=False)
    path = Path(__file__).resolve().parents[1] / 'prepare-qwen-vision.py'
    spec = importlib.util.spec_from_file_location('prepare_vision', path)
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)
    vision.attach(source, args.output, REPO, REVISION)
    receipt = {'source_model': REPO, 'source_revision': REVISION, 'bits': 5, 'group_size': 64,
               'mode': 'affine', 'text_only': False, 'vision_encoder_included': True, 'vision_dtype': 'BF16',
               'seconds': time.monotonic() - started,
               'packages': {p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'transformers']},
               'weight_bytes': sum(p.stat().st_size for p in args.output.glob('*.safetensors'))}
    (args.output / 'conversion-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
