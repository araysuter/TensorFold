#!/usr/bin/env python3
"""Prepare reviewed model cards, stage complete checkpoints privately, then publish verified revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from convert import sha256


PATTERNS = ['README.md', 'LICENSE', 'VALIDATION.json', 'config.json', 'conversion-receipt.json',
            'model.safetensors.index.json', 'model-*.safetensors', 'modeling_ricursive_mlx.py',
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']


def read(path):
    return json.loads(path.read_text())


def prepare(args):
    selection = read(args.validation / 'selection.json')
    capacity = read(args.validation / 'mixed-4bit-capacity.json')
    if capacity['status'] != 'complete':
        raise ValueError('Four-bit target capacity probe has not completed')
    prepared = []
    for bits in [4, 2, 1]:
        choice = selection[str(bits)]
        model = Path(choice['model_path'])
        conversion = read(model / 'conversion-receipt.json')
        if conversion['source_repo'] != 'ricursiveAI/Ripple-65B-Preview':
            raise ValueError('Unexpected source model')
        if conversion['source_revision'] != '4f308b9954b73cd200812095916c7baf68cf27f4':
            raise ValueError('Unexpected source revision')
        if (model / 'CONVERSION_INCOMPLETE').exists():
            raise ValueError('Conversion is incomplete')
        if sha256(model / 'modeling_ricursive_mlx.py') != conversion['adapter_sha256']:
            raise ValueError('Unrecorded adapter change')
        smoke = read(args.validation / f'mixed-{bits}bit-smoke.json')
        if smoke['status'] != 'complete' or len(smoke['cases']) != 5:
            raise ValueError('Smoke/generation probes have not completed')
        fidelity = choice[choice['method']]
        if not fidelity['finite']:
            raise ValueError('Nonfinite fidelity result')
        for shard in conversion['shards']:
            if sha256(model / shard['file']) != shard['sha256']:
                raise ValueError('Output shard changed after conversion')
        speed = next(c for c in smoke['cases'] if c['name'] == 'cold-prefill-2048')
        measured_capacity = capacity if bits == 4 else None
        validation = {'source_revision': conversion['source_revision'], 'selected_method': choice['method'],
                      'weight_bytes': conversion['weight_bytes'], 'effective_bits_per_weight': conversion['effective_bits_per_weight'],
                      'fidelity': fidelity, 'speed': {k: speed[k] for k in ['prompt_tokens', 'generation_tokens',
                          'ttft_seconds', 'elapsed_seconds', 'prompt_tps', 'generation_tps']},
                      'peak_smoke_footprint_bytes': smoke.get('peak_physical_footprint_bytes'),
                      'target_only_capacity': {k: capacity[k] for k in ['status', 'peak_physical_footprint_bytes',
                          'before_paging', 'after_paging']} if measured_capacity else None,
                      'native_tensorfold': False, 'dflash2_exactness_validated': False,
                      'broad_quality_benchmark': False, 'source_code': args.source_code_url}
        (model / 'VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
        repo = f'araysuter/Ripple-65B-Preview-MLX-mixed-{bits}bit'
        extra = ('**Experimental low-bit MLP quantization. Expect substantial quality loss; this is not a reliable agent model.**\n\n'
                 'The Hub automatic parameter counter does not recognize this one-bit packing at release. '
                 'All 64,782,689,360 source parameters are retained; this is not a pruned 19B model.\n\n'
                 if bits == 1 else '**Reduced-fidelity variant. A smoke probe produced an incorrect `is_prime` function; '
                 'prefer mixed-4bit when memory permits.**\n\n' if bits == 2 else '')
        paging = capacity['after_paging']['swapouts'] - capacity['before_paging']['swapouts']
        paging_mib = paging * capacity['after_paging']['page_bytes'] / 1024**2
        cap_text = ('A target-only 131,008-token prompt plus up to 64 output tokens completed on the 64 GiB M5 Max. '
                    f'Observed peak process footprint: {capacity["peak_physical_footprint_bytes"] / 1024**3:.2f} GiB. '
                    f'The host-wide swap-out counter increased by {paging_mib:.1f} MiB during the run; '
                    'this is not per-process attribution or a zero-swap qualification. '
                    'This did not include a drafter or native gateway snapshots.' if measured_capacity else
                    'This profile did not receive a separate 128K capacity run; weight size is not a context-fit guarantee.')
        card = f'''---
license: apache-2.0
library_name: mlx
base_model: ricursiveAI/Ripple-65B-Preview
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- ricursive
- ripple
- quantized
- custom_code
---
# Ripple-65B Preview — MLX mixed {bits}-bit

{extra}Converted independently from the original instruction-tuned BF16 checkpoint,
revision `{conversion['source_revision']}`. No existing quantized model was requantized.
The pretraining-only Ripple Base model was not used. All **{conversion['source_parameters']:,} source parameters** are retained.

| Stored component | Precision |
| --- | --- |
| MLP matrices | {bits}-bit, groups of {32 if bits < 4 else 64} |
| Other attention projections | 4-bit, groups of 64 |
| Attention sigmoid gates, embedding and output head | 8-bit, groups of 64 |
| Norms and value-residual scalars | Original BF16 |

Weights occupy **{conversion['weight_bytes'] / 1e9:.2f} GB / {conversion['weight_bytes'] / 1024**3:.2f} GiB**.
The actual average is **{conversion['effective_bits_per_weight']:.3f} bits per parameter**, including affine
scales, biases and protected tensors. The name refers to the large MLP weights, not every parameter.
The encoder selected for this profile was `{choice['method']}`. MSE optimization uses per-group
clipping search and binary centroid refinement with an affine error fallback; it is not activation-calibrated GPTQ/AWQ.

## Run with MLX

This release includes a custom Ripple adapter using MLX-LM's `model_file` support.
Review `modeling_ricursive_mlx.py`; loading it requires the explicit trust option.
Tested with MLX 0.32.4 and MLX-LM 0.32.2.

```sh
pip install mlx==0.32.4 mlx-lm==0.32.2
python -m mlx_lm.generate --model {repo} --trust-remote-code \\
  --prompt 'Explain binary search in two sentences.' --max-tokens 512
```

The adapter preserves PeriNorm, Q/K norms, embedding scaling, local RoPE,
global NoPE, value-residual blending and clipped SwiGLU. Local KV is bounded.
The reference's FP32 activation computation and BF16 cast order are preserved.

## Measured fidelity and runtime scope

| Diagnostic | Result |
| --- | --- |
| Synthetic teacher-forced positions | {fidelity['positions']} |
| Top-token agreement with streamed original BF16 | {100*fidelity['top1_agreement']:.2f}% |
| Mean KL from original BF16 | {fidelity['mean_kl_nats']:.6f} nats |
| Synthetic next-token loss, BF16 / quant | {fidelity['original_bf16_mean_nll']:.4f} / {fidelity['quantized_mean_nll']:.4f} nats |
| Uncached 2K prompt processing, one user | {speed['prompt_tps']:.1f} tokens/s |
| Decode on that short retrieval continuation | {speed['generation_tps']:.1f} tokens/s ({speed['generation_tokens']} generated tokens) |

These are small synthetic diagnostics, not GPQA, Terminal-Bench, general task accuracy,
or the publisher's BF16 benchmark scores. The original weights were evaluated layer by layer
through the checked MLX reference adapter; this is not a whole-model PyTorch oracle.
Timing used plain MLX without a drafter, a resident model, fresh KV and 256-token prompt chunks.
Short continuations do not establish sustained throughput. Raw numeric scope is in `VALIDATION.json`.

{cap_text}

## DFlash2 and TensorFold

The matching draft is [Ripple-65B Preview DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2).
It requires a Ripple-aware DFlash2 runtime; it is not a standalone Qwen3 language model.
This release does **not** claim a tested DFlash2 path, distribution-preserving speculative
decoding, or native TensorFold API support. Native Ripple bring-up is tracked in
[TensorFold #2171](https://github.com/ashhart/TensorFold/issues/2171), with drafting planned separately.
Target-only 128K results cannot be applied to a drafter plus native snapshot configuration.

See the [conversion tools, source pins, adapter tests and limits]({args.source_code_url}).
Keep the original model's Apache-2.0 license; the license is included here.
'''
        (model / 'README.md').write_text(card)
        prepared.append({'repo_id': repo, 'model_path': str(model), 'method': choice['method']})
    args.prepared.write_text(json.dumps(prepared, indent=2) + '\n')
    print(json.dumps({'prepared_repositories': [p['repo_id'] for p in prepared]}))


def field(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def remote_verify(api, repo, model):
    info = api.model_info(repo, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    paths = set()
    for pattern in PATTERNS:
        paths.update(model.glob(pattern))
    for path in paths:
        item = remote.get(path.name)
        if item is None:
            raise ValueError('Missing uploaded artifact: ' + path.name)
        if item.lfs:
            if field(item.lfs, 'sha256') != sha256(path):
                raise ValueError('Uploaded LFS hash mismatch: ' + path.name)
        else:
            data = path.read_bytes()
            expected = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if item.blob_id != expected:
                raise ValueError('Uploaded Git blob mismatch: ' + path.name)
    return info.sha


def publish(args):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    if api.whoami()['name'] != 'araysuter':
        raise ValueError('Unexpected publishing account')
    records = read(args.state) if args.state.exists() else {}
    for row in read(args.prepared):
        repo, model = row['repo_id'], Path(row['model_path'])
        if not repo.startswith('araysuter/Ripple-65B-Preview-MLX-mixed-'):
            raise ValueError('Unexpected publication destination')
        conversion = read(model / 'conversion-receipt.json')
        if sha256(model / 'modeling_ricursive_mlx.py') != conversion['adapter_sha256']:
            raise ValueError('Adapter changed since validation preparation')
        for shard in conversion['shards']:
            if sha256(model / shard['file']) != shard['sha256']:
                raise ValueError('Weights changed since validation preparation')
        if args.mode == 'stage':
            if repo not in records:
                try:
                    api.model_info(repo)
                except RepositoryNotFoundError:
                    api.create_repo(repo, repo_type='model', private=True, exist_ok=False)
                else:
                    raise ValueError('Refusing to overwrite a repository not created by this staging run')
                records[repo] = {'created_private': True}
                args.state.write_text(json.dumps(records, indent=2) + '\n')
            api.upload_folder(repo_id=repo, folder_path=model, allow_patterns=PATTERNS,
                              commit_message='Add verified Ripple MLX checkpoint, adapter and scoped validation')
            commit = remote_verify(api, repo, model)
            records[repo].update(verified_commit=commit, remote_verified=True)
        else:
            record = records[repo]
            commit = remote_verify(api, repo, model)
            if not record.get('remote_verified') or commit != record['verified_commit']:
                raise ValueError('Staged repository changed before publication')
            card = api.model_info(repo).card_data
            if (card.base_model != 'ricursiveAI/Ripple-65B-Preview'
                    or card.base_model_relation != 'quantized'):
                raise ValueError('Quantization lineage must be explicit before publication')
            api.update_repo_settings(repo, private=False)
            with urllib.request.urlopen('https://huggingface.co/api/models/' + repo, timeout=30) as response:
                public = json.load(response)
            if public.get('private') or public['sha'] != commit:
                raise ValueError('Anonymous publication readback failed')
            record['public'] = True
        args.state.write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({'repo_id': repo, 'stage': args.mode, 'commit': records[repo].get('verified_commit')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'stage', 'publish'])
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--source-code-url', default='https://github.com/araysuter/TensorFold')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare' and args.validation is None:
        parser.error('--validation is required to prepare model cards')
    if args.mode != 'prepare' and args.state is None:
        parser.error('--state is required for staged publication')
    (prepare if args.mode == 'prepare' else publish)(args)
