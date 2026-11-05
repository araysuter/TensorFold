#!/usr/bin/env python3
"""Measure quantization fidelity on synthetic teacher-forced text against streamed original BF16 weights."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
from transformers import AutoTokenizer
from modeling_ricursive_mlx import Layer, ModelArgs, RMSNorm
from convert import header


CORPUS = [
    'A clear explanation separates observations from assumptions. A measured cache hit does not prove a faster cold prompt. '
    'To compare two systems, hold the input, output limit, sampling settings and hardware constant. Report every run. ',
    'def is_prime(n):\n    if n < 2:\n        return False\n    for divisor in range(2, int(n ** 0.5) + 1):\n'
    '        if n % divisor == 0:\n            return False\n    return True\n\nassert is_prime(29)\nassert not is_prime(49)\n',
    'Seventeen times twenty-three is 391: 17 times 20 is 340, and 17 times 3 is 51. '
    'The sum is 391. If a train travels 180 kilometers in three hours, its average speed is 60 kilometers per hour. ',
    'An HTTP client sends a JSON request with a model identifier and a list of messages. '
    'The server checks authentication before allocating inference resources. Never include credentials in logs. '
    'A response can stream multiple chunks before its terminal event. Cancellation must release active work. ',
    'BEGIN TRANSACTION;\nCREATE TABLE inventory (sku TEXT PRIMARY KEY, quantity INTEGER NOT NULL CHECK (quantity >= 0));\n'
    'INSERT INTO inventory VALUES (\'PEN\', 12);\nUPDATE inventory SET quantity = quantity - 1 WHERE sku = \'PEN\';\nCOMMIT;\n',
    '{"name":"read_file","arguments":{"path":"src/main.py"}}\n'
    '{"status":"success","data":{"language":"python","lines":42}}\n'
    'Preserve the schema of each object. A number is different from a quoted numeric string. ',
    'function uniqueValues(values) {\n  return [...new Set(values)];\n}\n'
    'function sum(values) {\n  return values.reduce((total, value) => total + value, 0);\n}\n'
    'The empty sum is zero. A set removes repeated values while retaining insertion order. ',
    'Sliding attention retains a bounded local window, while global attention can see the full prior sequence. '
    'A model with alternating attention types must retain the correct cache separately for each layer. '
    'Normalization parameters and embedding scaling are part of the checkpoint definition. ',
]


def batch(source, output):
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    passages = [text + '\n\n' + CORPUS[(i + 3) % len(CORPUS)] + '\n\n' + CORPUS[(i + 6) % len(CORPUS)]
                for i, text in enumerate(CORPUS)]
    rows = [tokenizer.encode(text, add_special_tokens=False)[:256] for text in passages]
    length = max(map(len, rows))
    ids = np.full((len(rows), length), tokenizer.pad_token_id or 0, dtype=np.int32)
    positions = []
    labels = []
    for i, row in enumerate(rows):
        ids[i, :len(row)] = row
        for j in range(max(0, len(row) - 129), len(row) - 1):
            positions.append((i, j))
            labels.append(row[j + 1])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'inputs.npz', ids=ids, positions=np.array(positions), labels=np.array(labels))
    (output / 'corpus.json').write_text(json.dumps({'source': 'original synthetic text, code, JSON and arithmetic',
                                                  'prompt_token_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                                                  'sequences': len(rows), 'positions': len(labels)}, indent=2) + '\n')
    return ids, positions, labels


class Source:
    def __init__(self, root):
        self.root = root
        if (root / 'DOWNLOAD_INCOMPLETE').exists():
            raise ValueError('Original source download is incomplete')
        self.index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {file: header(root / file) for file in set(self.index.values())}

    def tensor(self, name):
        file = self.index[name]
        specs, offset = self.headers[file]
        spec = specs[name]
        if spec['dtype'] != 'BF16':
            raise ValueError('Reference requires original BF16 weights')
        view = np.memmap(self.root / file, dtype='<u2', mode='r', shape=tuple(spec['shape']),
                         offset=offset + spec['data_offsets'][0])
        value = mx.array(np.array(view, copy=True)).view(mx.bfloat16)
        mx.eval(value)
        return value


def original(args):
    source = Source(args.source)
    ids, positions, labels = batch(args.source, args.output)
    config = ModelArgs.from_dict(json.loads((args.source / 'config.json').read_text()))
    start = time.monotonic()
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    embedding.weight = source.tensor('model.embed_tokens.weight')
    hidden = embedding(mx.array(ids))
    if config.embed_scale is not None:
        hidden = hidden * config.embed_scale
    mx.eval(hidden)
    del embedding
    seed = hidden if config.use_valres else None
    for index, kind in enumerate(config.layer_types):
        layer = Layer(config, index)
        prefix = f'model.layers.{index}.'
        weights = [(key[len(prefix):], source.tensor(key)) for key in source.index if key.startswith(prefix)]
        layer.load_weights(weights, strict=True)
        mask = create_attention_mask(hidden, window_size=config.sliding_window if kind == 'sliding_attention' else None)
        hidden = layer(hidden, seed, mask, None)
        mx.eval(hidden)
        del layer, weights
        gc.collect()
        mx.clear_cache()
        if (index + 1) % 5 == 0:
            print(json.dumps({'event': 'reference_layer', 'completed': index + 1,
                              'peak_mlx_bytes': mx.get_peak_memory()}), flush=True)
    row, col = np.array(positions).T
    hidden = hidden[mx.array(row), mx.array(col)]
    norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight = source.tensor('model.norm.weight')
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    head.weight = source.tensor('lm_head.weight')
    logits = np.array(head(norm(hidden)).astype(mx.float32))
    np.save(args.output / 'original-bf16-logits.npy', logits)
    result = {'scope': 'Streamed original BF16 weights through MLX reference adapter; no full BF16 model in RAM',
              'source_revision': '4f308b9954b73cd200812095916c7baf68cf27f4', 'seconds': time.monotonic() - start,
              'peak_mlx_bytes': mx.get_peak_memory(), 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'positions': len(labels), 'finite': bool(np.isfinite(logits).all()),
              'upstream_pytorch_whole_model_reference': False}
    (args.output / 'original-reference.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'event': 'reference_complete', **result}), flush=True)


def quantized(args):
    data = np.load(args.output / 'inputs.npz')
    positions = data['positions']
    row, col = positions.T
    gold = data['labels']
    reference = mx.array(np.load(args.output / 'original-bf16-logits.npy'))
    if not bool(mx.all(mx.isfinite(reference)).item()):
        raise ValueError('Original reference contains nonfinite logits')
    model, _ = load_model(args.model, strict=True, trust_remote_code=True)
    start = time.monotonic()
    logits = model(mx.array(data['ids']))[mx.array(row), mx.array(col)].astype(mx.float32)
    log_ref = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    log_q = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(log_ref) * (log_ref - log_q), axis=-1)
    agreement = mx.mean((mx.argmax(logits, axis=-1) == mx.argmax(reference, axis=-1)).astype(mx.float32))
    selected = mx.arange(len(gold))
    base_nll = -mx.mean(log_ref[selected, mx.array(gold)])
    quant_nll = -mx.mean(log_q[selected, mx.array(gold)])
    mx.eval(kl, agreement, base_nll, quant_nll)
    result = {'scope': 'Synthetic teacher-forced positions; fidelity diagnostic, not an industry benchmark',
              'mean_kl_nats': mx.mean(kl).item(), 'max_kl_nats': mx.max(kl).item(),
              'top1_agreement': agreement.item(), 'original_bf16_mean_nll': base_nll.item(),
              'quantized_mean_nll': quant_nll.item(), 'seconds': time.monotonic() - start,
              'finite': bool(mx.all(mx.isfinite(logits)).item()), 'positions': len(gold),
              'peak_mlx_bytes': mx.get_peak_memory(),
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'drafter': False, 'native_tensorfold': False}
    name = args.label or args.model.name
    (args.output / (name + '-fidelity.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'event': 'quant_fidelity_complete', 'model': name, **result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['original', 'quantized'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label')
    args = parser.parse_args()
    mx.set_cache_limit(128 * 1024**2)
    mx.set_memory_limit(54 * 1024**3)
    (original if args.mode == 'original' else quantized)(args)
