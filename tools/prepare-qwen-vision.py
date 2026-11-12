#!/usr/bin/env python3
"""Attach an original BF16 Qwen vision tower without requantizing text weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def header(path):
    with path.open('rb') as stream:
        length, = struct.unpack('<Q', stream.read(8))
        if not 0 < length <= min(100 * 1024**2, path.stat().st_size - 8):
            raise ValueError('Invalid safetensors header')
        return json.loads(stream.read(length)), 8 + length


def attach(source, destination, repo, revision, drop_legacy=False):
    source, destination = Path(source), Path(destination)
    src_config = json.loads((source / 'config.json').read_text())
    config = json.loads((destination / 'config.json').read_text())
    vision = src_config.get('vision_config')
    width = config.get('text_config', config)['hidden_size']
    if not vision or vision['out_hidden_size'] != width or vision['depth'] != 27:
        raise ValueError('Vision geometry does not match the text model')
    index_path = destination / 'model.safetensors.index.json'
    if index_path.is_symlink() or (destination / 'config.json').is_symlink():
        raise ValueError('Prepare a private copy of configuration and index before attaching vision')
    index = json.loads(index_path.read_text())
    weights = index['weight_map']
    if any(k.startswith('vision_tower.') for k in weights):
        raise ValueError('Destination already has a vision tower')
    legacy = [k for k in weights if k.startswith(('visual.', 'model.visual.'))]
    if legacy and not drop_legacy:
        raise ValueError('Legacy vision index needs explicit native-only replacement')
    src_index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    if any(Path(name).name != name or not name.endswith('.safetensors')
           for name in [*src_index.values(), *weights.values()]):
        raise ValueError('Shard names must be local safetensors basenames')
    wanted = {k: v for k, v in src_index.items() if k.startswith(('model.visual.', 'vision_tower.'))}
    if len(wanted) != 333:
        raise ValueError('Expected 333 BF16 vision tensors')
    tables = {name: header(source / name) for name in set(wanted.values())}
    entries, jobs, offset = {}, [], 0
    for name, file in sorted(wanted.items()):
        table, start = tables[file]
        entry = table[name]
        begin, end = entry['data_offsets']
        size = end - begin
        if (entry['dtype'] != 'BF16' or any(type(d) is not int or d <= 0 for d in entry['shape'])
                or size != math.prod(entry['shape']) * 2 or begin < 0
                or end + start > (source / file).stat().st_size):
            raise ValueError('Malformed BF16 vision tensor')
        target = 'vision_tower.' + name.split('.', 2)[2] if name.startswith('model.visual.') else name
        entries[target] = dict(dtype='BF16', shape=entry['shape'], data_offsets=[offset, offset + size])
        jobs.append((target, source / file, start + begin, size))
        offset += size
    entries['__metadata__'] = {'format': 'mlx'}
    raw = json.dumps(entries, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    output = destination / 'model-vision-bf16.safetensors'
    temporary = output.with_suffix('.incomplete')
    if output.exists() or temporary.exists():
        raise ValueError('Refusing to overwrite vision output')
    records = []
    try:
        with temporary.open('xb') as out:
            out.write(struct.pack('<Q', len(raw)))
            out.write(raw)
            for name, file, start, size in jobs:
                digest = hashlib.sha256()
                with file.open('rb') as stream:
                    stream.seek(start)
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(1024**2, remaining))
                        if not chunk:
                            raise ValueError('Truncated source tensor')
                        out.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                records.append({'name': name, 'bytes': size, 'sha256': digest.hexdigest()})
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, output)
        for name in legacy:
            weights.pop(name)
        weights.update({name: output.name for name, *_ in jobs})
        index.setdefault('metadata', {})['total_size'] = sum(
            end - begin for file in set(weights.values())
            for name, entry in header(destination / file)[0].items()
            if name in weights and weights[name] == file
            for begin, end in [entry['data_offsets']])
        config['vision_config'] = vision
        config['language_model_only'] = False
        for name in ['preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json', 'LICENSE']:
            if (source / name).is_file():
                (destination / name).write_bytes((source / name).read_bytes())
        (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        index_path.write_text(json.dumps(index, indent=2) + '\n')
        receipt = {'source_model': repo, 'source_revision': revision, 'dtype': 'BF16',
                   'vision_tensor_count': len(jobs), 'vision_bytes': offset,
                   'text_tensor_bytes_changed': False, 'patch_layout': 'original HF NCDHW or existing MLX layout',
                   'legacy_index_removed': bool(legacy), 'tensors': records}
        (destination / 'vision-attachment.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source-repo', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--drop-legacy-visual-index', action='store_true')
    args = parser.parse_args()
    value = attach(args.source, args.destination, args.source_repo, args.source_revision, args.drop_legacy_visual_index)
    print(json.dumps({k: v for k, v in value.items() if k != 'tensors'}))
