#!/usr/bin/env python3
"""Preserve the pinned Swift snapshot's historical RMSNorm conversion for native serving.

Creates a separate checkpoint; never modifies the Hugging Face cache. Only the
161 RMSNorm vectors change: BF16(float32(stored BF16) + 1), as in the old
tensorfold.families.qwen3_5.swift.convert_norms. No inference dependency required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

REVISION = "82276731e47e1db4ac502f24c63cfaf886639be9"
SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
            ".q_norm.weight", ".k_norm.weight", "model.norm.weight")


def convert(data):
    result = bytearray()
    for (bits,) in struct.iter_unpack("<H", data):
        value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        added = struct.unpack("<I", struct.pack("<f", value + 1.0))[0]
        rounded = (added + 0x7fff + ((added >> 16) & 1)) >> 16
        result.extend(struct.pack("<H", rounded))
    return bytes(result)


def prepare(source, destination):
    if source.name != REVISION:
        raise ValueError("Only the pinned Swift snapshot is supported")
    config = json.loads((source / "config.json").read_text())
    if config.get("text_config", {}).get("hidden_size") != 5120:
        raise ValueError("Unexpected Swift configuration")
    jobs = []
    names = set()
    for shard in sorted(source.glob("*.safetensors")):
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
            for name, tensor in header.items():
                if not name.startswith("language_model.") or not name.endswith(SUFFIXES):
                    continue
                if tensor["dtype"] != "BF16" or len(tensor["shape"]) != 1 or name in names:
                    raise ValueError("Unexpected or duplicate normalization tensor: " + name)
                begin, end = tensor["data_offsets"]
                stream.seek(8 + size + begin)
                original = stream.read(end - begin)
                if len(original) != tensor["shape"][0] * 2:
                    raise ValueError("Truncated normalization tensor: " + name)
                jobs.append((shard.name, name, 8 + size + begin, original, convert(original)))
                names.add(name)
    if len(jobs) != 161:
        raise ValueError(f"Expected 161 Swift RMSNorm scales, found {len(jobs)}")
    destination.mkdir(parents=True, exist_ok=False)
    receipt = []
    try:
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            target = destination / path.name
            if path.suffix == ".safetensors":
                shutil.copyfile(path, target)
            else:
                target.symlink_to(path.resolve())
        for shard, name, offset, original, converted in jobs:
            with (destination / shard).open("r+b") as stream:
                stream.seek(offset)
                if stream.read(len(original)) != original:
                    raise ValueError("Copied tensor differs from source")
                stream.seek(offset)
                stream.write(converted)
            receipt.append({"name": name, "shard": shard,
                            "source_sha256": hashlib.sha256(original).hexdigest(),
                            "converted_sha256": hashlib.sha256(converted).hexdigest()})
        manifest = {"source_model": "ukisai/Swift-1.5-4bit-MLX", "source_revision": REVISION,
                    "conversion": "BF16(float32(stored BF16) + 1.0); round to nearest even",
                    "normalization_tensors": receipt}
        (destination / "swift-native.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(f"Prepared {len(receipt)} normalization vectors; all other tensor bytes preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
