#!/usr/bin/env python3
"""Exercise real BF16 streaming conversion, MLX strict reload and corruption refusal on tiny models."""
import argparse
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model
    from modeling_ricursive_mlx import Model, ModelArgs
    import convert
    import fidelity
    import quantization
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: Ripple conversion tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


class ConversionTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        config = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      vocab_size=128, sliding_window=8, embed_scale=8.0)
        model = Model(ModelArgs(**config))
        model.set_dtype(mx.bfloat16)
        tensors = dict(tree_flatten(model.parameters()))
        mx.eval(tensors)
        file = source / 'model.safetensors'
        mx.save_safetensors(str(file), tensors)
        (source / 'config.json').write_text(json.dumps(config | {'model_type': 'ricursive'}))
        (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':
                                                         {k: file.name for k in tensors}}))
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'chat_template.jinja', 'LICENSE']:
            (source / name).write_text('{}')
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'models': [{'repo_id': 'example/ripple-fixture', 'revision': '0' * 40,
                                                    'files': [{'path': file.name, 'size': file.stat().st_size,
                                                               'sha256': convert.sha256(file)}]}]}))
        return source, manifest, tensors

    def test_all_profiles_roundtrip_with_original_norms_and_explicit_mixed_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, manifest, tensors = self.fixture(root)
            convert.run(SimpleNamespace(source=source, manifest=manifest, output=root / 'output',
                                        bits=[4, 2, 1], wait_seconds=1, shard_mib=1, method='mse'))
            for bits in [4, 2, 1]:
                path = root / 'output' / f'mixed-{bits}bit'
                self.assertFalse((path / 'CONVERSION_INCOMPLETE').exists())
                self.assertTrue((path / 'VALIDATION_PENDING').exists())
                model, config = load_model(path, strict=True, trust_remote_code=True)
                weights = dict(tree_flatten(model.parameters()))
                for key, value in tensors.items():
                    if 'norm' in key or key.endswith('value_res_lambda'):
                        np.testing.assert_array_equal(np.array(value.astype(mx.float32)),
                                                      np.array(weights[key].astype(mx.float32)))
                self.assertEqual(config['quantization']['model.layers.0.mlp.up_proj']['bits'], bits)
                self.assertEqual(config['quantization']['model.layers.0.self_attn.g_proj']['bits'], 8)
                logits = model(mx.array([[7, 8, 9]]))
                self.assertTrue(bool(mx.all(mx.isfinite(logits)).item()))
                receipt = json.loads((path / 'conversion-receipt.json').read_text())
                self.assertEqual(receipt['weight_bytes'], sum(x.nbytes for x in weights.values()))

    def test_corrupt_original_shard_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, manifest, _ = self.fixture(root)
            path = source / 'model.safetensors'
            with path.open('r+b') as file:
                file.seek(-1, 2)
                file.write(b'!')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                convert.run(SimpleNamespace(source=source, manifest=manifest, output=root / 'output',
                                            bits=[4], wait_seconds=1, shard_mib=1, method='affine'))
            self.assertTrue((root / 'output' / 'CONVERSION_INCOMPLETE').exists())

    def test_streamed_original_reference_matches_whole_tiny_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _, tensors = self.fixture(root)
            config = json.loads((source / 'config.json').read_text())
            model = Model(ModelArgs.from_dict(config))
            model.load_weights(list(tensors.items()), strict=True)
            ids = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.int32)
            positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
            expected = np.array(model(mx.array(ids)).astype(mx.float32))[[0, 0, 1, 1], [1, 2, 1, 2]]
            output = root / 'reference'
            output.mkdir()
            with patch('fidelity.batch', return_value=(ids, positions, [9, 10, 13, 14])):
                fidelity.original(SimpleNamespace(source=source, output=output))
            np.testing.assert_array_equal(np.load(output / 'original-bf16-logits.npy'), expected)

    def test_group_error_never_exceeds_standard_affine_and_binary_outliers_improve(self):
        mx.random.seed(833)
        weights = mx.random.normal((128, 128)).astype(mx.bfloat16)
        weights[:, 0] = 12
        weights[:, 64] = -12
        for bits in [1, 2, 4]:
            group = 64 if bits == 4 else 32
            baseline = mx.quantize(weights, bits=bits, group_size=group)
            candidate = quantization.quantize(weights, bits=bits, group_size=group, method='mse')
            original_error = quantization.error_groups(weights, *baseline, group, bits)
            improved_error = quantization.error_groups(weights, *candidate, group, bits)
            self.assertTrue(bool(mx.all(improved_error <= original_error).item()))
            if bits == 1:
                self.assertLess(mx.mean(improved_error).item(), mx.mean(original_error).item() * 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
