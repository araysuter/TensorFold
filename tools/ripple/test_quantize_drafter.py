#!/usr/bin/env python3
"""Check drafter precision policy, original-tensor coverage, packed reload and conversion refusals."""
import contextlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import quantize_drafter
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from convert import sha256
    from dflash_model import DraftArgs, DraftModel, load_draft
    HAS_MLX = True
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy'):
        raise
    HAS_MLX = False


class PolicyTests(unittest.TestCase):
    def test_precision_budget_preserves_selector_and_dynamic_kernel_projections(self):
        for name in ['fc.weight', 'candidate_selector.hidden_projection.weight',
                     'layers.0.attention_conv.kernel_projection.weight', 'layers.4.mlp_conv.kernel_projection.weight']:
            for bits in [4, 8]:
                with self.subTest(name=name, bits=bits):
                    self.assertEqual(quantize_drafter.policy(name, [64, 320], bits),
                                     dict(bits=8, group_size=64, mode='affine'))
        for name in ['layers.0.self_attn.q_proj.weight', 'layers.4.mlp.up_proj.weight']:
            self.assertEqual(quantize_drafter.policy(name, [64, 64])['bits'], 4)
            self.assertEqual(quantize_drafter.policy(name, [64, 64], 8)['bits'], 8)

    def test_original_nonprojection_tensors_remain_lossless_bf16(self):
        for name, shape in [('norm.weight', [64]), ('hidden_norm.weight', [64]),
                            ('candidate_selector.predecessor_codebook', [128, 64]),
                            ('candidate_selector.successor_codebook', [128, 64]),
                            ('layers.0.attention_conv.base_kernel', [2, 4, 64])]:
            for bits in [4, 8]:
                self.assertIsNone(quantize_drafter.policy(name, shape, bits))

    def test_unsupported_precision_refused_before_import_or_output_mutation(self):
        with self.assertRaisesRegex(ValueError, '4 or 8'):
            quantize_drafter.run(SimpleNamespace(bits=2))
        with self.assertRaisesRegex(ValueError, '4 or 8'):
            quantize_drafter.policy('layers.0.mlp.up_proj.weight', [64, 64], 2)


@unittest.skipUnless(HAS_MLX, 'Requires the Apple Silicon MLX conversion environment')
class ConversionTests(unittest.TestCase):
    def fixture(self, root, *, dtype=None):
        source = root / 'source'
        source.mkdir()
        config = dict(architectures=['DFlash2DraftModel'], hidden_size=64, intermediate_size=128,
                      num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      vocab_size=128, sliding_window=8, layer_types=['sliding_attention'] * 4 + ['full_attention'],
                      no_rope_layers=[4], attention_bias=False, is_causal=False, hidden_act='silu',
                      dflash_config=dict(block_size=16, conv_kernel_size=4, conv_group_size=16,
                      selector_rank=64, selector_top_k=8, mask_token_id=127, target_layer_ids=[1, 20, 39, 58, 77]))
        mx.random.seed(728)
        model = DraftModel(DraftArgs.from_dict(config))
        tensors = dict(tree_flatten(model.parameters()))
        tensors = {name: mx.random.normal(value.shape).astype(dtype or mx.bfloat16) * 0.02
                   for name, value in tensors.items()}
        tensors = {name: value.astype(dtype or mx.bfloat16) for name, value in tensors.items()}
        model.load_weights(list(tensors.items()), strict=True)
        mx.eval(tensors)
        file = source / 'model.safetensors'
        mx.save_safetensors(str(file), tensors)
        (source / 'config.json').write_text(json.dumps(config))
        (source / 'LICENSE').write_text('Apache-2.0 fixture\n')
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(dict(models=[{}, dict(repo_id='example/drafter-fixture', revision='0' * 40,
            files=[dict(path=file.name, size=file.stat().st_size, sha256=sha256(file))])])))
        return model, tensors, SimpleNamespace(source=source, manifest=manifest, output=root / 'output')

    def convert(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            quantize_drafter.run(args)
        return args.output

    def test_default_and_explicit_four_bit_produce_identical_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, args = self.fixture(root)
            default = self.convert(args)
            explicit = self.convert(SimpleNamespace(**(vars(args) | dict(bits=4, output=root / 'explicit'))))
            for name in ['model.safetensors', 'config.json', 'LICENSE']:
                self.assertEqual((default / name).read_bytes(), (explicit / name).read_bytes())
            receipt = json.loads((default / 'conversion-receipt.json').read_text())
            self.assertEqual(receipt['linear_bits'], 4)
            self.assertEqual(receipt['method'], 'affine')

    def test_all_original_tensors_survive_strict_reload_and_runtime_native_quantization_matches(self):
        squared_errors = {}
        for bits in [4, 8]:
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                model, tensors, args = self.fixture(root)
                args.bits = bits
                path = self.convert(args)
                expected_modules = dict(tree_flatten(model.leaf_modules(),
                                        is_leaf=lambda item: isinstance(item, nn.Module)))
                expected_bits = {name: (8 if name in ['fc', 'candidate_selector.hidden_projection']
                                       or name.endswith('kernel_projection') else bits)
                                 for name, module in expected_modules.items() if isinstance(module, nn.Linear)}
                self.assertEqual(len(expected_bits), 47)
                nn.quantize(model, group_size=64, bits=bits,
                            class_predicate=lambda name, module: dict(bits=expected_bits[name], group_size=64,
                            mode='affine') if name in expected_bits else False)
                loaded = load_draft(path)
                actual = dict(tree_flatten(loaded.parameters()))
                expected = dict(tree_flatten(model.parameters()))
                self.assertEqual(set(actual), set(expected))
                for name in expected:
                    self.assertTrue(bool(mx.array_equal(actual[name], expected[name]).item()), name)
                packed = mx.load(str(path / 'model.safetensors'))
                self.assertEqual(set(packed), set(expected))
                for name, tensor in tensors.items():
                    if not name.endswith('.weight') or len(tensor.shape) != 2:
                        self.assertEqual(packed[name].dtype, mx.bfloat16)
                        self.assertTrue(bool(mx.array_equal(packed[name], tensor).item()), name)
                config = json.loads((path / 'config.json').read_text())
                explicit = {name: setting['bits'] for name, setting in config['quantization'].items()
                            if isinstance(setting, dict)}
                self.assertEqual(explicit, expected_bits)
                squared_error = 0.0
                for name, selected in expected_bits.items():
                    restored = mx.dequantize(packed[name + '.weight'], packed[name + '.scales'],
                                            packed[name + '.biases'], bits=selected, group_size=64)
                    error = mx.sum((restored.astype(mx.float32) - tensors[name + '.weight'].astype(mx.float32))**2)
                    squared_error += float(error.item())
                squared_errors[bits] = squared_error
                receipt = json.loads((path / 'conversion-receipt.json').read_text())
                self.assertEqual(receipt['source_parameters'], sum(math.prod(x.shape) for x in tensors.values()))
                self.assertEqual(receipt['weight_bytes'], sum(x.nbytes for x in packed.values()))
                self.assertEqual(receipt['mlp_bits'], bits)
                self.assertEqual(receipt['sha256'], sha256(path / 'model.safetensors'))
                self.assertFalse(receipt['target_generation_equivalence_validated'])
                self.assertFalse((path / 'CONVERSION_INCOMPLETE').exists())
                self.assertTrue((path / 'VALIDATION_PENDING').exists())
        self.assertLess(squared_errors[8], squared_errors[4])

    def test_corrupt_or_incomplete_original_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, args = self.fixture(root)
            args.output.mkdir()
            sentinel = args.output / 'retained'
            sentinel.write_text('existing output')
            with self.assertRaisesRegex(ValueError, 'must be new'):
                self.convert(args)
            self.assertEqual(sentinel.read_text(), 'existing output')
            args.output = root / 'new'
            marker = args.source / 'DOWNLOAD_INCOMPLETE'
            marker.touch()
            with self.assertRaisesRegex(ValueError, 'failed verification'):
                self.convert(args)
            self.assertFalse(args.output.exists())
            marker.unlink()
            source = args.source / 'model.safetensors'
            source.write_bytes(source.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'failed verification'):
                self.convert(args)
            self.assertFalse(args.output.exists())

    def test_physical_guard_and_unexpected_source_dtype_leave_incomplete_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, args = self.fixture(Path(folder))
            with patch('probe.footprint', return_value=57 * 1024**3), self.assertRaises(MemoryError):
                self.convert(args)
            self.assertTrue((args.output / 'CONVERSION_INCOMPLETE').exists())
            self.assertFalse((args.output / 'conversion-receipt.json').exists())
        with tempfile.TemporaryDirectory() as folder:
            _, _, args = self.fixture(Path(folder), dtype=mx.float32)
            with self.assertRaisesRegex(ValueError, 'Unexpected original drafter dtype'):
                self.convert(args)
            self.assertTrue((args.output / 'CONVERSION_INCOMPLETE').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
