#!/usr/bin/env python3
"""Compare the MLX adapter with the original PyTorch implementation on small deterministic models."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
try:
    import mlx.core as mx
    import numpy as np
    import torch
    from mlx.utils import tree_flatten
    from modeling_ricursive_mlx import Model, ModelArgs
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'torch', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: Ripple adapter tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


def reference(path):
    name = 'ripple_reference'
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    for module in ['configuration_ricursive', 'modeling_ricursive']:
        spec = importlib.util.spec_from_file_location(name + '.' + module, path / (module + '.py'))
        value = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = value
        spec.loader.exec_module(value)
    return sys.modules[name + '.configuration_ricursive'].RicursiveConfig, value.RicursiveForCausalLM


class AdapterTests(unittest.TestCase):
    def paired(self, *, bf16=False, **options):
        torch.manual_seed(391)
        config = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      vocab_size=128, sliding_window=8, max_position_embeddings=256,
                      embed_scale=8.0, ffn_clamp_threshold=0.2,
                      use_valres=True, use_gated_attention=True,
                      layer_types=['sliding_attention'] * 4 + ['full_attention'])
        config.update(options)
        config_cls, model_cls = reference(SOURCE)
        target = model_cls(config_cls(**config)).eval()
        dtype = torch.bfloat16 if bf16 else torch.float32
        target = target.to(dtype=dtype)
        model = Model(ModelArgs(**config))
        weights = []
        for key, tensor in target.state_dict().items():
            value = tensor.detach().float().numpy()
            weights.append((key, mx.array(value).astype(mx.bfloat16 if bf16 else mx.float32)))
        self.assertEqual(set(dict(tree_flatten(model.parameters()))), {k for k, _ in weights})
        model.load_weights(weights, strict=True)
        mx.eval(model.parameters())
        return target, model

    def test_original_forward_fp32_with_gate_value_seed_nope_and_clipping(self):
        target, model = self.paired()
        tokens = np.array([[7, 9, 2, 63, 7, 18, 32, 1, 11, 12, 3, 5, 97, 14, 4, 87, 3, 25, 26]])
        with torch.no_grad():
            expected = target(torch.tensor(tokens), use_cache=False).logits.numpy()
        with mx.stream(mx.cpu):
            actual = np.array(model(mx.array(tokens)))
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)

    def test_original_forward_bf16(self):
        target, model = self.paired(bf16=True)
        tokens = np.array([[7, 9, 2, 63, 7, 18, 32, 1, 11]])
        with torch.no_grad():
            expected = target(torch.tensor(tokens), use_cache=False).logits.float().numpy()
        actual = np.array(model(mx.array(tokens)).astype(mx.float32))
        normalized = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-9)
        self.assertLess(normalized, 0.025)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

    def test_prefill_chunks_and_rotating_cache_match_fresh(self):
        _, model = self.paired()
        tokens = mx.array([[7, 9, 2, 63, 7, 18, 32, 1, 11, 12, 3, 5, 97, 14, 4, 87, 3, 25, 26]])
        with mx.stream(mx.cpu):
            full = np.array(model(tokens))
            cache = model.make_cache()
            pieces = []
            for start, end in [(0, 5), (5, 12), (12, 13), (13, 19)]:
                pieces.append(np.array(model(tokens[:, start:end], cache=cache)))
        np.testing.assert_allclose(np.concatenate(pieces, axis=1), full, atol=2e-5, rtol=3e-4)
        self.assertLessEqual(cache[0].keys.shape[2], 14)
        self.assertEqual(cache[4].offset, 19)

    def test_switching_off_value_residual_and_gates(self):
        target, model = self.paired(use_valres=False, use_gated_attention=False)
        tokens = np.array([[1, 17, 6, 9]])
        with torch.no_grad():
            expected = target(torch.tensor(tokens), use_cache=False).logits.numpy()
        with mx.stream(mx.cpu):
            actual = np.array(model(mx.array(tokens)))
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)

    def test_gpu_bf16_cache_continuation_top_tokens(self):
        _, model = self.paired(bf16=True)
        tokens = mx.array([[7, 9, 2, 63, 7, 18, 32, 1, 11, 12, 3, 5, 97, 14, 4, 87, 3, 25, 26]])
        full = np.array(model(tokens).astype(mx.float32))
        cache = model.make_cache()
        prefix = model(tokens[:, :12], cache=cache)
        mx.eval(prefix)
        suffix = np.array(model(tokens[:, 12:], cache=cache).astype(mx.float32))
        np.testing.assert_array_equal(suffix.argmax(-1), full[:, 12:].argmax(-1))

    def test_invalid_pattern_refused(self):
        with self.assertRaises(ValueError):
            ModelArgs(num_hidden_layers=5, layer_types=['full_attention'])
        with self.assertRaises(ValueError):
            ModelArgs(hidden_act='gelu')

    def test_gpu_actual_head_width_and_ten_to_one_gqa(self):
        target, model = self.paired(bf16=True, hidden_size=128, intermediate_size=256,
                                   num_attention_heads=20, num_key_value_heads=2, head_dim=128)
        tokens = np.array([[7, 9, 2, 63, 7, 18, 32, 1, 11]])
        with torch.no_grad():
            expected = target(torch.tensor(tokens), use_cache=False).logits.float().numpy()
        actual = np.array(model(mx.array(tokens)).astype(mx.float32))
        normalized = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-9)
        self.assertLess(normalized, 0.025)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.source
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests))
    args.output.write_text(json.dumps({'time_unix': time.time(), 'tests': result.testsRun,
                                      'failures': len(result.failures), 'errors': len(result.errors),
                                      'passed': result.wasSuccessful()}, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
