#!/usr/bin/env python3
"""Compare Ripple's MLX drafter with pinned upstream SGLang ASTs and CPU kernel adapters."""
import argparse
import hashlib
from functools import wraps
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
try:
    import numpy as np
    import torch
    import dflash_oracle as oracle
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('numpy', 'torch'):
        raise
    if __name__ == '__main__':
        print('SKIP: DFlash architecture tests require the documented NumPy and PyTorch environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple NumPy and PyTorch environment') from exc
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from dflash_model import DraftArgs, DraftModel, GroupedConv, load_draft, ordered_kv
    HAS_MLX = True
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm'):
        raise
    HAS_MLX = False


SOURCE = None
METRICS = []
SELECTOR_DIAGNOSTICS = []
EXECUTION_DEVICE = None


def cpu_fp32(method):
    @wraps(method)
    def run(*args, **kwargs):
        global EXECUTION_DEVICE
        previous = EXECUTION_DEVICE
        EXECUTION_DEVICE = 'cpu (explicit mx.stream(mx.cpu))'
        try:
            with mx.stream(mx.cpu):
                return method(*args, **kwargs)
        finally:
            EXECUTION_DEVICE = previous
    return run


def config(**overrides):
    value = dict(architectures=['DFlash2DraftModel'], hidden_size=64, intermediate_size=128,
                 num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                 vocab_size=128, sliding_window=8, layer_types=['sliding_attention'] * 4 + ['full_attention'],
                 no_rope_layers=[4], attention_bias=False, is_causal=False, rms_norm_eps=1e-6, rope_theta=10000.0,
                 hidden_act='silu', dflash_config=dict(block_size=16, conv_kernel_size=4, conv_group_size=16,
                 selector_rank=64, selector_top_k=8, mask_token_id=127, target_layer_ids=[1, 20, 39, 58, 77]))
    value.update(overrides)
    return value


def initialize(model, bf16=False):
    torch.manual_seed(91221)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            tensor.normal_(0, 0.04)
            if 'norm.weight' in name:
                tensor.add_(1)
            if name.endswith('base_kernel'):
                tensor[:, 0].add_(1)
    return model.to(dtype=torch.bfloat16 if bf16 else torch.float32)


def mlx_weights(model):
    weights = []
    for name, tensor in model.state_dict().items():
        converted = [(name, tensor)]
        if '.qkv_proj.' in name:
            index = int(name.split('.')[1])
            attention = model.layers[index].self_attn
            tensors = tensor.split([attention.q_size, attention.kv_size, attention.kv_size])
            converted = [(name.replace('qkv_proj', part + '_proj'), value)
                         for part, value in zip(('q', 'k', 'v'), tensors)]
        elif '.gate_up_proj.' in name:
            converted = [(name.replace('gate_up_proj', part + '_proj'), value)
                         for part, value in zip(('gate', 'up'), tensor.chunk(2))]
        for key, value in converted:
            weights.append((key, mx.array(value.float().numpy()).astype(
                mx.bfloat16 if value.dtype == torch.bfloat16 else mx.float32)))
    return weights


def paired(bf16=False, **overrides):
    settings = config(**overrides)
    expected = initialize(oracle.make_reference(SOURCE, settings), bf16)
    actual = DraftModel(DraftArgs.from_dict(settings))
    actual.load_weights(mlx_weights(expected), strict=True)
    mx.eval(actual.parameters())
    return settings, expected, actual


def tensor_pair(shape, bf16=False, seed=392):
    value = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    value = value.to(torch.bfloat16 if bf16 else torch.float32)
    actual = mx.array(value.float().numpy()).astype(mx.bfloat16 if bf16 else mx.float32)
    return value, actual


def compare(test, actual, expected, label, bf16=False):
    actual = np.asarray(actual.astype(mx.float32))
    expected = expected.detach().float().numpy()
    error = actual - expected
    normalized = np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12)
    metric = dict(label=label, dtype='bf16' if bf16 else 'fp32', normalized_l2=float(normalized),
                  mlx_device=EXECUTION_DEVICE or str(mx.default_device()),
                  max_absolute=float(np.max(np.abs(error))),
                  worst_column_l2=float(np.max(np.linalg.norm(error.reshape(-1, error.shape[-1]), axis=0))))
    METRICS.append(metric)
    if bf16:
        test.assertLess(normalized, 0.025, metric)
        np.testing.assert_allclose(actual, expected, atol=0.065, rtol=0.035, err_msg=label)
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=4e-4, err_msg=label)


class OracleTests(unittest.TestCase):
    def test_source_integrity_and_full_upstream_forward(self):
        model = initialize(oracle.make_reference(SOURCE, config()))
        features = [torch.randn(9, 64) for _ in range(5)]
        hidden, past = oracle.materialize_context(model, features)
        result = oracle.run_reference(model, torch.randn(16, 64), past, 9)
        self.assertEqual(tuple(hidden.shape), (9, 64))
        self.assertEqual(tuple(result.shape), (16, 64))
        self.assertTrue(torch.isfinite(result).all())

    def test_convolution_block_reset_and_fault_control(self):
        upstream = oracle.load_reference(SOURCE)
        model = upstream.DFlashGroupedConv(32, 16, 4, 16)
        with torch.no_grad():
            model.base_kernel.fill_(0.2)
            model.kernel_projection.weight.fill_(0.01)
        values = torch.arange(32 * 32).reshape(32, 32).float() / 1000
        prepared, coefficients = model.prepare(values)
        delta = model.kernel_projection(values).reshape(32, 2, 4, 2)
        manual = torch.zeros_like(values)
        faulty = torch.zeros_like(values)
        for row in range(32):
            for channel in range(32):
                for tap in range(4):
                    coefficient = 0.2 + delta[row, 0, tap, channel // 16]
                    if row >= tap:
                        faulty[row, channel] += coefficient * values[row - tap, channel]
                    if row % 16 >= tap:
                        manual[row, channel] += coefficient * values[row - tap, channel]
        torch.testing.assert_close(prepared, manual)
        self.assertGreater((faulty - prepared).abs().max().item(), 0.1)
        torch.testing.assert_close(model.finish(values, coefficients), manual)

    def test_bf16_fused_activation_rounding_and_fault_control(self):
        gate = torch.linspace(-5, 5, 128).to(torch.bfloat16)
        up = torch.linspace(2, -3, 128).to(torch.bfloat16)
        result = oracle.SiluAndMul()(torch.cat((gate, up)))
        value = gate.float()
        expected = (value / (1 + torch.exp(-value)) * up.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        faulty = torch.nn.functional.silu(gate) * up
        self.assertGreater(torch.count_nonzero(result != faulty).item(), 0)

    def test_reference_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'oracle.py'
            path.write_text(Path(SOURCE).read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                oracle.load_reference(path)


@unittest.skipUnless(HAS_MLX, 'Requires Apple Silicon MLX and MLX-LM')
class DraftArchitectureTests(unittest.TestCase):
    @cpu_fp32
    def test_grouped_convolution_pre_post_and_short_blocks(self):
        upstream = oracle.load_reference(SOURCE)
        for length in [1, 3, 16, 32]:
            expected = upstream.DFlashGroupedConv(64, 16, 4, 16)
            initialize(expected)
            actual = GroupedConv(DraftArgs.from_dict(config()))
            actual.load_weights([(name, mx.array(value.detach().numpy()))
                                 for name, value in expected.state_dict().items()], strict=True)
            source, values = tensor_pair((length, 64))
            pre, coefficients = actual.prepare(values[None])
            with torch.no_grad():
                padded = torch.nn.functional.pad(source, (0, 0, 0, max(0, 4 - length)))
                ref_pre, ref_coefficients = expected.prepare(padded)
                ref_post = expected.finish(ref_pre, ref_coefficients)[:length]
                ref_pre = ref_pre[:length]
            compare(self, pre[0], ref_pre, f'conv-pre-{length}')
            compare(self, actual.finish(pre, coefficients)[0], ref_post, f'conv-post-{length}')

    def check_architecture(self, bf16=False, separated_head=False, **overrides):
        settings, expected, actual = paired(bf16, **overrides)
        source_features, features = [], []
        for index in range(5):
            reference, value = tensor_pair((19, settings['hidden_size']), bf16, seed=392 + index)
            source_features.append(reference)
            features.append(value[None])
        with torch.no_grad():
            projected, past = oracle.materialize_context(expected, source_features)
        cache = actual.make_cache()
        actual.append_context(features, cache)
        compare(self, actual.hidden_norm(actual.fc(mx.concatenate(features, -1)))[0], projected,
                'target-taps-fc-hidden-norm', bf16)
        for index, entry in enumerate(cache):
            key, value = ordered_kv(entry)
            ref_key, ref_value = past[index]
            compare(self, key[0], ref_key[:, -key.shape[2]:], f'context-key-{index}', bf16)
            compare(self, value[0], ref_value[:, -value.shape[2]:], f'context-value-{index}', bf16)
            past[index] = (ref_key[:, -key.shape[2]:], ref_value[:, -value.shape[2]:])
        inputs, values = tensor_pair((16, settings['hidden_size']), bf16, seed=1009)
        with torch.no_grad():
            reference = oracle.run_reference(expected, inputs, past, 19)
        result = actual(values[None], cache)[0]
        compare(self, result, reference, 'five-draft-blocks-final-norm', bf16)
        self.assertEqual([entry.offset for entry in cache], [19] * 5)
        head = torch.nn.Linear(settings['hidden_size'], settings['vocab_size'], bias=False)
        head = head.to(dtype=inputs.dtype)
        head.org_vocab_size = settings['vocab_size']
        with torch.no_grad():
            head.weight.copy_(torch.randn(head.weight.shape, generator=torch.Generator().manual_seed(87)) * 0.1)
        expected.lm_head = head
        mlx_head = nn.Linear(settings['hidden_size'], settings['vocab_size'], bias=False)
        selector_hidden = mx.array(reference[1:].float().numpy()).astype(values.dtype)
        mlx_head.load_weights([('weight', mx.array(head.weight.detach().float().numpy()).astype(values.dtype))])
        with torch.no_grad():
            random_ids, _ = expected.compute_candidates(reference[1:])
            random_logits = (reference[1:] @ head.weight.T).float().numpy()
        observed_ids, _ = actual.candidate_selector.lattice(selector_hidden, mlx_head, 7)
        observed_logits = np.asarray(mlx_head(selector_hidden).astype(mx.float32))
        cutoff = settings['dflash_config']['selector_top_k']
        sorted_ref = np.sort(random_logits, -1)[:, ::-1]
        sorted_observed = np.sort(observed_logits, -1)[:, ::-1]
        mismatches = np.flatnonzero(np.any(np.sort(np.asarray(observed_ids), -1) !=
                                          np.sort(random_ids.numpy(), -1), axis=-1))
        SELECTOR_DIAGNOSTICS.append(dict(dtype=str(inputs.dtype), hidden_size=settings['hidden_size'],
            top_k=cutoff, fixture='random head; informational when cutoff ties',
            mismatched_slots=mismatches.tolist(),
            reference_cutoff_margins=(sorted_ref[:, cutoff - 1] - sorted_ref[:, cutoff]).tolist(),
            observed_cutoff_margins=(sorted_observed[:, cutoff - 1] - sorted_observed[:, cutoff]).tolist(),
            max_absolute_head_logit_error=float(np.max(np.abs(random_logits - observed_logits))),
            mismatched_reference_ids=random_ids.numpy()[mismatches].tolist(),
            mismatched_observed_ids=np.asarray(observed_ids)[mismatches].tolist()))
        if separated_head:
            column = int(reference[1:].float().abs().amin(0).argmax())
            with torch.no_grad():
                head.weight.zero_()
                head.weight[:, column] = torch.arange(1, settings['vocab_size'] + 1) * 0.125
            mlx_head.load_weights([('weight', mx.array(head.weight.detach().float().numpy()).astype(values.dtype))])
        with torch.no_grad():
            candidate_ids, unary = expected.compute_candidates(reference[1:])
            lattice = expected.candidate_selector.build_lattice(candidate_ids=candidate_ids[None],
                      unary_logits=unary[None], hidden_states=reference[None, 1:], anchor_token_ids=torch.tensor([7]))
        ids, scores = actual.candidate_selector.lattice(selector_hidden, mlx_head, 7)
        ids = np.asarray(ids)
        reference_ids = candidate_ids.numpy()
        np.testing.assert_array_equal(np.sort(ids, -1), np.sort(reference_ids, -1))
        if separated_head:
            with torch.no_grad():
                ordered = (reference[1:] @ head.weight.T).float().sort(-1, descending=True).values
            margin = float((ordered[:, cutoff - 1] - ordered[:, cutoff]).min())
            self.assertGreater(margin, 0)
            SELECTOR_DIAGNOSTICS.append(dict(fixture='separated single-column head',
                hidden_size=settings['hidden_size'], top_k=cutoff, minimum_cutoff_margin=margin,
                candidate_sets_exact=True))
        positions = np.argsort(ids, axis=-1)
        scores = np.asarray(scores.astype(mx.float32))
        canonical = []
        ref_scores = lattice[0].float().numpy()
        for slot in range(len(ids)):
            pred = positions[slot - 1] if slot else np.arange(len(ids[0]))
            canonical.append(scores[slot][pred][:, positions[slot]])
            ref_pred = np.argsort(reference_ids[slot - 1]) if slot else np.arange(len(ids[0]))
            ref_scores[slot] = ref_scores[slot][ref_pred][:, np.argsort(reference_ids[slot])]
        np.testing.assert_allclose(canonical, ref_scores, atol=0.065 if bf16 else 2e-5,
                                   rtol=0.035 if bf16 else 4e-4)

    @cpu_fp32
    def test_fp32_full_architecture_context_and_selector(self):
        self.check_architecture()

    def test_bf16_full_architecture_context_and_selector(self):
        self.check_architecture(bf16=True)

    def test_bf16_actual_head_width_and_ten_to_one_gqa(self):
        selector = config()['dflash_config'] | dict(selector_rank=256, selector_top_k=32)
        self.check_architecture(bf16=True, separated_head=True, hidden_size=128, intermediate_size=256,
                                num_attention_heads=20, num_key_value_heads=2, head_dim=128,
                                dflash_config=selector)

    def test_nope_and_bidirectional_sliding_attention_fault_controls(self):
        _, expected, actual = paired()
        source, value = tensor_pair((16, 64))
        for index in [0, 4]:
            attention = actual.layers[index].self_attn
            first, _ = attention.project_kv(value[None], 0)
            shifted, _ = attention.project_kv(value[None], 19)
            delta = np.max(np.abs(np.asarray(first - shifted)))
            self.assertEqual(delta == 0, index == 4)
        cache = actual.make_cache()
        first = actual.layers[0].self_attn(value[None], cache[0])
        changed = value.at[-1].add(10)
        second = actual.layers[0].self_attn(changed[None], cache[0])
        self.assertGreater(float(mx.max(mx.abs(first[0, 0] - second[0, 0])).item()), 1e-4)
        reference = expected.layers[0].self_attn
        batch = SimpleNamespace(past=[None] * 5, offset=0)
        with torch.no_grad():
            correct = reference(torch.arange(16), source, batch)
            reference.attn.attn_type = oracle.AttentionType.DECODER
            faulty = reference(torch.arange(16), source, batch)
        self.assertGreater((correct - faulty).abs().max().item(), 1e-3)

    def test_target_shares_raw_embedding_and_actual_head_without_target_scale(self):
        from dflash_target import DraftedTarget

        class Selector:
            def lattice(self, hidden, head, anchor):
                self.hidden, self.head, self.anchor = hidden, head, anchor
                return hidden, hidden

        class SpyDraft(nn.Module):
            def __init__(self):
                super().__init__()
                self.args = DraftArgs.from_dict(config())
                self.candidate_selector = Selector()

            def __call__(self, embeddings, cache):
                self.seen = embeddings
                return embeddings

        target = SimpleNamespace(args=SimpleNamespace(hidden_size=64, vocab_size=128,
                                 num_key_value_heads=2, head_dim=16, num_hidden_layers=80, embed_scale=8.0),
                                 model=SimpleNamespace(embed_tokens=nn.Embedding(128, 64), layers=[None] * 80),
                                 lm_head=nn.Linear(64, 128, bias=False), model_type='ricursive')
        draft = SpyDraft()
        wrapper = DraftedTarget(target, draft)
        wrapper.draft_lattice(7, [None] * 85, 16)
        tokens = mx.array([[7] + [127] * 15])
        expected = target.model.embed_tokens(tokens)
        np.testing.assert_array_equal(np.asarray(draft.seen), np.asarray(expected))
        self.assertIs(draft.candidate_selector.head, target.lm_head)
        self.assertEqual(draft.candidate_selector.anchor, 7)
        np.testing.assert_array_equal(np.asarray(draft.candidate_selector.hidden), np.asarray(expected[0, 1:]))
        self.assertFalse(np.array_equal(np.asarray(draft.seen), np.asarray(expected * target.args.embed_scale)))

    def test_strict_complete_mixed_quantized_reload_and_corruption_refusal(self):
        _, _, model = paired(bf16=True)
        settings = config()
        quant = dict(bits=4, group_size=64, mode='affine')
        for name, module in tree_flatten(model.leaf_modules(), is_leaf=lambda item: isinstance(item, nn.Module)):
            if isinstance(module, nn.Linear):
                quant[name] = dict(bits=8 if name == 'fc' or 'kernel_projection' in name
                                   or name == 'candidate_selector.hidden_projection' else 4,
                                   group_size=64, mode='affine')
        nn.quantize(model, group_size=64, bits=4, class_predicate=lambda name, module: quant.get(name, False))
        weights = dict(tree_flatten(model.parameters()))
        self.assertEqual(weights['layers.0.self_attn.q_proj.weight'].dtype, mx.uint32)
        self.assertEqual(model.layers[0].self_attn.q_proj.bits, 4)
        self.assertEqual(model.fc.bits, 8)
        self.assertEqual(model.layers[0].attention_conv.kernel_projection.bits, 8)
        self.assertEqual(weights['candidate_selector.predecessor_codebook'].dtype, mx.bfloat16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            file = path / 'model.safetensors'
            mx.save_safetensors(str(file), weights)
            settings['quantization'] = quant
            (path / 'config.json').write_text(json.dumps(settings))
            receipt = path / 'conversion-receipt.json'
            receipt.write_text(json.dumps(dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest())))
            loaded = load_draft(path)
            inputs = mx.ones((1, 16, 64), dtype=mx.bfloat16)
            np.testing.assert_array_equal(np.asarray(model(inputs, model.make_cache()).astype(mx.float32)),
                                          np.asarray(loaded(inputs, loaded.make_cache()).astype(mx.float32)))
            with self.assertRaises(ValueError):
                loaded.load_weights([(k, v) for k, v in weights.items() if k != 'norm.weight'], strict=True)
            with self.assertRaises(ValueError):
                loaded.load_weights(list(weights.items()) + [('invented.weight', mx.zeros((2,)))], strict=True)
            file.write_bytes(file.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_draft(path)
            (path / 'CONVERSION_INCOMPLETE').touch()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_draft(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle-only', action='store_true')
    options = parser.parse_args()
    SOURCE = options.source
    suite = unittest.TestSuite()
    for case in [OracleTests] + ([] if options.oracle_only else [DraftArchitectureTests]):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    options.output.write_text(json.dumps(dict(time_unix=time.time(), source_url=oracle.SOURCE_URL,
        source_commit=oracle.SOURCE_COMMIT, source_sha256=oracle.SOURCE_SHA256, source_license=oracle.SOURCE_LICENSE,
        activation_sources=oracle.ACTIVATION_SOURCES, activation_mode=oracle.ACTIVATION_MODE,
        source_method='upstream ASTs; distributed linears, CPU attention/RoPE/RMSNorm adapters; explicit config NoPE',
        scope='synthetic architecture; no whole SGLang runtime, checkpoint fidelity or target-token identity claim',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        passed=result.wasSuccessful(), mlx_available=HAS_MLX, oracle_only=options.oracle_only,
        architecture_comparison_passed=HAS_MLX and not options.oracle_only and result.wasSuccessful()
                                      and not result.skipped,
        torch_version=torch.__version__, mlx_device=str(mx.default_device()) if HAS_MLX else None,
        fp32_architecture_device='explicit MLX CPU stream; GPU strict FP32 remains separately unqualified',
        selector_fixture='random head diagnostics plus separated single-column head for BF16 actual geometry',
        selector_diagnostics=SELECTOR_DIAGNOSTICS,
        tolerances=dict(fp32_atol=2e-5, fp32_rtol=4e-4, bf16_normalized_l2_max=0.025,
                        bf16_atol=0.065, bf16_rtol=0.035), metrics=METRICS), indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
