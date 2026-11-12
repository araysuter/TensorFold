#!/usr/bin/env python3
"""Compare exact verification attention and committed physical KV to original single-token execution."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from dflash_exact import VerificationAttention, VerificationMode, install_exact_attention, trim_exact
    from dflash_target import DraftedTarget
    from modeling_ricursive_mlx import Attention, ModelArgs, RMSNorm
    from worker_state import ResidentState
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: Exact attention tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


class Slice(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start, self.end = start, end

    def __call__(self, values):
        return values[..., self.start:self.end]


class Identity(nn.Module):
    def __call__(self, values):
        return values


def array(value):
    return np.array(value.astype(mx.float32))


def clone(entry):
    state = list(entry.state)
    for index in [0, 1]:
        if state[index] is not None:
            state[index] = state[index].astype(mx.float32).astype(mx.bfloat16)
    copied = type(entry).from_state(state)
    if hasattr(entry, '_exact_start'):
        copied._exact_start = entry._exact_start
    mx.eval(copied.state)
    return copied


def attention(local=True, heads=20, kvheads=2, gated=False, valres=False, window=8):
    qwidth, kvwidth = heads * 128, kvheads * 128
    config = ModelArgs(hidden_size=qwidth + 2 * kvwidth, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=heads, num_key_value_heads=kvheads, head_dim=128, vocab_size=64,
                       sliding_window=window, layer_types=['sliding_attention' if local else 'full_attention'],
                       use_gated_attention=gated, use_valres=valres)
    inner = Attention.__new__(Attention)
    nn.Module.__init__(inner)
    inner.args, inner.sliding = config, local
    inner.q_proj, inner.k_proj = Slice(0, qwidth), Slice(qwidth, qwidth + kvwidth)
    inner.v_proj, inner.o_proj = Slice(qwidth + kvwidth, qwidth + 2 * kvwidth), Identity()
    inner.q_norm, inner.k_norm = RMSNorm(128, 1e-6), RMSNorm(128, 1e-6)
    if gated:
        inner.g_proj = Slice(0, qwidth)
    if valres:
        inner.value_res_lambda = mx.array([0.37])
    inner.set_dtype(mx.bfloat16)
    return inner


def inputs(inner, count, start=0):
    values = np.random.default_rng(392 + start).normal(size=(1, count, inner.args.hidden_size)).astype(np.float32)
    return mx.array(values).astype(mx.bfloat16)


def prepared(inner, offset):
    entry = RotatingKVCache(inner.args.sliding_window, keep=0) if inner.sliding else KVCache()
    if not offset:
        return entry
    count = min(offset, inner.args.sliding_window) if inner.sliding else offset
    values = inputs(inner, count, offset)
    keys = inner.k_norm(inner.k_proj(values).reshape(1, count, inner.args.num_key_value_heads, 128))
    keys = keys.transpose(0, 2, 1, 3)
    vals = inner.v_proj(values).reshape(1, count, inner.args.num_key_value_heads, 128).transpose(0, 2, 1, 3)
    if inner.sliding:
        keys = inner._rope(keys, offset - count)
        pointer = offset if offset <= count else (offset - 1) % count + 1
        if offset > count and pointer != count:
            keys = mx.concatenate([keys[..., -pointer:, :], keys[..., :-pointer, :]], axis=2)
            vals = mx.concatenate([vals[..., -pointer:, :], vals[..., :-pointer, :]], axis=2)
        entry._idx = pointer
    entry.keys, entry.values, entry.offset = keys, vals, offset
    mx.eval(entry.state)
    return entry


def call(inner, values, cache, seed=None):
    mask = cache.make_mask(values.shape[1], window_size=inner.args.sliding_window if inner.sliding else None,
                           return_array=False)
    return inner(values, seed, mask, cache)


class ExactAttentionTests(unittest.TestCase):
    def assert_cache(self, actual, reference):
        self.assertEqual(actual.offset, reference.offset)
        if isinstance(actual, RotatingKVCache):
            self.assertEqual(actual._idx, reference._idx)
            count = min(actual.offset, actual.max_size)
        else:
            count = actual.offset
        for a, b in zip(actual.keys_and_values(), reference.keys_and_values()):
            np.testing.assert_array_equal(array(a[..., :count, :]), array(b[..., :count, :]))

    def serial(self, inner, values, entry, seed=None):
        outputs, committed = [], []
        for index in range(values.shape[1]):
            selected_seed = None if seed is None else seed[:, index:index + 1]
            outputs.append(call(inner, values[:, index:index + 1], entry, selected_seed))
            mx.eval(outputs[-1], entry.state)
            committed.append(clone(entry))
        return mx.concatenate(outputs, axis=1), committed

    def check(self, inner, offset, width, with_seed=False, initial=None):
        values = inputs(inner, width, offset + 900)
        seed = inputs(inner, width, offset + 1200) if with_seed else None
        initial = prepared(inner, offset) if initial is None else initial
        expected, references = self.serial(inner, values, clone(initial), seed)
        mode = VerificationMode(enabled=True)
        wrapped = VerificationAttention(inner, mode)
        provisional = clone(initial)
        actual = wrapped(values, seed, None, provisional)
        np.testing.assert_array_equal(array(actual), array(expected))
        if inner.sliding:
            self.assertEqual(provisional._idx, provisional.keys.shape[2])
            self.assertEqual(provisional._exact_start, (offset, initial._idx))
        for accepted in range(1, width + 1):
            with self.subTest(accepted=accepted):
                entry = clone(provisional)
                if inner.sliding:
                    self.assertTrue(trim_exact(entry, offset + accepted))
                    self.assertFalse(hasattr(entry, '_exact_start'))
                    self.assertFalse(trim_exact(entry, offset + accepted))
                else:
                    self.assertFalse(trim_exact(entry, offset + accepted))
                    keys, vals = entry.keys, entry.values
                    entry.trim(width - accepted)
                    self.assertIs(entry.keys, keys)
                    self.assertIs(entry.values, vals)
                reference = clone(references[accepted - 1])
                self.assert_cache(entry, reference)
                followup = inputs(inner, 1, offset + accepted + 1700)
                continued = wrapped(followup, None, None, entry)
                oracle = call(inner, followup, reference)
                np.testing.assert_array_equal(array(continued), array(oracle))
                self.assert_cache(entry, reference)
        return provisional, references

    def test_all_partial_commits_before_at_and_after_wrap(self):
        inner = attention()
        for offset in [0, 1, 7, 8, 9, 15, 16, 131073]:
            for width in [4, 16]:
                with self.subTest(offset=offset, width=width):
                    self.check(inner, offset, width)

    def test_real_gqa_and_window_at_prefill_boundaries(self):
        inner = attention(heads=80, kvheads=8, window=1024)
        for offset in [127, 1023, 1024, 1025, 131072]:
            with self.subTest(offset=offset):
                self.check(inner, offset, 4)

    def test_bulk_prefill_single_step_pointer_is_not_absolute_offset_modulo_window(self):
        inner = attention()
        entry = RotatingKVCache(8, keep=0)
        mx.eval(call(inner, inputs(inner, 37), entry))
        self.assertGreater(entry.keys.shape[2], entry.max_size)
        wrapped = VerificationAttention(inner, VerificationMode(True))
        with self.assertRaisesRegex(ValueError, 'Normalize'):
            wrapped(inputs(inner, 4), None, None, entry)
        self.assertEqual(entry.offset, 37)
        mx.eval(call(inner, inputs(inner, 1, 37), entry))
        self.assertEqual(entry.offset, 38)
        self.assertEqual(entry._idx, 1)
        self.assertNotEqual(entry._idx, (entry.offset - 1) % entry.max_size + 1)
        self.check(inner, 38, 16, initial=entry)

    def test_full_nope_attention_uses_exact_serial_prefixes(self):
        inner = attention(local=False)
        for offset in [0, 127, 1023, 1024]:
            with self.subTest(offset=offset):
                self.check(inner, offset, 16)

    def test_original_gate_and_value_residual_formulas_are_retained(self):
        for local in [False, True]:
            with self.subTest(local=local):
                self.check(attention(local=local, gated=True, valres=True), 9, 4, with_seed=True)

    def test_disabled_and_single_token_paths_delegate_without_markers(self):
        inner = attention()
        mode = VerificationMode()
        wrapped = VerificationAttention(inner, mode)
        initial = prepared(inner, 9)
        for enabled, width in [(False, 4), (True, 1)]:
            mode.enabled = enabled
            actual_cache, expected_cache = clone(initial), clone(initial)
            values = inputs(inner, width)
            mask = actual_cache.make_mask(width, window_size=8)
            actual = wrapped(values, None, mask, actual_cache)
            expected = call(inner, values, expected_cache)
            np.testing.assert_array_equal(array(actual), array(expected))
            self.assert_cache(actual_cache, expected_cache)
            self.assertFalse(hasattr(actual_cache, '_exact_start'))

    def test_commit_rejects_unanchored_extended_or_noncontiguous_state(self):
        inner = attention()
        wrapped = VerificationAttention(inner, VerificationMode(True))
        provisional = prepared(inner, 9)
        wrapped(inputs(inner, 4), None, None, provisional)
        for desired in [9, 14]:
            with self.assertRaises(ValueError):
                trim_exact(clone(provisional), desired)
        broken = clone(provisional)
        broken._idx -= 1
        with self.assertRaises(ValueError):
            trim_exact(broken, 10)
        with self.assertRaisesRegex(ValueError, 'preceding'):
            wrapped(inputs(inner, 4), None, None, provisional)

    def test_physical_cache_comparison_detects_a_rotation_fault(self):
        inner = attention()
        entry = prepared(inner, 13)
        reference = clone(entry)
        entry.keys = mx.roll(entry.keys, 1, axis=2)
        entry.values = mx.roll(entry.values, 1, axis=2)
        with self.assertRaises(AssertionError):
            self.assert_cache(entry, reference)

    def test_committed_cache_snapshot_is_standard_and_continues_exactly(self):
        inner = attention()
        provisional, references = self.check(inner, 31, 4)
        trim_exact(provisional, 33)
        with tempfile.TemporaryDirectory() as directory:
            state = ResidentState(Path(directory), 'exact-test', inner.args, 512)
            path = Path(directory) / ('a' * 64 + '.tfs')
            state.commit([i % 64 for i in range(33)], [provisional])
            receipt = state.save(path)
            state.clear()
            self.assertEqual(state.load(path), receipt)
            self.assertFalse(hasattr(state.cache[0], '_exact_start'))
            reference = clone(references[1])
            self.assert_cache(state.cache[0], reference)
            values = inputs(inner, 1, 33)
            actual = array(call(inner, values, state.cache[0]))
            expected = array(call(inner, values, reference))
            np.testing.assert_array_equal(actual, expected)
            state.clear()

    def test_verification_context_restores_mode_and_environment_after_failure(self):
        holder = SimpleNamespace(exact_mode=VerificationMode())
        for previous in [None, 'previous-setting']:
            with self.subTest(previous=previous), patch.dict(os.environ):
                if previous is None:
                    os.environ.pop('MLX_RIPPLE_EXACT_VERIFY', None)
                else:
                    os.environ['MLX_RIPPLE_EXACT_VERIFY'] = previous
                with self.assertRaisesRegex(RuntimeError, 'Injected'):
                    with DraftedTarget.verification(holder):
                        self.assertTrue(holder.exact_mode.enabled)
                        self.assertEqual(os.environ['MLX_RIPPLE_EXACT_VERIFY'], '1')
                        raise RuntimeError('Injected evaluation failure')
                self.assertFalse(holder.exact_mode.enabled)
                self.assertEqual(os.environ.get('MLX_RIPPLE_EXACT_VERIFY'), previous)

    def test_nested_verification_refusal_preserves_outer_context(self):
        holder = SimpleNamespace(exact_mode=VerificationMode())
        with patch.dict(os.environ):
            os.environ.pop('MLX_RIPPLE_EXACT_VERIFY', None)
            with DraftedTarget.verification(holder):
                with self.assertRaisesRegex(ValueError, 'Nested'):
                    with DraftedTarget.verification(holder):
                        self.fail('Nested verification entered')
                self.assertTrue(holder.exact_mode.enabled)
                self.assertEqual(os.environ['MLX_RIPPLE_EXACT_VERIFY'], '1')
            self.assertFalse(holder.exact_mode.enabled)
            self.assertNotIn('MLX_RIPPLE_EXACT_VERIFY', os.environ)

    def test_installation_is_shared_and_idempotent_after_tap_wrapping(self):
        first, second = attention(), attention(local=False)
        layer = SimpleNamespace(self_attn=first)
        tapped = SimpleNamespace(inner=SimpleNamespace(self_attn=second))
        target = SimpleNamespace(model=SimpleNamespace(layers=[layer, tapped]))
        mode = install_exact_attention(target)
        self.assertIs(install_exact_attention(target), mode)
        self.assertIs(layer.self_attn.inner, first)
        self.assertIs(tapped.inner.self_attn.inner, second)
        self.assertIs(layer.self_attn.mode, tapped.inner.self_attn.mode)
        self.assertFalse(mode.enabled)


if __name__ == '__main__':
    unittest.main(verbosity=2)
