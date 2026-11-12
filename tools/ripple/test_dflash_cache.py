#!/usr/bin/env python3
"""Exercise real MLX DFlash cache rollback, ownership and disk restoration across rotating boundaries."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
try:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import KVCache, RotatingKVCache, save_prompt_cache
    from dflash_model import DraftArgs, DraftModel, ordered_kv
    from dflash_target import DraftedTarget, trim_verification
    from modeling_ricursive_mlx import Model, ModelArgs
    from worker_state import ResidentState
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: DFlash cache tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


def rows(start, count, heads=2, width=4):
    value = mx.arange(start, start + count, dtype=mx.float32)[None, None, :, None]
    return mx.broadcast_to(value, (1, heads, count, width)).astype(mx.bfloat16)


def update(entry, start, count):
    keys = rows(start, count)
    entry.update_and_fetch(keys, -keys)
    mx.eval(entry.state)


def suffix(entry, count):
    keys, values = ordered_kv(entry)
    return tuple(np.array(x[..., -count:, :].astype(mx.float32)) for x in (keys, values))


def small_model():
    target_args = ModelArgs(hidden_size=32, intermediate_size=64, num_hidden_layers=5,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            vocab_size=64, sliding_window=8, embed_scale=4.0)
    draft_args = DraftArgs(hidden_size=32, intermediate_size=64, num_hidden_layers=5,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                          vocab_size=64, sliding_window=16,
                          layer_types=['sliding_attention'] * 4 + ['full_attention'], no_rope_layers=[4],
                          target_layer_ids=[0, 1, 2, 3, 4], block_size=16, conv_kernel_size=4,
                          conv_group_size=16, selector_rank=8, selector_top_k=4, mask_token_id=63)
    mx.random.seed(149)
    model = DraftedTarget(Model(target_args), DraftModel(draft_args))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


class RollbackTests(unittest.TestCase):
    def assert_suffix(self, entry, end, window):
        count = min(end, window)
        actual_k, actual_v = suffix(entry, count)
        expected = np.array(rows(end - count, count).astype(mx.float32))
        np.testing.assert_array_equal(actual_k, expected)
        np.testing.assert_array_equal(actual_v, -expected)
        self.assertEqual(entry.offset, end)

    def test_every_partial_acceptance_before_at_and_after_wrap(self):
        for prefix in [0, 1, 6, 7, 8, 9, 15, 16, 23, 129]:
            for accepted in [1, 2, 7, 15, 16]:
                with self.subTest(prefix=prefix, accepted=accepted):
                    cache = [RotatingKVCache(8, keep=0), RotatingKVCache(16, keep=0), KVCache()]
                    for entry in cache:
                        for token in range(prefix):
                            update(entry, token, 1)
                        update(entry, prefix, 16)
                    trim_verification(cache, prefix + accepted)
                    for entry in cache:
                        window = entry.max_size if isinstance(entry, RotatingKVCache) else entry.offset
                        self.assert_suffix(entry, prefix + accepted, window)
                        update(entry, prefix + accepted, 1)
                        window = entry.max_size if isinstance(entry, RotatingKVCache) else entry.offset
                        self.assert_suffix(entry, prefix + accepted + 1, window)
                        update(entry, prefix + accepted + 1, 5)
                        window = entry.max_size if isinstance(entry, RotatingKVCache) else entry.offset
                        self.assert_suffix(entry, prefix + accepted + 6, window)

    def test_large_prefill_then_partial_round_retains_exact_past(self):
        for prefix in [257, 512, 1025]:
            for accepted in [1, 8, 15]:
                cache = [RotatingKVCache(8, keep=0), KVCache()]
                for entry in cache:
                    update(entry, 0, prefix)
                    update(entry, prefix, 16)
                trim_verification(cache, prefix + accepted)
                for entry in cache:
                    window = entry.max_size if isinstance(entry, RotatingKVCache) else entry.offset
                    self.assert_suffix(entry, prefix + accepted, window)
                self.assertEqual(cache[0].keys.shape[2], 7 + accepted)

    def test_rotated_single_step_rewind_is_refused(self):
        entry = RotatingKVCache(8, keep=0)
        for token in range(10):
            update(entry, token, 1)
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            trim_verification([entry], 9)

    def test_rewind_cannot_extend_or_remove_entire_cold_verification(self):
        entry = RotatingKVCache(8, keep=0)
        update(entry, 0, 16)
        with self.assertRaisesRegex(ValueError, 'extend'):
            trim_verification([entry], 17)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            trim_verification([entry], 0)

    def test_full_cache_rewind_reuses_existing_storage(self):
        entry = KVCache()
        update(entry, 0, 49)
        keys, values = entry.keys, entry.values
        trim_verification([entry], 34)
        self.assertIs(entry.keys, keys)
        self.assertIs(entry.values, values)
        update(entry, 34, 1)
        self.assert_suffix(entry, 35, 35)

    def test_model_continuation_after_partial_verification_matches_clean_prefix(self):
        model = small_model()
        for prefix in [1, 7, 8, 15, 16, 31, 257]:
            for accepted in [1, 4, 15]:
                with self.subTest(prefix=prefix, accepted=accepted):
                    prompt = [3 + i % 43 for i in range(prefix)]
                    block = [7 + i % 31 for i in range(16)]
                    verified = model.make_cache()
                    clean = model.make_cache()
                    mx.eval(model(mx.array([prompt]), cache=verified))
                    mx.eval(model(mx.array([prompt]), cache=clean))
                    mx.eval(model(mx.array([block]), cache=verified))
                    trim_verification(verified, prefix + accepted)
                    mx.eval(model(mx.array([block[:accepted]]), cache=clean))
                    continuation = mx.array([[41, 42, 43]])
                    actual = np.array(model(continuation, cache=verified).astype(mx.float32))
                    expected = np.array(model(continuation, cache=clean).astype(mx.float32))
                    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
                    np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.03)
                    for entry in verified:
                        self.assertEqual(entry.offset, prefix + accepted + 3)
                    for entry, reference in zip(verified[5:], clean[5:]):
                        count = min(entry.offset, 16) if isinstance(entry, RotatingKVCache) else entry.offset
                        for actual, expected in zip(suffix(entry, count), suffix(reference, count)):
                            np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.03)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        kinds = ['sliding_attention'] * 4 + ['full_attention']
        self.args = SimpleNamespace(layer_types=kinds * 16 + kinds, cache_windows=[8] * 80 + [16] * 5,
                                    sliding_window=8, num_key_value_heads=2, head_dim=4, vocab_size=2048)
        self.state = ResidentState(self.root, 'dflash-test-build', self.args, 2048)
        self.path = self.root / ('a' * 64 + '.tfs')

    def tearDown(self):
        self.state.clear()
        self.directory.cleanup()

    def fill(self, count=41):
        cache = [RotatingKVCache(window, keep=0) if kind == 'sliding_attention' else KVCache()
                 for kind, window in zip(self.args.layer_types, self.args.cache_windows)]
        for entry in cache:
            update(entry, 0, count)
        tokens = list(range(count))
        self.state.commit(tokens, cache)
        return tokens, cache

    def corrupt(self, change, metadata_change=None):
        tokens, cache = self.fill()
        change(cache)
        temporary = self.path.with_suffix('.safetensors')
        save_prompt_cache(str(temporary), cache, metadata={
            'format': 'tensorfold-ripple-kv-v1', 'identity': 'dflash-test-build',
            'tokens': json.dumps(tokens, separators=(',', ':'))})
        if metadata_change is not None:
            arrays, metadata = mx.load(str(temporary), return_metadata=True)
            metadata_change(metadata)
            altered = self.path.with_suffix('.metadata.safetensors')
            mx.save_safetensors(str(altered), arrays, metadata)
            altered.replace(temporary)
        temporary.rename(self.path)
        self.path.chmod(0o600)
        self.state.clear()
        with self.assertRaises(ValueError):
            self.state.load(self.path)
        self.assertIsNone(self.state.cache)
        self.assertEqual(self.state.tokens, [])

    def test_85_cache_snapshot_restores_offsets_windows_and_next_update(self):
        tokens, cache = self.fill()
        update_block = 16
        for entry in cache:
            update(entry, len(tokens), update_block)
        trim_verification(cache, len(tokens) + 3)
        self.state.commit(tokens + list(range(41, 57)), cache)
        receipt = self.state.save(self.path)
        before = [suffix(entry, min(entry.offset, window) if isinstance(entry, RotatingKVCache) else entry.offset)
                  for entry, window in zip(cache, self.args.cache_windows)]
        self.state.clear()
        self.assertEqual(self.state.load(self.path), receipt)
        self.assertEqual(len(self.state.cache), 85)
        for index, entry in enumerate(self.state.cache):
            window = self.args.cache_windows[index] if isinstance(entry, RotatingKVCache) else entry.offset
            count = min(entry.offset, window)
            for actual, expected in zip(suffix(entry, count), before[index]):
                np.testing.assert_array_equal(actual, expected)
            update(entry, 44, 1)
            self.assertEqual(entry.offset, 45)

    def test_snapshot_before_decode_after_large_prefill(self):
        self.fill(513)
        receipt = self.state.save(self.path)
        self.state.clear()
        self.assertEqual(self.state.load(self.path), receipt)
        for entry in self.state.cache:
            self.assertEqual(entry.offset, 513)
            update(entry, 513, 1)
            self.assertEqual(entry.offset, 514)

    def test_snapshot_at_exact_context_limit_can_restore(self):
        self.fill(self.state.context)
        receipt = self.state.save(self.path)
        self.state.clear()
        self.assertEqual(self.state.load(self.path), receipt)
        self.assertEqual(len(self.state.tokens), self.state.context)

    def test_target_only_snapshot_rejected_for_drafted_identity(self):
        self.corrupt(lambda cache: cache.__delitem__(slice(80, None)))

    def test_wrong_draft_window_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'max_size', 8))

    def test_misaligned_draft_offset_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'offset', 40))

    def test_fractional_saved_offset_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'offset', 41.5))

    def test_integral_float_full_offset_rejected(self):
        self.corrupt(lambda cache: setattr(cache[84], 'offset', 41.0))

    def test_floating_rotating_index_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], '_idx', 41.0))

    def test_floating_rotating_window_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'max_size', 16.0))

    def test_floating_rotating_keep_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'keep', 0.0))

    def test_boolean_rotating_keep_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'keep', False))

    def test_wrong_saved_kv_rank_rejected(self):
        def flatten(cache):
            entry = cache[80]
            entry.keys, entry.values = entry.keys.reshape(1, 2, -1), entry.values.reshape(1, 2, -1)
        self.corrupt(flatten)

    def test_full_class_metadata_with_missing_tensor_entry_rejected(self):
        def metadata_change(metadata):
            metadata['1.84'] = 'KVCache'
        self.corrupt(lambda cache: cache.pop(), metadata_change)

    def test_short_full_cache_rejected(self):
        self.corrupt(lambda cache: setattr(cache[84], 'keys', cache[84].keys[..., :20, :]))

    def test_mismatched_kv_extents_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], 'values', cache[80].values[..., :20, :]))

    def test_rotating_index_outside_allocated_storage_rejected(self):
        self.corrupt(lambda cache: setattr(cache[80], '_idx', cache[80].keys.shape[2] + 1))

    def test_insufficient_rotating_history_rejected(self):
        def shorten(cache):
            entry = cache[80]
            entry.keys, entry.values = entry.keys[..., :4, :], entry.values[..., :4, :]
            entry._idx = 4
        self.corrupt(shorten)

    def test_taken_prefix_has_single_owner_and_failed_request_leaves_no_resident_cache(self):
        tokens, original = self.fill()
        taken, count = self.state.take(tokens + [49])
        self.assertIs(taken, original)
        self.assertEqual(count, len(tokens))
        self.assertEqual(self.state.resumable(tokens + [49]), 0)
        self.assertIsNone(self.state.cache)
        self.state.clear()
        self.assertEqual(self.state.resumable(tokens + [49]), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
