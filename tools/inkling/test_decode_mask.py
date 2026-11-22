"""Check the worker's instance-local decode mask against pinned Inkling SDPA."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx_lm.models.inkling2.layers import (
        Attention, FullAttention, SlidingAttention, _key_positions, _positions,
    )
    from mlx_lm.models.inkling2.model import Model
except ImportError:
    HAS_INKLING = False
else:
    import decode_mask
    HAS_INKLING = True


def bare_module(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def attention_module(sliding):
    attention = bare_module(SlidingAttention if sliding else FullAttention)
    attention.head_dim = 128
    attention.num_heads = 24
    attention.num_key_value_heads = 8
    attention.sliding_window = 1024 if sliding else 0
    attention.scaling = attention.head_dim ** -0.5
    return attention


def model_with(*attentions):
    model = bare_module(Model)
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.layers = []
    for attention in attentions:
        layer = nn.Module()
        layer.self_attn = attention
        model.model.language_model.layers.append(layer)
    return model


@unittest.skipUnless(HAS_INKLING, 'Requires the pinned Inkling MLX environment')
class DecodeMaskTests(unittest.TestCase):
    def setUp(self):
        device = patch.object(mx.metal, 'device_info', return_value={'device_name': 'Apple M5 Max'})
        device.start()
        self.addCleanup(device.stop)

    def test_install_is_instance_local_and_parameter_neutral(self):
        full, sliding = attention_module(False), attention_module(True)
        full.sample_weight = mx.array([1.25])
        sliding.sample_weight = mx.array([-0.5])
        model = model_with(full, sliding)
        unrelated = attention_module(False)
        parameters_before = [(name, id(value)) for name, value in tree_flatten(model.parameters())]

        def pinned(name):
            return {'mlx': '0.32.4', 'mlx-lm': '0.32.3.dev26+g0c4c99192'}[name]

        with patch.object(decode_mask, 'version', pinned), patch.object(mx.metal, 'is_available', return_value=True):
            self.assertEqual(decode_mask.install(model), 2)
            self.assertEqual(decode_mask.install(model), 0)
            for attention in full, sliding:
                self.assertIs(attention._attention.__self__, attention)
                self.assertIs(attention._attention.__func__, decode_mask._attention)
                self.assertNotIn('_attention', attention)
        parameters_after = [(name, id(value)) for name, value in tree_flatten(model.parameters())]
        self.assertEqual(parameters_before, parameters_after)
        self.assertIs(unrelated._attention.__func__, Attention._attention)
        self.assertIs(Attention._attention, decode_mask._ORIGINAL_ATTENTION)

    def test_other_runtime_and_custom_method_stay_unpatched(self):
        full, custom = attention_module(False), attention_module(True)
        model = model_with(full, custom)
        custom_method = lambda *args: None
        custom._attention = custom_method
        with patch.object(decode_mask, 'version', return_value='0.32.5'):
            self.assertEqual(decode_mask.install(model), 0)
            self.assertIs(full._attention.__func__, Attention._attention)
            self.assertIs(custom._attention, custom_method)
        with patch.object(decode_mask, 'version', side_effect=('0.32.4', '0.32.3.dev26+g0c4c99192')):
            with patch.object(mx.metal, 'is_available', return_value=True):
                self.assertEqual(decode_mask.install(model), 1)
                self.assertIs(custom._attention, custom_method)

    def test_other_metal_chip_stays_unpatched(self):
        attention = attention_module(False)
        with patch.object(mx.metal, 'device_info', return_value={'device_name': 'Apple M4 Max'}):
            self.assertEqual(decode_mask.install(model_with(attention)), 0)
        self.assertIs(attention._attention.__func__, Attention._attention)

    def test_prefill_batch_other_dtype_and_other_cache_use_original(self):
        attn = attention_module(False)

        def fake_array(shape, dtype=mx.bfloat16):
            return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)

        bias = fake_array((1, 1, 24, 1024))
        # The valid-shape other-cache case only needs indexing to return a valid bias row.
        class Bias:
            shape = (1, 1, 24, 1024)
            ndim = 4
            dtype = mx.bfloat16
            def __getitem__(self, key):
                return fake_array((24, 1024))

        keys = fake_array((1, 8, 50, 128))
        sinks = fake_array((1, 1, 24))
        cases = [
            (fake_array((1, 24, 2, 128)), bias, None),
            (fake_array((2, 24, 1, 128)), bias, None),
            (fake_array((1, 24, 1, 128), mx.float32), bias, None),
            (fake_array((1, 24, 1, 128)), Bias(), object()),
        ]
        for query, position_bias, cache in cases:
            with self.subTest(shape=query.shape, cache=cache), patch.object(
                decode_mask, '_ORIGINAL_ATTENTION', return_value='fallback'
            ) as original:
                result = decode_mask._attention(attn, query, keys, keys, position_bias, sinks, cache)
                self.assertEqual(result, 'fallback')
                original.assert_called_once_with(attn, query, keys, keys, position_bias, sinks, cache)


@unittest.skipUnless(HAS_INKLING and mx.metal.is_available(), 'Requires pinned MLX with Metal')
class DecodeMaskMetalTests(unittest.TestCase):
    @staticmethod
    def sample(shape, seed):
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    def assert_same(self, attn, query, key, value, bias, sinks, cache):
        expected = Attention._attention(attn, query, key, value, bias, sinks, cache)
        actual = decode_mask._attention(attn, query, key, value, bias, sinks, cache)
        self.assertEqual(actual.dtype, expected.dtype, 'attention output dtype')
        self.assertTrue(mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item())

    def test_mask_matches_original_for_all_bf16_words(self):
        heads, extent = 24, 1024
        cases = [(size, size, 0, 0) for size in (31, 1023, 1024, 1025, 4097)]
        cases.extend((size, end, index, extent) for size, end, index in (
            (31, 31, 31), (1023, 1023, 1023), (1024, 1024, 1024),
            (1024, 1025, 1), (1024, 1536, 512), (1024, 2047, 1023),
        ))
        with mx.stream(mx.gpu):
            row = mx.arange(heads, dtype=mx.uint32)[:, None]
            col = mx.arange(extent, dtype=mx.uint32)[None, :]
            for size, offset, index, window in cases:
                cache = KVCache() if not window else RotatingKVCache(max_size=window)
                cache.offset = offset
                if window:
                    cache._idx = index
                query_pos = _positions(offset - 1, 1)
                keys = _key_positions(cache, size, 1, offset).reshape(size)
                distance = query_pos.reshape(1) - keys
                visible = (distance >= 0) & (keys >= 0)
                if window:
                    visible = visible & (distance < window)
                clipped = mx.clip(distance, 0, extent - 1)

                # Each tile places consecutive words in the columns the mask
                # will select. The loop includes all subnormal and NaN words.
                columns = min(size, extent)
                for start in range(0, 1 << 16, heads * columns):
                    with self.subTest(size=size, offset=offset, index=index, start=start):
                        words = (start + row * columns + col).astype(mx.uint16)
                        bias = words.view(mx.bfloat16)
                        gathered = mx.take(bias, clipped, axis=-1)
                        relative = mx.where(distance < extent, gathered, mx.zeros_like(gathered))
                        expected = mx.where(visible, relative, -mx.inf).astype(mx.bfloat16)
                        actual = decode_mask._decode_mask(bias, size, cache, window)
                        self.assertIsNotNone(actual)
                        actual = mx.contiguous(actual.astype(mx.bfloat16))
                        self.assertTrue(mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item())

    def test_bitwise_sdpa_at_relative_extent_and_ring_boundaries(self):
        full, sliding = attention_module(False), attention_module(True)
        with mx.stream(mx.gpu):
            query = self.sample((1, 24, 1, 128), 1234)
            bias = self.sample((1, 1, 24, 1024), 1235)
            sinks = self.sample((1, 1, 24), 1236)
            for size in (1, 1023, 1024, 1025, 4097, 131072):
                with self.subTest(layout='dense', length=size):
                    cache = KVCache()
                    cache.offset = size
                    key = self.sample((1, 8, size, 128), 1237)
                    value = self.sample((1, 8, size, 128), 1238)
                    self.assert_same(full, query, key, value, bias, sinks, cache)
            for size, offset, index in [(15, 15, 15), (1023, 1023, 1023),
                                         (1024, 1024, 1024), (1024, 1025, 1),
                                         (1024, 1536, 512), (1024, 2047, 1023)]:
                with self.subTest(layout='rotating', length=size, index=index):
                    cache = RotatingKVCache(max_size=1024)
                    cache.offset = offset
                    cache._idx = index
                    key = self.sample((1, 8, size, 128), 1237)
                    value = self.sample((1, 8, size, 128), 1238)
                    self.assert_same(sliding, query, key, value, bias, sinks, cache)

    def test_bitwise_sdpa_after_prefill_spill_and_real_ring_write(self):
        sliding = attention_module(True)
        cache = RotatingKVCache(max_size=1024)
        with mx.stream(mx.gpu):
            cache.update_and_fetch(self.sample((1, 8, 1121, 128), 21),
                                   self.sample((1, 8, 1121, 128), 22))
            key, value = cache.update_and_fetch(self.sample((1, 8, 1, 128), 23),
                                                self.sample((1, 8, 1, 128), 24))
            self.assertEqual((key.shape[2], cache._idx), (1024, 1))
            self.assert_same(sliding, self.sample((1, 24, 1, 128), 25), key, value,
                             self.sample((1, 1, 24, 1024), 26), self.sample((1, 1, 24), 27), cache)


if __name__ == '__main__':
    unittest.main()
