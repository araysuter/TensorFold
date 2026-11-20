import unittest
from types import SimpleNamespace

try:
    import mlx.core as mx
    from mlx_lm.models.inkling2 import layers
    from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
    import attention
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires the pinned Inkling MLX environment')
class DecodeRowsTests(unittest.TestCase):
    def test_global_and_rotated_local_rows_equal_serial_primitive(self):
        original = layers.Attention._attention
        attention.install()
        for window in (0, 64):
            with self.subTest(window=window):
                mx.random.seed(1234)
                q = mx.random.normal((2, 6, 1, 16)).astype(mx.bfloat16)
                k = mx.random.normal((2, 2, 64, 16)).astype(mx.bfloat16)
                v = mx.random.normal((2, 2, 64, 16)).astype(mx.bfloat16)
                bias = mx.random.normal((2, 1, 6, 32)).astype(mx.bfloat16)
                sinks = mx.random.normal((2, 1, 6)).astype(mx.bfloat16)
                owner = SimpleNamespace(sliding_window=window, scaling=0.25)
                if window:
                    cache = BatchRotatingKVCache(64, [0, 7])
                    cache.offset, cache._idx = mx.array([100, 93]), 17
                else:
                    cache = SimpleNamespace(offset=mx.array([64, 57]))
                expected = []
                for row in range(2):
                    if window:
                        single = RotatingKVCache(max_size=64, keep=0)
                        single.offset, single._idx = (100, 93)[row], 17
                    else:
                        single = SimpleNamespace(offset=(64, 57)[row])
                    expected.append(original(owner, q[row:row + 1], k[row:row + 1], v[row:row + 1],
                                             bias[row:row + 1], sinks[row:row + 1], single))
                actual = layers.Attention._attention(owner, q, k, v, bias, sinks, cache)
                self.assertTrue(mx.array_equal(actual, mx.concatenate(expected)).item())
                changed = layers.Attention._attention(owner, q, k, v + 1, bias, sinks, cache)
                self.assertFalse(mx.array_equal(actual, changed).item())


if __name__ == '__main__':
    unittest.main()
