"""Use the pinned Inkling cache layout to build a single-row SDPA mask."""
from importlib.metadata import PackageNotFoundError, version
from types import MethodType

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2.layers import Attention, FullAttention, SlidingAttention
from mlx_lm.models.inkling2.model import Model


_ORIGINAL_ATTENTION = Attention._attention


def _decode_mask(bias, size, cache, window):
    extent = bias.shape[-1]
    if type(cache) is KVCache and cache.offset == size and window == 0:
        # Dense keys are chronological. Distances beyond the learned extent use zero.
        mask = bias[..., :min(size, extent)][..., ::-1]
        if size > extent:
            zeros = mx.zeros((bias.shape[0], size - extent), bias.dtype)
            mask = mx.concatenate([zeros, mask], axis=-1)
    elif (
        type(cache) is RotatingKVCache and cache.keep == 0
        and size <= min(extent, window, cache.offset)
    ):
        # _idx points at the next write. Physical columns before it are the newest.
        recent = bias[..., :size]
        current = cache._idx % size
        if current == 0:
            mask = recent[..., ::-1]
        else:
            mask = mx.concatenate(
                [recent[..., :current][..., ::-1], recent[..., current:][..., ::-1]], axis=-1
            )
    else:
        return None
    # A broadcast BF16 select retains the pinned subnormal and NaN encoding behavior.
    return mx.where(mx.ones((1,), dtype=mx.bool_), mask, -mx.inf)


def _attention(self, query, key, value, position_bias, sinks, cache):
    heads = self.num_heads
    dims = self.head_dim
    if (
        mx.default_device() == mx.gpu and query.shape == (1, heads, 1, dims)
        and key.ndim == 4 and key.shape[:2] == (1, self.num_key_value_heads)
        and key.shape[2] > 0 and key.shape[3] == dims and value.shape == key.shape
        and query.dtype == key.dtype == value.dtype == mx.bfloat16
        and position_bias.ndim == 4 and position_bias.shape[:3] == (1, 1, heads)
        and position_bias.shape[-1] > 0 and position_bias.dtype == mx.bfloat16
        and sinks.shape == (1, 1, heads) and sinks.dtype == mx.bfloat16
    ):
        mask = _decode_mask(position_bias[0, 0], key.shape[2], cache, self.sliding_window)
        if mask is not None:
            mask = mx.contiguous(mask.astype(query.dtype))
            result = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scaling,
                mask=mask[None, :, None, :],
                sinks=sinks[0, 0].astype(query.dtype),
            )
            return result.transpose(0, 2, 1, 3)
    return _ORIGINAL_ATTENTION(self, query, key, value, position_bias, sinks, cache)


def install(model):
    """Patch the loaded model's own attention modules and return their count."""
    try:
        pinned = version('mlx') == '0.32.4' and version('mlx-lm') == '0.32.3.dev26+g0c4c99192'
    except PackageNotFoundError:
        return 0
    if not pinned or not mx.metal.is_available() or type(model) is not Model:
        return 0
    if mx.metal.device_info().get('device_name') != 'Apple M5 Max':
        return 0

    count = 0
    for layer in model.model.language_model.layers:
        attention = getattr(layer, 'self_attn', None)
        if (
            type(attention) in (FullAttention, SlidingAttention)
            and getattr(attention._attention, '__func__', None) is _ORIGINAL_ATTENTION
        ):
            # MLX stores a bound method as a Python attribute outside its parameter tree.
            attention._attention = MethodType(_attention, attention)
            count += 1
    return count
