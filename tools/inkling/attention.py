"""Use the single-row attention primitive for each row of batched decode."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers


def install():
    original = layers.Attention._attention
    if getattr(original, '_inkling_decode_rows', False):
        return

    def attention(self, query, key, value, position_bias, sinks, cache):
        batch, _, length, _ = query.shape
        if length != 1 or batch == 1:
            return original(self, query, key, value, position_bias, sinks, cache)
        size = key.shape[2]
        end = cache.offset if cache is not None else length
        queries = layers._positions(end - length, length)
        keys = layers._key_positions(cache, size, length, end)
        rows = []
        for index in range(batch):
            qpos = queries[index, 0] if queries.ndim == 2 else queries[0]
            kpos = keys[index] if keys.ndim == 2 else keys
            distance = qpos - kpos
            radius = position_bias.shape[-1]
            bias = mx.take(position_bias[index, 0], mx.clip(distance, 0, radius - 1), axis=-1)
            bias = mx.where(distance < radius, bias, mx.zeros_like(bias))
            visible = (distance >= 0) & (kpos >= 0)
            if self.sliding_window:
                visible = visible & (distance < self.sliding_window)
            mask = mx.where(visible, bias, -mx.inf).astype(query.dtype)
            rows.append(mx.fast.scaled_dot_product_attention(
                query[index:index + 1], key[index:index + 1], value[index:index + 1],
                scale=self.scaling, mask=mask[None, :, None, :],
                sinks=sinks[index, 0].astype(query.dtype)))
        return mx.concatenate(rows, axis=0).transpose(0, 2, 1, 3)

    attention._inkling_decode_rows = True
    layers.Attention._attention = attention
