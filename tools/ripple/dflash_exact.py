"""Preserve serial target attention arithmetic while retaining bounded speculative KV for rollback."""
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class VerificationMode:
    enabled: bool = False


def physical_ring(value, pointer):
    if pointer == value.shape[2]:
        return value
    return mx.concatenate([value[..., -pointer:, :], value[..., :-pointer, :]], axis=2)


def virtual_pointer(offset, pointer, accepted, window):
    return offset + accepted if offset + accepted <= window else (pointer + accepted - 1) % window + 1


class VerificationAttention(nn.Module):
    def __init__(self, inner, mode):
        super().__init__()
        self.inner, self.mode = inner, mode

    def __call__(self, x, seed, mask, cache):
        inner, args = self.inner, self.inner.args
        batch, length, _ = x.shape
        if not self.mode.enabled or length == 1:
            return inner(x, seed, mask, cache)
        if batch != 1 or not 2 <= length <= 16 or cache is None:
            raise ValueError('Unqualified exact verification geometry')
        if hasattr(cache, '_exact_start'):
            raise ValueError('Commit the preceding verification before starting another')
        local = inner.sliding
        if local and (not isinstance(cache, RotatingKVCache) or cache.keep or cache.max_size != args.sliding_window):
            raise ValueError('Exact local verification requires the original rotating cache')
        if not local and type(cache) is not KVCache:
            raise ValueError('Exact full verification requires the original dense KV cache')
        offset, pointer = cache.offset, getattr(cache, '_idx', 0)
        if local and cache.keys is not None and cache.keys.shape[2] > cache.max_size:
            raise ValueError('Normalize the prefill cache with its original single-token step first')
        value_input = x
        if seed is not None:
            blend = mx.sigmoid(inner.value_res_lambda[0]).astype(x.dtype)
            source, current = seed.astype(mx.float32), x.astype(mx.float32)
            value_input = (source + blend.astype(mx.float32) * (current - source)).astype(x.dtype)
        query = inner.q_proj(x).reshape(batch, length, args.num_attention_heads, args.head_dim)
        keys = inner.k_proj(x).reshape(batch, length, args.num_key_value_heads, args.head_dim)
        values = inner.v_proj(value_input).reshape(batch, length, args.num_key_value_heads,
                                                 args.head_dim).transpose(0, 2, 1, 3)
        queries, normalized_keys = [], []
        for index in range(length):
            q = inner.q_norm(query[:, index:index + 1]).transpose(0, 2, 1, 3)
            k = inner.k_norm(keys[:, index:index + 1]).transpose(0, 2, 1, 3)
            if local:
                q, k = inner._rope(q, offset + index), inner._rope(k, offset + index)
            queries.append(q)
            normalized_keys.append(k)
        keys, values = cache.update_and_fetch(mx.concatenate(normalized_keys, axis=2), values)
        if local:
            cache._exact_start = (offset, pointer)
        past = keys.shape[2] - length
        outputs = []
        for index, query in enumerate(queries):
            end = past + index + 1
            start = max(0, end - args.sliding_window) if local else 0
            k, v = keys[..., start:end, :], values[..., start:end, :]
            if local and offset + index + 1 > cache.max_size:
                position = virtual_pointer(offset, pointer, index + 1, cache.max_size)
                k, v = physical_ring(k, position), physical_ring(v, position)
            output = mx.fast.scaled_dot_product_attention(query, k, v, scale=args.head_dim**-0.5)
            outputs.append(output.transpose(0, 2, 1, 3).reshape(batch, 1, -1))
        output = mx.concatenate(outputs, axis=1)
        if args.use_gated_attention:
            gate = inner.g_proj(x)
            if args.gate_full_precision:
                gate = gate.astype(mx.float32)
            output = output * mx.sigmoid(gate).astype(output.dtype)
        return inner.o_proj(output)


def trim_exact(entry, desired_offset):
    if not hasattr(entry, '_exact_start'):
        return False
    offset, pointer = entry._exact_start
    accepted, width = desired_offset - offset, entry.offset - offset
    if (not isinstance(entry, RotatingKVCache) or not 1 <= accepted <= width <= 16
            or entry.keys is None or entry.values is None or entry.keys.shape != entry.values.shape
            or entry._idx != entry.keys.shape[2]):
        raise ValueError('Exact verification must commit a contiguous anchored prefix')
    end = entry.keys.shape[2] - (width - accepted)
    if end < min(desired_offset, entry.max_size):
        raise ValueError('Exact verification lost required committed history')
    start = max(0, end - entry.max_size)
    keys, values = entry.keys[..., start:end, :], entry.values[..., start:end, :]
    position = virtual_pointer(offset, pointer, accepted, entry.max_size)
    if desired_offset > entry.max_size:
        keys, values = physical_ring(keys, position), physical_ring(values, position)
    entry.keys, entry.values, entry.offset, entry._idx = keys, values, desired_offset, position
    del entry._exact_start
    return True


def install_exact_attention(target):
    if hasattr(target, '_exact_attention_mode'):
        return target._exact_attention_mode
    mode = VerificationMode()
    for layer in target.model.layers:
        while not hasattr(layer, 'self_attn') and hasattr(layer, 'inner'):
            layer = layer.inner
        if isinstance(layer.self_attn, VerificationAttention):
            raise ValueError('Target attention was already wrapped by another verification mode')
        layer.self_attn = VerificationAttention(layer.self_attn, mode)
    target._exact_attention_mode = mode
    return mode
