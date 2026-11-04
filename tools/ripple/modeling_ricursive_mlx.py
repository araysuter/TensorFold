"""MLX Ripple reference adapter, derived from Ricursive's Apache-2.0 checkpoint specification."""
from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = 'ricursive'
    hidden_size: int = 5120
    intermediate_size: int = 40960
    num_hidden_layers: int = 80
    num_attention_heads: int = 80
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    hidden_act: str = 'silu'
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 1024
    layer_types: Optional[list] = None
    embed_scale: Optional[float] = 71.55417527999327
    ffn_clamp_threshold: Optional[float] = 20.0
    use_valres: bool = True
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 2097152

    def __post_init__(self):
        if self.model_type != 'ricursive' or self.tie_word_embeddings or self.hidden_act != 'silu':
            raise ValueError('Unsupported Ripple model configuration')
        if self.layer_types is None:
            self.layer_types = ['full_attention' if (i + 1) % 5 == 0 else 'sliding_attention'
                                for i in range(self.num_hidden_layers)]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError('Invalid Ripple layer pattern length')
        if any(x not in ('full_attention', 'sliding_attention') for x in self.layer_types):
            raise ValueError('Unsupported Ripple attention type')
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError('Invalid Ripple head geometry')


class RMSNorm(nn.Module):
    def __init__(self, dims, eps):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        value = x.astype(mx.float32)
        variance = mx.mean(value * value, axis=-1, keepdims=True)
        normalized = value * mx.rsqrt(variance + self.eps)
        return self.weight * normalized.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.args = args
        self.sliding = args.layer_types[index] == 'sliding_attention'
        for name, rows in [('q_proj', args.num_attention_heads * args.head_dim),
                           ('k_proj', args.num_key_value_heads * args.head_dim),
                           ('v_proj', args.num_key_value_heads * args.head_dim)]:
            setattr(self, name, nn.Linear(args.hidden_size, rows, bias=False))
        self.o_proj = nn.Linear(args.num_attention_heads * args.head_dim, args.hidden_size, bias=False)
        self.q_norm = RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        if args.use_gated_attention:
            self.g_proj = nn.Linear(args.hidden_size, args.num_attention_heads * args.head_dim, bias=False)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))

    def _rope(self, value, offset):
        args = self.args
        positions = mx.arange(offset, offset + value.shape[2], dtype=mx.float32)
        frequency = args.rope_theta ** (-mx.arange(0, args.head_dim, 2, dtype=mx.float32) / args.head_dim)
        angles = positions[:, None] * frequency[None, :]
        angles = mx.concatenate([angles, angles], axis=-1)
        cosine = mx.cos(angles).astype(value.dtype).astype(mx.float32)
        sine = mx.sin(angles).astype(value.dtype).astype(mx.float32)
        source = value.astype(mx.float32)
        first, last = mx.split(source, 2, axis=-1)
        rotated = mx.concatenate([-last, first], axis=-1)
        return (source * cosine + rotated * sine).astype(value.dtype)

    def __call__(self, x, seed, mask, cache):
        args = self.args
        batch, length, _ = x.shape
        value_input = x
        if seed is not None:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            source, current = seed.astype(mx.float32), x.astype(mx.float32)
            value_input = (source + blend.astype(mx.float32) * (current - source)).astype(x.dtype)
        q = self.q_proj(x).reshape(batch, length, args.num_attention_heads, args.head_dim)
        k = self.k_proj(x).reshape(batch, length, args.num_key_value_heads, args.head_dim)
        v = self.v_proj(value_input).reshape(batch, length, args.num_key_value_heads, args.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if self.sliding:
            offset = cache.offset if cache is not None else 0
            q, k = self._rope(q, offset), self._rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = scaled_dot_product_attention(q, k, v, cache=cache, scale=args.head_dim**-0.5, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if args.use_gated_attention:
            gate = self.g_proj(x)
            if args.gate_full_precision:
                gate = gate.astype(mx.float32)
            y = y * mx.sigmoid(gate).astype(y.dtype)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.threshold = args.ffn_clamp_threshold
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.threshold:
            gate = mx.minimum(gate, self.threshold)
            up = mx.clip(up, -self.threshold, self.threshold)
        value = gate.astype(mx.float32)
        activated = (value * mx.sigmoid(value)).astype(gate.dtype)
        return self.down_proj(activated * up)


class Layer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.self_attn = Attention(args, index)
        self.mlp = MLP(args)
        for name in ['input_layernorm', 'post_attention_layernorm',
                     'pre_feedforward_layernorm', 'post_feedforward_layernorm']:
            setattr(self, name, RMSNorm(args.hidden_size, eps=args.rms_norm_eps))

    def __call__(self, x, seed, mask, cache):
        h = self.self_attn(self.input_layernorm(x), seed, mask, cache)
        x = x + self.post_attention_layernorm(h)
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, index) for index in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if self.args.embed_scale is not None:
            h = h * self.args.embed_scale
        seed = h if self.args.use_valres else None
        caches = cache if cache is not None else [None] * len(self.layers)
        if len(caches) != len(self.layers):
            raise ValueError('Invalid Ripple cache count')
        masks = {}
        for kind, c in zip(self.args.layer_types, caches):
            if kind not in masks:
                masks[kind] = create_attention_mask(h, c, window_size=self.args.sliding_window
                                                    if kind == 'sliding_attention' else None)
        for layer, kind, c in zip(self.layers, self.args.layer_types, caches):
            h = layer(h, seed, masks[kind], c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if kind == 'sliding_attention' else KVCache() for kind in self.args.layer_types]

    def sanitize(self, weights):
        return weights
