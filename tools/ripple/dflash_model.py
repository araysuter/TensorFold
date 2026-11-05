"""Ripple DFlash2 blocks, grouped convolutions and candidate lattice expressed with MLX."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache
from modeling_ricursive_mlx import RMSNorm


@dataclass
class DraftArgs:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    layer_types: list
    no_rope_layers: list
    target_layer_ids: list
    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    mask_token_id: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, config):
        if config.get('architectures') != ['DFlash2DraftModel'] or config.get('hidden_act', 'silu') != 'silu':
            raise ValueError('Requires an actual DFlash2 checkpoint')
        if config.get('attention_bias', False) or config.get('is_causal', False):
            raise ValueError('Unsupported draft attention configuration')
        d = config['dflash_config']
        fields = {k: config[k] for k in ['hidden_size', 'intermediate_size', 'num_hidden_layers',
                  'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size',
                  'sliding_window', 'layer_types', 'no_rope_layers']}
        fields.update({k: d[k] for k in ['target_layer_ids', 'block_size', 'conv_kernel_size',
                       'conv_group_size', 'selector_rank', 'selector_top_k', 'mask_token_id']})
        fields.update(rms_norm_eps=config.get('rms_norm_eps', 1e-6), rope_theta=config.get('rope_theta', 10000.0))
        result = cls(**fields)
        if len(result.layer_types) != result.num_hidden_layers:
            raise ValueError('Draft layer-pattern length mismatch')
        if sorted(result.no_rope_layers) != [i for i, kind in enumerate(result.layer_types) if kind == 'full_attention']:
            raise ValueError('Ripple draft full layers must be NoPE')
        if (any(t not in ['sliding_attention', 'full_attention'] for t in result.layer_types)
                or result.hidden_size % result.conv_group_size or result.num_attention_heads % result.num_key_value_heads
                or not 1 <= result.selector_top_k <= result.vocab_size
                or not 1 <= result.block_size <= 16 or not 0 <= result.mask_token_id < result.vocab_size):
            raise ValueError('Invalid draft geometry')
        return result


def ordered_kv(cache):
    if cache.keys is None:
        return None, None
    if isinstance(cache, RotatingKVCache):
        return cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    return cache.keys[..., :cache.offset, :], cache.values[..., :cache.offset, :]


class GroupedConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.groups = args.hidden_size // args.conv_group_size
        self.base_kernel = mx.zeros((2, args.conv_kernel_size, args.hidden_size))
        self.kernel_projection = nn.Linear(args.hidden_size, 2 * args.conv_kernel_size * self.groups, bias=False)

    def convolve(self, values, delta, side):
        batch, length, _ = values.shape
        blocks = values.reshape(batch, length, self.groups, self.args.conv_group_size)
        base = self.base_kernel[side].reshape(self.args.conv_kernel_size, self.groups, self.args.conv_group_size)
        coefficients = base[None, None] + delta[..., None]
        output = coefficients[:, :, 0] * blocks
        for tap in range(1, self.args.conv_kernel_size):
            if tap >= length:
                break
            shifted = mx.concatenate([mx.zeros_like(blocks[:, :tap]), blocks[:, :-tap]], axis=1)
            mask = ((mx.arange(length) % self.args.block_size) >= tap)[None, :, None, None]
            output = output + coefficients[:, :, tap] * shifted * mask
        return output.reshape(values.shape)

    def prepare(self, values):
        delta = self.kernel_projection(values).reshape(*values.shape[:-1], 2, self.args.conv_kernel_size, self.groups)
        return self.convolve(values, delta[..., 0, :, :], 0), delta[..., 1, :, :]

    def finish(self, values, coefficients):
        return self.convolve(values, coefficients, 1)


class Attention(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.args, self.index = args, index
        for name, heads in [('q_proj', args.num_attention_heads), ('k_proj', args.num_key_value_heads),
                            ('v_proj', args.num_key_value_heads)]:
            setattr(self, name, nn.Linear(args.hidden_size, heads * args.head_dim, bias=False))
        self.o_proj = nn.Linear(args.num_attention_heads * args.head_dim, args.hidden_size, bias=False)
        self.q_norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        self.rope = None if index in args.no_rope_layers else nn.RoPE(args.head_dim, traditional=False, base=args.rope_theta)

    def project_kv(self, values, offset):
        batch, length, _ = values.shape
        keys = self.k_proj(values).reshape(batch, length, self.args.num_key_value_heads, self.args.head_dim)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        if self.rope is not None:
            keys = self.rope(keys, offset=offset)
        vals = self.v_proj(values).reshape(batch, length, self.args.num_key_value_heads,
                                         self.args.head_dim).transpose(0, 2, 1, 3)
        return keys, vals

    def __call__(self, values, context):
        batch, length, _ = values.shape
        query = self.q_proj(values).reshape(batch, length, self.args.num_attention_heads, self.args.head_dim)
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        if self.rope is not None:
            query = self.rope(query, offset=context.offset)
        keys, vals = self.project_kv(values, context.offset)
        past_k, past_v = ordered_kv(context)
        past_length = 0 if past_k is None else past_k.shape[2]
        if past_k is not None:
            keys, vals = mx.concatenate([past_k, keys], axis=2), mx.concatenate([past_v, vals], axis=2)
        mask = None
        if self.args.layer_types[self.index] == 'sliding_attention':
            positions = mx.arange(context.offset - past_length, context.offset + length)
            queries = mx.arange(context.offset, context.offset + length)
            mask = positions[None, :] >= queries[:, None] - self.args.sliding_window + 1
        output = mx.fast.scaled_dot_product_attention(query, keys, vals, scale=self.args.head_dim**-0.5, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, values):
        gate = self.gate_proj(values).astype(mx.float32)
        up = self.up_proj(values).astype(mx.float32)
        return self.down_proj((gate * mx.sigmoid(gate) * up).astype(values.dtype))


class Layer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = Attention(args, index)
        self.mlp = MLP(args)
        self.attention_conv, self.mlp_conv = GroupedConv(args), GroupedConv(args)

    def __call__(self, hidden, context):
        values, coefficients = self.attention_conv.prepare(self.input_layernorm(hidden))
        hidden = hidden + self.attention_conv.finish(self.self_attn(values, context), coefficients)
        values, coefficients = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return hidden + self.mlp_conv.finish(self.mlp(values), coefficients)


class CandidateSelector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_projection = nn.Linear(args.hidden_size, args.selector_rank, bias=False)
        self.predecessor_codebook = mx.zeros((args.vocab_size, args.selector_rank))
        self.successor_codebook = mx.zeros((args.vocab_size, args.selector_rank))
        self.top_k = args.selector_top_k

    def lattice(self, hidden, head, anchor):
        logits = head(hidden).astype(mx.float32)
        ids = mx.argpartition(-logits, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        unary = mx.take_along_axis(logits, ids, axis=-1)
        previous = mx.concatenate([mx.full((1, self.top_k), anchor, dtype=ids.dtype), ids[:-1]], axis=0)
        predecessors = self.predecessor_codebook[previous] * self.hidden_projection(hidden)[:, None, :]
        successors = self.successor_codebook[ids]
        scores = unary[:, None, :] + mx.matmul(predecessors, successors.transpose(0, 2, 1)).astype(mx.float32)
        return ids, scores


class DraftModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc = nn.Linear(len(args.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.candidate_selector = CandidateSelector(args)

    def make_cache(self):
        return [RotatingKVCache(self.args.sliding_window, keep=0) if t == 'sliding_attention' else KVCache()
                for t in self.args.layer_types]

    def append_context(self, features, cache):
        hidden = self.hidden_norm(self.fc(mx.concatenate(features, axis=-1)))
        for layer, entry in zip(self.layers, cache):
            keys, values = layer.self_attn.project_kv(hidden, entry.offset)
            entry.update_and_fetch(keys, values)

    def __call__(self, embeddings, cache):
        hidden = embeddings
        for layer, entry in zip(self.layers, cache):
            hidden = layer(hidden, entry)
        return self.norm(hidden)


def load_draft(path):
    path = Path(path)
    if (path / 'CONVERSION_INCOMPLETE').exists():
        raise ValueError('Incomplete drafter conversion')
    config = json.loads((path / 'config.json').read_text())
    record = json.loads((path / 'conversion-receipt.json').read_text())
    weight_file = path / 'model.safetensors'
    digest = hashlib.sha256()
    with weight_file.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError('Drafter weights failed their recorded checksum')
    args = DraftArgs.from_dict(config)
    model = DraftModel(args)
    quant = config['quantization']
    nn.quantize(model, group_size=quant['group_size'], bits=quant['bits'],
                class_predicate=lambda name, module: quant.get(name, False))
    model.load_weights(list(mx.load(str(weight_file)).items()), strict=True)
    mx.eval(model.parameters())
    return model
