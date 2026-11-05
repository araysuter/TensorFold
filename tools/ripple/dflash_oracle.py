"""CPU adapters for executing pinned upstream SGLang DFlash ASTs without its GPU runtime."""
import ast
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


SOURCE_COMMIT = '0bcd822377da7b5718e674eaf9c870d349424dd1'
SOURCE_SHA256 = 'bfdef4b6ebe54f0ed88e04295310102c88b4b85798604f736cdb46320f3929ef'
SOURCE_URL = f'https://github.com/sgl-project/sglang/blob/{SOURCE_COMMIT}/python/sglang/srt/models/dflash.py'
SOURCE_LICENSE = 'Apache-2.0'
ACTIVATION_SOURCES = {
    'python/sglang/srt/layers/activation.py':
        '7cb92806f75eaa95529e841c6b506e77da5030b99e7e98110e64a7f3232e489a',
    'python/sglang/kernels/ops/activation/activation.py':
        '1b938a3778c68ce5f24af370c674097dd9b0dd0771d960262369a200b35cd5d4',
    'python/sglang/kernels/jit/csrc/elementwise/activation.cuh':
        'f1b56af7476695688d11bb004dc6cee144cd8922a56683a12c504c53aec26c47',
}
ACTIVATION_MODE = 'CPU formula emulates aligned CUDA JIT SiLU FP32 activation*up with single output cast'



class Norm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, value, residual=None):
        if residual is not None:
            value = value + residual
        source = value.float()
        result = (source * torch.rsqrt(source.square().mean(-1, keepdim=True) + self.variance_epsilon))
        result = result.to(value.dtype) * self.weight
        return result if residual is None else (result, value)


class Linear(nn.Linear):
    def __init__(self, input_size, output_size, bias=False, **unused):
        super().__init__(input_size, output_size, bias=bias)

    def forward(self, value):
        return super().forward(value), None


class MergedLinear(Linear):
    def __init__(self, input_size, output_sizes, **kwargs):
        super().__init__(input_size, sum(output_sizes), **kwargs)


class QKVLinear(Linear):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, **kwargs):
        super().__init__(hidden_size, head_size * (total_num_heads + 2 * total_num_kv_heads), **kwargs)


class SiluAndMul(nn.Module):
    def forward(self, value):
        gate, up = value.chunk(2, dim=-1)
        value = gate.float()
        activated = value / (1 + torch.exp(-value))
        return (activated * up.float()).to(gate.dtype)


class Rotary(nn.Module):
    def __init__(self, width, base, **unused):
        super().__init__()
        self.width, self.base, self.disabled = width, base, False

    def forward(self, positions, query, key):
        if self.disabled:
            return query, key
        frequency = self.base ** (-torch.arange(0, self.width, 2).float() / self.width)
        phase = positions.float()[:, None] * frequency[None]
        cosine, sine = phase.cos(), phase.sin()

        def rotate(value):
            shaped = value.reshape(len(positions), -1, self.width)
            a, b = shaped.float().chunk(2, dim=-1)
            result = torch.cat((a * cosine[:, None] - b * sine[:, None],
                                b * cosine[:, None] + a * sine[:, None]), dim=-1)
            return result.to(value.dtype).reshape_as(value)
        return rotate(query), rotate(key)


class AttentionType:
    DECODER = 'causal'
    ENCODER_ONLY = 'bidirectional'


class CPUAttention(nn.Module):
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, layer_id,
                 sliding_window_size=-1, attn_type=AttentionType.ENCODER_ONLY):
        super().__init__()
        self.num_heads, self.head_dim, self.scaling = num_heads, head_dim, scaling
        self.num_kv_heads, self.layer_id = num_kv_heads, layer_id
        self.window, self.attn_type = sliding_window_size, attn_type

    def forward(self, query, key, value, batch):
        length = len(query)
        query = query.reshape(length, self.num_heads, self.head_dim).transpose(0, 1)
        key = key.reshape(length, self.num_kv_heads, self.head_dim).transpose(0, 1)
        value = value.reshape(length, self.num_kv_heads, self.head_dim).transpose(0, 1)
        past = batch.past[self.layer_id]
        offset = batch.offset
        if past is not None:
            key, value = torch.cat((past[0], key), 1), torch.cat((past[1], value), 1)
        keys = torch.arange(offset + length - key.shape[1], offset + length)
        queries = torch.arange(offset, offset + length)
        mask = torch.ones((length, len(keys)), dtype=torch.bool)
        if self.window >= 0:
            mask &= keys[None] >= queries[:, None] - self.window
        if self.attn_type == AttentionType.DECODER:
            mask &= keys[None] <= queries[:, None]
        repeats = self.num_heads // self.num_kv_heads
        key, value = key.repeat_interleave(repeats, 0), value.repeat_interleave(repeats, 0)
        scores = query.float() @ key.float().transpose(-1, -2) * self.scaling
        probabilities = scores.masked_fill(~mask, float('-inf')).softmax(-1).to(value.dtype)
        result = probabilities @ value
        return result.transpose(0, 1).reshape(length, -1)


def qk_norm(query, key, q_norm, k_norm, width):
    return (q_norm(query.reshape(-1, width)).reshape_as(query),
            k_norm(key.reshape(-1, width)).reshape_as(key))


def config_namespace(config):
    result = SimpleNamespace(**config)
    result.get_text_config = lambda: result
    return result


def draft_config(config):
    data = SimpleNamespace(**config.dflash_config, num_target_layers=None,
                           output_multiplier=1.0, final_logit_softcapping=None)
    data.resolve_block_size = lambda default: data.block_size
    data.resolve_target_layer_ids = lambda **unused: data.target_layer_ids
    return data


def load_reference(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Oracle source SHA256 differs from pinned SGLang reference')
    names = {'_radix_topk', '_project_candidate_logits', '_get_dflash_attention_type',
             '_get_dflash_layer_attention_params', 'DFlashAttention', 'DFlashMLP', '_grouped_conv',
             'DFlashGroupedConv', 'DFlashDecoderLayer', 'DFlashDraftModel', '_score_edges',
             '_follow_maps', 'CandidateSelector', 'DFlash2DraftModel'}
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    for node in selected:
        node.decorator_list = []
        if isinstance(node, ast.ClassDef):
            node.body = [item for item in node.body if getattr(item, 'name', '') != 'load_weights']
    namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional, Tuple=Tuple, Iterable=Iterable,
                     RMSNorm=Norm, QKVParallelLinear=QKVLinear, RowParallelLinear=Linear,
                     MergedColumnParallelLinear=MergedLinear, ReplicatedLinear=Linear,
                     SiluAndMul=SiluAndMul, RadixAttention=CPUAttention, AttentionType=AttentionType,
                     ForwardBatch=SimpleNamespace, LogitsProcessorOutput=SimpleNamespace,
                     get_parallel=lambda: SimpleNamespace(tp_size=1),
                     get_rope_config=lambda c: (c.rope_theta, None),
                     get_rope=lambda width, **kw: Rotary(width, **kw), apply_qk_norm=qk_norm,
                     get_dflash_layer_types=lambda c: c.layer_types,
                     get_dflash_attention_sliding_window_size=lambda c: c.sliding_window - 1,
                     parse_dflash_draft_config=lambda draft_hf_config: draft_config(draft_hf_config),
                     is_nemotron_35_draft_config=lambda c: False,
                     can_dflash_slice_qkv_weight=lambda p: (True, None),
                     should_apply_lm_head_quant_method=lambda h, q: False,
                     is_dense_head_weight=lambda w: w is not None and w.dtype.is_floating_point,
                     _is_npu=False, _flashinfer_top_k=None, logger=logging.getLogger('dflash-oracle'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def make_reference(source, config):
    upstream = load_reference(source)
    model = upstream.DFlash2DraftModel(config_namespace(config))
    for index in config['no_rope_layers']:
        model.layers[index].self_attn.rotary_emb.disabled = True
    return model.eval()


def materialize_context(model, features, offset=0):
    hidden = model.project_target_hidden(torch.cat(features, dim=-1))
    positions = torch.arange(offset, offset + len(hidden))
    entries = []
    for layer in model.layers:
        attention = layer.self_attn
        value = model.prepare_context_hidden_for_kv(layer, hidden)
        key, value = attention.kv_proj_only(value)
        key = attention.apply_k_rope(positions, attention.apply_k_norm(key))
        shape = (len(hidden), attention.num_kv_heads, attention.head_dim)
        entries.append((key.reshape(shape).transpose(0, 1), value.reshape(shape).transpose(0, 1)))
    return hidden, entries


def run_reference(model, embeddings, past, offset):
    batch = SimpleNamespace(past=past, offset=offset)
    return model(None, torch.arange(offset, offset + len(embeddings)), batch,
                 input_embeds=embeddings).hidden_states
