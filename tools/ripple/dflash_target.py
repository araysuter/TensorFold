"""Capture Ripple target taps and keep its real and draft context caches synchronized."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import RotatingKVCache
from contextlib import contextmanager
import os
from dflash_exact import install_exact_attention, trim_exact


class Taps:
    def __init__(self):
        self.values = {}


class TapLayer(nn.Module):
    def __init__(self, inner, taps, index):
        super().__init__()
        self.inner, self.taps, self.index = inner, taps, index

    def __call__(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        self.taps.values[self.index] = result
        return result


class DraftedTarget(nn.Module):
    def __init__(self, target, draft, *, exact_verification=False):
        super().__init__()
        self.target, self.draft = target, draft
        self.draft_block_size = min(4, draft.args.block_size)
        self.prefill_clear_interval, self.adaptive_blocks = 0, False
        self.exact_mode = install_exact_attention(target) if exact_verification else None
        self.args, self.model_type = target.args, target.model_type
        self.taps = Taps()
        a, b = target.args, draft.args
        if (a.hidden_size, a.vocab_size, a.num_key_value_heads, a.head_dim) != (
                b.hidden_size, b.vocab_size, b.num_key_value_heads, b.head_dim):
            raise ValueError('Target/drafter dimensions do not match')
        if len(set(b.target_layer_ids)) != len(b.target_layer_ids) or any(
                not 0 <= i < a.num_hidden_layers for i in b.target_layer_ids):
            raise ValueError('Invalid target tap indices')
        for index in b.target_layer_ids:
            target.model.layers[index] = TapLayer(target.model.layers[index], self.taps, index)
        self.counters = {'draft_proposed_tokens_total': 0, 'draft_accepted_tokens_total': 0,
                         'draft_verify_calls_total': 0}

    @property
    def layers(self):
        return self.target.layers

    def make_cache(self):
        return self.target.make_cache() + self.draft.make_cache()

    def __call__(self, inputs, cache):
        count = self.args.num_hidden_layers
        self.taps.values.clear()
        logits = self.target(inputs, cache=cache[:count])
        features = [self.taps.values[i] for i in self.draft.args.target_layer_ids]
        self.draft.append_context(features, cache[count:])
        self.taps.values.clear()
        return logits

    def draft_lattice(self, anchor, cache, length):
        noise = mx.full((1, length), self.draft.args.mask_token_id, dtype=mx.int32)
        noise[0, 0] = anchor
        hidden = self.draft(self.target.model.embed_tokens(noise), cache[self.args.num_hidden_layers:])
        return self.draft.candidate_selector.lattice(hidden[0, 1:], self.target.lm_head, anchor)

    @contextmanager
    def verification(self):
        if self.exact_mode is None:
            yield
            return
        if self.exact_mode.enabled or os.environ.get('MLX_RIPPLE_EXACT_VERIFY') == '1':
            raise ValueError('Nested exact verification is not supported')
        previous = os.environ.get('MLX_RIPPLE_EXACT_VERIFY')
        self.exact_mode.enabled = True
        os.environ['MLX_RIPPLE_EXACT_VERIFY'] = '1'
        try:
            yield
        finally:
            self.exact_mode.enabled = False
            if previous is None:
                os.environ.pop('MLX_RIPPLE_EXACT_VERIFY', None)
            else:
                os.environ['MLX_RIPPLE_EXACT_VERIFY'] = previous


def trim_verification(cache, offset):
    for entry in cache:
        if trim_exact(entry, offset):
            continue
        removed = entry.offset - offset
        if removed < 0:
            raise ValueError('Cannot extend a verified prefix while trimming')
        if not removed:
            continue
        if isinstance(entry, RotatingKVCache):
            if entry._idx != entry.keys.shape[2]:
                raise ValueError('Speculative rewind requires a contiguous verification block')
            end = entry.keys.shape[2] - removed
            if end <= 0:
                raise ValueError('Verification must commit its known target anchor')
            entry.keys, entry.values = entry.keys[..., :end, :], entry.values[..., :end, :]
            entry._idx, entry.offset = end, offset
        else:
            entry.trim(removed)
