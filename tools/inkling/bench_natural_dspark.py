#!/usr/bin/env python3
"""Experimental greedy DSpark comparison; this verifier is excluded from API serving."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--verify-width", type=int, choices=(2,4,8), default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--width", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--draft-bits", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()
    if (
        not 1 <= args.tokens <= 1024
    ):
        parser.error("Positive prompt lengths plus output must fit 131072 tokens")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must name an existing local checkpoint directory")
    if not args.draft.is_file():
        parser.error("--draft must name an existing local safetensors file")
    if args.output.is_dir():
        parser.error("--output must be a file path")
    args.model = args.model.resolve()
    args.draft = args.draft.resolve()
    import os

    os.environ["MLX_ENABLE_TF32"] = "0"
    import time, json, types, hashlib, struct
    import importlib.metadata

    if (
        importlib.metadata.version("mlx") != "0.32.4"
        or importlib.metadata.version("mlx-lm") != "0.32.3.dev26+g0c4c99192"
    ):
        raise ValueError("Use the pinned MLX 0.32.4 / MLX-LM 0c4c99192 environment")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.inkling2 import layers as ink_layers
    from mlx_lm.models.cache import RotatingKVCache
    from mlx_lm import load
    from mlx_lm.models.inkling2.model import TextModel
    from mlx_lm.models.inkling2.layers import DecoderLayer

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(args.source.resolve()))
    from dspark_bench import model as d
    import re, subprocess, threading

    def available_gib():
        raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
        page = int(re.search(r"page size of (\d+) bytes", raw)[1])
        counts = {k: int(v) for k, v in re.findall(r"^([^:\n]+):\s*(\d+)\.", raw, re.M)}
        return (counts["Pages free"] + counts["Pages inactive"]) * page / 1024**3

    initial_free = available_gib()
    if initial_free < 32:
        raise SystemExit(
            "Stop other inference first; this probe requires at least 32 GiB available"
        )
    minimum_free = [initial_free]
    monitor_stop = threading.Event()
    report = {"status": "loading", "cases": []}

    def save_report(value):
        temporary = output.with_name(
            output.name + "." + str(threading.get_ident()) + ".tmp"
        )
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, output)

    def monitor():
        while not monitor_stop.wait(1):
            try:
                available = available_gib()
                minimum_free[0] = min(minimum_free[0], available)
                if available >= 8:
                    continue
                report["status"] = "stopped_low_memory"
            except Exception:
                report["status"] = "memory_status_unavailable"
            save_report(report)
            os._exit(75)

    threading.Thread(target=monitor, daemon=True).start()
    mx.set_memory_limit(46 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    model, tok = load(str(args.model))
    if model.args.model_type != "inkling2_loop" or len(model.make_cache()) != 48:
        raise ValueError("This probe requires the fixed-loop Inkling 2 Mini checkpoint")
    config = json.loads((args.model / "config.json").read_text())
    draft = d.Inkling2DSparkModel(d.Inkling2DSparkConfig.from_target_config(config))
    draft.load_weights(list(mx.load(str(args.draft)).items()), strict=True)
    if args.draft_bits == 8:
        nn.quantize(
            draft,
            group_size=64,
            bits=8,
            class_predicate=lambda p, m: (
                isinstance(m, nn.Linear) and p.startswith("layers.")
            ),
        )
    eos = set(tok.eos_token_ids)
    text = model.model.language_model
    proxy = types.SimpleNamespace(
        model=types.SimpleNamespace(
            embed_norm=text.embed_norm, embed=text.embed_tokens
        ),
        lm_head=model.lm_head,
    )
    draft.bind(proxy)
    mx.eval(draft.parameters())

    class Trace:
        enabled = False
        visit = 0
        taps = {}

    tr = Trace()
    original_layer = DecoderLayer.__call__
    original_text = TextModel.__call__

    def layer_call(self, hidden_states, *args, **kwargs):
        if tr.enabled:
            if tr.visit in (5, 23, 41):
                tr.taps[tr.visit] = hidden_states
            tr.visit += 1
        if getattr(tr, "verifying", False):
            tr.current_conv = kwargs["cache"][1]
            tr.conv_windows[id(tr.current_conv)] = []
        try:
            return original_layer(self, hidden_states, *args, **kwargs)
        finally:
            tr.current_conv = None

    def text_call(self, *args, **kwargs):
        if tr.enabled:
            tr.visit = 0
            tr.taps = {}
        result = original_text(self, *args, **kwargs)
        if tr.enabled:
            tr.taps[48] = result
        return result

    DecoderLayer.__call__ = layer_call
    TextModel.__call__ = text_call

    def forward(ids, cache, context, offset, need_logits=True):
        tr.enabled = True
        try:
            logits = model(mx.array([ids], dtype=mx.int32), cache=cache)[0]
        finally:
            tr.enabled = False
        taps = mx.concatenate([tr.taps[i] for i in (5, 23, 41, 48)], axis=-1)
        context = (
            mx.concatenate([context, taps], axis=1)[:, -1023:]
            if context is not None
            else taps[:, -1023:]
        )
        mx.eval(
            ([logits] if need_logits else []) + [context] + [c.state for c in cache]
        )
        return logits, context, offset + len(ids)

    def proposals(anchor, context, offset, width):
        cfg = draft.config
        h = mx.broadcast_to(draft.mask_embed, (1, width, cfg.hidden_size))
        h = mx.concatenate(
            [draft._embed_input_tokens(mx.array([[anchor]], dtype=mx.int32)), h[:, 1:]],
            axis=1,
        )
        n = context.shape[1]
        for layer in draft.layers:
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_norm(
                a.q_proj(x).reshape(1, width, a.n_heads, a.head_dim)
            ).transpose(0, 2, 1, 3)
            k = a.k_norm(
                a.k_proj(x).reshape(1, width, a.n_kv_heads, a.head_dim)
            ).transpose(0, 2, 1, 3)
            v = (
                a.v_proj(x)
                .reshape(1, width, a.n_kv_heads, a.head_dim)
                .transpose(0, 2, 1, 3)
            )
            ck, cv = a.project_context_kv(context)
            ck = a.k_norm_ctx(ck.reshape(1, n, a.n_kv_heads, a.head_dim)).transpose(
                0, 2, 1, 3
            )
            cv = cv.reshape(1, n, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
            q = draft.rope(q, offset=offset)
            k = draft.rope(k, offset=offset)
            ck = draft.rope(ck, offset=offset - n)
            aout = mx.fast.scaled_dot_product_attention(
                q,
                mx.concatenate([ck, k], axis=2),
                mx.concatenate([cv, v], axis=2),
                scale=a.scale,
            )
            h = h + a.o_proj(aout.transpose(0, 2, 1, 3).reshape(1, width, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = draft.norm(h)
        logits = draft._logits(h[:, 1:]).astype(mx.float32)
        draft.markov_head.prepare_block(h[:, 1:], context, [n], ["probe"])
        tokens = []
        previous = anchor
        for i in range(width - 1):
            out = draft.markov_head.apply_step_logits(
                logits[:, i], mx.array([previous], dtype=mx.int32)
            )
            previous = int(mx.argmax(out, axis=-1).item())
            tokens.append(previous)
        return tokens

    def greedy(logits):
        scores = logits.reshape(1, -1)
        return int(
            mx.argmax(scores - mx.logsumexp(scores, keepdims=True), axis=-1).item()
        )

    def run(ids, speculative, maximum=64):
        cache = model.make_cache()
        context = None
        offset = 0
        started = time.monotonic()
        for pos in range(0, len(ids) - 1, 512):
            chunk = ids[pos : min(pos + 512, len(ids) - 1)]
            logits, context, offset = forward(
                chunk, cache, context, offset, need_logits=False
            )
        logits, context, offset = forward(ids[-1:], cache, context, offset)
        anchor = greedy(logits[-1])
        outputs = [anchor]
        first = time.monotonic()
        accepted = proposed = rounds = 0
        draft_seconds = verify_seconds = rollback_seconds = 0.0
        while len(outputs) < maximum and outputs[-1] not in eos:
            if not speculative:
                logits, context, offset = forward([anchor], cache, context, offset)
                anchor = greedy(logits[-1])
                outputs.append(anchor)
                continue
            draft_started = time.monotonic()
            remaining = maximum - len(outputs)
            guesses = proposals(anchor, context, offset, args.width) if remaining > 1 else []
            guesses = guesses[:min(args.verify_width - 1, remaining - 1)]
            draft_seconds += time.monotonic() - draft_started
            inputs = [anchor] + guesses
            proposed += len(guesses)
            rounds += 1
            branch = cache
            tr.verifying = True
            tr.conv_windows = {}
            tr.kv = {}
            verify_started = time.monotonic()
            try:
                logits, new_context, new_offset = forward(
                    inputs, branch, context, offset
                )
            finally:
                tr.verifying = False
            verify_seconds += time.monotonic() - verify_started
            rollback_started = time.monotonic()
            new_taps = mx.concatenate([tr.taps[i] for i in (5, 23, 41, 48)], axis=-1)
            consumed = 0
            for i in range(len(inputs)):
                token = greedy(logits[i])
                outputs.append(token)
                consumed = i + 1
                anchor = token
                if token in eos or i >= len(guesses) or token != guesses[i] or len(outputs) >= maximum:
                    break
                accepted += 1
            if consumed == len(inputs):
                cache, context, offset = branch, new_context, new_offset
            else:
                for entry in branch:
                    kv, conv = entry[0], entry[1]
                    old, idx, keys, values = tr.kv[id(kv)]
                    past = keys.shape[2] - len(inputs)
                    end = past + consumed
                    start = (
                        max(0, end - kv.max_size)
                        if isinstance(kv, RotatingKVCache)
                        else 0
                    )
                    keys, values = keys[..., start:end, :], values[..., start:end, :]
                    kv.offset = old + consumed
                    if isinstance(kv, RotatingKVCache):
                        pointer = (
                            kv.offset
                            if kv.offset <= kv.max_size
                            else (idx + consumed - 1) % kv.max_size + 1
                        )
                        if kv.offset > kv.max_size:
                            keys, values = (
                                physical(keys, pointer),
                                physical(values, pointer),
                            )
                        kv._idx = pointer
                    if isinstance(kv, RotatingKVCache):
                        kv.keys, kv.values = keys, values
                    for slot, window in enumerate(tr.conv_windows[id(conv)]):
                        conv[slot] = window[:, consumed : consumed + 3]
                cache = branch
                context = mx.concatenate([context, new_taps[:, :consumed]], axis=1)[
                    :, -1023:
                ]
                offset += consumed
                mx.eval(context, [c.state for c in cache])
            rollback_seconds += time.monotonic() - rollback_started
            del branch
        finish = time.monotonic()
        result = dict(
            tokens=outputs,
            prefill_seconds=first - started,
            prompt_tps=len(ids) / (first - started),
            decode_tps=(len(outputs) - 1) / (finish - first),
            accepted=accepted,
            proposed=proposed,
            draft_seconds=draft_seconds,
            verify_seconds=verify_seconds,
            rollback_seconds=rollback_seconds,
            rounds=rounds,
            peak_gb=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if outputs[-1] in eos else "length",
            output_tokens=len(outputs),
            text=tok.decode(outputs),
            proposal_acceptance=(accepted / proposed if proposed else None),
            proposals_discarded=(proposed - accepted),
            token_sha256=hashlib.sha256(
                b"".join(struct.pack("<I", t) for t in outputs)
            ).hexdigest(),
        )

        for name in ("taps", "kv", "conv_windows"):
            state = getattr(tr, name, None)
            if isinstance(state, dict):
                state.clear()
        return result

    def physical(x, pointer):
        return (
            x
            if pointer == x.shape[2]
            else mx.concatenate([x[..., -pointer:, :], x[..., :-pointer, :]], axis=2)
        )

    orig_attend = ink_layers.Attention._attend
    orig_attention = ink_layers.Attention._attention
    orig_conv = ink_layers._conv_state

    def conv_state(window, lengths, n):
        if getattr(tr, "verifying", False) and tr.current_conv is not None:
            tr.conv_windows[id(tr.current_conv)].append(window)
        return orig_conv(window, lengths, n)

    def exact_attend(self, *args, **kwargs):
        cache = args[5] if len(args) > 5 else kwargs.get("cache")
        if getattr(tr, "verifying", False):
            old, idx = cache.offset, getattr(cache, "_idx", 0)
            tr.kv[id(cache)] = (old, idx, None, None)
        result = orig_attend(self, *args, **kwargs)
        if (
            getattr(tr, "verifying", False)
            and args[0].shape[1] > 1
            and isinstance(cache, RotatingKVCache)
        ):
            old, idx, keys, values = tr.kv[id(cache)]
            end = keys.shape[2]
            start = max(0, end - cache.max_size)
            keys, values = keys[..., start:end, :], values[..., start:end, :]
            pointer = (
                cache.offset
                if cache.offset <= cache.max_size
                else (idx + cache.offset - old - 1) % cache.max_size + 1
            )
            if cache.offset > cache.max_size:
                keys, values = physical(keys, pointer), physical(values, pointer)
            cache.keys, cache.values, cache._idx = keys, values, pointer
        return result

    def exact_attention(self, q, k, v, bias, sinks, cache):
        if not getattr(tr, "verifying", False):
            return orig_attention(self, q, k, v, bias, sinks, cache)
        old, idx, _, _ = tr.kv[id(cache)]
        tr.kv[id(cache)] = (old, idx, k, v)
        if q.shape[2] == 1:
            return orig_attention(self, q, k, v, bias, sinks, cache)
        length = q.shape[2]
        past = k.shape[2] - length
        rows = []
        for i in range(length):
            end = past + i + 1
            start = max(0, end - self.sliding_window) if self.sliding_window else 0
            ki, vi = k[..., start:end, :], v[..., start:end, :]
            if self.sliding_window:
                single = RotatingKVCache(cache.max_size, keep=0)
                single.offset = old + i + 1
                pointer = (
                    single.offset
                    if single.offset <= cache.max_size
                    else (idx + i) % cache.max_size + 1
                )
                single._idx = pointer
                if single.offset > cache.max_size:
                    ki, vi = physical(ki, pointer), physical(vi, pointer)
            else:
                single = types.SimpleNamespace(offset=old + i + 1)
            rows.append(
                orig_attention(
                    self,
                    q[:, :, i : i + 1],
                    ki,
                    vi,
                    bias[:, i : i + 1],
                    sinks[:, i : i + 1],
                    single,
                )
            )
        return mx.concatenate(rows, axis=1)

    ink_layers.Attention._attend = exact_attend
    ink_layers.Attention._attention = exact_attention
    ink_layers._conv_state = conv_state

    def per_token(original):
        def call(self, x, *args, **kwargs):
            if getattr(tr, "verifying", False) and x.ndim >= 3 and x.shape[1] > 1:
                return mx.concatenate(
                    [
                        original(self, x[:, i : i + 1], *args, **kwargs)
                        for i in range(x.shape[1])
                    ],
                    axis=1,
                )
            return original(self, x, *args, **kwargs)

        return call

    nn.Linear.__call__ = per_token(nn.Linear.__call__)
    nn.QuantizedLinear.__call__ = per_token(nn.QuantizedLinear.__call__)
    ink_layers.SparseMoeBlock.__call__ = per_token(ink_layers.SparseMoeBlock.__call__)
    ink_layers.RelativeLogits.__call__ = per_token(ink_layers.RelativeLogits.__call__)

    report = dict(
        status="running",
        experimental=True,
        production_enabled=False,
        width=args.width,
        draft_bits=args.draft_bits,
        verify_width=args.verify_width,
        source_method="Native chat template, reasoning_effort none, EOS respected",
        output_tokens=args.tokens,
        cases=[],
    )
    def tokenize(messages):
        return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       reasoning_effort="none")

    plans = [
        ("prose", [{"role":"user", "content": "Explain how a hash table handles collisions. Compare separate chaining and open addressing, and give one concrete worked example for each. Use concise prose in about 250 words."}]),
        ("code", [{"role":"user", "content": "Write a Python implementation of a bounded LRU cache with get and put methods. Use collections.OrderedDict, reject non-positive capacity, and include four small unittest cases. Return the complete Python file."}]),
    ]
    if args.long_prompt:
        paragraph = ("Repository note: The cache service keeps an ordered mapping of keys to values. "
                     "Reading an entry makes it most recently used. Adding an entry beyond capacity evicts "
                     "the oldest entry. An update replaces the existing value without increasing size. "
                     "Tests cover recency changes, replacement, eviction and invalid capacity. "
                     "The server exposes this utility behind a small HTTP endpoint and keeps each request isolated.\n")
        filler=tok.encode(paragraph * 180)
        question="Using these repository notes, write the complete Python LRUCache class and four concise unittest cases."
        remaining=7900
        for _ in range(16):
            messages=[{"role":"user","content":"Project reference notes:\n"+tok.decode(filler[:remaining])+"\nTask: "+question}]
            count=len(tokenize(messages))
            if count==8192:break
            remaining+=8192-count
        plans.append(("natural-notes-8k",messages))
    warm_ids=tokenize([{"role":"user","content":"Count from one to twenty."}])
    warm_plain=run(warm_ids,False,16)
    warm_draft=run(warm_ids,True,16)
    if warm_plain["tokens"]!=warm_draft["tokens"]:
        report["status"]="failed_warmup_exactness"
        save_report(report)
        return 2
    report["warmup_exact"]=True
    for name,messages in plans:
        ids=tokenize(messages)
        length=len(ids)
        if length + args.tokens > 131072:
            raise ValueError("Prompt plus output exceeds 128K context")
        plain = run(ids, False, args.tokens)
        candidate = run(ids, True, args.tokens)
        case = dict(
            name=name,
            input_tokens=length,
            plain=plain,
            candidate=candidate,
            exact=plain["tokens"] == candidate["tokens"],
        )
        report["cases"].append(case)
        save_report(report)
        print(
            "case",
            name,
            "context",
            length,
            "exact",
            case["exact"],
            "plain",
            plain["prompt_tps"],
            plain["decode_tps"],
            "candidate",
            candidate["prompt_tps"],
            candidate["decode_tps"],
            flush=True,
        )
        if not case["exact"]:
            report["status"] = "failed_exactness"
            break
    else:
        report["status"] = "passed"
    save_report(report)
    print(report["status"], flush=True)

    monitor_stop.set()
    report["minimum_available_gib"] = minimum_free[0]
    save_report(report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
