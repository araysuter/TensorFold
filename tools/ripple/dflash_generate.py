"""Target-sampled DFlash2 decoding with bounded verification and transactional KV rewind."""
import time
from contextlib import nullcontext
import mlx.core as mx
from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
from mlx_lm.sample_utils import greedy_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from dflash_target import trim_verification


def sample(logits, history, sampler, processors):
    for processor in processors or ():
        logits = processor(history, logits)
    probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return sampler(probabilities), probabilities.squeeze(0)


def propose(model, anchor, cache, length, history, sampler, processors):
    ids, lattice = model.draft_lattice(anchor, cache, length)
    key = mx.random.state[0]
    if key.dtype != mx.uint32 or key.shape != (2,):
        raise ValueError('Unqualified MLX random-key representation')
    high, low = key.tolist()
    saved = (high << 32) | low
    tokens, previous = [], mx.array(0, dtype=mx.int32)
    try:
        for position in range(length - 1):
            scores = lattice[position, previous]
            logits = mx.full((1, model.args.vocab_size), -float('inf'), dtype=mx.float32)
            logits[0, ids[position]] = scores
            sampled, _ = sample(logits, history, sampler, processors)
            token = sampled.reshape(())
            tokens.append(token)
            previous = mx.argmax(ids[position] == token)
            history = mx.concatenate([history, token.reshape(1)])
        proposed = mx.stack(tokens).astype(mx.int32)
        mx.eval(proposed, mx.random.state)
        return proposed
    finally:
        mx.random.seed(saved)


def generate_step(prompt, model, stream=generation_stream, *, max_tokens=256, sampler=None,
                  logits_processors=None, prompt_cache=None, prefill_step_size=512,
                  prompt_progress_callback=None, eos_token_ids=(), **unsupported):
    if unsupported or max_tokens < 1 or not 0 < len(prompt) or prefill_step_size < 1:
        raise ValueError('Unsupported DFlash2 generation arguments')
    cache = prompt_cache if prompt_cache is not None else model.make_cache()
    width = getattr(model, 'draft_block_size', model.draft.args.block_size)
    adaptive = getattr(model, 'adaptive_blocks', False)
    if adaptive:
        width = min(4, model.draft.args.block_size)
    if type(width) is not int or not 1 <= width <= model.draft.args.block_size:
        raise ValueError('Draft verification width exceeds the trained block')
    sampler = sampler or greedy_sampler
    progress = prompt_progress_callback or (lambda *_: None)
    history = prompt.astype(mx.int32)
    total, done = len(prompt), 0
    clear_interval = getattr(model, 'prefill_clear_interval', 0)
    if clear_interval not in (0, 4096, 8192):
        raise ValueError('Unqualified prompt allocator interval')
    cleared = 0
    with mx.stream(stream):
        progress(0, total)
        while total - done > 1:
            count = min(prefill_step_size, total - done - 1)
            model(prompt[None, done:done + count], cache=cache)
            mx.eval([entry.state for entry in cache])
            done += count
            progress(done, total)
            if clear_interval == 0 or done - cleared >= clear_interval:
                mx.clear_cache()
                cleared = done
        logits = model(prompt[None, -1:], cache=cache)[:, -1, :]
        sampled, probabilities = sample(logits, history, sampler, logits_processors)
        anchor = sampled.item()
        progress(total, total)
        history = mx.concatenate([history, mx.array([anchor], dtype=mx.int32)])
        emitted = 1
        yield anchor, probabilities, False
        while emitted < max_tokens and anchor not in eos_token_ids:
            offset = cache[0].offset
            length = min(width, max_tokens - emitted)
            if length > 1:
                proposed = propose(model, anchor, cache, length, history, sampler, logits_processors)
                inputs = mx.concatenate([mx.array([anchor], dtype=mx.int32), proposed])
                model.counters['draft_proposed_tokens_total'] += length - 1
            else:
                inputs = mx.array([anchor], dtype=mx.int32)
            with model.verification() if hasattr(model, 'verification') else nullcontext():
                logits = model(inputs[None], cache=cache)[0]
                mx.eval(logits, [entry.state for entry in cache])
            model.counters['draft_verify_calls_total'] += 1
            committed = 1
            accepted_count = 0
            try:
                for index in range(length):
                    sampled, probabilities = sample(logits[index:index + 1], history, sampler, logits_processors)
                    token = sampled.item()
                    accepted = index < length - 1 and token == int(inputs[index + 1].item())
                    if accepted:
                        model.counters['draft_accepted_tokens_total'] += 1
                        accepted_count += 1
                    committed = index + 1
                    history = mx.concatenate([history, mx.array([token], dtype=mx.int32)])
                    anchor, emitted = token, emitted + 1
                    yield token, probabilities, accepted
                    if not accepted or token in eos_token_ids or emitted >= max_tokens:
                        break
            finally:
                trim_verification(cache, offset + committed)
            if adaptive and length == width:
                if accepted_count == length - 1:
                    width = min(model.draft.args.block_size, width * 2)
                elif accepted_count < (length - 1) / 2:
                    width = min(model.draft.args.block_size, max(4, width // 2))
                model.draft_block_size = width
            if emitted % 256 == 0:
                mx.clear_cache()


def stream_generate(model, tokenizer, prompt, max_tokens=256, stream=generation_stream, **kwargs):
    if max_tokens < 1:
        raise ValueError('A positive output budget is required')
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if isinstance(prompt, str):
        special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt = tokenizer.encode(prompt, add_special_tokens=special)
    if not isinstance(prompt, mx.array):
        prompt = mx.array(prompt, dtype=mx.int32)
    detokenizer = tokenizer.detokenizer
    generator = generate_step(prompt, model, stream, max_tokens=max_tokens,
                              eos_token_ids=tokenizer.eos_token_ids, **kwargs)
    with wired_limit(model, [stream]):
        started = time.perf_counter()
        try:
            for index, (token, probabilities, from_draft) in enumerate(generator):
                if index == 0:
                    prompt_time = time.perf_counter() - started
                    decode_started = time.perf_counter()
                final = token in tokenizer.eos_token_ids or index + 1 == max_tokens
                if token not in tokenizer.eos_token_ids:
                    detokenizer.add_token(token)
                if final:
                    detokenizer.finalize()
                yield GenerationResponse(
                    text=detokenizer.last_segment, token=token, logprobs=probabilities, from_draft=from_draft,
                    prompt_tokens=prompt.size, prompt_tps=prompt.size / max(prompt_time, 1e-9),
                    generation_tokens=index + 1,
                    generation_tps=(index + 1) / max(time.perf_counter() - decode_started, 1e-9),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=('stop' if token in tokenizer.eos_token_ids else 'length') if final else None)
                if final:
                    break
        finally:
            generator.close()
