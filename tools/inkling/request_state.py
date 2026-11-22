"""Independent request state advanced through the shared upstream MLX decoder."""
import functools
import secrets
import mlx.core as mx
from mlx_lm import server as api
from mlx_lm.generate import generate_step, generation_stream
from mlx_lm.models.cache import make_prompt_cache


def with_prefix(processor, prefix, tokens, logits):
    return processor(mx.concatenate([prefix, tokens]), logits)


class RequestState:
    def __init__(self, model, tokenizer, prompt, vision, args, ctx, prefill_size):
        self.model, self.tokenizer, self.args, self.ctx = model, tokenizer, args, ctx
        self.prompt = mx.array(prompt, dtype=mx.int32)
        self.cache = make_prompt_cache(model)
        self.position, self.generated = 0, 0
        self.prefill_size = prefill_size
        self.iterator = None
        self.detokenizer = tokenizer.detokenizer
        self.stop_reader = api.TextStateMachine.from_stop_words(args.stop_words).reader()
        self.rng = None
        self.seed = args.seed if args.seed is not None else secrets.randbits(64)
        self.embeddings = None
        if vision is not None:
            self.embeddings = model.embed_inputs(self.prompt, pixel_values=vision.pixel_values)
            mx.eval(self.embeddings)
        self.done = False

    def advance(self):
        total = len(self.prompt)
        if self.ctx._should_stop:
            self.done = True
            return None
        if self.position < total - 1:
            count = min(self.prefill_size, total - self.position - 1)
            embeddings = None if self.embeddings is None else self.embeddings[None, self.position:self.position + count]
            self.model(self.prompt[None, self.position:self.position + count], cache=self.cache,
                       input_embeddings=embeddings)
            mx.eval([entry.state for entry in self.cache])
            self.position += count
            return self.position, total
        if self.iterator is None:
            processors = api._make_logits_processors(self.args)
            prefix = self.prompt[:-1]
            if processors:
                processors = [functools.partial(with_prefix, p, prefix) for p in processors]
            self.iterator = generate_step(
                self.prompt[-1:], self.model, max_tokens=self.args.max_tokens,
                sampler=api._make_sampler(self.args, self.tokenizer), logits_processors=processors,
                prompt_cache=self.cache, prefill_step_size=self.prefill_size,
                input_embeddings=None if self.embeddings is None else self.embeddings[-1:],
                stream=generation_stream)
        if self.rng is None:
            mx.random.seed(self.seed)
        else:
            mx.random.seed(self.rng)
        token, logprobs = next(self.iterator)
        high, low = mx.random.state[0].tolist()
        self.rng = (high << 32) | low
        token = int(token)
        self.generated += 1
        finish = 'stop' if token in self.tokenizer.eos_token_ids else (
            'length' if self.generated >= self.args.max_tokens else None)
        if finish == 'stop':
            self.detokenizer.finalize()
        else:
            self.detokenizer.add_token(token)
            if finish:
                self.detokenizer.finalize()
        text, finish = api._apply_stop_words(self.stop_reader, self.detokenizer.last_segment, finish_reason=finish)
        self.done = finish is not None
        logprob = logprobs[token].item() if getattr(self.args, 'logprobs', False) else 0.0
        return api.Response(text, token, logprob, finish,
                            api._format_top_logprobs(logprobs, self.args.top_logprobs, self.tokenizer))

    def close(self):
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None
        self.cache.clear()
        self.embeddings = None
        self.prompt = None
        self.rng = None
