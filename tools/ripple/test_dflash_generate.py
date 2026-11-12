#!/usr/bin/env python3
"""Compare DFlash target sampling to an independent Markov oracle and exercise transactional shutdown."""
import sys
from types import SimpleNamespace
import unittest
try:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from dflash_generate import generate_step
    from dflash_model import ordered_kv
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: DFlash decoder tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


VOCAB = 13


def transition(token, shift=0):
    scores = np.full(VOCAB, -2.0, dtype=np.float32)
    scores[(token + 1 + shift) % VOCAB] = 2.4
    scores[(token + 2 + shift) % VOCAB] = 1.7
    return scores


class MarkovTarget:
    def __init__(self, mode='match'):
        self.args = SimpleNamespace(vocab_size=VOCAB)
        self.draft = SimpleNamespace(args=SimpleNamespace(block_size=16))
        self.mode, self.last_width = mode, 0
        self.counters = {'draft_proposed_tokens_total': 0, 'draft_accepted_tokens_total': 0,
                         'draft_verify_calls_total': 0}
        self.transitions = mx.array(np.stack([transition(i) for i in range(VOCAB)]))

    def make_cache(self):
        return [RotatingKVCache(8, keep=0), KVCache(), RotatingKVCache(16, keep=0)]

    def __call__(self, inputs, cache):
        self.last_width = inputs.shape[1]
        keys = mx.broadcast_to(inputs[:, None, :, None], (1, 1, inputs.shape[1], 2)).astype(mx.bfloat16)
        for entry in cache:
            entry.update_and_fetch(keys, -keys)
        return self.transitions[inputs]

    def draft_lattice(self, anchor, cache, length):
        ids = np.broadcast_to(np.arange(VOCAB, dtype=np.int32), (length - 1, VOCAB)).copy()
        scores = np.empty((length - 1, VOCAB, VOCAB), dtype=np.float32)
        for slot in range(length - 1):
            shift = 4 if self.mode == 'reject' or (self.mode == 'partial' and slot == 2) else 0
            for previous in range(VOCAB):
                scores[slot, previous] = transition(anchor if slot == 0 else previous, shift)
        return mx.array(ids), mx.array(scores)


def categorical(logprobs):
    return mx.random.categorical(logprobs, axis=-1)


def greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


def random_state():
    mx.eval(mx.random.state)
    return [np.array(value) for value in mx.random.state]


def oracle(prompt, count, sampler, seed, eos=(), processor=None):
    mx.random.seed(seed)
    history, tokens = list(prompt), []
    for _ in range(count):
        logits = mx.array(transition(history[-1])[None])
        if processor is not None:
            logits = processor(mx.array(history), logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = sampler(logprobs).item()
        tokens.append(token)
        history.append(token)
        if token in eos:
            break
    return tokens, random_state()


class DecoderTests(unittest.TestCase):
    def assert_cache(self, cache, tokens):
        self.assertEqual({entry.offset for entry in cache}, {len(tokens)})
        for entry in cache:
            count = min(len(tokens), entry.max_size) if isinstance(entry, RotatingKVCache) else len(tokens)
            keys, values = ordered_kv(entry)
            actual = np.array(keys[0, 0, -count:, 0].astype(mx.float32)).astype(int).tolist()
            self.assertEqual(actual, tokens[-count:])
            np.testing.assert_array_equal(np.array(values[..., -count:, :].astype(mx.float32)),
                                          -np.array(keys[..., -count:, :].astype(mx.float32)))

    def run_case(self, prompt, count, mode, sampler, seed=41, eos=(), processors=None, cache=None, model=None):
        model = model or MarkovTarget(mode)
        cache = cache if cache is not None else model.make_cache()
        mx.random.seed(seed)
        iterator = generate_step(mx.array(prompt), model, max_tokens=count, sampler=sampler, prompt_cache=cache,
                                 logits_processors=processors, prefill_step_size=5, eos_token_ids=eos)
        try:
            output = list(iterator)
        finally:
            iterator.close()
        return model, cache, output, random_state()

    def test_greedy_identity_for_all_accept_reject_and_partial_blocks(self):
        for mode in ['match', 'reject', 'partial']:
            for count in [1, 2, 3, 15, 16, 17, 33, 49]:
                with self.subTest(mode=mode, count=count):
                    prompt = [1, 2, 3, 4, 5, 6, 7]
                    expected, _ = oracle(prompt, count, greedy, 41)
                    model, cache, output, _ = self.run_case(prompt, count, mode, greedy)
                    tokens = [row[0] for row in output]
                    self.assertEqual(tokens, expected)
                    self.assert_cache(cache, prompt + tokens[:-1])
                    if count > 17:
                        if mode == 'reject':
                            self.assertEqual(model.counters['draft_accepted_tokens_total'], 0)
                        else:
                            self.assertGreater(model.counters['draft_accepted_tokens_total'], 0)

    def test_seeded_sampling_tokens_and_final_rng_equal_serial_oracle(self):
        for mode in ['match', 'reject', 'partial']:
            for seed in [0, 41, 915]:
                with self.subTest(mode=mode, seed=seed):
                    prompt = [2, 5, 7, 1, 8, 6, 3, 4, 2]
                    expected, expected_rng = oracle(prompt, 47, categorical, seed)
                    model, cache, output, actual_rng = self.run_case(prompt, 47, mode, categorical, seed)
                    tokens = [row[0] for row in output]
                    self.assertEqual(tokens, expected)
                    self.assert_cache(cache, prompt + tokens[:-1])
                    for actual, reference in zip(actual_rng, expected_rng):
                        np.testing.assert_array_equal(actual, reference)
                    if mode == 'match':
                        self.assertEqual(model.counters['draft_accepted_tokens_total'],
                                         model.counters['draft_proposed_tokens_total'])

    def test_adaptive_width_and_allocator_policy_preserve_target_tokens_and_rng(self):
        for mode in ['match', 'reject', 'partial']:
            for sampler in [greedy, categorical]:
                for clear_interval in [0, 4096, 8192]:
                    with self.subTest(mode=mode, sampler=sampler.__name__, clear_interval=clear_interval):
                        prompt = [i % VOCAB for i in range(37)]
                        model = MarkovTarget(mode)
                        model.adaptive_blocks, model.prefill_clear_interval = True, clear_interval
                        expected, expected_rng = oracle(prompt, 81, sampler, 92)
                        _, cache, output, actual_rng = self.run_case(prompt, 81, mode, sampler, 92, model=model)
                        tokens = [row[0] for row in output]
                        self.assertEqual(tokens, expected)
                        self.assert_cache(cache, prompt + tokens[:-1])
                        for actual, reference in zip(actual_rng, expected_rng):
                            np.testing.assert_array_equal(actual, reference)

    def test_seeded_warm_prefix_matches_fresh_oracle_across_both_windows(self):
        prompt = [i % VOCAB for i in range(113)]
        for mode in ['match', 'partial', 'reject']:
            with self.subTest(mode=mode):
                model = MarkovTarget(mode)
                cache = model.make_cache()
                mx.eval(model(mx.array([prompt[:-1]]), cache))
                expected, expected_rng = oracle(prompt, 37, categorical, 94)
                _, cache, output, actual_rng = self.run_case(prompt[-1:], 37, mode, categorical, 94,
                                                            cache=cache, model=model)
                tokens = [row[0] for row in output]
                self.assertEqual(tokens, expected)
                self.assert_cache(cache, prompt + tokens[:-1])
                for actual, reference in zip(actual_rng, expected_rng):
                    np.testing.assert_array_equal(actual, reference)

    def test_history_processor_sees_the_same_accepted_prefix(self):
        def processor(history, logits):
            penalized = int(history[-3].item()) if len(history) >= 3 else 0
            return logits - mx.where(mx.arange(VOCAB)[None] == penalized, 3.0, 0.0)
        prompt = [2, 5, 7, 1, 8, 6, 3, 4, 2]
        expected, expected_rng = oracle(prompt, 37, categorical, 44, processor=processor)
        _, cache, output, actual_rng = self.run_case(prompt, 37, 'partial', categorical, 44, processors=[processor])
        tokens = [row[0] for row in output]
        self.assertEqual(tokens, expected)
        self.assert_cache(cache, prompt + tokens[:-1])
        for actual, reference in zip(actual_rng, expected_rng):
            np.testing.assert_array_equal(actual, reference)

    def test_eos_at_anchor_and_in_partial_and_full_verification_blocks(self):
        prompt = [1, 2, 3]
        for eos_token in [4, 5, 7, 10, 0]:
            for mode in ['match', 'partial', 'reject']:
                with self.subTest(eos=eos_token, mode=mode):
                    expected, _ = oracle(prompt, 32, greedy, 41, eos=[eos_token])
                    _, cache, output, _ = self.run_case(prompt, 32, mode, greedy, eos=[eos_token])
                    tokens = [row[0] for row in output]
                    self.assertEqual(tokens, expected)
                    self.assertEqual(tokens[-1], eos_token)
                    self.assert_cache(cache, prompt + tokens[:-1])

    def test_close_mid_block_keeps_only_emitted_verified_prefix_and_can_resume(self):
        for yielded in [1, 2, 3, 7, 16, 17, 18]:
            with self.subTest(yielded=yielded):
                model = MarkovTarget()
                cache = model.make_cache()
                prompt = [i % VOCAB for i in range(37)]
                iterator = generate_step(mx.array(prompt), model, max_tokens=60, sampler=greedy, prompt_cache=cache)
                tokens = [next(iterator)[0] for _ in range(yielded)]
                iterator.close()
                self.assert_cache(cache, prompt + tokens[:-1])
                _, _, continued, _ = self.run_case(tokens[-1:], 23, 'match', greedy, cache=cache, model=model)
                expected, _ = oracle(prompt, yielded + 23, greedy, 41)
                self.assertEqual(tokens + [row[0] for row in continued], expected)
                self.assert_cache(cache, prompt + expected[:-1])

    def test_processor_exception_during_verification_rewinds_to_emitted_prefix(self):
        model = MarkovTarget()
        cache = model.make_cache()
        prompt = [1, 2, 3, 4, 5, 6, 7]
        def processor(history, logits):
            if model.last_width > 1 and len(history) == len(prompt) + 3:
                raise RuntimeError('Injected processor failure')
            return logits
        iterator = generate_step(mx.array(prompt), model, max_tokens=40, sampler=greedy, prompt_cache=cache,
                                 logits_processors=[processor])
        tokens = []
        with self.assertRaisesRegex(RuntimeError, 'Injected'):
            while True:
                tokens.append(next(iterator)[0])
        iterator.close()
        self.assertEqual(len(tokens), 3)
        self.assert_cache(cache, prompt + tokens[:-1])

    def test_proposal_exception_restores_rng_and_leaves_cache_untouched(self):
        model = MarkovTarget()
        cache = model.make_cache()
        prompt = [1, 2, 3]
        calls = 0
        def sampler(logprobs):
            nonlocal calls
            calls += 1
            value = categorical(logprobs)
            mx.eval(value)
            if calls == 3:
                raise RuntimeError('Injected proposal failure')
            return value
        mx.random.seed(77)
        iterator = generate_step(mx.array(prompt), model, max_tokens=40, sampler=sampler, prompt_cache=cache)
        anchor = next(iterator)[0]
        expected_rng = random_state()
        with self.assertRaisesRegex(RuntimeError, 'proposal'):
            next(iterator)
        iterator.close()
        self.assert_cache(cache, prompt)
        for actual, reference in zip(random_state(), expected_rng):
            np.testing.assert_array_equal(actual, reference)
        self.assertTrue(0 <= anchor < VOCAB)

    def test_prefill_cancellation_stops_after_bounded_chunk(self):
        model = MarkovTarget()
        cache = model.make_cache()
        def progress(done, total):
            if 0 < done < total:
                raise InterruptedError('Injected cancellation')
        iterator = generate_step(mx.array([i % VOCAB for i in range(30)]), model, max_tokens=40,
                                 prompt_cache=cache, prefill_step_size=5, prompt_progress_callback=progress)
        with self.assertRaises(InterruptedError):
            next(iterator)
        iterator.close()
        self.assertEqual({entry.offset for entry in cache}, {5})


if __name__ == '__main__':
    unittest.main(verbosity=2)
