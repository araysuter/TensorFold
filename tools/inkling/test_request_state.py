import unittest
from types import SimpleNamespace

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import server as api
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import KVCache
    from request_state import RequestState
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


if AVAILABLE:
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def make_cache(self):
            return [KVCache()]
        def __call__(self, inputs, cache, input_embeddings=None):
            self.calls += 1
            x = inputs[:, None, :, None].astype(mx.float32)
            _, values = cache[0].update_and_fetch(x, x)
            previous = mx.concatenate([mx.zeros_like(values[:, :, :1]), values[:, :, :-1]], axis=2)
            previous = previous[:, 0, -inputs.shape[1]:, 0].astype(mx.int32)
            desired = (inputs + previous + 1) % 32
            return -mx.abs(mx.arange(32) - desired[..., None]).astype(mx.float32) * 0.3

    class Detokenizer:
        last_segment = ''
        def add_token(self, token):
            self.last_segment = str(token) + ' '
        def finalize(self):
            self.last_segment = ''

    class Tokenizer:
        eos_token_ids = set()
        @property
        def detokenizer(self):
            return Detokenizer()
        def encode(self, text):
            return [1]

    def arguments(seed):
        return SimpleNamespace(seed=seed, max_tokens=32, top_logprobs=-1, stop_words=[],
            sampling=SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                                     xtc_probability=0.0, xtc_threshold=0.1),
            logits=SimpleNamespace(logit_bias=None, repetition_penalty=0.0, repetition_context_size=20,
                                   presence_penalty=0.0, presence_context_size=20,
                                   frequency_penalty=0.0, frequency_context_size=20))


@unittest.skipUnless(AVAILABLE, 'Requires the pinned MLX environment')
class StateTests(unittest.TestCase):
    def test_interleaved_seeded_sampling_matches_each_solo_core(self):
        model, tokenizer = ToyModel(), Tokenizer()
        prompts = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
        expected = []
        for i, prompt in enumerate(prompts):
            args = arguments(1234 + i)
            mx.random.seed(args.seed)
            expected.append([int(t) for t, _ in generate_step(mx.array(prompt), model,
                max_tokens=32, sampler=api._make_sampler(args, tokenizer), prefill_step_size=2)])
        states = [RequestState(model, tokenizer, prompt, None, arguments(1234 + i),
                               SimpleNamespace(_should_stop=False), 2) for i, prompt in enumerate(prompts)]
        self.assertIsNot(states[0].cache, states[1].cache)
        actual = [[], []]
        while not all(s.done for s in states):
            for i, state in enumerate(states):
                if state.done:
                    continue
                response = state.advance()
                if isinstance(response, api.Response):
                    actual[i].append(response.token)
        self.assertEqual(actual, expected)
        self.assertNotEqual(expected[0], expected[1])
        for state in states:
            state.close()
            self.assertEqual(state.cache, [])

    def test_cancelled_request_does_no_more_model_work(self):
        model = ToyModel()
        state = RequestState(model, Tokenizer(), [1, 2, 3], None, arguments(1),
                             SimpleNamespace(_should_stop=True), 2)
        self.assertIsNone(state.advance())
        self.assertEqual(model.calls, 0)
        self.assertTrue(state.done)
        state.close()


if __name__ == '__main__':
    unittest.main()
