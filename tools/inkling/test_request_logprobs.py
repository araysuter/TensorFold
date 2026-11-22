import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class Logprobs:
    def __init__(self):
        self.scalar_reads = 0
    def __getitem__(self, token):
        assert token == 7
        self.scalar_reads += 1
        return SimpleNamespace(item=lambda: -0.25)


class LazyLogprobTests(unittest.TestCase):
    def run_one(self, requested=False, top_n=-1, missing_field=False):
        mx = ModuleType('mlx.core')
        mx.random = SimpleNamespace(seed=lambda value: None,
            state=[SimpleNamespace(tolist=lambda: [1, 2])])
        api = SimpleNamespace(
            Response=lambda text, token, probability, finish, top: SimpleNamespace(
                text=text, token=token, logprob=probability, finish_reason=finish, top_tokens=top),
            _apply_stop_words=lambda reader, text, finish_reason: (text, finish_reason),
            _format_top_logprobs=lambda values, n, tokenizer: ('top', n, id(values)))
        mlx = ModuleType('mlx'); mlx.core = mx
        lm = ModuleType('mlx_lm'); lm.server = api
        generate = ModuleType('mlx_lm.generate')
        generate.generate_step = lambda *a, **k: None
        generate.generation_stream = None
        cache = ModuleType('mlx_lm.models.cache'); cache.make_prompt_cache = lambda m: []
        with patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': mx, 'mlx_lm': lm,
                'mlx_lm.generate': generate, 'mlx_lm.models.cache': cache}):
            spec = importlib.util.spec_from_file_location('candidate',
                Path(__file__).with_name('request_state.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            state = object.__new__(module.RequestState)
            values = Logprobs()
            state.prompt = [1]
            state.ctx = SimpleNamespace(_should_stop=False)
            state.position = 0
            state.iterator = iter([(7, values)])
            state.rng = None
            state.seed = 1234
            state.generated = 0
            state.args = SimpleNamespace(max_tokens=1, top_logprobs=top_n)
            if not missing_field:
                state.args.logprobs = requested
            state.tokenizer = SimpleNamespace(eos_token_ids=set())
            state.detokenizer = SimpleNamespace(add_token=lambda token: None,
                finalize=lambda: None, last_segment='answer')
            state.stop_reader = None
            result = state.advance()
            return values, result

    def test_unrequested_probability_does_not_read_gpu_scalar(self):
        for missing in (False, True):
            with self.subTest(missing_field=missing):
                values, result = self.run_one(missing_field=missing)
                self.assertEqual(values.scalar_reads, 0)
                self.assertEqual(result.token, 7)
                self.assertEqual(result.finish_reason, 'length')

    def test_requested_probability_is_preserved(self):
        values, result = self.run_one(requested=True)
        self.assertEqual(values.scalar_reads, 1)
        self.assertEqual(result.logprob, -0.25)

    def test_top_probabilities_keep_the_original_formatter(self):
        values, result = self.run_one(top_n=3)
        self.assertEqual(values.scalar_reads, 0)
        self.assertEqual(result.top_tokens, ('top', 3, id(values)))


if __name__ == '__main__':
    unittest.main()
