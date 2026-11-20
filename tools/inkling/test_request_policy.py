import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('request_policy', Path(__file__).with_name('request_policy.py'))
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class RequestBoundaryTests(unittest.TestCase):
    def valid(self, **kwargs):
        return dict(model='inkling-2-mini', prompt=[1, 2, 3], max_tokens=32, **kwargs)

    def test_raw_token_prompt(self):
        self.assertEqual(policy.validate_request(self.valid(), 'inkling-2-mini', 131072), 32)

    def test_client_cannot_select_another_model_or_adapter(self):
        for change in ({'model': '/tmp/other-model'}, {'adapters': 'remote/repo'}, {'draft_model': 'remote/repo'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                policy.validate_request(self.valid() | change, 'inkling-2-mini', 131072)

    def test_no_remote_or_local_image_fetch(self):
        for url in ('http://127.0.0.1/private', 'file:///etc/passwd', '/tmp/image.png', 'https://example.com/image.png'):
            body = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}]}]}
            with self.subTest(url=url), self.assertRaises(ValueError):
                policy.validate_request(body, 'inkling-2-mini', 131072)

    def test_bad_images_and_non_image_parts(self):
        for part in ({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,???'}},
                     {'type': 'input_audio'}, {'type': 'video_url'}):
            with self.subTest(part=part), self.assertRaises(ValueError):
                policy.validate_request({'messages': [{'content': [part]}]}, 'inkling-2-mini', 131072)

    def test_output_budget_must_be_bounded_integer(self):
        for maximum in (True, 0, -1, 131072, 10.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                policy.validate_request(self.valid() | {'max_tokens': maximum}, 'inkling-2-mini', 131072)

    def test_raw_prompt_plus_output_must_fit_context(self):
        with self.assertRaises(ValueError):
            policy.validate_request(self.valid() | {'prompt': [1] * 100, 'max_tokens': 29}, 'inkling-2-mini', 128)
        self.assertEqual(policy.validate_request(self.valid() | {'prompt': [1] * 100, 'max_tokens': 28}, 'inkling-2-mini', 128), 28)

    def test_token_arrays_cannot_be_nested_or_negative(self):
        for prompt in ([True], [-1], [[1, 2]], []):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                policy.validate_request(self.valid() | {'prompt': prompt}, 'inkling-2-mini', 131072)

    def test_eos_override_only_available_in_isolated_benchmark(self):
        body = self.valid(ignore_eos=True)
        with self.assertRaises(ValueError):
            policy.validate_request(body, 'inkling-2-mini', 131072)
        self.assertEqual(policy.validate_request(body, 'inkling-2-mini', 131072, benchmark=True), 32)


if __name__ == '__main__':
    unittest.main()
