#!/usr/bin/env python3
"""Check real MLX KV restoration and authenticated HTTP behavior with a small Ripple checkpoint."""
import argparse
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
try:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from modeling_ricursive_mlx import Model, ModelArgs
    from worker_state import ResidentState
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: Ripple worker tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


SOURCE = None


def small_model(vocab=128):
    args = ModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=5, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, vocab_size=vocab, sliding_window=8, embed_scale=8.0)
    mx.random.seed(17)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, args


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.model, self.args = small_model()
        self.state = ResidentState(self.root, 'test-build', self.args, 512)
        self.path = self.root / ('a' * 64 + '.tfs')

    def tearDown(self):
        self.state.clear()
        self.directory.cleanup()

    def fill(self):
        ids = list(range(7, 47))
        cache = self.model.make_cache()
        for start in range(0, len(ids), 5):
            mx.eval(self.model(mx.array([ids[start:start + 5]]), cache=cache))
        self.state.commit(ids, cache)
        return ids

    def test_rotated_cache_disk_restore_is_bit_identical(self):
        ids = self.fill()
        receipt = self.state.save(self.path)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(receipt['tokens'], len(ids))
        expected = np.array(self.model(mx.array([[47, 48, 49]]), cache=self.state.cache).astype(mx.float32))
        self.state.clear()
        self.assertEqual(self.state.load(self.path), receipt)
        actual = np.array(self.model(mx.array([[47, 48, 49]]), cache=self.state.cache).astype(mx.float32))
        np.testing.assert_array_equal(actual, expected)

    def test_prefix_ownership_and_unprocessed_last_token(self):
        ids = self.fill()
        self.state.commit(ids + [47], self.state.cache)
        self.assertEqual(self.state.tokens, ids)
        self.assertEqual(self.state.resumable(ids + [47, 48]), len(ids))
        self.assertEqual(self.state.resumable(ids), 0)
        self.assertEqual(self.state.resumable([99] + ids), 0)
        cache, count = self.state.take(ids + [47])
        self.assertEqual(count, len(ids))
        self.assertIsNotNone(cache)
        self.assertIsNone(self.state.cache)

    def test_snapshot_after_long_prefill_before_any_decode_step(self):
        ids = [7 + i % 100 for i in range(300)]
        cache = self.model.make_cache()
        mx.eval(self.model(mx.array([ids]), cache=cache))
        self.state.commit(ids, cache)
        receipt = self.state.save(self.path)
        expected = np.array(self.model(mx.array([[47]]), cache=cache).astype(mx.float32))
        self.state.clear()
        self.assertEqual(self.state.load(self.path), receipt)
        actual = np.array(self.model(mx.array([[47]]), cache=self.state.cache).astype(mx.float32))
        np.testing.assert_array_equal(actual, expected)

    def test_wrong_build_corruption_and_unsafe_paths_refused(self):
        self.fill()
        self.state.save(self.path)
        self.state.clear()
        other = ResidentState(self.root, 'wrong-build', self.args, 512)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            other.load(self.path)
        with self.assertRaises(ValueError):
            self.state.save(self.root.parent / self.path.name)
        link = self.root / ('b' * 64 + '.tfs')
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            self.state.load(link)
        self.path.write_bytes(b'broken')
        with self.assertRaises(ValueError):
            self.state.load(self.path)
        self.assertIsNone(self.state.cache)


class HTTPTests(unittest.TestCase):
    DRAFT = False

    @classmethod
    def fixture_model(cls):
        return small_model(100352)

    @classmethod
    def extra_arguments(cls, root):
        return []

    @classmethod
    def setUpClass(cls):
        if SOURCE is None:
            raise unittest.SkipTest('--source supplies the pinned tokenizer for the HTTP fixture')
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        model_path = root / 'model'
        model_path.mkdir()
        model, args = cls.fixture_model()
        tensors = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(model_path / 'model.safetensors'), tensors)
        config = vars(args) | {'model_file': 'modeling_ricursive_mlx.py'}
        (model_path / 'config.json').write_text(json.dumps(config))
        (model_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':
                                              {k: 'model.safetensors' for k in tensors}}))
        (model_path / 'conversion-receipt.json').write_text('{"fixture":true}')
        shutil.copyfile(Path(__file__).with_name('modeling_ricursive_mlx.py'), model_path / 'modeling_ricursive_mlx.py')
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']:
            if (SOURCE / name).exists():
                shutil.copyfile(SOURCE / name, model_path / name)
        cls.snapshots = root / 'contexts'
        cls.snapshots.mkdir(mode=0o700)
        cls.key = secrets.token_urlsafe(48)
        keyfile = root / 'backend-key'
        keyfile.write_text('gateway:' + cls.key + '\n')
        keyfile.chmod(0o600)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.log = (root / 'worker.log').open('wb')
        command = [sys.executable, str(Path(__file__).with_name('worker.py')), '--model', str(model_path),
                   '--name', 'ripple-test', '--context', '512', '--port', str(cls.port),
                   '--snapshot-dir', str(cls.snapshots), '--api-key-file', str(keyfile)] + cls.extra_arguments(root)
        cls.process = subprocess.Popen(command, stdout=cls.log, stderr=cls.log)
        for _ in range(300):
            try:
                if cls.request('GET', '/health')[0] == 200:
                    return
            except OSError:
                pass
            if cls.process.poll() is not None:
                cls.log.flush()
                raise RuntimeError('Tiny worker exited; inspect ' + str(root / 'worker.log'))
            time.sleep(0.1)
        cls.process.kill()
        raise RuntimeError('Tiny worker startup timed out')

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.log.close()
        cls.directory.cleanup()

    @classmethod
    def request(cls, method, path, body=None, key=True):
        connection = http.client.HTTPConnection('127.0.0.1', cls.port, timeout=30)
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + cls.key
        connection.request(method, path, None if body is None else json.dumps(body).encode(), headers)
        response = connection.getresponse()
        result = response.status, response.read().decode()
        connection.close()
        return result

    def chat(self, **options):
        return {'model': 'ripple-test', 'messages': [{'role': 'user', 'content': 'Say hello.'}],
                'max_tokens': 8, 'temperature': 0, **options}

    def test_private_auth_health_and_no_model_path_disclosure(self):
        self.assertEqual(self.request('GET', '/health', key=False)[0], 401)
        status, raw = self.request('GET', '/v1/models')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['data'][0]['id'], 'ripple-test')
        self.assertNotIn(str(Path.home()), raw)
        self.assertEqual(json.loads(self.request('GET', '/health')[1])['dflash2'], self.DRAFT)

    def test_unsupported_features_and_total_context_budget_refused(self):
        for options in [{'tools': []}, {'draft_model': 'external/model'}, {'max_tokens': 512},
                        {'max_tokens': True}, {'model': '/arbitrary/path'}, {'stream_options': []},
                        {'temperature': float('nan')}, {'top_p': float('inf')}, {'seed': -1},
                        {'messages': [{'role': 'user', 'content': [{'type': 'image_url'}]}]},
                        {'messages': [{'role': 'user', 'content': 'hello ' * 600}]}]:
            with self.subTest(options=options.keys()):
                self.assertEqual(self.request('POST', '/v1/chat/completions', self.chat(**options))[0], 400)

    def test_streaming_matches_nonstream_and_snapshot_barrier(self):
        self.request('POST', '/v1/internal/session/clear', {})
        status, raw = self.request('POST', '/v1/chat/completions', self.chat())
        self.assertEqual(status, 200, raw)
        expected = json.loads(raw)['choices'][0]['message']
        path = self.snapshots / ('c' * 64 + '.tfs')
        status, raw = self.request('POST', '/v1/internal/session/save', {'path': str(path)})
        self.assertEqual(status, 200, raw)
        saved = json.loads(raw)
        self.assertGreater(saved['tokens'], 0)
        self.request('POST', '/v1/internal/session/clear', {})
        self.assertEqual(json.loads(self.request('POST', '/v1/internal/session/load', {'path': str(path)})[1]), saved)
        self.request('POST', '/v1/internal/session/clear', {})
        status, raw = self.request('POST', '/v1/chat/completions', self.chat(stream=True, stream_options={}))
        self.assertEqual(status, 200, raw)
        content, reasoning = '', ''
        for line in raw.splitlines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                for choice in json.loads(line[6:])['choices']:
                    content += choice.get('delta', {}).get('content', '')
                    reasoning += choice.get('delta', {}).get('reasoning', '')
        self.assertIn('data: [DONE]', raw)
        self.assertEqual(content, expected.get('content') or '')
        self.assertEqual(reasoning, expected.get('reasoning') or '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[sys.argv[0], *remaining])
