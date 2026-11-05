#!/usr/bin/env python3
"""Exercise authenticated DFlash worker requests, SSD state and interrupted HTTP streams with tiny real models."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import socket
import sys
import unittest
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import load_prompt_cache
    from safetensors import safe_open
    from transformers import AutoTokenizer
    import test_worker as base
    from dflash_model import DraftArgs, DraftModel
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('mlx', 'mlx_lm', 'numpy', 'safetensors', 'transformers'):
        raise
    if __name__ == '__main__':
        print('SKIP: DFlash worker tests require the documented Apple Silicon MLX test environment')
        sys.exit(0)
    raise unittest.SkipTest('Requires the Ripple MLX test environment') from exc


BUILD_RECEIPT = None


class HTTPDFlashTests(base.HTTPTests):
    DRAFT = True

    @classmethod
    def setUpClass(cls):
        if BUILD_RECEIPT is None:
            raise unittest.SkipTest('--mlx-build-receipt supplies the isolated dispatcher build')
        super().setUpClass()

    @classmethod
    def fixture_model(cls):
        model, args = base.small_model(100352)
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
        mx.eval(model.parameters())
        return model, args

    @classmethod
    def extra_arguments(cls, root):
        draft_path = root / 'drafter'
        draft_path.mkdir()
        args = DraftArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=100352,
                         sliding_window=16, layer_types=['sliding_attention'] * 4 + ['full_attention'],
                         no_rope_layers=[4], target_layer_ids=[0, 1, 2, 3, 4], block_size=16,
                         conv_kernel_size=4, conv_group_size=16, selector_rank=16, selector_top_k=32,
                         mask_token_id=100278)
        mx.random.seed(81)
        model = DraftModel(args)
        model.set_dtype(mx.bfloat16)
        quantization = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
        for name, value in tree_flatten(model.parameters()):
            if name.endswith('.weight') and value.ndim == 2:
                path = name[:-7]
                bits = 8 if path in ('fc', 'candidate_selector.hidden_projection') or 'kernel_projection' in path else 4
                quantization[path] = {'bits': bits, 'group_size': 64, 'mode': 'affine'}
        nn.quantize(model, group_size=64, bits=4, class_predicate=lambda path, module: quantization.get(path, False))
        mx.eval(model.parameters())
        weight_path = draft_path / 'model.safetensors'
        mx.save_safetensors(str(weight_path), dict(tree_flatten(model.parameters())))
        special = ['target_layer_ids', 'block_size', 'conv_kernel_size', 'conv_group_size',
                   'selector_rank', 'selector_top_k', 'mask_token_id']
        config = {key: value for key, value in vars(args).items() if key not in special}
        config.update(architectures=['DFlash2DraftModel'], is_causal=False, hidden_act='silu',
                      dflash_config={key: getattr(args, key) for key in special}, quantization=quantization)
        (draft_path / 'config.json').write_text(json.dumps(config))
        (draft_path / 'conversion-receipt.json').write_text(json.dumps({
            'fixture': True, 'sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest()}))
        tokenizer = AutoTokenizer.from_pretrained(base.SOURCE, trust_remote_code=False)
        cls.repeated_text = tokenizer.decode([0, 0, 0, 0], skip_special_tokens=False)
        cls.stop_text = tokenizer.decode([0, 0], skip_special_tokens=False)
        if not cls.stop_text or cls.stop_text * 2 != cls.repeated_text:
            raise ValueError('The pinned tokenizer must expose a repeatable token zero for the HTTP fixture')
        for word in [' the', ' test', ' zero']:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) == 1:
                repeated = tokenizer.decode(tokens * 2, skip_special_tokens=False)
                if tokenizer.encode(repeated, add_special_tokens=False) == tokens * 2:
                    cls.stop_text, cls.stop_token = repeated, tokens[0]
                    break
        else:
            raise ValueError('The stop fixture requires an unambiguous repeated token')
        return ['--dflash-drafter', str(draft_path), '--dflash-exact-verify',
                '--mlx-build-receipt', str(BUILD_RECEIPT)]

    def setUp(self):
        status, raw = self.request('POST', '/v1/internal/session/clear', {})
        self.assertEqual(status, 200, raw)

    def complete(self, **options):
        status, raw = self.request('POST', '/v1/chat/completions', self.chat(max_tokens=23, **options))
        self.assertEqual(status, 200, raw)
        return json.loads(raw)

    def clear(self):
        status, raw = self.request('POST', '/v1/internal/session/clear', {})
        self.assertEqual(status, 200, raw)

    def save(self, letter='d'):
        path = self.snapshots / (letter * 64 + '.tfs')
        status, raw = self.request('POST', '/v1/internal/session/save', {'path': str(path)})
        self.assertEqual(status, 200, raw)
        return path, json.loads(raw)

    @staticmethod
    def streamed(raw):
        content, reasoning, finish = '', '', None
        for line in raw.splitlines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                for choice in json.loads(line[6:]).get('choices', []):
                    delta = choice.get('delta', {})
                    content += delta.get('content') or ''
                    reasoning += delta.get('reasoning') or ''
                    finish = choice.get('finish_reason') or finish
        return {'content': content, 'reasoning': reasoning}, finish

    def test_chat_and_snapshot_auth_precede_state_mutation(self):
        self.complete(draft=True)
        _, before = self.save('a')
        blocked_path = self.snapshots / ('9' * 64 + '.tfs')
        for route, body in [('/v1/chat/completions', self.chat(draft=True)),
                            ('/v1/internal/session/clear', {}),
                            ('/v1/internal/session/save', {'path': str(blocked_path)}),
                            ('/v1/internal/session/load', {'path': str(blocked_path)})]:
            with self.subTest(route=route):
                status, raw = self.request('POST', route, body, key=False)
                self.assertEqual(status, 401)
                self.assertNotIn(str(self.snapshots), raw)
        self.assertFalse(blocked_path.exists())
        _, after = self.save('a')
        self.assertEqual(after, before)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        connection.request('GET', '/stats', headers={'Authorization': 'Bearer invalid-fixture-key'})
        response = connection.getresponse()
        self.assertEqual(response.status, 401)
        response.read()
        connection.close()

    def test_draft_false_boolean_validation_and_proposal_stats(self):
        for value in [0, 1, 'false', None, [], {}]:
            with self.subTest(value=value):
                status, _ = self.request('POST', '/v1/chat/completions', self.chat(draft=value))
                self.assertEqual(status, 400)
        before = json.loads(self.request('GET', '/stats')[1])
        plain = self.complete(draft=False)
        middle = json.loads(self.request('GET', '/stats')[1])
        self.assertEqual(middle.get('draft_proposed_tokens_total', 0), before.get('draft_proposed_tokens_total', 0))
        self.clear()
        drafted = self.complete(draft=True)
        self.save('b')
        after = json.loads(self.request('GET', '/stats')[1])
        self.assertEqual(plain['choices'][0]['message'], drafted['choices'][0]['message'])
        self.assertGreater(after['draft_proposed_tokens_total'], middle.get('draft_proposed_tokens_total', 0))
        self.assertGreater(after['draft_verify_calls_total'], middle.get('draft_verify_calls_total', 0))
        for key in ['draft_proposed_tokens_total', 'draft_accepted_tokens_total', 'draft_verify_calls_total']:
            self.assertIs(type(after[key]), int)
            self.assertGreaterEqual(after[key], 0)
        self.assertLessEqual(after['draft_accepted_tokens_total'], after['draft_proposed_tokens_total'])
        self.assertEqual(after['parallel'], 1)
        self.assertEqual(after['running'], 0)

    def test_seeded_plain_drafted_and_streamed_responses_match(self):
        for temperature in [0, 0.7]:
            for seed in [0, 73]:
                with self.subTest(temperature=temperature, seed=seed):
                    plain = self.complete(draft=False, temperature=temperature, seed=seed)
                    self.clear()
                    drafted = self.complete(draft=True, temperature=temperature, seed=seed)
                    self.assertEqual(plain['choices'][0]['message'], drafted['choices'][0]['message'])
                    self.assertEqual(plain['choices'][0]['finish_reason'], drafted['choices'][0]['finish_reason'])
                    self.clear()
                    status, raw = self.request('POST', '/v1/chat/completions',
                                               self.chat(draft=True, temperature=temperature, seed=seed,
                                                         max_tokens=23, stream=True, stream_options={}))
                    self.assertEqual(status, 200, raw)
                    actual, finish = self.streamed(raw)
                    message = drafted['choices'][0]['message']
                    self.assertEqual(actual['content'], message.get('content') or '')
                    self.assertEqual(actual['reasoning'], message.get('reasoning') or '')
                    self.assertEqual(finish, drafted['choices'][0]['finish_reason'])
                    self.assertIn('data: [DONE]', raw)
                    self.clear()

    def test_draft_snapshot_has_ten_caches_correct_windows_and_restores(self):
        expected = self.complete(draft=True, temperature=0.7, seed=19)
        path, receipt = self.save()
        self.assertGreater(receipt['tokens'], 0)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with safe_open(str(path), framework='numpy') as stream:
            metadata = stream.metadata()
        classes = {key: value for key, value in metadata.items() if key.startswith('1.')}
        self.assertEqual(len(classes), 10)
        kinds = ['RotatingKVCache'] * 4 + ['KVCache']
        self.assertEqual(classes, {f'1.{index}': kind for index, kind in enumerate(kinds * 2)})
        inspection = path.with_suffix('.safetensors')
        inspection.unlink(missing_ok=True)
        inspection.hardlink_to(path)
        try:
            cache = load_prompt_cache(str(inspection))
        finally:
            inspection.unlink()
        self.assertEqual({entry.offset for entry in cache}, {receipt['tokens']})
        self.assertEqual([entry.max_size for entry in cache if hasattr(entry, 'max_size')], [8] * 4 + [16] * 4)
        self.clear()
        status, raw = self.request('POST', '/v1/internal/session/load', {'path': str(path)})
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw), receipt)
        continuation = [{'role': 'user', 'content': 'Say hello.'}, expected['choices'][0]['message'],
                        {'role': 'user', 'content': 'Continue.'}]
        restored = self.complete(messages=continuation, draft=True, temperature=0.7, seed=29)
        self.clear()
        fresh = self.complete(messages=continuation, draft=True, temperature=0.7, seed=29)
        self.assertEqual(restored['choices'][0]['message'], fresh['choices'][0]['message'])

    def test_stop_mid_block_snapshot_and_followup_remain_valid(self):
        stopped = self.complete(draft=True, stop=self.stop_text, logit_bias={str(self.stop_token): 100})
        self.assertEqual(stopped['choices'][0]['finish_reason'], 'stop')
        path, receipt = self.save('e')
        self.assertGreater(receipt['tokens'], 0)
        self.clear()
        self.assertEqual(self.request('POST', '/v1/internal/session/load', {'path': str(path)})[0], 200)
        after = self.complete(draft=True, seed=44)
        self.clear()
        fresh = self.complete(draft=True, seed=44)
        self.assertEqual(after['choices'][0]['message'], fresh['choices'][0]['message'])

    def test_disconnect_mid_stream_then_snapshot_barrier_and_fresh_request(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        body = self.chat(draft=True, max_tokens=128, stream=True, stream_options={})
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key}
        connection.request('POST', '/v1/chat/completions', json.dumps(body), headers)
        transport = connection.sock
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        for _ in range(100):
            line = response.readline()
            if line.startswith(b'data: ') and b'choices' in line and b'content' in line:
                break
        else:
            self.fail('The streaming worker produced no content before disconnect')
        transport.shutdown(socket.SHUT_RDWR)
        response.close()
        connection.close()
        path, receipt = self.save('f')
        self.assertGreater(receipt['tokens'], 0)
        stats = json.loads(self.request('GET', '/stats')[1])
        self.assertEqual(stats['running'], 0)
        self.clear()
        status, raw = self.request('POST', '/v1/internal/session/load', {'path': str(path)})
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw), receipt)
        after = self.complete(draft=True, temperature=0.7, seed=61)
        self.clear()
        fresh = self.complete(draft=True, temperature=0.7, seed=61)
        self.assertEqual(after['choices'][0]['message'], fresh['choices'][0]['message'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mlx-build-receipt', type=Path)
    args, remaining = parser.parse_known_args()
    base.SOURCE = args.source
    BUILD_RECEIPT = args.mlx_build_receipt
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
