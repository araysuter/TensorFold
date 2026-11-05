#!/usr/bin/env python3
"""Check public benchmark comparisons preserve failures, separate settings and exclude private data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_dflash import PROMPTS, SOURCE_REVISIONS
from compare_dflash import build, markdown, token_hash


def receipt():
    inputs, outputs = [11, 12, 13], [21, 22, 23]
    prompt_sha = token_hash(inputs)
    base = dict(prompt='prose', prompt_sha256=prompt_sha, prompt_tokens=len(inputs), chunk=512,
                temperature=0.0, seed=17, state='fresh', rep=0, max_output_tokens=16, cached_prefix_tokens=0,
                generation_tokens=len(outputs), output_tokens=outputs, output_sha256=token_hash(outputs),
                finish_reason='stop', elapsed_seconds=2.0, ttft_seconds=0.2, steady_decode_tps=10.0,
                request_output_tps=1.5, library_prompt_tps=100, library_prompt_seconds=0.03,
                acceptance=None, peak_mlx_bytes=40 * 1024**3, sampled_peak_physical_footprint_bytes=42 * 1024**3,
                offsets_after=[5] * 80, warmup=False)
    plain = dict(base, case_id=0, mode='plain', block_width=None, adaptive=False, prefill_clear_interval=0)
    draft = dict(base, case_id=1, mode='draft', block_width=4, adaptive=False, prefill_clear_interval=0,
                 elapsed_seconds=1.0, steady_decode_tps=20.0, acceptance=0.5)
    return dict(schema='ripple-dflash-benchmark-v1', public_prompts=PROMPTS, phase='decode', status='complete',
                checkpoint_provenance={'target': {'source_revision': SOURCE_REVISIONS['target'], 'weight_bytes': 39},
                                      'drafter': {'source_revision': SOURCE_REVISIONS['drafter'], 'weight_bytes': 3}},
                config_sha256={'target': {'config.json': 'a' * 64, 'conversion-receipt.json': 'b' * 64},
                               'drafter': {'config.json': 'c' * 64, 'conversion-receipt.json': 'd' * 64}},
                source_sha256={'benchmark_dflash.py': 'e' * 64}, packages={'mlx': '0.32.4', 'mlx-lm': '0.32.2'},
                limits={'context': 131072, 'cache_pool_bytes': 128 * 1024**2}, hardware_model='Mac16,11',
                physical_ram_bytes=64 * 1024**3, prompt_arrays={prompt_sha: inputs}, cases=[plain, draft],
                comparisons=[], chunk_comparisons=[], policy_comparisons=[])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, value, name='private-host-private-user.json'):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def test_exact_pair_has_distinct_request_decode_prompt_and_memory_metrics(self):
        result = build([(self.write(receipt()), 'fixed4')])
        source = result['sources'][0]
        self.assertEqual(source['cache_pool_bytes'], 128 * 1024**2)
        comparison = source['comparisons'][0]
        self.assertTrue(comparison['token_identity'])
        self.assertEqual(comparison['steady_decode_speedup'], 2)
        self.assertEqual(comparison['request_speedup'], 2)
        self.assertEqual(comparison['prompt_speedup'], 1)
        self.assertEqual(source['cases'][1]['memory_gib']['sampled_peak_physical_footprint'], 42)
        self.assertIn('10.00 → 20.00', markdown(result))

    def test_failed_receipt_and_first_mismatch_are_preserved(self):
        raw = receipt()
        raw.update(status='failed', failure_type='ValueError')
        raw['cases'][1]['output_tokens'] = [21, 24]
        raw['cases'][1]['generation_tokens'] = 2
        raw['cases'][1]['output_sha256'] = token_hash([21, 24])
        raw['comparisons'] = [{'token_identity': False, 'plain_case': 0, 'draft_case': 1, 'first_mismatch': 1}]
        source = build([(self.write(raw), 'failed-width16')])['sources'][0]
        self.assertEqual(source['status'], 'failed')
        self.assertFalse(source['comparisons'][0]['token_identity'])
        self.assertEqual(source['comparisons'][0]['first_mismatch'], 1)
        self.assertEqual(len(source['recorded_failed_checks']), 1)

    def test_priming_warmups_excluded_and_sampling_cells_not_pooled(self):
        raw = receipt()
        raw['cases'] += [dict(raw['cases'][0], case_id=2, priming=True, warmup=True, rep=-2),
                         dict(raw['cases'][1], case_id=3, warmup=True, rep=-1),
                         dict(raw['cases'][0], case_id=4, temperature=0.7),
                         dict(raw['cases'][1], case_id=5, temperature=0.7, adaptive=True)]
        source = build([(self.write(raw), 'mixed')])['sources'][0]
        self.assertEqual(source['excluded_priming_cases'], 1)
        self.assertEqual(source['excluded_warmup_cases'], 1)
        self.assertEqual(len(source['cases']), 4)
        self.assertEqual(len(source['comparisons']), 2)
        self.assertIn('primer excluded', source['timing_scope'])

    def test_draft_only_pp_is_unpaired(self):
        raw = receipt()
        raw['phase'] = 'prefill'
        raw['cases'] = raw['cases'][1:]
        comparison = build([(self.write(raw), 'pp')])['sources'][0]['comparisons'][0]
        self.assertIsNone(comparison['plain_case'])
        self.assertIsNone(comparison['token_identity'])
        self.assertIsNone(comparison['steady_decode_speedup'])

    def test_private_paths_text_and_unknown_metadata_do_not_escape(self):
        raw = receipt()
        raw.update(model='/Users/private/person/models', host='private-host', api_key='secret-private-key')
        raw['cases'][1].update(text='private completion text', snapshot={'path': '/Users/private/cache'})
        result = build([(self.write(raw), 'public-label')])
        exported = json.dumps(result) + markdown(result)
        for forbidden in ['private-host', 'private-user', '/Users/', str(self.root), 'secret-private-key',
                          'private completion text', '"output_tokens":', '"prompt_arrays":']:
            self.assertNotIn(forbidden, exported)

    def test_incompatible_targets_contexts_prompts_and_hashes_refused(self):
        first = self.write(receipt(), 'first.json')
        mutations = [lambda r: r['config_sha256']['target'].update({'config.json': 'f' * 64}),
                     lambda r: r['limits'].update(context=65536),
                     lambda r: r['cases'][1].update(output_sha256='f' * 64)]
        for index, mutate in enumerate(mutations):
            raw = receipt()
            mutate(raw)
            with self.subTest(index=index), self.assertRaises(ValueError):
                build([(first, 'first'), (self.write(raw, 'second.json'), 'second')])
        raw = receipt()
        old = raw['cases'][0]['prompt_sha256']
        new = token_hash([11, 12, 14])
        raw['prompt_arrays'] = {new: [11, 12, 14]}
        for case in raw['cases']:
            case['prompt_sha256'] = new
        self.assertNotEqual(old, new)
        with self.assertRaisesRegex(ValueError, 'Public prompt hash'):
            build([(first, 'first'), (self.write(raw, 'second.json'), 'second')])

    def test_drafter_variants_stay_separate_and_identity_is_not_quality(self):
        other = copy.deepcopy(receipt())
        other['config_sha256']['drafter']['config.json'] = 'f' * 64
        result = build([(self.write(receipt(), 'a.json'), 'mixed4'), (self.write(other, 'b.json'), 'mixed8')])
        self.assertEqual(len(result['sources']), 2)
        self.assertIn('different drafter conversion identities', result['notes'][-1])


if __name__ == '__main__':
    unittest.main()
