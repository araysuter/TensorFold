#!/usr/bin/env python3
"""Check benchmark admission, exact public prompts and token-identity reporting without a model or GPU."""
import unittest
from benchmark_dflash import chunk_comparisons, compare_cases, parser, plan, policy_comparisons, public_tokens, token_hash


class Tokenizer:
    def encode(self, value, add_special_tokens=False):
        return [ord(c) for c in value]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return self.encode('user:' + messages[0]['content'] + '\nassistant:')


class BenchmarkTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(['--model', '/model', '--drafter', '/draft', '--output', '/receipt', *extra])

    def test_default_matrix_and_alternating_order(self):
        cases = plan(self.args('--reps', '2'))
        self.assertEqual(len(cases), 16)
        self.assertEqual([c['mode'] for c in cases[:2]], ['plain', 'draft'])
        self.assertEqual([c['mode'] for c in cases[8:10]], ['draft', 'plain'])
        self.assertEqual({c['temperature'] for c in cases}, {0.0, 0.7})

    def test_extended_matrix_needs_explicit_case_budget(self):
        options = ['--phase', 'prefill', '--prefill-sizes', '2048,8192,32768,65536,131072',
                   '--prefill-chunks', '128,256,512,1024', '--max-tokens', '64']
        with self.assertRaisesRegex(ValueError, '40 cases'):
            plan(self.args(*options))
        self.assertEqual(len(plan(self.args(*options, '--max-cases', '40'))), 40)

    def test_invalid_or_unbounded_settings_refused(self):
        for options in [('--max-tokens', '257'), ('--max-seconds', 'inf'), ('--temperatures', 'nan'),
                        ('--prefill-chunks', '4096'), ('--seeds', '-1'), ('--modes', 'plain,plain'),
                        ('--cache-states', 'unknown'), ('--prefill-sizes', '131073')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan(self.args(*options))
        args = self.args()
        args.drafter = None
        with self.assertRaisesRegex(ValueError, 'drafter'):
            plan(args)

    def test_full_window_reserves_output_and_retains_exact_array(self):
        tokens = public_tokens(Tokenizer(), 131072, 64, 131072)
        self.assertEqual(len(tokens), 131008)
        self.assertEqual(tokens, public_tokens(Tokenizer(), 131072, 64, 131072))
        self.assertEqual(len(token_hash(tokens)), 64)
        self.assertNotEqual(token_hash(tokens), token_hash(tokens[:-1]))

    def test_identity_detects_different_lengths_and_first_mismatch(self):
        base = dict(prompt='prose', chunk=512, temperature=0, seed=17, state='fresh', rep=0, warmup=False,
                    steady_decode_tps=10, elapsed_seconds=2)
        plain = dict(base, mode='plain', case_id=0, output_tokens=[1, 2, 3])
        draft = dict(base, mode='draft', case_id=1, output_tokens=[1, 2, 3], elapsed_seconds=1)
        result = compare_cases([plain, draft])[0]
        self.assertTrue(result['token_identity'])
        self.assertEqual(result['request_speedup'], 2)
        draft['output_tokens'] = [1, 2]
        self.assertEqual(compare_cases([plain, draft])[0]['first_mismatch'], 2)
        draft['output_tokens'] = [1, 4, 3]
        self.assertEqual(compare_cases([plain, draft])[0]['first_mismatch'], 1)
        self.assertEqual(compare_cases([plain]), [])

    def test_plain_only_is_admitted_without_loading_a_drafter(self):
        args = self.args('--modes', 'plain')
        args.drafter = None
        self.assertEqual(len(plan(args)), 4)

    def test_width_matrix_reuses_plain_baseline(self):
        cases = plan(self.args('--draft-block-widths', '4,8,16', '--temperatures', '0'))
        self.assertEqual(len(cases), 8)
        self.assertEqual(sum(c['mode'] == 'plain' for c in cases), 2)
        self.assertEqual({c['block_width'] for c in cases if c['mode'] == 'draft'}, {4, 8, 16})
        for index, case in enumerate(cases):
            case.update(case_id=index, output_tokens=[1, 2], steady_decode_tps=10, elapsed_seconds=1)
        comparisons = compare_cases(cases)
        self.assertEqual(len(comparisons), 6)
        self.assertTrue(all(c['token_identity'] for c in comparisons))
        with self.assertRaisesRegex(ValueError, 'widths'):
            plan(self.args('--draft-block-widths', '32'))

    def test_cross_chunk_identity_does_not_mix_widths(self):
        cases = plan(self.args('--phase', 'prefill', '--prefill-chunks', '128,512', '--draft-block-widths', '4,16'))
        for index, case in enumerate(cases):
            case.update(case_id=index, output_tokens=[1, case['block_width'] or 0])
        comparisons = chunk_comparisons(cases)
        self.assertEqual(len(comparisons), 3)
        self.assertTrue(all(c['token_identity'] for c in comparisons))
        cases[-1]['output_tokens'] = [8]
        self.assertFalse(chunk_comparisons(cases)[-1]['token_identity'])

    def test_prefill_chunk_and_clear_policies_preserve_one_old_plain_baseline(self):
        cases = plan(self.args('--phase', 'prefill', '--prefill-sizes', '8192,32768',
                               '--prefill-chunks', '512,1024,2048', '--draft-block-widths', '4'))
        self.assertEqual(len(cases), 12)
        self.assertTrue(all(c['prefill_clear_interval'] == 0 for c in cases))
        policies = plan(self.args('--temperatures', '0', '--prefill-clear-intervals', '0,4096,8192',
                                  '--draft-block-widths', '4', '--adaptive-blocks'))
        self.assertEqual(len(policies), 14)
        plain = [c for c in policies if c['mode'] == 'plain']
        self.assertEqual(len(plain), 2)
        self.assertTrue(all(c['prefill_clear_interval'] == 0 and not c['adaptive'] for c in plain))
        adaptive = [c for c in policies if c['adaptive']]
        self.assertEqual(len(adaptive), 6)
        self.assertTrue(all(c['block_width'] == 4 for c in adaptive))
        with self.assertRaisesRegex(ValueError, 'clear intervals'):
            plan(self.args('--prefill-clear-intervals', '1'))

    def test_policy_identity_cannot_overwrite_adaptive_or_clear_variants(self):
        cases = plan(self.args('--temperatures', '0', '--prefill-clear-intervals', '0,4096',
                               '--draft-block-widths', '4', '--adaptive-blocks'))
        for index, case in enumerate(cases):
            case.update(case_id=index, output_tokens=[1, 2], steady_decode_tps=10, elapsed_seconds=1)
        self.assertEqual(len(compare_cases(cases)), 8)
        self.assertTrue(all(row['token_identity'] for row in policy_comparisons(cases)))
        cases[-1]['output_tokens'] = [1, 9]
        self.assertFalse(compare_cases(cases)[-1]['token_identity'])
        self.assertFalse(policy_comparisons(cases)[-1]['token_identity'])

    def test_cache_pool_choice_preserves_case_matrix(self):
        baseline = plan(self.args())
        self.assertEqual(self.args().cache_limit_mib, 128)
        for size in ['512', '1024']:
            args = self.args('--cache-limit-mib', size)
            self.assertEqual(args.cache_limit_mib, int(size))
            self.assertEqual(plan(args), baseline)


if __name__ == '__main__':
    unittest.main()
