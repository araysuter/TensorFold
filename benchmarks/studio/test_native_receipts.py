"""Native queue receipts must match their response, prompt, cache and output counts."""
import unittest

from normalize_native import normalize


class QueueReceipts(unittest.TestCase):
    def row(self):
        return {'config': {}, 'requests': [{'status': 'ok', 'request_id': 'cmpl-fixture',
                'actual_prompt_tokens': 20000, 'cached_tokens': 0, 'output_tokens': 1024,
                'server_metrics': {'prefill_seconds': 20.0}}]}

    def test_queue_is_added_without_including_decode(self):
        log = 'done cmpl-fixture prompt=20000 cached=0 thinking=False tokens=1024 sha=fixture queued=12.500 tok/s=80.0'
        data = normalize(self.row(), log)
        self.assertEqual(data['requests'][0]['queue_prefill_seconds'], 32.5)
        self.assertAlmostEqual(data['requests'][0]['effective_input_tps'], 20000 / 32.5)

    def test_missing_or_mismatched_receipt_is_refused(self):
        for log in ('', 'done cmpl-other prompt=20000 cached=0 tokens=1024 sha=fixture queued=1.0',
                    'done cmpl-fixture prompt=19999 cached=0 tokens=1024 sha=fixture queued=1.0',
                    'done cmpl-fixture prompt=20000 cached=1 tokens=1024 sha=fixture queued=1.0',
                    'done cmpl-fixture prompt=20000 cached=0 tokens=12 sha=fixture queued=1.0'):
            with self.subTest(log=log), self.assertRaises(ValueError):
                normalize(self.row(), log)


if __name__ == '__main__':
    unittest.main()
