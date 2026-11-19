"""A failed or refused stream must never become a successful concurrency datapoint."""
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/bench_studio_concurrency.py'


class BenchmarkEvidenceTests(unittest.TestCase):
    def run_case(self, mode):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if mode == 'limited' and body['draft'] and body['prompt'][0] == 1997:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for text in ['public', ' fixture']:
                    self.wfile.write(('data: ' + json.dumps({'choices': [{'text': text}]}) + '\n\n').encode())
                    self.wfile.flush()
                    time.sleep(0.002)
                final = {'choices': [], 'usage': {'completion_tokens': body['max_tokens'], 'prompt_tokens': 4,
                         'prompt_tokens_details': {'cached_tokens': 0}},
                         'tensorfold': {'token_sha256': 'different' if mode == 'mismatch' and body['draft'] else 'same',
                                        'prefill_seconds': 0.1, 'tokens_per_second': 100}}
                self.wfile.write(('data: ' + json.dumps(final) + '\n\ndata: [DONE]\n\n').encode())

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'report.json'
                result = subprocess.run([sys.executable, str(SCRIPT), f'http://127.0.0.1:{server.server_port}',
                                         'fixture', '--lengths', '4', '--levels', '1,2', '--tokens', '8',
                                         '--output', str(output)], capture_output=True, timeout=15)
                return result.returncode, json.loads(output.read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_token_mismatch_fails_and_retains_the_failed_cell(self):
        code, report = self.run_case('mismatch')
        self.assertNotEqual(code, 0)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['cells'][0]['exact'])

    def test_capacity_rejection_is_reported_as_one_of_two_not_two_users(self):
        code, report = self.run_case('limited')
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'capacity_limited')
        self.assertEqual((report['cells'][1]['users'], report['cells'][1]['admitted']), (2, 1))
        self.assertEqual(report['cells'][1]['rows'][1]['status'], 503)


if __name__ == '__main__':
    unittest.main()
