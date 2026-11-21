import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


class PlainTextTools(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = '<|message_model|>Shell<|content_invoke_tool_json|>{"name":"Shell","args":{}}'
        self.send_response(200)
        self.end_headers()
        if body.get('stream'):
            chunk = {'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}]}
            self.wfile.write(('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())
        else:
            chunk = {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}
            self.wfile.write(json.dumps(chunk).encode())


class CheckerTests(unittest.TestCase):
    def test_plain_text_tool_control_exits_nonzero(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), PlainTextTools)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'result.json'
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('check_tool_calls.py')),
                    'http://127.0.0.1:' + str(server.server_port), '--models', 'control',
                    '--timeout', '5', '--max-tokens', '64', '--output', str(output)],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 2, result.stderr)
                report = json.loads(output.read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(len(report['cases']), 2)
                self.assertTrue(all(not case['passed'] for case in report['cases']))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
