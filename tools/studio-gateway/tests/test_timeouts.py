"""Failure paths must release scarce model memory and queued request slots."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket
import time
import unittest

from support import FRIEND_KEY, eventually, gateway


class TimeoutTests(unittest.TestCase):
    def started_request(self, server):
        def assertion():
            self.assertTrue(any(row["event"] == "request" for row in server.rows()))
        eventually(assertion)

    def test_readiness_timeout_reaps_child_and_other_model_still_works(self):
        with gateway(model_options={"swift": ("--ready-delay", "2")},
                     readiness_timeout={"swift": 0.3, "ornith": 3}) as server:
            status, _, _ = server.completion(model="swift")
            self.assertIn(status, (502, 503, 504))
            for row in server.rows():
                if row["event"] == "start":
                    with self.assertRaises(ProcessLookupError):
                        os.kill(row["pid"], 0)
            status, _, raw = server.completion(model="ornith")
            self.assertEqual(status, 200, raw)

    def test_early_backend_exit_does_not_hold_queue_or_load_next_concurrently(self):
        with gateway(model_options={"swift": ("--exit-immediately",)}) as server:
            status, _, _ = server.completion(model="swift")
            self.assertIn(status, (502, 503, 504))
            status, _, raw = server.completion(model="ornith")
            self.assertEqual(status, 200, raw)
            for row in server.rows():
                if row["event"] == "start":
                    self.assertEqual(row["live_predecessors"], [])

    def test_request_timeout_stops_abandoned_work_before_next_model(self):
        with gateway(request_timeout=0.15) as server:
            status, _, _ = server.completion(fixture_delay=2)
            self.assertIn(status, (502, 503, 504))
            started = time.monotonic()
            status, _, raw = server.completion(model="ornith")
            self.assertEqual(status, 200, raw)
            self.assertLess(time.monotonic() - started, 2)

    def test_queue_timeout_never_starts_waiting_model(self):
        with gateway(queue_timeout=0.08) as server:
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(server.completion, fixture_delay=0.4)
                self.started_request(server)
                status, _, _ = server.completion(model="ornith")
                self.assertIn(status, (429, 503, 504))
                self.assertFalse(any(row["event"] == "start" and row["label"] == "ornith"
                                     for row in server.rows()))
                self.assertEqual(running.result()[0], 200)

    def test_revoked_queued_key_cannot_start_another_model(self):
        with gateway() as server:
            with ThreadPoolExecutor(max_workers=2) as pool:
                running = pool.submit(server.completion, fixture_delay=0.45)
                self.started_request(server)
                waiting = pool.submit(server.completion, model="ornith", _key=FRIEND_KEY)
                time.sleep(0.06)
                server.friend_key_file.unlink()
                self.assertEqual(running.result()[0], 200)
                status, _, raw = waiting.result()
                self.assertEqual(status, 401, raw)
                self.assertFalse(any(row["event"] == "start" and row["label"] == "ornith"
                                     for row in server.rows()))

    def test_disconnected_queued_client_does_not_load_its_model(self):
        with gateway() as server:
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(server.completion, fixture_delay=0.4)
                self.started_request(server)
                payload = json.dumps({"model": "ornith", "messages": [
                    {"role": "user", "content": "Public abandoned queue fixture."}]}).encode()
                with socket.create_connection(server.address, timeout=3) as connection:
                    connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                        b"Host: localhost\r\nAuthorization: Bearer " + FRIEND_KEY.encode() +
                        b"\r\nContent-Type: application/json\r\nContent-Length: " +
                        str(len(payload)).encode() + b"\r\n\r\n" + payload)
                    time.sleep(0.06)
                self.assertEqual(running.result()[0], 200)
                self.assertFalse(any(row["event"] == "start" and row["label"] == "ornith"
                                     for row in server.rows()))

    def test_manager_shutdown_during_active_call_is_bounded(self):
        with gateway() as server:
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(server.completion, fixture_delay=20)
                self.started_request(server)
                started = time.monotonic()
                server.server.manager.close()
                self.assertLess(time.monotonic() - started, 3)
                try:
                    response = running.result(timeout=3)
                    self.assertIn(response[0], (500, 502, 503, 504))
                except (ConnectionError, OSError):
                    pass
                for row in server.rows():
                    if row["event"] == "start":
                        with self.assertRaises(ProcessLookupError):
                            os.kill(row["pid"], 0)


if __name__ == "__main__":
    unittest.main()
