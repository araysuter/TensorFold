"""Real child-process tests for serialization, cancellation and context isolation."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import time
import unittest

from support import FRIEND_KEY, OWNER_KEY, eventually, gateway


def context(identifier):
    return {"X-TensorFold-Context": identifier}


class LifecycleTests(unittest.TestCase):
    def assert_one_engine(self, rows):
        for row in rows:
            if row["event"] == "start":
                self.assertEqual(row["live_predecessors"], [], rows)

    def test_alias_and_switch_stop_previous_engine_before_next_starts(self):
        with gateway() as server:
            for name, expected in (("swift-alias", "swift"), ("ornith", "ornith"),
                                   ("swift", "swift")):
                status, _, raw = server.completion(model=name)
                self.assertEqual(status, 200, raw)
                self.assertEqual(json.loads(raw)["model"], expected)
            rows = server.rows()
            self.assertEqual([row["label"] for row in rows if row["event"] == "start"],
                             ["swift", "ornith", "swift"])
            self.assert_one_engine(rows)

    def test_simultaneous_models_are_serialized_with_at_most_one_child(self):
        with gateway() as server:
            names = ["swift", "ornith", "swift", "ornith", "swift", "ornith"]
            with ThreadPoolExecutor(max_workers=len(names)) as pool:
                results = list(pool.map(lambda name: server.completion(
                    model=name, fixture_delay=0.05), names))
            self.assertTrue(all(status == 200 for status, _, _ in results), results)
            self.assert_one_engine(server.rows())
            self.assertEqual(sum(row["event"] == "request_end" for row in server.rows()), len(names))

    def test_same_context_identifier_cannot_cross_api_key_scope(self):
        with gateway() as server:
            status, _, _ = server.completion(_headers=context("same"), fixture_marker="owner-data")
            self.assertEqual(status, 200)
            status, _, raw = server.completion(_key=FRIEND_KEY, _headers=context("same"),
                                               fixture_marker="friend-data")
            self.assertEqual(status, 200, raw)
            self.assertIsNone(json.loads(raw)["fixture_previous_context"])
            status, _, raw = server.completion(_headers=context("same"))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["fixture_previous_context"], "owner-data")

    def test_lru_bound_is_four_parked_plus_one_active_per_model(self):
        with gateway() as server:
            for index in range(6):
                self.assertEqual(server.completion(_headers=context(f"context-{index}"),
                    fixture_marker=str(index))[0], 200)
            self.assertLessEqual(len(list(server.state.rglob("*.tfs"))), 5)
            status, _, raw = server.completion(_headers=context("context-0"))
            self.assertEqual(status, 200, raw)
            self.assertIsNone(json.loads(raw)["fixture_previous_context"])

    def test_corrupt_disk_context_fails_closed_instead_of_silent_reprefill(self):
        with gateway() as server:
            self.assertEqual(server.completion(_headers=context("first"), fixture_marker="first")[0], 200)
            self.assertEqual(server.completion(_headers=context("second"), fixture_marker="second")[0], 200)
            snapshots = list(server.state.rglob("*.tfs"))
            self.assertEqual(len(snapshots), 1)
            snapshots[0].write_bytes(b"corrupt fixture")
            before = sum(row["event"] == "request" for row in server.rows())
            status, _, _ = server.completion(_headers=context("first"))
            self.assertIn(status, (409, 500, 502, 503))
            self.assertEqual(sum(row["event"] == "request" for row in server.rows()), before)

    def test_snapshot_save_failure_preserves_active_engine_and_context(self):
        with gateway() as server:
            self.assertEqual(server.completion(_headers=context("keep"),
                fixture_marker="unsaved-context", fixture_save_error=True)[0], 200)
            status, _, _ = server.completion(model="ornith", _headers=context("other"))
            self.assertIn(status, (409, 500, 502, 503))
            self.assertFalse(any(row["event"] == "stop" for row in server.rows()))
            self.assertFalse(any(row["event"] == "start" and row["label"] == "ornith"
                                 for row in server.rows()))
            status, _, raw = server.completion(_headers=context("keep"), fixture_save_error=False)
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["fixture_previous_context"], "unsaved-context")

    def test_repeated_failed_switches_do_not_accumulate_unbounded_contexts(self):
        with gateway() as server:
            self.assertEqual(server.completion(_headers=context("keep"),
                fixture_marker="unsaved-context", fixture_save_error=True)[0], 200)
            for index in range(8):
                status, _, _ = server.completion(model="ornith", _headers=context(f"new-{index}"))
                self.assertIn(status, (409, 500, 502, 503))
            contexts = server.server.manager.contexts.rows.values()
            self.assertLessEqual(sum(row.model == "ornith" for row in contexts), 5)
            status, _, raw = server.completion(_headers=context("keep"), fixture_save_error=False)
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["fixture_previous_context"], "unsaved-context")

    def test_native_prefix_miss_explicitly_clears_before_cold_generation(self):
        with gateway() as server:
            self.assertEqual(server.completion(_headers=context("first"), fixture_marker="first")[0], 200)
            self.assertEqual(server.completion(_headers=context("second"), fixture_marker="second")[0], 200)
            status, _, raw = server.completion(_headers=context("first"), fixture_prefix_miss=True)
            self.assertEqual(status, 200, raw)
            self.assertIsNone(json.loads(raw)["fixture_previous_context"])
            self.assertTrue(any(row["event"] == "preflight" for row in server.rows()))

    def test_idle_expiry_saves_context_stops_engine_then_restores_it(self):
        with gateway(idle_seconds=0.15) as server:
            status, _, raw = server.completion(_headers=context("resume"), fixture_marker="saved")
            self.assertEqual(status, 200, raw)

            def stopped():
                self.assertEqual(server.server.manager.status()["loaded_models"], 0)
            eventually(stopped, timeout=3)
            status, _, raw = server.completion(_headers=context("resume"))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["fixture_previous_context"], "saved")
            self.assert_one_engine(server.rows())

    def test_shutdown_releases_child_and_removes_contexts_but_not_root_sibling(self):
        with gateway() as server:
            sentinel = server.state.parent / "keep-this-file"
            sentinel.write_text("public fixture")
            self.assertEqual(server.completion(_headers=context("first"), fixture_marker="first")[0], 200)
            self.assertEqual(server.completion(_headers=context("second"), fixture_marker="second")[0], 200)
            pids = [row["pid"] for row in server.rows() if row["event"] == "start"]
            self.assertTrue(list(server.state.rglob("*.tfs")))
            server.server.manager.close()
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            self.assertEqual(list(server.state.rglob("*.tfs")), [])
            self.assertEqual(sentinel.read_text(), "public fixture")

    def test_disconnect_releases_stream_and_next_model_can_run(self):
        with gateway(request_timeout=3) as server:
            connection = http.client.HTTPConnection(*server.address, timeout=3)
            payload = json.dumps({"model": "swift", "stream": True,
                "messages": [{"role": "user", "content": "Public cancellation fixture."}],
                "fixture_delay": 0.03, "fixture_chunks": 1000})
            connection.request("POST", "/v1/chat/completions", body=payload,
                headers={"Authorization": "Bearer " + OWNER_KEY,
                         "Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.readline().startswith(b"data:"))
            response.close()
            connection.close()
            started = time.monotonic()
            status, _, raw = server.completion(model="ornith")
            self.assertEqual(status, 200, raw)
            self.assertLess(time.monotonic() - started, 3)
            self.assert_one_engine(server.rows())


if __name__ == "__main__":
    unittest.main()
