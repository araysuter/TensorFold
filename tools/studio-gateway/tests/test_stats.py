"""Collector compatibility without a native /stats route or leaked metadata."""
import importlib.util
import json
from pathlib import Path
import time
import unittest

from support import eventually, gateway
from stats import compatibility_stats, parse_prometheus, sanitize_stats


class StatsTests(unittest.TestCase):
    def test_missing_stats_falls_back_to_metrics_memory_and_existing_collector(self):
        with gateway() as server:
            self.assertEqual(server.completion()[0], 200)
            status, _, raw = server.request("/stats", method="GET")
            self.assertEqual(status, 200)
            data = json.loads(raw)
            self.assertEqual(data["running"], 1)
            self.assertEqual(data["waiting"], 2)
            self.assertEqual(data["generation_tokens_total"], 6000)
            self.assertEqual(data["prompt_tokens_total"], 12000)
            self.assertEqual(data["decode_tokens_per_second"], 100)
            self.assertEqual(data["prefill_tokens_per_second"], 500)
            self.assertEqual(data["memory"]["physical_footprint_bytes"], 1100)
            self.assertEqual(data["memory"]["device_active_bytes"], 900)
            self.assertEqual(data["rate_scope"], "completed_requests_since_model_load")
            self.assertNotIn(b"never-publish", raw)
            self.assertNotIn(b"private_path", raw)
            self.assertNotIn(b"unknown_private_value", raw)
            self.assertNotIn(b"NaN", raw)
            path = Path(__file__).resolve().parents[2] / "studio-collector.py"
            spec = importlib.util.spec_from_file_location("studio_collector_compat", path)
            collector = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(collector)
            metrics = collector.tensorfold_metrics(data)
            self.assertEqual(metrics["native_running"], 1)
            self.assertEqual(metrics["native_decode_tokens_per_second"], 100)
            self.assertEqual(metrics["native_process_physical_footprint_bytes"], 1100)

    def test_stats_polls_do_not_load_or_keep_an_engine_resident(self):
        with gateway(idle_seconds=0.25) as server:
            self.assertEqual(server.request("/stats", method="GET")[0], 200)
            self.assertEqual(server.rows(), [])
            self.assertEqual(server.completion()[0], 200)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and server.server.manager.status()["loaded_models"]:
                self.assertEqual(server.request("/stats", method="GET")[0], 200)
                time.sleep(0.02)
            self.assertEqual(server.server.manager.status()["loaded_models"], 0)

    def test_numeric_sanitizer_rejects_private_nonfinite_and_invalid_values(self):
        data = sanitize_stats({
            "running": True, "waiting": -1, "prompt_tokens_total": 10 ** 1000,
            "decode_tokens_per_second": float("inf"), "generation_tokens_total": 7,
            "private_key": "never-publish", "memory": {"physical_footprint_bytes": 100, "secret": 123},
            "slots": [{"slot": True, "state": "ram"}, {"slot": 2, "state": []}],
            "parallel": 1, "context": 131072,
        })
        self.assertEqual(data["generation_tokens_total"], 7)
        self.assertEqual(data["memory"], {"physical_footprint_bytes": 100})
        self.assertEqual(data["slots"], [])
        for name in ("running", "waiting", "prompt_tokens_total", "decode_tokens_per_second", "private_key"):
            self.assertNotIn(name, data)

    def test_prometheus_ignores_labels_nonfinite_and_unsupported_metrics(self):
        values = parse_prometheus(b'\n'.join([
            b'tensorfold:requests_running{key="private"} 5',
            b'tensorfold:requests_running NaN', b'tensorfold:requests_waiting +Inf',
            b'tensorfold:generation_tokens_total -1', b'tensorfold_unknown_private 42',
            b'tensorfold_prompt_tokens_total 6e2',
        ]))
        self.assertEqual(values, {"prompt_tokens_total": 600})
        self.assertNotIn("fill_tokens_left", compatibility_stats(b"", {}))


if __name__ == "__main__":
    unittest.main()
