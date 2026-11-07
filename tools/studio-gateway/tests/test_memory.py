"""Admission must reject before loading and account for resident physical memory."""
import dataclasses
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from config import ModelConfig
from errors import GatewayError
from memory import GIB, check_memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelConfig("fixture", ("/not/executed",), memory_required_gib=40)

    def test_cold_full_context_budget_and_reserve_are_both_required(self):
        for available, refused in [(45, True), (46, False)]:
            with self.subTest(available=available), patch("memory.snapshot", return_value=(available * GIB, 1)):
                if refused:
                    with self.assertRaises(GatewayError) as error:
                        check_memory(self.model, 6)
                    self.assertEqual(error.exception.code, "insufficient_memory")
                else:
                    check_memory(self.model, 6)

    def test_resident_credit_is_measured_and_cannot_exceed_profile_budget(self):
        backend = SimpleNamespace(model=self.model, alive=lambda: True, process=SimpleNamespace(pid=123))
        with patch("memory.snapshot", return_value=(6 * GIB, 1)), patch("memory.group_footprint", return_value=90 * GIB):
            check_memory(self.model, 6, backend)
        with patch("memory.snapshot", return_value=(5 * GIB, 1)), patch("memory.group_footprint", return_value=90 * GIB):
            with self.assertRaises(GatewayError):
                check_memory(self.model, 6, backend)

    def test_pressure_and_missing_telemetry_fail_closed(self):
        for pressure in [0, 2, 4]:
            with patch("memory.snapshot", return_value=(100 * GIB, pressure)):
                with self.assertRaises(GatewayError):
                    check_memory(self.model, 6)
        with patch("memory.snapshot", side_effect=OSError):
            with self.assertRaises(GatewayError) as error:
                check_memory(self.model, 6)
            self.assertEqual(error.exception.code, "memory_status_unavailable")

    def test_wrong_model_and_dead_process_receive_no_credit(self):
        for model, alive in [(dataclasses.replace(self.model, id="other"), True), (self.model, False)]:
            backend = SimpleNamespace(model=model, alive=lambda: alive)
            with patch("memory.snapshot", return_value=(6 * GIB, 1)), patch("memory.group_footprint") as measure:
                with self.assertRaises(GatewayError):
                    check_memory(self.model, 6, backend)
                measure.assert_not_called()

    def test_http_refusal_does_not_start_a_backend_and_next_request_can_retry(self):
        with support.gateway() as server:
            with patch("runtime.check_memory", side_effect=GatewayError(503, "RAM busy", "insufficient_memory"), create=True):
                status, _, raw = server.completion()
            self.assertEqual(status, 503, raw)
            self.assertEqual(server.rows(), [])
            self.assertEqual(server.completion()[0], 200)

    def test_warm_request_is_rechecked_before_any_generation(self):
        with support.gateway() as server:
            self.assertEqual(server.completion()[0], 200)
            before = len(server.rows())
            with patch("runtime.check_memory", side_effect=GatewayError(503, "RAM busy", "insufficient_memory"), create=True):
                self.assertEqual(server.completion()[0], 503)
            self.assertEqual(len(server.rows()), before)

    def test_failed_post_load_check_reaps_backend_before_next_load(self):
        with support.gateway() as server:
            with patch("runtime.check_memory", side_effect=[None, GatewayError(503, "RAM busy", "insufficient_memory")], create=True):
                self.assertEqual(server.completion()[0], 503)
            self.assertIsNone(server.server.manager.backend)
            self.assertEqual(server.completion()[0], 200)
            starts = [r for r in server.rows() if r['event'] == 'start']
            self.assertTrue(all(not r['live_predecessors'] for r in starts))

    def test_low_memory_switch_refuses_before_saving_or_stopping_old_model(self):
        with support.gateway() as server:
            self.assertEqual(server.completion()[0], 200)
            before = server.rows()
            with patch('runtime.check_memory', side_effect=GatewayError(503, 'RAM busy', 'insufficient_memory')):
                self.assertEqual(server.completion(model='ornith')[0], 503)
            self.assertEqual(server.rows(), before)
            self.assertEqual(server.server.manager.backend.model.id, 'swift')


if __name__ == "__main__":
    unittest.main()
