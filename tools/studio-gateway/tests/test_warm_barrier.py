"""An in-flight canonical warm pass must not force an avoidable cold prefill."""
from pathlib import Path
import unittest
from unittest.mock import Mock

import support  # Adds the gateway module directory to the test import path.
from config import ModelConfig
from private_state import Conversation
from runtime import ModelManager


class WarmBarrierTests(unittest.TestCase):
    def test_active_preflight_rechecks_after_save_barrier_before_clearing(self):
        for warmed, expected_status in ((True, "resident"), (False, "cold")):
            with self.subTest(warm_completed_before_save_returned=warmed):
                events = []
                results = iter((False, warmed))

                def preflight(_payload):
                    events.append("preflight")
                    return next(results)

                model = ModelConfig("swift", ("/not/executed",))
                wanted = Conversation("swift", "synthetic-owner", "synthetic-context",
                                      Path("/not/written.tfs"), dirty=True)
                manager = object.__new__(ModelManager)
                manager.active = wanted
                manager.backend = Mock()
                manager.backend.model = model
                manager.backend.alive.return_value = True
                manager.backend.can_resume.side_effect = preflight
                manager.backend.control.side_effect = lambda action: events.append(action)
                manager._save = Mock(side_effect=lambda: events.append("save_barrier"))

                manager._activate(model, wanted, {"messages": []}, "/v1/chat/completions", lambda: False)

                self.assertEqual(events[:3], ["preflight", "save_barrier", "preflight"])
                manager._save.assert_called_once_with()
                self.assertEqual(manager.backend.can_resume.call_count, 2)
                self.assertEqual(manager.backend.context_status, expected_status)
                self.assertNotEqual(manager.backend.context_status, "resumed")
                if warmed:
                    manager.backend.control.assert_not_called()
                    self.assertEqual(len(events), 3)
                else:
                    manager.backend.control.assert_called_once_with("clear")
                    self.assertEqual(events[-1], "clear")


if __name__ == "__main__":
    unittest.main()
