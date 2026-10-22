"""Exercise the CLI and parent-death supervisor in real OS processes."""
from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

from support import DIRECTORY, available_port, eventually, gateway


@contextmanager
def cli(server):
    server.close()
    document = dataclasses.asdict(server.config)
    document["cache_root"] = str(document["cache_root"])
    for key in document["key_files"]:
        key["path"] = str(key["path"])
    document["port"] = available_port()
    config_file = server.root / "cli-config.json"
    config_file.write_text(json.dumps(document))
    server.address = ("127.0.0.1", document["port"])
    with (server.root / "cli-log.txt").open("wb") as output:
        process = subprocess.Popen([sys.executable, str(DIRECTORY.parent / "gateway.py"),
                                    "--config", str(config_file)],
                                   stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 4
            while True:
                if process.poll() is not None:
                    raise AssertionError("CLI gateway exited during startup")
                try:
                    if server.request("/v1/models", method="GET")[0] == 200:
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


class ProcessShutdownTests(unittest.TestCase):
    def assert_pid_stopped(self, pid):
        def stopped():
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        eventually(stopped, timeout=6)

    def populate(self, server):
        for name in ("one", "two"):
            status, _, raw = server.completion(_headers={"X-TensorFold-Context": name},
                                               fixture_marker=name)
            self.assertEqual(status, 200, raw)
        self.assertTrue(list(server.state.rglob("*.tfs")))
        return [row["pid"] for row in server.rows() if row["event"] == "start"][-1]

    def test_cli_sigterm_stops_engine_and_erases_ephemeral_contexts(self):
        with gateway() as server:
            with cli(server) as process:
                child_pid = self.populate(server)
                process.terminate()
                self.assertEqual(process.wait(timeout=6), 0)
                self.assert_pid_stopped(child_pid)
                self.assertEqual(list(server.state.rglob("*.tfs")), [])

    def test_sigkill_gateway_stops_engine_and_next_start_clears_stale_contexts(self):
        with gateway() as server:
            with cli(server) as process:
                child_pid = self.populate(server)
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
                self.assert_pid_stopped(child_pid)
                # SIGKILL cannot run cleanup. The following start must erase the
                # marked stale run instead of recovering someone else's state.
                self.assertTrue(list(server.state.rglob("*.tfs")))
            with cli(server):
                self.assertEqual(list(server.state.rglob("*.tfs")), [])


if __name__ == "__main__":
    unittest.main()
