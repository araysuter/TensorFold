"""Opaque conversation affinity must be scoped, bounded and path-safe."""
from pathlib import Path
import tempfile
import unittest

import support
from gateway import ContextStore, GatewayError, Workspace


class ContextStoreTests(unittest.TestCase):
    def test_identifiers_are_scoped_to_both_model_and_credential_identity(self):
        with tempfile.TemporaryDirectory() as root:
            store = ContextStore(Path(root))
            first = store.select("swift", "owner-a", {}, "same-context")
            other_owner = store.select("swift", "owner-b", {}, "same-context")
            other_model = store.select("ornith", "owner-a", {}, "same-context")
            self.assertEqual(len({row.path for row in (first, other_owner, other_model)}), 3)
            self.assertIs(store.select("swift", "owner-a", {}, "same-context"), first)
            for row in (first, other_owner, other_model):
                self.assertEqual(row.path.parent, Path(root))
                self.assertEqual(row.path.suffix, ".tfs")
                self.assertNotIn("same-context", row.path.name)
                self.assertNotIn("owner", row.path.name)

    def test_context_path_traversal_and_oversized_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store = ContextStore(Path(root))
            for identifier in ("../outside", "/absolute", "a/b", "a\\b", "a\0b", "a" * 129, ""):
                with self.subTest(identifier=repr(identifier)):
                    with self.assertRaises(GatewayError):
                        store.select("swift", "owner", {}, identifier)
            self.assertEqual(store.rows, {})
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_shared_system_prompt_does_not_associate_unrelated_chats(self):
        with tempfile.TemporaryDirectory() as root:
            store = ContextStore(Path(root))
            system = {"role": "system", "content": "Shared public instructions."}
            one = store.select("swift", "owner", {"messages": [system]})
            two = store.select("swift", "owner", {"messages": [system]})
            self.assertIsNot(one, two)
            one = store.select("swift", "owner", {"messages": [system, {
                "role": "user", "content": "First public question."}]})
            two = store.select("swift", "owner", {"messages": [system, {
                "role": "user", "content": "Another public question."}]})
            self.assertIsNot(one, two)

    def test_longest_prefix_affinity_keeps_no_plaintext_transcript(self):
        with tempfile.TemporaryDirectory() as root:
            store = ContextStore(Path(root))
            messages = [{"role": "user", "content": "Unpublished fixture text 26b8767a."}]
            initial = store.select("swift", "owner", {"messages": messages})
            continued = store.select("swift", "owner", {"messages": [*messages,
                {"role": "assistant", "content": "Synthetic response."},
                {"role": "user", "content": "Follow-up fixture."}]})
            self.assertIs(initial, continued)
            self.assertNotIn("Unpublished fixture", repr(store.rows))
            self.assertNotIn("Synthetic response", repr(store.rows))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_workspace_lock_blocks_second_owner_and_cleanup_preserves_unmarked_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-contexts"
            workspace = Workspace(root)
            keep = root / ("run-" + "a" * 32)
            keep.mkdir()
            (keep / "keep").write_text("unrelated directory")
            try:
                with self.assertRaises(ValueError):
                    Workspace(root)
            finally:
                workspace.close()
            self.assertEqual((keep / "keep").read_text(), "unrelated directory")
            self.assertFalse(workspace.path.exists())

    def test_stale_marked_contexts_are_removed_after_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-contexts"
            root.mkdir(mode=0o700)
            stale = root / ("run-" + "b" * 32)
            stale.mkdir(mode=0o700)
            (stale / ".gateway-run").write_bytes(Workspace.MARKER)
            (stale / "old-private-context.tfs").write_bytes(b"synthetic state")
            workspace = Workspace(root)
            try:
                self.assertFalse(stale.exists())
            finally:
                workspace.close()


if __name__ == "__main__":
    unittest.main()
