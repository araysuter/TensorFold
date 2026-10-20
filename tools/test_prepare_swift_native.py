"""Check the Swift conversion's arithmetic and source-preservation boundary."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_swift", Path(__file__).with_name("prepare-swift-native.py"))
prepare_swift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_swift)


class SwiftConversionTests(unittest.TestCase):
    def test_bf16_rounding(self):
        # +/-0 become 1; -1 becomes 0; 1 becomes 2; a half-way value rounds to even.
        original = struct.pack("<5H", 0, 0x8000, 0xbf80, 0x3f80, 0x3b80)
        self.assertEqual(prepare_swift.convert(original), struct.pack("<5H", 0x3f80, 0x3f80, 0, 0x4000, 0x3f80))

    def fixture(self, root, count):
        source = root / prepare_swift.REVISION
        source.mkdir()
        (source / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 5120}}))
        header = {}
        data = bytearray()
        for i in range(count):
            name = f"language_model.model.layers.{i}.input_layernorm.weight"
            header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [len(data), len(data) + 2]}
            data.extend(struct.pack("<H", 0))
        header["language_model.other.weight"] = {"dtype": "U32", "shape": [1], "data_offsets": [len(data), len(data) + 4]}
        data.extend(b"abcd")
        encoded = json.dumps(header).encode()
        raw = struct.pack("<Q", len(encoded)) + encoded + data
        (source / "model.safetensors").write_bytes(raw)
        return source, raw

    def test_copy_changes_only_norm_payloads_and_refuses_reapplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, raw = self.fixture(root, 161)
            dest = root / "converted"
            prepare_swift.prepare(source, dest)
            self.assertEqual((source / "model.safetensors").read_bytes(), raw)
            converted = (dest / "model.safetensors").read_bytes()
            payload = 8 + struct.unpack("<Q", raw[:8])[0]
            self.assertEqual(converted[:payload], raw[:payload])
            self.assertEqual(converted[payload:], struct.pack("<H", 0x3f80) * 161 + b"abcd")
            self.assertEqual(len(json.loads((dest / "swift-native.json").read_text())["normalization_tensors"]), 161)
            with self.assertRaises(FileExistsError):
                prepare_swift.prepare(source, dest)
            self.assertEqual((dest / "model.safetensors").read_bytes(), converted)

    def test_wrong_count_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.fixture(root, 160)
            dest = root / "converted"
            with self.assertRaisesRegex(ValueError, "Expected 161"):
                prepare_swift.prepare(source, dest)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
