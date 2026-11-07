"""BF16 tower attachment preserves tensor bytes and rejects malformed source metadata."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_vision', Path(__file__).with_name('prepare-qwen-vision.py'))
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)


def shard(path, tensors):
    offset, table, payload = 0, {}, b''
    for name, data in tensors.items():
        table[name] = dict(dtype='BF16', shape=[len(data) // 2], data_offsets=[offset, offset + len(data)])
        offset += len(data)
        payload += data
    raw = json.dumps(table).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


class VisionTests(unittest.TestCase):
    def fixture(self, root):
        source, target = root / 'source', root / 'target'
        source.mkdir()
        target.mkdir()
        config = {'text_config': {'hidden_size': 4}, 'vision_config': {'out_hidden_size': 4, 'depth': 27}}
        for path in [source, target]:
            (path / 'config.json').write_text(json.dumps(config))
        tensors = {'model.visual.fixture.' + str(i): struct.pack('<H', i) for i in range(333)}
        shard(source / 'model-source.safetensors', tensors)
        shard(target / 'model-text.safetensors', {'language_model.fixture': b'\x01\x02'})
        for path, weights in [(source, {k: 'model-source.safetensors' for k in tensors}),
                              (target, {'language_model.fixture': 'model-text.safetensors'})]:
            (path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weights}))
        return source, target

    def test_bf16_bytes_are_identical_and_text_shard_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.fixture(Path(directory))
            before = (target / 'model-text.safetensors').read_bytes()
            receipt = vision.attach(source, target, 'example/model', 'a' * 40)
            self.assertEqual(receipt['vision_tensor_count'], 333)
            self.assertEqual((target / 'model-text.safetensors').read_bytes(), before)
            path = target / 'model-vision-bf16.safetensors'
            table, start = vision.header(path)
            data = path.read_bytes()
            for i in range(333):
                entry = table['vision_tower.fixture.' + str(i)]
                begin, end = entry['data_offsets']
                self.assertEqual(data[start + begin:start + end], struct.pack('<H', i))
            self.assertEqual(json.loads((target / 'model.safetensors.index.json').read_text())['metadata']['total_size'], 668)
            with self.assertRaises(ValueError):
                vision.attach(source, target, 'example/model', 'a' * 40)

    def test_wrong_width_and_escaped_shard_fail_before_output(self):
        for bad in ('width', 'path', 'truncated'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                source, target = self.fixture(Path(directory))
                if bad == 'width':
                    p = target / 'config.json'
                    c = json.loads(p.read_text())
                    c['text_config']['hidden_size'] = 8
                    p.write_text(json.dumps(c))
                elif bad == 'path':
                    p = source / 'model.safetensors.index.json'
                    c = json.loads(p.read_text())
                    c['weight_map']['model.visual.fixture.0'] = '../outside.safetensors'
                    p.write_text(json.dumps(c))
                else:
                    p = source / 'model-source.safetensors'
                    p.write_bytes(p.read_bytes()[:-2])
                with self.assertRaises(ValueError):
                    vision.attach(source, target, 'example/model', 'a' * 40)
                self.assertFalse((target / 'model-vision-bf16.safetensors').exists())

    def test_legacy_index_replacement_requires_explicit_native_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = self.fixture(Path(directory))
            p = target / 'model.safetensors.index.json'
            value = json.loads(p.read_text())
            value['weight_map']['visual.legacy'] = 'model-text.safetensors'
            p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                vision.attach(source, target, 'example/model', 'a' * 40)
            receipt = vision.attach(source, target, 'example/model', 'a' * 40, True)
            self.assertTrue(receipt['legacy_index_removed'])
            self.assertNotIn('visual.legacy', json.loads(p.read_text())['weight_map'])


if __name__ == '__main__':
    unittest.main()
