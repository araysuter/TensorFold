"""Global SSD limits include opaque native state files without following symlinks."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from support import DIRECTORY
from private_state import ContextStore, Conversation
from disk_contexts import native_states


class DiskContexts(unittest.TestCase):
    def test_eight_total_across_native_and_legacy_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ContextStore(root, 8)
            offset = time.time() - time.monotonic()
            for i in range(4):
                path = root / (str(i) + '.tfs');path.touch(mode=0o600)
                row = Conversation('swift' if i % 2 else 'arete', 'owner', str(i), path, touched=time.monotonic()-100+i)
                store.rows[(row.model,row.owner,row.id)] = row
            disk = root/'liquid-disk'/('a'*16)
            disk.mkdir(parents=True)
            for i in range(6):
                path = disk / (f'{i:032x}.st');path.touch(mode=0o600);os.utime(path,(offset+time.monotonic()-50+i,)*2)
            store.prune()
            self.assertEqual(len(store.rows)+len(native_states(root)),8)
            self.assertEqual(len(store.rows),2)

    def test_native_lru_touch_and_partial_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=ContextStore(root,1);disk=root/'liquid-disk'/('b'*16);disk.mkdir(parents=True)
            old=disk/('0'*32+'.st');new=disk/('1'*32+'.st')
            for p in (old,new):p.touch(mode=0o600)
            os.utime(old,(time.time()-100,)*2);old.with_suffix('.used').touch(mode=0o600)
            partial=disk/('2'*32+'.st.part');partial.touch(mode=0o600)
            store.prune();self.assertTrue(old.exists());self.assertFalse(new.exists());self.assertTrue(partial.exists())

    def test_native_symlinks_do_not_count_or_get_followed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);disk=root/'liquid-disk'/('c'*16);disk.mkdir(parents=True)
            victim=root/'outside';victim.write_bytes(b'keep');(disk/('0'*32+'.st')).symlink_to(victim)
            self.assertEqual(native_states(root),[]);ContextStore(root,0).prune();self.assertEqual(victim.read_bytes(),b'keep')

    def test_eight_slots_are_valid_but_nine_are_rejected(self):
        import dataclasses
        from config import GatewayConfig, ModelConfig, KeyFile, validate
        conf=GatewayConfig(models=(ModelConfig('test',('/bin/true',)),),key_files=(KeyFile('key',Path('/tmp/key')),),cache_root=Path('/tmp/contexts/test'),default_model='test',max_parked=8)
        self.assertEqual(validate(conf).max_parked,8)
        with self.assertRaises(ValueError):validate(dataclasses.replace(conf,max_parked=9))
