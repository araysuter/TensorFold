"""Single-owner Ripple KV state with private, atomic, identity-checked SSD snapshots."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
from safetensors import safe_open


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


class ResidentState:
    def __init__(self, root, identity, model_args, context):
        self.root = Path(root).absolute()
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Snapshot root must be a private owned directory')
        self.identity, self.args, self.context = identity, model_args, context
        self.tokens, self.cache = [], None

    def clear(self):
        self.tokens, self.cache = [], None
        gc.collect()
        mx.clear_cache()
        return {'tokens': 0, 'bytes': 0, 'fingerprint': '0' * 16}

    def resumable(self, tokens):
        count = len(self.tokens)
        return count if self.cache and 0 < count < len(tokens) and tokens[:count] == self.tokens else 0

    def take(self, tokens):
        count = self.resumable(tokens)
        cache = self.cache if count else None
        self.tokens, self.cache = [], None
        if not count:
            gc.collect()
            mx.clear_cache()
        return cache, count

    def commit(self, tokens, cache):
        offsets = {int(c.offset) for c in cache}
        if len(offsets) != 1:
            raise ValueError('Inconsistent KV positions')
        count = offsets.pop()
        if not 0 < count <= len(tokens) <= self.context:
            raise ValueError('Invalid KV token history')
        self.tokens, self.cache = tokens[:count], cache

    def path(self, value):
        path = Path(value).absolute()
        if path.parent != self.root or not re.fullmatch(r'[0-9a-f]{64}\.tfs', path.name):
            raise ValueError('Invalid snapshot destination')
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('Unsafe snapshot file')
        return path

    def save(self, value):
        path = self.path(value)
        if not self.cache or not self.tokens:
            return {'tokens': 0, 'bytes': 0, 'fingerprint': '0' * 16}
        fd, temporary = tempfile.mkstemp(prefix='.mlx-', suffix='.safetensors', dir=self.root)
        os.close(fd)
        temporary = Path(temporary)
        try:
            save_prompt_cache(str(temporary), self.cache, metadata={
                'format': 'tensorfold-ripple-kv-v1', 'identity': self.identity,
                'tokens': json.dumps(self.tokens, separators=(',', ':'))})
            os.chmod(temporary, 0o600)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            fingerprint = digest(temporary)[:16]
            size = temporary.stat().st_size
            os.replace(temporary, path)
            return {'tokens': len(self.tokens), 'bytes': size, 'fingerprint': fingerprint}
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, value):
        if self.cache is not None:
            raise ValueError('Clear resident KV before import')
        path = self.path(value)
        info = path.lstat()
        if not 16 <= info.st_size <= 16 * 1024**3:
            raise ValueError('Invalid snapshot size')
        with safe_open(str(path), framework='numpy') as stream:
            metadata = stream.metadata()
        if (metadata.get('0.format') != 'tensorfold-ripple-kv-v1'
                or metadata.get('0.identity') != self.identity):
            raise ValueError('Snapshot model/build mismatch')
        tokens = json.loads(metadata.get('0.tokens', 'null'))
        if (not isinstance(tokens, list) or not 0 < len(tokens) <= self.context
                or any(type(t) is not int or not 0 <= t < self.args.vocab_size for t in tokens)):
            raise ValueError('Invalid saved token history')
        expected = ['RotatingKVCache' if t == 'sliding_attention' else 'KVCache' for t in self.args.layer_types]
        classes = {k: v for k, v in metadata.items() if k.startswith('1.')}
        if classes != {f'1.{i}': name for i, name in enumerate(expected)}:
            raise ValueError('Unexpected snapshot cache classes')
        fingerprint = digest(path)[:16]
        fd, temporary = tempfile.mkstemp(prefix='.import-', suffix='.safetensors', dir=self.root)
        os.close(fd)
        os.unlink(temporary)
        try:
            os.link(path, temporary, follow_symlinks=False)
            cache, _ = load_prompt_cache(temporary, return_metadata=True)
        finally:
            Path(temporary).unlink(missing_ok=True)
        try:
            if len(cache) != len(expected):
                raise ValueError('Missing saved cache entries')
            for index, c in enumerate(cache):
                if type(c.offset) is not int or c.offset != len(tokens):
                    raise ValueError('Invalid saved cache offset')
                local = expected[index] == 'RotatingKVCache'
                window = getattr(self.args, 'cache_windows', [self.args.sliding_window] * len(expected))[index]
                bound = window + 2048 if local else self.context + 256
                if c.keys is None or c.values is None or c.keys.shape != c.values.shape:
                    raise ValueError('Inconsistent KV shapes')
                if c.keys.ndim != 4:
                    raise ValueError('Invalid saved KV rank')
                if local and any(type(value) is not int for value in [c._idx, c.max_size, c.keep]):
                    raise ValueError('Invalid rotating cache coordinates')
                if local and (c.max_size != window or c.keep != 0 or not 0 <= c._idx <= c.keys.shape[2]):
                    raise ValueError('Invalid rotating cache state')
                minimum = min(len(tokens), window) if local else len(tokens)
                if c.keys.shape[2] < minimum or (local and len(tokens) <= window and c._idx != len(tokens)):
                    raise ValueError('Missing saved KV history')
                for array in (c.keys, c.values):
                    if (array is None or array.dtype != mx.bfloat16 or array.ndim != 4
                            or array.shape[0] != 1 or array.shape[1] != self.args.num_key_value_heads
                            or array.shape[3] != self.args.head_dim or not 0 < array.shape[2] <= bound):
                        raise ValueError('Invalid saved KV shape')
            mx.eval([c.state for c in cache])
            after = path.lstat()
            if (info.st_ino, info.st_size, info.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Snapshot changed during import')
            self.tokens, self.cache = tokens, cache
        except BaseException:
            del cache
            self.clear()
            raise
        return {'tokens': len(tokens), 'bytes': info.st_size, 'fingerprint': fingerprint}
