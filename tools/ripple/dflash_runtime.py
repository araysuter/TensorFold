"""Verify the isolated MLX dispatcher's loaded binaries against its private build receipt."""
import ctypes
import hashlib
import json
from pathlib import Path
import mlx.core as mx


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime(receipt):
    receipt = Path(receipt).resolve(strict=True)
    data = json.loads(receipt.read_text())
    if (data.get('source_commit') != '36969c094354086e15d2a327a9c50feb20b9cf11'
            or data.get('installed_into_active_environment') is not False
            or data.get('patch_sha256') != sha256(Path(__file__).with_name('mlx-0.32.4-exact-verify.patch'))):
        raise ValueError('Unqualified isolated MLX build identity')
    root = receipt.parent / 'override'
    files = data['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing isolated MLX file identities')
    expected = {}
    for name, digest in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError('Isolated MLX file checksum mismatch')
        expected[path] = digest
    if Path(mx.__file__).resolve() not in expected:
        raise ValueError('The worker did not import the isolated MLX extension')
    system = ctypes.CDLL(None)
    system._dyld_image_count.restype = ctypes.c_uint32
    system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    system._dyld_get_image_name.restype = ctypes.c_char_p
    loaded = []
    for index in range(system._dyld_image_count()):
        value = system._dyld_get_image_name(index)
        if value and 'libmlx' in Path(value.decode()).name:
            path = Path(value.decode()).resolve()
            if path not in expected or sha256(path) != expected[path]:
                raise ValueError('A different MLX library is loaded')
            loaded.append(expected[path])
    if not loaded:
        raise ValueError('No verified isolated MLX runtime library is loaded')
    return hashlib.sha256(receipt.read_bytes() + ''.join(sorted(loaded)).encode()).hexdigest()
