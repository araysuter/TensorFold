#!/usr/bin/env python3
"""Build the pinned MLX verification dispatcher without installing into the active environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile


BASE_SHA256 = '947e92b5d51134f329698b2497a8315b135a0012994af385ef296adb962b5850'
RELATIVE = 'mlx/backend/metal/quantized.cpp'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, cwd):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def main(args):
    if not 1 <= args.jobs <= 12 or not 60 <= args.max_seconds <= 3600:
        raise ValueError('Build requires 1..12 jobs and a 60..3600 second limit')
    source = args.source.resolve(strict=True)
    python = args.python.expanduser().absolute()
    if not python.is_file():
        raise ValueError('Existing environment Python not found')
    args.root = args.root.expanduser().absolute()
    patch = args.patch.resolve(strict=True)
    revision = command(['git', 'rev-parse', 'HEAD'], source)
    tag = command(['git', 'rev-parse', 'v0.32.4^{commit}'], source)
    if revision != tag or revision != '36969c094354086e15d2a327a9c50feb20b9cf11':
        raise ValueError('Source must be the original pinned v0.32.4 commit')
    if command(['git', 'status', '--porcelain'], source):
        raise ValueError('Use a clean, separate source checkout')
    original = source / RELATIVE
    if sha(original) != BASE_SHA256:
        raise ValueError('Pinned quantized.cpp checksum mismatch')
    if args.root.exists():
        raise ValueError('Use a new private build output directory')
    args.root.mkdir(mode=0o700, parents=True)
    wheels, override = args.root / 'wheels', args.root / 'override'
    wheels.mkdir(mode=0o700)
    override.mkdir(mode=0o700)
    receipt = {'status': 'preparing', 'source_tag': 'v0.32.4', 'source_commit': revision,
               'base_quantized_cpp_sha256': BASE_SHA256, 'patch_sha256': sha(patch),
               'jobs': args.jobs, 'max_seconds': args.max_seconds,
               'installed_into_active_environment': False, 'gpu_execution': False}

    def save():
        (args.root / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    save()
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=source, check=True)
    receipt['candidate_quantized_cpp_sha256'] = sha(original)
    env = dict(os.environ)
    env.update(CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), PYPI_RELEASE='0', DEV_RELEASE='0',
               MLX_BUILD_FRONTEND_PACKAGE='0', MLX_BUILD_BACKEND_PACKAGE='0')
    env.pop('CMAKE_ARGS', None)
    argv = [str(python), '-m', 'pip', 'wheel', '--no-deps', '--no-cache-dir', '--wheel-dir', str(wheels), str(source)]
    receipt.update(status='building', command=['python', '-m', 'pip', 'wheel', '--no-deps', '--no-cache-dir',
                                              '--wheel-dir', 'PRIVATE/wheels', 'PRIVATE/source'])
    save()
    started = time.monotonic()
    with (args.root / 'build.log').open('wb') as log:
        process = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.max_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            receipt.update(status='time_guard_stopped', elapsed_seconds=time.monotonic() - started)
            save()
            raise RuntimeError('Bounded build timed out; private log and partial files preserved')
    receipt.update(exit_code=code, elapsed_seconds=time.monotonic() - started)
    if code != 0:
        receipt['status'] = 'build_failed'
        save()
        raise RuntimeError('Wheel build failed; inspect the private build.log')
    built = list(wheels.glob('mlx-*.whl'))
    if len(built) != 1:
        raise RuntimeError('Expected one combined mlx wheel')
    wheel = built[0]
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            destination = (override / name).resolve()
            if not destination.is_relative_to(override.resolve()):
                raise ValueError('Unsafe wheel entry')
        archive.extractall(override)
    libraries = [path for path in override.rglob('*') if path.is_file() and
                 (path.suffix in ('.so', '.dylib', '.metallib') or path.name.startswith('libmlx'))]
    if not any(path.name.startswith('core.') and path.suffix == '.so' for path in libraries):
        raise ValueError('Combined wheel did not contain the Python core extension')
    if not any(path.name.startswith('libmlx') for path in libraries):
        raise ValueError('Combined wheel did not contain a private MLX runtime library')
    receipt.update(status='built_not_qualified', wheel=wheel.name, wheel_sha256=sha(wheel),
                   files={str(path.relative_to(override)): sha(path) for path in libraries})
    save()
    print(json.dumps({'status': receipt['status'], 'override': str(override), 'receipt': str(args.root / 'build-receipt.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--patch', type=Path, default=Path(__file__).with_name('mlx-0.32.4-exact-verify.patch'))
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--max-seconds', type=int, default=1800)
    main(parser.parse_args())
