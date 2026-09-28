#!/usr/bin/env python3
"""Astra Studio metrics collector. Standard-library only. Run with --install on the Studio."""
from contextlib import contextmanager
import argparse
import getpass
import hmac
import json
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

RETENTION = 86400

def command(*args):
    return subprocess.check_output(args, text=True, timeout=2, stderr=subprocess.DEVNULL)

def host_metrics():
    result = {}
    try:
        total = int(command('/usr/sbin/sysctl', '-n', 'hw.memsize').strip())
        vm = command('/usr/bin/vm_stat')
        page = int(re.search(r'page size of (\d+) bytes', vm)[1])
        counts = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
        # Explicit estimate, not Activity Monitor's private Memory Used formula.
        used = sum(counts[k] for k in ('Pages active', 'Pages wired down', 'Pages occupied by compressor')) * page
        result.update(host_ram_total_gib=total / 2**30, host_ram_used_gib=used / 2**30,
                      host_compressed_gib=counts['Pages occupied by compressor'] * page / 2**30)
    except Exception:
        pass
    try:
        swap = command('/usr/sbin/sysctl', '-n', 'vm.swapusage')
        match = re.search(r'used\s*=\s*([\d.]+)([KMG])', swap)
        result['host_swap_used_gib'] = float(match[1]) * {'K': 2**10, 'M': 2**20, 'G': 2**30}[match[2]] / 2**30
    except Exception:
        pass
    try:
        result['host_pressure'] = int(command('/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level').strip())
    except Exception:
        pass
    return result

STATES = ['empty', 'queued', 'restoring', 'prefilling', 'generating', 'saving', 'ram', 'ssd', 'uncached', 'error', 'shared']
def tensorfold_metrics(data):
    slots = data['slots']
    result = dict(tracking_version=data.get('tracking_version', 1), tensorfold_version=1, slots_total=len(slots), parallel=data['parallel'], context=data['context'])
    for item in slots:
        sid = int(item['slot'])
        if not 1 <= sid <= 256:
            continue
        result[f'slot_{sid}_state'] = STATES.index(item['state']) if item['state'] in STATES else 9
        for key in ('last_used', 'prompt_tokens', 'cached_tokens', 'generated_tokens', 'prompt_tps', 'decode_tps', 'rate_at', 'ram_bytes', 'ssd_bytes', 'checkpoint_tokens', 'shared_checkpoint_tokens', 'shared_ram', 'shared_ssd'):
            value = item.get(key)
            if isinstance(value, (float, int)) and math.isfinite(value):
                result[f'slot_{sid}_{key}'] = value
    for key, value in data.get('cache_stats', {}).items():
        if isinstance(value, (float, int)) and math.isfinite(value):
            result['checkpoint_' + key] = value
    for key, value in data.get('memory', {}).items():
        if isinstance(value, (float, int)) and math.isfinite(value):
            result['mlx_' + key] = value
    return result

class Collector:
    def __init__(self, home):
        self.key_path = home / '.config/tensorfold/api-key'
        if not self.key_path.exists():
            self.key_path = home / '.config/llama.cpp/api-key'
        self.db_path = home / 'ai/metrics/samples.sqlite3'
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.active_until = 0
        self.last_success = 0
        self.host_last_read = 0
        self.host_values = {}
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA secure_delete=ON')
            db.execute('CREATE TABLE IF NOT EXISTS samples (time INTEGER PRIMARY KEY, metrics TEXT NOT NULL)')
        os.chmod(self.db_path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute('PRAGMA secure_delete=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def key(self):
        key = self.key_path.read_text().strip()
        if not key:
            raise ValueError('Empty API key')
        return key

    def interval(self):
        with self.lock:
            return 1 if time.monotonic() < self.active_until else 5

    def heartbeat(self):
        with self.lock:
            self.active_until = time.monotonic() + 10
        self.wake.set()

    def collect(self):
        now = int(time.time() * 1000)
        # Prune even when TensorFold is offline.
        with self.connect() as db:
            db.execute('DELETE FROM samples WHERE time <= ?', (now - RETENTION * 1000,))
        request = Request('http://127.0.0.1:8080/studio/metrics', headers={'Authorization': 'Bearer ' + self.key()})
        with urlopen(request, timeout=4) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError('Response too large')
        metrics = tensorfold_metrics(json.loads(raw))
        if time.monotonic() - self.host_last_read >= 5:
            self.host_values = host_metrics()
            self.host_last_read = time.monotonic()
        metrics.update(self.host_values)
        now = int(time.time() * 1000)
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO samples VALUES (?, ?)', (now, json.dumps(metrics)))
        self.last_success = now

    def run(self):
        last_attempt = -10.0
        while True:
            remaining = last_attempt + self.interval() - time.monotonic()
            if remaining > 0:
                self.wake.wait(remaining)
                self.wake.clear()
                continue
            last_attempt = time.monotonic()
            try:
                self.collect()
            except Exception:
                # Do not log request headers, keys or unbounded error messages.
                pass

    def history(self, since):
        cutoff = max(since, int(time.time() * 1000) - RETENTION * 1000)
        with self.connect() as db:
            rows = db.execute('SELECT time, metrics FROM samples WHERE time > ? ORDER BY time LIMIT 2001', (cutoff,)).fetchall()
        return {'samples': [{'time': t, 'metrics': json.loads(m)} for t, m in rows[:2000]],
                'hasMore': len(rows) > 2000, 'source': 'collector',
                'intervalSeconds': self.interval(), 'lastSuccess': self.last_success}

def serve(home):
    os.umask(0o077)
    collector = Collector(home)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                authorized = hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + collector.key())
            except Exception:
                authorized = False
            if not authorized:
                self.send_error(401)
                return
            url = urlparse(self.path)
            if url.path != '/monitor/history':
                self.send_error(404)
                return
            params = parse_qs(url.query)
            try:
                since = max(0, int(params.get('since', ['0'])[0]))
            except ValueError:
                self.send_error(400)
                return
            if params.get('active') == ['1']:
                collector.heartbeat()
            body = json.dumps(collector.history(since)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    threading.Thread(target=collector.run, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', 8081), Handler).serve_forever()

def install(home):
    if sys.platform != 'darwin' or os.geteuid() == 0:
        raise SystemExit('Run using python3 as your normal Mac Studio user, without sudo.')
    if not (home / '.config/llama.cpp/api-key').is_file():
        raise SystemExit('Missing ~/.config/llama.cpp/api-key. Run this on the Mac Studio.')
    target = home / 'ai/metrics/studio-collector.py'
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if Path(__file__).resolve() != target.resolve():
        shutil.copyfile(__file__, target)
    target.chmod(0o700)
    label = 'com.astrastudio.metrics'
    service = '/Library/LaunchDaemons/' + label + '.plist'
    plist = {'Label': label, 'ProgramArguments': [sys.executable, str(target), '--home', str(home)],
             'UserName': getpass.getuser(), 'RunAtLoad': True, 'KeepAlive': True,
             'ThrottleInterval': 5, 'StandardOutPath': '/dev/null', 'StandardErrorPath': '/dev/null'}
    with tempfile.NamedTemporaryFile(suffix='.plist') as tmp:
        tmp.write(plistlib.dumps(plist)); tmp.flush()
        subprocess.run(['sudo', 'install', '-o', 'root', '-g', 'wheel', '-m', '644', tmp.name, service], check=True)
    subprocess.run(['sudo', 'launchctl', 'bootout', 'system/' + label], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['sudo', 'launchctl', 'bootstrap', 'system', service], check=True)
    print('Installed. Collector runs at boot: 1 second with a viewer, 5 seconds without; 24-hour retention.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--home', type=Path, default=Path.home())
    args = parser.parse_args()
    install(args.home) if args.install else serve(args.home)
