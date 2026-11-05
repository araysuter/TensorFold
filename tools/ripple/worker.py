#!/usr/bin/env python3
"""Temporary, loopback-only MLX chat worker under the existing Studio gateway supervisor."""
import argparse
import functools
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
import mlx.core as mx
from mlx_lm import server as api
from mlx_lm.models.cache import make_prompt_cache
from modeling_ricursive_mlx import ModelArgs
from probe import footprint
from worker_state import ResidentState
from dflash_model import DraftArgs, load_draft
from dflash_target import DraftedTarget
from dflash_generate import stream_generate as dflash_stream_generate
from dflash_runtime import verify_runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio-gateway'))
from config import KeyFile
from private_state import KeyStore
from errors import GatewayError
from text_chat import validate_body


def build_identity(model, drafter=None):
    value = hashlib.sha256()
    for file in [Path(__file__), Path(__file__).with_name('worker_state.py'), Path(__file__).with_name('probe.py'),
                 Path(__file__).with_name('modeling_ricursive_mlx.py'),
                 model / 'config.json', model / 'model.safetensors.index.json',
                 model / 'conversion-receipt.json', model / 'modeling_ricursive_mlx.py',
                 model / 'tokenizer.json', model / 'tokenizer_config.json',
                 *[Path(__file__).with_name(name) for name in
                   ['dflash_model.py', 'dflash_target.py', 'dflash_generate.py',
                    'dflash_exact.py', 'dflash_runtime.py']]]:
        value.update(file.read_bytes())
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for name in sorted(set(index['weight_map'].values())):
        info = (model / name).stat()
        value.update(f'{name}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}'.encode())
    for package in ['mlx', 'mlx-lm', 'transformers']:
        value.update(f'{package}={importlib.metadata.version(package)}'.encode())
    if drafter is not None:
        for name in ['config.json', 'conversion-receipt.json']:
            value.update((drafter / name).read_bytes())
        info = (drafter / 'model.safetensors').stat()
        value.update(f'drafter:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}'.encode())
    value.update(os.environ.get('TF_SESSION_BUILD_ID', '').encode())
    return value.hexdigest()


class LocalProvider(api.ModelProvider):
    def _load(self, *args, **kwargs):
        super()._load(*args, **kwargs)
        if self.cli_args.dflash_drafter is not None:
            self.model = DraftedTarget(self.model, load_draft(self.cli_args.dflash_drafter),
                                       exact_verification=self.cli_args.dflash_exact_verify)
        self.is_batchable = False

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        if model_path not in ('default_model', self.cli_args.name) or adapter_path or draft_model_path not in (None, 'default_model'):
            raise ValueError('Only the configured local model is available')
        return super().load('default_model', None, None)


class Engine(api.ResponseGenerator):
    def __init__(self, provider, state):
        self.state, self.ready = state, threading.Event()
        self.stats = {'running': 0, 'waiting': 0, 'prompt_tokens_total': 0, 'generation_tokens_total': 0}
        super().__init__(provider, None)

    def _generate(self):
        stream = mx.default_stream(mx.default_device())
        self.model_provider.load_default()
        self.ready.set()
        try:
            while not self._stop:
                try:
                    item = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if callable(item[1]):
                    try:
                        item[0].put(item[1]())
                    except Exception:
                        item[0].put(ValueError('MLX state operation refused'))
                else:
                    self._serve_single(item, stream)
        finally:
            self.state.clear()
            self.model_provider.reset()

    def control(self, function):
        response = queue.Queue()
        self.requests.put((response, function))
        result = self._await_response(response)
        if isinstance(result, Exception):
            raise result
        return result

    def prompt(self, body):
        request = api.CompletionRequest('chat', '', body['messages'], None, None)
        args = SimpleNamespace(chat_template_kwargs=None)
        return self._tokenize(self.model_provider.tokenizer, request, args)[0]

    def _serve_single(self, item, stream):
        responses, request, args = item
        self.stats['running'] = 1
        try:
            model, tokenizer = self.model_provider.model, self.model_provider.tokenizer
            prompt, _, _, initial, _ = self._tokenize(tokenizer, request, args)
            if len(prompt) + args.max_tokens > self.cli_args.context:
                raise ValueError('Prompt plus output budget exceeds the context window')
            stops, machine = self._make_state_machine(self.model_provider.model_key, tokenizer, args.stop_words)
            cache, reused = self.state.take(prompt)
            if cache is None:
                cache = make_prompt_cache(model)
            ctx = api.GenerationContext(False, tokenizer.has_thinking, None, machine, initial, prompt, reused)
            responses.put(ctx)
            if args.seed is not None:
                mx.random.seed(args.seed)
            processors = api._make_logits_processors(args)
            if reused and processors:
                prefix = mx.array(prompt[:reused])
                processors = [functools.partial(with_prefix, processor, prefix) for processor in processors]
            started, first, generated, history = time.monotonic(), None, 0, prompt[:]

            def progress(done, total):
                if ctx._should_stop:
                    raise InterruptedError('Generation cancelled')
                self.stats.update(fill_prompt_tokens=done, fill_tokens_left=total - done)
                responses.put((done, total))

            matcher = stops.matcher()
            use_draft = isinstance(model, DraftedTarget) and getattr(request, 'use_draft', True)
            generate = dflash_stream_generate if use_draft else api.stream_generate
            iterator = generate(model, tokenizer, prompt[reused:], max_tokens=args.max_tokens,
                                sampler=api._make_sampler(args, tokenizer), logits_processors=processors,
                                prompt_cache=cache, prefill_step_size=self.cli_args.prefill_step_size,
                                prompt_progress_callback=progress, stream=stream)
            try:
                for gen in iterator:
                    generated += 1
                    if first is None:
                        first = time.monotonic()
                    finish = 'stop' if matcher.advance(gen.token) else gen.finish_reason
                    history.append(gen.token)
                    responses.put(api.Response(gen.text, gen.token, gen.logprobs[gen.token].item(), finish,
                                               api._format_top_logprobs(gen.logprobs, args.top_logprobs, tokenizer)))
                    if finish is not None or ctx._should_stop:
                        break
            finally:
                iterator.close()
            mx.eval([c.state for c in cache])
            self.state.commit(history, cache)
            self.stats['prompt_tokens_total'] += len(prompt) - reused
            self.stats['generation_tokens_total'] += generated
            self.stats.update(getattr(model, 'counters', {}))
            self.stats.update(prefill_tokens_per_second=(len(prompt) - reused) / max((first or started) - started, 1e-6),
                              decode_tokens_per_second=max(0, generated - 1) / max(time.monotonic() - (first or started), 1e-6),
                              fill_tokens_left=0)
            responses.put(None)
        except Exception as exc:
            self.state.clear()
            logging.error('MLX request failed (%s)', type(exc).__name__)
            responses.put(ValueError('MLX request failed; inspect the private worker log'))
        finally:
            self.stats['running'] = 0


def with_prefix(processor, prefix, tokens, logits):
    return processor(mx.concatenate([prefix, tokens]), logits)


class Handler(api.APIHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def handle_completion(self, request, stop_words):
        request.use_draft = self._use_draft
        return super().handle_completion(request, stop_words)

    def _authorized(self):
        try:
            self.server.keys.authenticate(self.headers)
            return True
        except GatewayError:
            self._json(401, {'error': 'Invalid API key'})
            return False

    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_OPTIONS(self):
        self._json(405, {'error': 'Method not allowed'})

    def do_GET(self):
        if not self._authorized():
            return
        engine = self.response_generator
        if self.path == '/health':
            healthy = engine.ready.is_set() and engine.is_healthy
            self._json(200 if healthy else 503, {'status': 'ok' if healthy else 'loading',
                       'runtime': 'temporary-mlx', 'dflash2': engine.cli_args.dflash_drafter is not None,
                       'context': engine.cli_args.context})
        elif self.path == '/v1/models':
            self._json(200, {'object': 'list', 'data': [{'id': engine.cli_args.name, 'object': 'model'}]})
        elif self.path == '/stats':
            self._json(200, dict(engine.stats, context=engine.cli_args.context, parallel=1,
                       memory={'physical_footprint_bytes': footprint(), 'device_active_bytes': mx.get_active_memory(),
                               'device_peak_bytes': mx.get_peak_memory(), 'device_cache_bytes': mx.get_cache_memory()}))
        else:
            self._json(404, {'error': 'Route not found'})

    def do_POST(self):
        if not self._authorized():
            return
        engine = self.response_generator
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise ValueError('A single Content-Length is required')
            length = int(lengths[0])
            if not 0 < length <= 8 * 1024**2:
                raise ValueError('Invalid request size')
            raw = self.rfile.read(length)
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            if self.path.startswith('/v1/internal/session/'):
                action = self.path.rsplit('/', 1)[-1]
                if action == 'clear':
                    result = engine.control(engine.state.clear)
                elif action in ('save', 'load'):
                    result = engine.control(lambda: getattr(engine.state, action)(body['path']))
                else:
                    self._json(404, {'error': 'Route not found'})
                    return
                self._json(200, result)
                return
            maximum = validate_body(body, engine.cli_args.name, engine.cli_args.context)
            prompt = engine.control(lambda: engine.prompt(body))
            if len(prompt) + maximum > engine.cli_args.context:
                raise ValueError('Prompt plus output budget exceeds the 128K context window')
            if self.path == '/v1/imprint/resume':
                self._json(200, engine.control(lambda: {'ram_tokens': engine.state.resumable(prompt)}))
            elif self.path == '/v1/chat/completions':
                self._use_draft = body.get('draft', True)
                body['model'] = engine.cli_args.name
                if body.get('stream_options') is not None:
                    body['stream_options'].setdefault('include_usage', False)
                raw = json.dumps(body).encode()
                self.headers.replace_header('Content-Length', str(len(raw)))
                self.rfile = io.BytesIO(raw)
                super().do_POST()
            else:
                self._json(404, {'error': 'Ripple currently supports /v1/chat/completions only'})
        except (ValueError, KeyError, TypeError, OSError):
            self._json(400, {'error': 'Invalid or unsupported Ripple request or session state'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--name', default='ripple-65b-preview')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--context', type=int, default=131072)
    parser.add_argument('--prefill-step-size', type=int, default=512)
    parser.add_argument('--dflash-drafter', type=Path)
    parser.add_argument('--dflash-exact-verify', action='store_true')
    parser.add_argument('--mlx-build-receipt', type=Path)
    args = parser.parse_args()
    os.environ.pop('MLX_RIPPLE_EXACT_VERIFY', None)
    if args.dflash_drafter is not None and not args.dflash_exact_verify:
        raise ValueError('Serving DFlash2 requires the isolated, verified MLX arithmetic path')
    runtime_identity = ''
    if args.dflash_exact_verify:
        if args.dflash_drafter is None or args.mlx_build_receipt is None:
            raise ValueError('Exact verification requires a drafter and isolated MLX build receipt')
        runtime_identity = verify_runtime(args.mlx_build_receipt)
    if importlib.metadata.version('mlx-lm') != '0.32.2':
        raise ValueError('This worker is qualified with MLX-LM 0.32.2 only')
    if not 128 <= args.context <= 131072 or args.prefill_step_size not in (128, 256, 512):
        raise ValueError('Unqualified context or prompt chunk size')
    if args.dflash_drafter is not None and args.prefill_step_size != 512:
        raise ValueError('The DFlash2 serving profile is qualified with 512-token prompt chunks')
    if (args.model / 'CONVERSION_INCOMPLETE').exists():
        raise ValueError('Incomplete model conversion')
    os.umask(0o077)
    logging.basicConfig(level=logging.WARNING)
    mx.set_memory_limit(54 * 1024**3)
    mx.set_cache_limit(128 * 1024**2)
    config = json.loads((args.model / 'config.json').read_text())
    model_args = ModelArgs.from_dict(config)
    schema = SimpleNamespace(**vars(model_args))
    schema.cache_windows = [model_args.sliding_window] * model_args.num_hidden_layers
    if args.dflash_drafter is not None:
        draft = DraftArgs.from_dict(json.loads((args.dflash_drafter / 'config.json').read_text()))
        schema.layer_types = model_args.layer_types + draft.layer_types
        schema.cache_windows += [draft.sliding_window] * draft.num_hidden_layers
    identity = build_identity(args.model, args.dflash_drafter) + runtime_identity
    state = ResidentState(args.snapshot_dir, identity, schema, args.context)
    cli = SimpleNamespace(**vars(args), adapter_path=None, draft_model=None, pipeline=False,
                          trust_remote_code=True, chat_template=None, chat_template_args={},
                          use_default_chat_template=False, num_draft_tokens=0, max_tokens=4096,
                          temp=0.0, top_p=1.0, top_k=0, min_p=0.0, allowed_origins=[], kv_bits=None)
    engine = Engine(LocalProvider(cli), state)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, engine))
    server.keys = KeyStore((KeyFile('gateway', args.api_key_file),))

    def guard():
        while True:
            time.sleep(0.5)
            if footprint() > 56 * 1024**3:
                os._exit(75)

    threading.Thread(target=guard, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        engine.stop_and_join()


if __name__ == '__main__':
    main()
