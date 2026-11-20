#!/usr/bin/env python3
"""Authenticated Inkling MLX worker under the existing Studio gateway supervisor."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'  # Before importing MLX: qualified FP32 path.

import argparse
import functools
import hashlib
import importlib.metadata
import io
import json
import logging
from pathlib import Path
import struct
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm import server as api
from request_policy import validate_request
from request_state import RequestState
from mlx_lm.generate import generation_stream
from response_format import configure as configure_response_format

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio-gateway'))
from config import KeyFile
from private_state import KeyStore
from errors import GatewayError


class Provider(api.ModelProvider):
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        if model_path not in ('default_model', self.cli_args.name) or adapter_path or draft_model_path not in (None, 'default_model'):
            raise ValueError('Only the configured local model is available')
        return super().load('default_model', None, None)

    def _load(self, *args, **kwargs):
        super()._load(*args, **kwargs)
        configure_response_format(self.tokenizer)
        if self.cli_args.benchmark_ignore_eos:
            self.tokenizer.eos_token_ids = set()
        self.ready = True


class Engine(api.ResponseGenerator):
    def __init__(self, provider):
        self.receipts = threading.local()
        super().__init__(provider, None)

    def _generate(self):
        self.model_provider.load_default()
        tasks = []
        pending = None
        try:
            while not self._stop:
                while len(tasks) < 2:
                    item = pending or self._next_request(timeout=0.1 if not tasks else None)
                    pending = None
                    if item is None:
                        break
                    output, request, args = item
                    images = any(api._image_parts(request.messages))
                    if tasks and (images or any(t[2] for t in tasks)):
                        pending = item
                        break
                    try:
                        model, tokenizer = self.model_provider.load(args.model.model, args.model.adapter, args.model.draft)
                        prompt, _, _, initial, vision = self._tokenize(tokenizer, request, args)
                        ctx = api.GenerationContext(tokenizer.has_tool_calling, tokenizer.has_thinking,
                                tokenizer.tool_parser, (api.TextStateMachine.from_tokenizer(tokenizer)
                                if request.request_type == 'chat' else api.TextStateMachine({})), initial,
                                prompt, prompt_cache_count=0)
                        state = RequestState(model, tokenizer, prompt, vision, args, ctx, self.cli_args.prefill_step_size)
                        output.put(ctx)
                        tasks.append((state, output, images))
                    except Exception:
                        logging.error('Inkling request preparation failed')
                        output.put(ValueError('Invalid or unsupported Inkling request'))
                    if images:
                        break
                for task in tasks[:]:
                    state, output, _ = task
                    try:
                        with mx.stream(generation_stream):
                            result = state.advance()
                        if result is not None:
                            output.put(result)
                        if state.done:
                            output.put(None)
                            state.close()
                            tasks.remove(task)
                    except Exception:
                        logging.exception('Inkling generation failed')
                        output.put(ValueError('Inkling generation failed'))
                        state.close()
                        tasks.remove(task)
        finally:
            for state, _, _ in tasks:
                state.close()
            self.model_provider.reset()

    def _tokenize(self, tokenizer, request, args):
        if request.request_type == 'text' and isinstance(request.prompt, list):
            prompt = request.prompt
            if any(type(t) is not int or not 0 <= t < len(tokenizer) for t in prompt):
                raise ValueError('Token outside vocabulary')
            result = prompt, [prompt], ['assistant'], 'normal', None
        else:
            result = super()._tokenize(tokenizer, request, args)
        if len(result[0]) + args.max_tokens > self.cli_args.context:
            raise ValueError('Expanded prompt plus output budget exceeds the context window')
        return result

    def generate(self, request, generation_args, progress_callback=None):
        start = time.monotonic()
        ctx, responses = super().generate(request, generation_args, progress_callback=progress_callback)
        receipt = {'tokens': 0, 'first': None, 'last': None, 'hash': hashlib.sha256(), 'start': start}
        self.receipts.current = receipt
        def measured():
            for response in responses:
                now = time.monotonic()
                receipt['first'] = receipt['first'] or now
                receipt['last'] = now
                receipt['tokens'] += 1
                receipt['hash'].update(struct.pack('<I', response.token))
                yield response
        return ctx, measured()

    def receipt(self):
        r = self.receipts.current
        first, last = r['first'], r['last']
        return dict(token_sha256=r['hash'].hexdigest(),
                    prefill_seconds=max((first or r['start']) - r['start'], 1e-6),
                    tokens_per_second=max(r['tokens'] - 1, 0) / max((last or 0) - (first or 0), 1e-6),
                    runtime='mlx-inkling', drafter=False,
                    timing_scope='worker queue, tokenization and prefill to first token')


class Handler(api.APIHandler):
    protocol_version = 'HTTP/1.0'
    def log_message(self, *args):
        pass
    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True
    def _authorized(self):
        try:
            self.server.keys.authenticate(self.headers)
            return True
        except GatewayError:
            self._json(401, {'error': 'Invalid API key'})
            return False
    def do_OPTIONS(self):
        self._json(405, {'error': 'Method not allowed'})
    def do_GET(self):
        if not self._authorized():
            return
        e = self.response_generator
        if self.path == '/health':
            ready = getattr(e.model_provider, 'ready', False) and e.is_healthy
            self._json(200 if ready else 503, {'status': 'ok' if ready else 'loading',
                       'context': e.cli_args.context, 'parallel_text': 2, 'parallel_images': 1})
        elif self.path == '/v1/models':
            self._json(200, {'object': 'list', 'data': [{'id': e.cli_args.name, 'object': 'model'}]})
        elif self.path == '/stats':
            self._json(200, dict(device_active_bytes=mx.get_active_memory(),
                                device_peak_bytes=mx.get_peak_memory(), device_cache_bytes=mx.get_cache_memory()))
        else:
            self._json(404, {'error': 'Route not found'})
    def do_POST(self):
        if not self._authorized():
            return
        if self.path not in ('/v1/completions', '/v1/chat/completions'):
            self._json(404, {'error': 'Route not found'})
            return
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise ValueError('A single Content-Length is required')
            size = int(lengths[0])
            if not 0 < size <= 8 * 1024**2:
                raise ValueError('Invalid body size')
            body = json.loads(self.rfile.read(size))
            validate_request(body, self.response_generator.cli_args.name,
                             self.response_generator.cli_args.context,
                             self.response_generator.cli_args.benchmark_ignore_eos)
            raw = json.dumps(body).encode()
            self.rfile = io.BytesIO(raw)
            self.headers.replace_header('Content-Length', str(len(raw)))
        except (ValueError, TypeError, KeyError, OSError):
            self._json(400, {'error': 'Invalid or unsupported Inkling request'})
            return
        # The pinned upstream logs complete malformed bodies; validate JSON above
        # and never enable DEBUG logging on this authenticated worker.
        super().do_POST()
    def handle_completion(self, request, stop_words):
        request.reference_plain = self.body.get('draft') is False
        return super().handle_completion(request, stop_words)
    def completion_usage_response(self, *args, **kwargs):
        response = super().completion_usage_response(*args, **kwargs)
        response['tensorfold'] = self.response_generator.receipt()
        return response
    def generate_response(self, text, finish_reason, *args, **kwargs):
        response = super().generate_response(text, finish_reason, *args, **kwargs)
        if finish_reason is not None:
            response['tensorfold'] = self.response_generator.receipt()
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', default='inkling-2-mini')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--context', type=int, default=131072)
    parser.add_argument('--prefill-step-size', type=int, default=512)
    parser.add_argument('--benchmark-ignore-eos', action='store_true')
    args = parser.parse_args()
    if importlib.metadata.version('mlx') != '0.32.4' or importlib.metadata.version('mlx-lm') != '0.32.3.dev26+g0c4c99192':
        raise ValueError('Use the qualified MLX 0.32.4 / MLX-LM PR 2086 at 0c4c99192')
    if not 128 <= args.context <= 131072 or args.prefill_step_size not in (256, 512, 1024, 2048):
        raise ValueError('Unqualified context or prefill size')
    os.umask(0o077)
    logging.basicConfig(level=logging.WARNING)
    mx.set_memory_limit(48 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    cli = SimpleNamespace(**vars(args), adapter_path=None, draft_model=None, pipeline=False,
            trust_remote_code=False, chat_template=None, chat_template_args={'reasoning_effort': 'none'},
            use_default_chat_template=False, num_draft_tokens=0, max_tokens=4096,
            temp=0.0, top_p=1.0, top_k=0, min_p=0.0, allowed_origins=[], kv_bits=None,
            kv_group_size=64, quantized_kv_start=0, prompt_cache_bytes=None,
            decode_concurrency=2, prompt_concurrency=1)
    engine = Engine(Provider(cli))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, engine))
    server.keys = KeyStore((KeyFile('gateway', args.api_key_file),))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.stop_and_join()


if __name__ == '__main__':
    main()
