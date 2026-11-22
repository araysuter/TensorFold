"""Arguments and uninstrumented serving-worker reference for the DSpark benchmark."""

import argparse
import hashlib
import importlib.util
import struct
import time
import types
from contextlib import contextmanager
from pathlib import Path


def parse_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--verify-width", type=int, choices=(2,4,8), default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--width", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--draft-bits", type=int, choices=(2, 3, 4, 5, 6, 8, 16), default=16)
    parser.add_argument("--cache-projected", action="store_true")
    parser.add_argument("--trim-head", action="store_true")
    parser.add_argument("--device-markov", action="store_true")
    parser.add_argument("--cases", default="prose,code,natural-notes-8k")
    parser.add_argument("--worker-source", type=Path, required=True, help="Deployed directory containing request_state.py")
    parser.add_argument("--historical-plain", action="store_true", help="Also time the synchronous historical comparator")
    args = parser.parse_args()
    if (
        not 1 <= args.tokens <= 1024
    ):
        parser.error("Positive prompt lengths plus output must fit 131072 tokens")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must name an existing local checkpoint directory")
    if not args.draft.is_file():
        parser.error("--draft must name an existing local safetensors file")
    if args.output.is_dir():
        parser.error("--output must be a file path")
    args.model = args.model.resolve()
    args.draft = args.draft.resolve()
    return args


def load_worker(worker_path):
    spec = importlib.util.spec_from_file_location("inkling_worker_reference", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def make_worker_reference(model, tok, worker, uninstrumented):
    import mlx.core as mx
    from mlx_lm import server as api
    from mlx_lm.generate import generation_stream

    @contextmanager
    def ordinary_model():
        wrapped = [(m, k, getattr(m, k)) for m, k, original in uninstrumented]
        for m, k, original in uninstrumented:
            setattr(m, k, original)
        try:
            yield
        finally:
            for m, k, instrumented in wrapped:
                setattr(m, k, instrumented)

    def worker_reference(ids, maximum):
        request_args = types.SimpleNamespace(
            seed=1234, max_tokens=maximum, logprobs=False, top_logprobs=-1, stop_words=[],
            sampling=api.SamplingArguments(0.0, 1.0, 0, 0.0, 0.0, 0.1),
            logits=api.LogitsProcessorArguments(None, 0.0, 20, 0.0, 20, 0.0, 20),
        )
        with ordinary_model(), mx.stream(generation_stream):
            state = worker.RequestState(model, tok, ids, None, request_args,
                                        types.SimpleNamespace(_should_stop=False), 512)
            outputs, first = [], None
            mx.synchronize(generation_stream)
            started = time.monotonic()
            try:
                while not state.done:
                    value = state.advance()
                    if isinstance(value, api.Response):
                        if first is None:
                            first = time.monotonic()
                        outputs.append(int(value.token))
                end = time.monotonic()
            finally:
                state.close()
                mx.synchronize(generation_stream)
        return dict(
            tokens=outputs, output_tokens=len(outputs), prefill_seconds=first - started,
            prompt_tps=len(ids)/(first-started),
            decode_tps=(len(outputs)-1)/(end-first) if len(outputs)>1 else None,
            finish_reason=value.finish_reason,
            token_sha256=hashlib.sha256(b"".join(struct.pack("<I",t) for t in outputs)).hexdigest(),
        )

    return worker_reference
