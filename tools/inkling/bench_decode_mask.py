#!/usr/bin/env python3
"""Compare the pinned Inkling worker's production decode mask with upstream on M5 Max."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace


GIB = 1024 ** 3
PREFILL = 512
OUTPUT_TOKENS = 256
REPEATS = 3
CONTEXT_LIMIT = 131072
MEMORY_FLOOR_GIB = 8
MEMORY_LIMIT_GIB = 48
CACHE_LIMIT_MIB = 256
ATTENTION_COUNT = 24
SEED = 1234


class ExactnessError(RuntimeError):
    """Two runs generated different tokens for the same input."""


def git_revision(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("provide a full 40-character commit SHA")
    return value.lower()


def context_lengths(value):
    try:
        lengths = tuple(int(length) for length in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("provide comma-separated input lengths") from error
    if (
        len(set(lengths)) != len(lengths)
        or any(not 1 <= length <= CONTEXT_LIMIT - OUTPUT_TOKENS for length in lengths)
    ):
        raise argparse.ArgumentTypeError("lengths must be unique and fit the 128K context with 256 outputs")
    return lengths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local 8-bit checkpoint directory")
    parser.add_argument("--worker-source", required=True, type=Path, help="directory containing decode_mask.py and request_state.py")
    parser.add_argument("--output", required=True, type=Path, help="JSON receipt path")
    parser.add_argument("--base-commit", required=True, type=git_revision)
    parser.add_argument("--head-commit", required=True, type=git_revision)
    parser.add_argument("--model-revision", required=True, type=git_revision)
    parser.add_argument("--lengths", type=context_lengths, default=(2000, 8000, 32000, 64000),
                        help="comma-separated input lengths (default: 2000,8000,32000,64000)")
    args = parser.parse_args(argv)
    if not (args.model / "config.json").is_file():
        parser.error("--model must name a local checkpoint containing config.json")
    if not all((args.worker_source / name).is_file() for name in ("decode_mask.py", "request_state.py")):
        parser.error("--worker-source must contain decode_mask.py and request_state.py")
    if args.output.is_dir():
        parser.error("--output must be a file path")
    return args


def available_gib():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
    page_size = re.search(r"page size of (\d+) bytes", raw)
    counters = dict(re.findall(r"^(Pages (?:free|inactive)):\s*(\d+)\.", raw, re.M))
    if page_size is None or counters.keys() != {"Pages free", "Pages inactive"}:
        raise RuntimeError("vm_stat did not report free and inactive pages")
    return (int(counters["Pages free"]) + int(counters["Pages inactive"])) * int(page_size[1]) / GIB


def write_receipt(output, report):
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, encoding="utf-8", delete=False) as temp:
        json.dump(report, temp, indent=2, allow_nan=False)
        temp.write("\n")
    os.replace(temp.name, output)


def install_bindings(model, decode_mask):
    attentions = [layer.self_attn for layer in model.model.language_model.layers]
    if len(attentions) != ATTENTION_COUNT or len({id(attention) for attention in attentions}) != ATTENTION_COUNT:
        raise RuntimeError("expected 24 distinct Inkling attention modules")
    originals = [attention._attention for attention in attentions]
    installed = decode_mask.install(model)
    masked = [attention._attention for attention in attentions]
    if installed != ATTENTION_COUNT or any(
        binding.__self__ is not attention
        or binding.__func__ is not decode_mask._attention
        or original.__func__ is not decode_mask._ORIGINAL_ATTENTION
        for attention, original, binding in zip(attentions, originals, masked)
    ):
        raise RuntimeError("production mask did not bind all 24 attention modules")
    return list(zip(attentions, originals, masked))


def generation_args(count):
    return SimpleNamespace(
        seed=SEED,
        max_tokens=count,
        top_logprobs=-1,
        stop_words=[],
        logprobs=False,
        sampling=SimpleNamespace(
            temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
            xtc_probability=0.0, xtc_threshold=0.1,
        ),
        logits=SimpleNamespace(
            logit_bias=None, repetition_penalty=0.0, repetition_context_size=20,
            presence_penalty=0.0, presence_context_size=20,
            frequency_penalty=0.0, frequency_context_size=20,
        ),
    )


def run_request(model, tokenizer, request_state, mx, response_type, stream, prompt, count):
    state = request_state(
        model, tokenizer, prompt, None, generation_args(count),
        SimpleNamespace(_should_stop=False), PREFILL,
    )
    tokens = []
    first = None
    finish = None
    mx.reset_peak_memory()
    started = time.perf_counter()
    try:
        while not state.done:
            with mx.stream(stream):
                result = state.advance()
            if isinstance(result, response_type):
                emitted = time.perf_counter()
                if first is None:
                    first = emitted
                tokens.append(result.token)
                finish = result.finish_reason
        if len(tokens) != count or finish != "length":
            raise RuntimeError(f"expected {count} emitted tokens and a length finish, got {len(tokens)}")
        prefill = first - started
        decode = emitted - first
        packed = struct.pack(f"<{len(tokens)}I", *tokens)
        return tuple(tokens), {
            "input_tokens": len(prompt),
            "output_tokens": len(tokens),
            "finish_reason": finish,
            "prefill_s": prefill,
            "decode_s": decode,
            "prompt_tps": len(prompt) / prefill,
            "generation_tps": (len(tokens) - 1) / decode,
            "token_sha256": hashlib.sha256(packed).hexdigest(),
            "mlx_peak_gib": mx.get_peak_memory() / GIB,
        }
    finally:
        state.close()
        mx.synchronize(stream)
        mx.clear_cache()


def prompt_tokens(length):
    return [1000 + (i * 79) % 30000 for i in range(length)]


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MLX_ENABLE_TF32"] = "0"
    report = {
        "status": "starting",
        "base_commit": args.base_commit,
        "head_commit": args.head_commit,
        "model_revision": args.model_revision,
        "mode_order": "baseline/mask and mask/baseline on alternating pairs",
        "lengths": args.lengths,
        "prefill_step_size": PREFILL,
        "output_tokens": OUTPUT_TOKENS,
        "repeats_per_length": REPEATS,
        "paired_repeats_verified": 0,
        "seed": SEED,
        "temperature": 0,
        "eos_ignored_for_benchmark": True,
        "memory_floor_gib": MEMORY_FLOOR_GIB,
        "mlx_memory_limit_gib": MEMORY_LIMIT_GIB,
        "mlx_cache_limit_mib": CACHE_LIMIT_MIB,
        "mlx_env": {name: os.environ.get(name) for name in (
            "MLX_ENABLE_TF32", "MLX_METAL_FAST_SYNCH", "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
        )},
        "timing_scope": "Prompt through first emitted token, then first-to-last emitted token.",
        "cells": [],
    }
    stop = threading.Event()
    receipt_lock = threading.Lock()

    def save():
        write_receipt(args.output, report)

    def monitor():
        while not stop.wait(1):
            try:
                free = available_gib()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                free = None
            with receipt_lock:
                if stop.is_set():
                    return
                if free is not None:
                    report["minimum_available_gib"] = min(report["minimum_available_gib"], free)
                    if free >= MEMORY_FLOOR_GIB:
                        continue
                    report["status"] = "memory_guard_stopped"
                else:
                    report["status"] = "memory_guard_failed"
                try:
                    save()
                finally:
                    os._exit(3)

    save()
    monitor_thread = None
    phase = "memory check"
    try:
        initial_available = available_gib()
        report["minimum_available_gib"] = initial_available
        if initial_available < MEMORY_FLOOR_GIB:
            report["status"] = "memory_guard_stopped"
            return 3
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        phase = "runtime and checkpoint"
        mlx_version, mlx_lm_version = version("mlx"), version("mlx-lm")
        if (mlx_version, mlx_lm_version) != ("0.32.4", "0.32.3.dev26+g0c4c99192"):
            raise RuntimeError("use MLX 0.32.4 and MLX-LM 0c4c99192")
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generation_stream
        from mlx_lm.server import Response

        source = args.worker_source.resolve()
        sys.path.insert(0, str(source))
        import decode_mask
        import request_state

        modules = (decode_mask, request_state)
        if any(Path(module.__file__).resolve().parent != source for module in modules):
            raise RuntimeError("worker modules must come from --worker-source")
        module_hashes = {
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in modules
        }
        device = mx.metal.device_info().get("device_name") if mx.metal.is_available() else None
        if device != "Apple M5 Max" or mx.default_device() != mx.gpu:
            raise RuntimeError("this benchmark requires the Apple M5 Max GPU")
        mx.set_memory_limit(MEMORY_LIMIT_GIB * GIB)
        mx.set_cache_limit(CACHE_LIMIT_MIB * 1024 ** 2)
        config_path = args.model / "config.json"
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if config.get("quantization", {}).get("bits") != 8:
            raise RuntimeError("use the 8-bit Inkling checkpoint")
        model, tokenizer = load(str(args.model.resolve()))
        if model.args.text.loop_config.get("num_passes") != 3:
            raise RuntimeError("use the fixed-three-pass Inkling model")
        tokenizer.eos_token_ids = set()
        bindings = install_bindings(model, decode_mask)
        with receipt_lock:
            report.update(
                status="running",
                mlx=mlx_version,
                mlx_lm=mlx_lm_version,
                metal_device=device,
                model_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                worker_source_sha256=module_hashes,
                weight_quantization_bits=8,
                loop_passes=3,
                installed_attention_modules=len(bindings),
            )
            save()

        def run_mode(mode, prompt, count):
            for attention, original, masked in bindings:
                attention._attention = masked if mode == "mask" else original
            return run_request(
                model, tokenizer, request_state.RequestState, mx, Response,
                generation_stream, prompt, count,
            )

        phase = "warmup"
        warm_prompt = prompt_tokens(128)
        baseline, _ = run_mode("baseline", warm_prompt, 8)
        masked, _ = run_mode("mask", warm_prompt, 8)
        if masked != baseline:
            raise ExactnessError("warmup tokens differed")
        pair_index = 0
        for length in args.lengths:
            reference = None
            prompt = prompt_tokens(length)
            for repeat in range(1, REPEATS + 1):
                modes = ("baseline", "mask") if pair_index % 2 == 0 else ("mask", "baseline")
                for mode in modes:
                    phase = f"{length} input tokens, repeat {repeat}, {mode}"
                    tokens, row = run_mode(mode, prompt, OUTPUT_TOKENS)
                    if reference is None:
                        reference = tokens
                    exact = tokens == reference
                    row.update(mode=mode, repeat=repeat, exact_to_first=exact)
                    with receipt_lock:
                        report["cells"].append(row)
                        save()
                    print(
                        f"{mode:8} {length:6} repeat {repeat} "
                        f"{row['prompt_tps']:7.1f} prompt tok/s {row['generation_tps']:5.2f} decode tok/s "
                        f"{row['token_sha256'][:12]}",
                        flush=True,
                    )
                    if not exact:
                        raise ExactnessError(f"generated tokens differed at {phase}")
                pair_index += 1
                with receipt_lock:
                    report["paired_repeats_verified"] = pair_index
        stop.set()
        report["status"] = "passed"
        return 0
    except ExactnessError as error:
        stop.set()
        report.update(status="failed_exactness", failure_phase=phase)
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        stop.set()
        report.update(status="interrupted", failure_phase=phase)
        return 130
    except Exception as error:
        stop.set()
        report.update(status="failed", failure_phase=phase, error_type=type(error).__name__)
        print(f"{phase}: {error}", file=sys.stderr)
        return 1
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join()
        save()


if __name__ == "__main__":
    raise SystemExit(main())
