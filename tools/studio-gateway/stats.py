"""Read-only numeric stats compatibility for native engines without /stats.

The pinned native port publishes Prometheus counters and /memory. Its aggregate
rates cover finished requests since model load, not the currently running
prompt. Unknown metrics, labels, paths, errors, and nonfinite values never pass
through the public API.
"""
from __future__ import annotations

import http.client
import json
import math


LIMIT = 1024 * 1024
COUNTERS = {
    "running", "waiting", "decode_tokens_per_second", "prefill_tokens_per_second",
    "fill_prompt_tokens", "fill_tokens_left", "generation_tokens_total", "prompt_tokens_total",
}
MEMORY = {
    "physical_footprint_bytes", "lifetime_peak_physical_footprint_bytes", "device_active_bytes",
    "device_cache_bytes", "device_peak_bytes", "pinned_memory_bytes", "rss_bytes", "active_bytes",
    "peak_bytes", "cache_bytes", "process_rss_bytes", "used_bytes", "free_bytes", "limit_bytes",
    "prompt_cache_budget_bytes", "ram_bytes", "memory_cap_bytes",
}
SLOT_NUMBERS = {
    "slot", "last_used", "prompt_tokens", "cached_tokens", "generated_tokens", "prompt_tps",
    "decode_tps", "rate_at", "ram_bytes", "ssd_bytes", "checkpoint_tokens", "shared_checkpoint_tokens",
    "shared_ram", "shared_ssd",
}
SLOT_STATES = {
    "empty", "queued", "restoring", "prefilling", "generating", "saving", "ram", "ssd",
    "uncached", "error", "shared",
}
PROMETHEUS = {
    "requests_running", "num_requests_running", "requests_waiting", "num_requests_waiting",
    "generation_tokens_total", "prompt_tokens_total", "prompt_tokens_cached_total",
    "request_decode_seconds_sum", "request_decode_time_seconds_sum", "request_prefill_seconds_sum",
    "request_prefill_time_seconds_sum", "process_footprint_bytes", "process_footprint_peak_bytes",
    "device_memory_bytes", "device_memory_peak_bytes", "pinned_memory_bytes",
}


def number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def sanitize_stats(value: object) -> dict:
    """Accept collector-compatible numeric fields from newer native /stats."""
    if not isinstance(value, dict):
        return {}
    out = {key: value[key] for key in COUNTERS if number(value.get(key))}
    memory = value.get("memory", {})
    if isinstance(memory, dict):
        clean = {key: memory[key] for key in MEMORY if number(memory.get(key))}
        if clean:
            out["memory"] = clean
    if (isinstance(value.get("slots"), list) and number(value.get("parallel"))
            and number(value.get("context"))):
        slots = []
        for row in value["slots"][:256]:
            if (not isinstance(row, dict) or type(row.get("slot")) is not int
                    or not 1 <= row["slot"] <= 256 or not isinstance(row.get("state"), str)
                    or row["state"] not in SLOT_STATES):
                continue
            clean = {key: row[key] for key in SLOT_NUMBERS if number(row.get(key))}
            clean["state"] = row["state"]
            slots.append(clean)
        out.update(slots=slots, parallel=value["parallel"], context=value["context"])
    if number(value.get("tracking_version")):
        out["tracking_version"] = value["tracking_version"]
    return out


def parse_prometheus(raw: bytes) -> dict[str, float | int]:
    result = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        # Only unlabeled scalar metrics are needed. In particular, never return
        # API key labels or per-key request series from this diagnostic route.
        parts = line.split()
        if len(parts) not in {2, 3} or "{" in parts[0] or parts[0].startswith("#"):
            continue
        key = parts[0]
        if key.startswith("tensorfold:"):
            key = key[len("tensorfold:"):]
        elif key.startswith("tensorfold_"):
            key = key[len("tensorfold_"):]
        else:
            continue
        if key not in PROMETHEUS:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if number(value):
            result[key] = int(value) if value.is_integer() else value
    return result


def compatibility_stats(metrics: bytes, memory: object) -> dict:
    values = parse_prometheus(metrics)
    out = {"tracking_version": 2}
    for target, alternatives in {
        "running": ("requests_running", "num_requests_running"),
        "waiting": ("requests_waiting", "num_requests_waiting"),
        "generation_tokens_total": ("generation_tokens_total",),
        "prompt_tokens_total": ("prompt_tokens_total",),
    }.items():
        found = next((values[name] for name in alternatives if name in values), None)
        if found is not None:
            out[target] = found
    generated = values.get("generation_tokens_total")
    decode_seconds = values.get("request_decode_seconds_sum", values.get("request_decode_time_seconds_sum"))
    prompt = values.get("prompt_tokens_total")
    prefill_seconds = values.get("request_prefill_seconds_sum", values.get("request_prefill_time_seconds_sum"))
    if generated is not None and decode_seconds:
        rate = generated / decode_seconds
        if number(rate):
            out["decode_tokens_per_second"] = rate
    if prompt is not None and prefill_seconds:
        tokens = max(0, prompt - values.get("prompt_tokens_cached_total", 0))
        rate = tokens / prefill_seconds
        if number(rate):
            out["prefill_tokens_per_second"] = rate
    if "decode_tokens_per_second" in out or "prefill_tokens_per_second" in out:
        # The collector ignores this explanatory metadata and records numbers.
        out["rate_scope"] = "completed_requests_since_model_load"
    clean_memory = {}
    for source, target in {
        "process_footprint_bytes": "physical_footprint_bytes",
        "process_footprint_peak_bytes": "lifetime_peak_physical_footprint_bytes",
        "device_memory_bytes": "device_active_bytes", "device_memory_peak_bytes": "device_peak_bytes",
        "pinned_memory_bytes": "pinned_memory_bytes",
    }.items():
        if source in values:
            clean_memory[target] = values[source]
    if isinstance(memory, dict):
        for section, mapping in {
            "process": {"physical_footprint_bytes": "physical_footprint_bytes",
                        "lifetime_peak_physical_footprint_bytes": "lifetime_peak_physical_footprint_bytes"},
            "device": {"active_bytes": "device_active_bytes", "cache_bytes": "device_cache_bytes",
                       "peak_bytes": "device_peak_bytes"},
            "prompt_cache_plan": {"budget_bytes": "prompt_cache_budget_bytes", "ram_bytes": "ram_bytes",
                                  "cap_bytes": "memory_cap_bytes"},
        }.items():
            row = memory.get(section)
            if isinstance(row, dict):
                for source, target in mapping.items():
                    if number(row.get(source)):
                        clean_memory[target] = row[source]
    if clean_memory:
        out["memory"] = clean_memory
    return out


def _get(backend, path: str) -> tuple[int, bytes]:
    # Monitoring must not occupy a handler for a reasoning request's hour-long
    # timeout. Three fixed read-only calls fit inside the collector's 4s poll.
    conn = backend._connection(timeout=1.0)
    try:
        conn.request("GET", path, headers=backend._headers())
        response = conn.getresponse()
        raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            return 502, b""
        return response.status, raw
    finally:
        conn.close()


def read_stats(backend) -> dict:
    try:
        status, raw = _get(backend, "/stats")
    except (OSError, http.client.HTTPException):
        return {}
    if status == 200:
        try:
            return sanitize_stats(json.loads(raw))
        except (ValueError, TypeError, RecursionError):
            return {}
    if status not in {404, 501}:
        return {}
    metrics, memory = b"", {}
    try:
        status, raw = _get(backend, "/metrics")
        if status == 200:
            metrics = raw
    except (OSError, http.client.HTTPException):
        pass
    try:
        status, raw = _get(backend, "/memory")
        if status == 200:
            memory = json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException, RecursionError):
        pass
    return compatibility_stats(metrics, memory)
