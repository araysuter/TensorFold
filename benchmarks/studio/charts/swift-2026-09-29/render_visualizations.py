"""Reproduce the updated Swift sheet and its historical solo comparison.

This only reads saved benchmark results and renders figures; it runs no models.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tensorfold-chart-cache")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONTEXTS = [20000, 40000, 60000, 80000, 100000, 128000]
BLUE, ORANGE, PINK = "#2563eb", "#e77b24", "#db2777"
CURRENT = ["benchmark-results-swift-325da38/results.json",
           "benchmark-results-swift-100k-two/results.json",
           "benchmark-results-swift-128k-solo/results.json",
           "benchmark-results-swift-128k-dual/results.json"]
PREVIOUS = ["benchmarks/studio/studio-benchmark-resumed/results.json",
            "benchmarks/studio/studio-benchmark-100k/results.json",
            "benchmarks/studio/studio-benchmark-128k-higher-memory/results.json"]
MATCH_FIELDS = ["model", "tokenizer", "revision", "output_tokens", "no_draft",
                "context_limit", "base_url"]


def combine(paths, previous=False):
    rows, batches, sources = [], [], []
    config = None
    for path in paths:
        raw = (ROOT / path).read_bytes()
        data = json.loads(raw)
        if config is None:
            config = dict(data["config"])
        assert all(data["config"].get(k) == config.get(k) for k in MATCH_FIELDS), path
        sources.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                        "config": data["config"]})
        for key, target in (("requests", rows), ("batches", batches)):
            for record in data[key]:
                if previous and "higher-memory" not in path and record["context_tokens"] == 128000:
                    continue  # Match the already published Swift figure's selection.
                target.append(dict(record, source=path))
    config.update(contexts=CONTEXTS, concurrency=[1, 2], reps=None)
    return {"config": config, "requests": rows, "batches": batches,
            "sources": sources, "repetitions": "Variable; inspect each source and group"}


def validate(data):
    seen = set()
    for row in data["requests"]:
        key = tuple(row[k] for k in ("source", "context_tokens", "concurrency", "repetition", "user"))
        assert key not in seen, key
        seen.add(key)
        assert row["status"] == "ok" and row["actual_prompt_tokens"] == row["context_tokens"], key
        assert row["cached_tokens"] == 0 and 0 < row["output_tokens"] <= 1024, key
        expected = row["actual_prompt_tokens"] / row["queue_prefill_seconds"]
        assert math.isclose(row["effective_input_tps"], expected, rel_tol=1e-9), key
    for batch in data["batches"]:
        subset = [r for r in data["requests"] if all(r[k] == batch[k] for k in
                  ("source", "context_tokens", "concurrency", "repetition"))]
        assert len(subset) == batch["concurrency"]
        expected = sum(r["output_tokens"] for r in subset) / batch["elapsed_seconds"]
        assert math.isclose(batch["aggregate_tps"], expected, rel_tol=1e-9)
    for n in CONTEXTS:
        for c in (1, 2):
            assert any(r["context_tokens"] == n and r["concurrency"] == c for r in data["requests"])


def samples(data, metric, context, concurrency):
    records = data["batches"] if metric == "aggregate_tps" else data["requests"]
    return [r[metric] for r in records if r["context_tokens"] == context
            and r["concurrency"] == concurrency and r.get("status", "ok") == "ok"
            and r.get(metric) is not None and math.isfinite(r[metric])]


def stats(data, metric, concurrency):
    groups = [samples(data, metric, n, concurrency) for n in CONTEXTS]
    return ([statistics.median(g) for g in groups],
            [min(g) for g in groups], [max(g) for g in groups])


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#bcc5cf", "text.color": "#172b42",
        "axes.labelcolor": "#34465b", "xtick.color": "#536377", "ytick.color": "#536377"})


def axis_style(axis, title, unit, numeric=False):
    axis.set_title(title, loc="left", pad=16)
    axis.set_ylabel(unit)
    axis.set_xlabel("Input context (tokens)")
    x = [n / 1000 for n in CONTEXTS] if numeric else list(range(len(CONTEXTS)))
    axis.set_xticks(x, [f"{n // 1000}K" for n in CONTEXTS])
    axis.grid(axis="y", color="#e7ebf0", lw=.8)
    axis.set_axisbelow(True)
    axis.set_xlim((15, 133) if numeric else (-.3, len(CONTEXTS) - .65))
    return x


def draw_series(axis, x, values, low, high, color, marker="o", linestyle="-"):
    axis.errorbar(x, values, yerr=[[v - l for v, l in zip(values, low)],
                                 [h - v for v, h in zip(values, high)]],
                  color=color, lw=2.4, marker=marker, ms=6, capsize=3,
                  elinewidth=1, linestyle=linestyle)


def save(fig, filename):
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"{filename}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def standard(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.065, right=.96, top=.81, bottom=.19, hspace=.43, wspace=.22)
    fig.text(.065, .956, "TensorFold · Swift 1.5 4-bit", fontsize=24, weight="bold")
    fig.text(.065, .922, "M5 Max Mac Studio · updated configuration · 1 vs 2 concurrent requests",
             fontsize=15, color="#536377")
    fig.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", lw=2, label="1 request"),
                        Line2D([0], [0], color=ORANGE, marker="s", lw=2,
                               label="2 simultaneous submissions (may queue)")],
               loc="upper left", bbox_to_anchor=(.06, .90), frameon=False, ncol=2, fontsize=12)
    metrics = [("decode_tps", "Generation speed per request", "Tokens / second"),
               ("effective_input_tps", "Effective prompt throughput¹", "Input tokens / second"),
               ("ttft_seconds", "Time to first token", "Seconds"),
               ("aggregate_tps", "Combined end-to-end output throughput²", "Output tokens / second")]
    for axis, (metric, title, unit) in zip(axes.flat, metrics):
        xx = axis_style(axis, title, unit)
        maximum = 0
        for c, color in ((1, BLUE), (2, ORANGE)):
            values, low, high = stats(data, metric, c)
            x = [i + (c - 1.5) * .035 for i in xx]
            draw_series(axis, x, values, low, high, color, "o" if c == 1 else "s")
            maximum = max(maximum, *high)
            other_values = stats(data, metric, 2 if c == 1 else 1)[0]
            for i, v, other in zip(x, values, other_values):
                offset = 11 if v >= other else -18
                axis.annotate(f"{v:.1f}", (i, v), xytext=(0, offset), textcoords="offset points",
                              ha="center", color=color, fontsize=10, weight="bold")
        axis.set_ylim(-maximum * .09, maximum * 1.16)
        axis.axhline(0, color="#b9c4d1", lw=.7)
    notes = [
        "44 successful requests · 1,024 output-token limit · uncached · greedy · reasoning off · drafting on",
        "Trials: 3 per group through 80K and 100K solo; 1 at 100K paired and 128K. Whiskers: observed min–max.",
        "Some 20K/40K paired replies ended early (157–301 tokens); all other replies reached 1,024 tokens.",
        "¹ Uncached input tokens ÷ (queue + prefill time), not isolated GPU prefill.",
        "² Total output ÷ batch wall time, including queueing and prefill. Source details: updated-results.json."]
    for y, note in zip((.132, .108, .084, .060, .036), notes):
        fig.text(.065, y, note, fontsize=10, color="#536377")
    save(fig, "swift-updated-benchmark")


def comparison(previous, current):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.065, right=.96, top=.75, bottom=.39, wspace=.22)
    fig.text(.065, .948, "TensorFold · Swift 1.5 solo comparison", fontsize=24, weight="bold")
    fig.text(.065, .906, "M5 Max Mac Studio · 1 request · previous vs updated configuration",
             fontsize=15, color="#536377")
    fig.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", lw=2, linestyle="--",
                               label="Previous Swift 1.5 configuration"),
                        Line2D([0], [0], color=PINK, marker="D", lw=2,
                               label="Updated Swift 1.5 configuration")],
               loc="upper left", bbox_to_anchor=(.06, .876), frameon=False, ncol=2, fontsize=12)
    for axis, (metric, title, unit) in zip(axes, [
            ("decode_tps", "Generation speed · higher is better", "Output tokens / second"),
            ("effective_input_tps", "Effective prompt throughput · higher is better", "Input tokens / second")]):
        x = axis_style(axis, title, unit, numeric=True)
        old, oldlo, oldhi = stats(previous, metric, 1)
        new, newlo, newhi = stats(current, metric, 1)
        draw_series(axis, x, old, oldlo, oldhi, BLUE, "o", "--")
        draw_series(axis, x, new, newlo, newhi, PINK, "D")
        for xx, a, b in zip(x, old, new):
            for value, color, offset in ((a, BLUE, 12 if a > b else -20),
                                         (b, PINK, 12 if b >= a else -20)):
                axis.annotate(f"{value:.1f}", (xx, value), xytext=(0, offset),
                              textcoords="offset points", ha="center", color=color,
                              fontsize=10, weight="bold")
        axis.set_ylim(0, max(*oldhi, *newhi) * 1.2)
    columns = [.35, .465, .58, .695, .81, .925]
    fig.text(.065, .293, "Updated vs previous", fontsize=11, weight="bold")
    for x, n in zip(columns, CONTEXTS):
        fig.text(x, .293, f"{n // 1000}K", ha="center", fontsize=11, weight="bold")
    for y, metric, label in ((.256, "decode_tps", "Generation change"),
                             (.219, "effective_input_tps", "Prompt throughput change")):
        fig.text(.065, y, label, fontsize=11, color="#536377")
        for x, n in zip(columns, CONTEXTS):
            old = statistics.median(samples(previous, metric, n, 1))
            new = statistics.median(samples(current, metric, n, 1))
            fig.text(x, y, f"{(new / old - 1) * 100:+.1f}%", ha="center", fontsize=11, weight="bold")
    notes = [
        "Previous trials: 3 at 20K/40K/60K; 1 at 80K/100K/128K. Updated trials: 3 at 20K–100K; 1 at 128K.",
        "Points: medians; whiskers: observed min–max, not confidence intervals. Solo outputs: 1,024 tokens; zero cache hits.",
        "The blue series matches the published Swift chart; its 128K point uses the prior higher-memory rerun.",
        "Historical configuration comparison, not a controlled A/B. Model revision, sampling and output limit match.",
        "Effective prompt throughput = uncached input tokens ÷ (queue + prefill time), not isolated GPU prefill."]
    for y, note in zip((.148, .123, .098, .073, .048), notes):
        fig.text(.065, y, note, fontsize=10, color="#536377")
    save(fig, "swift-solo-comparison")


def evidence(previous, current):
    metrics = ["decode_tps", "effective_input_tps", "ttft_seconds", "aggregate_tps"]
    summary = []
    for name, data in (("previous", previous), ("updated", current)):
        for n in CONTEXTS:
            for c in (1, 2):
                row = {"configuration": name, "context_tokens": n, "concurrency": c}
                for metric in metrics:
                    values = samples(data, metric, n, c)
                    row.update({metric: statistics.median(values), metric + "_min": min(values),
                                metric + "_max": max(values), metric + "_samples": len(values)})
                summary.append(row)
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    solo = []
    for n in CONTEXTS:
        row = {"context_tokens": n}
        for metric in metrics:
            old = statistics.median(samples(previous, metric, n, 1))
            new = statistics.median(samples(current, metric, n, 1))
            row.update({metric + "_previous": old, metric + "_updated": new,
                        metric + "_change_percent": (new / old - 1) * 100})
        solo.append(row)
    with (OUT / "solo-comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(solo[0])); writer.writeheader(); writer.writerows(solo)
    (OUT / "previous-results.json").write_text(json.dumps(previous, indent=2))
    (OUT / "updated-results.json").write_text(json.dumps(current, indent=2))
    print(json.dumps(solo, indent=2))


def main():
    current = (combine(CURRENT) if all((ROOT / path).is_file() for path in CURRENT)
               else json.loads((OUT / "updated-results.json").read_text()))
    previous = (combine(PREVIOUS, previous=True) if all((ROOT / path).is_file() for path in PREVIOUS)
                else json.loads((OUT / "previous-results.json").read_text()))
    assert len(current["requests"]) == 44 and len(previous["requests"]) == 32
    assert all(current["config"].get(k) == previous["config"].get(k) for k in MATCH_FIELDS)
    validate(current); validate(previous)
    evidence(previous, current)
    style(); standard(current); comparison(previous, current)
    print(f"Rendered both figures in {OUT}")


if __name__ == "__main__":
    main()
