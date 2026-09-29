"""Render a new Studio benchmark in the four-panel style of the recorded Swift/Qwen charts."""

import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results directory or results.json")
    parser.add_argument("--out", type=Path, help="SVG path; defaults to benchmark.svg in the results directory")
    parser.add_argument("--title", help="chart title; defaults to the recorded model ID")
    parser.add_argument("--subtitle", default="M5 Max Mac Studio · 1 vs 2 concurrent requests")
    args = parser.parse_args()
    source = args.results / "results.json" if args.results.is_dir() else args.results
    data = json.loads(source.read_text())
    rows, batches, config = data["requests"], data["batches"], data["config"]
    contexts = sorted(set(config["contexts"]))
    output = args.out or source.parent / "benchmark.svg"
    if output.suffix.lower() != ".svg":
        parser.error("--out must end in .svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#2563eb", "#e77b24"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#bcc5cf", "text.color": "#172b42", "axes.labelcolor": "#34465b",
                         "xtick.color": "#536377", "ytick.color": "#536377"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.065, right=.96, top=.81, bottom=.17, hspace=.43, wspace=.22)
    fig.text(.065, .956, args.title or f"TensorFold · {config['model']}", fontsize=24, weight="bold")
    fig.text(.065, .922, args.subtitle, fontsize=15, color="#536377")
    handles = [Line2D([0], [0], color=color, marker="o" if c == 1 else "s", lw=2,
                      label="1 request" if c == 1 else "2 simultaneous submissions (may queue)")
               for c, color in zip((1, 2), colors)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .90), frameon=False, ncol=2, fontsize=12)
    metrics = [("decode_tps", "Generation speed per request", "Tokens / second"),
               ("effective_input_tps", "Effective prompt throughput¹", "Input tokens / second"),
               ("ttft_seconds", "Time to first token", "Seconds"),
               ("aggregate_tps", "Combined end-to-end output throughput²", "Output tokens / second")]
    for axis, (metric, title, unit) in zip(axes.flat, metrics):
        axis.set_title(title, loc="left", pad=16)
        axis.set_ylabel(unit)
        axis.set_xlabel("Input context (tokens)")
        axis.set_xticks(range(len(contexts)), [f"{n // 1000}K" for n in contexts])
        axis.grid(axis="y", color="#e7ebf0", lw=.8)
        axis.set_axisbelow(True)
        maxima = []
        for concurrency, color in zip((1, 2), colors):
            values, lower, upper = [], [], []
            for n in contexts:
                records = batches if metric == "aggregate_tps" else rows
                samples = []
                for record in records:
                    if record["context_tokens"] != n or record["concurrency"] != concurrency:
                        continue
                    if metric != "aggregate_tps" and record["status"] != "ok":
                        continue
                    value = record.get(metric)
                    if value is None and metric == "effective_input_tps":
                        seconds, tokens = record.get("queue_prefill_seconds"), record.get("actual_prompt_tokens")
                        value = (tokens - record.get("cached_tokens", 0)) / seconds if seconds and tokens else None
                    if value is not None and math.isfinite(value):
                        samples.append(value)
                median = statistics.median(samples) if samples else math.nan
                values.append(median)
                lower.append(median - min(samples) if samples else math.nan)
                upper.append(max(samples) - median if samples else math.nan)
                maxima.extend(samples)
            x = [i + (concurrency - 1.5) * .035 for i in range(len(contexts))]
            if any(math.isfinite(value) for value in values):
                axis.errorbar(x, values, yerr=[lower, upper], color=color, lw=2.4,
                              marker="o" if concurrency == 1 else "s", ms=6, capsize=3, elinewidth=1)
            for xx, value in zip(x, values):
                if math.isfinite(value):
                    offset = 11 if concurrency == 1 else -18
                    if metric in ("aggregate_tps", "ttft_seconds"):
                        offset = -19 if concurrency == 1 else 11
                    axis.annotate(f"{value:.1f}", (xx, value), xytext=(0, offset), textcoords="offset points",
                                  ha="center", color=color, fontsize=10, weight="bold")
        axis.set_xlim(-.3, max(.4, len(contexts) - .65))
        maximum = max(maxima, default=1) or 1
        axis.set_ylim(-maximum * .09, maximum * 1.16)
        axis.axhline(0, color="#b9c4d1", lw=.7)
    failed = sum(row["status"] != "ok" for row in rows)
    drafting = "off" if config.get("no_draft") else "on"
    fig.text(.065, .115, f"{len(rows) - failed} successful requests · {failed} failed/invalid · "
             f"{config['output_tokens']:,} output-token limit · greedy · reasoning off · drafting {drafting}",
             fontsize=11, color="#536377")
    fig.text(.065, .090, "Points: medians; whiskers: observed min–max. Failed cells are gaps. "
             "Synthetic uncached prompts measure throughput, not coding quality.", fontsize=10, color="#536377")
    fig.text(.065, .065, "¹ Uncached input tokens ÷ (queue + prefill time), not isolated GPU prefill.",
             fontsize=10, color="#536377")
    fig.text(.065, .040, "² Total output ÷ batch wall time, including queueing and prefill. "
             "See results.json for lengths, cache hits and individual requests.", fontsize=10, color="#536377")
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
