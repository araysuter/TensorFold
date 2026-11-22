"""Render draft precision, paired decode speed, and discard rates from the benchmark receipts."""

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter


DRAFT_BITS = (2, 3, 4, 5, 6, 8, 16)
PROMPTS = {
    "prose": "Prose",
    "code": "Code",
    "natural-notes-8k": "Natural notes · 8K",
}
INK = "#172a3c"
MUTED = "#4c6174"
PLAIN = "#8e9eae"
DRAFT = "#0c7e77"
DISCARD = "#bb6442"


def read_report(path):
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        print(f"{path.name}: {error}", file=sys.stderr)
        return {}
    if not isinstance(report, dict):
        print(f"{path.name}: expected a JSON object", file=sys.stderr)
        return {}
    return report


def read_rows(report, key, index):
    rows = report.get(key, [])
    if not isinstance(rows, list):
        return {}
    return {
        row[index]: row for row in rows
        if isinstance(row, dict) and type(row.get(index)) in (int, str)
    }


def number(value, minimum=0, maximum=math.inf):
    if type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum:
        return value
    return None


def draft_label(bits):
    return "BF16" if bits == 16 else f"{bits}-bit"


def load_drafts(results_dir):
    reports = {}
    for bits in DRAFT_BITS:
        data = read_report(results_dir / f"draft-{bits}.json")
        reports[bits] = read_rows(data, "cases", "name")
    return reports


def draft_values(case):
    plain = case.get("plain")
    candidate = case.get("candidate")
    plain = plain if isinstance(plain, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}
    acceptance = number(candidate.get("proposal_acceptance"), maximum=1)
    return (
        number(plain.get("decode_tps")),
        number(candidate.get("decode_tps")),
        None if acceptance is None else 1 - acceptance,
    )


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b1bdc8",
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e4eaf0",
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
        "svg.hashsalt": "inkling-round-two",
    })


def save_figure(figure, output):
    figure.savefig(output, metadata={"Date": None})
    plt.close(figure)
    lines = output.read_text().splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def plot_drafts(reports, output):
    if not any(reports.values()):
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey="row")
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.79, wspace=.18, hspace=.32)
    fig.text(.085, .95, "Inkling 2 Mini · DSpark draft precision", fontsize=22, weight="bold")
    fig.text(.085, .906, "Paired decode throughput and discarded proposals across three natural prompts",
             fontsize=11, color=MUTED)
    fig.legend(handles=[Patch(color=PLAIN, label="Paired plain decode"),
                        Patch(color=DRAFT, label="DSpark decode"),
                        Patch(color=DISCARD, label="DSpark proposals discarded")],
               bbox_to_anchor=(.084, .868), loc="upper left", ncol=3, frameon=False, borderaxespad=0)

    max_tps = 0
    for (throughput, discard), (name, label) in zip(zip(*axes), PROMPTS.items()):
        throughput.set_title(label, loc="left", weight="bold", fontsize=12, pad=10)
        for offset, bits in enumerate(DRAFT_BITS):
            case = reports[bits].get(name, {})
            plain_tps, draft_tps, discard_rate = draft_values(case)
            if plain_tps is not None:
                throughput.bar(offset - .19, plain_tps, width=.36, color=PLAIN)
                max_tps = max(max_tps, plain_tps)
            if case.get("exact") is True and draft_tps is not None:
                throughput.bar(offset + .19, draft_tps, width=.36, color=DRAFT)
                max_tps = max(max_tps, draft_tps)
            if case.get("exact") is True and discard_rate is not None:
                discard.bar(offset, discard_rate, width=.63, color=DISCARD)
            if case.get("exact") is False:
                throughput.text(offset + .19, .05, "×", transform=throughput.get_xaxis_transform(),
                                ha="center", va="bottom", color=INK, fontsize=14)
                discard.text(offset, .05, "×", transform=discard.get_xaxis_transform(),
                             ha="center", va="bottom", color=INK, fontsize=14)
        for axis in (throughput, discard):
            axis.set_xlim(-.55, len(DRAFT_BITS) - .45)
            axis.set_xticks(range(len(DRAFT_BITS)), [draft_label(bits) for bits in DRAFT_BITS])
            axis.tick_params(axis="x", length=0, pad=8)
    axes[0, 0].set_ylim(0, max(1, max_tps * 1.1))
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0, 0].set_ylabel("Decode tokens/s")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_yticks([0, .25, .5, .75, 1])
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].set_ylabel("Discarded proposals")
    fig.text(.085, .075,
             "Same 8-bit target · three passes · 256-output-token cap per natural prompt.",
             fontsize=9, color=MUTED)
    fig.text(.085, .046,
             "Each precision uses its own plain run. × marks an exactness failure. Empty slots indicate unavailable values.",
             fontsize=9, color=MUTED)
    save_figure(fig, output)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        parser.error("results_dir must be a directory containing draft receipts")
    set_style()
    if not plot_drafts(load_drafts(args.results_dir), args.results_dir / "draft-precision.svg"):
        parser.error("no draft cases found in results_dir")


if __name__ == "__main__":
    main()
