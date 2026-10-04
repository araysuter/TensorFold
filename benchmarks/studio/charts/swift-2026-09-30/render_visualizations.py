"""Render the single-trial Swift run in the established benchmark and solo-comparison layouts."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tensorfold-chart-cache")


def main() -> None:
    source = OUT.parent / "swift-2026-09-29" / "render_visualizations.py"
    spec = importlib.util.spec_from_file_location("swift_chart_layout", source)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    previous = json.loads((OUT / "previous-results.json").read_text())
    current = json.loads((OUT / "updated-results.json").read_text())
    assert current["config"]["reps"] == 1
    assert len(current["requests"]) == 18 and len(current["batches"]) == 12
    seen = set()
    for row in current["requests"]:
        key = tuple(row[k] for k in ("context_tokens", "concurrency", "repetition", "user"))
        assert key not in seen and row["repetition"] == 1
        seen.add(key)
        assert row["status"] == "ok" and row["cached_tokens"] == 0
        assert row["actual_prompt_tokens"] == row["context_tokens"]
    for field in ("model", "tokenizer", "revision", "output_tokens", "no_draft", "context_limit"):
        assert previous["config"][field] == current["config"][field], field
    short = [r["output_tokens"] for r in current["requests"] if r["output_tokens"] < 1024]
    lengths = (f"{len(short)} replies ended early ({min(short)}–{max(short)} tokens); other replies reached 1,024 tokens."
               if short else "All replies reached the 1,024-token output limit; zero cached prompt tokens.")
    original_save = layout.save

    def save(fig, filename):
        for text in fig.texts:
            note = text.get_text()
            if filename == "swift-updated-benchmark":
                if note.startswith("M5 Max Mac Studio · updated configuration"):
                    text.set_text("M5 Max Mac Studio · September 30, 2026 · 1 vs 2 concurrent requests")
                elif note.startswith("44 successful requests"):
                    text.set_text("18 successful requests · 1,024 output-token limit · uncached · greedy · reasoning off · drafting on")
                elif note.startswith("Trials:"):
                    text.set_text("One trial per context/concurrency pair. Whiskers show the two users' range, not repeated-trial variation.")
                elif note.startswith("Some 20K/40K"):
                    text.set_text(lengths)
            else:
                if note.startswith("M5 Max Mac Studio · 1 request"):
                    text.set_text("M5 Max Mac Studio · 1 request · original benchmark vs September 30")
                elif note.startswith("Previous trials:"):
                    text.set_text("Original: 3 solo trials at 20K/40K/60K; 1 at 80K/100K/128K. Current: 1 solo trial at every size.")
                elif note.startswith("Points: medians;"):
                    text.set_text("Points: medians; whiskers: observed min–max, not confidence intervals. No repeated-trial range for September 30.")
                elif note.startswith("The blue series"):
                    text.set_text("Blue: original published Swift chart, including its 128K higher-memory rerun. Pink: current 0.6.0 / M5 run.")
                elif note.startswith("Historical configuration comparison"):
                    text.set_text("Historical comparison, not a controlled A/B. New client uses local HTTP; earlier client used the public endpoint.")
        original_save(fig, filename)

    layout.validate(previous)
    layout.validate(dict(current, requests=[dict(r, source="single-trial") for r in current["requests"]],
                         batches=[dict(r, source="single-trial") for r in current["batches"]]))
    layout.OUT = OUT
    layout.save = save
    layout.evidence(previous, current)
    layout.style()
    layout.standard(current)
    layout.comparison(previous, current)
    print(f"Rendered both figures in {OUT}")


if __name__ == "__main__":
    main()
