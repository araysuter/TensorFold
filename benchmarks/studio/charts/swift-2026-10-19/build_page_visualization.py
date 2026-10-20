"""Regenerate the existing Page chart toggle from the saved benchmark receipts."""
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
CONTEXTS = [20000, 40000, 60000, 80000, 100000, 128000]
METRICS = ["decode_tps", "effective_input_tps", "ttft_seconds", "aggregate_tps"]


def series(data, concurrency):
    result = {}
    for metric in METRICS:
        rows = data["batches" if metric == "aggregate_tps" else "requests"]
        result[metric] = []
        for context in CONTEXTS:
            values = [r[metric] for r in rows if r["context_tokens"] == context
                      and r["concurrency"] == concurrency and r.get("status", "ok") == "ok"]
            result[metric].append({"context": context, "value": statistics.median(values),
                                   "low": min(values), "high": max(values)})
    return result


def main():
    original = json.loads((OUT / "previous-results.json").read_text())
    current = json.loads((OUT / "updated-results.json").read_text())
    one, two, previous = series(current, 1), series(current, 2), series(original, 1)
    changes = []
    for i, context in enumerate(CONTEXTS):
        changes.append({"context": context,
                        "generation": 100 * (one["decode_tps"][i]["value"] / previous["decode_tps"][i]["value"] - 1),
                        "prompt": 100 * (one["effective_input_tps"][i]["value"] / previous["effective_input_tps"][i]["value"] - 1)})
    data = {"one": one, "two": two, "original": previous, "current": one, "changes": changes}
    html = (OUT.parent / "swift-2026-10-04" / "swift-performance.html").read_text()
    html, count = re.subn(r" const DATA=.*?;\n const metrics=", " const DATA=" + json.dumps(data, separators=(",", ":")) + ";\n const metrics=", html)
    assert count == 1
    short = sum(r["output_tokens"] < 1024 for r in current["requests"])
    html = html.replace("October 4:", "October 19 · native 1.1.1:")
    if short:
        html = html.replace("all replies 1,024 tokens", f"{short} replies ended before the 1,024-token limit")
    html = html.replace("Historical comparison; original client used public HTTP, current client local HTTP.",
                        "Historical comparison across Python and native engines; original client used public HTTP, current client local HTTP.")
    html = html.replace("Prompt throughput = uncached tokens ÷ queue plus prefill.",
                        "Prompt throughput = uncached tokens ÷ queue plus prefill; native queue time comes from matching completion receipts.")
    html = html.replace("drawAll();});legend.append(b);", "drawAll();saveState();});legend.append(b);")
    html = html.replace(" function drawAll(){", " function saveState(){const view=root.querySelector('[role=tab][aria-selected=true]').id==='solo-tab'?'solo':'concurrency';window.openai?.setWidgetState?.({modelContent:{view},privateContent:{visible:{...visible}}}).catch(()=>{});}\n function drawAll(){")
    html = html.replace("function applyState(state){const mode=", "function applyState(state){if(state?.privateContent?.visible){Object.keys(visible).forEach(k=>{if(typeof state.privateContent.visible[k]==='boolean')visible[k]=state.privateContent.visible[k];});Object.entries(series).forEach(([mode,list])=>{root.querySelectorAll('.legend[data-mode='+mode+'] button').forEach((b,i)=>b.setAttribute('aria-pressed',String(visible[list[i].key])));});drawAll();}const mode=")
    html = html.replace("window.openai?.setWidgetState?.({modelContent:{view}}).catch(()=>{});", "saveState();")
    (OUT / "swift-performance.html").write_text(html)


if __name__ == "__main__":
    main()
