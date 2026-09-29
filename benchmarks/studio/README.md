# Studio benchmarks

Run on a separate Mac while the Studio serves the selected model. Keep other clients idle. From the repository root, use a Python environment with:

```bash
python -m pip install -r benchmarks/studio/requirements.txt
```

The API key is prompted securely. Each command runs 20K, 40K, 60K, 80K, 100K and 128K input tokens, with one solo and two simultaneously submitted requests at each size: **18 requests**, up to 1,024 output tokens each. Drafting is enabled, reasoning is off, and sampling is greedy. These are synthetic raw-completion prompts, not a coding correctness evaluation.

## Swift

```bash
python benchmarks/studio/benchmark.py \
  --contexts 20000 40000 60000 80000 100000 128000 \
  --reps 1 --out benchmark-results-swift
```

## Qwen

```bash
python benchmarks/studio/benchmark.py \
  --model qwen3.8-27b \
  --tokenizer Vontra/Qwen3.8-27B-MLX-4bit \
  --revision 70ae7fac63274ff2eac54152031433374cb80f2f \
  --contexts 20000 40000 60000 80000 100000 128000 \
  --reps 1 --out benchmark-results-qwen
```

Use a new output directory for each run. Increase `--reps` to 3 for repeated trials. The output includes JSON, CSV, SVG charts and an HTML report. Generation speed is the server-reported per-request rate; effective input throughput includes queue/prefill waiting. Combined output throughput includes the entire batch wall time.

## Reproduce the README sheets

The committed `studio-benchmark-*` directories contain the source results used for the published charts, with no API credentials. From the repository root:

```bash
python benchmarks/studio/plot_results.py
python benchmarks/studio/plot_qwen_results.py
```

These render SVGs into `benchmarks/studio/charts/`. The Swift script combines three historical datasets and replaces its old 128K points with the higher-memory rerun. The Qwen script uses the single complete Qwen run. The plotting scripts intentionally reproduce these recorded datasets; a new benchmark run does not silently replace them.
