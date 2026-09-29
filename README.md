# TensorFold — Mac Studio

Our TensorFold fork for the 64 GB M5 Max Mac Studio: Swift 1.5 or Qwen3.8 27B, both MLX 4-bit with DFlash2 speculative decoding.

## Reconnect and start

SSH into the Studio, then activate the existing installation:

```bash
ssh generator@generator
cd ~/ai/tensorfold-studio
source .venv/bin/activate
```

Start **one** model:

```bash
# Swift 1.5
bash tools/start-studio.sh
```

```bash
# Or Qwen3.8 27B
bash tools/start-studio-qwen.sh
```

Both use port 8080; stop the current server before switching. Press **Ctrl+C once** and wait for shutdown to finish. The launchers enable deletion of that model's inference checkpoints in their configured cache directories on graceful shutdown. Downloaded weights stay cached. Forced termination or power loss can bypass cleanup; older cache folders are not automatically cleaned.

## Update

Stop the server, then run from the activated environment:

```bash
cd ~/ai/tensorfold-studio
git pull --ff-only
python -m pip install -e .
bash tools/start-studio.sh
# Use tools/start-studio-qwen.sh instead for Qwen.
```

## Connection and settings

| Setting | Both launchers |
| --- | --- |
| Public API base URL | `https://ai-api.ashersuter.com/v1` through the existing Cloudflare tunnel |
| Local API | `http://127.0.0.1:8080/v1` |
| Model ID | `swift-1.5` or `qwen3.8-27b` |
| API key file | `~/.config/tensorfold/api-key` |
| Context limit | 131,072 tokens, including output |
| Parallel generation | Up to 2 requests, subject to memory admission |
| History inventory | 8 histories; not a limit of 8 snapshot files |
| RAM checkpoint cache | 2 checkpoints, 8 GiB budget |
| SSD spill budget | 128 GiB per model in its configured session-cache directory |
| Memory target | 58 GiB process / up to 55 GiB MLX; Metal may cap this lower |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2`, enabled by default |

The observed Qwen startup cap was **51.8 GiB process / 48.8 GiB MLX**. These are budgets, not preallocated memory or a guarantee against swap. Clients do not need to enable drafting; explicitly sending `"draft": false` disables it for that request.

The launcher copies the existing llama.cpp key if the TensorFold key is missing. Never put keys in this repository. If downloads need Hugging Face authentication, run `hf auth login` in the activated environment and supply a read token interactively.

For a new installation rather than reconnecting, use Python 3.11+:

```bash
mkdir -p ~/ai
cd ~/ai
git clone https://github.com/araysuter/TensorFold.git tensorfold-studio
cd tensorfold-studio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Provision `~/.config/tensorfold/api-key` before launching if the old llama.cpp key is unavailable. The public hostname requires the existing tunnel; cloning this repo does not configure it.

## Benchmarks

[Run the benchmark suite and regenerate these charts](benchmarks/studio/README.md). Charts are embedded SVGs, so GitHub renders them directly and they stay sharp when enlarged.

### Swift 1.5 4-bit

![Swift benchmark](benchmarks/studio/charts/tensorfold-studio-benchmark.svg)

### Qwen3.8 27B 4-bit

![Qwen benchmark](benchmarks/studio/charts/tensorfold-qwen-benchmark.svg)

These measure synthetic uncached throughput, not coding quality. Swift combines earlier runs with a higher-memory 128K rerun; Qwen uses one trial per configuration. They are not a controlled repeated-trial comparison. See the chart footnotes for timing definitions and sample counts.

For general model/backend documentation, see the [preserved upstream README](UPSTREAM_README.md) and [runbook](RUNBOOK.md).
