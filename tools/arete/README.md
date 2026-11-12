# Arete 27B, MLX 5-bit with vision

This profile serves [NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B)
through the same serialized native gateway as Swift and Ornith. The
[published MLX checkpoint](https://huggingface.co/araysuter/Arete-27B-MLX-5bit)
contains affine 5-bit language weights, group size 64, and all 333 original BF16
vision tensors. Weight files total 19,416,210,059 bytes. This is a lossy language
quantization, not a claim of unchanged accuracy or a new quality benchmark score.

Source revision: `108f0afbdcfaa1ceeda4a635004304dfff6d4da2`. Conversion used
MLX 0.32.4, MLX-LM 0.32.2 and Transformers 5.22.1. Eligible language weights
average 5.501 bits with scales and unquantized parameters. The original vision
tensor bytes are copied unchanged, with their public names mapped to
`vision_tower.*`; the loader accepts their original patch layout. Processor
files and the original license are included. MLX-VLM 0.7.12 loaded this complete
checkpoint and correctly answered both synthetic image fixtures.

## Download or reproduce

Use an excluded destination on a backed-up Mac. Allow about 55 GB for the source
and 20 GB for the output, plus environments and temporary headroom. The source
is not needed just to run the published quant.

```sh
hf download araysuter/Arete-27B-MLX-5bit \
  --revision fe55469f0c69fb73c3680f49f384ec78175381ed \
  --local-dir "$HOME/ai/models/arete-27b-mlx-5bit-108f0afb"
```

Or, in a separate conversion environment with the versions above:

```sh
python tools/arete/convert.py \
  --source "$HOME/ai/models/arete-27b-bf16-108f0afb" \
  --output "$HOME/ai/models/arete-27b-mlx-5bit-108f0afb"
```

The converter pins the original source revision, refuses an existing output,
quantizes the language model, then attaches the BF16 tower. Bare
`mlx_lm.convert` strips the vision tower and is not this complete export.
Use MLX-VLM for standalone image input:

```sh
python -m mlx_vlm.generate --model araysuter/Arete-27B-MLX-5bit \
  --image /path/to/image.png --prompt 'Describe this image.' --max-tokens 256
```

## Shared API profile

Append the object in [gateway-profile.example.json](gateway-profile.example.json)
to the gateway's private configuration. It uses the session-enabled native
binary and the same pinned Qwen3.8 DFlash2 drafter used by Swift. Model ID:
`arete-27b`; alias: `arete-27b-5bit`. Native inference does not need Python MLX.
The gateway's four-conversation limit is global, including this model. Its
44 GiB working budget and the 6 GiB host reserve enable conservative admission;
background tasks can cause HTTP 503 rather than a new model load.

The profile has 131,072 total input/output tokens and bounded native image
input. See [image input and limits](../studio-gateway/VISION.md). Ripple remains
text-only. Image histories currently run cold through the gateway preflight;
text RAM/SSD reuse works. Video and general vision accuracy are not qualified
by the two-color image fixtures.

## M5 Max evidence

On the 64 GiB Studio, 26 native checks passed, including greedy and seeded
drafted/plain text equality, image identity, changed-image controls, fresh
follow-up equality, text SSD restoration, two-concurrent versus solo identity,
and four global snapshots. A 130,944-token input with an 8-token output budget
passed at the full configured window. Its restored follow-up reused 130,950
tokens. Native lifetime peak physical footprint was 45,256,262,096 bytes
(42.15 GiB). Cold full-window prompt processing took 911.49 seconds, about
143.7 tokens/sec; it is a synthetic capacity probe, not a quality benchmark.

Controlled short text decode cells used a 256-token output budget before tower
attachment; language tensor bytes were unchanged by attachment. Greedy prose
was 9.16 tok/s plain and 31.77 with DFlash2; sampled prose was 9.07 and 28.97.
Code was 9.07/91.58 greedy and 9.04/92.56 sampled, hitting the output cap. These
are one attempt per cell, not representative averages. Every pair matched
token-for-token. No new kernel or arithmetic implementation is introduced.

The deployed HTTPS API subsequently passed 13 rollout checks with admission and
vision enabled. Its fresh greedy prose/code cells measured 32.73 and 91.52 tok/s
and matched the earlier target token hashes. See the
[production receipt](../../benchmarks/arete-2026-11-07/production.json).

The opt-in [probe](probe.py) requires an owned isolated loopback gateway and
identifies only that pane's backend process group for its memory watchdog:

```sh
python3 tools/arete/probe.py --run --pane AI:2 \
  --key-file /path/to/private/key-file --key-label owner \
  --cache-root /path/to/isolated/cache --output /path/to/new-receipt.json \
  --vision --full-window
```

Native binary, immutable revisions, full receipts, failed expectations and
scope notes are retained with the qualification data. Other chips/backends,
new full-window Swift/Ornith vision capacity, and BF16-to-quant quality scores
are not established by these checks.
