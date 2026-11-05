# Ripple DFlash2 on the temporary MLX backend

The matching [Ripple DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2)
drafter now has a target-verified MLX implementation. It shares the selected
mixed-4bit target's embedding and output head. The tested configuration uses
four-token verification blocks and the mixed-4/8-bit drafter, whose weights
occupy 2.48 GB. Target weights and BF16 context caches retain their precision.

The first four matched public prose/code cases, with greedy and seeded
temperature-0.7 sampling, produced identical target tokens. Steady generation
was 16.1–22.3 tokens/sec, versus 12.2–12.3 without drafting on the 64 GiB M5 Max.
These are short-prompt speed tests, not quality benchmark scores. Longer
context, prompt-processing and API deployment receipts are collected separately.

## Build the matching runtime

Stock MLX 0.32.4 uses different arithmetic for single-token and batched
quantized projections. Stock attention also changes its reduction path with
query count and physical sliding-cache order. The original experimental draft
path failed token identity; smaller blocks alone did not correct it.

The isolated runtime below keeps the existing single-token quantized kernel's
arithmetic for verification rows. It adds no shader and changes no weight bits.
The target's verification attention uses the original one-query kernel and
physical KV order. Normal prefill and plain decoding retain their original
dispatch. The process enables the override only while evaluating verification.

Use Xcode's macOS and Metal toolchains. The builder checks the original source
revision and file checksum, applies the included small dispatch patch, builds
a combined wheel and extracts it into an isolated directory. It does not
install over your existing MLX environment.

```sh
git clone --branch v0.32.4 --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/ml-explore/mlx.git "$HOME/ai/mlx-verify-source"
python3 tools/ripple/build_mlx.py --source "$HOME/ai/mlx-verify-source" \
  --python "$HOME/ai/ripple-mlx-env/bin/python" \
  --root "$HOME/ai/mlx-verify-build" --jobs 8
```

The source is Apple's Apache-2.0 [MLX 0.32.4](https://github.com/ml-explore/mlx/tree/v0.32.4),
commit `36969c094354086e15d2a327a9c50feb20b9cf11`. Preserve its included licenses.
The worker verifies its loaded extension and library hashes against the build
receipt and included patch. A build receipt establishes artifact identity;
the model, cache, sampling and performance tests establish runtime behavior.

## Prepare the drafter

Download the original matching drafter at the pinned revision, then convert
it independently. There is no need to download the target's BF16 weights again.

```sh
"$HOME/ai/ripple-mlx-env/bin/hf" download ricursiveAI/Ripple-65B-Preview-DFlash2 \
  --revision 35609b3be3dd08a822b3ba1257c1ea5dd6f87632 \
  --local-dir "$HOME/ai/models/Ripple-DFlash2-original"
"$HOME/ai/ripple-mlx-env/bin/python" tools/ripple/quantize_drafter.py \
  --manifest tools/ripple/source-manifest.json \
  --source "$HOME/ai/models/Ripple-DFlash2-original" \
  --output "$HOME/ai/models/Ripple-DFlash2-MLX-mixed4" --bits 4
```

FC, selector projection and convolution projections use 8 bits; other linear
projections use 4 bits. Norms, selector codebooks and convolution base kernels
remain BF16. Conversion checks the original weight checksum and complete
geometry. `--bits 8` is available for reproducing the larger-drafter experiment;
it did not improve the two measured greedy cases enough to justify its extra
1.86 GB of weights. A conversion receipt alone does not qualify inference.

## Configure the existing gateway

Add `--dflash-drafter`, `--dflash-exact-verify` and `--mlx-build-receipt` to
the Ripple worker's operator-controlled argv. Set that profile's `PYTHONPATH`
to the build's **absolute** `override` directory. Environment values do not
expand `~`. The matching example is in `gateway-profile.example.json`.

Keep the existing gateway, authentication and model aliases. One target plus
its matching drafter occupies RAM; other models and parked conversations stay
on SSD. The context limit counts input plus output together. Shutdown removes
the gateway's temporary conversation state, not model weights.

The private worker health reports `dflash2: true` once the configured model
has loaded. Requests can set `"draft": false` for a target-only comparison;
they cannot select an arbitrary drafter or checkpoint. Public statistics
expose numeric proposed/accepted token and verification counters.

## Reproduce the checks

The independent architecture oracle executes hash-pinned SGLang source ASTs
with CPU primitive adapters. Supply the original source file named in
`dflash_oracle.py`; it is not vendored into this repository. The oracle applies
the checkpoint's final NoPE layer explicitly, tests FP32 on CPU and BF16 on GPU,
and labels its limits. It does not qualify the complete SGLang/CUDA runtime.

```sh
source "$HOME/ai/ripple-mlx-env/bin/activate"
python3 tools/ripple/test_dflash_model.py --source /path/to/pinned/sglang-dflash.py \
  --output /path/to/new-architecture-receipt.json
python3 tools/ripple/test_dflash_cache.py
python3 tools/ripple/test_dflash_exact.py
python3 tools/ripple/test_dflash_generate.py
PYTHONPATH="$HOME/ai/mlx-verify-build/override" \
  "$HOME/ai/ripple-mlx-env/bin/python" tools/ripple/test_worker_dflash.py \
  --source "$HOME/ai/models/Ripple-65B-Preview-MLX-mixed-4bit" \
  --mlx-build-receipt "$HOME/ai/mlx-verify-build/build-receipt.json"
PYTHONPATH="$HOME/ai/mlx-verify-build/override" \
  "$HOME/ai/ripple-mlx-env/bin/python" tools/ripple/benchmark_dflash.py \
  --model "$HOME/ai/models/Ripple-65B-Preview-MLX-mixed-4bit" \
  --drafter "$HOME/ai/models/Ripple-DFlash2-MLX-mixed4" \
  --exact-verify --mlx-build-receipt "$HOME/ai/mlx-verify-build/build-receipt.json" \
  --draft-block-widths 4 --max-tokens 256 --output /path/to/new-decode-receipt.json
```

Run one model process at a time. Receipts preserve exact public prompt/output
tokens, source hashes, seeds, cold-cache state, throughput and memory limits.
The fresh-KV prefill phase can prime a short request first with
`--prime-tokens 16`; that measures cold KV with already used model weights.
It separates this from initial model loading and compilation time. Failed
experiments remain failed and are retained alongside selected results.

This is an owner-authorized temporary MLX profile. Native Ripple, other Apple
chips and other GPU/OS backends have separate qualification requirements.
