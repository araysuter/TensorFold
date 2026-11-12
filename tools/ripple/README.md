# Ripple-65B Preview MLX experiment

This converts the original **instruction-tuned BF16 release**, not the separate
pretraining-only Base checkpoint and not an existing quantization. Every target
variant is made independently from the original tensors. Source revisions,
large-file hashes and sizes are pinned in `source-manifest.json`.

The objective is the highest tested fidelity that fits a single 131,072-token
window on a 64 GiB Apple Silicon machine, with room for the matching DFlash2
drafter and execution buffers. Weight size alone does not prove that fit.

## Precision profiles

| Profile | MLP matrices | Other attention matrices | Attention sigmoid gates, embedding, output head |
| --- | --- | --- | --- |
| mixed-4bit | 4-bit, groups of 64 | 4-bit, groups of 64 | 8-bit, groups of 64 |
| mixed-2bit | 2-bit, groups of 32 | 4-bit, groups of 64 | 8-bit, groups of 64 |
| mixed-1bit | 1-bit, groups of 32 | 4-bit, groups of 64 | 8-bit, groups of 64 |

Norms and value-residual scalars remain BF16. The one-bit profile is an
**experimental MLP quantization**, not a claim that the entire model is one bit
or retains the original quality. Receipts report total weight bytes and actual
average bits per parameter, including scales, biases and protected tensors.
Start evaluation with mixed-4bit; recommend a profile only after examining its
fidelity, outputs and measured memory headroom.

`--method mse` searches per-group clipping ranges for 2/4-bit matrices and uses
two-centroid Lloyd refinement for one-bit MLP weights. It retains the standard
affine result for any group whose reconstructed-weight error would increase.
Protected 8-bit matrices remain standard affine. This is weight-error
optimization, not activation-calibrated AWQ/GPTQ, and reduced weight error alone
does not establish better language-model quality. Compare it against the
ordinary affine conversion on the held-out fidelity inputs and generation probes.

The separate DFlash2 preparation uses 4-bit projections, with its FC, selector
projection and dynamic-convolution projections at 8-bit. Norms, selector
codebooks and convolution base kernels remain BF16. Its original five-layer
attention pattern, NoPE layer, four-tap convolutions and target taps are retained.
It is a draft network sharing the target's embedding/head, not a standalone Qwen3.

## MLX adapter and validation

`modeling_ricursive_mlx.py` implements the publisher's Ripple computation:
PeriNorm, Q/K normalization, local RoPE, global NoPE, value-residual input blending,
attention output gates, embedding scaling and clipped SwiGLU. Local layers use
bounded rotating caches. Norm parameters are used directly, with no Gemma-style
`+1` transform.

The custom model uses MLX-LM's `model_file` mechanism and requires the explicit
`--trust-remote-code` option when loaded. The tested environment is MLX 0.32.4,
MLX-LM 0.32.2, NumPy 2.5.4 and safetensors 0.8.0 on Apple Silicon. A stock older
MLX-LM release without custom `model_file` support is insufficient.

```sh
python3 -m venv .venv
.venv/bin/pip install mlx==0.32.4 mlx-lm==0.32.2 numpy==2.5.4 safetensors==0.8.0
.venv/bin/python -m mlx_lm.generate --model /path/to/complete/quant \
  --trust-remote-code --prompt 'Explain binary search.' --max-tokens 512
```

The original architecture and its Apache-2.0 license are provided by
[Ricursive Intelligence](https://huggingface.co/ricursiveAI/Ripple-65B-Preview).
The adapter is independently expressed with MLX operators and is checked against
the publisher's PyTorch implementation using deterministic small models.
FP32 architecture/cache checks run on CPU; BF16 checks run on the GPU. On the
M5, GPU FP32 projections differed from the CPU oracle, so those failed FP32 GPU
attempts must not be represented as exactness passes.

```sh
python3 tools/ripple/test_adapter.py --source /path/to/original-bf16 \
  --output /path/to/adapter-tests.json
python3 tools/ripple/test_conversion.py
python3 tools/ripple/download.py --manifest tools/ripple/source-manifest.json \
  --root /path/to/models --receipt /path/to/download-receipt.json
python3 tools/ripple/convert.py --manifest tools/ripple/source-manifest.json \
  --source /path/to/original-bf16 --output /path/to/new-quants --bits 4 2 1 --method mse
```

The converter streams one source tensor at a time, verifies every source-shard
checksum, checks complete tensor coverage and writes bounded output shards.
It can consume completed source shards while others download. It refuses an
existing output directory. Interrupted conversions keep `CONVERSION_INCOMPLETE`;
do not serve or publish them. `VALIDATION_PENDING` means conversion finished but
does not certify inference, quality, 128K capacity or native serving.

`fidelity.py original` streams the original BF16 weights one layer at a time,
so it does not need to fit the full ~130 GB checkpoint in RAM. It records logits
for up to 1,024 fixed positions in synthetic prose, arithmetic, code and JSON. Each
quant is then compared at the same teacher-forced histories using KL, top-token
agreement and next-token loss. These are small fidelity diagnostics, not GPQA,
Terminal-Bench or a whole-model PyTorch reference qualification.

`probe.py` records uncached generation samples and a 2K prompt-speed probe. Its
separate capacity phase fills 131,008 prompt tokens with room for 64 output
tokens. It retains BF16 KV, caps MLX allocations at 54 GiB and stops above a
56 GiB process physical footprint. A target-only capacity result does not
qualify a DFlash2-enabled server or prove long-context task accuracy.

## Optional temporary MLX API worker

The Studio gateway can supervise `worker.py` alongside the existing native
Swift/Ornith profiles. It stops and reaps the previous engine before starting
another, so only one model's weights and active conversation occupy RAM.
The worker listens only on loopback and uses the gateway's ephemeral private
credential. Public authentication, request serialization, tenant isolation,
idle unloading and four parked SSD conversations remain owned by the gateway.

Use **mixed-4bit** for this profile. The tested weights are 39.05 GB / 36.37 GiB,
including higher-precision protected tensors. In the synthetic teacher-forced
diagnostic it retained 89.75% top-token agreement with the streamed BF16
reference. The selected mixed-2bit reached 76.27% but generated an incorrect
`is_prime` function; mixed-1bit reached 24.12% and repeated text. These are
conversion diagnostics, not task accuracy. There is no matched Ripple GPQA
result to compare with the Swift/Ornith scores in the main README.

Install the pinned MLX environment outside the checkout, download the selected
checkpoint, and append the model object in `gateway-profile.example.json` to
your private gateway configuration's `models` array. Edit only the environment,
checkout and model paths for your machine; keep the gateway substitutions.

```sh
python3 -m venv "$HOME/ai/ripple-mlx-env"
"$HOME/ai/ripple-mlx-env/bin/pip" install mlx==0.32.4 mlx-lm==0.32.2
"$HOME/ai/ripple-mlx-env/bin/hf" download araysuter/Ripple-65B-Preview-MLX-mixed-4bit \
  --local-dir "$HOME/ai/models/Ripple-65B-Preview-MLX-mixed-4bit"
python3 tools/studio-gateway/gateway.py --config /path/to/private/gateway.json --check-config
python3 tools/studio-gateway/gateway.py --config /path/to/private/gateway.json
```

Choose `ripple-65b-preview` in `POST /v1/chat/completions`, using the same public
base URL and API key as the other configured models. Send complete text message
history and optionally a stable `X-TensorFold-Context` header. The 131,072-token
limit counts the rendered prompt plus requested output budget. It rejects an
oversized request instead of truncating history. The temporary worker supports
one text completion per request, streaming, sampling, token usage and non-stream
logprobs.
Use `stream: true` for long prompts so prefill keepalives keep the tunnel active.
Tools, images, JSON-schema constraints, Responses and Anthropic Messages are not
implemented by this temporary Ripple profile; they are rejected.

The worker uses one active BF16 attention cache, without a second in-RAM KV
checkpoint. It saves private safetensors snapshots under the gateway's ephemeral
run directory. Import checks build/model identity, token history, cache classes,
positions and shapes; the gateway compares the full-file receipt fingerprint.
Control operations serialize behind generation, including after a stream's
final event. A changed or shorter token prefix starts cold. Shutdown removes
the gateway's temporary context directory. No prompts or credentials are logged.

`worker.py` pins the MLX-LM server interface to version 0.32.2, reusing its chat
response and reasoning formatting while replacing model selection and cache
ownership. The model path is fixed by operator configuration; requests cannot
download or select arbitrary checkpoints or adapters. The default 512-token
prompt chunks were compared with 256 on identical cold 2K/8K/32K inputs. The
process has a 54 GiB MLX allocation limit and a 56 GiB physical-footprint guard.
The whole 64 GiB machine still needs memory for macOS and other applications.

The [Studio receipt](validation-2026-11-04.json) records a 130,944-token cold API
prompt, a switch through Swift, and restoration of 131,008 cached tokens for a
131,023-token follow-up. Only 15 new prompt tokens needed processing. At most one
model process was observed. This was a capacity/cache check during uploads, not
a task-accuracy benchmark; its replies reached their requested output caps.
The separate target-only capacity probe peaked at 46.33 GiB. Production HTTPS
generation, streaming, both existing credentials and a short warm turn passed;
the short warm request took about 0.71 seconds, including its two output tokens.

```sh
python3 tools/ripple/test_worker.py --source /path/to/original-bf16
python3 tools/ripple/studio_api.py --endpoint http://127.0.0.1:8080 \
  --key-file /path/to/private/labeled-key-file --model /path/to/mixed-4bit \
  --output /path/to/new-private-receipt.json --capacity
```

The Studio integration probe expects existing `owner` and `friend` labels; it
reads credentials locally and emits only scoped results and numeric usage. Run
it on an idle, isolated gateway because it switches all three models and can
fill nearly the entire context. Its receipt records actual process overlap,
fresh/warm token equality, tenant separation and full-window SSD restoration.

## Native API and DFlash2 status

The native Swift/Ornith profiles continue to use the previously qualified
session-enabled binary. Synchronizing upstream source does not replace that
binary. Ripple uses the temporary MLX worker above, not a native TensorFold
family. The [matching DFlash2 implementation](DFLASH2.md) now supports strict
target verification, synchronized draft/target KV, cancellation and SSD state.
It uses an isolated MLX dispatcher build and preserves the single-token target
arithmetic during verification. A stock MLX installation serves plain decoding.
Follow the DFlash2 guide for the selected mixed-4/8-bit drafter, installation,
measured speed and the distinction between short-context and full-window checks.

The current upstream Ripple bring-up is tracked in
[TensorFold #2171](https://github.com/ashhart/TensorFold/issues/2171); the stated
initial scope is BF16 native Metal, with DFlash2 planned separately. The MLX
adapter and converted tensors here do not establish native TensorFold support.
Do not relabel Ripple as Qwen or Gemma to bypass architecture admission.
