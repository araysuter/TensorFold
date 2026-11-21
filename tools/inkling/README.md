# Inkling MLX gateway adapter

This optional worker exposes an already converted Inkling 2 Mini checkpoint
through the existing authenticated Studio gateway. It does not add an Inkling
family to the native Zig engine or change another model's runtime.

Use a dedicated environment with MLX 0.32.4 and MLX-LM PR 2086 (stacked on PR 2061) pinned to
`0c4c99192230f1943dd468d297f0b8a64ae7b7c2`. Install that exact MLX-LM revision;
released MLX-LM versions without the Inkling port are unsupported:

```sh
python3 -m venv "$INKLING_ENV"
"$INKLING_ENV/bin/python" -m pip install 'mlx==0.32.4' \
  'git+https://github.com/ml-explore/mlx-lm.git@0c4c99192230f1943dd468d297f0b8a64ae7b7c2'
```

The worker
sets `MLX_ENABLE_TF32=0` before importing MLX to preserve the qualified FP32 path.

## Quick local API

After the root README's pinned environment and download steps, run from the
repository root in that same shell. This direct worker binds only to loopback;
it is useful for trying the API. For model switching, RAM admission and idle
unloading, use the gateway configuration below.

```sh
install -d -m 700 "$HOME/.config/tensorfold"
export INKLING_KEY_FILE="$HOME/.config/tensorfold/inkling-local.key"
# Create a new key file; the exclusive open preserves any existing file.
.venv-inkling/bin/python - <<'PYKEY'
import os, secrets
from pathlib import Path
p = Path(os.environ["INKLING_KEY_FILE"])
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    f.write(secrets.token_hex(32) + "\n")
PYKEY
.venv-inkling/bin/python tools/inkling/worker.py \
  --model "$MODEL_DIR" --name inkling-2-mini --context 131072 \
  --port 8080 --api-key-file "$INKLING_KEY_FILE"
```

If the key file already exists, skip its creation and reuse it. In another
terminal, send a streaming request (keep the key file private):

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $(cat "$HOME/.config/tensorfold/inkling-local.key")" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128,"stream":true}'
```

In an OpenAI-compatible client, use base URL `http://127.0.0.1:8080/v1`,
model `inkling-2-mini`, and the generated key. The dedicated worker has no
gateway RAM-admission guard; stop it before running another large model.
The 128K setting is a total input-plus-output limit, not preallocated history.

## Gateway profile

Configure a gateway model with these fields, replacing environment, source and
checkpoint paths with absolute paths on the serving host:

```json
{
  "id": "inkling-2-mini",
  "argv": [
    "/absolute/environment/bin/python", "/absolute/source/tools/inkling/worker.py",
    "--model", "/absolute/checkpoint", "--name", "inkling-2-mini",
    "--port", "{port}", "--api-key-file", "{backend_key_file}",
    "--context", "131072"
  ],
  "stateless": true,
  "context_window": 131072,
  "max_concurrent": 2,
  "memory_required_gib": 22,
  "memory_per_request_gib": 5.5,
  "request_reserve_gib": 11,
  "readiness_timeout": 300
}
```

These budgets are deployment profile inputs, not a promise that any machine can
admit two full windows. The gateway checks live memory pressure and remaining
full-window growth for every admission. Keep its per-model reserve at least
8 GiB and qualify actual peak workspace on your hardware before deployment.
The model is unloaded by the gateway's existing idle timer. This stateless
worker retains no completed KV history and consumes no SSD context slots.

Both `/v1/chat/completions` and `/v1/completions` are available. Images use
OpenAI `image_url` content parts containing inline PNG, JPEG or WebP data URLs;
remote URLs and filesystem paths are refused. Limits are four images, 4 MiB
per decoded image, 4 megapixels per image, and an 8 MiB request body. Expanded
image tokens count toward the prompt plus output context limit.

Up to two concurrent text requests retain independent KV/short-convolution
states while sharing one loaded model. The worker advances bounded prompt
chunks and upstream `generate_step` iterators in turn on one GPU stream. It
uses the same single-request model shapes and cache layouts as the solo
reference, including independent seeded-sampling state. This is interleaved
serving, not an array-batched kernel or a native Zig lane. Image requests drain
text requests and run sequentially. No speculative drafter is qualified in
this profile.

For isolated benchmarking only, add `--benchmark-ignore-eos` to a separate
worker profile. Never use that flag in the public serving profile. It disables
EOS stopping to match fixed-output speed tests. `draft` does not enable a drafter in this profile; every request uses the
upstream ordinary decoding core. The final response/usage event includes a generated-token SHA-256 and
worker-observed prompt/decode timings. Prompt timing includes worker queueing,
tokenization and image processing, but excludes cold model loading; client TTFT
includes gateway and model-loading delays. No prompt cache is reused.

Run request boundary checks with:

```sh
python3 -m unittest discover -s tools/inkling -p 'test_*.py'
```

Use `tools/bench_studio_concurrency.py` against the isolated benchmark profile
with `--context 131072 --levels 1,2 --lengths 4000,8000,32000 --tokens 256`.
It compares generated-token hashes to each stream's solo/plain reference and
fails on disagreement. This is a throughput check, not a quality benchmark.

An [experimental DSpark benchmark](DSpark.md) is available separately. Its
exact-output verifier is not used by the API worker: short-context gains
did not hold at 32K. See the [follow-up speed/rejection graph and results](../../benchmarks/inkling-2026-11-20/README.md). For long prompts over a reverse proxy, use streaming;
the worker emits prompt-progress keepalives while filling the context.
