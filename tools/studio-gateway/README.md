# One API, two models, one resident engine

The Studio gateway exposes Swift-1.5 and Ornith-1.5-122B mixed 2-bit through one
OpenAI-compatible API. It runs one native TensorFold model process and its
DFlash2 drafter at a time. Model and drafter weights stay on SSD between uses.
The gateway is a small Python supervisor; inference remains in the native engine.

Requests for the resident model reuse it. A request for the other model parks
the active conversation, stops the old model process, waits for it to exit, and
starts the requested engine. Requests queue while another request or a model
switch is in progress. Streaming replies hold their place in the queue until
they finish or disconnect. The gateway does not load both models concurrently.

Each model has one active conversation and up to four parked conversations on
SSD. The context window is configured as 131,072 total input-plus-output tokens.
After 30 minutes without a request, the gateway parks the active conversation
and releases the model process. The next request loads the selected model and
restores its conversation. Cold requests therefore include model-loading and
state-restoration time.

## Native runtime requirement

This gateway needs native binaries with its internal session save, load, and
clear endpoints. A stock release binary does not provide that contract. Native
session snapshots preserve the model's tokens, attention cache, recurrent state,
and drafter state; saving only message text would require prompt recomputation.
See the native compatibility patch and build notes in [`native/`](native/).

Use the native-compatible Swift checkpoint prepared by
[`prepare-swift-native.py`](../prepare-swift-native.py). Ornith needs the native
Ornith port and support for its mixed 2-bit routed and 4-bit shared experts.
Download weights from their published repositories into a model directory
outside the source checkout:

- [Swift-1.5 affine 4-bit](https://huggingface.co/ukisai/Swift-1.5-4bit-MLX)
- [Ornith-1.5-122B mixed 2-bit](https://huggingface.co/araysuter/Ornith-1.5-122B-MLX-mixed-2bit)
- [Swift's Qwen3.8-27B DFlash2 drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [Ornith's DFlash2 drafter](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)

A configured 128K window is a capacity limit, not a promise of a measured
128K prompt-processing speed or a guarantee that every workload fits in 64 GiB.
Use the validated checkpoint, drafter, engine, and memory settings together.
Our [M5 Max capacity receipt](../../benchmarks/studio/charts/studio-model-gateway-2026-10-22/README.md)
checks 130,944 cold prompt tokens and 130,950 restored cached tokens for each
model. Observed process peaks were 42.6 GiB for Swift and 48.3 GiB for Ornith.
The published build target and these measurements were checked on the M5 Max.

## Download and prepare the checkpoints

Python 3.11 or later is sufficient for the supervisor. It uses the standard
library and does not need Python MLX. Model loading requires compatible Apple
Silicon hardware and the prepared native executables above.

Run these commands from this repository's root. A separate download environment
provides the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli);
it is not an inference environment:

```sh
mkdir -p "$HOME/ai"
python3 -m venv "$HOME/ai/hf-tools-env"
"$HOME/ai/hf-tools-env/bin/python" -m pip install --upgrade huggingface_hub
```

Download Swift into the ordinary cache at its tested revision, then prepare a
separate native-compatible copy. The converter requires the snapshot directory
name to be the exact pinned revision; downloading Swift with `--local-dir` to an
arbitrary name is not a substitute:

```sh
"$HOME/ai/hf-tools-env/bin/hf" download ukisai/Swift-1.5-4bit-MLX \
  --revision 82276731e47e1db4ac502f24c63cfaf886639be9 \
  --cache-dir "$HOME/.cache/huggingface/hub"
python3 tools/prepare-swift-native.py \
  "$HOME/.cache/huggingface/hub/models--ukisai--Swift-1.5-4bit-MLX/snapshots/82276731e47e1db4ac502f24c63cfaf886639be9" \
  "$HOME/ai/models/swift-1.5-native-82276731"
```

This standard-library converter adjusts 161 RMSNorm vectors and copies the
weight shards without MLX. It refuses an existing destination. Reuse a prepared
directory that already contains `swift-native.json`; do not apply the conversion
twice. Allow space for both the original Swift weights and their prepared copy.
The prepared copy links its small configuration/tokenizer assets to the source
snapshot, so keep that cached snapshot available.

Download our published smallest Ornith quant and both drafters into the paths
used by the example configuration:

```sh
"$HOME/ai/hf-tools-env/bin/hf" download araysuter/Ornith-1.5-122B-MLX-mixed-2bit \
  --revision 848e5a0d53f457d213bb43e554012722f3f3fc60 \
  --local-dir "$HOME/ai/models/ornith-1.5-122b-mlx-mixed-2bit-dfa07bab"
"$HOME/ai/hf-tools-env/bin/hf" download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir "$HOME/ai/drafters/Qwen3.8-27B-DFlash2"
"$HOME/ai/hf-tools-env/bin/hf" download incoai/Ornith-1.5-122B-DFlash2 \
  --revision a5ed7396f5f9e5b1abf7ff50b7899a3186961432 \
  --local-dir "$HOME/ai/drafters/Ornith-1.5-122B-DFlash2"
```

These are tested revisions, including the Ornith drafter revision used in the
benchmark, rather than moving `main` references. The public quant is ready to
download; the original 250 GB BF16 source and Python MLX are not required.

## Build the session-enabled native engine

Install Xcode with its Metal toolchain and Zig **0.17.0**. With that Zig on
`PATH`, build the pinned port and its snapshot patch:

```sh
zig version
xcrun metal --version
mkdir -p "$HOME/ai/tensorfold-gateway"
bash tools/studio-gateway/native/build.sh \
  "$HOME/ai/tensorfold-gateway/TensorFold" "$(command -v zig)"
```

The example configuration points at
`~/ai/tensorfold-gateway/TensorFold/zig-out/native/bin/tensorfold-native`.
Keep its build output tree together. The build script runs the native CPU
snapshot tests and prints the executable's SHA-256; it never starts a server.
See [`native/README.md`](native/README.md) for the patched runtime contract and
the separate GPU/context checks needed on the target machine.

## Private setup

Copy the placeholder configuration outside the checkout:

```sh
install -d -m 700 "$HOME/.config/tensorfold"
install -m 600 tools/studio-gateway/config.example.json \
  "$HOME/.config/tensorfold/studio-gateway.json"
```

Edit that private copy to use the actual native executable, checkpoint, and
drafter paths. `~` expands to the current user's home; shell variables such as
`$HOME` are not expanded in argument arrays. The example executable paths are
installation placeholders. Commands are argument arrays, not shell scripts.
Keep tokens out of `argv` and `env`; name key files instead.

Use existing owner and guest key files, or create new ones in the private
configuration directory. Keep their permissions at `0600`. The example guest
file is optional. Each request rereads the configured key files, so removing a
guest's file revokes future requests without replacing the owner key. Changing
a label or a filename is a configuration change and needs a gateway restart.

For a new installation without an owner key, this creates one and refuses to
overwrite an existing file. It does not print the key:

```sh
(umask 077; set -C; openssl rand -hex 32 > "$HOME/.config/tensorfold/api-key")
```

The example reserves a 9 GiB native context-cache budget for Swift and 4 GiB for
Ornith. The explicit `--prompt-cache-over-cap` setting is needed for those
budgets on this configuration. The engine's active execution state and saved
context can overlap in memory, so one logical conversation does not mean one
physical copy of every cache tensor. Leave these settings tied to measured
hardware capacity. The compatibility runtime uses the conversation flags to
enable its snapshot machinery; the gateway switches it to explicit save/load
mode and owns when snapshots are written. This does not fine-tune model weights.

The gateway listens on `127.0.0.1:8080`; its selected backend uses
`127.0.0.1:18080`. Keep the existing TLS tunnel pointed at port 8080. Add that
tunnel's public hostname to private `allowed_hosts`. The backend receives a
fresh internal key that is distinct from client keys. Internal session
endpoints are for the supervisor and are not exposed through the public API.

Validate the configuration, then start the gateway in the server terminal:

```sh
python3 tools/studio-gateway/gateway.py \
  --config "$HOME/.config/tensorfold/studio-gateway.json" --check-config
python3 tools/studio-gateway/gateway.py \
  --config "$HOME/.config/tensorfold/studio-gateway.json"
```

Stop an existing inference server on port 8080 before starting the gateway.
On the Studio, keep the server in tmux `AI:0` and checks in `AI:1`. The gateway
leaves the external tunnel and separate metrics collector running.

## Client use

Keep the existing `/v1` base URL and API key. Choose a model in each request:

| Model ID | Backend |
| --- | --- |
| `swift-1.5` | Swift-1.5 with Qwen3.8-27B DFlash2 |
| `ornith-1.5-122b-mixed-2bit` | Ornith mixed 2-bit with Ornith DFlash2 |
| `ornith-1.5-122b` | Alias for the Ornith model above |

For a local request, read the key from its private file:

```sh
python3 - <<'PY'
import json
from pathlib import Path
from urllib.request import Request, urlopen
key = (Path.home() / ".config/tensorfold/api-key").read_text().strip()
body = {"model": "swift-1.5", "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
request = Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                           "User-Agent": "TensorFold-Studio-Gateway/1.0"})
with urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
PY
```

Change `model` to `ornith-1.5-122b-mixed-2bit` to select Ornith.

`GET /v1/models` lists both configured models even when no engine is loaded.
The proxy accepts `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and
`/v1/messages` in the formats supported by the selected native engine. Responses
requests are stateless: send full input history; `previous_response_id`, stored
response chains, and background jobs are unsupported. The gateway forces
`store: false` for Responses requests.

SSD cache restoration currently applies to chat completions. Other request
formats run cold after a model or context switch. For chat completions,
`X-TensorFold-Context-Status` reports `cold`, `resident`, or `resumed`; a restored
file is only reported as resumed when its token prefix can actually be reused.

Full native snapshot restoration currently applies to `/v1/chat/completions`.
The other formats can select either model, but their full input history may
need to be processed again. Replies include `X-TensorFold-Context-Status` as
`resident`, `resumed`, or `cold`, so a cold continuation is distinguishable from
a restored conversation. An edited or shortened conversation can intentionally
start cold when it no longer matches the saved token prefix.

Use the normal chat-completions request format. The optional
`X-TensorFold-Context` header selects a conversation; reuse the same identifier
when resuming it. Conversation namespaces also include the authenticated key
and model, so separate API keys do not share state. Continue to send the full
conversation messages in each request, as normal for the OpenAI API.

Clients that omit the context header use automatic message-prefix affinity.
Explicit IDs are preferable when a client can supply them: independent chats
can start with identical messages, and explicit IDs make that separation clear.
Changing model selects that model's separate conversation state; an attention
cache from Swift cannot be reused by Ornith.

## Storage and limits

`cache_root` is a dedicated directory outside Git. Its per-run directory and
conversation state are private to the local user. Native snapshots can contain
sensitive conversation content. They use ordinary SSD files, not encrypted
storage. The supervisor clears its temporary state on clean shutdown and
removes its own stale run directories at the next startup. It does not delete
model weights. A shutdown clears conversation continuity; this is a temporary
cache, not a durable chat archive.

Four parked conversations is a bound per model, not unlimited storage. Older
parked state is evicted when that bound is reached. Keeping the message history
in the client allows the engine to compute its state again if a snapshot has
been evicted or cleared. A model switch always incurs loading latency.

Do not put private configuration, keys, native snapshots, raw benchmark data,
or downloaded weights in this repository. See the
[sharing and benchmark disclosure notes](../../docs/studio/SHARING.md).
