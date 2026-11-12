# One API, one resident engine

The default Studio gateway exposes Swift-1.5 and Arete-27B MLX 5-bit through one
OpenAI-compatible API. It runs one native TensorFold model process and its
Qwen3.8-27B DFlash2 drafter at a time. Both profiles use one 131,072-token window
and serialize requests. Model and drafter weights stay on SSD between uses.
The gateway is a small Python supervisor; inference stays in the native engine.
Historical Ornith and Ripple tools remain in the repository as optional setups;
their models are not part of the default configuration.

Requests for the resident model reuse it. A request for the other model parks
the active conversation, stops the old model process, waits for it to exit, and
starts the requested engine. Requests queue while another request or a model
switch is in progress. Streaming replies hold their place in the queue until
they finish or disconnect. The gateway does not load both models concurrently.

The shared LRU retains up to four conversations total across all models and API
keys, including the active conversation and its saved copy. The context window
is configured as 131,072 total input-plus-output tokens.
After 30 minutes without a request, the gateway parks the active conversation
and releases the model process. The next request loads the selected model and
restores its conversation. Cold requests therefore include model-loading and
state-restoration time.

On macOS, set each profile's `memory_required_gib` to a conservative physical
budget for weights, drafter, full context and scratch space. The gateway checks
reclaimable RAM and OS memory pressure after queuing, before loading or restoring
state, and after loading. It credits only the measured resident backend group's
physical footprint and leaves `memory_reserve_gib` (default 6 GiB) for the host.
Insufficient headroom returns HTTP 503 with `insufficient_memory`; unavailable
telemetry fails closed. Zero profile budget disables admission for legacy/test
profiles. The example enables it for both native profiles. This is an admission
check, not a reservation against another process allocating RAM afterward.

## Native runtime requirement

For image input, prepare Swift's original BF16 tower and enable the native
vision options in the example. Arete already includes its BF16 tower. See
[native image profiles](VISION.md) for preparation, input limits and the current
image-history cache limitation.

The native profiles need binaries with internal session save, load, and
clear endpoints. A stock release binary does not provide that contract. Native
session snapshots preserve the model's tokens, attention cache, recurrent state,
and drafter state; saving only message text would require prompt recomputation.
See the native compatibility patch and build notes in [`native/`](native/).

Use the native-compatible Swift checkpoint prepared by
[`prepare-swift-native.py`](../prepare-swift-native.py). The published
[Arete quant](../arete/README.md) is ready for the native profile.
Download weights from their published repositories into a model directory
outside the source checkout:

- [Swift-1.5 affine 4-bit](https://huggingface.co/ukisai/Swift-1.5-4bit-MLX)
- [Arete-27B MLX 5-bit](https://huggingface.co/araysuter/Arete-27B-MLX-5bit)
- [Shared Qwen3.8-27B DFlash2 drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)

A configured 128K window is a capacity limit, not a promise of a measured
128K prompt-processing speed or a guarantee that every workload fits in 64 GiB.
Use the validated checkpoint, drafter, engine, and memory settings together.
The historical [Swift capacity receipt](../../benchmarks/studio/charts/studio-model-gateway-2026-10-22/README.md)
and [Arete receipt](../../benchmarks/arete-2026-11-07/README.md) each check a
130,944-token cold input and 130,950 restored cached tokens on the 64 GiB M5 Max.
These are capacity probes, not general speed or quality benchmarks; their
original runtime identities and scope remain attached to those receipts.

## Download and prepare the checkpoints

Python 3.11 or later is sufficient for the supervisor. It uses the standard
library and does not need Python MLX. The native profiles require compatible
Apple Silicon hardware and the prepared executables above.

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

Download our published Arete quant and the shared drafter into the paths used
by the example configuration. On a backed-up Mac, verify the model and download
cache destinations are excluded from Time Machine before downloading:

```sh
"$HOME/ai/hf-tools-env/bin/hf" download araysuter/Arete-27B-MLX-5bit \
  --revision fe55469f0c69fb73c3680f49f384ec78175381ed \
  --local-dir "$HOME/ai/models/arete-27b-mlx-5bit-108f0afb"
"$HOME/ai/hf-tools-env/bin/hf" download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir "$HOME/ai/drafters/Qwen3.8-27B-DFlash2"
```

These are tested revisions rather than moving `main` references. The public
quant is ready to download; the original Arete BF16 source and Python MLX are
not required for native inference. Keep the shared drafter when removing an
unused model; both current profiles depend on it.

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

The example reserves a 9 GiB native context-cache budget for either model. The explicit `--prompt-cache-over-cap` setting is needed for those
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
| `arete-27b` | Arete MLX 5-bit with Qwen3.8-27B DFlash2 |
| `arete-27b-5bit` | Alias for Arete; `/v1/models` lists the canonical ID |

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

Change `model` to `arete-27b` to select Arete.

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
cache from Swift cannot be reused by Arete.

## Storage and limits

`cache_root` is a dedicated directory outside Git. Its per-run directory and
conversation state are private to the local user. Native snapshots can contain
sensitive conversation content. They use ordinary SSD files, not encrypted
storage. The supervisor clears its temporary state on clean shutdown and
removes its own stale run directories at the next startup. It does not delete
model weights. A shutdown clears conversation continuity; this is a temporary
cache, not a durable chat archive.

Four conversations is a shared bound across all models and API keys, including
the resident conversation and its saved copy. Older unused state is evicted
when that global bound is reached. Keeping the message history
in the client allows the engine to compute its state again if a snapshot has
been evicted or cleared. A model switch always incurs loading latency.

Do not put private configuration, keys, native snapshots, raw benchmark data,
or downloaded weights in this repository. See the
[sharing and benchmark disclosure notes](../../docs/studio/SHARING.md).
