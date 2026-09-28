# Astra Studio configuration

This fork adds conversation monitoring and Bearer authentication to TensorFold 0.3.6.2.
It reuses upstream `--spill-gib` for SSD snapshots. The BF16 cache and decoding kernels
are unchanged. `tools/start-studio.sh` selects Swift's tested snapshot, two compute lanes,
a 131,072-token prompt-plus-response limit, eight tracked conversations and a 128 GiB
SSD spill budget. Idle RAM checkpoints remain until the cache limit or memory admission
requires eviction. There is no fixed reservation of two complete 128k windows: the
existing memory controller can queue the second request when both would not fit.

## Install on the Studio

Stop the running TensorFold, and stop Caddy (or llama.cpp) if it owns 8080. Keep cloudflared
and the collector running. Use a fresh venv so the old manual mlx-lm loader patch cannot
be applied twice. This fork includes the revision-scoped Swift compatibility fix;
vision is still unsupported. Do not run the old fix-swift-tensorfold.py on this fork.

```bash
cd ~/ai
git clone https://github.com/araysuter/TensorFold.git tensorfold-studio
cd tensorfold-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -e . 'mlx-lm==0.31.3' 'mlx==0.32.2'
bash tools/start-studio.sh
```

The startup script is ordinary configuration: copy it to `~/ai/start-tensorfold.sh`
if desired, and activate this venv before using it. `STUDIO_PORT=8082` overrides the
port for an optional local comparison. The normal setup serves authenticated requests
on 8080, preserving the tunnel, public hostname and `swift-1.5` model ID. It reuses the
existing TensorFold key, or copies the llama.cpp key if missing. No Caddy is required.

```bash
curl -i http://127.0.0.1:8080/v1/models  # must return 401
curl --fail-with-body http://127.0.0.1:8080/studio/metrics \
  -H "Authorization: Bearer $(cat ~/.config/tensorfold/api-key)"
```

`/studio` is a public empty HTML shell; entering the key fetches authenticated metrics.
The key stays in tab memory, not localStorage. `/studio/metrics`, `/health`, model lists
and completion endpoints all require the key when configured. No key is placed in URLs.

## Conversations versus checkpoints

`--conversation-slots 8` tracks eight conversations, protecting every pending conversation
from replacement. When all eight are pending, another request is rejected. When an idle
entry is replaced, its disk snapshot remains reusable until normal disk-budget pruning.
These are conversation entries, not user accounts or eight reserved cache files. Several
prefix checkpoints can belong to one conversation. RAM figures in a slot are retained
idle checkpoints; process memory also includes working streams, model weights and scratch
buffers. Shared prefixes can appear in multiple slot counts and must not be summed as
unique GPU memory.

Clients should send a stable, unique `X-Conversation-ID` for each conversation. Clients
without it are matched by a hash of their messages through the first user message.
Identical openings are ambiguous: explicit IDs avoid this. Changing system instructions
or compaction can create another entry. Only one pending request per conversation ID is
allowed. IDs select inventory entries only; cache reuse still requires an exact token
prefix and compatible model/runtime metadata. Prompt text, token IDs and conversation
IDs are never returned by monitoring endpoints.

Upstream saves evicted checkpoint tensors synchronously on the engine thread before their
last references are released. Large saves/restores can pause other decoding during I/O.
The cache remains BF16; recurrent state is serialized too, while upstream rebuilds transient
buffers and drafter state. Saved prefixes stop at valid prompt chunk boundaries, so restore
may reprocess the recent tail. If SSD writes fail or the byte budget prunes a checkpoint,
a later request needs prefill again; the monitor shows `uncached`, not a successful SSD save.
This is application-managed cache storage, not macOS swap, and does not disable OS swap.

The inventory manifest survives restarts for the same model/runtime key; obsolete cache
versions are not reused. Use a private cache directory (`start-studio.sh` sets umask 077).
Graceful shutdown also saves retained conversations within the spill budget; spill happens during operation, not on every
completed reply. A crash may therefore lose a recent in-RAM-only prefix.

## Metrics and history

States: empty, queued, restoring, prefilling, generating, saving, ram, ssd, uncached, error.
Prompt speed measures the latest completed prefill chunk. Decode speed measures accepted
output tokens over the latest engine round, per stream. Neither is a rolling average or
a claim of instantaneous throughput. `rate_at` exposes reading age; no new work means no
new measurement. Monitoring never reads GPU tensors from HTTP threads.

The companion collector stores 24 hours of SQLite samples at one-second intervals with a
viewer heartbeat, five seconds otherwise. It adds host memory/swap measurements and serves
`/monitor/history` through the existing tunnel. The private Sites dashboard uses its
existing server-side `LLAMA_API_KEY` secret; no browser API key configuration is needed.
Copy `tools/studio-collector.py` to the Studio and run its `--install` command to update
its existing launch daemon. The dashboard defaults to three hours.

## Validation on the Studio

Run one deterministic coding prompt, then continue conversations A, B, C, and A again.
Inspect `cached_tokens`, restored prefix length and SSD/RAM transitions. Compare a restored
continuation with an uninterrupted run of the same token history, temperature and seed.
Test two active long conversations before increasing context. Monitor swap and memory;
short-prompt throughput does not establish long-context or concurrent performance.
