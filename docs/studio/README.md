# Astra Studio configuration

This fork adds conversation monitoring and Bearer authentication to TensorFold 0.5.0.
It reuses upstream `--spill-gib` for SSD snapshots. The BF16 cache and decoding kernels
are unchanged. `tools/start-studio.sh` selects Swift's tested snapshot, two compute lanes,
a 131,072-token prompt-plus-response limit, eight tracked conversations and a 128 GiB
SSD spill budget. Idle RAM checkpoints remain until the cache limit or memory admission
requires eviction. There is no fixed reservation of two complete 128k windows: the
existing memory controller can queue the second request when both would not fit.

The Swift optimization branch and its Studio validation steps are described in
[optimization.md](optimization.md).

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
buffers. Shared prefixes are reported in a separate pool instead of being charged to multiple slots.

Clients should send a stable, unique `X-Conversation-ID` for each conversation. Without
one, the monitor follows fingerprints of assistant replies already returned by this server,
including tool calls. Changes to system/editor context do not create another identity when
a unique latest reply identifies the continuation. New chats with identical openings stay
separate. Ambiguous branches, histories with no recognized reply, and auxiliary requests
are separate entries; the server never identifies a chat merely by shared cached tokens.
Explicit IDs remain the only guaranteed identity across compaction or rewritten history.

Tracking format 2 discards the older, unreliable inventory on its first startup. This only
resets monitor entries, never safetensors snapshots or the historical metrics database.
The entries repopulate as requests arrive. Restarting persists the new reply fingerprints.
No prompt text, token IDs, conversation IDs or reply fingerprints appear in monitoring APIs.

Cache accounting assigns each retained checkpoint record to exactly one matching entry,
a shared pool, or an unassigned pool. A shared prefix is not billed independently to every
matching entry. Slot `prompt_tokens` is the last request size, not resident cache occupancy.
`checkpoint_tokens` is the longest exclusively assigned checkpoint; `shared_checkpoint_tokens`
is the longest reusable shared checkpoint. Checkpoint bytes describe retained records, not
unique physical allocations (underlying arrays may share storage). Host/process memory is
the authoritative allocation measurement. Active working buffers are not idle checkpoints.

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

States: empty, queued, restoring, prefilling, generating, saving, ram, ssd, uncached, error, shared.
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

### Long requests through Cloudflare

Use `stream: true` for long-running inference through the public API. Streaming
responses send an immediate SSE comment and a keepalive comment every 15 seconds,
including while queued or prefilling. These comments contain no generated tokens
and standard SSE clients ignore them. Final usage and TensorFold timing metrics
remain in the final data event. Non-streaming requests still wait for the whole
response and can exceed the proxy read timeout; increasing a client's timeout
alone does not change that limit.
