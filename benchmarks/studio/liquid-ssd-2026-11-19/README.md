# Liquid SSD context qualification

M5 Max, 64 GiB unified RAM, Metal and Zig 0.17.0. This extends the qualified
DSpark runtime at `2d5af28e8e82941cbfc261a9694a3dbafa54cca3`; it does not replace
the fork's current upstream engine wholesale. Target revision
`667408d494bde191c2cb9dd4f4549827816b9275`, DSpark revision
`e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`, drafter quantized to 8-bit at load.
No weights, kernels or arithmetic are changed.

The family exposes its convolution/attention snapshot and optional DSpark state
to the existing checksummed rows store. The image header distinguishes a real
drafter snapshot from a target-only state. Missing drafter state cannot masquerade
as a valid neural context. Kept marks follow the existing 2,048-row prompt grid.
A synchronous eviction writer uses the shared bounded pieces instead of copying
a whole snapshot into a second asynchronous staging buffer. Restores use a
16 MiB buffer with incremental checksum validation before publishing the blank
state. Whole-image speculative read-ahead is disabled for this opt-in codec.

The gateway retains at most eight opaque saved windows across native `.st`
images and existing Swift/Arete `.tfs` snapshots. A `.used` sidecar records native
cache hits for the shared LRU. Partial writes and symlinks are not eviction
candidates. A missing file after external LRU eviction is forgotten safely;
a resident state can be queued for a later write. Cache files remain private.

## Real-checkpoint checks

[Capacity and LRU receipt](capacity-and-lru.json):

- Created real Swift and Arete snapshots, then a Liquid snapshot in one gateway.
- Used a two-second idle timer to exercise the same unload path as production's
  1,800 seconds. Model count reached zero, with SSD snapshots retained.
- Reloaded Liquid, reused 6,144 of 8,000 tokens, and matched the cold/plain
  reference's full output hash. Two concurrent restored replies also matched.
- Nine distinct Liquid token-array contexts left eight native files total;
  older Swift/Arete windows were evicted by the shared limit.
- A 95,000-input/64-output request survived unload/reload, reused 94,208 tokens,
  and matched the previous cache-disabled runtime's reference hash.
- Minimum sampled available RAM was about 17.1 GiB, with normal pressure.
  Existing swap was unchanged during this probe.

[Sampled and tool receipt](sampled-and-tools.json): seeded temperature-0.6
cold/restored replies had equal hashes; required tool calls retained equal
hashes and valid arguments after SSD restore; tool-result continuation worked.
Swap was unchanged during that probe. Synthetic token arrays check execution
identity, not model answer quality.

## Host checks and limits

- Gateway tests: 71 passed, including shared eight-slot LRU, native touch order,
  incomplete files, symlinks, config boundaries and existing concurrency tests.
  The two LRU tests fail against the previous memory-only pruning function.
- Scoped native LFM/DSpark/lane tests passed; one inherited host skip remains.
  Snapshot-image tests check boundary copies, validation before mutation,
  integer overflow and zero padding for a declined drafter. The streamed-reader
  suite passed 22 tests, including a >16 MiB payload and corruption in its last
  piece; staging stayed bounded and a checksum failure rejected the result.
- The larger `test-cuda-host` suite has five inherited benchmark-fixture failures.
  A clean base run reproduces all five test names and the same three hash
  mismatches. These failures were not relabeled as passes or fixed in this work.
- Device scope is M5 Max Metal. Other chips, CUDA, HIP and Windows device paths
  have not been qualified by this receipt. The idle probe accelerates the timer;
  it does not claim a new 30-minute wall-clock observation.

The gateway keeps a three-GiB native RAM snapshot budget. Its 11.0625 GiB admission
reserve includes that workspace, 64 MiB of restore staging/slack and the requested
eight-GiB host reserve.
The maximum remains eight active Liquid requests; actual admission depends on
available RAM. One model process is resident at a time. Model weights already
exist on SSD; unloading releases the process instead of rewriting the weights.
Saved contexts survive model unloads and disappear when the gateway shuts down.
