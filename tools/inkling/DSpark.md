# Experimental DSpark benchmark

Inkling 2 Mini ships a three-layer DSpark head in the original checkpoint's
`mtp.safetensors`. Its published MLX-LM port does not include DSpark. The
standalone [benchmark](bench_dspark.py) adapts the head definitions from oMLX,
captures decoder visits 5, 23, 41 and 48, verifies proposals against the target,
and rewinds rejected KV and short-convolution state.

This tool is excluded from the API worker. The tested verifier was slower than
ordinary decoding, so the deployed profile keeps ordinary decoding. It preserves
the single-token attention and projection paths while checking draft blocks;
the faster batched-arithmetic experiment failed token equality and was rejected.
No target weights, cache precision or layer passes are changed.

On M5 Max with 64 GiB, MLX 0.32.4 and MLX-LM
`0c4c99192230f1943dd468d297f0b8a64ae7b7c2`, one user, fresh KV and 256 generated
tokens, with one measured attempt per cell:

| Input tokens | Plain prompt tok/s | DSpark prompt tok/s | Plain generation tok/s | DSpark generation tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4,000 | 1,738 | 1,751 | 42.5 | 41.4 |
| 8,000 | 1,706 | 1,694 | 41.7 | 40.7 |
| 32,000 | 1,414 | 1,411 | 38.3 | 17.7 |

All three candidate outputs matched their paired plain outputs token for token.
The plain hashes also matched the serving API's solo oracles. These paired
runs capture target features in both arms; see the separate serving matrix
receipt for API rates. Results are speed/correctness probes, not quality scores
or qualification for other chips, sampled generation, images or two drafted
requests. Draft layer projections use 8-bit affine weights; the remaining draft
weights and the target are unchanged. Verification width is two.

## Reproduce

Use the worker's dedicated, pinned environment and the existing 5-bit target.
Stop other inference before this GPU benchmark. Choose Time Machine-excluded
absolute directories for draft downloads and results. Only the 636 MB head is
needed from the original checkpoint; do not download the full source again.

```sh
mkdir -p "$DRAFT_DIR" "$BENCH_DIR"
tmutil isexcluded "$DRAFT_DIR" "$BENCH_DIR"
"$INKLING_ENV/bin/hf" download thinkingmachines/Inkling-2-Mini mtp.safetensors \
  --revision 6e7c8248f75265bdd4fbab5273304243b5d8c2cb --local-dir "$DRAFT_DIR"
"$INKLING_ENV/bin/python" tools/inkling/bench_dspark.py \
  --model "$MODEL_DIR" --draft "$DRAFT_DIR/mtp.safetensors" \
  --lengths 4000,8000,32000 --tokens 256 --width 2 --draft-bits 8 \
  --output "$BENCH_DIR/dspark.json"
```

The tool requires 32 GiB available before loading and aborts below an 8 GiB
available-memory floor. It exits nonzero on token disagreement. It reads local
weights, performs no upload, and stores synthetic token outputs and timings
without machine paths. Width four and BF16 draft projections are experimental
comparison options, not serving profiles. [Measured receipt](receipts/dspark-m5max.json).

The [vendor notice](dspark_bench/NOTICE) identifies the Apache-2.0 draft model
source. No weights are included in this repository.
