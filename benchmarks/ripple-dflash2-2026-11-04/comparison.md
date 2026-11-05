Each row is one attempt. Priming and warmups are excluded; settings are kept separate.

| Run | Prompt | Temp / seed | Chunk | Draft policy | Tokens | Identity | Decode plain → draft tok/s | Steady × | Request × | PP plain → draft tok/s | Draft peak MLX / physical GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failed_stock_width16 | prose (fresh, r0) | 0.0 / 17 | 512 | w? fixed, clear0 | 248 | FAIL | 12.24 → 12.05 | 0.98 | 1.12 | 11.82 → 75.85 | 38.89 / 39.42 |
| failed_stock_width4 | prose (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 238 | FAIL | 12.20 → 23.61 | 1.93 | 2.23 | 12.04 → 75.62 | 38.85 / 39.35 |
| selected_fixed4 | prose (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 243 | PASS | 12.31 → 16.29 | 1.32 | 1.48 | 16.96 → 73.03 | 38.85 / 39.36 |
| selected_fixed4 | prose (fresh, r0) | 0.7 / 17 | 512 | w4 fixed, clear0 | 206 | PASS | 12.26 → 16.11 | 1.31 | 1.32 | 55.70 → 76.05 | 38.85 / 39.34 |
| selected_fixed4 | code (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 168 | PASS | 12.24 → 22.26 | 1.82 | 1.79 | 64.08 → 78.56 | 38.86 / 39.33 |
| selected_fixed4 | code (fresh, r0) | 0.7 / 17 | 512 | w4 fixed, clear0 | 159 | PASS | 12.30 → 21.44 | 1.74 | 1.72 | 64.32 → 78.35 | 38.86 / 39.32 |
| fixed8_16_adaptive | prose (fresh, r0) | 0.0 / 17 | 512 | w8 fixed, clear0 | 243 | PASS | 12.27 → 10.52 | 0.86 | 0.99 | 10.85 → 75.79 | 38.85 / 39.38 |
| fixed8_16_adaptive | prose (fresh, r0) | 0.0 / 17 | 512 | w16 fixed, clear0 | 243 | PASS | 12.27 → 5.60 | 0.46 | 0.53 | 10.85 → 75.60 | 38.90 / 39.40 |
| fixed8_16_adaptive | prose (fresh, r0) | 0.0 / 17 | 512 | w4 adaptive, clear0 | 243 | PASS | 12.27 → 13.69 | 1.12 | 1.29 | 10.85 → 75.58 | 38.87 / 39.35 |
| fixed8_16_adaptive | code (fresh, r0) | 0.0 / 17 | 512 | w8 fixed, clear0 | 168 | PASS | 12.25 → 16.59 | 1.35 | 1.35 | 64.04 → 78.20 | 38.86 / 39.31 |
| fixed8_16_adaptive | code (fresh, r0) | 0.0 / 17 | 512 | w16 fixed, clear0 | 168 | PASS | 12.25 → 10.40 | 0.85 | 0.86 | 64.04 → 77.73 | 38.87 / 39.37 |
| fixed8_16_adaptive | code (fresh, r0) | 0.0 / 17 | 512 | w4 adaptive, clear0 | 168 | PASS | 12.25 → 18.77 | 1.53 | 1.52 | 64.04 → 78.22 | 38.86 / 39.32 |
| drafter8 | prose (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 243 | PASS | 12.31 → 16.16 | 1.31 | 1.51 | 11.61 → 75.87 | 40.58 / 41.10 |
| drafter8 | code (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 168 | PASS | 12.26 → 21.82 | 1.78 | 1.76 | 64.02 → 78.16 | 40.59 / 41.06 |
| prefill_chunks | 8192 (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 16 | unpaired | — → 16.54 | — | — | — → 315.51 | 40.60 / 41.21 |
| prefill_chunks | 8192 (fresh, r0) | 0.0 / 17 | 1024 | w4 fixed, clear0 | 16 | unpaired | — → 21.10 | — | — | — → 320.05 | 41.13 / 41.61 |
| prefill_chunks | 8192 (fresh, r0) | 0.0 / 17 | 2048 | w4 fixed, clear0 | 16 | unpaired | — → 20.90 | — | — | — → 308.30 | 41.63 / 42.12 |
| prefill_chunks | 32768 (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 16 | unpaired | — → 13.37 | — | — | — → 304.33 | 42.22 / 42.78 |
| prefill_chunks | 32768 (fresh, r0) | 0.0 / 17 | 1024 | w4 fixed, clear0 | 16 | unpaired | — → 16.81 | — | — | — → 306.86 | 42.72 / 43.20 |
| prefill_chunks | 32768 (fresh, r0) | 0.0 / 17 | 2048 | w4 fixed, clear0 | 16 | unpaired | — → 13.09 | — | — | — → 296.92 | 43.18 / 43.78 |
| prefill_allocator | 8192 (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear0 | 16 | unpaired | — → 21.24 | — | — | — → 323.02 | 40.64 / 41.36 |
| prefill_allocator | 8192 (fresh, r0) | 0.0 / 17 | 512 | w4 fixed, clear8192 | 16 | unpaired | — → 21.42 | — | — | — → 323.82 | 40.64 / 41.36 |

Sources:
- failed_stock_width16: failed; no separate weight/kernel primer recorded; cache pool — MiB; 1 failed paired cells; cross-chunk checks 0 pass/0 fail; 1 recorded failed checks.
- failed_stock_width4: failed; no separate weight/kernel primer recorded; cache pool — MiB; 1 failed paired cells; cross-chunk checks 0 pass/0 fail; 1 recorded failed checks.
- selected_fixed4: complete; no separate weight/kernel primer recorded; cache pool — MiB; 0 failed paired cells; cross-chunk checks 0 pass/0 fail; 0 recorded failed checks.
- fixed8_16_adaptive: complete; no separate weight/kernel primer recorded; cache pool — MiB; 0 failed paired cells; cross-chunk checks 0 pass/0 fail; 0 recorded failed checks.
- drafter8: complete; no separate weight/kernel primer recorded; cache pool — MiB; 0 failed paired cells; cross-chunk checks 0 pass/0 fail; 0 recorded failed checks.
- prefill_chunks: failed; short weight/kernel primer excluded; fresh KV per fresh-state cell; cache pool — MiB; 0 failed paired cells; cross-chunk checks 3 pass/1 fail; 1 recorded failed checks.
- prefill_allocator: complete; short weight/kernel primer excluded; fresh KV per fresh-state cell; cache pool 512.00 MiB; 0 failed paired cells; cross-chunk checks 0 pass/0 fail; 0 recorded failed checks.

Each row is one attempt; repetitions and workloads are not pooled.
Speed ratios compare plain and drafted cells within the same receipt and settings.
Prompt rate is the generator-reported prompt metric; TTFT and request elapsed remain separate.
Short-chat prompt rates can include first-use costs; use the primed prefill phase for PP claims.
Rates from failed or unpaired identity cells are diagnostic measurements, not qualified speedups.
Sampled physical peaks use periodic process-footprint readings; RSS is a process lifetime peak.
Target identity uses config and conversion-receipt hashes; target shards were not rehashed here.
Different drafter conversion identities are separate measurements, not a quality comparison.
The inputs include different drafter conversion identities.
