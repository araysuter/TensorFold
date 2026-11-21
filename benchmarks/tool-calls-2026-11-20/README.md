# Function-tool protocol qualification

Base: `8abb2afc8b80743ac87a5da8543985465c4df625`. Candidate: the commit containing
this receipt. This checks API framing and tool-result replay, not model quality.
Raw synthetic responses are in [candidate.json](candidate.json); rerun with
[`tools/check_tool_calls.py`](../../tools/check_tool_calls.py).

| API model | Nonstreaming | Streaming | Extra automatic call after changed result |
| --- | --- | --- | --- |
| inkling-2-mini | pass | pass | none |
| lfm2.5-24b-a2b | pass | pass | one in each mode; explicit `none` retry passes |
| swift-1.5 | pass | pass | none |
| arete-27b | pass | pass | none |

Each case requests a function, validates its name/JSON arguments/ID and
`tool_calls` finish reason, replays its result, then repeats with a changed
result and checks IDs are not reused within the conversation. Streaming must
finish with `[DONE]`. An additional valid automatic call is recorded, followed
by an answer-only retry; it is not silently counted as a direct final answer.
Liquid's extra automatic call also occurs with native prompt caching disabled
and in the independent MLX reference on the same checkpoint/history. The model
acknowledges the changed value; this is not evidence of a stale cache.

Before the fix, Inkling returned native tool markup instead of structured
calls. Its old formatter fails all three chunk-size subcases of the new named
header regression. The other three models passed basic tool-result round trips
before the change. The patch changes Inkling response framing and explicit
`tool_choice: none` handling only; native model parsers remain unchanged.

Validation: 18 worker tests passed in the pinned MLX environment, 5 pure parser
tests passed locally, 1 checker negative control passed (plain-text tool output
must exit 2), and 71 gateway tests passed. Boundary tests include every split
point through reasoning, two tool blocks and final content. Lean checker:
221 inherited findings on base and candidate, zero new findings. No tensor,
sampler, cache layout, drafter or GPU kernel changed. Generation hashes and
speed benchmarks were not rerun: this receipt qualifies protocol translation,
not a numerical or performance optimization.

Test host: Apple M5 Max, 64 GB, macOS; MLX 0.32.4, MLX-LM
`0.32.3.dev26+g0c4c99192`. Inkling 8-bit weights revision
`ececf5ebe6175372addee86a260addaa5537ac55`; no drafter. Native engine revision
`2d5af28e8e82941cbfc261a9694a3dbafa54cca3`. Zig pin: 0.17.0;
no native rebuild was required. Other hardware/platforms were not exercised.
