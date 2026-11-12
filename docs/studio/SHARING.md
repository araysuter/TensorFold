# Sharing this fork

The source repository is public at [araysuter/TensorFold](https://github.com/araysuter/TensorFold).
Hugging Face model cards can link to its source and the
[two-model gateway instructions](../../tools/studio-gateway/README.md).
Link the commit or release used for a published measurement when reproducibility matters.
Model weights remain on Hugging Face; their licenses are separate from the engine license.

## Keep private state outside the checkout

| Material | Local location | Publish? |
| --- | --- | --- |
| Gateway source, placeholder configuration, synthetic tests | `tools/studio-gateway/` | Yes |
| Operator configuration | `~/.config/tensorfold/studio-gateway.json` | No |
| Owner and guest API key files | `~/.config/tensorfold/` | No |
| Downloaded model and drafter weights | `~/ai/models/` or the Hugging Face cache | Use the model Hub |
| Temporary conversations and native attention/recurrent state | Dedicated `cache_root`, such as `~/Library/Caches/TensorFold/studio-gateway` | No |
| Raw benchmark questions, answers, reasoning, and server logs | Private benchmark directory outside the checkout | No |
| Reviewed aggregate scores, timings, charts, and public checkpoint revisions | `benchmarks/studio/charts/` | Yes |

Runtime directories use mode `0700` and credential/state files use mode `0600`.
The gateway deletes its temporary conversation state on shutdown and removes its
own stale run directories at the next startup. Model weights are retained.
Deletion is ordinary file deletion, not a promise of secure erasure from SSDs or backups.
File permissions protect the local account boundary; they do not encrypt snapshots.

Use a separate, labeled, revocable key file for a guest. Keep key values in those
files, not shell command arguments, screenshots, model cards, or GitHub issues.
Examples refer to file paths and contain no usable credentials. The public
hostname belongs in private configuration; the reusable tools default to loopback.

The root `.gitignore` blocks common credential files, local configuration,
model weights, native `.tfs` conversation snapshots, generated evaluation tasks,
and `*-private.json` traces. An ignore rule does not remove a tracked file or
rewrite Git history, and `git add -f` overrides it. Add explicit source paths
when publishing; do not stage the entire working directory.

## Benchmark disclosure

Do not publish gated GPQA question text, answer keys, model response traces, or
the private task files. Share aggregate original-set and cleaned-set results
with their denominators, output budgets, retry policy, checkpoint revisions, and
backend settings. Keep cutoffs in the main accuracy score. Label any score that
excludes cutoffs as a diagnostic on a smaller subset.

The local Ornith conversion directory also contains generated tasks and raw
outputs from earlier, public datasets. Those files need their own data-license
and privacy review before inclusion, even when a model repository already
contains a copy. Publishing the gateway does not require publishing them.

## Before publishing a change

Review both tracked and newly added filenames with `git status --short` and
`git diff --cached --name-only`. Review the staged diff locally without copying
secrets into a chat or an issue. Confirm the expected exclusions, for example:

```sh
git check-ignore \
  .env \
  tools/studio-gateway/config.json \
  state/conversation.tfs \
  benchmarks/private-benchmarks/gpqa-tasks.jsonl \
  benchmarks/private-benchmarks/ornith-private.json
```

This fork's October 22, 2026 hygiene review at commit
`d13204b19ed124834864a6c3313f464652d601f9` checked 6,655 tracked files and the
23,120 unique blobs reachable from locally available refs (3,828 commits).
The targeted scan covered common provider-token formats, private-key markers,
password-bearing URLs, credential-like filenames, model weights, and private
evaluation paths. All credential-shaped matches were reviewed as synthetic
security fixtures or key parsing/generation code; no actual credential or gated
GPQA file was found. Binary contents, unreachable objects, remote-only refs,
GitHub artifacts, issues, and attachments were outside that review. A targeted
scan is not a guarantee that arbitrary secrets cannot be present.

Historical benchmark receipts include the operator's public API hostname and
machine configuration. A hostname is not an access credential; those receipts
do not grant access without a key. Prefer generic hosts in new examples and
sanitized counters in new receipts. No history rewrite is part of this change.
