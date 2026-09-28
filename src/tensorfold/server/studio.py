"""Small, CPU-only conversation inventory. GPU arrays stay owned by the scheduler."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any

from tensorfold.server.errors import RequestError


class Studio:
    def __init__(self, slots: int = 8, directory=None, model_id=""):
        self.path = None if directory is None else Path(directory) / "studio.json"
        self.model_id = model_id
        if slots < 1:
            raise ValueError("conversation slots must be positive")
        self.lock = threading.RLock()
        self.slots = [dict(slot=i + 1, state="empty", last_used=None) for i in range(slots)]
        self.prompts: dict[int, list[int]] = {}
        self.jobs: dict[str, int] = {}
        self.keys: dict[str, int] = {}
        self.errors = 0
        self.replies: dict[int, list[str]] = {}
        self.cache_stats: dict[str, int] = {}
        if self.path is not None:
            try:
                data = json.loads(self.path.read_text())
                if data.get("format") == 2 and data["model_id"] == model_id and len(data["slots"]) == slots:
                    self.slots = data["slots"]
                    self.keys = data["keys"]
                    self.prompts = {int(k): v for k, v in data["prompts"].items()}
                    self.replies = {int(k): v for k, v in data.get("replies", {}).items()}
                    for i, row in enumerate(self.slots):
                        row.update(state="uncached" if i in self.prompts else "empty", ram_bytes=0, ssd_bytes=0,
                                   prompt_tps=None, decode_tps=None)
            except (OSError, ValueError, KeyError, TypeError):
                pass

    def save(self):
        if self.path is None:
            return
        with self.lock:
            temp = self.path.with_suffix(".partial")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as handle:
                    json.dump(dict(format=2, model_id=self.model_id, slots=self.slots, keys=self.keys,
                                   prompts=self.prompts, replies=self.replies), handle)
                temp.replace(self.path)
            except OSError:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass


    def submit(self, job: Any) -> None:
        key = job.conversation_id or job.job_id
        with self.lock:
            index = self.keys.get(key)
            if key.startswith("auto:"):
                # A cached token prefix is NOT evidence of conversation identity. Match
                # assistant replies previously sent by this server, independent of system context.
                trail = tuple(getattr(job, "conversation_replies", ()))
                scores = {i: max((n for n, f in enumerate(trail) if f in replies), default=-1)
                          for i, replies in self.replies.items()
                          if not any(k.startswith("explicit:") and v == i for k, v in self.keys.items())}
                best = max(scores.values(), default=-1)
                candidates = [i for i, score in scores.items() if score == best and best >= 0]
                # A common ancestor can be in several branches; only a unique latest reply wins.
                index = candidates[0] if len(candidates) == 1 else None
            if index is not None and index in self.jobs.values():
                if key.startswith("auto:") and not getattr(job, "conversation_replies", ()):
                    index = None  # two independent chats can start with the same question
                else:
                    raise RequestError("This conversation already has a pending request; wait for it to finish.")
            if index is None:
                available = [i for i in range(len(self.slots)) if i not in self.jobs.values()]
                if not available:
                    raise RequestError("All conversation slots are busy; retry when a request finishes.")
                index = min(available, key=lambda i: self.slots[i]["last_used"] or 0)
                self.replies.pop(index, None)
            # Bound aliases and prevent old volatile keys from accumulating indefinitely.
            self.keys = {k: v for k, v in self.keys.items() if v != index}
            self.keys[key] = index
            self.jobs[job.job_id] = index
            self.prompts[index] = list(job.prompt_ids)
            self.slots[index] = dict(slot=index + 1, state="queued", last_used=time.time(),
                                    prompt_tokens=len(job.prompt_ids), cached_tokens=0, generated_tokens=0,
                                    prompt_tps=None, decode_tps=None, rate_at=None,
                                    ram_bytes=0, ssd_bytes=0, checkpoint_tokens=0, shared_checkpoint_tokens=0)

    def update(self, job: Any, **values: Any) -> None:
        with self.lock:
            index = self.jobs.get(job.job_id)
            if index is not None:
                self.slots[index].update(values)

    def finish(self, job: Any) -> None:
        with self.lock:
            index = self.jobs.pop(job.job_id, None)
            if index is not None:
                self.slots[index].update(state="error" if job.error else "idle", last_used=time.time(),
                                         prompt_tps=None, decode_tps=None)
                self.errors += int(job.error is not None)

    def record_reply(self, key: str, reply: dict) -> None:
        fingerprint = reply_fingerprint(reply)
        if not fingerprint:
            return
        with self.lock:
            index = self.keys.get(key)
            if index is not None:
                self.replies[index] = [*self.replies.get(index, [])[-15:], fingerprint]
                self.save()

    def saving(self, entry: Any) -> None:
        with self.lock:
            owners = [i for i, prompt in self.prompts.items() if prompt[:len(entry.tokens)] == entry.tokens]
            if len(owners) == 1 and owners[0] not in self.jobs.values():
                self.slots[owners[0]]["state"] = "saving"

    def residency(self, entries: list[Any], blocks: list[Any]) -> None:
        # A checkpoint is counted once. Shared prefixes belong to a shared pool,
        # never independently to every conversation which could reuse them.
        with self.lock:
            self.cache_stats = {f"{group}_{kind}_bytes": 0 for group in ("assigned", "shared", "unassigned")
                                for kind in ("ram", "ssd")}
            for i in self.prompts:
                self.slots[i].update(ram_bytes=0, ssd_bytes=0, checkpoint_tokens=0,
                                     shared_checkpoint_tokens=0, shared_ram=0, shared_ssd=0)
            records = [("ram", e.tokens, e.nbytes, e.pinned) for e in entries]
            for path, tokens in blocks:
                try:
                    records.append(("ssd", tokens, path.stat().st_size, False))
                except FileNotFoundError:
                    pass
            for kind, tokens, size, pinned in records:
                owners = [i for i, prompt in self.prompts.items() if prompt[:len(tokens)] == tokens]
                shared = pinned or len(owners) > 1
                group = "shared" if shared else "assigned" if owners else "unassigned"
                self.cache_stats[f"{group}_{kind}_bytes"] += size
                for i in owners:
                    row = self.slots[i]
                    if shared:
                        row[f"shared_{kind}"] = 1
                        row["shared_checkpoint_tokens"] = max(row["shared_checkpoint_tokens"], len(tokens))
                    else:
                        row[f"{kind}_bytes"] += size
                        row["checkpoint_tokens"] = max(row["checkpoint_tokens"], len(tokens))
            for i in self.prompts:
                row = self.slots[i]
                if i not in self.jobs.values() and row["state"] != "error":
                    row["state"] = ("ram" if row["ram_bytes"] else "ssd" if row["ssd_bytes"] else
                                    "shared" if row["shared_ram"] or row["shared_ssd"] else "uncached")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(timestamp=time.time(), tracking_version=2, slots=[dict(s) for s in self.slots],
                        cache_stats=dict(self.cache_stats), errors=self.errors)


def reply_fingerprint(message: dict) -> str | None:
    """Hash assistant content/tool calls, ignoring transient metadata and private reasoning."""
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    calls = message.get("tool_calls") or []
    if calls:
        normalized = []
        for call in calls:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            normalized.append(dict(id=call.get("id"), name=fn.get("name"), arguments=args))
        value = dict(tool_calls=normalized)  # clients can omit prose accompanying a call
    elif content:
        value = dict(content=content)
    else:
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def conversation_replies(messages: list[dict]) -> tuple[str, ...]:
    return tuple(f for m in messages if m.get("role") == "assistant" if (f := reply_fingerprint(m)))


def conversation_key(messages: list[dict], explicit: str | None = None) -> str:
    """Explicit IDs are stable; otherwise each request is matched by known assistant replies."""
    if explicit is not None:
        if not isinstance(explicit, str) or not 1 <= len(explicit) <= 256:
            raise RequestError("X-Conversation-ID must be 1 to 256 characters")
        return "explicit:" + hashlib.sha256(explicit.encode()).hexdigest()
    # Never confuse identical openings (or background requests) with a single conversation.
    return "auto:" + uuid.uuid4().hex
