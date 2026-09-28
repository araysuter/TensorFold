"""Small, CPU-only conversation inventory. GPU arrays stay owned by the scheduler."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
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
        if self.path is not None:
            try:
                data = json.loads(self.path.read_text())
                if data["model_id"] == model_id and len(data["slots"]) == slots:
                    self.slots = data["slots"]
                    self.keys = data["keys"]
                    self.prompts = {int(k): v for k, v in data["prompts"].items()}
                    for row in self.slots:
                        row.update(state="uncached", ram_bytes=0, ssd_bytes=0,
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
                    json.dump(dict(model_id=self.model_id, slots=self.slots, keys=self.keys,
                                   prompts=self.prompts), handle)
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
            if index is not None and index in self.jobs.values():
                raise RequestError("This conversation already has a pending request; wait for it to finish.")
            if index is None:
                available = [i for i in range(len(self.slots)) if i not in self.jobs.values()]
                if not available:
                    raise RequestError("All conversation slots are busy; retry when a request finishes.")
                index = min(available, key=lambda i: self.slots[i]["last_used"] or 0)
                self.keys = {k: v for k, v in self.keys.items() if v != index}
                self.keys[key] = index
            self.jobs[job.job_id] = index
            self.prompts[index] = list(job.prompt_ids)
            self.slots[index] = dict(slot=index + 1, state="queued", last_used=time.time(),
                                    prompt_tokens=len(job.prompt_ids), cached_tokens=0, generated_tokens=0,
                                    prompt_tps=None, decode_tps=None, rate_at=None,
                                    ram_bytes=0, ssd_bytes=0, checkpoint_tokens=0)

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

    def saving(self, entry: Any) -> None:
        with self.lock:
            for i, prompt in self.prompts.items():
                if i not in self.jobs.values() and prompt[:len(entry.tokens)] == entry.tokens:
                    self.slots[i]["state"] = "saving"

    def residency(self, entries: list[Any], blocks: list[Any]) -> None:
        # Called on the engine thread at lifecycle boundaries, never from an HTTP reader.
        with self.lock:
            for i, prompt in self.prompts.items():
                ram = [e for e in entries if not e.pinned and prompt[:len(e.tokens)] == e.tokens]
                disk = [(p, t) for p, t in blocks if prompt[:len(t)] == t and p.exists()]
                row = self.slots[i]
                row.update(ram_bytes=sum(e.nbytes for e in ram),
                           ssd_bytes=sum(p.stat().st_size for p, _ in disk),
                           checkpoint_tokens=max([len(e.tokens) for e in ram] + [len(t) for _, t in disk] + [0]))
                if i not in self.jobs.values() and row["state"] != "error":
                    row["state"] = "ram" if ram else "ssd" if disk else "uncached"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(timestamp=time.time(), slots=[dict(s) for s in self.slots], errors=self.errors)


def conversation_key(messages: list[dict], explicit: str | None = None) -> str:
    """Explicit IDs are preferred. First user message is a fallback, not an identity guarantee."""
    import json
    if explicit is not None:
        if not isinstance(explicit, str) or not 1 <= len(explicit) <= 256:
            raise RequestError("X-Conversation-ID must be 1 to 256 characters")
        value = explicit
    else:
        prefix = []
        for message in messages:
            prefix.append(message)
            if message.get("role") == "user":
                break
        value = json.dumps(prefix, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()
