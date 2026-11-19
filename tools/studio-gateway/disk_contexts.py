"""One LRU across gateway snapshots and native LFM2 whole-state SSD images."""
import os
import re
import stat
import time


def native_states(root):
    disk = root / "liquid-disk"
    if not disk.exists() or disk.is_symlink():
        return []
    found = []
    for directory in disk.iterdir():
        if not re.fullmatch(r"[a-f0-9]{16}", directory.name) or directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.glob("*.st"):
            if not re.fullmatch(r"[a-f0-9]{32}\.st", path.name):
                continue
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    continue
                used = path.with_suffix(".used")
                touched = info.st_mtime
                if used.exists() and not used.is_symlink():
                    touched = max(touched, used.stat().st_mtime)
                found.append((touched, path))
            except FileNotFoundError:
                continue
    return found


def prune(store, protected=None):
    offset = time.time() - time.monotonic()
    candidates = [(row.touched + offset, "gateway", row) for row in store.rows.values()]
    candidates += [(touched, "native", path) for touched, path in native_states(store.root)]
    candidates.sort(key=lambda entry: entry[0])
    limit = max(store.max_parked, int(protected is not None))
    while len(candidates) > limit:
        victim = next((entry for entry in candidates if entry[1] != "gateway" or entry[2] is not protected), None)
        if victim is None:
            return
        candidates.remove(victim)
        _, kind, value = victim
        if kind == "gateway":
            store.rows.pop((value.model, value.owner, value.id), None)
            value.path.unlink(missing_ok=True)
        else:
            value.unlink(missing_ok=True)
            value.with_suffix(".used").unlink(missing_ok=True)
