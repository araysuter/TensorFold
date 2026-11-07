"""Conservative macOS admission against a profile's full-context physical budget."""
import ctypes
import re
import struct
import subprocess
import sys

from errors import GatewayError

GIB = 1024 ** 3


def snapshot():
    if sys.platform != "darwin":
        raise OSError("Memory admission requires macOS")
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
    page = int(re.search(r"page size of (\d+) bytes", raw)[1])
    counts = {name: int(count) for name, count in re.findall(r"^([^:\n]+):\s*(\d+)\.", raw, re.M)}
    available = (counts["Pages free"] + counts["Pages inactive"]) * page
    pressure = int(subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=2))
    return available, pressure


def group_footprint(group):
    raw = subprocess.check_output(["/bin/ps", "-axo", "pid=,pgid="], text=True, timeout=2)
    pids = [int(pid) for pid, pgid in (line.split() for line in raw.splitlines()) if int(pgid) == group]
    if not pids:
        raise OSError("Backend group disappeared")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    total = 0
    for pid in pids:
        buffer = ctypes.create_string_buffer(4096)
        if library.proc_pid_rusage(pid, 4, ctypes.byref(buffer)) != 0:
            raise OSError("Cannot measure backend footprint")
        total += struct.unpack_from("Q", buffer.raw, 72)[0]
    return total


def check_memory(model, reserve_gib, backend=None):
    if not model.memory_required_gib:
        return
    try:
        available, pressure = snapshot()
        credit = 0
        if backend and backend.model.id == model.id and backend.alive():
            credit = min(group_footprint(backend.process.pid), int(model.memory_required_gib * GIB))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        raise GatewayError(503, "Memory status unavailable; retry later", "memory_status_unavailable") from None
    required = int((model.memory_required_gib + reserve_gib) * GIB) - credit
    if pressure != 1 or available < required:
        raise GatewayError(503, "Insufficient RAM headroom for this model; retry after other tasks finish",
                           "insufficient_memory")
