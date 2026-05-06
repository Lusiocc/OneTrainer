from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _bytes_to_gib(n: int | float | None) -> float | None:
    if n is None:
        return None
    try:
        return float(n) / (1024.0**3)
    except Exception:
        return None


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    pid: int
    rss_bytes: int | None = None
    working_set_bytes: int | None = None
    private_bytes: int | None = None

    def to_log_dict(self) -> dict[str, float | int | None]:
        return {
            "pid": self.pid,
            "rss_gib": _bytes_to_gib(self.rss_bytes),
            "working_set_gib": _bytes_to_gib(self.working_set_bytes),
            "private_gib": _bytes_to_gib(self.private_bytes),
        }


def get_process_memory_snapshot(pid: int | None = None) -> ProcessMemorySnapshot:
    pid = int(pid or os.getpid())

    if psutil is None:
        return ProcessMemorySnapshot(pid=pid)

    try:
        p = psutil.Process(pid)
        mi = p.memory_info()
        rss = getattr(mi, "rss", None)

        # On Windows, memory_full_info() may contain a closer “private bytes” equivalent.
        private_bytes = None
        working_set = None
        try:
            mfi = p.memory_full_info()
            # Windows: `private` is private bytes; on other platforms it may not exist.
            private_bytes = getattr(mfi, "private", None)
        except Exception:
            pass

        # For Task Manager “Memory” we most often want the working set; `rss` is the best proxy.
        # On Windows, rss is generally the working set.
        working_set = rss

        return ProcessMemorySnapshot(
            pid=pid,
            rss_bytes=int(rss) if rss is not None else None,
            working_set_bytes=int(working_set) if working_set is not None else None,
            private_bytes=int(private_bytes) if private_bytes is not None else None,
        )
    except Exception:
        return ProcessMemorySnapshot(pid=pid)


def trim_process_working_set_windows() -> bool:
    """
    Best-effort request to the OS to reduce this process working set.
    On Windows this often lowers Task Manager 'Memory' after large allocations are freed.
    """
    if platform.system() != "Windows":
        return False

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        # SIZE_T(-1) sentinel requests trimming.
        size_t_neg1 = ctypes.c_size_t(-1).value
        res = kernel32.SetProcessWorkingSetSize(handle, size_t_neg1, size_t_neg1)
        return bool(res)
    except Exception:
        return False

