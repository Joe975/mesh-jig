"""Keep mesh-jig's own CPU use small while something that needs the machine is running (a VR session).

The machine decides, never the model driving the jig: every entry point (`cli.main`, `mcp_server.main`, the
gauntlet runners) and every Blender build calls `govern()`, which reads the environment and sets this process's
CPU affinity and priority. What it starts afterwards (Blender, the sound runner) inherits both.

    MESH_JIG_CPUS=12-15                       the logical CPUs to keep to while limited ("12-15" or "12,13,14,15")
    MESH_JIG_YIELD_TO=vrcompositor.exe        limit only while one of these processes runs (comma separated);
                                              unset means always limit when MESH_JIG_CPUS is set
    MESH_JIG_PRIORITY=below_normal            while limited: below_normal (default) or idle

With MESH_JIG_CPUS unset nothing is changed. Limiting changes how long a build takes, never what it makes.
"""
from __future__ import annotations

import os
import subprocess
import sys

CPUS_ENV, YIELD_ENV, PRIORITY_ENV = "MESH_JIG_CPUS", "MESH_JIG_YIELD_TO", "MESH_JIG_PRIORITY"
PRIORITIES = {"normal": 0x0020, "below_normal": 0x4000, "idle": 0x0040}      # Windows priority classes
NICE = {"normal": 0, "below_normal": 10, "idle": 19}


def parse_cpus(text: str) -> list[int]:
    """'12-15' or '12,13,14,15' or '0-1,8' -> sorted logical CPU numbers."""
    cpus = set()
    for part in filter(None, (p.strip() for p in text.split(","))):
        lo, dash, hi = part.partition("-")
        if not lo.isdigit() or (dash and not hi.isdigit()):
            raise ValueError(f"{CPUS_ENV}: {part!r} is not a CPU number or range like 12-15")
        first, last = int(lo), int(hi or lo)
        if last < first:
            raise ValueError(f"{CPUS_ENV}: {part!r} runs backwards")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError(f"{CPUS_ENV} names no CPUs")
    return sorted(cpus)


def mask(cpus: list[int]) -> int:
    return sum(1 << c for c in cpus)


def running_names() -> set[str]:
    """Lower-case names of the processes running now."""
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return {line.split('","')[0].strip('"').lower() for line in out.splitlines() if line.startswith('"')}
    names = set()
    for pid in filter(str.isdigit, os.listdir("/proc")) if os.path.isdir("/proc") else ():
        try:
            with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                names.add(f.read().strip().lower())
        except OSError:
            pass
    if not names:                                       # macOS and others: ps
        out = subprocess.run(["ps", "-axco", "comm="], capture_output=True, text=True, timeout=10).stdout
        names = {line.strip().lower() for line in out.splitlines() if line.strip()}
    return names


def decide(env=os.environ, running=running_names, cpu_count=os.cpu_count) -> dict:
    """What the limit should be now: {"limited", "cpus", "priority", "because"}. Raises ValueError on bad settings."""
    text = env.get(CPUS_ENV, "").strip()
    if not text:
        return {"limited": False, "cpus": None, "priority": "normal", "because": f"{CPUS_ENV} is not set"}
    cpus = parse_cpus(text)
    total = cpu_count() or 0
    if total and cpus[-1] >= total:
        raise ValueError(f"{CPUS_ENV}={text}: this machine has CPUs 0-{total - 1}")
    priority = env.get(PRIORITY_ENV, "below_normal").strip() or "below_normal"
    if priority not in PRIORITIES:
        raise ValueError(f"{PRIORITY_ENV}={priority}: use one of {', '.join(PRIORITIES)}")
    wanted = [n.strip().lower() for n in env.get(YIELD_ENV, "").split(",") if n.strip()]
    if not wanted:
        return {"limited": True, "cpus": cpus, "priority": priority, "because": f"{CPUS_ENV} is set"}
    found = sorted(set(wanted) & running())
    if not found:
        return {"limited": False, "cpus": None, "priority": "normal", "because": f"none of {', '.join(wanted)} is running"}
    return {"limited": True, "cpus": cpus, "priority": priority, "because": f"{', '.join(found)} is running"}


def _apply_windows(cpus, priority):
    import ctypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    me = ctypes.c_void_p(k.GetCurrentProcess())
    if cpus is None:                                    # back to every CPU the system allows this process
        own, system = ctypes.c_size_t(), ctypes.c_size_t()
        k.GetProcessAffinityMask(me, ctypes.byref(own), ctypes.byref(system))
        wanted = system.value
    else:
        wanted = mask(cpus)
    if not k.SetProcessAffinityMask(me, ctypes.c_size_t(wanted)):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    if not k.SetPriorityClass(me, PRIORITIES[priority]):
        raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")


def _apply_posix(cpus, priority):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus if cpus is not None else range(os.cpu_count() or 1))
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if NICE[priority] > current:                        # a process can lower its priority, not raise it back
        os.setpriority(os.PRIO_PROCESS, 0, NICE[priority])


def apply(decision: dict, platform: str = sys.platform) -> None:
    (_apply_windows if platform == "win32" else _apply_posix)(decision["cpus"], decision["priority"])


_last: dict = {}


def govern(env=os.environ, running=running_names, applier=apply) -> dict:
    """Decide and apply the limit for this process now; returns the decision (also kept for `current()`).
    Bad settings are reported, not raised: a misconfigured limit must not stop a build."""
    global _last
    try:
        decision = decide(env, running)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        decision = {"limited": False, "cpus": None, "priority": "normal", "because": f"not applied: {e}"}
        _last = decision
        return decision
    if decision["limited"] or _last.get("limited") or env.get(CPUS_ENV, "").strip():
        try:
            applier(decision)
        except OSError as e:
            decision = {**decision, "limited": False, "because": f"not applied: {e}"}
    _last = decision
    return decision


def current() -> dict:
    """The last decision `govern` made in this process, for the record."""
    return dict(_last) if _last else {"limited": False, "cpus": None, "priority": "normal", "because": "not checked"}


def blender_threads(decision: dict) -> list[str]:
    """Blender sizes its thread pools to the whole machine; while limited, tell it how many CPUs it has."""
    return ["--threads", str(len(decision["cpus"]))] if decision.get("limited") else []


def describe(decision: dict) -> str:
    if not decision.get("limited"):
        return f"no limit ({decision.get('because', '')})"
    return f"CPUs {','.join(map(str, decision['cpus']))} at {decision['priority']} priority ({decision['because']})"
