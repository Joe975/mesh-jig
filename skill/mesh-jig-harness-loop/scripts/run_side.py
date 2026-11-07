"""One side's runs of a harness-loop generation, built by `mesh-jig agent` a few at a time.

    python run_side.py <runs dir> --seed <seed project> --count 4 [--at-a-time 2] [--stagger 600] [--reruns 2]
        [--run-timeout 10800] [--server-url http://localhost:1234 --server-command <file>] [--monitor 60]
        -- --model <model> --url <endpoint> --attempts 3 --effort low --max-wall 5400

What follows `--` is given to `mesh-jig agent <run>` as it stands, the same for every run.

Each run is a fresh copy of the seed in <runs dir>/r01, r02, ...: its reference views cut (`mesh-jig sheet`), then
`harness-commit.txt` (the checkout's commit and the builder's command), then the builder, its output in r01.log
beside it. At most `--at-a-time` builders run, and one starts no sooner than `--stagger` seconds after any that is
still going: two builders that open together write their longest replies together.

A run that ends with no measured attempt died. It is run again under the next free name, `--reruns` times a side at
most, and the summary says which run each stands in for. A run that built badly is a result and stays.

With `--server-url`, the endpoint's /health is asked before each start. When it does not answer, the server is
started from the one command line in `--server-command` (a file outside the repository's tracked files: the command
names a machine's paths; quote with double quotes and write paths with forward slashes) and waited for. Two starts
in a row that fail end the side, and the summary says so. `--monitor N` notes every N seconds, in monitor.log, the
card's memory, the host's free memory and the largest context the server has held.

<runs dir>/summary.json is written when every run has ended, and after each one: a run's name, whether it finished,
its attempts and their scores, its wall time, the largest prompt and reply of one call, what it stands in for.

Exit codes: 0 every run delivered, 1 the side is short of runs or the server would not start, 2 the request was wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from mesh_jig import harness, project, viewer

SUMMARY_FILE, COMMIT_FILE, MONITOR_FILE = "summary.json", "harness-commit.txt", "monitor.log"
BUILDER = "mesh-jig agent <run>"
SERVER_WAIT_S = 300.0
SERVER_STARTS = 2           # starts in a row that fail before the side is given up


class Refused(ValueError):
    """The request cannot be carried out as given."""


def run_name(index: int) -> str:
    return f"r{index:02d}"


def schedule(count: int, launch, delivered, *, at_a_time: int = 2, stagger: float = 600.0, reruns: int = 2,
             run_timeout: float = 0.0, ensure_server=lambda: True, on_end=lambda rows: None,
             now=time.monotonic, sleep=time.sleep, poll_s: float = 5.0) -> dict:
    """Start `count` runs and the re-runs of those that died, and wait for all of them.

    `launch(name)` starts a builder and returns its handle (`poll()`, `kill()`, `wait()`, as a Popen has);
    `delivered(name)` says whether an ended run has a measured attempt. Returns {"runs": [a row a run, in the order
    they ended], "stopped": why the side ended early, or ""}. A row: name, started_s and wall_s on this call's clock,
    exit, delivered, rerun_of, timed_out."""
    if count < 1 or at_a_time < 1 or stagger < 0 or reruns < 0:
        raise Refused("count and at-a-time must be positive; stagger and reruns cannot be negative")
    t0 = now()
    pending = [(run_name(i), None) for i in range(1, count + 1)]
    following, running, rows, failed_starts, stopped = count + 1, {}, [], 0, ""
    while pending or running:
        for name, (handle, started, rerun_of) in list(running.items()):
            code, timed_out = handle.poll(), False
            if code is None and run_timeout and now() - started >= run_timeout:
                handle.kill()
                code, timed_out = handle.wait(), True
            if code is None:
                continue
            del running[name]
            ok = bool(delivered(name))
            rows.append({"name": name, "started_s": round(started - t0, 1), "wall_s": round(now() - started, 1),
                         "exit": code, "delivered": ok, "rerun_of": rerun_of, "timed_out": timed_out})
            if not ok and reruns and not stopped:
                reruns -= 1
                pending.append((run_name(following), name))
                following += 1
            on_end(rows)
        clear = all(now() - started >= stagger for _handle, started, _of in running.values())
        if pending and len(running) < at_a_time and clear:
            if ensure_server():
                failed_starts = 0
                name, rerun_of = pending.pop(0)
                running[name] = (launch(name), now(), rerun_of)
                continue
            failed_starts += 1
            if failed_starts >= SERVER_STARTS:
                stopped = f"the server did not start, {failed_starts} times in a row"
                pending.clear()
        if pending or running:
            sleep(poll_s)
    return {"runs": rows, "stopped": stopped}


def prepare(seed: Path, run: Path, builder_args: list[str], cut=None) -> None:
    """A fresh copy of the seed at `run`, its reference views cut, and what it is about to be built by."""
    if run.exists():
        raise Refused(f"{run} exists: a run needs a fresh directory")
    shutil.copytree(seed, run)
    (cut or cut_views)(run)
    version = harness.version() or {}
    commit = str(version.get("commit") or "unknown") + (" (working tree changed)" if version.get("dirty") else "")
    (run / COMMIT_FILE).write_text(f"{commit}\n{BUILDER} {shlex.join(builder_args)}\n", encoding="utf-8")


def cut_views(run: Path) -> None:
    if (project.load(run).data.get("sheet") or {}).get("source"):
        subprocess.run([sys.executable, "-m", "mesh_jig", "sheet", str(run)], check=True, capture_output=True)


def builder(run: Path, builder_args: list[str], out):
    """`mesh-jig agent <run>` with the side's options, its output to `out`."""
    return subprocess.Popen([sys.executable, "-m", "mesh_jig", "agent", str(run), *builder_args],
                            stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def delivered(run: Path) -> bool:
    """Whether a run has a measured attempt, which is what `generation.py sheets` needs of it."""
    try:
        return viewer.project_record(project.load(run), run.name)["best"] is not None
    except (project.ProjectError, OSError, ValueError):
        return False


def report_of(run: Path) -> dict:
    """The run's last `report.json`, or {}."""
    found = sorted((run / "agent").glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(found[-1].read_text(encoding="utf-8")) if found else {}
    except (OSError, json.JSONDecodeError):
        return {}


def run_row(run: Path, row: dict) -> dict:
    """A scheduling row with what the run's own report says of it."""
    rep = report_of(run)
    usage = [c.get("usage") or {} for c in rep.get("calls") or []]
    return {**row, "finished": bool(rep.get("done")), "error": rep.get("error") or "",
            "attempts": [{"name": a.get("name"), "numeric_score": a.get("numeric_score")}
                         for a in rep.get("attempts") or []],
            "recoveries": rep.get("recoveries") or [], "builder_turns": rep.get("turns"),
            "model_calls": len(usage),
            "output_tokens": sum(int(u.get("completion_tokens") or 0) for u in usage),
            "largest_prompt_and_reply": max((int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
                                             for u in usage), default=0)}


def write_summary(runs_dir: Path, rows: list[dict], count: int, stopped: str = "", done: bool = False) -> dict:
    full = [run_row(runs_dir / r["name"], r) for r in rows]
    summary = {"complete": done, "asked": count, "delivered": sum(r["delivered"] for r in full),
               "died": sum(not r["delivered"] for r in full), "reruns": sum(r["rerun_of"] is not None for r in full),
               "stopped": stopped, "runs": full}
    temp = runs_dir / (SUMMARY_FILE + ".tmp")
    temp.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    temp.replace(runs_dir / SUMMARY_FILE)
    return summary


def answers(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, ValueError):
        return False


def start_server(command_file: Path, log: Path, url: str, wait_s: float = SERVER_WAIT_S, sleep=time.sleep) -> bool:
    """Start the server from the command line in `command_file`, detached from this process, and wait for it."""
    command = shlex.split(command_file.read_text(encoding="utf-8").strip())
    if not command:
        raise Refused(f"{command_file} holds no command")
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    with log.open("ab") as out:
        subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=flags, start_new_session=os.name != "nt")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if answers(url):
            return True
        sleep(3.0)
    return False


def card_memory() -> str:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
        used, total = done.stdout.strip().splitlines()[0].split(",")
        return f"vram {int(used)}/{int(total)} MiB"
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return "vram ?"


def host_memory() -> str:
    """Free physical memory and what is left of the commit limit, where the platform says."""
    try:
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "free", "commit_limit", "commit_free", "virtual",
                                                            "virtual_free", "extended")]
            status = Status(length=ctypes.sizeof(Status))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return (f"host free {status.free / 2 ** 30:.1f} GiB, "
                    f"commit left {status.commit_free / 2 ** 30:.1f} of {status.commit_limit / 2 ** 30:.1f} GiB")
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return f"host free {int(fields['MemAvailable'].split()[0]) / 2 ** 20:.1f} GiB"
    except (OSError, KeyError, ValueError, AttributeError):
        return "host ?"


def context_peak(url: str) -> str:
    """llama-server's largest context so far (`--metrics`), and how many requests it is working on."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
            lines = response.read().decode("utf-8", "replace").splitlines()
        seen = {k: v for k, _, v in (line.partition(" ") for line in lines if line.startswith("llamacpp:"))}
        return (f"context peak {int(float(seen['llamacpp:n_tokens_max']))}, "
                f"processing {int(float(seen.get('llamacpp:requests_processing', 0)))}")
    except (OSError, urllib.error.URLError, KeyError, ValueError):
        return "context ?"


def monitor(path: Path, url: str | None, every: float, stop: threading.Event) -> None:
    while not stop.wait(every):
        line = ", ".join([time.strftime("%Y-%m-%d %H:%M:%S"), card_memory(), host_memory()]
                         + ([context_peak(url)] if url else []))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def side(runs_dir: Path, seed: Path, count: int, builder_args: list[str], *, at_a_time: int = 2,
         stagger: float = 600.0, reruns: int = 2, run_timeout: float = 0.0, server_url: str | None = None,
         server_command: Path | None = None, monitor_every: float = 0.0, log=print) -> dict:
    """Build one side's runs under `runs_dir` and write its summary; the summary."""
    if runs_dir.exists() and any(runs_dir.iterdir()):
        raise Refused(f"{runs_dir} is not empty: a side needs a fresh directory")
    if not (seed / project.JIG_FILE).is_file():
        raise Refused(f"{seed} is not a project (no {project.JIG_FILE})")
    if server_command and not server_url:
        raise Refused("--server-command needs --server-url")
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs = []

    def launch(name: str):
        run = runs_dir / name
        prepare(seed, run, builder_args)
        out = (runs_dir / f"{name}.log").open("wb")
        logs.append(out)
        log(f"{time.strftime('%H:%M:%S')} start {name}")
        return builder(run, builder_args, out)

    def ensure_server() -> bool:
        if not server_url or answers(server_url):
            return True
        if not server_command:
            log(f"{server_url} does not answer and there is no --server-command to start it from")
            return False
        log(f"{time.strftime('%H:%M:%S')} {server_url} does not answer: starting the server")
        started = start_server(server_command, runs_dir / "server.log", server_url)
        log("the server answers" if started else "the server did not start")
        return started

    def on_end(rows: list[dict]) -> None:
        last = write_summary(runs_dir, rows, count)["runs"][-1]
        scores = " ".join(f"{a['numeric_score']:.3f}" for a in last["attempts"] if a["numeric_score"] is not None)
        log(f"{time.strftime('%H:%M:%S')} end   {last['name']} after {last['wall_s']:.0f} s: "
            + (f"scores {scores}" if last["delivered"] else f"DIED ({last['error'] or 'exit ' + str(last['exit'])})"))

    stop = threading.Event()
    if monitor_every:
        threading.Thread(target=monitor, args=(runs_dir / MONITOR_FILE, server_url, monitor_every, stop),
                         daemon=True).start()
    try:
        result = schedule(count, launch, lambda name: delivered(runs_dir / name), at_a_time=at_a_time,
                          stagger=stagger, reruns=reruns, run_timeout=run_timeout, ensure_server=ensure_server,
                          on_end=on_end)
    finally:
        stop.set()
        for out in logs:
            out.close()
    return write_summary(runs_dir, result["runs"], count, result["stopped"], done=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, help="a fresh directory for this side's runs, as g03/runs")
    ap.add_argument("--seed", type=Path, required=True)
    ap.add_argument("--count", type=int, required=True, help="runs a side")
    ap.add_argument("--at-a-time", type=int, default=2)
    ap.add_argument("--stagger", type=float, default=600.0, help="seconds between two starts (default 600)")
    ap.add_argument("--reruns", type=int, default=2, help="dead runs run again, at most (default 2)")
    ap.add_argument("--run-timeout", type=float, default=10800.0,
                    help="a builder still going after this many seconds is killed (default 10800; 0 for never)")
    ap.add_argument("--server-url", help="the endpoint's root, asked for /health before each start")
    ap.add_argument("--server-command", type=Path, help="a file holding the command line that starts the server")
    ap.add_argument("--monitor", type=float, default=0.0, help=f"seconds between lines of {MONITOR_FILE} (default: none)")
    return ap


def arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """The side's own options, and what follows the first `--`: the options of `mesh-jig agent <run>`."""
    cut = argv.index("--") if "--" in argv else len(argv)
    return parser().parse_args(argv[:cut]), argv[cut + 1:]


def main(argv: list[str] | None = None) -> int:
    a, builder_args = arguments(sys.argv[1:] if argv is None else argv)
    try:
        summary = side(a.runs, a.seed, a.count, builder_args, at_a_time=a.at_a_time, stagger=a.stagger,
                       reruns=a.reruns, run_timeout=a.run_timeout, server_url=a.server_url,
                       server_command=a.server_command, monitor_every=a.monitor,
                       log=lambda s: print(s, flush=True))
    except (Refused, project.ProjectError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"{summary['delivered']} of {summary['asked']} runs delivered, {summary['died']} died, "
          f"{summary['reruns']} run again" + (f"; stopped: {summary['stopped']}" if summary["stopped"] else "")
          + f" -> {a.runs / SUMMARY_FILE}", flush=True)
    return 0 if summary["delivered"] >= summary["asked"] and not summary["stopped"] else 1


if __name__ == "__main__":
    sys.exit(main())
