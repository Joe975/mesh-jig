"""evaluate: one call that builds, renders and measures a model script, for an agent that authors the script itself.

What such an agent needs is not a prompt but a verdict on each attempt, in seconds, that does not drift and costs
nothing unless it asks for a judge. This is that verdict:

1. paint the atlas from a plan when given one, build every variant in Blender, hold the GLBs to the contract;
2. render the judged variant on the CPU from the cameras the reference views are named after;
3. measure: outline error per station against the reference (`profile`), silhouette overlap (`silhouette`), colour
   zones and edge structure in place (`resemblance`), and any gate the contract sets on them;
4. optionally, ONE pairwise judge comparison against another attempt's renders: the only step that costs money or
   can be wrong about the same input twice, so the agent decides when to spend it.

Everything lands in the attempt directory: out/ (script copy, GLBs, atlas, glb_stats.json), renders/, compare/
(side-by-sides and outline diffs), numeric.json, eval.json, and one line appended to the project's ledger, which is
what the agent reads back next time. `ok` means built, in contract, rendered and measured; the judge's verdict never
changes it (a measured attempt judged worse is still a valid attempt).

The record `evaluate` returns holds absolute paths, for a caller to open. eval.json holds the same paths relative to
the attempt directory (`portable`), so a record names no machine and survives a copy; `read` gives them back absolute.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import statistics
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from . import build, comparison_context, criteria, glb, harness, judge as judge_mod, paint, profile, readiness, render, resemblance
from . import references, review, silhouette
from .project import Project

EVAL_FILE = "eval.json"
NUMERIC_FILE = "numeric.json"
ORTHO_VIEWS = silhouette.ORTHO_VIEWS
PATH_FIELDS = ("script", "against")                     # a path each
PATH_MAPS = ("renders", "compare", "ruled", "overlays", "closeups", "zone_maps")    # view -> path; a close-up is region -> path
TEXT_FIELDS = ("error", "pairwise_error")               # what a build or a judge said, which can name a path
LEDGER_HEAD = ("| attempt | build | outline mm | overlap | resemblance | zones | numeric | gates | triangles | judge "
               "| note |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
# Keep this as a module constant so lock timeout tests can shorten it without changing append_ledger's API.
_LEDGER_LOCK_TIMEOUT_SECONDS = 10.0
_LEDGER_LOCK_RETRY_SECONDS = 0.05
VIEW_MOVE = 0.005       # `standing_lines` names a view whose overlap or zones moved by this much against another attempt
FAILURE_SNAPSHOTS = "failed-evaluations"
_SNAPSHOT_ROOTS = ("out", "renders", "compare")
_SNAPSHOT_FILES = (NUMERIC_FILE, "resemblance.json", "silhouette.json")


def _linkish(path: Path, info=None) -> bool:
    """True for symlinks and Windows junction/reparse-point paths."""
    info = info or path.lstat()
    attrs = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _snapshot_files(attempt: Path, excluded: list[str] | None = None) -> list[tuple[Path, str]]:
    """Enumerate only evaluator-owned output roots and known measurement files, without following symlinks."""
    found: list[tuple[Path, str]] = []
    for name in _SNAPSHOT_ROOTS:
        root = attempt / name
        try:
            root_stat = root.lstat()
        except FileNotFoundError:
            continue
        if _linkish(root, root_stat):
            if excluded is not None:
                excluded.append(name)
            continue
        if not root.is_dir():
            raise OSError(f"owned output root {name} is not a directory")
        stack = [(root, name)]
        while stack:
            directory, relative = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError as error:
                raise OSError(f"could not inventory owned output directory {relative}: {error}") from error
            for entry in entries:
                path = Path(entry.path)
                rel = f"{relative}/{entry.name}"
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or bool(getattr(entry_stat, "st_file_attributes", 0) &
                                                   getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                        if excluded is not None:
                            excluded.append(rel)
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((path, rel))
                    elif entry.is_file(follow_symlinks=False):
                        found.append((path, rel))
                except OSError as error:
                    raise OSError(f"could not inspect owned output {rel}: {error}") from error
    for name in _SNAPSHOT_FILES:
        path = attempt / name
        try:
            info = path.lstat()
            if _linkish(path, info):
                if excluded is not None:
                    excluded.append(name)
            elif path.is_file():
                found.append((path, name))
        except FileNotFoundError:
            continue
        except OSError as error:
            raise OSError(f"could not inspect owned output {name}: {error}") from error
    return sorted(found, key=lambda item: item[1])


def _snapshot_hashes(attempt: Path) -> tuple[dict[str, str], str | None]:
    """Hash the known owned outputs at evaluation entry to distinguish old bytes from later changes."""
    hashes = {}
    try:
        files = _snapshot_files(attempt)
    except OSError as error:
        return {}, f"{type(error).__name__}: {error}"
    for path, relative in files:
        try:
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            return hashes, f"{type(error).__name__}: could not hash {relative}: {error}"
    return hashes, None


def _long_windows_path(path: Path) -> Path:
    """Use the Win32 extended-length spelling for snapshot paths near MAX_PATH."""
    if os.name != "nt" or str(path).startswith("\\\\?\\") or len(str(path)) < 240:
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _preserve_failed_evaluation(attempt: Path, root: Path, rec: dict, source_bytes: bytes | None,
                                source_error: str | None, initial_hashes: dict[str, str],
                                inventory_error: str | None = None) -> dict:
    """Publish a complete immutable snapshot, atomically exposing it only after every file is written."""
    snapshots = _long_windows_path(attempt / FAILURE_SNAPSHOTS)
    try:
        existing = snapshots.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and (_linkish(snapshots, existing) or not snapshots.is_dir()):
        raise OSError("failed-evaluations exists but is not a plain directory")
    snapshots.mkdir(exist_ok=True)
    snapshot_name = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}"
    staging = snapshots / f".incomplete-{uuid.uuid4().hex}"
    final = snapshots / snapshot_name
    for candidate in (staging, final):
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        raise FileExistsError(f"snapshot path already exists: {candidate.name}")
    staging.mkdir()
    artifact_rows = []
    excluded_symlinks = []
    try:
        if source_bytes is not None:
            source_target = staging / "source.py"
            source_target.write_bytes(source_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest()
        else:
            source_hash = None
        for source, relative in _snapshot_files(attempt, excluded_symlinks):
            target = staging / "artifacts" / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = source.read_bytes()
            except OSError as error:
                raise OSError(f"could not snapshot observed artifact {relative}: {error}") from error
            digest = hashlib.sha256(data).hexdigest()
            target.write_bytes(data)
            artifact_rows.append({"path": f"artifacts/{relative}", "observed_path": relative,
                                  "sha256": digest, "bytes": len(data), "status": "observed",
                                  "entry_hash_known": inventory_error is None,
                                  "changed_since_entry": (initial_hashes.get(relative) != digest
                                                          if inventory_error is None else None)})
        # No artifact is certified as produced: a failed call may observe stale files from an earlier evaluation.
        rec["failure_preservation"] = {"status": "preserved", "snapshot": f"{FAILURE_SNAPSHOTS}/{snapshot_name}"}
        snapshot_rec = portable(rec, attempt, root)
        eval_bytes = json.dumps(snapshot_rec, indent=1).encode("utf-8")
        (staging / EVAL_FILE).write_bytes(eval_bytes)
        manifest = {"schema": 1, "evaluation": rec.get("name"), "stage": rec.get("stage"),
                    "evaluation_record": {"path": EVAL_FILE, "sha256": hashlib.sha256(eval_bytes).hexdigest(),
                                           "bytes": len(eval_bytes)},
                    "source": {"path": "source.py" if source_bytes is not None else None,
                               "sha256": source_hash, "bytes": len(source_bytes) if source_bytes is not None else None,
                               "error": source_error},
                    "outputs": {"semantics": "observed state at failure; changed_since_entry is not certification",
                                "certified_produced_by_failing_call": [], "files": artifact_rows,
                                "excluded_symlinks": sorted(excluded_symlinks),
                                "entry_inventory_error": (portable({"error": inventory_error}, attempt, root)["error"]
                                                          if inventory_error else None)}}
        manifest_bytes = json.dumps(manifest, indent=1).encode("utf-8")
        (staging / "manifest.json").write_bytes(manifest_bytes)
        os.replace(staging, final)
        return {"status": "preserved", "snapshot": f"{FAILURE_SNAPSHOTS}/{snapshot_name}",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    except BaseException:
        # A staging directory is private and never looks like a completed snapshot.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _stored(path: str, attempt: Path, root: Path) -> str:
    """A path as eval.json keeps it, with forward slashes: relative to the attempt directory when it is inside the
    attempt, or inside the project the attempt is in (another attempt is `../<name>`); its name alone otherwise, so a
    record never spells a way out through the directories above the project."""
    p = Path(path).resolve()
    if p.is_relative_to(attempt):
        return p.relative_to(attempt).as_posix()
    if p.is_relative_to(root) and attempt.is_relative_to(root):
        return Path(os.path.relpath(p, attempt)).as_posix()
    return p.name


def portable(rec: dict, attempt: str | Path, root: str | Path) -> dict:
    """`rec` as eval.json holds it: every path relative to the attempt directory (`_stored`), and the attempt's and
    the project's own directories taken off the paths an error text names."""
    attempt, root = Path(attempt).resolve(), Path(root).resolve()
    out = dict(rec)
    for key in PATH_FIELDS:
        if out.get(key):
            out[key] = _stored(out[key], attempt, root)
    for key in PATH_MAPS:
        if out.get(key):
            out[key] = {view: _stored(p, attempt, root) for view, p in out[key].items()}
    for key in TEXT_FIELDS:
        text = out.get(key)
        for base in (attempt, root) if text else ():
            for spelt in (str(base), base.as_posix()):
                text = text.replace(spelt + "\\", "").replace(spelt + "/", "").replace(spelt, ".")
        if text:
            out[key] = text
    preservation = out.get("failure_preservation")
    if isinstance(preservation, dict) and preservation.get("error"):
        preservation = dict(preservation)
        out["failure_preservation"] = preservation
        text = preservation["error"]
        for base in (attempt, root):
            base = Path(base).resolve()
            for spelt in (str(base), base.as_posix()):
                text = text.replace(spelt + "\\", "").replace(spelt + "/", "").replace(spelt, ".")
        preservation["error"] = text
    return out


def read(attempt: str | Path) -> dict:
    """An attempt's eval.json with its paths absolute again, as `evaluate` returned them. A record written before
    paths were stored relative holds absolute ones, which are left as they are."""
    attempt = Path(attempt)
    rec = json.loads((attempt / EVAL_FILE).read_text(encoding="utf-8"))

    def full(p: str) -> str:
        return p if Path(p).is_absolute() else os.path.normpath(attempt / p)

    for key in PATH_FIELDS:
        if rec.get(key):
            rec[key] = full(rec[key])
    for key in PATH_MAPS:
        if rec.get(key):
            rec[key] = {view: full(p) for view, p in rec[key].items()}
    return rec


def numeric_score(n: dict) -> float | None:
    """Mean of silhouette overlap and resemblance: both 0..1, both against the reference, both drift-free. The
    outline error is reported beside it, not folded in: it is in millimetres, on the model's own scale.

    A record that carries the project's own `mix` (`criteria.score`: term -> weight) is the weighted mean of those
    terms instead. A term the attempt has no number for is left out, as a zone term is with no palette."""
    if n.get("mix"):
        live = [(n[criteria.SCORE_TERMS[t]], w) for t, w in n["mix"].items() if n.get(criteria.SCORE_TERMS[t]) is not None]
        return round(sum(v * w for v, w in live) / sum(w for _, w in live), 4) if live else None
    terms = [n[k] for k in ("ortho_iou", "resemblance") if n.get(k) is not None]
    return round(statistics.fmean(terms), 4) if terms else None


def numeric_measures(sil: dict, res: dict, profile_diff: dict, weights: dict[str, float] | None = None, *,
                     mix: dict[str, float] | None = None, regions: float | None = None) -> dict:
    """The deterministic measures of one attempt in one record: outline error (mm, from the GLB), silhouette overlap
    and resemblance (ortho means over the rendered views), the colour-zone agreement, how close the colours are, the
    edges in place, the marked regions read on their own (`regions`, from `review.regions_score`, when the project
    marks any), and `numeric_score`, the one number to rank attempts on. `mix` is the project's `criteria.score`."""
    out: dict = {"mae_mm": (profile_diff or {}).get("mae_mm"), "ortho_iou": sil.get("ortho_iou"), "resemblance": None,
                 "zone": None}
    if weights is None and out["ortho_iou"] is None and sil.get("views"):
        # a project measured on one perspective view has no ortho reference: the views it has are the measure
        out["ortho_iou"] = round(statistics.fmean(v["iou"] for v in sil["views"].values()), 4)
    views = res.get("views") or {}
    if views:
        out["resemblance"] = res.get("ortho") if res.get("ortho") is not None else res.get("score")
        ortho = [v for v in views if v in ORTHO_VIEWS] or list(views)
        zones = [d["zone"] for v, d in views.items() if v in ortho and d.get("zone") is not None]
        out["zone"] = round(statistics.fmean(zones), 4) if zones else None
        for key, term in (("colour", "colour"), ("edges", "structure")):
            read = [d[term] for v, d in views.items() if v in ortho and d.get(term) is not None]
            out[key] = round(statistics.fmean(read), 4) if read else None
    if weights is not None:
        from .scoring import aggregate
        out["ortho_iou"] = aggregate(sil.get("views") or {}, "iou", weights)
        out["resemblance"] = aggregate(views, "score", weights)
        out["zone"] = aggregate(views, "zone", weights)
        out["colour"] = aggregate(views, "colour", weights)
        out["edges"] = aggregate(views, "structure", weights)
    out["regions"] = regions
    for key in ("colour", "edges", "regions"):          # newer than the four above: a record has them when measured
        if out.get(key) is None:
            out.pop(key, None)
    if mix:
        out["mix"] = dict(mix)
    out["numeric_score"] = numeric_score(out)
    return out


def gates(spec: dict, numeric: dict) -> list[dict]:
    """The contract's optional limits on the measures, each as {"name", "passed", "message"}. A measure the contract
    sets no limit on is recorded and passes: a number is worth having long before anyone gates on it."""
    out = []
    for key, name, limit_key, higher in (("mae_mm", "outline error (mm)", "profile_mae_max", False),
                                         ("zone", "colour zones in place", "placement_min", True),
                                         ("resemblance", "resemblance", "resemblance_min", True)):
        value, limit = numeric.get(key), spec.get(limit_key)
        if value is None or limit is None:
            continue
        passed = value >= limit if higher else value <= limit
        out.append({"name": limit_key, "passed": bool(passed),
                    "message": f"{name} {value:.3f} ({'min' if higher else 'max'} {limit})"})
    return out


def numbers_text(n: dict) -> str:
    parts = []
    if n.get("mae_mm") is not None:
        parts.append(f"outline error {n['mae_mm']:.2f} mm")
    if n.get("ortho_iou") is not None:
        parts.append(f"overlap {n['ortho_iou']:.3f}")
    if n.get("resemblance") is not None:
        parts.append(f"resemblance {n['resemblance']:.3f}")
    if n.get("zone") is not None:
        parts.append(f"zones {n['zone']:.2f}")
    mix = n.get("mix") or {}
    for term in ("colour", "edges", "regions"):         # what a project's own score counts beyond the three above
        if term in mix and n.get(term) is not None:
            parts.append(f"{term} {n[term]:.3f}")
    if n.get("numeric_score") is not None:
        # four decimals: attempts late in a run differ in the fourth, and at three two of them read as a tie
        parts.append(f"numeric score {n['numeric_score']:.4f}" + (f" ({criteria.score_text(mix)})" if mix else ""))
    return ", ".join(parts)


def acceptance_gates(rec: dict) -> list[dict]:
    """Exclude unthresholded regions, also when reading immutable older evaluations."""
    observations = {c.get("name") for c in (rec.get("review") or {}).get("checks") or []
                    if review.is_observation(c)}
    return [g for g in rec.get("gates") or []
            if g.get("name") not in observations and not review.is_observation(g)]


def failed_gates(rec: dict) -> list[str]:
    """The names of the gates an attempt fails, in the order they were checked."""
    return [str(g.get("name") or "?") for g in acceptance_gates(rec) if not g.get("passed")]


def gate_tally(rec: dict) -> str:
    """How many of an attempt's gates pass and which fail, for the ledger: `17/19 (left-patch, right-eye)`, `19/19`,
    `-` for an attempt with none."""
    total, failed = len(acceptance_gates(rec)), failed_gates(rec)
    if not total:
        return "-"
    return f"{total - len(failed)}/{total}" + (f" ({', '.join(failed)})" if failed else "")


def triangles_text(rec: dict) -> str:
    """The triangles a measured build used, against the ceiling when there is one; '-' for a record without them."""
    reading = rec.get("readiness") or {}
    used, cap = (reading.get("cost") or {}).get("triangles"), reading.get("max_triangles")
    return "-" if used is None else f"{used}/{cap}" if cap else str(used)


def ledger_line(rec: dict) -> str:
    """One line per attempt, the table an agent reads back: name, verdict, the numbers, the gates, the judge if
    asked. The score is not a mean over the gates, so the line says them beside it."""
    n = rec.get("numeric") or {}
    f = lambda k, d=3: "-" if n.get(k) is None else f"{n[k]:.{d}f}"  # noqa: E731
    pw = rec.get("pairwise")
    verdict = "-" if not pw else f"{pw['win_rate']:.2f}{' decisive' if pw['decisive'] else ''} vs {rec.get('against_name', '?')}"
    status = "ok" if rec.get("ok") else f"FAILED {rec.get('stage', '')}"
    return (f"| {rec.get('name', '')} | {status} | {f('mae_mm', 2)} | {f('ortho_iou')} | {f('resemblance')} | {f('zone', 2)} | "
            f"{f('numeric_score', 4)} | {gate_tally(rec)} | {triangles_text(rec)} | {verdict} | {(rec.get('note') or '').replace('|', '/')} |")


def append_ledger(path: Path, rec: dict) -> None:
    """Append the attempt's line. A ledger begun under other columns gets the header again first, so no row sits
    under a header it does not fit. Cooperating writers serialize only this read/header/append section."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_writer_ownership(path):
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        heads = [line for line in text.splitlines() if line.startswith("| attempt |")]
        with open(path, "a", encoding="utf-8") as f:
            if not text.strip():
                f.write("# Attempts (mesh-jig eval)\n\n" + LEDGER_HEAD)
            elif not heads or heads[-1] != LEDGER_HEAD.splitlines()[0]:
                f.write(("" if text.endswith("\n") else "\n") + "\n" + LEDGER_HEAD)
            f.write(ledger_line(rec) + "\n")


@contextmanager
def _ledger_writer_ownership(path: Path):
    """Serialize cooperating writers without locking ledger bytes against unlocked readers.

    Windows uses a named mutex derived from the normalized absolute ledger path, avoiding a sidecar filename and
    its path-length limit. Other platforms use advisory flock on the ledger itself; ordinary readers do not take
    that lock. Both ownership mechanisms are released by the OS after process exit.
    """
    if os.name == "nt":
        import ctypes

        key = os.path.normcase(os.path.realpath(path))
        name = "Local\\mesh-jig-ledger-" + hashlib.sha256(key.encode("utf-8")).hexdigest()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        create_mutex.restype = ctypes.c_void_p
        wait = kernel32.WaitForSingleObject
        wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        wait.restype = ctypes.c_uint32
        release = kernel32.ReleaseMutex
        release.argtypes = [ctypes.c_void_p]
        release.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        handle = create_mutex(None, 0, name)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
        owned = False
        try:
            timeout_ms = max(0, int(_LEDGER_LOCK_TIMEOUT_SECONDS * 1000))
            result = wait(handle, timeout_ms)
            if result == 0x102:  # WAIT_TIMEOUT
                raise TimeoutError("ledger busy; wait and retry append")
            if result not in (0, 0x80):  # WAIT_OBJECT_0, WAIT_ABANDONED
                raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
            owned = True
            try:
                yield
            finally:
                if owned and not release(handle):
                    raise OSError(ctypes.get_last_error(), "ReleaseMutex failed")
        finally:
            close(handle)
    else:
        import fcntl

        with open(path, "a", encoding="utf-8") as lock_file:
            _acquire_posix_ledger_lock(lock_file, fcntl)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _acquire_posix_ledger_lock(lock_file, fcntl) -> None:
    """Acquire flock within a finite deadline; only contention errors are retried."""
    deadline = time.monotonic() + _LEDGER_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ledger busy; wait and retry append") from None
        time.sleep(min(_LEDGER_LOCK_RETRY_SECONDS, remaining))


def renders_of(attempt: str | Path) -> dict[str, str]:
    """view -> render for an attempt directory (its renders/), or for a directory of PNGs named by view."""
    d = Path(attempt)
    d = d / "renders" if (d / "renders").is_dir() else d
    return {p.stem: str(p) for p in sorted(d.glob("*.png")) if not p.stem.startswith(silhouette.OVERLAY_PREFIX)}


def pairwise(project: Project, a: dict[str, str], b: dict[str, str], out_dir: Path, *, backend, samples: int = 16,
             pairwise_fn: Callable | None = None) -> dict:
    """The judge on two sets of renders, restricted to the views the reference has: the record eval.json keeps."""
    refs = project.references
    a = {k: v for k, v in a.items() if k in refs}
    b = {k: v for k, v in b.items() if k in refs}
    res = (pairwise_fn or judge_mod.compare)(backend, refs, a, b, project.brief, Path(out_dir), samples=samples)
    rec = {k: res.get(k) for k in ("win_rate", "ci", "decisive", "ties", "judgements", "calls", "position_bias",
                                   "views", "reasons", "errors", "model")}
    rec["cost"] = (res.get("usage") or {}).get("cost")
    rec["better"] = judge_mod.better(res)
    return rec


@contextmanager
def _evaluation_ownership(out_dir: str | Path):
    """Exclusively own one attempt output directory for the duration of an evaluation.

    A fixed, short sidecar inside the output directory keeps ownership stable across processes and temporary
    directory settings. Creating the output directory and this sidecar are the only filesystem changes before the
    lock is acquired. The OS releases the advisory lock when a process exits.
    """
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".mjlock"
    lock_file = open(lock_path, "a+b")
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import ctypes
            import msvcrt

            class Overlapped(ctypes.Structure):
                _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                            ("Offset", ctypes.c_uint32), ("OffsetHigh", ctypes.c_uint32),
                            ("hEvent", ctypes.c_void_p)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            lock_file_ex = kernel32.LockFileEx
            lock_file_ex.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.c_uint32, ctypes.POINTER(Overlapped)]
            lock_file_ex.restype = ctypes.c_int
            unlock_file_ex = kernel32.UnlockFileEx
            unlock_file_ex.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.POINTER(Overlapped)]
            unlock_file_ex.restype = ctypes.c_int
            handle, overlapped = msvcrt.get_osfhandle(lock_file.fileno()), Overlapped()
            if not lock_file_ex(handle, 0x00000003, 0, 1, 0, ctypes.byref(overlapped)):
                error = ctypes.get_last_error()
                if error == 33:  # ERROR_LOCK_VIOLATION
                    raise ValueError(f"an evaluation is already running for {output}; wait for that command to finish")
                raise OSError(error, "LockFileEx failed")

            def unlock():
                if not unlock_file_ex(handle, 0, 1, 0, ctypes.byref(overlapped)):
                    raise OSError(ctypes.get_last_error(), "UnlockFileEx failed")
        else:
            import fcntl
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise ValueError(f"an evaluation is already running for {output}; wait for that command to finish") \
                    from error
            unlock = lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()
    finally:
        lock_file.close()


@contextmanager
def _owned_evaluation(project: Project, script: str | Path, out_dir: str | Path, *,
                      atlas: str | Path | None = None, paint_plan: str | Path | None = None,
                      name: str = "", note: str = "", against: str | Path | None = None,
                      against_name: str = "", backend=None, samples: int = 16,
                      build_fn: Callable = build.build_variants, render_fn: Callable = render.render_views,
                      pairwise_fn: Callable | None = None, log=print,
                      postprocess: Callable[[dict], None] | None = None):
    """Own one evaluation and its caller's immediate output-dependent post-processing as one operation."""
    with _evaluation_ownership(out_dir):
        rec = _evaluate_locked(project, script, out_dir, atlas=atlas, paint_plan=paint_plan, name=name, note=note,
                               against=against, against_name=against_name, backend=backend, samples=samples,
                               build_fn=build_fn, render_fn=render_fn, pairwise_fn=pairwise_fn, log=log)
        if postprocess is not None:
            postprocess(rec)
        yield rec


def evaluate(project: Project, script: str | Path, out_dir: str | Path, *, atlas: str | Path | None = None,
             paint_plan: str | Path | None = None, name: str = "", note: str = "", against: str | Path | None = None,
             against_name: str = "", backend=None, samples: int = 16, build_fn: Callable = build.build_variants,
             render_fn: Callable = render.render_views, pairwise_fn: Callable | None = None, log=print,
             _postprocess: Callable[[dict], None] | None = None) -> dict:
    """Build, render and measure while owning this output directory against another live evaluation."""
    with _owned_evaluation(project, script, out_dir, atlas=atlas, paint_plan=paint_plan, name=name, note=note,
                           against=against, against_name=against_name, backend=backend, samples=samples,
                           build_fn=build_fn, render_fn=render_fn, pairwise_fn=pairwise_fn, log=log,
                           postprocess=_postprocess) as rec:
        return rec


def _evaluate_locked(project: Project, script: str | Path, out_dir: str | Path, *, atlas: str | Path | None = None,
                     paint_plan: str | Path | None = None, name: str = "", note: str = "",
                     against: str | Path | None = None, against_name: str = "", backend=None, samples: int = 16,
                     build_fn: Callable = build.build_variants, render_fn: Callable = render.render_views,
                     pairwise_fn: Callable | None = None, log=print) -> dict:
    """Build, render, measure; judge pairwise against another attempt's renders when given a backend. The record,
    also written to out_dir/eval.json with its paths relative to out_dir (`portable`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = project.spec
    rec: dict = {"name": name or out_dir.name, "script": str(script), "note": note, "ok": False, "stage": "",
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "harness": harness.version()}
    try:
        source_bytes = Path(script).read_bytes()
        source_error = None
    except OSError as error:
        source_bytes, source_error = None, f"source unreadable at evaluation entry ({type(error).__name__})"
    initial_hashes, inventory_error = _snapshot_hashes(out_dir)
    try:
        rec[comparison_context.FIELD] = comparison_context.signature(project)
    except (OSError, ValueError, TypeError):
        # If source inputs cannot be read or canonicalized, leave explicit invalid
        # provenance so this evaluation cannot silently join a legacy bucket.
        rec[comparison_context.FIELD] = {"schema": 0}
    t0 = time.time()

    def finish() -> dict:
        rec["wall_s"] = round(time.time() - t0, 1)
        if not rec.get("ok"):
            try:
                rec["failure_preservation"] = _preserve_failed_evaluation(
                    out_dir, project.root, rec, source_bytes, source_error, initial_hashes, inventory_error)
            except Exception as error:
                # Keep and write the original failed evaluation. Preservation failure is reported on it.
                detail = portable({"error": f"{type(error).__name__}: {error}"}, out_dir, project.root)["error"]
                rec["failure_preservation"] = {"status": "failed", "error": detail[:1000]}
        (out_dir / EVAL_FILE).write_text(json.dumps(portable(rec, out_dir, project.root), indent=1), encoding="utf-8")
        return rec

    def fail(stage: str, error: str) -> dict:
        rec.update({"stage": stage, "error": error[-3000:]})
        log(f"[{rec['name']}] {stage} FAILED: {error[-600:]}")
        return finish()

    # Existing direct callers can build/render without references. Validate the new
    # opt-in contracts here while retaining that legacy behavior (CLI validates all).
    if (any(k in spec for k in ("height_m", "height_tolerance", "axis_extents_m", "axis_tolerances", "profile"))
            or any(project.data.get(k) is not None for k in ("scoring", "review", "criteria"))):
        problems = project.problems()
        if problems:
            return fail("project", "; ".join(problems))

    built = out_dir / "out"
    if paint_plan is not None:
        atlas = built / glb.atlas_name(spec)
        err = paint.paint_file(paint_plan, atlas, project.palette)
        if err:
            return fail("paint", err)
    if atlas is not None and not spec.get("textured"):
        return fail("build", "an atlas was given but the model block is not `textured`: the build would ignore it")
    painted = glb.texture_from(spec) == "script"
    if spec.get("textured") and atlas is None and not painted:
        return fail("build", "the model block is `textured`: give an atlas PNG or a paint plan")
    err = build_fn(script, built, spec, **({"atlas": atlas} if atlas is not None else {}),
                   **({"palette": project.palette} if painted else {}))
    if err:
        return fail("build", err)
    shot = render_fn(spec, built, out_dir / "renders")
    if shot.get("error"):
        return fail("render", shot["error"])

    refs = project.references
    views = {k: v for k, v in shot["renders"].items() if k in refs}
    compare_dir = out_dir / "compare"
    weights = project.scoring_weights
    missing = sorted(set(weights or {}) - set(views))
    if missing:
        return fail("measure", f"selected scoring views were not rendered: {missing}")
    paints = {k: v for k, v in (shot.get("albedo") or {}).items() if k in views}
    sil = silhouette.measure_views(views, refs, compare_dir, weights=weights)
    res = resemblance.measure_views(views, refs, project.zones, out_dir, weights=weights, colours=paints)
    judged = built / glb.judged_glb(spec)
    profile_diff = profile.glb_diff(project.reference_profile(), judged) if judged.is_file() else {}
    marked = review.regions(views, refs, (project.review or {}).get("regions") or [], project.zones, paints)
    numeric = numeric_measures(sil, res, profile_diff, weights, mix=project.score_mix,
                               regions=review.regions_score(marked))
    (out_dir / NUMERIC_FILE).write_text(json.dumps(numeric, indent=1), encoding="utf-8")
    rec.update({"ok": True, "stage": "measured", "renders": shot["renders"], "cropped": shot.get("cropped") or [],
                "compare": render.write_comparisons(views, refs, compare_dir),
                "ruled": references.write_ruled_pairs(project, views, compare_dir),
                "overlays": {v: str(compare_dir / f"{silhouette.OVERLAY_PREFIX}{v}.png") for v in sil["views"]},
                "numeric": numeric, "silhouette": sil["views"], "resemblance": res["views"],
                "profile_diff": profile_diff, "gates": gates(spec, numeric)})
    if paints:
        rec["albedo"] = paints
    if project.zones:
        rec["zone_reading"] = {v: zone_reading(views[v], refs[v], project.zones, paints.get(v)) for v in sorted(views)}
        rec["zone_maps"] = {v: zone_picture(views[v], refs[v], project.zones,
                                            compare_dir / f"{ZONES_PREFIX}{v}.png", paints.get(v)) for v in sorted(views)}
        # and each marked region with a zone limit, cut to its box: the face gates were read off the whole view
        rec["zone_maps"].update({
            r["name"]: zone_picture(views[r["view"]], refs[r["view"]], project.zones,
                                    compare_dir / f"{ZONES_PREFIX}region_{r['name']}.png", paints.get(r["view"]), r["box"])
            for r in (project.review or {}).get("regions") or [] if "min_zone" in r and r["view"] in views})
    if weights is not None:
        rec["scoring"] = {"weights": weights, "weak_overlap_views": sil["weak_views"],
                          "weak_resemblance_views": res["weak_views"]}
    if profile_diff.get("mode") == "standing":
        rec["gates"].append({"name": "standing_stations", "passed": not profile_diff["missing_stations"]
                              and not profile_diff["extra_stations"],
                              "message": f"missing {profile_diff['missing_stations']}; extra {profile_diff['extra_stations']}"})
    if project.review is not None:
        rec["review"] = review.measure(judged, spec, project.review, views, refs, project.zones, marked)
        rec["closeups"] = review.closeups(views, refs, project.review.get("regions") or [], compare_dir)
        rec["gates"].extend({"name": c["name"], "passed": c["passed"], "message": str(c)}
                            for c in rec["review"]["checks"] if not review.is_observation(c))
    if judged.is_file():
        rec["readiness"] = readiness.measure(judged, spec)
        # nodes no scored camera sees much of: what the pictures show as missing, and why
        rec["hidden"] = review.hidden(judged, spec, [v for v in (weights or ORTHO_VIEWS) if v in views])
        rec["floating"] = review.floating(judged, spec)      # and nodes that touch no other: a tail off its body
        rec["flipped"] = review.flipped(judged, spec, [v for v in (weights or ORTHO_VIEWS) if v in views])  # holes
    log(f"[{rec['name']}] measured: {numbers_text(numeric)}")

    if against is not None and backend is not None:
        other = {k: v for k, v in renders_of(against).items() if k in refs}
        if other:
            rec["pairwise"] = pairwise(project, views, other, out_dir / "pairwise", backend=backend, samples=samples,
                                       pairwise_fn=pairwise_fn)
            rec["against"] = str(against)
            rec["against_name"] = against_name or Path(against).name
            rec["better"] = rec["pairwise"]["better"]
            log(f"[{rec['name']}] pairwise vs {rec['against_name']}: win rate {rec['pairwise']['win_rate']:.2f}, "
                f"decisive {rec['pairwise']['decisive']} -> {'better' if rec['better'] else 'not better'}")
        else:
            rec["pairwise_error"] = f"no view PNGs under {against} match the reference views"
    return finish()


PREVIEW_DIR = "preview"


def preview(project: Project, attempt: str | Path, views: list[str] | None = None,
            render_fn: Callable = render.render_views) -> dict:
    """A free look at an attempt that is built: its GLBs drawn and measured as an evaluation would, into
    <attempt>/preview/, with no build, no ledger line and no eval.json. The renders are the evaluation's size, so
    the numbers are the ones an evaluation of this build gives. The record has the fields `feedback` reads
    (`numeric` only when every scored view was drawn; no outline table and no gates, which need the evaluation),
    or {"ok": False, "error": ...}."""
    attempt = Path(attempt).resolve()
    out = attempt / PREVIEW_DIR
    shot = render_fn(project.spec, attempt / "out", out, views=views or None)
    if shot.get("error"):
        return {"ok": False, "error": shot["error"]}
    refs = project.references
    paired = {k: v for k, v in shot["renders"].items() if k in refs}
    from . import brief                         # brief reads the project; nothing in it reads this module
    weights = project.scoring_weights
    whole = all(v in paired for v in brief.scored_views(project)[0])
    sil = silhouette.measure_views(paired, refs, out, weights=weights if whole else None)
    paints = {k: v for k, v in (shot.get("albedo") or {}).items() if k in paired}
    res = resemblance.measure_views(paired, refs, project.zones, None, weights=weights if whole else None,
                                    colours=paints)
    marked = [c for c in review.regions(paired, refs, (project.review or {}).get("regions") or [], project.zones,
                                        paints)
              if "error" not in c]              # a region whose view was not drawn is not read, not failed
    rec: dict = {"ok": True, "name": attempt.name, "renders": shot["renders"], "cropped": shot.get("cropped") or [],
                 "compare": render.write_comparisons(paired, refs, out),
                 "overlays": {v: str(out / f"{silhouette.OVERLAY_PREFIX}{v}.png") for v in sil["views"]},
                 "silhouette": sil["views"], "resemblance": res["views"], "gates": [], "regions": marked,
                 "numeric": numeric_measures(sil, res, {}, weights, mix=project.score_mix,
                                             regions=review.regions_score(marked)) if whole and paired else {}}
    if rec["numeric"].get("numeric_score") is not None:
        try:
            rec[comparison_context.FIELD] = comparison_context.signature(project)
        except (OSError, ValueError, TypeError):
            rec[comparison_context.FIELD] = {"schema": 0}
    if project.zones:
        rec["zone_reading"] = {v: zone_reading(paired[v], refs[v], project.zones, paints.get(v))
                               for v in sorted(paired)}
    if weights is not None and whole:
        rec["scoring"] = {"weights": weights, "weak_overlap_views": sil["weak_views"],
                          "weak_resemblance_views": res["weak_views"]}
    if project.review is not None:
        rec["closeups"] = review.closeups(paired, refs, project.review.get("regions") or [], out)
    return rec


def preview_text(project: Project, attempt: str | Path, rec: dict) -> str:
    """A preview as the builder reads it: that it is not an attempt, the numbers an evaluation of this build would
    give and where they stand against the best measured attempt, then each view, how the colours read and where
    the pictures are."""
    if not rec.get("ok"):
        return "PREVIEW FAILED: " + str(rec.get("error", ""))
    measured = rec["silhouette"]
    if not measured:
        return "rendered (no reference view of the same name to compare)"
    scored = ([v for v in rec["scoring"]["weights"] if v in measured] if rec.get("scoring")
              else [v for v in ORTHO_VIEWS if v in measured])
    numeric = rec.get("numeric") or {}
    if numeric.get("numeric_score") is not None:
        lines = [f"[{rec['name']}] a free look, not an attempt (nothing is added to the ledger): "
                 f"{numbers_text(numeric)}. An evaluation of this build gives these numbers, and the outline table "
                 "and the gates besides."]
        stands = standing(project, attempt, rec)
        lines += standing_lines(rec, stands, scored) if stands and (stands.get("best") or stands.get("excluded_incompatible")) else []
    else:
        lines = [f"[{rec['name']}] a free look, not an attempt: not every scored view was drawn, so there is no "
                 "score. Leave `views` out to have one."]
    lines += region_lines(rec)
    lines += view_lines(rec, scored)
    lines += zone_reading_lines(rec.get("zone_reading") or {}, scored)
    lines += picture_lines(rec, scored)
    return "\n".join(lines)


def zone_reading(render_path, reference, zones, colour=None) -> dict | None:
    """How a render's cells read as palette colours against the reference's, over the cells the zone score counts
    (inside both silhouettes): the share of each colour on each side, and for each reference colour what the render
    shows there. It is `resemblance`'s own classification (`normalised`, `zone_letters`) reported, not a new measure:
    the zone score alone says how many cells disagree, never which paint reads as which colour. `colour` is the
    render's unlit pass, read in its place under `colour_from: "albedo"`."""
    ra, ma = resemblance.normalised(render_path)
    if colour is not None:
        ra = resemblance.normalised_colour(render_path, colour)
    rb, mb = resemblance.normalised(reference)
    mine, ref = resemblance.zone_letters(ra, ma, zones), resemblance.zone_letters(rb, mb, zones)
    both = (mine >= 0) & (ref >= 0)
    cells = int(both.sum())
    if not cells:
        return None
    names = [z[1] for z in zones]

    def shares(letters, where) -> dict:
        n = int(where.sum())
        found = {names[k]: round(int((where & (letters == k)).sum()) / n, 3) for k in range(len(names))}
        return dict(sorted(((k, v) for k, v in found.items() if v), key=lambda kv: -kv[1]))

    return {"cells": cells, "reference": shares(ref, both), "yours": shares(mine, both),
            "read_as": {name: shares(mine, both & (ref == k)) for k, name in enumerate(names) if (both & (ref == k)).any()}}


ZONES_PREFIX = "zones_"
ZONE_CELL_PX = 6


def zone_picture(render_path, reference, zones, out_path, colour=None, box=None) -> str:
    """zones_<view>.png: the reference's cells in their palette colours, the build's, and where they disagree (red
    where both have a colour and differ, grey where only one side has the body), on the canvas the zone score reads.
    Builders asked where colour went wrong when the number only said how much. `box` (a region's, normalised on
    the whole-subject canvas) cuts it to that region and draws its cells larger: a face is a dozen cells of a view."""
    import numpy as np
    from PIL import Image, ImageDraw
    ra, ma = resemblance.normalised(render_path)
    if colour is not None:
        ra = resemblance.normalised_colour(render_path, colour)
    rb, mb = resemblance.normalised(reference)
    mine, ref = resemblance.zone_letters(ra, ma, zones), resemblance.zone_letters(rb, mb, zones)
    from .palette import zone_rgb
    palette = np.clip(np.asarray(zone_rgb(zones), float), 0, 255).astype(np.uint8)
    if box is not None:
        n = ref.shape[0]
        x0, y0, x1, y1 = (int(round(v * n)) for v in box)
        ref, mine = ref[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)], mine[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    h, w = ref.shape

    def paint(letters):
        img = np.full((h, w, 3), 255, np.uint8)
        img[letters >= 0] = palette[letters[letters >= 0]]
        return img
    diff = np.full((h, w, 3), 255, np.uint8)
    diff[(ref >= 0) & (mine >= 0)] = (225, 225, 225)
    diff[(ref >= 0) ^ (mine >= 0)] = (150, 150, 150)
    diff[(ref >= 0) & (mine >= 0) & (ref != mine)] = (220, 30, 30)
    k, label = (ZONE_CELL_PX if box is None else max(ZONE_CELL_PX, 240 // max(h, w))), 18
    sheet = Image.new("RGB", (3 * w * k + 2 * 8, h * k + label), "white")
    d = ImageDraw.Draw(sheet)
    for i, (img, title) in enumerate(((paint(ref), "REFERENCE zones"), (paint(mine), "YOUR BUILD zones"),
                                      (diff, "DISAGREE: red differ, grey one side only"))):
        x = i * (w * k + 8)
        sheet.paste(Image.fromarray(img).resize((w * k, h * k), Image.NEAREST), (x, label))
        d.text((x + 2, 2), title, fill=(0, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def zone_reading_lines(readings: dict, scored: list[str], least: float = 0.05) -> list[str]:
    """The zone readings as text, for the views the score is taken on: each side's colours by share, then, for the
    reference's three largest colours, what the render shows where the reference has them."""
    def listed(shares: dict, floor: float = least) -> str:
        return ", ".join(f"{name} {v:.0%}" for name, v in shares.items() if v >= floor) or "nothing over 5%"

    lines = []
    for view in [v for v in scored if readings.get(v)] or [v for v in readings if readings[v]]:
        r = readings[view]
        lines.append(f"  {view}: the reference is {listed(r['reference'])}; yours reads {listed(r['yours'])}")
        for name in list(r["reference"])[:3]:
            lines.append(f"    where the reference is {name}, yours reads {listed(r['read_as'].get(name) or {}, 0.10)}")
    if not lines:
        return []
    return ["HOW YOUR COLOURS READ (cells inside both outlines; a cell is the palette colour nearest its lit pixels AS "
            "RENDERED, so the light changes what a paint reads as, and a cell agrees only when both sides read the same):"
            ] + lines


def scored_views_lines(views: dict) -> list[str]:
    """Which of the compared views the overlap, the resemblance and so the numeric score are taken on, when it is not
    all of them: the orthographic three (`numeric_measures`). A run that is not told works on the views it looks at
    first, and `front` and `hero` are not among the ones that count."""
    scored = [v for v in ORTHO_VIEWS if v in views]
    others = [v for v in views if v not in scored]
    if not scored or not others:
        return []
    return [f"overlap, resemblance, zones and the numeric score are the means over {', '.join(scored)} ONLY; "
            f"{', '.join(others)} {'is' if len(others) == 1 else 'are'} compared for you to look at and "
            f"{'does' if len(others) == 1 else 'do'} not count (the fallback views: jig.json states no `criteria.views`)"]


def standing(project: Project, attempt: str | Path, rec: dict) -> dict | None:
    """Where a measured attempt stands among the project's other measured attempts, read from their eval.json:
    {"others": how many there are, "best": the highest scoring of them ({"name", "score", "failed": the gates it
    fails, "gates": how many it has, "views": view -> {"iou", "zone"}}, None when there is none), "clean": the highest
    scoring of them that passes every gate}. None for an attempt that was not measured.

    It is read from the files and not kept by the caller, so `eval`, `show`, the MCP tool and the agent loop say the
    same thing, and a builder that comes back to a project in a new session is still told of its earlier attempts."""
    if not rec.get("ok") or (rec.get("numeric") or {}).get("numeric_score") is None:
        return None
    here = Path(attempt).resolve()
    others, excluded = [], 0
    for f in project.attempts_dir.glob(f"*/{EVAL_FILE}"):
        if f.parent.resolve() == here:
            continue
        try:
            other = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(other, dict):
            continue
        score = (other.get("numeric") or {}).get("numeric_score")
        if other.get("ok") and score is not None:
            if not comparison_context.compatible(rec, other):
                excluded += 1
                continue
            others.append((-float(score), str(other.get("started") or ""), f.parent.name, other))
    others.sort(key=lambda o: o[:3])                    # highest first; of two that tie, the one measured first

    def entry(o) -> dict:
        r = o[3]
        sil, res = r.get("silhouette") or {}, r.get("resemblance") or {}
        return {"name": o[2], "score": -o[0], "failed": failed_gates(r), "gates": len(acceptance_gates(r)),
                "arms_clear": readiness.arms_clear(r.get("readiness") or {}), "numeric": r.get("numeric") or {},
                "views": {v: {"iou": (sil.get(v) or {}).get("iou"), "zone": (res.get(v) or {}).get("zone")}
                          for v in dict.fromkeys(list(sil) + list(res))}}
    passing = [o for o in others if acceptance_gates(o[3]) and not failed_gates(o[3])]
    result = {"others": len(others), "best": entry(others[0]) if others else None,
              "clean": entry(passing[0]) if passing else None}
    kind, _ = comparison_context.classify(rec)
    if kind == "legacy":
        result["provenance_note"] = "historical full evaluation provenance unavailable for legacy records."
    elif kind == "invalid":
        result["provenance_note"] = "comparisons unavailable: this record has invalid or unsupported provenance."
    if excluded:
        result["excluded_incompatible"] = excluded
        result["compatibility_note"] = _compatibility_note(rec, excluded)
    return result


def _compatibility_note(rec: dict, excluded: int) -> str:
    kind, _ = comparison_context.classify(rec)
    if kind == "legacy":
        return f"{excluded} measured attempt{'s' if excluded != 1 else ''} excluded: readiness target/cap provenance differs or is unknown."
    if kind == "invalid":
        return f"{excluded} measured attempt{'s' if excluded != 1 else ''} excluded: comparison provenance is invalid or uses an unsupported schema."
    return f"{excluded} measured attempt{'s' if excluded != 1 else ''} excluded: evaluation contexts differ; legacy provenance is unavailable."


def term_lines(n: dict, was: dict, name: str) -> str:
    """Each term of the score against the best attempt's: a score that barely moves is often two terms trading,
    and a builder was told only the total and the views."""
    mix = n.get("mix") or {"overlap": 1, "resemblance": 1}
    fields = {**criteria.SCORE_TERMS, "resemblance": "resemblance"}
    said = []
    for term in mix:
        key = fields.get(term)
        if key and n.get(key) is not None and was.get(key) is not None:
            said.append(f"{term} {n[key]:.3f} (was {was[key]:.3f}, {n[key] - was[key]:+.3f})")
    return f"  the score's terms against {name}: " + "; ".join(said) if said else ""


def standing_lines(rec: dict, st: dict | None, scored: list[str]) -> list[str]:
    """`standing` as the agent reads it: how far the attempt is from the best of the others, which gates the two
    differ on, the highest attempt that passes every gate when this one does not, and the scored views that moved.
    The score is not a mean over the gates or the views, so a change that gained a little by trading one of them
    away reads here as a trade."""
    if st is None:
        return []
    best = st.get("best")
    if not best:
        if comparison_context.classify(rec)[0] == "invalid":
            label = "standing: comparison unavailable for this record."
        else:
            label = ("standing: no compatible measured attempts in the history." if st.get("excluded_incompatible")
                     else "standing: the first measured attempt here.")
        lines = [label]
        if st.get("excluded_incompatible"):
            lines.append("  " + st["compatibility_note"])
        if st.get("provenance_note"):
            lines.append("  " + st["provenance_note"])
        return lines
    score = rec["numeric"]["numeric_score"]
    gap = round(score - best["score"], 4)
    total = st["others"] + 1
    if gap > 0:
        line = f"standing: the best of the {total} measured attempts, {gap:.4f} above {best['name']} ({best['score']:.4f})."
    elif gap < 0:
        line = f"standing: {-gap:.4f} below the best attempt, {best['name']} ({best['score']:.4f})."
    else:
        line = f"standing: level with {best['name']} ({best['score']:.4f}), the best of the other attempts."
    mine = failed_gates(rec)
    if rec.get("gates") and best["gates"]:
        broke, mended = [g for g in mine if g not in best["failed"]], [g for g in best["failed"] if g not in mine]
        if broke:
            line += f" It fails {', '.join(broke)}, which {best['name']} passes."
        if mended:
            line += f" It passes {', '.join(mended)}, which {best['name']} fails."
    arms, was = readiness.arms_clear(rec.get("readiness") or {}), best.get("arms_clear")
    if arms == 0 and was:
        line += f" No arm hangs clear of the body, where {best['name']}'s hang clear for {was:.2f} m."
    clean = st.get("clean")
    if mine and clean and clean["name"] != best["name"]:
        line += f" The highest attempt that passes every gate is {clean['name']} ({clean['score']:.4f})."
    lines = [line]
    if st.get("excluded_incompatible"):
        lines.append("  " + st["compatibility_note"])
    if st.get("provenance_note"):
        lines.append("  " + st["provenance_note"])
    terms = term_lines(rec.get("numeric") or {}, best.get("numeric") or {}, best["name"])
    if terms:
        lines.append(terms)
    sil, res = rec.get("silhouette") or {}, rec.get("resemblance") or {}
    moved, held = [], False
    for view in scored:
        then = best["views"].get(view) or {}
        for what, now, was in (("overlap", (sil.get(view) or {}).get("iou"), then.get("iou")),
                               ("zones", (res.get(view) or {}).get("zone"), then.get("zone"))):
            if now is None or was is None:
                continue
            held = True
            if abs(now - was) >= VIEW_MOVE:
                moved.append(f"{view} {what} {now:.3f} (was {was:.3f})")
    if moved:
        lines.append(f"  by scored view against {best['name']}: " + "; ".join(moved))
    elif held:
        lines.append(f"  against {best['name']} no scored view's overlap or zones moved by {VIEW_MOVE:g}")
    return lines


def gate_lines(rec: dict) -> list[str]:
    """The gates as text: a line for each one that failed, saying what it read and what it needed, then every one
    that passed on one line with the numbers it passed on. A review check is put into words from its own record
    (`review.describe`), which holds its limits; a limit on a measure carries its words already."""
    checks = {c.get("name"): c for c in (rec.get("review") or {}).get("checks") or []}
    diff = rec.get("profile_diff") or {}

    def said(g: dict) -> str:
        if g["name"] in checks:
            return review.describe(checks[g["name"]]) or ("" if g["passed"] else str(g.get("message") or ""))
        if g["name"] == "standing_stations":
            gone, extra = diff.get("missing_stations") or [], diff.get("extra_stations") or []
            return "; ".join(x for x in (
                f"the reference has body at {', '.join(map(str, gone))} and yours has none" if gone else "",
                f"yours has body at {', '.join(map(str, extra))} and the reference has none" if extra else "") if x)
        message = str(g.get("message") or "")
        return message if not g["passed"] else message.replace(" (", ", ").rstrip(")")

    gates = acceptance_gates(rec)
    lines = [f"gate {g['name']}: FAILED - {said(g)}" for g in gates if not g["passed"]]
    passed = [g["name"] + (f" ({said(g)})" if said(g) else "") for g in gates if g["passed"]]
    observed = [f"observation {c['name']}: {c.get('error') or review.describe(c)} (no acceptance limit)"
                for c in checks.values() if review.is_observation(c)]
    return lines + (["gates passed: " + "; ".join(passed)] if passed else []) + observed


def region_lines(rec: dict) -> list[str]:
    """The marked regions as their own numbers, weakest first: an evaluation's are among its review checks, a
    preview's in `regions`. Said whether or not the score counts them: a gate says only passed or failed."""
    checks = [c for c in rec.get("regions") or (rec.get("review") or {}).get("checks") or []
              if c.get("score") is not None]
    if not checks:
        return []
    n = rec.get("numeric") or {}
    counted = "regions" in (n.get("mix") or {})
    said = "; ".join(f"{c['name']} {c['score']:.2f} (colours {c['colour']:.2f} close"
                     + (f", edges in place {c['edges']:.2f}" if c.get("edges") is not None else "") + ")"
                     for c in sorted(checks, key=lambda c: c["score"]))
    mean = f", their mean {n['regions']:.3f}" if n.get("regions") is not None else ""
    lines = ["REGIONS (each marked region read on its own, twice as fine as a whole view; weakest first"
             + mean + (", which the score counts" if counted else "") + "): " + said]
    for c in checks:
        if c.get("passed") is False and c.get("read_as"):
            # a failed region gate: which of the reference's colours the build gets wrong there, largest first
            said = "; ".join(
                f"where the reference is {colour} ({r['share']:.0%} of it), yours reads "
                + (", ".join(f"{k} {v:.0%}" for k, v in r["yours"].items() if v >= 0.10) or "a mix under 10% each")
                for colour, r in list(c["read_as"].items())[:3] if r["share"] >= 0.05)
            if said:
                lines.append(f"  {c['name']} (its gate FAILED; {c['view']} view, the zone colours as rendered): {said}")
    return lines


def view_lines(rec: dict, scored: list[str]) -> list[str]:
    """One line a view: its outline overlap and proportion, then its colour zones and edges. The views the score is
    taken over come first, and a view that is only compared is marked: its line is there to look at, not to fix."""
    sil, res = rec.get("silhouette") or {}, rec.get("resemblance") or {}
    order = list(dict.fromkeys([v for v in scored if v in sil or v in res] + list(sil) + list(res)))
    mark = bool(scored) and any(v not in scored for v in order)
    lines = []
    for view in order:
        said = silhouette.feedback_lines({"views": {view: sil[view]}}) if view in sil else []
        said += resemblance.feedback_lines({"views": {view: res[view]}}) if view in res else []
        if said:
            lines.append(f"{view}{' (not scored)' if mark and view not in scored else ''}: "
                         + "; ".join(line.split(": ", 1)[1] for line in said))
    return lines


def parts_line(diagnostics: dict) -> str:
    """The named parts' size and how many pieces each is in, on one line."""
    return "parts (triangles, welded pieces): " + "; ".join(
        f"{name} {d.get('triangles')}, {d.get('position_welded_components')}" for name, d in diagnostics.items())


def picture_lines(rec: dict, scored: list[str]) -> list[str]:
    """Where the comparison pictures are: the directory once, then each file's name, scored views first. An
    evaluation names a dozen files under one directory, and the directory can be a hundred characters long."""
    def listed(paths: dict) -> tuple[str, str]:
        order = [v for v in scored if v in paths] + [v for v in paths if v not in scored]
        files = [Path(paths[v]) for v in order]
        folders = {str(p.parent) for p in files}
        if len(folders) != 1:
            return "", ", ".join(str(p) for p in files)
        unscored = [Path(paths[v]).name for v in order if v not in scored] if scored else []
        names = ", ".join(p.name for p in files if p.name not in unscored)
        return folders.pop(), names + (f"{'; ' if names else ''}not scored: {', '.join(unscored)}" if unscored else "")

    lines = []
    folder, names = listed(rec.get("compare") or {})
    if names:
        lines.append("LOOK AT (reference left, your build right)" + (f" in {folder}" if folder else "") + ": " + names)
    over, diffs = listed(rec.get("overlays") or {})
    if diffs:
        where = ", same directory" if over and over == folder else f" in {over}" if over else ""
        lines.append(f"outline diffs (grey both, red add body, blue remove body){where}: {diffs}")
    close = rec.get("closeups") or {}
    if close:
        failed = set(failed_gates(rec))
        order = [n for n in close if n in failed] + [n for n in close if n not in failed]    # what failed, first
        at = {str(Path(close[n]).parent) for n in order}
        where = ", same directory" if at == {folder} else f" in {at.pop()}" if len(at) == 1 else ""
        names = ", ".join((Path(close[n]).name if where else close[n]) + (" (its gate FAILED)" if n in failed else "")
                          for n in order)
        lines.append(f"close-ups of the marked regions (reference left, your build right, enlarged){where}: {names}")
    return lines


def feedback(rec: dict, stands: dict | None = None) -> str:
    """The record as the agent reads it, the verdict first: what failed, or the numbers with the gate count, each
    failed gate and what it needed, and where the attempt stands among the others (`stands`, from `standing`). Then
    each view, how the colours read, every row of the outline table, where the pictures are, and the judge's reasons
    when it was asked. A reader that keeps only the first lines still has the verdict."""
    name = rec.get("name", "")
    if not rec.get("ok"):
        lines = [f"[{name}] FAILED at {rec.get('stage', '?')}:", str(rec.get("error", ""))]
        preservation = rec.get("failure_preservation") or {}
        if preservation.get("status") == "preserved":
            lines.append("failure snapshot saved: " + str(preservation.get("snapshot", "")))
        elif preservation.get("status") == "failed":
            lines.append("failure snapshot could not be saved: " + str(preservation.get("error", "unknown I/O error")))
        return "\n".join(lines)
    measured = rec.get("silhouette") or {}
    # the project names its own scoring views, or the score is over the orthographic three
    scored = ([v for v in rec["scoring"]["weights"] if v in measured] if rec.get("scoring")
              else [v for v in ORTHO_VIEWS if v in measured])
    gates, failed = acceptance_gates(rec), failed_gates(rec)
    tally = ""
    if failed:
        tally = f"; {len(failed)} of {len(gates)} gate{'s' if len(gates) != 1 else ''} FAILED ({', '.join(failed)})"
    elif gates:
        tally = f"; all {len(gates)} gates pass" if len(gates) > 1 else "; its gate passes"
    lines = [f"[{name}] built, in contract, rendered and measured: {numbers_text(rec.get('numeric') or {})}{tally}"]
    mesh = readiness.summary(rec["readiness"], include_surface=False) if rec.get("readiness") else ""
    surface = readiness.surface_summary(rec["readiness"]) if rec.get("readiness") else ""
    if mesh:
        lines.append(mesh)
    cropped = rec.get("cropped") or []
    cut = (f"the subject touches the frame edge in {cropped}; it may be cropped, "
           "which can raise or lower the overlap")
    if any(v in scored for v in cropped):               # a view the score is taken over: the number above is off
        lines.append(cut)
    lines += gate_lines(rec)
    lines += standing_lines(rec, stands, scored)
    if surface:
        lines.append(surface)
    lines += region_lines(rec)
    if rec.get("scoring"):
        weak = lambda rows: ", ".join(f"{r['view']} {r['value']:.3f}" for r in rows[:2]) or "none"   # noqa: E731
        weights = ", ".join(f"{v} {float(w):g}" for v, w in rec["scoring"]["weights"].items())
        lines.append(f"Aggregate scoring weights: {weights}; weakest overlap views: "
                     f"{weak(rec['scoring']['weak_overlap_views'])}")
        lines.append(f"Weakest resemblance views: {weak(rec['scoring']['weak_resemblance_views'])}")
    else:
        lines += scored_views_lines(measured)
    lines += view_lines(rec, scored)
    lines += zone_reading_lines(rec.get("zone_reading") or {}, scored)
    maps = rec.get("zone_maps") or {}
    if maps:
        lines.append("WHERE THE COLOURS DISAGREE, cell by cell (reference zones, yours, and red where they differ), "
                     "in the comparison pictures' directory: "
                     + ", ".join(Path(maps[v]).name for v in ([v for v in scored if v in maps] or sorted(
                         v for v in maps if v in measured)) + sorted(v for v in maps if v not in measured)))
    for warning in (rec.get("review") or {}).get("warnings", []):
        lines.append("review warning: " + warning)
    if (rec.get("review") or {}).get("diagnostics"):
        lines.append(parts_line(rec["review"]["diagnostics"]))
    if rec.get("readiness"):
        lines += readiness.lines(rec["readiness"])
    if rec.get("hidden"):
        lines += ["HIDDEN (an object no scored camera sees much of): " + line
                  for line in review.hidden_lines(rec["hidden"])]
    if rec.get("floating"):
        lines += ["FLOATING (sampled separation; inspect contact): " + line for line in review.floating_lines(rec["floating"])]
    if rec.get("flipped"):
        lines += ["FLIPPED: " + line for line in review.flipped_lines(rec["flipped"])]
    if rec.get("profile_diff"):
        lines.append(profile.compare_text(rec["profile_diff"]))
    if cropped and cut not in lines:
        lines.append(cut)
    lines += picture_lines(rec, scored)
    pw = rec.get("pairwise")
    if pw:
        lines.append(judge_mod.summary({**pw, "views": pw.get("views") or {}}, name, rec.get("against_name", "the other")))
    if rec.get("pairwise_error"):
        lines.append("judge not run: " + rec["pairwise_error"])
    return "\n".join(lines)
