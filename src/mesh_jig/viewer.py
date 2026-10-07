"""mesh-jig view: a local web page that shows every result measured against a reference.

    mesh-jig view [<dir> ...] [--skip NAME] [--port 8796] [--open]
    mesh-jig view <dir> ... --export <dir> [--best]        the same page as files, for a static host

Each directory is searched for projects (a jig.json), leaving out any directory named by `--skip`. Projects whose reference views are the same images are one
REFERENCE, however many copies of the project there are: a comparison made by copying a seed project per model shows
up as one reference with a run per copy. For a reference the page shows the best attempt of every run view by view
under the reference views, the attempts as a table of their numbers (one row a run, its best attempt, with the run's
other attempts a click away, or every attempt ranked on the numeric score), the `mesh-jig agent` runs with their
transcripts, and beside whichever of those is open the picked attempt: every measure with what it is and which views
it is a mean over, the project's part checks, and per view the reference, the build and the outline diff with that
view's own numbers, its outline stations and how its surface reads in palette colours.
Under every score is the commit of the mesh-jig that measured it (`harness`), a link when the checkout has a remote
and the commit is in its history, plain text otherwise: results are comparable when it is the same, and a change to
the guidance alone is a different commit.
Every run says what drove it (`driver_of`): Claude Code, Codex or `mesh-jig agent` calling an API, with the model and
the effort, read from what the run recorded. The runs can be narrowed to one of those.
A reference's runs are listed in sections, one for each directory they were found in under the searched one (an
experiment's folder): a section shows its top scoring run alone until it is opened. Runs that are all of one
directory are listed without sections.
An attempt whose GLB is on disk is drawn in 3D, in the browser, with the lights and the cameras `render` measures
with: a named view is the picture that was measured, and it turns from there. "In 3D" is the best attempt of every
run side by side, turning together.
Under a drawn attempt is a button that shows its GLB in the machine's file manager (Explorer, with the file
selected), for taking the model into another program, and a link that downloads it.

Read-only, and the server is the standard library's; the page loads nothing from anywhere else. Nothing is cached on
disk: the page reads eval.json, the pictures `evaluate` wrote and agent/*/report.json as they are, so it can be left
open while a run is going (a file goes out with an ETag, so the browser keeps it until the file changes). The server
binds to 127.0.0.1 unless told otherwise and serves only pictures, text and built models from inside the projects it
found. The one thing it does besides is that button: a POST to /reveal/<project>/<path> opens the file manager at a
built model. A page cannot do that itself, so the server does, for a GLB it would serve and for a page open on its
own machine (`on_this_machine`), and the page offers the button only there.

`--export` writes the page with everything it would ask the server for (`export`): the index, the files it links
and its tiles already cut, under relative addresses, so the directory can be put on any static host. It names no
directory of the machine it was made on and leaves the agent transcripts out. `--best` keeps each run's best
attempt only.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from . import (agent as agent_mod, evaluate as evaluate_mod, glb as glb_mod, harness as harness_mod, project as project_mod,
               render, silhouette)

DEFAULT_PORT = 8796
SCAN_DEPTH = 5
SKIP_DIRS = {project_mod.ATTEMPTS, agent_mod.RUNS_DIR, "refs", "out", "renders", "compare", "preview", "build", "dist",
             "node_modules", "__pycache__"}
VIEW_ORDER = ("hero", "front", "side", "rear", "top")
PANEL_HEIGHT = 240
PANEL_MAX_HEIGHT = 720
TILE_SIZES = (96, 240, 420)             # the sizes the page asks for a tile in: what an export cuts ahead of time
STATIC_FLAG = "const STATIC = false;"   # the line of the page an export turns on
TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
         ".py": "text/plain; charset=utf-8", ".json": "application/json; charset=utf-8",
         ".jsonl": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8",
         ".glb": "model/gltf-binary"}
NUMERIC_KEYS = ("mae_mm", "ortho_iou", "resemblance", "zone", "numeric_score")
OUTLINE_KEYS = ("iou", "missing", "extra", "aspect", "reference_aspect")      # per view, from silhouette.compare
HARNESS_FILE = "harness-commit.txt"     # a run made by a script that noted the commit itself: its first word
COMMIT = re.compile(r"[0-9a-f]{7,40}")
REVEALED = (".glb",)    # what the file manager is opened at: a built model
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
BEST_DIR = "best"       # a run that keeps only its best attempt's build (the sweep does): best/best.json names the attempt
# what drives a run, as the page names it, and how its own name starts in a report's `harness` or in the command
# harness-commit.txt holds. "agent" is `mesh-jig agent`: the tool loop here, calling a model over an API.
DRIVERS = (("claude-code", "Claude Code", re.compile(r"claude(?:[ -]code)?(?![\w-])", re.I)),
           ("codex", "Codex", re.compile(r"codex(?![\w-])", re.I)),
           ("agent", "API", re.compile(r"mesh-jig(?:\.exe)? +agent(?![\w-])", re.I)))


# -- the index (pure: directories in, a JSON-able dict out) ----------------------------------------------------------

def search(paths: list[str | Path], depth: int = SCAN_DEPTH, skip: tuple[str, ...] = ()) -> dict[Path, tuple[str, str]]:
    '{project root: (label, group)} for every jig.json at or under `paths`.'
    found: dict[Path, tuple[str, str]] = {}
    for start in paths:
        start = Path(start).resolve()
        for cur, dirs, files in os.walk(start):
            here = Path(cur)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d not in skip and not d.startswith("."))
            if project_mod.JIG_FILE in files:
                rel = here.relative_to(start).parts
                found.setdefault(here, ("/".join(rel) or here.name,
                                        rel[0] if len(rel) > 1 else start.name if rel else here.parent.name))
                dirs[:] = []
            elif len(here.relative_to(start).parts) >= depth:
                dirs[:] = []
    return dict(sorted(found.items(), key=lambda kv: kv[0].as_posix()))


def find_projects(paths: list[str | Path], depth: int = SCAN_DEPTH, skip: tuple[str, ...] = ()) -> list[tuple[Path, str]]:
    """(project root, label) for every project `search` finds, in path order."""
    return [(root, label) for root, (label, _) in search(paths, depth, skip).items()]


def project_id(root: Path) -> str:
    return hashlib.sha1(root.resolve().as_posix().encode("utf-8")).hexdigest()[:10]


def reference_key(refs: dict[str, str]) -> str:
    """One key for a set of reference views: the same images under the same view names, wherever they are."""
    h = hashlib.sha1()
    for view in sorted(refs):
        h.update(view.encode("utf-8") + b"\0" + hashlib.sha1(Path(refs[view]).read_bytes()).digest())
    return h.hexdigest()[:12]


def ordered_views(names) -> list[str]:
    names = list(dict.fromkeys(names))
    return [v for v in VIEW_ORDER if v in names] + sorted(v for v in names if v not in VIEW_ORDER)


def _rel(root: Path, path: Path) -> str | None:
    """The path relative to the project when the file is there, else None: the page links only what exists."""
    return path.relative_to(root).as_posix() if path.is_file() else None


def harness_of(rec: dict, root: Path) -> dict | None:
    """{"commit", "dirty"} of the mesh-jig that measured an attempt: what its eval.json recorded, else the commit the
    project's harness-commit.txt names (dirty unknown: None). None when neither says, or says something that is not
    a commit: the page puts this into a link."""
    said = rec.get("harness")
    if isinstance(said, dict) and COMMIT.fullmatch(str(said.get("commit") or "")):
        return {"commit": said["commit"], "dirty": bool(said.get("dirty"))}
    try:
        first = (root / HARNESS_FILE).read_text(encoding="utf-8").split()[:1]
    except OSError:
        return None
    return {"commit": first[0], "dirty": None} if first and COMMIT.fullmatch(first[0]) else None


def command_of(root: Path) -> str:
    """The command a run script noted the run was started with: harness-commit.txt after its first line."""
    try:
        lines = (root / HARNESS_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return " ".join(" ".join(lines[1:]).split())


def driver_of(root: Path, runs: list[dict]) -> dict | None:
    """What drove a run: {"kind": "claude-code" | "codex" | "agent" | "other", "name", "detail", "model", "effort"},
    from what the run recorded and nothing else. The latest report says first: one brought in from a coding agent's
    log names the tool and its version (`harness`), and one with no such field is `mesh-jig agent`'s own, whose
    detail is the host it called. With no report, the command in harness-commit.txt says. None when neither does:
    a run's name is not read."""
    command = command_of(root)

    def flag(name: str) -> str | None:
        m = re.search(rf"--{name}[ =](\S+)", command)
        return m.group(1) if m else None

    def tool(text: str) -> tuple[str, str, str] | None:
        for kind, name, starts in DRIVERS:
            m = starts.match(text)
            if m:
                return kind, name, text[m.end():].strip()
        return None

    report = runs[-1] if runs else None
    if report is None:
        found, detail = tool(command), ""
    elif report.get("harness"):
        said = " ".join(str(report["harness"]).split())
        found = tool(said) or ("other", said, "")
        detail = found[2]
    else:
        found = next((kind, name, "") for kind, name, _ in DRIVERS if kind == "agent")
        detail = urllib.parse.urlsplit(str(report.get("url") or flag("url") or "")).netloc
    if found is None:
        return None
    configured = re.search(r'model_reasoning_effort\s*=\s*[\"\x27]?(\w+)', command)
    return {"kind": found[0], "name": found[1][:60], "detail": detail[:60],
            "model": (report or {}).get("model") or flag("model"),
            "effort": (report or {}).get("effort") or flag("effort") or (configured.group(1) if configured else None)}


def stage_of(spec: dict) -> dict:
    """What the page needs to draw a project's model as `render` draws it: the scale it is staged at, which of
    COLOR_0 and the texture colours it, and each shot of the judged variant as the camera `render.camera` makes of
    it. A shot that is not a camera is left out."""
    judged = glb_mod.judged_variant(spec)
    shots = []
    for shot in render.shots_for(spec):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):       # a camera standing on its target has no direction
                eye, basis, tan = render.camera(shot)
            target = [float(x) for x in shot["cam_target"]]
        except (KeyError, TypeError, ValueError):
            continue
        if shot["variant"] == judged and np.isfinite(basis).all():
            shots.append({"name": str(shot.get("name") or ""), "eye": eye.tolist(), "target": target,
                          "forward": basis[2].tolist(), "up": basis[1].tolist(), "tan": tan,
                          "ortho": float(shot.get("ortho") or 0.0)})
    return {"scale": render.stage_scale(spec), "colour": glb_mod.texture_mode(spec), "shots": shots}


def light() -> dict:
    """The lights `render.shade` uses, for the page to shade with: linear colours, each light's travel direction."""
    return {"background": [c / 255 for c in render.BACKGROUND], "ambient": render.AMBIENT.tolist(),
            "lights": [{"direction": (d / np.linalg.norm(d)).tolist(), "colour": (c * render.DIFFUSE).tolist()}
                       for d, c in render.LIGHTS]}


def kept_build(proj: project_mod.Project) -> tuple[str, str] | None:
    """(attempt, its GLB's path in the project) for a run that kept its best attempt's build in best/ when the
    attempts' own out/ did not travel with it."""
    try:
        name = json.loads((proj.root / BEST_DIR / "best.json").read_text(encoding="utf-8")).get("attempt")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    rel = _rel(proj.root, proj.root / BEST_DIR / glb_mod.judged_glb(proj.spec))
    return (str(name), rel) if name and rel else None


def attempt_record(proj: project_mod.Project, attempt: Path) -> dict | None:
    """One attempt as the page shows it, from its eval.json and the pictures beside it; None without an eval.json.
    Pictures are found by the attempt's layout, not by the paths eval.json stored: a copied project still works."""
    try:
        rec = json.loads((attempt / evaluate_mod.EVAL_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    root = proj.root
    numeric = rec.get("numeric") or {}
    sil, res = rec.get("silhouette") or {}, rec.get("resemblance") or {}
    rendered = {p.stem for p in (attempt / "renders").glob("*.png")} if (attempt / "renders").is_dir() else set()
    views = {}
    for v in ordered_views(list(sil) + sorted(rendered)):
        s, r = sil.get(v) or {}, res.get(v) or {}
        views[v] = {**{k: s.get(k) for k in OUTLINE_KEYS}, "zone": r.get("zone"), "structure": r.get("structure"),
                    "resemblance": r.get("score"),
                    "compare": _rel(root, attempt / "compare" / f"{v}{render.COMPARE_SUFFIX}.png"),
                    "overlay": _rel(root, attempt / "compare" / f"{silhouette.OVERLAY_PREFIX}{v}.png"),
                    "render": _rel(root, attempt / "renders" / f"{v}.png")}
    # the views the attempt's numbers are means over: the project's scoring views with their weights when it names
    # them, else top, side and rear, or every view when it has none of those (evaluate.numeric_measures). The others
    # are compared to be looked at.
    weights = (rec.get("scoring") or {}).get("weights") or None
    counted = ([v for v in views if v in weights] if weights else
               [v for v in views if v in sil and v in silhouette.ORTHO_VIEWS] or [v for v in views if v in sil])
    marks = [(views[v]["structure"], (weights or {}).get(v, 1.0)) for v in counted if views[v]["structure"] is not None]
    pw = rec.get("pairwise")
    return {"name": attempt.name, "ok": bool(rec.get("ok")), "stage": rec.get("stage") or "",
            "error": str(rec.get("error") or "")[-2000:], "note": rec.get("note") or "", "started": rec.get("started") or "",
            "numeric": {k: numeric.get(k) for k in NUMERIC_KEYS}, "views": views, "counted": counted, "weights": weights,
            "structure": round(sum(x * w for x, w in marks) / sum(w for _, w in marks), 4) if marks else None,
            "gates": [{"name": str(g.get("name") or ""), "passed": bool(g.get("passed")),
                       "message": str(g.get("message") or "")[:200]} for g in evaluate_mod.acceptance_gates(rec)],
            "wall_s": rec.get("wall_s"), "colours": bool(rec.get("zone_reading")), "harness": harness_of(rec, root),
            "script": _rel(root, attempt / str(proj.spec.get("script") or "model.py")),
            "glb": _rel(root, attempt / "out" / glb_mod.judged_glb(proj.spec)),
            "eval": _rel(root, attempt / evaluate_mod.EVAL_FILE),
            "pairwise": {"win_rate": pw.get("win_rate"), "decisive": pw.get("decisive"),
                         "against": rec.get("against_name") or ""} if pw else None}


def run_record(root: Path, run: Path) -> dict | None:
    """One `mesh-jig agent` run from its report.json."""
    try:
        rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    keep = ("model", "turns", "evaluations", "best", "cost_usd", "wall_s", "done", "error", "summary", "prompt_tokens",
            "completion_tokens", "tool_calls", "rejected_calls", "harness", "url", "effort")
    return {"name": run.name, **{k: rep.get(k) for k in keep}, "report": _rel(root, run / "report.json"),
            "transcript": _rel(root, run / "transcript.jsonl")}


def project_record(proj: project_mod.Project, label: str, group: str = "") -> dict:
    root = proj.root
    attempts = [a for a in (attempt_record(proj, d) for d in sorted(proj.attempts_dir.iterdir()) if d.is_dir())
                if a] if proj.attempts_dir.is_dir() else []
    attempts.sort(key=lambda a: (a["started"], a["name"]))
    scored = [a for a in attempts if a["ok"] and a["numeric"]["numeric_score"] is not None]
    best = max(scored, key=lambda a: a["numeric"]["numeric_score"])["name"] if scored else None
    runs_dir = root / agent_mod.RUNS_DIR
    runs = [r for r in (run_record(root, d) for d in sorted(runs_dir.iterdir()) if d.is_dir()) if r] if runs_dir.is_dir() else []
    kept = kept_build(proj)
    for a in attempts:
        if kept and not a["glb"] and a["name"] == kept[0]:
            a["glb"] = kept[1]
    return {"id": project_id(root), "label": label, "group": group, "name": proj.name, "path": str(root), "brief": proj.brief.strip(),
            "palette": proj.palette, "attempts": attempts, "best": best, "runs": runs, "driver": driver_of(root, runs),
            "stage": stage_of(proj.spec)}


def build_index(paths: list[str | Path], skip: tuple[str, ...] = ()) -> dict:
    """{"references": [{"key", "name", "views": {view: {"project", "path"}}, "projects": [...]}], "skipped": [...],
    "repo": where a commit can be looked at, or None, "linked": the recorded commits that are in this checkout's
    history, "light": the renderer's lights}: every project under `paths`,
    grouped by its reference views. A project with no reference views is skipped. Each project says the `group` it
    was found in (`search`), which the page lists a reference's runs by."""
    groups: dict[str, dict] = {}
    skipped = []
    for root, (label, group) in search(paths, skip=skip).items():
        try:
            proj = project_mod.load(root)
        except project_mod.ProjectError as e:
            skipped.append({"path": str(root), "why": str(e)})
            continue
        refs = proj.references
        if not refs:
            skipped.append({"path": str(root), "why": "no reference views"})
            continue
        key = reference_key(refs)
        g = groups.setdefault(key, {"key": key, "names": Counter(), "projects": [],
                                    "views": {v: {"project": project_id(proj.root),
                                                  "path": Path(refs[v]).relative_to(proj.root).as_posix()}
                                              for v in ordered_views(refs)}})
        g["names"][proj.name] += 1
        g["projects"].append(project_record(proj, label, group))
    out = []
    for g in groups.values():
        # a run is called by its directory when that says which one it is; by its path when two share a name
        names = Counter(Path(p["path"]).name for p in g["projects"])
        for p in g["projects"]:
            if names[Path(p["path"]).name] == 1:
                p["label"] = Path(p["path"]).name
        out.append({"key": g["key"], "name": g["names"].most_common(1)[0][0], "views": g["views"], "projects": g["projects"]})
    out.sort(key=lambda g: (g["name"], g["key"]))
    repo = harness_mod.repo_url()
    said = {a["harness"]["commit"] for g in out for p in g["projects"] for a in p["attempts"] if a.get("harness")}
    return {"references": out, "skipped": skipped, "paths": [str(Path(p).resolve()) for p in paths],
            "skip": sorted(skip), "repo": repo, "light": light(),
            "linked": sorted(c for c in said if harness_mod.in_history(c)) if repo else []}


def roots_of(paths: list[str | Path], skip: tuple[str, ...] = ()) -> dict[str, Path]:
    return {project_id(root): root for root in search(paths, skip=skip)}


def safe_file(roots: dict[str, Path], pid: str, rel: str) -> Path | None:
    """The file a URL names, when it is a servable type inside a project that was found; None otherwise."""
    root = roots.get(pid)
    if root is None or not rel:
        return None
    p = (root / rel).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        return None
    return p if p.is_file() and p.suffix.lower() in TYPES else None


def panel_jpeg(path: Path, height: int, width: int = 0) -> bytes:
    """A picture cropped to its subject and scaled to one height (silhouette.subject_panel), so a render and a
    reference crop of different framing sit side by side. Given a width, the subject is fitted inside width x height
    and centred on the picture's own ground: every tile on the page is one size, whatever the subject's shape."""
    height = max(32, min(PANEL_MAX_HEIGHT, height))
    im = silhouette.subject_panel(path, height)
    if width:
        width = max(32, min(PANEL_MAX_HEIGHT, width))
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
        ground = tuple(int(c) for c in silhouette.background_colour(np.asarray(im)))
        tile = Image.new("RGB", (width, height), ground)
        tile.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
        im = tile
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def etag(path: Path, *extra) -> str:
    """What a browser holds a picture under: the file's time and size, so a picture `evaluate` wrote again is fetched
    again and one that did not change is not."""
    st = path.stat()
    return '"' + "-".join(f"{x:x}" for x in (st.st_mtime_ns, st.st_size, *extra)) + '"'


# -- the server --------------------------------------------------------------------------------------------------------

def reveal_command(path: Path, platform: str = sys.platform) -> list[str] | str:
    """The command that shows a file in the machine's file manager: selected in its folder on Windows and macOS, the
    folder alone elsewhere."""
    if platform == "win32":
        # one string, as Explorer reads its own command line: the path is quoted and /select, is not. A Windows path
        # cannot hold a quote
        return f'explorer /select,"{path}"'
    if platform == "darwin":
        return ["open", "-R", str(path)]
    return ["xdg-open", str(path.parent)]


def show_in_folder(path: Path) -> None:
    """Open the file manager at a file. Not waited for: Explorer exits 1 when it has opened the window. OSError when
    the machine has no such command."""
    subprocess.Popen(reveal_command(path), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def on_this_machine(client: str, origin: str | None) -> bool:
    """Whether a request to open a folder comes from a page of this server open on this machine: the window opens
    here, so nowhere else is it any use. The caller's address is this machine's, and the page that sent it (a browser
    names it in Origin; another program sends none) is one read from this machine's own name, not another site's
    page posting here."""
    try:
        if not ipaddress.ip_address(client).is_loopback:
            return False
    except ValueError:
        return False
    return origin is None or urllib.parse.urlsplit(origin).hostname in LOCAL_HOSTS


def make_handler(paths: list[str | Path], skip: tuple[str, ...] = (), show=show_in_folder):
    panels: dict[tuple, bytes] = {}
    roots: dict[str, Path] = {}

    def found(pid: str) -> dict[str, Path]:
        """The projects a URL can name. The directories are searched when the index is read and for a project not
        seen yet, not once per picture: a page of a few hundred pictures is a few hundred requests."""
        if pid not in roots:
            roots.update(roots_of(paths, skip))
        return roots

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def send(self, status: int, body: bytes, ctype: str, tag: str = "") -> None:
            """A file goes out with its tag and is checked against it on the next request (304 when it is the same
            file); the page and the index are never kept."""
            if tag and self.headers.get("If-None-Match") == tag:
                self.send_response(304)
                self.send_header("ETag", tag)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache" if tag else "no-store")
            if tag:
                self.send_header("ETag", tag)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            url = urllib.parse.urlsplit(self.path)
            parts = [urllib.parse.unquote(x) for x in url.path.split("/") if x]
            if not parts:
                return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if parts == ["api", "index"]:
                roots.update(roots_of(paths, skip))
                return self.send(200, json.dumps(build_index(paths, skip)).encode("utf-8"), "application/json; charset=utf-8")
            if parts[0] in ("file", "panel") and len(parts) >= 3:
                f = safe_file(found(parts[1]), parts[1], "/".join(parts[2:]))
                if f is None:
                    return self.send(404, b"not found", "text/plain")
                if parts[0] == "file":
                    return self.send(200, f.read_bytes(), TYPES[f.suffix.lower()], etag(f))
                if not f.suffix.lower().startswith((".png", ".jp", ".webp")):
                    return self.send(404, b"not a picture", "text/plain")
                query = urllib.parse.parse_qs(url.query)

                def pixels(name: str, default: int) -> int:
                    try:
                        return int(query.get(name, [default])[0])
                    except ValueError:
                        return default
                height, width = pixels("h", PANEL_HEIGHT), pixels("w", 0)
                tag = etag(f, height, max(0, width))
                key = (str(f), tag)
                if key not in panels and self.headers.get("If-None-Match") != tag:
                    panels[key] = panel_jpeg(f, height, max(0, width))
                return self.send(200, panels.get(key, b""), "image/jpeg", tag)
            return self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            """The one thing the page asks to be done: /reveal/<project>/<path> shows a built model in the file
            manager."""
            parts = [urllib.parse.unquote(x) for x in urllib.parse.urlsplit(self.path).path.split("/") if x]
            if len(parts) < 3 or parts[0] != "reveal":
                return self.send(404, b"not found", "text/plain")
            if not on_this_machine(self.client_address[0], self.headers.get("Origin")):
                return self.send(403, b"only for a page open on the machine mesh-jig view runs on", "text/plain")
            f = safe_file(found(parts[1]), parts[1], "/".join(parts[2:]))
            if f is None or f.suffix.lower() not in REVEALED:
                return self.send(404, b"not found", "text/plain")
            try:
                show(f)
            except OSError as e:
                return self.send(500, str(e).encode("utf-8"), "text/plain; charset=utf-8")
            return self.send(200, b"shown", "text/plain")

    return Handler


class Server(ThreadingHTTPServer):
    """Binds its port for itself alone. http.server sets SO_REUSEADDR, and on Windows that lets a second server bind
    a port another one is answering on with no error: the first keeps getting the requests, so the address this
    one printed shows some other server's page."""
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(paths: list[str | Path], host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          skip: tuple[str, ...] = (), show=show_in_folder) -> ThreadingHTTPServer:
    """The server, bound and not yet running: call .serve_forever(). Port 0 takes a free one. OSError when the port
    is taken. `show` is what opens the file manager at a built model."""
    return Server((host, port), make_handler([Path(p) for p in paths], tuple(skip), show))


# -- the page as files -------------------------------------------------------------------------------------------------

def export(paths: list[str | Path], out: str | Path, skip: tuple[str, ...] = (), best_only: bool = False) -> dict:
    """Write the page and everything it asks the server for into `out`, for a static host: index.html, index.json,
    file/<project>/<path> for each file the index links and panel/<size>/<project>/<path>.jpg for each tile.
    {"projects", "attempts", "files", "bytes"} of what was written.

    The copy names no directory of this machine (a project's path is its label) and has no agent transcript: those
    are a run's raw log. `best_only` keeps each run's best measured attempt and drops the others, pictures and all.
    `out` must be new, empty or an earlier export, which is replaced."""
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not (out / "index.json").is_file():
        raise ValueError(f"{out} is not empty and is not an earlier export")
    index, roots = build_index(paths, skip), roots_of(paths, skip)
    files: set[tuple[str, str]] = set()       # (project id, path in the project)
    tiles: set[tuple[str, str]] = set()
    attempts = 0
    for ref in index["references"]:
        for view in ref["views"].values():
            files.add((view["project"], view["path"]))
            tiles.add((view["project"], view["path"]))
        for p in ref["projects"]:
            p["path"] = p["label"]
            if best_only:
                p["attempts"] = [a for a in p["attempts"] if a["name"] == p["best"]]
            attempts += len(p["attempts"])
            for r in p["runs"]:
                r["transcript"] = r["report"] = None
            for a in p["attempts"]:
                files.update((p["id"], a[k]) for k in ("script", "eval", "glb") if a[k])
                for w in a["views"].values():
                    files.update((p["id"], w[k]) for k in ("compare", "overlay", "render") if w[k])
                    if w["render"]:
                        tiles.add((p["id"], w["render"]))
    index.update(static=True, paths=[], skipped=[{"path": Path(x["path"]).name, "why": x["why"]} for x in index["skipped"]])
    for sub in ("file", "panel"):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    assert PAGE.count(STATIC_FLAG) == 1
    written = {"index.html": PAGE.replace(STATIC_FLAG, STATIC_FLAG.replace("false", "true")).encode("utf-8"),
               "index.json": json.dumps(index).encode("utf-8"), ".nojekyll": b""}
    for pid, rel in sorted(files):
        src = safe_file(roots, pid, rel)
        if src is not None:
            written[f"file/{pid}/{rel}"] = src.read_bytes()
    for pid, rel in sorted(tiles):
        src = safe_file(roots, pid, rel)
        if src is not None:
            written.update({f"panel/{px}/{pid}/{rel}.jpg": panel_jpeg(src, px, px) for px in TILE_SIZES})
    for name, body in written.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    return {"projects": sum(len(r["projects"]) for r in index["references"]), "attempts": attempts,
            "files": len(written), "bytes": sum(len(b) for b in written.values())}


# -- the page ----------------------------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>mesh-jig results</title>
<style>
:root { color-scheme:light dark; --bg:#eef0f2; --panel:#fff; --ink:#14191d; --dim:#5b656d; --line:#d4d9dd; --soft:#e7eaed;
  --accent:#1b4e9b; --band:#1b4e9b; --pick:#e3ecf9; --good:#1b7a50; --bad:#b4372a; --track:#dde2e6;
  --cc:#b04a17; --codex:#6741b5; --api:#0b737b;
  --din:Bahnschrift, "DIN Alternate", "Roboto Condensed", "Arial Narrow", system-ui, sans-serif;
  --text:"Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171a; --panel:#1c2024; --ink:#e7eaec; --dim:#97a1a9; --line:#2f363c;
  --soft:#262b30; --accent:#86b4f2; --band:#173762; --pick:#1f2d42; --good:#6dc79b; --bad:#f08a7c; --track:#333b42;
  --cc:#f0a074; --codex:#b9a4f6; --api:#63c8ce; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 var(--text); }
button { font:inherit; color:inherit; cursor:pointer; }
a { color:var(--accent); }
:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.dim, small { color:var(--dim); font-size:12px; }
.fail { color:var(--bad); } .ok { color:var(--good); }

/* the band: which reference */
.top { display:flex; gap:20px; align-items:center; padding:10px 20px; background:var(--band); color:#fff; }
.mark { font:600 21px/1 var(--din); letter-spacing:.01em; white-space:nowrap; }
.mark span { font-weight:300; opacity:.8; }
#refs { display:flex; gap:8px; flex:1; min-width:0; overflow-x:auto; padding:2px; }
.ref { display:flex; gap:9px; align-items:center; flex:none; text-align:left; padding:4px 12px 4px 4px; border:1px solid transparent;
  border-radius:5px; background:rgba(255,255,255,.12); }
.ref:hover { background:rgba(255,255,255,.22); }
.ref[aria-pressed="true"] { background:var(--panel); color:var(--ink); }
.ref img { width:40px; height:40px; border-radius:3px; display:block; }
.ref b { display:block; font:600 15px/1.2 var(--din); }
.ref small { color:inherit; opacity:.75; }
.ref:focus-visible, #reload:focus-visible { outline-color:#fff; }
#reload { border:1px solid rgba(255,255,255,.45); background:none; border-radius:4px; padding:5px 12px; white-space:nowrap; }
#reload:hover { background:rgba(255,255,255,.15); }

/* the board on the left, the picked attempt on the right */
#page { display:grid; grid-template-columns:minmax(0,1fr); gap:20px; padding:18px 20px; align-items:start; }
h1 { font:600 26px/1.1 var(--din); margin:0; overflow-wrap:anywhere; }
.facts { display:flex; flex-wrap:wrap; gap:4px 18px; margin:6px 0 14px; color:var(--dim); }
.facts b { color:var(--ink); font:600 15px var(--din); }
.tabs { display:flex; gap:2px; border-bottom:1px solid var(--line); }
.tabs button { border:0; background:none; padding:7px 14px; font:500 15px var(--din); color:var(--dim); border-bottom:3px solid transparent;
  margin-bottom:-1px; }
.tabs button:hover { color:var(--ink); }
.tabs button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--accent); }
.tabs i { font-style:normal; font-weight:300; margin-left:5px; }
.sheet { background:var(--panel); border:1px solid var(--line); border-top:0; border-radius:0 0 6px 6px; overflow:clip; }
.hint { color:var(--dim); padding:16px; margin:0; max-width:60ch; }

/* what drove a run: a colour and a shape each, and always the name */
.drv { --k:var(--dim); display:inline-flex; align-items:center; gap:5px; padding:2px 7px 2px 6px; border-radius:3px; white-space:nowrap;
  vertical-align:middle; font:600 12px/1.35 var(--din); letter-spacing:.02em; color:var(--k); background:color-mix(in srgb, var(--k) 14%, transparent); }
.drv::before { content:""; flex:none; width:7px; height:7px; background:currentColor; }
.claude-code { --k:var(--cc); } .codex { --k:var(--codex); } .agent { --k:var(--api); }
.drv.claude-code::before { border-radius:50%; }
.drv.codex::before { transform:rotate(45deg) scale(.9); }
.effort { font:500 12px/1.3 var(--din); font-style:normal; color:var(--dim); border:1px solid var(--line); border-radius:3px; padding:0 4px; margin-left:2px;
  white-space:nowrap; }
.drivers { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 14px; }
.driver { --k:var(--dim); flex:1 1 190px; max-width:320px; display:grid; gap:4px; text-align:left; padding:8px 12px 10px; background:var(--panel);
  border:1px solid var(--line); border-top:3px solid var(--k); border-radius:5px; }
.driver:hover { border-color:var(--k); }
.driver[aria-pressed="true"] { border-color:var(--k); background:color-mix(in srgb, var(--k) 11%, var(--panel)); }
.driver .head { display:flex; align-items:center; gap:8px; }
.driver .head small { margin-left:auto; }
.driver .best { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.driver .best b { font:600 20px/1.1 var(--din); font-variant-numeric:tabular-nums; margin-right:5px; }
.by { display:flex; flex-wrap:wrap; align-items:center; gap:4px 8px; margin:6px 0 2px; }
.by b { font:600 15px var(--din); }

/* a model in 3D: one canvas over the stage, drawn into the rectangle of each cell */
.stage { position:relative; user-select:none; }
canvas.gl { position:absolute; inset:0; width:100%; height:100%; z-index:1; pointer-events:none; }
.cell { position:relative; display:grid; place-items:center; overflow:hidden; border-radius:3px; background:var(--stage, #dddbd6); cursor:grab;
  touch-action:pan-y; }
.cell:active { cursor:grabbing; }
.cell::after { content:attr(data-say); padding:8px; color:#4d565d; font-size:12px; text-align:center; }
.bar { position:relative; z-index:2; display:flex; flex-wrap:wrap; gap:4px 5px; align-items:center; padding-top:6px; font-size:12px; color:var(--dim); }
.bar button { border:1px solid var(--line); background:var(--panel); border-radius:3px; padding:1px 8px; font:500 13px/1.5 var(--din); color:var(--dim); }
.bar button:hover { color:var(--ink); border-color:var(--accent); }
.bar button[aria-pressed="true"] { color:var(--ink); background:var(--pick); border-color:var(--accent); }
.bar .gap { margin-left:8px; }
.bar .how { margin-left:auto; }
.inset { position:absolute; top:8px; right:8px; z-index:2; width:27%; max-width:150px; border:1px solid rgba(0,0,0,.3); border-radius:3px; overflow:hidden;
  background:#fff; color:#222; text-decoration:none; font:500 11px/1.6 var(--din); text-align:center; }
.inset[hidden] { display:none; }
.inset img { display:block; width:100%; aspect-ratio:1; }
#inspect .stage { margin-top:12px; }
#inspect .cell { aspect-ratio:16 / 10; }
.lineup .bar { padding:7px 10px; border-bottom:1px solid var(--line); }
.slots { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:6px; padding:8px; }
.slot { padding:5px; border:1px solid transparent; border-radius:5px; cursor:pointer; min-width:0; }
.slot:hover { background:var(--soft); }
.slot.on { background:var(--pick); border-color:var(--accent); }
.slot .cell { aspect-ratio:4 / 5; }
.slot .cap { display:grid; gap:2px; padding:6px 2px 2px; justify-items:start; }
.slot .cap > * { max-width:100%; overflow-wrap:anywhere; }
.slot .score { font:600 21px/1.1 var(--din); font-variant-numeric:tabular-nums; }

table { border-collapse:separate; border-spacing:0; width:100%; table-layout:fixed; }
th, td { padding:6px 8px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; font-weight:400; }
tbody tr:last-child > * { border-bottom:0; }
thead th, thead td { position:sticky; top:0; z-index:2; background:var(--panel); }
thead th { font:500 13px var(--din); color:var(--dim); height:32px; vertical-align:middle; white-space:nowrap; }
tr.row { cursor:pointer; }
tr.row:hover > * { background:var(--soft); }
tr.row.on > * { background:var(--pick); }
tr.row.on > :first-child { box-shadow:inset 3px 0 0 var(--accent); }
tr.row:focus-visible { outline-offset:-2px; }

.tile { display:block; width:100%; aspect-ratio:1; object-fit:contain; border-radius:3px; background:var(--soft); }
.none { display:grid; place-items:center; aspect-ratio:1; padding:6px; border:1px dashed var(--line); border-radius:3px;
  color:var(--dim); font-size:12px; text-align:center; }
.meter { display:flex; align-items:center; gap:6px; font:500 13px var(--din); font-variant-numeric:tabular-nums; }
.meter .bar { flex:1; height:4px; background:var(--track); min-width:16px; }
.meter i { display:block; height:100%; background:var(--accent); }

/* best of each run: one row a run, one column a view, the reference held at the top */
.grid col.who { width:184px; }
.who .drv { margin-top:5px; }
.grid .tile, .grid .none { max-width:190px; }
.grid td .meter { margin-top:4px; max-width:190px; }
.grid thead .refrow > * { top:32px; height:auto; vertical-align:top; border-bottom:2px solid var(--ink); }
.grid thead .refrow th { color:var(--ink); font-size:15px; padding-top:10px; }
.tools { display:flex; flex-wrap:wrap; gap:6px 20px; align-items:center; padding:7px 8px; border-bottom:1px solid var(--line);
  font-size:13px; color:var(--dim); }
.tools button { border:0; background:none; padding:2px 6px; border-radius:3px; }
.tools button:hover { color:var(--ink); }
.tools button[aria-pressed="true"] { color:var(--ink); background:var(--soft); }
.tools .why { margin-left:auto; }
/* a section of runs: the experiment they were found in, closed to its top scoring run until it is opened */
tr.section > * { padding:0; background:var(--soft); border-top:1px solid var(--line); text-align:left; }
.fold { display:flex; gap:8px; align-items:baseline; width:100%; border:0; background:none; padding:6px 10px; text-align:left;
  font:600 14px var(--din); }
.fold i { width:10px; font-style:normal; color:var(--dim); }
.fold small { font-weight:400; }
.fold:hover b { color:var(--accent); }
.tools.folds .fold { width:auto; padding:2px 8px; background:var(--soft); }
.grid thead th.counted { color:var(--ink); }
.grid thead th small { margin-left:5px; }
.who .score { display:block; font:600 21px/1.1 var(--din); font-variant-numeric:tabular-nums; }
.who b { display:block; font-weight:600; overflow-wrap:anywhere; margin-top:3px; }
.who small { display:block; overflow-wrap:anywhere; }
.commit { display:block; font:400 12px/1.5 ui-monospace, Consolas, monospace; color:var(--dim); }
a.commit:hover { color:var(--accent); }
.headline .commit { display:inline; }

/* all attempts */
.list col.score { width:124px; } .list col.run { width:18%; }
.list th button { border:0; background:none; padding:0; font:inherit; }
.list th button:hover { color:var(--ink); }
.list th[aria-sort] button { color:var(--ink); }
.list td { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; vertical-align:middle; }
.list .n { text-align:right; font:400 14px var(--din); font-variant-numeric:tabular-nums; }
.list td small { display:block; overflow:hidden; text-overflow:ellipsis; }
.list tr.best .meter i { background:var(--good); }
.list tr.under td { color:var(--dim); }
.list tr.under td:nth-child(3) { padding-left:22px; }
.list .more { margin-left:5px; padding:0 5px; border:1px solid var(--line); border-radius:3px; background:var(--panel); color:var(--dim);
  font:500 12px/1.5 var(--din); font-variant-numeric:tabular-nums; }
.list .more:hover { color:var(--ink); border-color:var(--accent); }

/* agent runs */
.agents .cols, .agents summary { display:grid; grid-template-columns:minmax(0,1fr) repeat(6, 68px) 20px; gap:10px; align-items:center; padding:7px 10px; }
.agents .cols { font:500 13px var(--din); color:var(--dim); border-bottom:1px solid var(--line); }
.agents details { border-bottom:1px solid var(--line); }
.agents details:last-child { border-bottom:0; }
.agents summary { cursor:pointer; list-style:none; }
.agents summary::-webkit-details-marker { display:none; }
.agents summary:hover { background:var(--soft); }
.agents summary::after { content:"+"; font:300 18px/1 var(--din); color:var(--dim); text-align:center; }
.agents details[open] summary::after { content:"\2212"; }
.agents .n { text-align:right; font:400 14px var(--din); font-variant-numeric:tabular-nums; }
.agents summary b { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.agents summary small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.agents .body { padding:4px 10px 14px; }
.agents .body p { margin:6px 0; }
.act { border:1px solid var(--line); background:var(--panel); border-radius:4px; padding:4px 10px; margin-right:6px; }
.act:hover { border-color:var(--accent); }
pre { background:var(--soft); padding:10px; border-radius:4px; overflow:auto; max-height:280px; white-space:pre-wrap; font-size:12px; margin:8px 0; }
.msg { border-left:3px solid var(--line); padding:2px 8px; margin:6px 0; white-space:pre-wrap; font-size:12px; overflow-wrap:anywhere; }
.msg.assistant { border-color:var(--accent); } .msg.tool { border-color:var(--good); color:var(--dim); }
.msg b { font:500 12px var(--din); color:var(--dim); display:block; }

/* the picked attempt */
#inspect { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; min-width:0; }
#inspect:empty { display:none; }
#inspect h2 { font:600 20px/1.2 var(--din); margin:0 0 2px; overflow-wrap:anywhere; }
#inspect h2 span { font-weight:300; }
.where { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.headline { display:flex; align-items:baseline; gap:10px; margin:14px 0 2px; }
.headline b { font:600 34px/1 var(--din); font-variant-numeric:tabular-nums; }
.headline span { font:500 15px var(--din); }
.what { color:var(--dim); font-size:12px; margin:0; max-width:70ch; }
.measures { display:grid; grid-template-columns:max-content 112px minmax(0,1fr); gap:8px 14px; align-items:baseline; margin:12px 0 8px; }
.measures .name { font:500 14px var(--din); }
.measures .sub { padding-left:12px; font-weight:400; }
.measures .mm { font:500 13px var(--din); font-variant-numeric:tabular-nums; }
.palette { display:flex; flex-wrap:wrap; gap:2px 12px; font-size:12px; color:var(--dim); margin:10px 0 0; }
.palette .sw, .stack i { box-shadow:inset 0 0 0 1px rgba(128,128,128,.45); }
.note { margin:10px 0; max-width:70ch; }
.take { display:flex; flex-wrap:wrap; align-items:center; gap:4px 10px; margin:8px 0 0; font-size:12px; }
.take .act { margin:0; font-size:13px; }
.three { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:8px; }
.caps { position:sticky; top:0; z-index:1; background:var(--panel); padding:8px 0 6px; margin-top:14px; align-items:end; border-bottom:2px solid var(--ink);
  font:500 13px var(--din); color:var(--dim); }
.legend { display:flex; flex-wrap:wrap; gap:0 9px; font:400 12px var(--text); }
.sw { display:inline-block; width:9px; height:9px; margin-right:4px; }
.view { padding:10px 0 12px; border-bottom:1px solid var(--line); }
.view:last-child { border-bottom:0; padding-bottom:0; }
.view header { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.view h3 { font:600 15px var(--din); margin:0; min-width:44px; }
.stats { display:flex; flex-wrap:wrap; gap:4px 16px; margin-bottom:4px; }
.stats .meter { flex:1 1 140px; }
.meter .lab { font:400 12px var(--text); color:var(--dim); white-space:nowrap; }
.view .fine { font-size:12px; color:var(--dim); margin:0 0 6px; }
.tag { font-size:12px; color:var(--dim); border:1px solid var(--line); border-radius:3px; padding:0 5px; }
.reading, .stations { margin-top:8px; font-size:12px; color:var(--dim); }
.reading:empty, .stations:empty, #checks:empty { display:none; }
.chips { display:flex; flex-wrap:wrap; gap:3px 12px; margin-top:2px; }
.chips b { font:500 13px var(--din); color:var(--ink); font-variant-numeric:tabular-nums; margin-left:4px; }
#checks { margin:12px 0 4px; }
#checks h3 { font:600 15px var(--din); margin:10px 0 2px; }
.checks { display:grid; grid-template-columns:max-content minmax(0,1fr) max-content; gap:3px 12px; font-size:12px; margin-top:6px; }
.checks .name { font:500 13px var(--din); }
.checks .said { color:var(--dim); }
.mix { display:grid; grid-template-columns:58px minmax(60px,1fr) minmax(0,1.4fr); gap:8px; align-items:center; margin-top:3px; }
.mix span:last-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.stack { display:flex; gap:2px; height:10px; }
.stack i { min-width:3px; }
.view header a { margin-left:auto; font-size:12px; }
.tile.diff { background:#fafafa; }

footer { padding:4px 20px 24px; color:var(--dim); font-size:12px; overflow-wrap:anywhere; }

@media (min-width:1100px) {
  #page { grid-template-columns:minmax(0,1.3fr) minmax(0,1fr); }
  #inspect { position:sticky; top:12px; max-height:calc(100vh - 24px); overflow-y:auto; }
}
@media (max-width:760px), (max-height:560px) { thead th, thead td { position:static; } }
@media (max-width:760px) {
  .top { flex-wrap:wrap; gap:10px; } #refs { order:3; flex-basis:100%; }
  .mark { flex:1; }
  .sheet { overflow-x:auto; } .grid { min-width:calc(120px + var(--views) * 112px); } .grid col.who { width:120px; }
  .list { min-width:820px; } .agents { min-width:640px; }
  .bar .how { margin-left:0; flex-basis:100%; }
  .measures { grid-template-columns:max-content minmax(0,1fr); } .measures .what { grid-column:1 / -1; margin-top:-5px; }
}
</style></head><body>
<header class="top"><div class="mark">mesh-jig <span>results</span></div><nav id="refs" aria-label="References"></nav>
  <button id="reload" title="Read the results from disk again">Reload results</button></header>
<div id="page"><main id="board"></main><aside id="inspect" aria-label="The picked attempt"></aside></div>
<footer id="foot"></footer>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const enc = p => p.split("/").map(encodeURIComponent).join("/");
const STATIC = false;        // true in a copy `mesh-jig view --export` wrote: files beside the page, tiles already cut
const file = (pid, rel) => `${STATIC ? "" : "/"}file/${pid}/${enc(rel)}`;
// the file manager is opened by the server, on its own machine: asked for only by a page that is open there
const here = () => !STATIC && ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
const reveal = (pid, rel) => `/reveal/${pid}/${enc(rel)}`;
const tile = (pid, rel, px) => STATIC ? `panel/${px}/${pid}/${enc(rel)}.jpg` : `/panel/${pid}/${enc(rel)}?h=${px}&w=${px}`;
const f = (x, d = 3) => x == null ? "-" : Number(x).toFixed(d);
const count = (n, word) => `<b>${n}</b> ${word}${n === 1 ? "" : "s"}`;
const meter = (x, d = 2, label = "") => x == null ? "" : `<span class="meter">${label ? `<span class="lab">${label}</span>` : ""}<span class="bar"><i style="width:${Math.max(0, Math.min(1, x)) * 100}%"></i></span>${f(x, d)}</span>`;
const wide = matchMedia("(min-width: 1100px)");
let data = null;
const state = {ref: null, pick: null, tab: "runs", order: "score", measure: "iou", sort: "numeric_score", desc: true, driver: null,
  every: false, open: new Set(),         // the attempts list: every attempt, or one a run with these runs opened
  groups: new Set()};                    // the sections that are opened: the others show their top scoring run alone
// what the bar under a picture can show, per view
const MEASURES = [["iou", "outline overlap"], ["zone", "colour zones"], ["structure", "markings"], ["resemblance", "resemblance"]];
// what each number is, in the words the picked attempt is explained in
const MEANING = {
  numeric_score: "The number attempts are ranked on: the mean of outline overlap and resemblance, or the mix the project states in criteria.score.",
  ortho_iou: "How much of the build's silhouette and the reference's coincide, each cropped to its subject and scaled to one canvas (intersection over union). 1 is the same outline.",
  resemblance: "What the outline cannot see: the mean of colour zones and markings.",
  zone: "Of the cells inside both outlines, the share whose nearest palette colour is the same on the build as on the reference: colour in the right place.",
  structure: "Cell by cell, whether edges run the same way with the same strength: panel lines, trim and features in the right place.",
  mae_mm: "The mean gap between the model's own outline and the reference's, at the stations the project measures at. In millimetres, reported beside the score and not part of it.",
};
const words = xs => xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];
const pct = x => Math.round(x * 100) + "%";
const colour = hex => /^#[0-9a-f]{3,8}$/i.test(hex || "") ? hex : "#888";

function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  state.ref = q.get("ref"); state.pick = q.get("a"); state.tab = q.get("t") || "runs";
  state.measure = MEASURES.some(m => m[0] === q.get("m")) ? q.get("m") : "iou";
  state.driver = q.get("d"); state.every = q.get("e") === "all";
  state.groups = new Set(q.getAll("g"));
}
function writeHash() {
  const q = new URLSearchParams();
  if (state.ref) q.set("ref", state.ref);
  if (state.pick) q.set("a", state.pick);
  if (state.tab !== "runs") q.set("t", state.tab);
  if (state.measure !== "iou") q.set("m", state.measure);
  if (state.driver) q.set("d", state.driver);
  if (state.every) q.set("e", "all");
  for (const g of state.groups) q.append("g", g);
  history.replaceState(null, "", "#" + q.toString());
}
const current = () => data.references.find(r => r.key === state.ref);
const rows = ref => ref.projects.flatMap(p => p.attempts.map(a => ({p, a, id: p.id + "/" + a.name})));
const score = r => r.a.ok ? r.a.numeric.numeric_score : null;

// what drove a run, from what the run recorded (the index's `driver`): Claude Code, Codex, or mesh-jig agent calling
// a model over an API. The runs on the page can be narrowed to one of them.
const KINDS = ["claude-code", "codex", "agent", "other", "none"];
const KIND_IS = {"claude-code": "a coding agent, with the skill and the command line", codex: "a coding agent, with the skill and the command line",
  agent: "mesh-jig agent, calling a model over an API", other: "another tool", none: "the run did not record what drove it"};
const kindOf = p => p.driver ? p.driver.kind : "none";
const shown = ref => ref.projects.filter(p => !state.driver || kindOf(p) === state.driver);
function badge(p) {
  const d = p.driver;
  if (!d) return "";
  const how = d.kind === "agent" ? `mesh-jig agent, the tool loop here, calling ${d.detail || "a model over an API"}` : `${d.name}${d.detail ? " " + d.detail : ""}`;
  return `<span class="drv ${esc(d.kind)}" title="Driven by ${esc(how)}">${esc(d.name)}</span>`;
}
// what a run is called: its model and effort when it recorded them, else its directory
function runName(p) {
  const d = p.driver, model = d && d.model ? String(d.model).split("/").pop() : "";
  return model ? `<span title="${esc(d.model)}">${esc(model)}</span>${d.effort ? ` <i class="effort" title="effort">${esc(d.effort)}</i>` : ""}` : esc(p.label);
}
// A reference's runs are listed in sections, one for each group they were found in (the index's `group`: the
// experiment's directory). A section is closed until it is opened: closed, it lists its top scoring run alone. One
// that holds the picked attempt in another of its runs is open. A reference whose runs are all of one group has no
// sections, and its runs are listed as they are.
const topScore = p => p.attempts.find(a => a.name === p.best)?.numeric.numeric_score ?? -1;
const sectioned = ref => new Set(ref.projects.map(p => p.group)).size > 1;
function sections(ref) {
  const by = new Map();
  for (const p of shown(ref)) by.set(p.group, [...(by.get(p.group) || []), p]);
  const picked = ref.projects.find(p => state.pick && state.pick.startsWith(p.id + "/"));
  const list = [...by].map(([name, ps]) => {
    const top = ps.reduce((a, b) => topScore(b) > topScore(a) ? b : a);
    return {name, ps, top, open: state.groups.has(name) || !!picked && picked !== top && ps.includes(picked)};
  });
  return list.sort(state.order === "name" ? (x, y) => x.name.localeCompare(y.name) : (x, y) => topScore(y.top) - topScore(x.top));
}
const listed = ref => sectioned(ref) ? sections(ref).flatMap(s => s.open ? s.ps : [s.top]) : shown(ref);
const sectionHead = s => s.ps.length === 1 ? `<span class="fold"><i></i><b>${esc(s.name)}</b><small>1 run</small></span>`      // nothing to open
  : `<button class="fold" data-group="${esc(s.name)}" aria-expanded="${s.open}" title="${s.open ? "Show only the top scoring run" : `Show all ${s.ps.length} runs`}">
  <i>${s.open ? "▾" : "▸"}</i><b>${esc(s.name)}</b><small>${s.ps.length} runs${s.open ? "" : ", the top scoring one shown"}</small></button>`;
function toggleGroup(name) {
  const ref = current(), s = ref && sections(ref).find(x => x.name === name);
  if (!s) return;
  if (s.open) {
    state.groups.delete(name);
    // a section closes to its top run: an attempt picked in another of its runs gives way to that run's best
    if (s.ps.some(p => p !== s.top && state.pick && state.pick.startsWith(p.id + "/"))) {
      const a = s.top.best ?? s.top.attempts[0]?.name;
      state.pick = a ? s.top.id + "/" + a : null;
    }
  } else state.groups.add(name);
  writeHash(); drawBoard(); drawInspector();
}
// the best attempt of each run that is listed (of `ps` when given), in the order the board is in
function bestRuns(ref, ps = listed(ref)) {
  const runs = ps.map(p => ({p, a: p.attempts.find(x => x.name === p.best)}));
  if (state.order === "score") runs.sort((x, y) => (y.a?.numeric.numeric_score ?? -1) - (x.a?.numeric.numeric_score ?? -1));
  return runs;
}
function drivers(ref) {
  const groups = KINDS.map(k => [k, ref.projects.filter(p => kindOf(p) === k)]).filter(g => g[1].length);
  if (groups.length === 1 && groups[0][0] === "none") return "";
  return `<div class="drivers" role="group" aria-label="Show only the runs driven by">${groups.map(([k, ps]) => {
    const best = ps.map(p => ({p, a: p.attempts.find(x => x.name === p.best)})).filter(r => r.a)
      .sort((x, y) => y.a.numeric.numeric_score - x.a.numeric.numeric_score)[0];
    return `<button class="driver ${k === "none" ? "" : k}" data-driver="${k}" aria-pressed="${state.driver === k}" title="${state.driver === k ? "Show every run" : "Show only these runs"}">
      <span class="head">${k === "none" ? `<span class="drv">Not recorded</span>` : badge(ps[0])}<small>${ps.length} run${ps.length === 1 ? "" : "s"}</small></span>
      <span class="best">${best ? `<b>${f(best.a.numeric.numeric_score)}</b>best, ${runName(best.p)}` : "nothing measured yet"}</span>
      <small>${KIND_IS[k]}</small></button>`;
  }).join("")}</div>`;
}
// the mesh-jig that measured an attempt, as a link to its commit when the checkout has a remote and the commit is
// in its history (a record made on another history is plain text: the remote has no such page). Two results are
// comparable when this is the same, and a change to the guidance alone shows up here
function commit(a) {
  const h = a && a.harness;
  if (!h) return "";
  const text = esc(h.commit.slice(0, 7)) + (h.dirty ? "+" : "");
  const title = "Measured by mesh-jig at commit " + esc(h.commit) + (h.dirty ? ", with uncommitted changes to the harness" : "");
  return data.repo && (data.linked || []).includes(h.commit) ? `<a class="commit" target="_blank" rel="noopener" title="${title}" href="${esc(data.repo)}/commit/${esc(h.commit)}">${text}</a>`
    : `<span class="commit" title="${title}">${text}</span>`;
}
const bestOf = ref => rows(ref).filter(r => score(r) != null).sort((x, y) => score(y) - score(x))[0];

function drawRefs() {
  $("refs").innerHTML = data.references.map(r => {
    const first = Object.values(r.views)[0], n = rows(r).length;
    return `<button class="ref" data-ref="${r.key}" aria-pressed="${r.key === state.ref}">
      ${first ? `<img src="${tile(first.project, first.path, 96)}" alt="">` : ""}
      <span><b>${esc(r.name)}</b><small>${r.projects.length} run${r.projects.length === 1 ? "" : "s"}, ${n} attempt${n === 1 ? "" : "s"}</small></span></button>`;
  }).join("");
}

// best of each run: the reference row stays under the column names while the runs scroll past it
function grid(ref) {
  const views = Object.keys(ref.views);
  const runs = bestRuns(ref);
  const counted = bestOf(ref)?.a.counted || [];
  const tools = `<div class="tools"><span>Bar under each picture ${MEASURES.map(([k, label]) =>
      `<button data-measure="${k}" aria-pressed="${state.measure === k}">${label}</button>`).join("")}</span>
    <span>Order by ${[["score", "score"], ["name", "name"]].map(([k, label]) =>
      `<button data-order="${k}" aria-pressed="${state.order === k}">${label}</button>`).join("")}</span>
    ${counted.length ? `<span class="why">The score counts the ${esc(words(counted))} view${counted.length === 1 ? "" : "s"}</span>` : ""}</div>`;
  const head = `<tr><th>run</th>${views.map(v => `<th class="${counted.includes(v) ? "counted" : ""}">${esc(v)}${counted.includes(v) ? "<small>scored</small>" : ""}</th>`).join("")}</tr>
    <tr class="refrow"><th scope="row">Reference</th>${views.map(v => `<td><a target="_blank" title="Open the picture" href="${file(ref.views[v].project, ref.views[v].path)}">
      <img class="tile" src="${tile(ref.views[v].project, ref.views[v].path, 240)}" alt="reference ${esc(v)}"></a></td>`).join("")}</tr>`;
  const runRow = ({p, a}) => {
    const dir = p.driver && p.driver.model ? `<small>${esc(p.label)}</small>` : "";
    if (!a) return `<tr><th class="who" scope="row">${badge(p)}<b>${runName(p)}</b>${dir}<small>${p.attempts.length ? "no attempt built and measured" : "no attempt yet"}</small></th><td colspan="${views.length}"></td></tr>`;
    return `<tr class="row" tabindex="0" data-pick="${esc(p.id + "/" + a.name)}"><th class="who" scope="row">
      <span class="score">${f(a.numeric.numeric_score)}</span>${commit(a)}${badge(p)}<b>${runName(p)}</b>${dir}<small>${esc(a.name)}</small>
      ${(a.counted || []).join() === counted.join() ? "" : `<small class="fail">scored on ${esc(words(a.counted || []))}</small>`}</th>${views.map(v => {
        const w = a.views[v];
        return `<td>${w && w.render ? `<img class="tile" loading="lazy" src="${tile(p.id, w.render, 240)}" alt="${esc(p.label)} ${esc(v)}">${meter(w[state.measure])}`
          : `<span class="none">no render</span>`}</td>`;
      }).join("")}</tr>`;
  };
  const body = !sectioned(ref) ? runs.map(runRow).join("") : sections(ref).map(s =>
    `<tr class="section"><th colspan="${views.length + 1}">${sectionHead(s)}</th></tr>` + bestRuns(ref, s.open ? s.ps : [s.top]).map(runRow).join("")).join("");
  return `${tools}<table class="grid" style="--views:${views.length}"><colgroup><col class="who">${views.map(() => "<col>").join("")}</colgroup>
    <thead>${head}</thead><tbody>${body}</tbody></table>`;
}

const COLS = [["numeric_score", "score", score, 3], ["run", "run", r => r.p.label], ["attempt", "attempt and note", r => r.a.name],
  ["ortho_iou", "overlap", r => r.a.numeric.ortho_iou, 3], ["resemblance", "resemblance", r => r.a.numeric.resemblance, 3],
  ["zone", "zones", r => r.a.numeric.zone, 2], ["structure", "markings", r => r.a.structure, 2],
  ["mae_mm", "outline mm", r => r.a.numeric.mae_mm, 2]];
const COL_WIDTH = {ortho_iou: 74, resemblance: 96, zone: 62, structure: 82, mae_mm: 88};

function attempts(ref) {
  const all = rows(ref).filter(r => shown(ref).includes(r.p)), best = bestOf(ref);
  if (!all.length) return `<p class="hint">No attempt has been evaluated yet. <code>mesh-jig eval</code> writes one; reload when it has run.</p>`;
  // a measure no attempt of this reference has is a column of dashes: leave it out
  const cols = COLS.filter((c, i) => i < 3 || all.some(r => c[2](r) != null));
  const col = cols.find(c => c[0] === state.sort) || cols[0];
  const order = (x, y) => {
    const a = col[2](x), b = col[2](y);
    if (a == null || b == null) return (a == null) - (b == null);
    return (typeof a === "string" ? a.localeCompare(b) : a - b) * (state.desc ? -1 : 1);
  };
  // One row a run: its best attempt, or its last when none was measured. A run that is opened, or that holds the
  // picked attempt, has its other attempts under that row in the order they were made. Asked for, every attempt is
  // a row of its own and they are ranked together.
  const lead = p => { const own = all.filter(r => r.p === p); return own.find(r => r.a.name === p.best) || own[own.length - 1]; };
  const picked = all.find(r => r.id === state.pick);
  const opened = r => state.open.has(r.p.id) || !!picked && picked.p === r.p && picked !== r;
  const runRows = ps => ps.map(lead).filter(Boolean).sort(order).flatMap(r =>
    [{...r, more: r.p.attempts.length - 1, open: opened(r)}, ...(opened(r) ? all.filter(x => x.p === r.p && x !== r).map(x => ({...x, under: true})) : [])]);
  // ranked together, every attempt is listed whatever section its run is in
  const list = state.every ? all.slice().sort(order) : !sectioned(ref) ? runRows(shown(ref))
    : sections(ref).flatMap(s => [{section: s}, ...runRows(s.open ? s.ps : [s.top])]);
  const more = r => !r.more ? "" : `<button class="more" data-open="${esc(r.p.id)}" aria-expanded="${r.open}" title="${r.open ? "Hide" : "Show"} this run's other ${
    r.more === 1 ? "attempt" : r.more + " attempts"}">${r.open ? "hide" : "+" + r.more}</button>`;
  const tools = `<div class="tools"><span>Show ${[["best", "the best attempt of each run"], ["all", "every attempt"]].map(([k, label]) =>
    `<button data-every="${k}" aria-pressed="${state.every === (k === "all")}">${label}</button>`).join("")}</span></div>`;
  const head = cols.map((c, i) => `<th class="${i > 2 ? "n" : ""}" ${c[0] === col[0] ? `aria-sort="${state.desc ? "descending" : "ascending"}"` : ""}>
    <button data-sort="${c[0]}">${c[1]}${c[0] === col[0] ? (state.desc ? " ▾" : " ▴") : ""}</button></th>`).join("");
  const body = list.map(r => r.section ? `<tr class="section"><td colspan="${cols.length}">${sectionHead(r.section)}</td></tr>`
    : `<tr class="row ${best && r.id === best.id ? "best" : ""} ${r.under ? "under" : ""}" tabindex="0" data-pick="${esc(r.id)}">
    <td>${r.a.ok ? meter(score(r), 3) || "-" : `<span class="fail">failed at ${esc(r.a.stage || "build")}</span>`}${commit(r.a)}</td>
    <td title="${esc(r.p.label)}">${r.under ? "" : `${badge(r.p) || esc(r.p.label)}${more(r)}${r.p.driver ? `<small>${esc(r.p.label)}</small>` : ""}`}</td><td title="${esc(r.a.note || r.a.name)}">${esc(r.a.name)}<small>${esc(r.a.note)}</small></td>
    ${cols.slice(3).map(c => `<td class="n">${f(c[2](r), c[3])}</td>`).join("")}</tr>`).join("");
  return `${tools}<table class="list"><colgroup><col class="score"><col class="run"><col>${cols.slice(3).map(c => `<col style="width:${COL_WIDTH[c[0]]}px">`).join("")}</colgroup>
    <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function agentRuns(ref) {
  const list = shown(ref).flatMap(p => p.runs.map(r => ({p, r})))
    .sort((x, y) => (y.r.best?.numeric_score ?? -1) - (x.r.best?.numeric_score ?? -1));
  const n = (x, text) => `<span class="n">${x == null ? "-" : text}</span>`;
  return `<div class="agents"><div class="cols"><span>model and run</span><span class="n">best</span><span class="n">attempts</span>
      <span class="n">turns</span><span class="n">cost</span><span class="n">time</span><span class="n">tokens</span><span></span></div>` +
    list.map(({p, r}) => {
      const bestId = r.best && p.attempts.some(a => a.name === r.best.name) ? p.id + "/" + r.best.name : null;
      return `<details><summary><span><b>${esc(r.model || r.name)}</b><small>${badge(p)} ${esc(p.label === r.name ? r.name : p.label + " / " + r.name)}</small></span>
        ${n(r.best, f(r.best?.numeric_score) + commit(p.attempts.find(a => a.name === r.best?.name)))}${n(r.evaluations, r.evaluations)}${n(r.turns, r.turns)}
        ${n(r.cost_usd || null, "$" + Number(r.cost_usd).toFixed(3))}${n(r.wall_s, Math.round(r.wall_s) + " s")}
        ${n(r.prompt_tokens, Math.round((r.prompt_tokens + (r.completion_tokens || 0)) / 1000) + "k")}</summary>
        <div class="body"><p class="${r.done ? "ok" : "fail"}">${r.done ? "The model called done." : "The model did not call done."}${r.error ? " " + esc(r.error) : ""}</p>
        ${r.summary ? `<pre>${esc(r.summary)}</pre>` : ""}
        ${bestId ? `<button class="act" data-pick="${esc(bestId)}">Show best attempt ${esc(r.best.name)}</button>` : ""}
        ${r.transcript ? `<button class="act" data-transcript="${file(p.id, r.transcript)}">Show transcript</button><div class="transcript"></div>` : ""}</div></details>`;
    }).join("") + `</div>`;
}

async function showTranscript(btn) {
  const box = btn.nextElementSibling;
  if (box.innerHTML) { box.innerHTML = ""; btn.textContent = "Show transcript"; return; }
  const text = await (await fetch(btn.dataset.transcript)).text();
  box.innerHTML = text.split("\n").filter(Boolean).map(line => {
    let m; try { m = JSON.parse(line); } catch { return ""; }
    const parts = Array.isArray(m.content) ? m.content.map(c => c.type === "text" ? c.text : "[picture]").join("\n") : (m.content || "");
    const calls = (m.tool_calls || []).map(c => `${c.function.name}(${(c.function.arguments || "").slice(0, 600)})`).join("\n");
    const body = [parts, calls].filter(Boolean).join("\n");
    return `<div class="msg ${esc(m.role)}"><b>${esc(m.role)}</b>${esc(body.length > 6000 ? body.slice(0, 6000) + "\n[cut]" : body)}</div>`;
  }).join("");
  btn.textContent = "Hide transcript";
}

// -- 3D: reading a GLB and placing the camera. Nothing here touches the page: tests/test_viewer.py runs it under node
// against render.py, which it has to agree with.
const COMPONENT = {5120: ["getInt8", 1, 127], 5121: ["getUint8", 1, 255], 5122: ["getInt16", 2, 32767], 5123: ["getUint16", 2, 65535],
  5125: ["getUint32", 4, 1], 5126: ["getFloat32", 4, 1]};
const WIDTH = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16};
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// 4x4 matrices are column-major, as glTF and WebGL have them
function mul4(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function nodeMatrix(n) {
  if (n.matrix) return n.matrix;
  const [x, y, z, w] = n.rotation || [0, 0, 0, 1], [sx, sy, sz] = n.scale || [1, 1, 1], [tx, ty, tz] = n.translation || [0, 0, 0];
  return [(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0, tx, ty, tz, 1];
}
function accessor(g, bin, index, Out = Float32Array) {
  const a = g.accessors[index], w = WIDTH[a.type], [get, size, max] = COMPONENT[a.componentType], out = new Out(a.count * w);
  if (a.bufferView == null) return out;
  const v = g.bufferViews[a.bufferView], start = (v.byteOffset || 0) + (a.byteOffset || 0), stride = v.byteStride || size * w;
  const k = a.normalized && max > 1 ? 1 / max : 1;
  for (let i = 0; i < a.count; i++) for (let c = 0; c < w; c++) out[i * w + c] = bin[get](start + i * stride + c * size, true) * k;
  return out;
}
function join(chunks, Type = Float32Array) {
  const out = new Type(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}
// Every triangle primitive of the default scene, in world space times `scale`, with the albedo render.load gives it:
// `colour` is "vertex" (COLOR_0, white without one), "atlas" (the material's texture times its factor, the factor
// alone without a texture) or "multiply" (COLOR_0 times the texture). A group is a run of indices drawn with one
// texture and tint.
function readModel(buffer, scale, colour) {
  const head = new DataView(buffer);
  if (buffer.byteLength < 20 || head.getUint32(0, true) !== 0x46546C67) throw new Error("not a GLB");
  let g = null, bin = new DataView(new ArrayBuffer(0));
  for (let at = 12; at + 8 <= buffer.byteLength;) {
    const size = head.getUint32(at, true), type = head.getUint32(at + 4, true);
    if (type === 0x4E4F534A) g = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, at + 8, size)));
    else if (type === 0x004E4942) bin = new DataView(buffer, at + 8, size);
    at += 8 + size;
  }
  if (!g) throw new Error("no JSON chunk in the GLB");
  const nodes = g.nodes || [], roots = ((g.scenes || [{nodes: nodes.map((_, i) => i)}])[g.scene || 0] || {}).nodes || [];
  const P = [], N = [], C = [], T = [], I = [], groups = [], images = [], slots = new Map();
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  let vertices = 0, indices = 0;
  const imageOf = texture => {
    if (!slots.has(texture)) {
      const image = (g.images || [])[((g.textures || [])[texture] || {}).source], view = image && (g.bufferViews || [])[image.bufferView];
      slots.set(texture, view ? images.push({type: image.mimeType || "image/png",
        bytes: new Uint8Array(buffer, bin.byteOffset + (view.byteOffset || 0), view.byteLength)}) - 1 : -1);
    }
    return slots.get(texture);
  };
  const walk = (index, parent) => {
    const node = nodes[index], m = mul4(parent, nodeMatrix(node));
    // normals go through the inverse transpose: the cross products of the matrix's columns, over its determinant
    const X = [m[0], m[1], m[2]], Y = [m[4], m[5], m[6]], Z = [m[8], m[9], m[10]], det = dot(X, cross(Y, Z));
    const nm = Math.abs(det) > 1e-12 ? [cross(Y, Z), cross(Z, X), cross(X, Y)].map(v => v.map(x => x / det)) : [X, Y, Z];
    for (const prim of node.mesh == null ? [] : g.meshes[node.mesh].primitives || []) {
      const a = prim.attributes || {};
      if ((prim.mode ?? 4) !== 4 || a.POSITION == null) continue;
      const pos = accessor(g, bin, a.POSITION), n = pos.length / 3;
      const idx = prim.indices == null ? Uint32Array.from({length: n}, (_, i) => i) : accessor(g, bin, prim.indices, Uint32Array);
      const count = idx.length - idx.length % 3;
      if (!count) continue;
      const from = a.NORMAL == null ? null : accessor(g, bin, a.NORMAL), p = new Float32Array(n * 3), nrm = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
        for (let r = 0; r < 3; r++) {
          const v = (m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]) * scale;
          p[i * 3 + r] = v;
          if (v < min[r]) min[r] = v;
          if (v > max[r]) max[r] = v;
        }
        if (!from) continue;            // no normals: left at zero, and the shader takes the face's own
        const v = [0, 1, 2].map(r => nm[0][r] * from[i * 3] + nm[1][r] * from[i * 3 + 1] + nm[2][r] * from[i * 3 + 2]);
        const len = Math.hypot(...v) || 1;
        for (let r = 0; r < 3; r++) nrm[i * 3 + r] = v[r] / len;
      }
      const pbr = ((g.materials || [])[prim.material] || {}).pbrMetallicRoughness || {};
      const tex = colour !== "vertex" && pbr.baseColorTexture && a.TEXCOORD_0 != null ? imageOf(pbr.baseColorTexture.index) : -1;
      const col = new Float32Array(n * 3).fill(1);
      if (a.COLOR_0 != null && (tex >= 0 ? colour === "multiply" : colour === "vertex")) {
        const c = accessor(g, bin, a.COLOR_0), w = WIDTH[g.accessors[a.COLOR_0].type];
        for (let i = 0; i < n; i++) for (let r = 0; r < 3; r++) col[i * 3 + r] = c[i * w + r];
      }
      const tint = colour === "vertex" ? [1, 1, 1] : (pbr.baseColorFactor || [1, 1, 1]).slice(0, 3), key = tex + "/" + tint.join();
      const last = groups[groups.length - 1];
      if (last && last.key === key) last.count += count;
      else groups.push({key, tex, tint, start: indices, count});
      P.push(p); N.push(nrm); C.push(col); T.push(tex >= 0 ? accessor(g, bin, a.TEXCOORD_0).slice(0, n * 2) : new Float32Array(n * 2));
      I.push(idx.subarray(0, count).map(i => i + vertices));
      vertices += n; indices += count;
    }
    for (const child of node.children || []) walk(child, m);
  };
  for (const root of roots) walk(root, IDENTITY);
  return {pos: join(P), nrm: join(N), col: join(C), uv: join(T), idx: join(I, Uint32Array), groups, images, min, max, triangles: indices / 3};
}
// A stage's camera turns about `target` at the distance a shot stood at: yaw about +Y, pitch up from the ground.
// Looking straight down or up, the shot's own up says which way the picture's top points.
function cameraOf(shot) {
  const f = shot.forward, up = shot.up, steep = Math.abs(f[1]) > 0.999, s = f[1] < 0 ? 1 : -1;
  return {target: shot.target.slice(), home: shot.target.slice(), dist: Math.hypot(...shot.eye.map((e, i) => e - shot.target[i])),
    ortho: shot.ortho > 0, pitch: Math.asin(Math.max(-1, Math.min(1, -f[1]))),
    yaw: steep ? Math.atan2(-s * up[0], -s * up[2]) : Math.atan2(-f[0], -f[2])};
}
function basisOf(cam) {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const f = [-cp * sy, -sp, -cp * cy], r = [cy, 0, -sy];
  return {f, r, u: cross(r, f)};
}
// How far the corners of some boxes reach from the camera's home: in any direction (`far`), and level with the
// ground (`wide`). Whichever way the camera is turned, nothing is further up or down the picture than `far` or
// further across it than `wide`.
function reachOf(cam, corners) {
  const size = {wide: 1e-6, far: 1e-6};
  for (const c of corners) {
    const flat = Math.hypot(c[0] - cam.home[0], c[2] - cam.home[2]);
    size.wide = Math.max(size.wide, flat); size.far = Math.max(size.far, Math.hypot(c[1] - cam.home[1], flat));
  }
  return size;
}
// View times projection for a picture `aspect` wide, with the lens that shows the whole of `size` (reachOf) from
// every side, so a model keeps its size as it is turned; only `zoom` scales it. `unit` is the height of the picture
// at the target, for moving the target by a drag.
function matrices(cam, size, aspect, zoom) {
  const {f, r, u} = basisOf(cam), away = Math.hypot(...cam.target.map((t, i) => t - cam.home[i])), radius = size.far;
  const d = cam.ortho ? Math.max(cam.dist, 2 * (radius + away)) : cam.dist, eye = cam.target.map((t, i) => t - f[i] * d);
  const half = Math.max(size.far, size.wide / aspect) / (cam.ortho ? 1 : Math.max(d - radius, 0.2 * d)) * 1.04 * zoom;
  const reach = 1.5 * (radius + away);
  const near = Math.max(d - reach, d * 0.02), far = d + reach, x = 1 / (half * aspect), y = 1 / half;
  const view = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, eye), -dot(u, eye), dot(f, eye), 1];
  const lens = cam.ortho ? [x, 0, 0, 0, 0, y, 0, 0, 0, 0, -2 / (far - near), 0, 0, 0, -(far + near) / (far - near), 1]
    : [x, 0, 0, 0, 0, y, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
  return {vp: mul4(lens, view), f, r, u, unit: 2 * half * (cam.ortho ? 1 : d)};
}

// -- 3D: the stage. An attempt's GLB drawn as render.py draws it: the same albedo, the index's lights (render.shade's),
// back faces culled, a normal that faces away turned round. One canvas lies over the stage and each model is drawn
// into the rectangle of its cell, all from one camera, so models side by side turn together.
const VERTEX = `#version 300 es
in vec3 p; in vec3 n; in vec3 c; in vec2 t; uniform mat4 vp; out vec3 vw; out vec3 vn; out vec3 vc; out vec2 vt;
void main() { vw = p; vn = n; vc = c; vt = t; gl_Position = vp * vec4(p, 1.0); }`;
const fragment = lights => `#version 300 es
precision highp float;
in vec3 vw; in vec3 vn; in vec3 vc; in vec2 vt; out vec4 o;
uniform vec3 fwd, ambient, tint, dir[${lights}], lit[${lights}]; uniform sampler2D tex; uniform bool textured;
void main() {
  vec3 nn = dot(vn, vn) > 1e-8 ? normalize(vn) : normalize(cross(dFdx(vw), dFdy(vw)));
  if (dot(nn, fwd) > 0.0) nn = -nn;
  vec3 light = ambient;
  for (int i = 0; i < ${lights}; i++) light += max(-dot(nn, dir[i]), 0.0) * lit[i];
  vec3 lin = clamp(vc * tint * (textured ? texture(tex, vt).rgb : vec3(1.0)) * light, 0.0, 1.0);
  o = vec4(mix(lin * 12.92, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin)), 1.0);
}`;
const FREE = {target: [0, 0, 0], home: [0, 0, 0], dist: 3, ortho: false, pitch: 0.35, yaw: -2.5};     // a project with no shots

class Stage {
  constructor() {
    this.models = new Map(); this.items = []; this.shots = []; this.cam = null; this.zoom = 1; this.view = null; this.key = null;
    this.canvas = Object.assign(document.createElement("canvas"), {className: "gl"});
    const gl = this.gl = this.canvas.getContext("webgl2", {antialias: true});
    if (!gl) return;
    const program = gl.createProgram(), lights = data.light.lights, places = {};
    for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, fragment(lights.length)]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source); gl.compileShader(shader); gl.attachShader(program, shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) console.error("mesh-jig view:", gl.getShaderInfoLog(shader));
    }
    ["p", "n", "c", "t"].forEach((name, i) => gl.bindAttribLocation(program, i, name));
    gl.linkProgram(program); gl.useProgram(program);
    this.at = name => places[name] ??= gl.getUniformLocation(program, name);
    gl.uniform3fv(this.at("ambient"), data.light.ambient);
    gl.uniform3fv(this.at("dir"), lights.flatMap(l => l.direction));
    gl.uniform3fv(this.at("lit"), lights.flatMap(l => l.colour));
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.enable(gl.SCISSOR_TEST);
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }
  // The page was drawn again: the stage goes into `host` with these models, [{url, cell, scale, colour}]. The camera
  // stays where it was unless `key` (the reference) is another one.
  show(host, items, shots, key) {
    this.host = host; this.items = items; this.shots = shots;
    host.prepend(this.canvas);
    if (!this.gl) {
      for (const item of items) item.cell.dataset.say = "This browser has no WebGL 2, so the model cannot be drawn here.";
      return;
    }
    const wanted = new Set(items.map(item => item.url));
    for (const [url, model] of this.models) if (!wanted.has(url)) { this.drop(model); this.models.delete(url); }
    for (const item of items) if (!this.models.has(item.url)) this.load(item);
    if (this.key !== key || !this.cam) { this.key = key; this.snap((shots.find(s => s.name === "hero") || shots[0] || {}).name); }
    else { this.mark(); this.draw(); }
  }
  async load(item) {
    const model = {};
    this.models.set(item.url, model);
    try {
      const got = await fetch(item.url);
      if (!got.ok) throw new Error("it is not on disk any more");
      const read = readModel(await got.arrayBuffer(), item.scale, item.colour);
      const pictures = await Promise.all(read.images.map(i => createImageBitmap(new Blob([i.bytes], {type: i.type}))));
      if (this.models.get(item.url) !== model) return;          // another attempt was picked while this one loaded
      Object.assign(model, this.upload(read, pictures), {ready: true});
    } catch (e) {
      model.error = String(e.message || e);
    }
    this.draw();
  }
  upload(read, pictures) {
    const gl = this.gl, vao = gl.createVertexArray(), buffers = [];
    gl.bindVertexArray(vao);
    [[read.pos, 3], [read.nrm, 3], [read.col, 3], [read.uv, 2]].forEach(([values, size], i) => {
      buffers.push(gl.createBuffer());
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers[i]); gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(i); gl.vertexAttribPointer(i, size, gl.FLOAT, false, 0, 0);
    });
    buffers.push(gl.createBuffer());
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers[4]); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, read.idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    const textures = pictures.map(picture => {        // sRGB in, linear out, and the nearest texel as render.shade takes it
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE, picture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      return texture;
    });
    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(i => [0, 1, 2].map(r => (i >> r & 1 ? read.max : read.min)[r]));
    return {vao, buffers, textures, groups: read.groups, corners, triangles: read.triangles};
  }
  drop(model) {
    if (!model.ready) return;
    this.gl.deleteVertexArray(model.vao);
    model.buffers.forEach(b => this.gl.deleteBuffer(b)); model.textures.forEach(t => this.gl.deleteTexture(t));
  }
  clear() {                                              // the files may have been written again
    for (const model of this.models.values()) this.drop(model);
    this.models.clear();
  }
  snap(name) {
    const shot = this.shots.find(s => s.name === name);
    this.cam = shot ? cameraOf(shot) : {...FREE}; this.zoom = 1; this.view = shot ? name : null;
    this.mark(); this.draw();
  }
  turn(dx, dy) {
    this.cam.yaw -= dx * 0.01;
    this.cam.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.cam.pitch + dy * 0.01));
    this.view = null; this.mark(); this.draw();
  }
  move(dx, dy) {
    const {r, u} = basisOf(this.cam), k = this.unit || 0;
    this.cam.target = this.cam.target.map((t, i) => t + (u[i] * dy - r[i] * dx) * k);
    this.view = null; this.mark(); this.draw();
  }
  scale(by) { this.zoom = Math.max(0.03, Math.min(4, this.zoom * by)); this.draw(); }
  // which named view the camera stands at, on the buttons and on the reference picture of that view
  mark() {
    if (!this.host) return;
    this.host.querySelectorAll("[data-shot]").forEach(b => b.setAttribute("aria-pressed", b.dataset.shot === this.view));
    this.host.querySelectorAll(".inset").forEach(el => el.hidden = el.dataset.view !== this.view);
  }
  draw() {
    this.waiting ??= requestAnimationFrame(() => { this.waiting = null; this.paint(); });
  }
  paint() {
    const gl = this.gl, canvas = this.canvas;
    if (!gl || !canvas.isConnected || !this.cam) return;
    const ratio = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    const width = Math.round(box.width * ratio), height = Math.round(box.height * ratio);
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    gl.viewport(0, 0, width, height); gl.scissor(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const size = reachOf(this.cam, this.items.map(item => this.models.get(item.url)).filter(m => m && m.ready).flatMap(m => m.corners));
    for (const item of this.items) {
      const model = this.models.get(item.url), said = item.cell.closest(".slot, .stage").querySelector(".tris");
      item.cell.dataset.say = !model ? "" : model.error ? "The model could not be drawn: " + model.error : model.ready ? "" : "reading the model";
      if (said) said.textContent = model && model.ready ? model.triangles.toLocaleString("en") + " triangles" : "";
      if (!model || !model.ready) continue;
      const cell = item.cell.getBoundingClientRect(), w = Math.round(cell.width * ratio), h = Math.round(cell.height * ratio);
      if (w < 2 || h < 2) continue;
      const x = Math.round((cell.left - box.left) * ratio), y = Math.round((box.bottom - cell.bottom) * ratio);
      gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
      gl.clearColor(...data.light.background, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const m = matrices(this.cam, size, w / h, this.zoom);
      this.unit = m.unit / cell.height;
      gl.uniformMatrix4fv(this.at("vp"), false, m.vp); gl.uniform3fv(this.at("fwd"), m.f);
      gl.bindVertexArray(model.vao);
      for (const group of model.groups) {
        gl.uniform3fv(this.at("tint"), group.tint); gl.uniform1i(this.at("textured"), group.tex >= 0 ? 1 : 0);
        if (group.tex >= 0) gl.bindTexture(gl.TEXTURE_2D, model.textures[group.tex]);
        gl.drawElements(gl.TRIANGLES, group.count, gl.UNSIGNED_INT, group.start * 4);
      }
    }
  }
}
const stages = {};
const stage = name => stages[name] ??= new Stage();
document.addEventListener("visibilitychange", () => Object.values(stages).forEach(s => s.draw()));    // a hidden tab is not painted
const stageBar = shots => `${shots.map(s => `<button data-shot="${esc(s.name)}" aria-pressed="false" title="The camera the ${esc(s.name)} view is measured from">${esc(s.name)}</button>`).join("")}
  <button class="gap" data-zoom="1.25" title="Zoom out" aria-label="Zoom out">&minus;</button><button data-zoom="0.8" title="Zoom in" aria-label="Zoom in">+</button>
  <small class="how">Drag to turn, Shift and drag to move, Ctrl and scroll to zoom</small>`;
// the built model, to take into another program: shown in its folder where the page can ask for that, and to download
function take(p, a) {
  if (!a.glb) return "";
  const name = esc(a.glb.split("/").pop());
  return `<p class="take">${here() ? `<button class="act" data-reveal="${reveal(p.id, a.glb)}" title="Open the folder ${name} is in, with the file selected">Show ${name} in its folder</button>` : ""}
    <a download href="${file(p.id, a.glb)}">${here() ? "or download it" : `Download ${name}`}</a><span class="dim said" role="status"></span></p>`;
}
async function showFolder(btn) {
  let ok = false;
  try { ok = (await fetch(btn.dataset.reveal, {method: "POST"})).ok; } catch {}
  btn.parentNode.querySelector(".said").textContent = ok ? "Opened in the file manager."
    : "The folder could not be opened: mesh-jig view is not answering, or the file is gone.";
}
const modelOf = (p, a, cell) => ({url: file(p.id, a.glb), cell, scale: p.stage.scale, colour: p.stage.colour});

// the best attempt of each run in 3D, side by side at one scale and turning together
function lineup(ref) {
  const runs = bestRuns(ref).filter(r => r.a && r.a.glb);
  if (!runs.length) return `<p class="hint">No run here has its best attempt's GLB on disk. <code>mesh-jig eval</code> on an attempt's script builds it again.</p>`;
  const folds = sectioned(ref) ? `<div class="tools folds">${sections(ref).map(sectionHead).join("")}</div>` : "";
  return `${folds}<div class="stage lineup" data-stage="lineup"><div class="bar">${stageBar(runs[0].p.stage.shots)}</div><div class="slots">${runs.map(({p, a}) =>
    `<div class="slot" tabindex="0" data-pick="${esc(p.id + "/" + a.name)}"><div class="cell"></div><div class="cap">
      <span class="score">${f(a.numeric.numeric_score)}</span>${commit(a)}${badge(p)}<b>${runName(p)}</b><small>${esc(p.driver && p.driver.model ? p.label + " / " : "")}${esc(a.name)}</small>
      <small class="tris"></small></div></div>`).join("")}</div></div>`;
}
function showLineup(ref) {
  const host = document.querySelector('[data-stage="lineup"]');
  if (!host) return;
  const runs = bestRuns(ref).filter(r => r.a && r.a.glb), cells = host.querySelectorAll(".cell");
  stage("lineup").show(host, runs.map(({p, a}, i) => modelOf(p, a, cells[i])), runs[0].p.stage.shots, ref.key);
}

// the picked attempt: per view the reference, the build and the outline diff, three tiles of one size
function drawInspector() {
  const ref = current(), r = ref && rows(ref).find(x => x.id === state.pick), box = $("inspect");
  if (!r) { box.innerHTML = ref && rows(ref).length ? `<p class="hint">Pick a run or an attempt to see it beside the reference.</p>` : ""; return; }
  const a = r.a, p = r.p, n = a.numeric;
  const row = (name, key, value, d, sub) => value == null ? "" : `<span class="name ${sub ? "sub" : ""}">${name}</span>
    ${key === "mae_mm" ? `<span class="mm">${f(value, d)} mm</span>` : meter(value, d)}<p class="what">${MEANING[key]}</p>`;
  const measures = row("outline overlap", "ortho_iou", n.ortho_iou, 3) + row("resemblance", "resemblance", n.resemblance, 3)
    + row("colour zones", "zone", n.zone, 3, true) + row("markings", "structure", a.structure, 3, true)
    + row("outline error", "mae_mm", n.mae_mm, 2);
  const counted = a.counted || [], others = Object.keys(a.views).filter(v => !counted.includes(v));
  const weighted = a.weights && new Set(Object.values(a.weights)).size > 1;
  const scope = !measures ? "" : `<p class="what">These are ${weighted ? "weighted means" : "means"} over the ${esc(words(counted))} view${counted.length === 1 ? "" : "s"}.${others.length
      ? ` The ${esc(words(others))} view${others.length === 1 ? " is" : "s are"} compared to look at and ${others.length === 1 ? "does" : "do"} not count.` : ""}${n.mae_mm == null
      ? " An outline error in millimetres is measured too when the project gives its model's length." : ""}</p>`;
  // a part check's gate carries its whole record as its message: those are shown as the part checks below
  const gates = (a.gates || []).filter(g => !g.message.startsWith("{") && (!g.passed || /_(min|max)$/.test(g.name))).map(g =>
    `<p class="what ${g.passed ? "ok" : "fail"}">Contract check ${esc(g.name)}: ${esc(g.message)}, ${g.passed ? "met" : "not met"}.</p>`).join("");
  const palette = a.colours && Object.keys(p.palette || {}).length ? `<p class="palette"><span>The project's palette</span>${Object.entries(p.palette).map(([name, hex]) =>
      `<span><i class="sw" style="background:${colour(hex)}"></i>${esc(name)}</span>`).join("")}</p>` : "";
  const links = [a.script && `<a target="_blank" href="${file(p.id, a.script)}">model script</a>`,
    a.eval && `<a target="_blank" href="${file(p.id, a.eval)}">eval.json</a>`].filter(Boolean).join(", ");
  const shot = (href, src, alt, cls = "") => `<a target="_blank" title="Open the picture" href="${href}"><img class="tile ${cls}" loading="lazy" src="${src}" alt="${alt}"></a>`;
  const views = Object.entries(a.views).map(([v, w]) => {
    const rv = ref.views[v];
    const fine = [w.missing != null && w.extra != null ? `the build misses ${pct(w.missing)} of the reference's body, and ${pct(w.extra)} of the build lies outside it` : "",
      w.aspect != null && w.reference_aspect != null ? `height to width ${f(w.aspect, 2)}, the reference ${f(w.reference_aspect, 2)}` : ""].filter(Boolean).join("; ");
    return `<section class="view"><header><h3>${esc(v)}</h3>${counted.includes(v) ? "" : `<span class="tag">not in the score</span>`}
      ${w.compare ? `<a target="_blank" href="${file(p.id, w.compare)}">side-by-side picture</a>` : ""}</header>
      <div class="stats">${meter(w.iou, 2, "outline overlap")}${meter(w.zone, 2, "colour zones")}${meter(w.structure, 2, "markings")}</div>
      ${fine ? `<p class="fine">${fine.charAt(0).toUpperCase() + fine.slice(1)}.</p>` : ""}<div class="three">
      ${rv ? shot(file(rv.project, rv.path), tile(rv.project, rv.path, 420), `reference ${esc(v)}`) : `<span class="none">no reference for this view</span>`}
      ${w.render ? shot(file(p.id, w.render), tile(p.id, w.render, 420), `build ${esc(v)}`) : `<span class="none">no render on disk: run mesh-jig eval again</span>`}
      ${w.overlay ? shot(file(p.id, w.overlay), file(p.id, w.overlay), `outline diff ${esc(v)}`, "diff") : `<span class="none">no outline diff</span>`}</div>
      <div class="stations" data-view="${esc(v)}"></div><div class="reading" data-view="${esc(v)}"></div></section>`;
  }).join("");
  const d = p.driver, by = !d ? "" : `<div class="by">${badge(p)}${d.model ? `<b>${runName(p)}</b>` : ""}<span class="dim">${esc(
    d.kind === "agent" ? "mesh-jig agent" + (d.detail ? ", calling " + d.detail : "") : d.detail ? "version " + d.detail : "")}</span></div>`;
  // in 3D, with the reference picture of whichever named view the camera stands at
  const insets = Object.entries(ref.views).map(([v, rv]) => `<a class="inset" data-view="${esc(v)}" hidden target="_blank" title="Open the picture"
    href="${file(rv.project, rv.path)}"><img loading="lazy" src="${tile(rv.project, rv.path, 240)}" alt="reference ${esc(v)}">reference</a>`).join("");
  const solid = a.glb ? `<div class="stage" data-stage="inspect"><div class="cell">${insets}</div><div class="bar">${stageBar(p.stage.shots)}</div>
      <p class="what">The build itself, drawn here with the renderer's lights. A named view is the camera that view is measured from.
      <span class="tris"></span></p></div>`
    : a.ok ? `<p class="what">Not drawn in 3D: this attempt's GLB is not on disk. <code>mesh-jig eval</code> on its script builds it again.</p>` : "";
  box.innerHTML = `<h2>${esc(p.label)} <span>/</span> ${esc(a.name)}</h2>${by}
    <div class="dim where" title="${esc(p.path)}">${esc(a.started)}${a.started ? ", " : ""}${a.wall_s != null ? `evaluated in ${Math.round(a.wall_s)} s, ` : ""}${esc(p.path)}</div>
    ${a.note ? `<p class="note">${esc(a.note)}</p>` : ""}${solid}${take(p, a)}
    ${n.numeric_score != null ? `<div class="headline"><b>${f(n.numeric_score)}</b><span>score</span>${commit(a)}</div><p class="what">${MEANING.numeric_score}</p>` : ""}
    ${measures ? `<div class="measures">${measures}</div>` : ""}${scope}${gates}<div id="checks"></div>
    ${a.pairwise ? `<p class="note">Judged against ${esc(a.pairwise.against)}: win rate ${f(a.pairwise.win_rate, 2)}, ${a.pairwise.decisive ? "decisive" : "not decisive"}.</p>` : ""}
    ${a.ok ? "" : `<p class="fail">Failed at ${esc(a.stage || "build")}.</p><pre>${esc(a.error)}</pre>`}
    ${links ? `<p class="dim">Open the ${links}</p>` : ""}${palette}
    ${views ? `<div class="three caps"><span>reference</span><span>build</span><span>outline diff<span class="legend">
      <span><i class="sw" style="background:#969696"></i>both</span><span><i class="sw" style="background:#dc2828"></i>add</span>
      <span><i class="sw" style="background:#285adc"></i>remove</span></span></span></div>${views}` : ""}`;
  box.scrollTop = 0;
  if (a.glb) stage("inspect").show(box.querySelector(".stage"), [modelOf(p, a, box.querySelector(".cell"))], p.stage.shots, ref.key);
  if (a.eval) showRecord(p, a, r.id);
}

// one line per part check of the project's review contract: what was checked, what it read, whether it was met
function isObservation(c) {
  return c.observation || (c.view != null && c.iou != null && c.min_iou == null && c.min_zone == null);
}
function checkRow(c) {
  const [kind, ...rest] = String(c.name).split(":");
  let name = c.name, said = c.error || "";
  if (c.view && c.iou != null) said = `in its marked region of the ${c.view} view: outline overlap ${f(c.iou, 2)}${c.zone != null ? ", colour zones " + f(c.zone, 2) : ""}`;
  else if (kind === "landmark") { name = rest.join(" "); said = "the part lies inside the box the contract gives it"; }
  else if (kind === "visibility") { name = rest[0]; said = `${pct(c.visible_fraction || 0)} of it shows in the ${rest[1]} view`; }
  else if (kind === "clearance") {
    const gaps = (c.samples || []).map(x => x.gap_m).filter(g => g != null);
    name = rest.join(" under ");
    said = gaps.length ? `gap between the two ${Math.round(Math.min(...gaps) * 1000)} to ${Math.round(Math.max(...gaps) * 1000)} mm` : "no gap could be measured";
  }
  return `<span class="name">${esc(name)}</span><span class="said">${esc(said)}</span><span class="${isObservation(c) ? "" : c.passed ? "ok" : "fail"}">${isObservation(c) ? "observation" : c.passed ? "met" : "not met"}</span>`;
}

// the rest of what was measured is in the attempt's eval.json, not in the index: the part checks, the outline at
// each named station, and how the surface reads in palette colours
async function showRecord(p, a, id) {
  let rec;
  try { rec = await (await fetch(file(p.id, a.eval))).json(); } catch { return; }
  if (state.pick !== id) return;
  const signed = x => (x > 0 ? "+" : "") + Math.round(x);
  const checks = (rec.review || {}).checks || [], outline = rec.profile_diff || {};
  const observed = checks.filter(isObservation), gates = checks.filter(c => !isObservation(c));
  const failed = gates.filter(c => !c.passed);
  $("checks").innerHTML = (outline.height_error_mm != null ? `<p class="what">Height against the reference: ${signed(outline.height_error_mm)} mm.</p>` : "")
    + (checks.length ? `<h3>Part checks</h3><p class="what">${gates.length - failed.length} of ${gates.length} acceptance gates met; ${observed.length} observations without acceptance limits. Region measurements can also contribute to the configured score.</p>
      <div class="checks">${failed.concat(gates.filter(c => c.passed), observed).map(checkRow).join("")}</div>` : "");
  document.querySelectorAll("#inspect .stations").forEach(box => {
    const rows = ((outline.views || {})[box.dataset.view] || {}).rows || [];
    if (rows.length && rows[0].name) box.innerHTML = `Width at each station against the reference, in mm (+ is wider)<div class="chips">${rows.map(x =>
      `<span>${esc(x.name)}<b>${x.width_mm == null ? "missing" : signed(x.width_mm)}</b></span>`).join("")}</div>`;
  });
  if (a.colours) showColours(p, rec);
}

// per view, what share of the surface reads as each palette colour on the reference and on the build
function showColours(p, rec) {
  const mix = (label, shares) => {
    const list = Object.entries(shares || {}).sort((x, y) => y[1] - x[1]);
    return `<div class="mix"><span>${label}</span><span class="stack">${list.map(([name, share]) =>
      `<i title="${esc(name)} ${pct(share)}" style="flex:${Number(share) || 0};background:${colour((p.palette || {})[name])}"></i>`).join("")}</span>
      <span>${esc(list.slice(0, 3).map(([name, share]) => `${name} ${pct(share)}`).join(", "))}</span></div>`;
  };
  document.querySelectorAll("#inspect .reading").forEach(box => {
    const z = (rec.zone_reading || {})[box.dataset.view];
    if (z) box.innerHTML = `How the surface reads, as shares of the palette${mix("reference", z.reference)}${mix("build", z.yours)}`;
  });
}

function markPick() {
  document.querySelectorAll("tr[data-pick], .slot").forEach(el => el.classList.toggle("on", el.dataset.pick === state.pick));
}
function pick(id, reveal = true) {
  state.pick = id; writeHash(); markPick(); drawInspector();
  if (reveal && !wide.matches) $("inspect").scrollIntoView({block: "start"});
}

function drawBoard() {
  const ref = current();
  if (!ref) {
    $("board").innerHTML = `<h1>Nothing to show yet</h1><p class="hint" style="padding-left:0">No project with reference views was found under the directories below.
      A project is a directory with a jig.json and its reference pictures; <code>mesh-jig init</code> starts one.</p>`;
    return;
  }
  const all = rows(ref), best = bestOf(ref), runs = shown(ref), solid = ref.projects.some(p => p.attempts.some(a => a.glb && a.name === p.best));
  const tabs = [["runs", "Best of each run", runs.length], ["attempts", "Attempts", runs.reduce((s, p) => s + p.attempts.length, 0)],
    ["3d", "In 3D", solid ? bestRuns(ref).filter(r => r.a && r.a.glb).length : null], ["agent", "Agent runs", runs.reduce((s, p) => s + p.runs.length, 0) || null]]
    .filter(t => t[2] != null);
  if (!tabs.some(t => t[0] === state.tab)) state.tab = "runs";
  $("board").innerHTML = `<h1>${esc(ref.name)}</h1>
    <p class="facts"><span>${count(Object.keys(ref.views).length, "view")}</span><span>${count(ref.projects.length, "run")}</span>
      <span>${count(all.length, "attempt")}</span>${best ? `<span>best <b>${f(score(best))}</b> ${esc(best.p.label)} / ${esc(best.a.name)}</span>` : ""}</p>
    ${drivers(ref)}<div class="tabs" role="tablist">${tabs.map(([k, label, n]) =>
      `<button role="tab" data-tab="${k}" aria-selected="${state.tab === k}">${label}<i>${n}</i></button>`).join("")}</div>
    <div class="sheet">${state.tab === "attempts" ? attempts(ref) : state.tab === "agent" ? agentRuns(ref) : state.tab === "3d" ? lineup(ref) : grid(ref)}</div>`;
  if (state.tab === "3d") showLineup(ref);
  markPick();
}

function draw() {
  if (!current()) { state.ref = data.references[0]?.key || null; state.pick = null; }
  const ref = current();
  if (!ref || !ref.projects.some(p => kindOf(p) === state.driver)) state.driver = null;
  document.documentElement.style.setProperty("--stage", `rgb(${data.light.background.map(c => Math.round(c * 255)).join()})`);
  if (ref && !rows(ref).some(r => r.id === state.pick)) state.pick = bestOf(ref)?.id || rows(ref)[0]?.id || null;
  drawRefs(); drawBoard(); drawInspector(); writeHash();
  const skipped = leftOut(data.skipped);
  $("foot").textContent = STATIC ? "A saved copy of mesh-jig view: the results as they were when it was written."
    : `Projects under ${data.paths.join(" and ")}.`
    + (data.skip.length ? ` Not searched: any directory named ${data.skip.join(" or ")}.` : "") + (skipped ? ` Left out: ${skipped}.` : "");
  $("foot").title = STATIC ? "" : data.skipped.map(s => `${s.path} (${s.why})`).join("\n");      // every path, on hover
}

// The projects left out, for the footer: a count for each reason and the first few directory names. A checkout
// whose runs are kept as records only has dozens, and a full path for each buried the line above them.
const NAMED = 3;
function leftOut(skipped) {
  const names = new Map();
  for (const s of skipped) names.set(s.why, [...(names.get(s.why) || []), s.path.split(/[\\/]/).pop()]);
  return [...names].map(([why, of]) => `${of.length} (${why}): ${of.slice(0, NAMED).join(", ")}`
    + (of.length > NAMED ? ` and ${of.length - NAMED} more` : "")).join("; ");
}

// a model is turned by dragging its cell (moved with Shift held), and zoomed by scrolling with Ctrl held: a plain
// scroll over it still scrolls the page. A drag that ends on something clickable is not a click on it.
let drag = null, dragged = false;
const stageOf = el => stages[el.closest(".stage").dataset.stage];
document.addEventListener("pointerdown", e => {
  const cell = e.button === 0 && !e.target.closest("a, button") && e.target.closest(".cell");
  if (cell && stageOf(cell)) drag = {on: stageOf(cell), x: e.clientX, y: e.clientY, far: 0};
});
document.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY; drag.far += Math.abs(dx) + Math.abs(dy);
  if (drag.far > 3 && drag.on.cam) e.shiftKey ? drag.on.move(dx, dy) : drag.on.turn(dx, dy);
});
for (const type of ["pointerup", "pointercancel"]) document.addEventListener(type, () => {
  dragged = !!drag && drag.far > 3; drag = null;
  setTimeout(() => dragged = false);
});
document.addEventListener("wheel", e => {
  const cell = e.ctrlKey && e.target.closest(".cell");
  if (!cell || !stageOf(cell)) return;
  e.preventDefault(); stageOf(cell).scale(Math.exp(e.deltaY * 0.0015));
}, {passive: false});

document.addEventListener("click", e => {
  if (dragged || e.target.closest("a")) return;
  const el = e.target.closest("[data-ref],[data-tab],[data-measure],[data-order],[data-sort],[data-shot],[data-zoom],[data-driver],[data-every],[data-open],[data-group],[data-pick],[data-transcript],[data-reveal]");
  if (!el) return;
  const d = el.dataset;
  if (d.ref) { state.ref = d.ref; state.pick = null; draw(); }
  else if (d.group) toggleGroup(d.group);
  else if (d.every) { state.every = d.every === "all"; writeHash(); drawBoard(); }
  else if (d.open) { state.open.has(d.open) ? state.open.delete(d.open) : state.open.add(d.open); drawBoard(); }
  else if (d.shot) stageOf(el).snap(d.shot);
  else if (d.zoom) stageOf(el).scale(Number(d.zoom));
  else if (d.driver) { state.driver = state.driver === d.driver ? null : d.driver; writeHash(); drawBoard(); }
  else if (d.tab) { state.tab = d.tab; writeHash(); drawBoard(); }
  else if (d.measure) { state.measure = d.measure; writeHash(); drawBoard(); }
  else if (d.order) { state.order = d.order; drawBoard(); }
  else if (d.sort) {
    const text = d.sort === "run" || d.sort === "attempt";
    state.desc = state.sort === d.sort ? !state.desc : !text; state.sort = d.sort; drawBoard();
  }
  else if (d.pick) pick(d.pick);
  else if (d.transcript) showTranscript(el);
  else if (d.reveal) showFolder(el);
});
// a row is picked with Enter, and the arrows walk the list with the picked attempt following
document.addEventListener("keydown", e => {
  const row = e.target.closest ? e.target.closest("tr[data-pick], .slot") : null;
  if (!row || e.target.closest("button")) return;
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(row.dataset.pick); return; }
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  const list = [...row.parentNode.querySelectorAll("tr[data-pick], .slot")], next = list[list.indexOf(row) + (e.key === "ArrowDown" ? 1 : -1)];
  if (next) { e.preventDefault(); next.focus(); pick(next.dataset.pick, false); }
});

async function load() {
  try {
    data = await (await fetch(STATIC ? "index.json" : "/api/index")).json();
  } catch {
    $("foot").textContent = STATIC ? "The results could not be read: index.json is not beside this page."
      : "The results could not be read: mesh-jig view is not answering. Start it again, then reload.";
    return;
  }
  Object.values(stages).forEach(s => s.clear());
  draw();
}
readHash();
$("reload").onclick = load;
$("reload").hidden = STATIC;
load();
</script></body></html>
"""

__all__ = ["find_projects", "build_index", "reference_key", "attempt_record", "driver_of", "stage_of", "safe_file", "serve",
           "export", "reveal_command", "show_in_folder", "on_this_machine", "DEFAULT_PORT"]
