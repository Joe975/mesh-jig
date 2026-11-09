'mesh-jig agent --workflow rig: an API model writes the rig script for a frozen GLB, with no shell.'
from __future__ import annotations

import copy
import json
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from . import agent, agent_state, glb, harness, llm, mcp_server, rig, rig_brief, rig_look, rig_read, skin

SOURCE, CONTRACT, BRIEF, SCRIPT = "source.glb", "rig.json", "brief.md", "rig.py"
BUILT, FAILED, REPORT = "animated.glb", "failed.glb", "report.json"
TURNS_PER_BUILD = 14
TURNS_LEFT_NOTE = 3
SILENT_TURNS = 3              # replies in a row with no tool call that end a run
STUCK_AFTER = 3               # builds out of contract in a row before the builder is told to change approach
RETRIES = 2                   # a model call dropped by the connection is made again this many times
RETRY_WAIT_S = 5.0
KEEP_IMAGE_TURNS = 2
PICTURE_MAX_DIM = 1536
CONTEXT_TOKENS = 90_000       # compact above this: the conversation is re-sent every turn, and long prompts cost more
STRIP_VIEWS = ("left", "front")
LOOK_MAX = 8
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SYSTEM = """You rig and animate ONE finished 3D model by writing a Python rig script. The model is frozen: you add a
skeleton, skin weights and clips, and change no vertex. The harness builds your script (plain numpy, no Blender),
holds the result to the contract, and draws it. There is no shell: you have the tools below and nothing else.

Every path you pass is relative to the rig directory. The only file you may write is attempts/<name>/rig.py.

THE LOOP
1. Read the brief, THE MODEL (its bounds, where it stands, its slices) and the ruled pictures of the rest pose.
   Decide where each joint is from the slices and the grid: a joint's head goes where the part it turns meets the
   part it hangs from. Say your joint positions in a sentence before you write.
2. write_file attempts/a01/rig.py: the whole script, with every number you will tune (joint positions, blend
   widths, swing angles, timings) as a named constant at the top. Where the brief has A RECIPE, start from its
   code as it stands (its skeleton, its weights by region, its feet placed and legs solved) with your own numbers, and add
   what the contract and the brief ask for beyond it. Then build it.
3. Read what build returns, in this order: the verdict; every line starting with "!" (each is a fault no gate
   catches: fix all of them before polishing anything); the WEIGHTS picture (a patch in the wrong colour is a
   wrong weight); then each clip's film strip, the side row first. Say what each clip actually looks like, frame by
   frame, before you decide what to change. A passing verdict means the file is valid, not that the animation is good.
4. Start the next attempt with copy_attempt and change it with edit_file (small edits, at most four a call).
   An attempt that has been built in contract is never changed again.
5. Use every build: each one is a chance to fix what the last readings and pictures showed. When they are used,
   call done naming your best attempt, which need not be the last.

ORDER OF WORK: skeleton and weights first. A wrong weight ruins every clip, and the WEIGHTS picture and the STRAY
WEIGHTS lines show it before anything moves. Then the walk, then the run built from the walk's parts with its own
numbers, then the idle. Do not spend a build on a change you could have made together with another.

WHAT GOES WRONG, from the first rigs made here:
- t is in SECONDS, not a 0..1 phase: a 0.6 s clip written with sin(2*pi*t) does not loop. Use the phase
  p = 2*pi*t/L with L the clip's own length.
- A rotation's sense is fixed (see ROTATIONS): work one limb through by hand before trusting a sign, then check it
  in the side row of the strip. A lean or a knee that goes the wrong way is the commonest fault.
- A parent's rotation carries its children: tilting the hips swings both legs and drives the feet through the
  ground. Lean with the spine; move and turn the hips only through the recipe's `legs`, which solves the legs
  for whatever the hips do.
- A leg that swings from the hip lifts its foot off the ground or pushes it under: the lowest point of the model
  must stay on the rest pose's ground in a walk (the readings say where it is). Say where the foot is to be and
  solve the leg for it (the recipe's `legs`), rather than choosing angles and finding out where they put it.
- A stretch gate failed by a wide margin is failed by weights and pivots, not by motion. Shrinking every swing
  until it passes leaves the bad weights in and a timid clip on top of them: the run then looks like the walk.
  (Within a point of the gate with clean weights, easing the one joint the readings name is fair.) Each clip's `stretched at`
  lines say whose part stretches, which bone's weight pulls it, and where. When the two are neighbours at a
  joint, fix the joint: a longer blend band across it (see A BEND STRETCHES ITS OUTSIDE), its head moved to the
  middle of the limb or toward the side that opens, or that one joint bent less. When the puller is a limb and
  the part is a trunk bone's (an arm pulling "the part hips holds"), look at the place named. On the body: the
  limb's region reaches into it, so narrow the region there. Out on the limb: the region is too small and has
  left some of the limb to the trunk (STRAY WEIGHTS then lists a patch of a trunk bone out there), so widen it
  (a larger capsule radius). Either way leave the trunk's own blends alone.
- Look closely before you believe a joint: `look` with `aim` and `span` draws a knee, an elbow or a shoulder
  large. The film strips are small.
- An idle that only breathes by a millimetre reads as a frozen pose, and one that nods or turns the head by a
  hand's width reads as a gesture, not a stand. A few centimetres at the chest and head, the arms a little, the
  tail more. The feet do not move at all: shift the weight over them (the recipe's `idle`); a sway of the root
  drags them along. THE CLIPS BESIDE EACH OTHER shows how far each part goes.
- A model in many pieces: give every node the same tests on position, so neighbours move as one, and every node
  of one rigid thing (a sword and its scabbard) the same one bone by name. The readings list the nodes that
  touch at rest and part.
- The model's own left is X<0 and its right is X>0 (it faces +Y). Name bones to match: .L at X<0."""

REVIEWER = """You are an independent animation reviewer. You are shown film strips of a rigged 3D model: each strip
is one looping clip at evenly spaced times, left to right, with a row for each camera. A thin line marks the ground
the model stands on at rest. You did not make this and have not seen its script or any numbers.

Judge only what the pictures show against the brief. For each clip say in one or two sentences what the motion
actually reads as. Then give at most five prioritized faults, most damaging first. For each: the clip, the row and
the frames it shows in, the part, what is wrong (a limb not moving, both legs together, a knee or a lean the wrong
way, feet through the ground or the whole model floating, a patch of body dragged by a limb, something worn left
behind, a part torn or badly stretched, a clip that does not differ from another), and what would fix it in terms of
motion (which part should turn which way, by roughly how much, at which point of the cycle). Do not invent faults:
if a clip reads well, say so. Finish with what should be kept."""


def _fn(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S, N = {"type": "string"}, {"type": "number"}
TOOLS = [
    _fn("write_file", "Write attempts/<name>/rig.py (creates the directory). For a NEW script; change an existing one "
        "with edit_file.", {"path": S, "text": S}, ["path", "text"]),
    _fn("edit_file", "Change an attempt's rig.py by exact replacements, for an attempt not yet built in contract. "
        "`edits` is a list of {old, new} applied in order; each `old` must appear exactly once. If one fails, none is "
        "applied. At most four edits a call.",
        {"path": S, "edits": {"type": "array", "items": {"type": "object", "properties": {"old": S, "new": S},
                                                          "required": ["old", "new"]}}}, ["path", "edits"]),
    _fn("read_file", "Read an attempt's rig.py or report.json (at most 200 lines; start_line/end_line for more).",
        {"path": S, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _fn("copy_attempt", "Start the next attempt from an earlier one: copies its rig.py into a new attempts/<name>/.",
        {"source": S, "name": S}, ["source", "name"]),
    _fn("build", "Build an attempt's rig.py and hold it to the contract. Returns the verdict with every gate's number, "
        "what each bone holds, stray weights, what each clip does, and pictures: the weights, and a film strip of "
        "every clip from the side and the front. A build that fails its contract returns the same, for what it "
        "built, and does not count toward the builds.", {"script": S, "note": S}, ["script"]),
    _fn("look", f"More pictures of an attempt that has been built: one clip at the times you name (SECONDS within the "
        f"clip's length, at most {LOOK_MAX}), from the views you name ({', '.join(rig_look.VIEWS)}), with the skeleton "
        "drawn over it if asked. For a close-up of a joint give `aim` (the point to centre on, [x, y, z] in the "
        "script's axes) and `span` (metres across the picture, e.g. 0.5): a knee, an elbow or a shoulder cannot be "
        "judged at the size of the whole figure. Free; not a build.",
        {"attempt": S, "clip": S, "times": {"type": "array", "items": N}, "views": {"type": "array", "items": S},
         "skeleton": {"type": "boolean"}, "aim": {"type": "array", "items": N}, "span": N},
        ["attempt", "clip", "times"]),
    _fn("done", "Finish: name the attempt to deliver (built in contract), and say what is still wrong with it.",
        {"best": S, "summary": S}, ["best", "summary"]),
]


def problems(root: Path) -> list[str]:
    out = [f"no {name} in {root.name}/" for name in (SOURCE, CONTRACT, BRIEF) if not (root / name).is_file()]
    if not out:
        try:
            rig.contract(rig_brief.contract(root / CONTRACT))
        except (ValueError, json.JSONDecodeError) as e:
            out.append(f"{CONTRACT}: {e}")
    return out


def image_part(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": llm.image_to_data_url(str(path), max_dim=PICTURE_MAX_DIM)}}


def picture_parts(pictures: list, root: Path) -> list[dict]:
    out = []
    for pic in pictures:
        path = Path(pic.path)
        out += [{"type": "text", "text": f"{pic.caption}: {path.relative_to(root).as_posix()}"}, image_part(path)]
    return out


def script_error(error: BaseException, script: Path) -> str:
    """An exception from the script as the line of it that raised, without the harness's own frames."""
    frames = [f for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).name == script.name]
    where = "".join(f"  {script.name} line {f.lineno}, in {f.name}: {f.line}\n" for f in frames)
    return f"{type(error).__name__}: {error}\n{where}".rstrip()


class Tools:
    def __init__(self, root: Path, spec: dict, report: dict, *, builds: int, log=lambda s: None):
        self.root, self.spec, self.rep, self.budget, self.log = root, spec, report, builds, log
        self.source = root / SOURCE
        model, _ = rig_brief.load(self.source)
        self.lo, self.hi = model["lo"], model["hi"]
        self.closing = False
        self.streak = 0                          # builds out of contract since the last one in it
        self.drawn: dict[str, dict] = {}         # clip -> the latest build's frames, for the reviewer's strips

    # -- paths ---------------------------------------------------------------------------------------------------
    def script(self, path: str) -> Path:
        p = agent.in_attempt(self.root, path, file=True)
        if p.name != SCRIPT or p.parent.parent != self.root / "attempts":
            raise agent.Rejected(f"the only file you may write is attempts/<name>/{SCRIPT}")
        if not NAME.fullmatch(p.parent.name):
            raise agent.Rejected("an attempt's name is letters, digits, '.', '_' and '-'")
        return p

    def attempt(self, path: str) -> Path:
        p = agent.in_attempt(self.root, path, file=False)
        return p.parent if p.is_file() else p

    def frozen(self, attempt: Path) -> str:
        return (f"attempts/{attempt.name} is built in contract and stays as it was built. Start the next with "
                f"copy_attempt(source=\"attempts/{attempt.name}\", name=\"<a0N_what_changes>\")."
                if (attempt / BUILT).is_file() else "")

    # -- the tools -----------------------------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> tuple[str, list]:
        if name not in {t["function"]["name"] for t in TOOLS}:
            return f"ERROR: unknown tool {name!r}", []
        return getattr(self, "t_" + name)(args)

    def t_write_file(self, args):
        p = self.script(args.get("path", ""))
        if self.frozen(p.parent):
            return "ERROR: " + self.frozen(p.parent), []
        text = args.get("text", args.get("content"))
        if not isinstance(text, str):
            return "ERROR: write_file needs `text` (the whole script, a string); nothing written", []
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return f"wrote {len(text)} chars to {p.relative_to(self.root).as_posix()}", []

    def t_edit_file(self, args):
        p = self.script(args.get("path", ""))
        if not p.is_file():
            return f"ERROR: no such file {args.get('path')}; create it with write_file or copy_attempt", []
        if self.frozen(p.parent):
            return "ERROR: " + self.frozen(p.parent), []
        edits = args.get("edits")
        if edits is None and "old" in args:
            edits = [{"old": args.get("old"), "new": args.get("new", "")}]
        try:
            text, lines = agent.edit_text(p.read_text(encoding="utf-8"), edits)
        except ValueError as e:
            return f"ERROR: {e}", []
        p.write_text(text, encoding="utf-8")
        return f"edited {p.parent.name}/{p.name}: {len(lines)} edit(s) at line(s) {', '.join(map(str, lines))}", []

    def t_read_file(self, args):
        p = agent.in_attempt(self.root, args.get("path", ""), file=True)
        if p.name not in (SCRIPT, REPORT) or not p.is_file():
            return f"ERROR: read_file reads attempts/<name>/{SCRIPT} or {REPORT}; no such file {args.get('path')}", []
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        start = args.get("start_line", 1)
        end = args.get("end_line", (start if type(start) is int else 1) + 199)
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            return "ERROR: read_file needs 1-based start_line <= end_line", []
        end = min(end, start + 199, len(lines))
        head = f"Lines {start}-{end} of {len(lines)}.\n" if start > 1 or end < len(lines) else ""
        return head + agent.clip("".join(lines[start - 1:end])), []

    def t_copy_attempt(self, args):
        src = self.attempt(args.get("source", ""))
        name = str(args.get("name", "")).strip().strip("/").removeprefix("attempts/")
        if not NAME.fullmatch(name):
            return "ERROR: `name` is the new attempt's directory name alone, like a02_knees", []
        dst = self.root / "attempts" / name
        if not (src / SCRIPT).is_file():
            return f"ERROR: no {SCRIPT} in {args.get('source')}", []
        if dst.exists():
            return f"ERROR: attempts/{name} exists already; pick a new name", []
        dst.mkdir(parents=True)
        text = (src / SCRIPT).read_text(encoding="utf-8")
        (dst / SCRIPT).write_text(text, encoding="utf-8")
        return (f"copied {SCRIPT} ({text.count(chr(10)) + 1} lines) to attempts/{name}/. Change it with edit_file, "
                f"then build attempts/{name}/{SCRIPT}."), []

    def t_build(self, args):
        script = self.script(args.get("script", ""))
        attempt = script.parent
        if not script.is_file():
            return f"ERROR: no such file {args.get('script')}", []
        if self.frozen(attempt):
            return "REFUSED: " + self.frozen(attempt), []
        if self.rep["builds"] >= self.budget:
            return (f"REFUSED: the {self.budget} builds are used. Call done(best, summary) naming the attempt to "
                    "deliver."), []
        entry = {"name": attempt.name, "ok": False, "note": str(args.get("note") or "")[:300]}
        (attempt / FAILED).unlink(missing_ok=True)
        try:
            report = rig.build(self.source, script, self.spec, attempt / BUILT, failed=attempt / FAILED)
            built, verdict = attempt / BUILT, "BUILT AND IN CONTRACT: every gate passed."
            (attempt / BUILT).with_suffix(".rig.json").unlink(missing_ok=True)   # the same record is report.json
        except rig.ContractError as e:
            report, built = e.report, attempt / FAILED
            verdict = ("OUT OF CONTRACT (this is not counted as a build; fix it and build again):\n"
                       + "\n".join(f"! {i}" for i in report["issues"]))
        except Exception as e:  # the script's own error, or a refusal: nothing was built
            self.rep["failed_builds"] += 1
            self.rep["attempts"].append({**entry, "error": str(e)[:300]})
            return "BUILD FAILED, nothing was built:\n" + agent.clip(script_error(e, script), 3000), []
        entry["ok"] = not report["issues"]
        try:
            readings = rig_read.text(built, self.spec, report)
        except Exception as e:  # a reading must not take the build down with it
            readings = f"(the readings failed: {type(e).__name__}: {e})"
        (attempt / REPORT).write_text(json.dumps({**report, "readings": readings}, indent=1) + "\n", encoding="utf-8")
        pictures = self.pictures(attempt, built, report)
        note = ""
        if entry["ok"]:
            self.rep["builds"] += 1
            self.rep["latest"] = attempt.name
            left = self.budget - self.rep["builds"]
            note = (f"\n[{left} build{'s' if left != 1 else ''} left.]" if left else
                    "\n[That was the last build. Call done(best, summary) naming the attempt to deliver.]")
            self.streak = 0
        else:
            self.rep["failed_builds"] += 1
            self.streak += 1
            if self.streak >= STUCK_AFTER:
                # a run that tuned one number a build against a gate its whole approach could not pass made sixteen
                # such builds and never one in contract
                note = (f"\n[{self.streak} builds in a row out of contract. If the same gate keeps failing by about "
                        "as much, the approach is wrong and not its numbers: stop tuning. Read the `stretched at` and "
                        "STRAY WEIGHTS lines for where, look at the WEIGHTS picture (flat colours with a narrow blend "
                        "at each joint is what passes; a muddy gradient is weights by distance), and rewrite that part "
                        "of the script from the brief's recipe.]")
        entry["issues"] = report["issues"]
        self.rep["attempts"].append(entry)
        return agent.clip(verdict + "\n" + readings + note, 14_000), pictures

    def pictures(self, attempt: Path, built: Path, report: dict) -> list:
        out = []
        try:
            sheet = attempt / "weights.png"
            rig_look.weights_sheet(built, self.lo, self.hi).save(sheet)
            out.append(mcp_server.Picture(str(sheet), "weights and skeleton at rest"))
            for clip, c in self.spec["clips"].items():
                if clip not in report.get("clips", {}):
                    continue
                path = attempt / f"strip_{clip}.png"
                self.drawn[clip] = rig_look.cells(built, clip, c["seconds"], self.lo, self.hi, STRIP_VIEWS)
                rig_look.strip(built, clip, c["seconds"], self.lo, self.hi, STRIP_VIEWS, drawn=self.drawn[clip]).save(path)
                out.append(mcp_server.Picture(str(path), f"film strip of clip {clip}"))
        except Exception as e:  # as the readings
            self.log(f"pictures failed: {type(e).__name__}: {e}")
        return out

    def t_look(self, args):
        attempt = self.attempt(args.get("attempt", ""))
        built = attempt / BUILT if (attempt / BUILT).is_file() else attempt / FAILED
        if not built.is_file():
            return f"ERROR: attempts/{attempt.name} has not been built", []
        clip = str(args.get("clip", ""))
        if clip not in self.spec["clips"]:
            return f"ERROR: no clip {clip!r}; the clips are {', '.join(self.spec['clips'])}", []
        times = args.get("times") or []
        views = args.get("views") or list(STRIP_VIEWS)
        if not isinstance(times, list) or not times or not all(isinstance(t, (int, float)) for t in times):
            return "ERROR: `times` is a list of seconds", []
        bad = [v for v in views if v not in rig_look.VIEWS]
        if bad:
            return f"ERROR: no view {bad[0]!r}; the views are {', '.join(rig_look.VIEWS)}", []
        length = self.spec["clips"][clip]["seconds"]
        late = [t for t in times if not 0 <= t <= length]
        if late:
            return (f"ERROR: {clip} is {length:g} s long and `times` are seconds, not a phase: {late[0]:g} is outside "
                    f"0..{length:g}"), []
        aim, span = args.get("aim"), args.get("span")
        if aim is not None and not (isinstance(aim, list) and len(aim) == 3 and all(isinstance(v, (int, float)) for v in aim)):
            return "ERROR: `aim` is [x, y, z] in the script's axes (Blender metres)", []
        if span is not None and not (isinstance(span, (int, float)) and 0.05 <= span <= 5):
            return "ERROR: `span` is the metres across the picture, from 0.05 to 5", []
        doc, blob = glb.read_glb(built)
        rows = []
        for view in views[:3]:
            cells = []
            for t in times[:LOOK_MAX]:
                shot = rig_look.shot(view, self.lo, self.hi, aim, span)
                cell = rig_look.draw(skin.posed(built, clip, float(t)), shot, 384)
                if view in rig_look.ORTHO:
                    cell = rig_look.rule(cell, shot, self.lo, self.hi)
                    if args.get("skeleton"):
                        cell = rig_look.bones(cell, shot, doc, blob, clip, float(t), labels=len(times) <= 3)
                ImageDraw.Draw(cell).text((4, 16), f"{view} t={float(t):.2f}s", fill=(0, 0, 0))
                cells.append(cell)
            rows.append(rig_look.row(cells))
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        self.rep["looks"] += 1
        path = attempt / f"look_{self.rep['looks']:02d}_{clip}.png"
        sheet.save(path)
        return (f"{clip} at {', '.join(f'{float(t):g}' for t in times[:LOOK_MAX])} s"
                + (f", {span:g} m across" if span else "") + (f", centred on {aim}" if aim is not None else "")
                + "; rows: " + "; ".join(f"{v} ({rig_look.FACES[v]})" for v in views[:3])), [mcp_server.Picture(str(path), "a look")]

    def t_done(self, args):
        best = str(args.get("best") or self.rep.get("latest") or "")
        try:
            attempt = self.attempt(best if best.startswith("attempts/") else "attempts/" + best)
        except agent.Rejected:
            attempt = None
        if attempt is None or not (attempt / BUILT).is_file():
            if self.rep["builds"]:
                return (f"ERROR: `best` must name an attempt built in contract; {best!r} is not one. The last was "
                        f"attempts/{self.rep['latest']}."), []
            if not self.closing:
                return "NOT YET: nothing has been built in contract. Fix the script and build it.", []
            self.rep["done"], self.rep["summary"] = True, str(args.get("summary", ""))[:4000]
            return "ok", []
        left = self.budget - self.rep["builds"]
        if left > 0 and not self.closing:
            # a run that stopped at six of eight said in its own summary that its run "needs more lean and bounce":
            # an unused build is the cheapest improvement there is
            return (f"NOT YET: {left} of the {self.budget} builds are unused. Take the first line starting with "
                    "\"!\" in the last readings and the worst thing the last film strips show, fix them in a copy of "
                    "your best attempt, and build it. Call done when the builds are used; you name the best one "
                    "then, which need not be the last."), []
        self.rep["final"], self.rep["done"] = attempt.name, True
        self.rep["summary"] = str(args.get("summary", ""))[:4000]
        return "ok", []


def opening(root: Path, spec: dict, builds: int) -> list[dict]:
    """The brief, the model's facts, the script's rules, the contract and the ruled rest pose."""
    source = root / SOURCE
    model, _ = rig_brief.load(source)
    out = root / "rest"
    out.mkdir(exist_ok=True)
    rest = rig_look.ruled_rest(source, model["lo"], model["hi"])
    parts = [{"type": "text", "text": "THE BRIEF:\n" + (root / BRIEF).read_text(encoding="utf-8").strip()
              + "\n\n" + rig_brief.text(source, spec)},
             {"type": "text", "text": "THE REST POSE, ruled every 0.1 m in the script's own axes (Blender metres). "
                                      "Read joint positions off these:"}]
    pairs = [("front", "left"), ("back", "top")]
    for pair in pairs:
        path = out / f"rest_{'_'.join(pair)}.png"
        rig_look.row([rest[v] for v in pair]).save(path)
        parts += [{"type": "text", "text": "; ".join(f"{v}: {rig_look.FACES[v]}" for v in pair)}, image_part(path)]
    parts.append({"type": "text", "text": f"You have {builds} builds that hold the contract (a failed build is not "
                  "counted). Start: say where the joints go, write attempts/a01/rig.py, and build it."})
    return parts


def review(chat_fn, root: Path, attempt: Path, spec: dict, lo, hi, drawn: dict | None = None) -> tuple[str, dict]:
    """A fresh look at a build in contract: the brief and three-view strips, no script and no numbers. `drawn` is
    the frames the builder's strips were made of, so only the third view is rendered here."""
    parts = [{"type": "text", "text": "THE BRIEF:\n" + (root / BRIEF).read_text(encoding="utf-8").strip()}]
    for clip, c in spec["clips"].items():
        path = attempt / f"review_{clip}.png"
        rig_look.strip(attempt / BUILT, clip, c["seconds"], lo, hi, ("left", "front", "three-quarter"),
                       drawn=(drawn or {}).get(clip)).save(path)
        parts += [{"type": "text", "text": f"Clip {clip} ({c['seconds']:g} s loop):"}, image_part(path)]
    message, usage = chat_fn([{"role": "system", "content": REVIEWER}, {"role": "user", "content": parts}], [])
    text = message.get("content") or ""
    return (text.strip() if isinstance(text, str) else ""), usage


def run(root: Path, chat_fn: Callable, *, builds: int = 8, max_turns: int = 0, model: str = "", url: str = "",
        out: Path | None = None, reviewer: bool = False, context_tokens: int = CONTEXT_TOKENS, reply_reserve: int = 8192,
        log: Callable[[str], None] = lambda s: None) -> dict:
    """Drive the model until done(), the turn cap or a chat failure. Writes report.json (and transcript.jsonl) to
    `out` after returned replies, tool results and every turn (not crash-proof).
    The delivered rig is copied to `<root>/animated.glb` beside its `rig.py`."""
    root = Path(root).resolve()
    issues = problems(root)
    if issues:
        raise ValueError("; ".join(issues))
    if builds < 1:
        raise ValueError("builds must be positive")
    spec = rig_brief.contract(root / CONTRACT)
    max_turns = max_turns or TURNS_PER_BUILD * builds
    rep = {"workflow": "rig", "model": model, "url": url, "builds_allowed": builds, "max_turns": max_turns,
           "reviewer": reviewer, "reviews": 0, "builds": 0, "failed_builds": 0, "looks": 0, "turns": 0, "tool_calls": 0,
           "rejected_calls": 0, "invalid_tool_batches": 0, "dropped_calls": 0, "attempts": [], "latest": None, "final": None, "done": False, "summary": "",
           "error": "", "harness": harness.version(), "prompt_tokens": 0, "completion_tokens": 0,
           "completion_evidence": [], "context_compactions": 0, "wall_s": 0.0}
    tools = Tools(root, spec, rep, builds=builds, log=log)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": opening(root, spec, builds)}]
    transcript = list(messages)
    start = time.monotonic()

    def save():
        rep["wall_s"] = round(time.monotonic() - start, 1)
        if out is not None:
            agent.write_run(out, rep, transcript)

    def checkpoint():
        return ("RESUME STATE: " + json.dumps({"builds_in_contract": rep["builds"], "builds_allowed": builds,
                                               "latest_in_contract": rep["latest"],
                                               "attempts": [a["name"] for a in rep["attempts"]][-6:]})
                + " Earlier turns were dropped to save context. Your scripts are on disk: read_file the latest "
                "attempt's rig.py (and its report.json for the last readings) before changing it.")

    def ask(sent, schemas):
        """One model call, made again when the connection drops under it: a reset an hour into a run otherwise
        ends the run with its last build undelivered (the first trial here ended so)."""
        for again in range(RETRIES + 1):
            try:
                return chat_fn(sent, schemas)
            except OSError as e:
                rep["dropped_calls"] += 1
                log(f"  the call was dropped ({type(e).__name__}: {e}); {'again' if again < RETRIES else 'giving up'}")
                if again == RETRIES:
                    raise llm.LLMError(f"{RETRIES + 1} calls in a row were dropped: {e}") from e
                time.sleep(RETRY_WAIT_S * (again + 1))

    silent, last_prompt = 0, 0
    try:
        while rep["turns"] < max_turns and not rep["done"]:
            rep["turns"] += 1
            tools.closing = max_turns - rep["turns"] < TURNS_LEFT_NOTE
            agent.prune_images(messages, KEEP_IMAGE_TURNS)
            # compact on what the endpoint last counted, not on agent_state's own estimate: that counts 4096 a
            # picture, and a run whose calls were 20,000 tokens was cut back to its checkpoint twelve turns in 21
            limit = context_tokens if last_prompt >= context_tokens - reply_reserve else 0
            messages, compacted = agent_state.compact(messages, TOOLS, checkpoint(), limit, reply_reserve)
            rep["context_compactions"] += bool(compacted)
            before = len(messages)
            log(f"turn {rep['turns']}: waiting on the model (+{time.monotonic() - start:.0f}s)")
            message, usage = ask(messages, TOOLS)
            rep["completion_evidence"].append({"role": "implementor", "turn": rep["turns"],
                "completion": copy.deepcopy(usage.get("mesh_jig_completion"))})
            last_prompt = int(usage.get("prompt_tokens") or 0)
            rep["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            rep["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            message = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
            message["role"] = "assistant"
            transcript.append(message)
            save()
            calls = message.get("tool_calls")
            try:
                parsed = llm.parse_tool_batch([] if calls is None else calls)
            except (ValueError, TypeError) as failure:
                rep["tool_calls"] += len(calls) if isinstance(calls, list) else 0
                rep["invalid_tool_batches"] += 1
                silent = 0
                feedback = llm.rejected_tool_turn(message, failure)
                messages.extend(feedback)
                transcript.append(feedback[-1])
                save()
                continue
            messages.append(message)
            before = len(messages)
            if not parsed:
                # a reply that only says what it will do next is not the end of a run: one run wrote "the next step
                # is a softer shoulder blend" twice, with five builds unused, and was stopped there
                silent += 1
                if silent >= SILENT_TURNS:
                    rep["summary"], rep["error"] = str(message.get("content") or "")[:4000], "stopped without done()"
                    transcript.extend(messages[before:])
                    break
                left = builds - rep["builds"]
                messages.append({"role": "user", "content": (
                    f"Nothing was called. {left} of the {builds} builds are unused: do what you just described now, "
                    "with the tools (copy_attempt or edit_file, then build)." if left > 0 else
                    "Nothing was called. The builds are used: call done(best, summary).")})
                transcript.extend(messages[before:])
                save()
                continue
            silent = 0
            shown, built_ok = [], None
            for tc, fn, args in parsed:
                name = fn.get("name", "")
                rep["tool_calls"] += 1
                before_builds = rep["builds"]
                try:
                    log(f"  {name}({json.dumps(args)[:140]})")
                    result, pics = tools.call(name, args)
                    shown += pics
                except agent.Rejected as e:
                    rep["rejected_calls"] += 1
                    result = f"REJECTED: {e}"
                except (json.JSONDecodeError, ValueError) as e:
                    result = f"ERROR: bad arguments: {e}"
                except Exception as e:  # a tool bug: tell the model, keep going
                    result = f"ERROR: {type(e).__name__}: {e}"
                if rep["builds"] > before_builds:
                    built_ok = root / "attempts" / rep["latest"]
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
                transcript.extend(messages[before:])
                before = len(messages)
                save()
                log(f"  -> {result[:160]!r}")
                if rep["done"]:
                    break
            if built_ok is not None and reviewer and not rep["done"]:
                log("  fresh reviewer")

                def review_chat(sent, schemas):
                    message, used = ask(sent, schemas)
                    rep["completion_evidence"].append({"role": "reviewer", "turn": rep["turns"],
                        "completion": copy.deepcopy(used.get("mesh_jig_completion")),
                        "message": copy.deepcopy(message)})
                    rep["reviews"] += 1
                    rep["prompt_tokens"] += int(used.get("prompt_tokens") or 0)
                    rep["completion_tokens"] += int(used.get("completion_tokens") or 0)
                    # Preserve the raw reviewer reply before text extraction or file writes.
                    save()
                    return message, used

                try:
                    said, used = review(review_chat, root, built_ok, spec, tools.lo, tools.hi, tools.drawn)
                except llm.LLMError:
                    raise
                except Exception as e:  # a review that could not be drawn is not the build's fault
                    said = ""
                    log(f"  review failed: {type(e).__name__}: {e}")
                if said:
                    (built_ok / "review.md").write_text(said + "\n", encoding="utf-8")
                    messages[-1]["content"] += (
                        "\n\nAN INDEPENDENT REVIEW of this build, by someone shown only the brief and the film strips "
                        "(front, side and three-quarter). Weigh it against what you see and the readings above; where "
                        "it names a fault you can also see or the readings confirm, fix it:\n" + said)
            left = max_turns - rep["turns"]
            if not rep["done"] and 0 < left <= TURNS_LEFT_NOTE and messages[-1]["role"] == "tool":
                messages[-1]["content"] += (f"\n[{left} turn{'s' if left != 1 else ''} left. Call done(best, summary) "
                                            "naming your best attempt built in contract.]")
            if shown and not rep["done"]:
                messages.append({"role": "user", "content": picture_parts(shown, root)})
            transcript.extend(messages[before:])
            save()
        else:
            if not rep["done"]:
                rep["error"] = f"hit the turn cap ({max_turns}) without done()"
    except llm.LLMError as e:
        rep["error"] = f"chat failed: {e}"
    except Exception as e:  # the loop's own failure: a sound run lost its builds to one (sound_agent, the same loop)
        rep["error"] = f"{type(e).__name__}: {e}"
        log(f"run stopped: {rep['error']}")
    if not rep["final"] and rep["latest"]:
        rep["final"] = rep["latest"]          # a run that ended without done() still delivers its last build
    if rep["final"]:
        chosen = root / "attempts" / rep["final"]
        shutil.copy2(chosen / BUILT, root / BUILT)
        shutil.copy2(chosen / SCRIPT, root / SCRIPT)
    save()
    return rep


def summary(rep: dict) -> str:
    lines = [f"rig agent {rep['model']}: {'done' if rep['done'] else 'NOT done'} after {rep['turns']} turns, "
             f"{rep['builds']} build(s) in contract, {rep['failed_builds']} failed; delivered "
             f"{rep['final'] or 'nothing'}; {rep['prompt_tokens']}+{rep['completion_tokens']} tokens"
             + (f", ${rep['cost_usd']:.4f}" if rep.get("cost_usd") else "") + f", {rep['wall_s']:.0f} s"]
    if rep["error"]:
        lines.append("error: " + rep["error"])
    if rep["summary"]:
        lines.append(rep["summary"])
    return "\n".join(lines)


def drive(root: Path, backend: llm.Backend, *, builds: int = 8, max_turns: int = 0, max_tokens: int | None = None,
          max_cost: float = 0.0, effort: str | None = None, idle_s: float = 900, reviewer: bool = False,
          log: Callable[[str], None] = print) -> tuple[dict, Path]:
    """`run` over a real endpoint, with the visual workflow's transport and accounting: direct Anthropic 5.5 models
    go through the native Messages transport (explicit effort, a reserved spend cap), anything else through the
    OpenAI-compatible client. -> (report, the run's directory under <root>/agent/)."""
    from . import anthropic_sweep, gauntlet
    from .gauntlet_sweep import Meter, estimate_cost
    root = Path(root).resolve()
    native = backend.url == gauntlet.URL and backend.model in anthropic_sweep.RATES
    if max_cost and not native:
        raise ValueError("a reserved spend cap needs a direct Anthropic Haiku/Sonnet/Opus 5.5 endpoint")
    # as the visual workflow: thinking and the reply share the cap, and three Haiku rig replies were all thinking
    max_tokens = max_tokens or (anthropic_sweep.MAX_OUTPUT_TOKENS if native else gauntlet.MAX_TOKENS)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", backend.model).strip("-") or "model"
    out = root / agent.RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}_rig"
    budget = anthropic_sweep.Budget(out / "budget.json", max_cost, backend.model) if max_cost else None
    if native:
        chat = anthropic_sweep.chat_for(backend, budget, "rig-agent", max_tokens=max_tokens, effort=effort, log=log)
    else:
        chat = agent.chat_for(backend, effort=effort, max_tokens=max_tokens, idle_s=idle_s, log=log)
    out.mkdir(parents=True, exist_ok=True)
    meter = Meter(chat, out, backend.model, price_fn=estimate_cost if native else lambda usage, model: None)
    log(f"rig agent: {backend.describe()}, {builds} builds, reviewer {'on' if reviewer else 'off'} -> {out}")
    rep = run(root, meter, builds=builds, max_turns=max_turns, model=backend.model, url=backend.url, out=out,
              reviewer=reviewer, log=log)
    account = meter.summary()
    rep.update(effort=effort, max_cost=max_cost,
               cost_usd=account["estimated_usd"] if not account["unpriced_calls"] else None,
               cost_kind="estimated" if not account["unpriced_calls"] else "unpriced")
    if budget:
        budget.save()
    agent.atomic_write_text(out / "report.json", json.dumps(rep, indent=1) + "\n")
    return rep, out
