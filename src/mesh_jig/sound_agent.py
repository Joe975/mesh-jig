"""mesh-jig agent --workflow sound: an API model writes the sound script for a sound directory, with no shell.

    mesh-jig agent <sound dir> --workflow sound --model <id> [--attempts 6]

`<sound dir>` holds `sound.json` (the contract), `brief.md` (what the sounds are for) and, when sounds go with clips,
`animated.glb`. The model writes `attempts/<name>/sounds.py`; `build` holds it to the contract (`sound.check`) and
answers with the verdict, where each cue's sound starts, what each sound measures as (`sound_measure`) and a sheet of
spectrograms with the cues drawn on it. Nobody listens: the model is told so, and is given numbers and pictures.

Confinement is `agent`'s: every path is relative to the sound directory and the only file the model may write is
attempts/<name>/sounds.py. The script runs in a child process after `script_check` (numpy, math and random only).

Offline test seam: `run(root, chat_fn, ...)` with chat_fn(messages, tools) -> (assistant message, usage).
"""
from __future__ import annotations

import base64
import copy
import io
import json
import re
import shutil
import time
from pathlib import Path
from typing import Callable

from . import agent, agent_state, harness, llm, sound, sound_measure, sound_recipe

SCRIPT, REPORT, SHEET = sound.SCRIPT, "report.json", "spectrograms.png"
TURNS_PER_BUILD = 8
TURNS_LEFT_NOTE = 3
SILENT_TURNS = 3
RETRIES = 2
RETRY_WAIT_S = 5.0
KEEP_IMAGE_TURNS = 2
SETTLE_AFTER = 3              # builds in contract before one that measures as it should may be delivered early: told
                              # so from the first, builders delivered a01 or a02 and never worked the leap (g11)
PICTURE_MAX_DIM = 1440
CONTEXT_TOKENS = 31_000       # the prompt's budget: a local server slot holds 51,200, and a reply up to 20,000 of them
REPLY_RESERVE = 4_000         # compaction starts this far under it; the opening (brief, recipe, a picture) has to fit
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SYSTEM = """You make a set of sounds by writing ONE Python script that synthesises them. Nobody will listen to what you
make before it is used, and you cannot hear it either: the harness measures each sound and draws its spectrogram, and
that is all anyone knows about it. There is no shell: you have the tools below and nothing else.

Every path you pass is relative to the sound directory. The only file you may write is attempts/<name>/sounds.py.

THE SCRIPT
- It defines a top-level function sounds(cues) and returns a dict {sound name: samples}. Samples are a 1-D numpy
  float array at 44100 samples a second, in -1..1. The harness writes the WAV files; do not write files.
- cues[name] tells you what each sound has to fit. For a sound played once on its own: {"seconds": None,
  "shortest": s, "longest": s, "loop": False}. For a sound that goes with an animation clip: {"clip": ..., "seconds":
  the clip's length, "loop": True or False, "footfalls": [[t, "L" or "R"], ...], and for a leap "crouch",
  "takeoff", "apex", "landing"}, times in seconds from the clip's start. Make such a sound exactly
  int(round(cues[name]["seconds"] * 44100)) samples long and put its events at the times given, read from cues at
  run time (never typed in as numbers). A sound may start new things where there is no cue (a growl pulses, a yowl
  breaks): only each cue the contract names needs a sound starting on it.
- It imports numpy, math and random only. Seed every random generator (np.random.default_rng(seed)).
- A sound played once: scale it so its loudest sample is about 0.9, subtract its mean, and fade the first and last
  5 ms to zero. A loop has no ends: do not fade it; anything that runs past its end wraps round to its start
  (add it at index % n), and its last sample has to meet its first.

THE LOOP
1. Read the brief and the contract. For each sound, say in a sentence or two what it physically is and how it
   changes from its first instant to its last, and how you will make that.
2. write_file attempts/a01/sounds.py: the whole script, with the numbers you will tune as named constants at the
   top. Then build it.
3. Read what build returns: the verdict and every line starting with "!" (each must be fixed), where each cue's
   sound starts, then the numbers that describe each sound and the spectrogram sheet. Say what each sound is,
   by its numbers and its picture, against what it should be, before you change anything.
4. Start the next attempt with copy_attempt and change it with edit_file. An attempt that has been built in
   contract is never changed again.
5. Use the builds to make the sounds more like what they are for, one sound you can name at a time. When every
   sound measures as it should and you cannot say what is still unlike what it is for, deliver: call done naming
   your best attempt, which need not be the last."""


def _fn(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOLS = [
    _fn("write_file", "Write attempts/<name>/sounds.py (creates the directory). For a NEW script; change an existing "
        "one with edit_file.", {"path": S, "text": S}, ["path", "text"]),
    _fn("edit_file", "Change an attempt's sounds.py by exact replacements, for an attempt not yet built in contract. "
        "`edits` is a list of {old, new} applied in order; each `old` must appear exactly once. If one fails, none is "
        "applied. At most four edits a call.",
        {"path": S, "edits": {"type": "array", "items": {"type": "object", "properties": {"old": S, "new": S},
                                                          "required": ["old", "new"]}}}, ["path", "edits"]),
    _fn("read_file", "Read an attempt's sounds.py or report.json (at most 200 lines; start_line/end_line for more).",
        {"path": S, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _fn("copy_attempt", "Start the next attempt from an earlier one: copies its sounds.py into a new attempts/<name>/.",
        {"source": S, "name": S}, ["source", "name"]),
    _fn("build", "Run an attempt's sounds.py twice and hold it to the contract. Returns the verdict with every "
        "number, where a sound starts at each cue, what each sound measures as, and the spectrogram sheet. A build "
        "out of contract returns the same and does not count toward the builds.", {"script": S, "note": S}, ["script"]),
    _fn("done", "Finish: name the attempt to deliver (built in contract), and say what is still wrong with it.",
        {"best": S, "summary": S}, ["best", "summary"]),
]


def problems(root: Path) -> list[str]:
    out = [f"no {name} in {root.name}/" for name in (sound.CONTRACT, sound.BRIEF) if not (root / name).is_file()]
    if not out:
        try:
            sound.load(root)
        except (ValueError, json.JSONDecodeError) as e:
            out.append(str(e))
    return out


def image_part(path: Path) -> dict:
    """A PNG data URL: the local server this runs on has been brought down by JPEG pictures."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > PICTURE_MAX_DIM:
        s = PICTURE_MAX_DIM / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def marks(spec: dict, cues: dict) -> dict:
    """{sound: [(t, label)]}: every cue of a clip's sound, for the sheet."""
    out = {}
    for name, c in cues.items():
        if c.get("seconds") is None:
            continue
        m = [(t, f"{side}") for t, side in c.get("footfalls", [])]
        m += [(c[k], k) for k in sound.CUE_KEYS if k in c]
        out[name] = sorted(m)
    return out


def measured(folder: Path, spec: dict, cues: dict, sheet: Path | None) -> str:
    """Each sound's description, in contract order; the sheet written to `sheet`."""
    lines, drawn, off = [], [], []
    for name in spec["sounds"]:
        path = folder / f"{name}.wav"
        if not path.is_file():
            continue
        rate, x = sound_measure.load(path)
        d = sound_measure.describe(x, rate)
        lines.append(sound_measure.format_description(name, d))
        if spec["sounds"][name].get("expect"):
            off.append(sound_measure.format_against(name, d, spec["sounds"][name]["expect"]))
        for span, want in spec["sounds"][name].get("between", {}).items():
            part = sound.stretch(x, span, cues[name])
            if len(part) >= int(0.05 * rate):          # the pitch tracker needs a few of its 46 ms frames
                off.append(sound_measure.format_against(f"{name} {span}", sound_measure.describe(part, rate), want))
        drawn.append((name, rate, x))
    if sheet is not None and drawn:
        sound_measure.write_sheet(drawn, sheet, marks(spec, cues))
    if off:
        lines += ["", "AGAINST WHAT EACH SHOULD MEASURE AS (readings, not gates: OFF is where to work next):"] + off
    return "\n".join(lines)


def settled(described: str) -> bool:
    """Whether a build's measurements read every sound against what it should measure as and found nothing OFF."""
    return "AGAINST WHAT EACH SHOULD MEASURE AS" in described and "\n  OFF " not in described


class Tools:
    def __init__(self, root: Path, spec: dict, report: dict, *, builds: int, log=lambda s: None):
        self.root, self.spec, self.rep, self.budget, self.log = root, spec, report, builds, log
        self.closing = False
        self.settled: set[str] = set()           # attempts built in contract with every expectation met

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
                if (attempt / REPORT).is_file() and json.loads((attempt / REPORT).read_text(encoding="utf-8")).get("ok")
                else "")

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
            return "ERROR: `name` is the new attempt's directory name alone, like a02_meow_glide", []
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
            return f"REFUSED: the {self.budget} builds are used. Call done(best, summary) naming the attempt to deliver.", []
        entry = {"name": attempt.name, "ok": False, "note": str(args.get("note") or "")[:300]}
        wavs = attempt / "wav"
        res = sound.check(self.root, script, keep=wavs)
        sheet = attempt / SHEET
        sheet.unlink(missing_ok=True)
        try:
            described = measured(wavs, self.spec, res.cues, sheet) if res.sounds else ""
        except Exception as e:  # a description must not take the build down with it
            described = f"(the measurements failed: {type(e).__name__}: {e})"
        verdict = sound.text(res)
        (attempt / REPORT).write_text(json.dumps({**sound.to_json(res), "measured": described}, indent=1) + "\n",
                                      encoding="utf-8")
        entry["ok"] = res.ok
        entry["problems"] = res.problems + [f"{n}: {p}" for n, r in res.sounds.items() for p in r["problems"]]
        self.rep["attempts"].append(entry)
        if not res.sounds:
            self.rep["failed_builds"] += 1
            return "BUILD FAILED, no sounds were made:\n" + agent.clip(verdict, 4000), []
        note = ""
        if res.ok:
            self.rep["builds"] += 1
            self.rep["latest"] = attempt.name
            left = self.budget - self.rep["builds"]
            note = (f"\n[{left} build{'s' if left != 1 else ''} left.]" if left else
                    "\n[That was the last build. Call done(best, summary) naming the attempt to deliver.]")
            if left and settled(described) and self.rep["builds"] >= SETTLE_AFTER:
                self.settled.add(attempt.name)
                note += (f"\n[Every sound of {attempt.name} measures as it should. You may deliver it now with done. "
                         "Build again only to change a sound you can say, from its numbers and picture, is still unlike "
                         "what it is for: in earlier runs the builds made after this point, changing numbers that already "
                         "passed, were more often worse than better.]")
            head = "BUILT AND IN CONTRACT."
        else:
            self.rep["failed_builds"] += 1
            head = "OUT OF CONTRACT (not counted as a build; fix every \"!\" line and build again):"
        text = f"{head}\n{verdict}\n\nWHAT EACH SOUND MEASURES AS:\n{described}{note}"
        pics = [(sheet, "spectrograms (60 Hz to 16 kHz, log; yellow lines are the clip's cues)")] if sheet.is_file() else []
        return agent.clip(text, 12_000), pics

    def t_done(self, args):
        best = str(args.get("best") or self.rep.get("latest") or "")
        best = best.removeprefix("attempts/").strip("/")
        ok = bool(best) and best in {a["name"] for a in self.rep["attempts"] if a["ok"]}
        if not ok:
            if self.rep["builds"]:
                return (f"ERROR: `best` must name an attempt built in contract; {best!r} is not one. The last was "
                        f"attempts/{self.rep['latest']}."), []
            if not self.closing:
                return "NOT YET: nothing has been built in contract. Fix the script and build it.", []
            self.rep["done"], self.rep["summary"] = True, str(args.get("summary", ""))[:4000]
            return "ok", []
        left = self.budget - self.rep["builds"]
        if left > 0 and not self.closing and best not in self.settled:
            return (f"NOT YET: {left} of the {self.budget} builds are unused. Take the sound that is furthest from "
                    "what it is for, change it in a copy of your best attempt, and build it. Call done when the builds "
                    "are used; you name the best one then, which need not be the last."), []
        self.rep["final"], self.rep["done"] = best, True
        self.rep["summary"] = str(args.get("summary", ""))[:4000]
        return "ok", []


def info(root: Path, spec: dict) -> str:
    """What an author starts from, whoever drives: the brief, the contract in words, the cues as the script gets
    them, what each sound should measure as, and the recipe. `mesh-jig sound-info` prints it; the agent's opening is
    it and a word on the builds."""
    cues = sound.cues_for(root, spec)
    text = ("THE BRIEF:\n" + (root / sound.BRIEF).read_text(encoding="utf-8").strip()
            + "\n\nTHE CONTRACT (what the harness checks):\n" + sound.contract_text(spec, cues)
            + "\n\nTHE CUES your script is given (cues argument), as they are:\n" + json.dumps(cues, indent=1))
    wants = [f"- {name}: {sound_measure.wanted(s['expect'])}" for name, s in spec["sounds"].items() if s.get("expect")]
    wants += [f"- {name} from its {span.replace('-', ' to its ')}: {sound_measure.wanted(want)}"
              for name, s in spec["sounds"].items() for span, want in s.get("between", {}).items()]
    if wants:
        text += ("\n\nWHAT EACH SHOULD MEASURE AS (each build reads every sound against these; they are not gates):\n"
                 + "\n".join(wants))
    text += ("\n\nA RECIPE you may start from (copy what you use into your script; it is plain numpy and passes the "
             "contract's import list). voice() makes a voice from a pitch curve and a mouth opening; shaped() makes "
             "noise of any spectrum; thud() a soft impact; growl() a low rattling voice; break_up() takes a finished sound apart; "
             "place() puts a sound on a loop; finish() levels and fades. "
             "example_call shows them together:\n```python\n" + sound_recipe.RECIPE.strip() + "\n```")
    return text


def opening(root: Path, spec: dict, builds: int, system_brief: str = "") -> list[dict]:
    text = info(root, spec)
    if system_brief:
        text += "\n\n" + system_brief
    text += (f"\n\nYou have {builds} builds that hold the contract (a build out of contract is not counted). Start: say "
             "what each sound is and how you will make it, write attempts/a01/sounds.py, and build it.")
    parts = [{"type": "text", "text": text}]
    picture = example_sheet(root / "rest" / "example_call.png")
    if picture is not None:
        parts += [{"type": "text", "text": "THE RECIPE'S example_call, drawn as your builds will be drawn. A classifier of "
                                           "real recordings hears it as a meow: harmonics stacked to 5 kHz and more, "
                                           "the brightest band moving with the mouth, the pitch up and back:"},
                  image_part(picture)]
    return parts


def example_sheet(path: Path) -> Path | None:
    """The recipe's example call as a spectrogram sheet, made once a sound directory."""
    if not path.is_file():
        import numpy as np
        ns: dict = {}
        exec(sound_recipe.RECIPE, ns)      # the package's own text, held by the tests to the script check
        call = ns["example_call"](np.random.default_rng(1))
        path.parent.mkdir(parents=True, exist_ok=True)
        sound_measure.write_sheet([("example_call", sound.RATE, call)], path)
        from PIL import Image
        with Image.open(path) as im:                 # one sound fills one of the sheet's two columns
            cell = im.crop((0, 0, sound_measure.MARGIN_LEFT + sound_measure.PX_PER_S + sound_measure.GAP, im.height))
            cell.load()
        cell.save(path)
    return path if path.is_file() else None


def run(root: Path, chat_fn: Callable, *, builds: int = 6, max_turns: int = 0, model: str = "", url: str = "",
        out: Path | None = None, system: str = SYSTEM, notes: str = "", context_tokens: int = CONTEXT_TOKENS,
        reply_reserve: int = REPLY_RESERVE, log: Callable[[str], None] = lambda s: None) -> dict:
    """Drive the model until done(), the turn cap or a chat failure. Writes report.json (and transcript.jsonl) to
    `out` after returned replies, tool results and every turn (not crash-proof).
    The delivered attempt's script and WAVs are copied to `<root>/delivered/`.
    `system` and `notes` (added to the opening) are what a prompt experiment changes."""
    root = Path(root).resolve()
    issues = problems(root)
    if issues:
        raise ValueError("; ".join(issues))
    if builds < 1:
        raise ValueError("builds must be positive")
    spec = sound.load(root)
    max_turns = max_turns or TURNS_PER_BUILD * builds
    rep = {"workflow": "sound", "model": model, "url": url, "builds_allowed": builds, "max_turns": max_turns,
           "builds": 0, "failed_builds": 0, "turns": 0, "tool_calls": 0, "rejected_calls": 0, "invalid_tool_batches": 0, "dropped_calls": 0,
           "attempts": [], "latest": None, "final": None, "done": False, "summary": "", "error": "",
           "harness": harness.version(), "prompt_tokens": 0, "completion_tokens": 0,
           "completion_evidence": [], "context_compactions": 0,
           "wall_s": 0.0}
    tools = Tools(root, spec, rep, builds=builds, log=log)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": opening(root, spec, builds, notes)}]
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
                "attempt's sounds.py (and its report.json for the last measurements) before changing it.")

    def ask(sent, schemas):
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
                silent += 1
                if silent >= SILENT_TURNS:
                    rep["summary"], rep["error"] = str(message.get("content") or "")[:4000], "stopped without done()"
                    transcript.extend(messages[before:])
                    break
                left = builds - rep["builds"]
                messages.append({"role": "user", "content": (
                    f"Nothing was called. {left} of the {builds} builds are unused: do what you just described now, "
                    "with the tools (write_file, copy_attempt or edit_file, then build)." if left > 0 else
                    "Nothing was called. The builds are used: call done(best, summary).")})
                transcript.extend(messages[before:])
                save()
                continue
            silent = 0
            shown = []
            for tc, fn, args in parsed:
                name = fn.get("name", "")
                rep["tool_calls"] += 1
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
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
                transcript.extend(messages[before:])
                before = len(messages)
                save()
                log(f"  -> {result[:160]!r}")
                if rep["done"]:
                    break
            left = max_turns - rep["turns"]
            if not rep["done"] and 0 < left <= TURNS_LEFT_NOTE and messages[-1]["role"] == "tool":
                messages[-1]["content"] += (f"\n[{left} turn{'s' if left != 1 else ''} left. Call done(best, summary) "
                                            "naming your best attempt built in contract.]")
            if shown and not rep["done"]:
                parts = []
                for path, caption in shown:
                    parts += [{"type": "text", "text": f"{caption}: {Path(path).relative_to(root).as_posix()}"},
                              image_part(Path(path))]
                messages.append({"role": "user", "content": parts})
            transcript.extend(messages[before:])
            save()
        else:
            if not rep["done"]:
                rep["error"] = f"hit the turn cap ({max_turns}) without done()"
    except llm.LLMError as e:
        rep["error"] = f"chat failed: {e}"
    except Exception as e:  # the harness's own failure: a whole generation once lost its builds to one here
        rep["error"] = f"{type(e).__name__}: {e}"
        log(f"run stopped: {rep['error']}")
    if not rep["final"] and rep["latest"]:
        rep["final"] = rep["latest"]
    if rep["final"]:
        chosen = root / "attempts" / rep["final"]
        delivered = root / "delivered"
        shutil.rmtree(delivered, ignore_errors=True)
        shutil.copytree(chosen / "wav", delivered)
        shutil.copy2(chosen / SCRIPT, delivered / SCRIPT)
    save()
    return rep


def summary(rep: dict) -> str:
    lines = [f"sound agent {rep['model']}: {'done' if rep['done'] else 'NOT done'} after {rep['turns']} turns, "
             f"{rep['builds']} build(s) in contract, {rep['failed_builds']} failed; delivered "
             f"{rep['final'] or 'nothing'}; {rep['prompt_tokens']}+{rep['completion_tokens']} tokens, {rep['wall_s']:.0f} s"]
    if rep["error"]:
        lines.append("error: " + rep["error"])
    if rep["summary"]:
        lines.append(rep["summary"])
    return "\n".join(lines)


def drive(root: Path, backend: llm.Backend, *, builds: int = 6, max_turns: int = 0, max_tokens: int | None = None,
          effort: str | None = None, idle_s: float = 900, system: str = SYSTEM, notes: str = "",
          log: Callable[[str], None] = print) -> tuple[dict, Path]:
    """`run` over an OpenAI-compatible endpoint. -> (report, the run's directory under <root>/agent/)."""
    root = Path(root).resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", backend.model).strip("-") or "model"
    out = root / agent.RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}_sound"
    chat = agent.chat_for(backend, effort=effort, max_tokens=max_tokens, idle_s=idle_s, log=log)
    out.mkdir(parents=True, exist_ok=True)
    log(f"sound agent: {backend.describe()}, {builds} builds -> {out}")
    rep = run(root, chat, builds=builds, max_turns=max_turns, model=backend.model, url=backend.url, out=out,
              system=system, notes=notes, log=log)
    rep["effort"] = effort
    agent.atomic_write_text(out / "report.json", json.dumps(rep, indent=1) + "\n")
    return rep, out
