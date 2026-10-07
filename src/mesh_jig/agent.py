"""mesh-jig agent: a small tool loop that drives an API model through the same tools a coding agent gets.

    mesh-jig agent <project> --model <id> [--url <endpoint>] [--attempts 5] [--judge]

The model sees what Claude Code sees through `mesh-jig-mcp`: the brief and the reference views up front, then the
MCP tool functions themselves (`evaluate`, `build`, `preview`, `paint`, and `judge` when asked for), called in
process, plus plain file tools. There is no shell. Ported from an earlier tool loop, with its lessons
kept:

- `edit_file` carries every change to a file in one call, each `old` unique, all or nothing: whole-file rewrites
  re-sent an 18k-character script 42 times and lost what worked; one edit a turn cost 2.5x the tokens.
- A binary never enters the context: one `cat reference.png` cost 5.6M prompt tokens over a round.
- A tool message is text only, so a turn's pictures follow its tool results as one user message, and every picture
  is re-sent on every later turn: only the last KEEP_IMAGE_TURNS turns' pictures stay, older ones become captions.
  Not under a marked prompt cache (Anthropic's models, `llm.cache_marks`): there a picture already sent is read back
  at a tenth of the input price, and rewriting an old message to drop one puts every later byte out of the cache,
  so the pictures stay (up to MAX_PICTURES).
- The last TURNS_LEFT_NOTE turns are announced in the turn's last tool result.

Confinement: the project is bound by the harness; every path is relative to it and must stay inside it. Writes go
only under attempts/<name>/, so jig.json, the references, the brief and the ledger are out of the model's reach.
Every tool path argument names something under attempts/ too, since `evaluate`, `build` and `preview` write beside
it.

Offline test seam: `run(project, chat_fn, ...)` where chat_fn(messages, tools) -> (assistant message, usage). Under a
time budget (`max_wall`) it is called as chat_fn(messages, tools, left_s), with the seconds the run has left.
"""

from __future__ import annotations

import base64
import inspect
import json
import re
import time
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import agent_state
from . import evaluate as evaluate_mod, glb, llm, mcp_server, project as project_mod, silhouette, streak as streak_mod

ChatFn = Callable[..., tuple[dict, dict]]      # (messages, tools) or, under a time budget, (messages, tools, left_s)

MAX_TOOL_OUTPUT = 12_000
MAX_FILE_READ = 60_000
MAX_LIST = 200
MAX_VIEW = 6                 # pictures per view_image call
IMAGE_MAX_DIM = 768          # every picture the model sees
KEEP_IMAGE_TURNS = 2         # tool turns whose pictures stay in the context; the brief's references always stay
MAX_PICTURES = 90            # with every turn's pictures kept (a cached prefix): Anthropic refuses a request over 100
TURNS_PER_ATTEMPT = 12       # default turn cap per evaluated attempt
TURNS_LEFT_NOTE = 3
RUNS_DIR = "agent"           # <project>/agent/<stamp>_<model>/: report.json, transcript.jsonl
MEASURED = "built, in contract, rendered and measured"
DROPPED = "(picture dropped to save context; view_image the path above to see it again)"

# Every str parameter of every MCP tool is one of these: a path is confined, a text is passed through. A parameter in
# neither makes `tool_schemas` raise, so a new one cannot skip confinement.
PATH_ARGS = {"script", "attempt", "attempt_a", "attempt_b", "paint_plan", "atlas", "plan", "out", "judge_against"}
TEXT_ARGS = {"note", "name", "views", "samples"}
JUDGE_ARGS = {"judge_against", "samples"}            # evaluate's judge parameters: offered only with judge on
HIDDEN_ARGS = {"evaluate": {"name"}}
BOUND = "project"
NOT_OFFERED = {"brief"}                              # its text and pictures open the conversation already

BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".glb",
                   ".gltf", ".blend", ".zip", ".gz", ".pyc")


class Rejected(Exception):
    """A request outside what the model may touch: answered with REJECTED and counted, the loop goes on."""


# -- the tools -------------------------------------------------------------------------------------------------------

def _json_type(hint) -> dict:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        return _json_type(inner[0])
    if origin is list or hint is list:
        args = typing.get_args(hint)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    return {str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"},
            bool: {"type": "boolean"}}.get(hint, {"type": "string"})


def mcp_schema(fn: Callable, judge: bool) -> dict:
    """An OpenAI tool schema for one MCP tool function, from its signature and docstring, minus the bound project."""
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        if name == BOUND or (not judge and fn is mcp_server.evaluate and name in JUDGE_ARGS):
            continue
        if name in HIDDEN_ARGS.get(fn.__name__, ()):
            continue
        if name not in PATH_ARGS | TEXT_ARGS:
            raise ValueError(f"{fn.__name__}({name}) is neither a path nor a text argument: classify it in agent.py")
        props[name] = _json_type(hints.get(name, str))
        if name in PATH_ARGS:
            props[name]["description"] = "relative to the project directory, under attempts/"
        if p.default is inspect.Parameter.empty:
            required.append(name)
    doc = inspect.getdoc(fn) or fn.__name__
    doc += "\n(The project is fixed by the harness; do not pass it.)"
    return {"type": "function", "function": {"name": fn.__name__, "description": doc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
FILE_TOOLS = [
    _fn("read_file", "Read a text file under the project (scripts, plans, eval.json, attempts/ledger.md). Pictures go "
        "through view_image, never read_file. Reads at most 200 lines; use start_line/end_line (1-based) for excerpts.",
        {"path": S, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _fn("search_file", "Find literal text in one project text file; returns at most 20 matching line snippets.",
        {"path": S, "query": S}, ["path", "query"]),
    _fn("write_file", "Write a text file under attempts/<name>/ (creates the directory). Use it for a NEW file; change "
        "an existing one with edit_file.", {"path": S, "text": S}, ["path", "text"]),
    _fn("copy_attempt", "Start the next attempt from an earlier one: copies its script (and atlas plan) into a new "
        "attempts/<name>/, which you then change with edit_file. Cheaper than writing the script out again, and the "
        "earlier attempt stays as it was measured.", {"source": S, "name": S}, ["source", "name"]),
    _fn("edit_file", "Change a text file under an attempt that is not measured yet, by exact replacements. Put EVERY change to the file in "
        "ONE call: `edits` is a list of {old, new}, applied in order; each `old` must appear exactly once in the file as "
        "the earlier edits left it (copy it with enough surrounding lines to be unique). If any edit fails, none is "
        "applied.", {"path": S, "edits": {"type": "array", "items": {
            "type": "object", "properties": {"old": S, "new": S}, "required": ["old", "new"]}}}, ["path", "edits"]),
    _fn("list_dir", "List a directory under the project (default: the project itself); directories end in /.",
        {"path": S}, []),
    _fn("view_image", f"Look at up to {MAX_VIEW} pictures under the project: the *_vs_reference.png side-by-sides and "
        "sil_*.png outline diffs `evaluate` names, the reference views, renders.",
        {"paths": {"type": "array", "items": S}}, ["paths"]),
    _fn("done", "Finish, once every evaluated attempt in the budget is made: name your best attempt, its numbers and "
        "what is still wrong.", {"summary": S}, ["summary"]),
]


def tool_schemas(judge: bool = False) -> list[dict]:
    """Every tool the model is offered: the MCP tools (judge only when asked for) and the file tools."""
    mcp = [mcp_schema(fn, judge) for fn in mcp_server.TOOLS
           if fn.__name__ not in NOT_OFFERED and (judge or fn.__name__ != "judge")]
    return mcp + FILE_TOOLS


# -- confinement (pure) ----------------------------------------------------------------------------------------------

def inside(root: Path, path: str) -> Path:
    """`path` (relative to the project, or absolute) resolved, and inside the project; Rejected otherwise."""
    if not isinstance(path, str) or not path.strip():
        raise Rejected("an empty path")
    p = Path(path)
    p = (p if p.is_absolute() else root / p).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise Rejected(f"{path} is outside the project") from None
    return p


def in_attempt(root: Path, path: str, file: bool) -> Path:
    'Inside attempts/<name>/: the attempt directory itself, or (file=True) a file within one.'
    p = inside(root, path)
    first = Path(path).parts[0] if Path(path).parts and not Path(path).is_absolute() else ""
    if first and first != project_mod.ATTEMPTS and (root / project_mod.ATTEMPTS / first).is_dir():
        p = inside(root, f"{project_mod.ATTEMPTS}/{path}")
    try:
        rel = p.relative_to(root / project_mod.ATTEMPTS)
    except ValueError:
        raise Rejected(f"{path} is not under {project_mod.ATTEMPTS}/<name>/") from None
    if len(rel.parts) < (2 if file else 1):
        raise Rejected(f"{path} must be {'a file inside ' if file else ''}{project_mod.ATTEMPTS}/<name>/")
    return p


def is_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    try:
        with path.open("rb") as f:
            return b"\0" in f.read(4096)
    except OSError:
        return False


def clip(s: str, n: int = MAX_TOOL_OUTPUT) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars clipped]...\n" + s[-n // 2:]


def edit_text(text: str, edits: list) -> tuple[str, list[int]]:
    """Apply exact, unique replacements in order; ValueError (nothing applied) when one fails. -> (text, lines)."""
    if not isinstance(edits, list) or not edits or not all(isinstance(e, dict) for e in edits):
        raise ValueError("`edits` must be a non-empty list of {\"old\": ..., \"new\": ...}")
    lines = []
    for i, e in enumerate(edits, 1):
        old, new = str(e.get("old", "")), str(e.get("new", ""))
        which = f"edit {i} of {len(edits)}: " if len(edits) > 1 else ""
        n = text.count(old) if old else 0
        if n != 1:
            why = "`old` is empty" if not old else "`old` was not found" if n == 0 else f"`old` matches {n} places"
            raise ValueError(f"{which}{why}; nothing changed. read_file the file and copy the lines exactly, with "
                             "enough around them to be unique")
        at = text.index(old)
        lines.append(text.count("\n", 0, at) + 1)
        text = text[:at] + new + text[at + len(old):]
    return text, lines


# -- one call ----------------------------------------------------------------------------------------------------------

@dataclass
class Report:
    project: str = ""                          # its name, not where it is: a report is published
    model: str = ""
    url: str = ""
    turns: int = 0
    prompt_tokens: int = 0                     # everything sent, whether read from a cache or not
    cached_tokens: int = 0                     # of prompt_tokens: read from the endpoint's prompt cache
    cache_write_tokens: int = 0                # of prompt_tokens: written to it (billed above the input price)
    completion_tokens: int = 0
    reasoning_tokens: int = 0                  # of completion_tokens
    cost_usd: float = 0.0
    max_cost_usd: float = 0.0                  # the cap the run was given; 0 is none
    wall_s: float = 0.0
    max_wall_s: float = 0.0                    # the time budget the run was given; 0 is none
    patience: int = 0                          # attempts in a row with no improvement that end the run; 0: a fixed number
    margin: float = 0.0                        # under `patience`: what an attempt has to gain to count as an improvement
    stopped: str = ""                          # why `mesh_jig.streak`'s rule ended the run, when it did
    tool_calls: int = 0
    rejected_calls: int = 0
    evaluations: int = 0                       # evaluate calls that built, held the contract and measured
    attempts: list = field(default_factory=list)   # {"name", "ok", "numeric_score"} per evaluate call
    best: dict | None = None                   # the measured attempt with the highest numeric score
    done: bool = False
    summary: str = ""
    error: str = ""
    context_compactions: int = 0
    calls: list = field(default_factory=list)  # per model call: its tokens, its cost, the evaluations done before it

    def count(self, usage: dict) -> float:
        """Add one reply's usage (OpenAI's shape, with OpenRouter's cache and cost fields when present). -> its cost."""
        sent, made = usage.get("prompt_tokens_details") or {}, usage.get("completion_tokens_details") or {}
        try:
            cost = float(usage.get("cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        call = {"turn": self.turns, "evaluations": self.evaluations, "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "cached_tokens": int(sent.get("cached_tokens") or 0),
                "cache_write_tokens": int(sent.get("cache_write_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(made.get("reasoning_tokens") or 0), "cost_usd": cost}
        for k in ("prompt_tokens", "cached_tokens", "cache_write_tokens", "completion_tokens", "reasoning_tokens"):
            setattr(self, k, getattr(self, k) + call[k])
        self.cost_usd = round(self.cost_usd + cost, 6)
        self.calls.append(call)
        return cost


class Tools:
    """Dispatches one tool call: (text for the tool message, [Picture] for the user message after the turn)."""

    def __init__(self, project: project_mod.Project, report: Report, *, attempts: int, judge: bool,
                 patience: int = 0, margin: float = streak_mod.MARGIN, max_previews: int = 2,
                 require_evaluation: bool = True):
        self.proj, self.root, self.rep = project, project.root, report
        self.budget, self.judge = attempts, judge
        self.patience, self.margin = patience, margin    # patience 0: exactly `attempts`; else `attempts` is a ceiling
        self.closing = False                     # the last TURNS_LEFT_NOTE turns: done() is taken whatever is unused
        self.overlaps: dict[str, float] = {}     # measured attempt -> its overlap, for a preview to be held against
        self.offered = {t["function"]["name"] for t in tool_schemas(judge)}
        self.lifecycle = agent_state.Lifecycle(self.root, max_previews, require_evaluation)
        prior, _ = streak_mod.attempts(self.root)
        if prior:
            best = max(prior, key=lambda a: a["score"])
            self.rep.best = {"name": best["name"], "ok": True, "numeric_score": best["score"]}
            try:
                rec = evaluate_mod.read(self.root / project_mod.ATTEMPTS / best["name"])
            except (OSError, ValueError):
                rec = {}  # the directory may have moved since history was read
            overlap = (rec.get("numeric") or {}).get("ortho_iou")
            if overlap is not None:
                self.overlaps[best["name"]] = overlap

    def checkpoint(self) -> str:
        return "RESUME STATE: " + json.dumps({
            "best": self.rep.best, "evaluations_this_run": self.rep.evaluations,
            "attempt_budget": self.budget, "stop_reason": self.rep.stopped,
            "recent_attempts": self.rep.attempts[-3:], "pending_candidate": self.lifecycle.pending,
            "pending_previews": self.lifecycle.counts.get(self.lifecycle.pending, 0),
            "max_previews": self.lifecycle.max_previews,
        }) + " Use search_file and bounded read_file excerpts to inspect the saved scripts. Preserve evaluated attempts."


    def call(self, name: str, args: dict) -> tuple[str, list]:
        if name not in self.offered:
            return f"ERROR: unknown tool {name!r}; the tools are {sorted(self.offered)}", []
        candidate = None
        if name == "copy_attempt":
            target = str(args.get("name", "")).strip().strip("/").removeprefix("attempts/")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", target):
                return "ERROR: `name` is the new attempt's directory name alone, like a02_longer_legs", []
            candidate = in_attempt(self.root, "attempts/" + target, file=False)
        elif name in ("write_file", "edit_file") and args.get("path"):
            candidate = in_attempt(self.root, args["path"], file=True)
        elif name in ("build", "evaluate") and args.get("script"):
            candidate = in_attempt(self.root, args["script"], file=True)
        elif name == "paint" and args.get("out"):
            candidate = in_attempt(self.root, args["out"], file=True)
        elif name == "preview" and args.get("attempt"):
            candidate = in_attempt(self.root, args["attempt"], file=False)
        if candidate is not None:
            candidate_name = candidate.relative_to(self.root / project_mod.ATTEMPTS).parts[0]
            why = self.lifecycle.refuse(candidate_name, preview=name == "preview")
            if why:
                return why, []
        if name in {fn.__name__ for fn in mcp_server.TOOLS}:
            text, pictures = self.mcp(name, args)
            return self.relative(text), pictures
        return getattr(self, "t_" + name)(args)

    def relative(self, text: str) -> str:
        """Tool output with the project's own path taken off every path in it: the model passes relative paths, and
        an evaluation names fifteen files under a root that can be a hundred characters long."""
        for root in (str(self.root), self.root.as_posix()):
            for sep in ("\\", "/"):
                text = text.replace(root + sep, "")
            text = text.replace(root, ".")
        return text

    def measured(self, path: Path) -> str:
        """The name of the attempt `path` is in when that attempt is already measured, else ""."""
        try:
            name = path.relative_to(self.root / project_mod.ATTEMPTS).parts[0]
            rec = json.loads((self.root / project_mod.ATTEMPTS / name / evaluate_mod.EVAL_FILE).read_text(encoding="utf-8"))
        except (ValueError, IndexError, OSError, json.JSONDecodeError):
            return ""
        return name if rec.get("ok") else ""

    def frozen(self, path: Path) -> str:
        """Why a measured attempt may not be changed or evaluated again, or ""."""
        name = self.measured(path)
        return (f"{project_mod.ATTEMPTS}/{name} is already measured and stays as it was measured (the ledger names "
                f"it). Start the next attempt with copy_attempt(source=\"{project_mod.ATTEMPTS}/{name}\", "
                "name=\"<a0N_what_changes>\") and edit_file the copy, or write_file a new directory.") if name else ""

    def mcp(self, name: str, args: dict) -> tuple[str, list]:
        fn = next(f for f in mcp_server.TOOLS if f.__name__ == name)
        params = inspect.signature(fn).parameters
        kw = {}
        for k, v in args.items():
            if k not in params or k == BOUND or (not self.judge and k in JUDGE_ARGS) or k in HIDDEN_ARGS.get(name, ()):
                raise Rejected(f"{name} takes no {k!r} argument here")
            if k in PATH_ARGS and v not in ("", None):
                v = str(in_attempt(self.root, str(v), file=k in ("script", "paint_plan", "atlas", "plan", "out")))
            kw[k] = v
        if name == "evaluate":
            if self.rep.evaluations >= self.budget:
                return (f"REFUSED: the attempt budget ({self.budget} evaluated attempts) is used. Call done(summary) "
                        "naming your best attempt."), []
            if self.rep.stopped:
                return (f"REFUSED: {self.rep.stopped}, so the run is over. Call done(summary) naming your best "
                        "attempt."), []
            why = self.frozen(Path(kw.get("script") or self.root))
            if why:
                return "REFUSED: " + why, []
        if name in ("build", "paint"):
            why = self.frozen(Path(kw.get("script" if name == "build" else "out") or self.root))
            if why:
                return "REFUSED: " + why, []
        started = time.time()
        reply = mcp_server.guarded(fn)(str(self.root), **kw)
        # the project's own path comes off before the clip, not after: an evaluation names its pictures under a root
        # that can be a hundred characters long, and a text that only the roots pushed past MAX_TOOL_OUTPUT lost its
        # middle to the clip (on the pirate cat that was the gate lines)
        text = self.relative("\n".join(x for x in reply if isinstance(x, str)))
        pictures = [x for x in reply if isinstance(x, mcp_server.Picture)]
        if name == "evaluate":
            note, diffs = self.record(Path(kw["script"]).parent, started)
            text, pictures = text + note, pictures + diffs
        if name == "preview" and kw.get("attempt"):
            if "a free look" in text:
                candidate = Path(kw["attempt"]).relative_to(self.root / project_mod.ATTEMPTS).parts[0]
                self.lifecycle.previewed(candidate)
            note, diffs = self.looked(Path(kw["attempt"]) / "preview", started)
            text, pictures = text + note, pictures + diffs
        return clip(text), pictures

    def scored_views(self) -> dict[str, float]:
        """view -> weight for the views the overlap is scored on: the project's `scoring` when it names them, else
        the orthographic three, evenly."""
        return self.proj.scoring_weights or dict.fromkeys(silhouette.ORTHO_VIEWS, 1.0)

    def looked(self, out: Path, since: float) -> tuple[str, list]:
        """What a preview adds here: its overlap over the scored views beside the best measured attempt's, so a look
        answers "is this change worth an attempt", and the outline diffs it drew."""
        f = out / silhouette.REPORT
        try:
            if f.stat().st_mtime < since - 1:
                return "", []
            sil = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "", []
        views = sil.get("views") or {}
        weights = self.scored_views()
        scored = [v for v in weights if v in views] or list(views)
        if not scored:
            return "", []
        overlap = (sum(views[v]["iou"] * weights.get(v, 1.0) for v in scored)
                   / sum(weights.get(v, 1.0) for v in scored))
        best = self.rep.best
        held = (f"; your best measured attempt, {best['name']}, has {self.overlaps[best['name']]:.3f}"
                if best and best["name"] in self.overlaps else "")
        note = (f"\n[A free look, not an attempt: overlap {overlap:.3f} over {', '.join(scored)}{held}. Change the script "
                "and look again, or evaluate it.]")
        return note, self.outline_diffs({"overlays": {v: str(out / f"{silhouette.OVERLAY_PREFIX}{v}.png") for v in scored}})

    def record(self, attempt: Path, since: float) -> tuple[str, list]:
        """Count an evaluate call from the eval.json it wrote (an older one means it stopped before evaluating).
        -> (the budget note, the outline diffs of the views the score is taken on)."""
        f = attempt / evaluate_mod.EVAL_FILE
        try:
            if f.stat().st_mtime < since - 1:
                return "", []
            rec = evaluate_mod.read(attempt)
        except (OSError, json.JSONDecodeError):
            return "", []
        score = (rec.get("numeric") or {}).get("numeric_score")
        entry = {"name": rec.get("name") or attempt.name, "ok": bool(rec.get("ok")), "numeric_score": score}
        self.rep.attempts.append(entry)
        if not entry["ok"]:
            return "", []
        self.rep.evaluations += 1
        if self.lifecycle.pending == self.lifecycle.validate(attempt.name):
            self.lifecycle.pending = None
            self.lifecycle.save()
        if (rec.get("numeric") or {}).get("ortho_iou") is not None:
            self.overlaps[entry["name"]] = rec["numeric"]["ortho_iou"]
        if score is not None and (self.rep.best is None or score > (self.rep.best["numeric_score"] or -1)):
            self.rep.best = entry
        left = self.budget - self.rep.evaluations
        if self.patience and left:
            return self.counted(entry, left), self.outline_diffs(rec)
        note = (f"\n[{left} evaluated attempt{'s' if left != 1 else ''} left in the budget.]" if left else
                "\n[That was the last evaluated attempt in the budget. Call done(summary) naming your best attempt.]")
        return self.standing(entry, left) + note, self.outline_diffs(rec)

    def counted(self, entry: dict, left: int) -> str:
        """Under `patience`: what `mesh-jig streak` says of the project as it now stands, which is the whole of the
        stopping rule (the count, the best, the gates it fails, KEEP GOING or STOP). The model is handed it with
        every evaluation and does not count for itself, as a builder with a shell is told to ask the command."""
        v = streak_mod.verdict(self.root, self.patience, self.margin)
        if v["stop"]:
            self.rep.stopped = v["why"]
            return self.standing(entry, 0) + "\n[" + streak_mod.text(v) + " Make no more attempts: call done(summary).]"
        return (self.standing(entry, left) + "\n[" + streak_mod.text(v)
                + f" {left} more at most: the ceiling is {self.budget}.]")

    def standing(self, entry: dict, left: int) -> str:
        """Where a measured attempt stands against the best so far, and which one to go on from. A run that builds
        each attempt on the last carries a change that lost score into every later attempt
        (one run: 0.515, then 0.506 and 0.507 from it)."""
        best = self.rep.best
        if best is None or best is entry:
            return "\n[The best attempt so far.]" if self.rep.evaluations > 1 else ""
        behind = f"\n[Below {best['name']} ({best['numeric_score']:.3f}), the best so far: this change lost score."
        return behind + (f" Start the next attempt from {best['name']} (copy_attempt), not from this one, and "
                         "change something else.]" if left else "]")

    def outline_diffs(self, rec: dict) -> list:
        """The sil_<view>.png of each view the overlap is scored on (all of them when none is one of those). The
        side-by-sides show what was built; these show where its outline is wrong, and no run asked for them unprompted."""
        overlays = {v: p for v, p in (rec.get("overlays") or {}).items() if Path(p).is_file()}
        scored = {v: overlays[v] for v in self.scored_views() if v in overlays} or overlays
        return [mcp_server.Picture(p, f"outline diff, {v} (grey: both; RED: the reference has body here and yours has "
                                      "none, add it; BLUE: yours has body here and the reference has none, remove it)")
                for v, p in scored.items()]

    def t_read_file(self, args: dict) -> tuple[str, list]:
        p = inside(self.root, args.get("path", ""))
        if not p.is_file():
            return f"ERROR: no such file {args.get('path')}", []
        if is_binary(p):
            return f"BINARY: {p.name} is {p.stat().st_size} bytes; look at pictures with view_image", []
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        start = args.get("start_line", 1)
        end = args.get("end_line", start + 199 if type(start) is int else 0)
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ValueError("read_file needs 1-based start_line <= end_line")
        if start > len(lines):
            return f"No lines at or after {start}; file has {len(lines)} lines.", []
        end = min(end, start + 199, len(lines))
        content = "".join(lines[start - 1:end])
        header = f"Lines {start}-{end} of {len(lines)}; request another range for more.\n" if start > 1 or end < len(lines) else ""
        return header + clip(content, MAX_TOOL_OUTPUT), []

    def t_search_file(self, args: dict) -> tuple[str, list]:
        p = inside(self.root, args.get("path", ""))
        query = args.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("search_file needs a nonempty literal query")
        if not p.is_file() or is_binary(p):
            return "ERROR: search_file needs a text file", []
        matches = []
        with p.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if query in line:
                    matches.append(f"{number}: {clip(line.rstrip(), 300)}")
                    if len(matches) == 20:
                        return "\n".join(matches) + "\n[20-match limit; narrow the query]", []
        return "\n".join(matches) or "No matches.", []

    def t_write_file(self, args: dict) -> tuple[str, list]:
        p = in_attempt(self.root, args.get("path", ""), file=True)
        if self.frozen(p):
            return "ERROR: " + self.frozen(p), []
        # `content` is what other harnesses call it. A mech run (deepseek-v4.1-flash) switched to
        # it mid-run; the missing `text` was written as an empty file and a dozen turns went on working out why its
        # scripts had no build().
        text = args.get("text", args.get("content"))
        if not isinstance(text, str):
            return "ERROR: write_file needs `text` (the file's whole content, a string); nothing written", []
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return f"wrote {len(text)} chars to {p.relative_to(self.root).as_posix()}", []

    def t_edit_file(self, args: dict) -> tuple[str, list]:
        p = in_attempt(self.root, args.get("path", ""), file=True)
        if not p.is_file():
            return f"ERROR: no such file {args.get('path')}; create it with write_file first", []
        if self.frozen(p):
            return "ERROR: " + self.frozen(p), []
        edits = args.get("edits")
        if edits is None and "old" in args:          # one edit, written flat: {path, old, new}
            edits = [{"old": args.get("old"), "new": args.get("new", "")}]
        try:
            text, lines = edit_text(p.read_text(encoding="utf-8"), edits)
        except ValueError as e:
            return f"ERROR: {e}", []
        p.write_text(text, encoding="utf-8")
        return f"edited {p.name}: {len(lines)} edit(s) at line(s) {', '.join(map(str, lines))}", []

    def t_copy_attempt(self, args: dict) -> tuple[str, list]:
        src = in_attempt(self.root, args.get("source", ""), file=False)
        src = src.parent if src.is_file() else src
        name = str(args.get("name", "")).strip().strip("/")
        name = name.split("/", 1)[1] if name.startswith(project_mod.ATTEMPTS + "/") else name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            return "ERROR: `name` is the new attempt's directory name alone, like a02_longer_legs", []
        dst = self.root / project_mod.ATTEMPTS / name
        script = glb.script_name(self.proj.spec)
        if not (src / script).is_file():
            return f"ERROR: no {script} in {args.get('source')}", []
        if dst.exists():
            return f"ERROR: {project_mod.ATTEMPTS}/{name} exists already; pick a new name", []
        dst.mkdir(parents=True)
        copied = []
        for f in (script, "atlas_plan.json"):
            if (src / f).is_file():
                text = (src / f).read_text(encoding="utf-8")
                (dst / f).write_text(text, encoding="utf-8")
                copied.append(f"{f} ({text.count(chr(10)) + 1} lines)")
        return (f"copied {', '.join(copied)} to {project_mod.ATTEMPTS}/{name}/. Change it with edit_file, then evaluate "
                f"{project_mod.ATTEMPTS}/{name}/{script}."), []

    def t_list_dir(self, args: dict) -> tuple[str, list]:
        p = inside(self.root, args.get("path") or ".")
        if not p.is_dir():
            return f"ERROR: no such directory {args.get('path')}", []
        names = [q.name + ("/" if q.is_dir() else "") for q in sorted(p.iterdir())]
        more = f"\n...and {len(names) - MAX_LIST} more" if len(names) > MAX_LIST else ""
        return ("\n".join(names[:MAX_LIST]) + more) or "(empty)", []

    def t_view_image(self, args: dict) -> tuple[str, list]:
        paths = args.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        shown, missing = [], []
        for raw in paths[:MAX_VIEW]:
            p = inside(self.root, str(raw))
            if p.is_file() and p.suffix.lower() in project_mod.IMAGE_SUFFIXES:
                shown.append(mcp_server.Picture(str(p), p.relative_to(self.root).as_posix()))
            else:
                missing.append(str(raw))
        text = f"{len(shown)} picture(s) follow." + (f" Not pictures or not found: {missing}" if missing else "")
        if len(paths) > MAX_VIEW:
            text += f" Only the first {MAX_VIEW} are shown."
        return text, shown

    def t_done(self, args: dict) -> tuple[str, list]:
        left = self.budget - self.rep.evaluations
        if left > 0 and not self.closing and self.patience and not self.rep.stopped:
            return (f"NOT YET: the task is to keep going until {self.patience} evaluated attempts in a row have "
                    f"{gained(self.margin)}, and the last evaluation said KEEP GOING. Make the next one now: a new "
                    "attempt directory that fixes the largest error the last evaluation showed."), []
        if left > 0 and not self.closing and not self.rep.stopped:
            return (f"NOT YET: {left} of the {self.budget} evaluated attempts are unused, and the task is exactly "
                    f"{self.budget}. Make the next one now: a new attempt directory that fixes the largest error the "
                    "last evaluation showed."), []
        self.rep.done = True
        self.rep.summary = str(args.get("summary", ""))[:4000]
        return "ok", []


# -- the conversation ------------------------------------------------------------------------------------------------

def image_part(pic: mcp_server.Picture) -> dict:
    data = base64.b64encode(pic.jpeg(IMAGE_MAX_DIM)).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}


def picture_parts(pictures: list, root: Path) -> list[dict]:
    """Each picture after the line that says what it is and where, the path relative to the project like every other
    path the model is given."""
    out: list[dict] = []
    for pic in pictures:
        path = Path(pic.path)
        where = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        out += [{"type": "text", "text": f"{pic.caption}: {where}" if pic.caption else where}, image_part(pic)]
    return out


def prune_images(messages: list[dict], keep: int = KEEP_IMAGE_TURNS) -> int:
    """Replace the pictures of all but the last `keep` picture-bearing user messages with a note; the opening message
    (the brief and its references) is never pruned. Idempotent. -> pictures dropped."""
    carrying = [i for i, m in enumerate(messages) if i > 1 and m.get("role") == "user" and isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"])]
    dropped = 0
    for i in carrying[:-keep] if keep else carrying:
        new = []
        for c in messages[i]["content"]:
            if c.get("type") == "image_url":
                c = {"type": "text", "text": DROPPED}
                dropped += 1
            new.append(c)
        messages[i]["content"] = new
    return dropped


def pictures_in(message: dict) -> int:
    content = message.get("content")
    return sum(c.get("type") == "image_url" for c in content) if isinstance(content, list) else 0


def cap_images(messages: list[dict], limit: int = MAX_PICTURES) -> int:
    """With every turn's pictures kept, stay under the endpoint's limit on pictures in one request: the oldest turns'
    pictures become captions until at most `limit` are left. The references are never dropped. -> pictures dropped."""
    carrying = [i for i, m in enumerate(messages) if i > 1 and m.get("role") == "user" and pictures_in(m)]
    total, drop = sum(pictures_in(m) for m in messages), 0
    while total > limit and drop < len(carrying):
        total -= pictures_in(messages[carrying[drop]])
        drop += 1
    return prune_images(messages, keep=len(carrying) - drop) if drop else 0


# The two lines of colour_facts a script can copy, held to the renderer's own conversion by tests/test_brief.py.
HEX_CHANNELS = "r, g, b = (int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))"
TO_LINEAR = "c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4"


def colour_facts(project: project_mod.Project) -> str:
    'Deterministic stopping policy and evaluation feedback.'
    if not project.palette or project.spec.get("textured"):
        return ""
    return ("COLOUR: paint as the brief's script rules show, through the mesh's corner colour attribute "
            "(mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='CORNER')), and store each palette "
            f"colour LINEAR. For a hex colour h, with its '#' or without: {HEX_CHANNELS}, and each channel c is stored "
            f"as {TO_LINEAR}. A bmesh layer made with bm.loops.layers.color is a byte layer that holds "
            "sRGB values: linear values painted through it come out far too dark, so for linear values use the mesh "
            "attribute or bm.loops.layers.float_color. Stored that way the conversion is settled, not something to "
            "tune. When the feedback says a paint of yours reads as another colour, either the paint is in the wrong "
            "place or the light has shifted how a correctly stored colour reads: raise the zones by moving paint to "
            "where the brief's zone maps put each colour. But zones under 0.2, with most of your paints reading as "
            "other colours, mean the stored values themselves are wrong: work one palette colour through your own "
            "helper by hand and hold it against the two lines above (a slice one character off is the usual cause). "
            "Never answer it by painting a part in another palette colour than its own.")


def gained(margin: float) -> str:
    """The stopping rule's test as `mesh_jig.streak` words it, to follow "have"."""
    return f"not gained more than {margin:g} on the best numeric score" if margin else "not improved on the best numeric score"


def system_prompt(project: project_mod.Project, attempts: int, max_turns: int, judge: bool, patience: int = 0,
                  margin: float = streak_mod.MARGIN, max_previews: int = 2, require_evaluation: bool = True) -> str:
    '`patience` 0: exactly `attempts` evaluated attempts.'
    return "\n\n".join(x for x in [
        "You build ONE low-poly 3D model as a Blender Python script and improve it against reference views. mesh-jig "
        "builds the script in Blender, holds it to a contract, renders it on the CPU from fixed cameras and measures "
        "the renders against the reference views. Evaluate every attempt and use the configured numeric score "
        "alongside visual comparison to decide what to change next.",
        "The project is fixed. Every path you pass is relative to its directory. You may write only inside "
        "attempts/<name>/. Where the brief names a `mesh-jig` command, call "
        "the tool for the same job instead (`mesh-jig eval` is `evaluate`).",
        "THE LOOP:\n"
        "1. Read the brief and reference views. Plan briefly, then write the first complete model.\n"
        "2. Start later attempts by copying the BEST measured attempt and editing one idea. Measured attempts "
        "are immutable. Read focused excerpts with read_file(start_line, end_line) and locate code with search_file.\n"
        "3. Build and preview every scored view (leave `views` out). Compare the configured numeric score and its components, "
        "not overlap alone; inspect the comparison pictures and explicit acceptance failures.\n"
        "4. Refine within the configured preview budget, then evaluate. "
        + ("Evaluate even a losing preview; a loss is evidence for the plateau rule.\n" if require_evaluation else "\n") +
        "5. Record what changed and use the measured best as the next seed. A score or observation alone "
        "does not establish visual completion.",
        (f"PREVIEW POLICY: at most {max_previews} successful previews per candidate, across restarts."
         if max_previews else "PREVIEW POLICY: no preview cap is configured.")
        + (" Evaluate the pending previewed candidate before working on another; repair failed evaluations."
           if require_evaluation else " The pending-candidate restriction is disabled."),
        # which views count, which outline table there is and what a baseline gets right are the project's, not
        # the loop's: brief.reading_facts writes them into the brief the opening message carries
        "READING THE NUMBERS: the brief's HOW THIS PROJECT IS MEASURED section says which views the score is taken "
        "over, which outline table this project has and how a row of it reads, and what the baseline should get right "
        "first. It differs from subject to subject: go by it, not by what another subject would need.",
        "Where the brief has an outline table: a station is the whole silhouette at that position, and its two ends "
        "are whatever reaches furthest there, often a part that sticks out (a limb, a fin, a barrel, anything held or "
        "mounted) and not the trunk. Before you size a part from a station, find in the reference view of that name "
        "which part reaches each end; where something else crosses the station, take the trunk's size from the "
        "picture.",
        colour_facts(project),
        "WHEN A RESULT SURPRISES YOU: the builder, the renderer and the measures are the same for every attempt and "
        "are not what is wrong. Look for the cause in your own script: a helper's arithmetic, an index, a sign, an "
        "axis. Do not write test objects to probe the harness. An evaluation is "
        + ("an attempt, counted toward the end of the run," if patience else f"one of your {attempts} attempts")
        + " and is for a whole model meant to beat your best, never for a test.",
        "A build failure (JIG_BUILD_ERROR with the script line) does not use up an attempt: fix the script and evaluate "
        "it again.",
        "`build` checks a script without rendering. `preview` is a free look at an attempt that is built: it draws it "
        "beside the reference and gives each view's overlap and the outline diffs, writes nothing to the ledger and is "
        "not an attempt.",
        ("`judge` (and `evaluate` with judge_against) asks a paid vision model which of two attempts is closer to the "
         "reference. Ask only after the numbers have moved, against your last judged attempt or the baseline, and treat "
         "only a DECISIVE verdict as a result.") if judge else "",
        (f"BUDGET: there is no fixed number of attempts. Keep making evaluated attempts (a failed build does not count "
         "as one), each in its own attempts/<name>/ directory, improving on the measurements and on what you see in "
         f"the comparison pictures each time, until {patience} in a row have {gained(margin)} before them, or "
         f"{attempts} are made; you have {max_turns} turns for them. Do not count it yourself: every evaluation ends "
         "with the count and a line that starts KEEP GOING or STOP. On KEEP GOING make the next attempt. On STOP make "
         "no more, and call done(summary) naming your best attempt, its numbers and what is still wrong. Each turn "
         "re-sends the whole conversation: batch independent calls in one turn, and keep the script under 400 lines "
         "(helper functions beat repeated literals).") if patience else
        f"BUDGET: make exactly {attempts} evaluated attempts (a failed build does not count as one), each in its own "
        f"attempts/<name>/ directory, improving on the measurements and on what you see in the comparison pictures each "
        f"time; you have {max_turns} turns for them. Each turn re-sends the whole conversation: batch independent calls "
        "in one turn, and keep the script under 400 lines (helper functions beat repeated literals). When the "
        f"{attempts} are done, call done(summary) naming your best attempt, its numbers and what is still wrong.",
        # what two of the first runs with no cap did, which a five-attempt run has no time for: one traded two gates
        # away for score without seeing the gate lines, one painted the zone map onto the mesh for 0.09
        ("A LONG RUN: read the `gates:` line of every count. The score does not include the gates, so an attempt that "
         "scores higher and fails a gate the best one passed is a trade, not a gain: undo what broke the gate before "
         "going on. And paint parts, not cells: keep each part the palette colour the brief gives it. Colouring faces "
         "to follow the reference's colour zones cell by cell, whatever part they belong to, raises the colour "
         "measure and makes a model that looks like camouflage. When the score went up and the comparison picture "
         "looks worse, the picture is right.") if patience else "",
        # t02 sent the whole script seven times in two attempts (a third of its conversation) and ended there, at the
        # server's 131,072 tokens; in t01 and t02 the attempt that was a rewrite scored below the one before it
        "WRITE THE SCRIPT ONCE: everything you send stays in the conversation, and a run whose conversation outgrows "
        "the model's context ends there with its attempts unused. write_file the script for the first attempt only. "
        "Every later change, a fix before the first evaluation included, is an edit_file of the lines that change, and "
        "the next attempt is copy_attempt of your best one with one part changed: that also keeps what the last "
        "evaluation showed was right, which a rewrite throws away. Read a script back only when an edit did not find "
        "its lines.",
        # t04 reached the context after three attempts: an 18-edit call of 8,500 characters failed whole on one `old`
        # and was sent three times, and the script was read back four times (47,000 characters). "A few edits a call"
        # was not held to (t07: 34 calls, 74,000 characters, two of them failing whole, five read-backs), so it is a
        # number
        "KEEP EDITS SMALL: an edit_file call fails whole when one `old` is not found, and a long call sent again "
        "costs its whole length again. At most four edits a call, each `old` one or two lines copied exactly from the "
        "file as it now is, without its trailing comment. Put the numbers you will tune (sizes, positions, angles) in named constants or one table "
        "near the top of the script, so that a change is a short line and not a block of geometry.",
    ] if x)


def opening(project: project_mod.Project, attempts: int = 0, patience: int = 0) -> list[dict]:
    """The brief, the ledger so far, and every reference view: what `brief` returns, plus the agent's memory."""
    reply = mcp_server.brief(str(project.root))
    text = "\n".join(x for x in reply if isinstance(x, str))
    for root in (str(project.root), project.root.as_posix()):       # as Tools.relative: the model passes relative paths
        text = text.replace(root + "\\", "").replace(root + "/", "").replace(root, ".")
    if project.ledger.is_file():
        text += "\n\nTHE LEDGER SO FAR (earlier attempts in this project):\n" + clip(project.ledger.read_text(encoding="utf-8"))
    parts = [{"type": "text", "text": text}]
    pics = [x for x in reply if isinstance(x, mcp_server.Picture)]
    if pics:
        # the first live run (deepseek-v4.1-flash) spent its first turn view_image-ing these
        parts.append({"type": "text", "text": "THE REFERENCE VIEWS (already in front of you here; do not view_image "
                                              "them again):"})
    parts += picture_parts(pics, project.root)
    # the task in the user's turn and in the words the Claude Code runs were given
    task = (f"Make exactly {attempts} successfully evaluated attempts (a failed build does not count as one), each in "
            "its own attempts/<name>/ directory, improving on the measurements and on what you see in the comparison "
            f"pictures each time. When the {attempts} are done, stop and report which attempt is best, its numbers, and "
            "what is still wrong. " if attempts else "")
    if attempts and patience:
        task = ("Make successfully evaluated attempts (a failed build does not count as one), each in its own "
                "attempts/<name>/ directory, improving on the measurements and on what you see in the comparison "
                f"pictures each time. Keep going until {patience} attempts in a row show no improvement: every "
                f"evaluation says KEEP GOING or STOP. Stop early at: {attempts} attempts. When you stop, report which "
                "attempt is best, its numbers, and what is still wrong. ")
    parts.append({"type": "text", "text": task + "Start: write your first attempt, build it and preview it."})
    return parts


def strip_images(messages: list[dict]) -> list[dict]:
    """The conversation with every picture's bytes replaced by a marker, for the transcript."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            m = dict(m, content=[{"type": "image"} if c.get("type") == "image_url" else c for c in m["content"]])
        out.append(m)
    return out


def run_dir(project: project_mod.Project, model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return project.root / RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}"


def write_run(out: Path, rep: Report, messages: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(asdict(rep), indent=1) + "\n", encoding="utf-8")
    with open(out / "transcript.jsonl", "w", encoding="utf-8") as f:
        for m in strip_images(messages):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def over_cost(spent: float, dearest: float, max_cost: float) -> str:
    """Why the next model call may not be made under a cost cap, or "". The cost of a call is only known once it is
    paid for, so the run stops while the dearest call so far still fits under the cap: the cap is then crossed only
    by a call dearer than every one before it."""
    if not max_cost or spent + dearest < max_cost:
        return ""
    return (f"stopped at the cost cap: ${spent:.4f} spent of ${max_cost:.2f}, and one call has cost up to "
            f"${dearest:.4f}")


def over_wall(elapsed: float, max_wall: float) -> str:
    """Why the next model call may not be made under a time budget, or ""."""
    if not max_wall or elapsed < max_wall:
        return ""
    return f"stopped at the time budget: {elapsed:.0f} s of {max_wall:.0f} s used"


def run(project: project_mod.Project, chat_fn: ChatFn, *, attempts: int = 5, max_turns: int = 0, judge: bool = False,
        patience: int = 0, margin: float = streak_mod.MARGIN, model: str = "", url: str = "", out: Path | None = None, max_cost: float = 0.0, max_wall: float = 0.0,
        keep_images: int | None = KEEP_IMAGE_TURNS, max_previews: int = 2, require_evaluation: bool = True,
        context_tokens: int = 0, reply_reserve: int = 8192, log: Callable[[str], None] = lambda s: None,
        clock: Callable[[], float] = time.time) -> Report:
    """Drive the model until it calls done(), the turn cap, the cost cap, the time budget, or a chat failure. Writes
    report.json and transcript.jsonl to `out` after every turn, so a killed run still leaves what it spent.

    `max_cost` (USD, 0 for none) is held against the cost the endpoint reports with each reply's usage: see
    `over_cost`. An endpoint that reports no cost cannot be held to a cap, so the run ends after its first call.
    `max_wall` (seconds, 0 for none) is one clock for the whole run, tools included. Each model call is handed what
    is left of it (`chat_fn(messages, tools, left_s)`), so a slow call is cut when the run's time is up and not
    before, and the run ends there with the attempts it measured.
    `patience` (0 for none) makes `attempts` a ceiling: the run goes on until that many evaluated attempts in a row
    have not gained more than `margin` (`mesh_jig.streak`, whose count every evaluation's result then carries), and
    an evaluate after that is refused. `Report.stopped` says so when the rule ended the run.
    `keep_images`: how many picture-bearing turns keep their pictures; None keeps them all (see `cap_images`), which
    is what a marked prompt cache needs."""
    if context_tokens < 0 or reply_reserve < 0 or (context_tokens and reply_reserve >= context_tokens):
        raise ValueError("context_tokens must exceed reply_reserve; 0 disables context budgeting")
    max_turns = max_turns or TURNS_PER_ATTEMPT * attempts
    rep = Report(project=project.name, model=model, url=url, max_cost_usd=max_cost, max_wall_s=max_wall,
                 patience=patience, margin=margin if patience else 0.0)
    tools = Tools(project, rep, attempts=attempts, judge=judge, patience=patience, margin=margin,
                  max_previews=max_previews, require_evaluation=require_evaluation)
    schemas = tool_schemas(judge)
    messages: list[dict] = [{"role": "system",
                             "content": system_prompt(project, attempts, max_turns, judge, patience, margin, max_previews, require_evaluation)},
                            {"role": "user", "content": opening(project, attempts, patience)}]
    if tools.rep.best or tools.lifecycle.pending:
        messages.append({"role": "user", "content": tools.checkpoint()})
    transcript = list(messages)
    t0 = clock()
    nudged = False
    dearest = 0.0
    try:
        while rep.turns < max_turns:
            rep.error = over_cost(rep.cost_usd, dearest, max_cost) or over_wall(clock() - t0, max_wall)
            if rep.error:
                break
            rep.turns += 1
            tools.closing = max_turns - rep.turns < TURNS_LEFT_NOTE
            if keep_images is None:
                cap_images(messages)
            else:
                prune_images(messages, keep_images)
            try:
                messages, compacted = agent_state.compact(messages, schemas, tools.checkpoint(), context_tokens, reply_reserve)
            except ValueError as e:
                rep.error = str(e)
                break
            if compacted:
                rep.context_compactions += 1
                log("context compacted; saved candidate and global best retained")
            before_reply = len(messages)
            log(f"turn {rep.turns}: waiting on the model (+{clock() - t0:.0f}s)")
            msg, usage = (chat_fn(messages, schemas, max_wall - (clock() - t0)) if max_wall
                          else chat_fn(messages, schemas))
            cost = rep.count(usage)
            dearest = max(dearest, cost)
            if max_cost and cost <= 0:
                rep.error = ("stopped: the endpoint reported no cost for the call, so the cost cap "
                             f"(${max_cost:.2f}) cannot be held")
                break
            msg = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_details")}
            msg["role"] = "assistant"
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                if nudged:
                    rep.summary = str(msg.get("content") or "")[:4000]
                    rep.error = "stopped without done()"
                    transcript.extend(messages[before_reply:])
                    break
                nudged = True
                messages.append({"role": "user", "content": "Use the tools. Call done(summary) when you are finished."})
                transcript.extend(messages[before_reply:])
                continue
            shown: list = []
            for tc in calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                rep.tool_calls += 1
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                    log(f"  {name}({json.dumps(args)[:160]})")
                    result, pics = tools.call(name, args)
                    shown += pics
                except Rejected as e:
                    rep.rejected_calls += 1
                    result = f"REJECTED: {e}"
                except (json.JSONDecodeError, ValueError) as e:
                    result = f"ERROR: bad arguments: {e}"
                except Exception as e:  # a tool bug: tell the model, keep going
                    result = f"ERROR: {type(e).__name__}: {e}"
                log(f"  -> {result[:200]!r}")
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
                if rep.done:
                    break
            left = max_turns - rep.turns
            if not rep.done and 0 < left <= TURNS_LEFT_NOTE and messages[-1]["role"] == "tool":
                messages[-1]["content"] += (f"\n[{left} turn{'s' if left != 1 else ''} left. Make sure your best attempt "
                                            "is evaluated, then call done(summary).]")
            if shown and not rep.done:
                messages.append({"role": "user", "content": picture_parts(shown, project.root)})
            transcript.extend(messages[before_reply:])
            rep.wall_s = round(clock() - t0, 1)
            if out is not None:
                write_run(out, rep, transcript)
            if rep.done:
                break
        else:
            rep.error = f"hit the turn cap ({max_turns}) without done()"
    except llm.OutOfTime as e:
        rep.error = f"stopped at the time budget ({max_wall:.0f} s) during a model call: {e}"
    except llm.LLMError as e:
        rep.error = f"chat failed: {e}"
    rep.wall_s = round(clock() - t0, 1)
    if out is not None:
        write_run(out, rep, transcript)
    return rep


def chat_for(backend: llm.Backend, *, effort: str | None = None, max_tokens: int | None = None,
             idle_s: float = llm.CHAT_IDLE_S, cache: bool = False, log: Callable[[str], None] = lambda s: None) -> ChatFn:
    """The chat function over the package's urllib client: no temperature sent (reasoning models refuse one), no
    max_tokens unless given, the effort in the endpoint's form. Streamed, so a call with no chunk for `idle_s` is
    dropped and retried instead of holding the run for the whole of `backend.timeout_s`. `cache` sends the
    conversation with `llm.cache_marks` (run with keep_images=None, or the marks are wasted). `left_s` is what a run
    under a time budget has left: the call and its retries get that long and no longer."""
    extra = llm.effort_extra(backend, effort)

    def chat(messages: list[dict], tools: list[dict], left_s: float | None = None) -> tuple[dict, dict]:
        if cache:
            messages = llm.cache_marks(messages)
        out = llm.chat_stream(backend, messages, max_tokens=max_tokens, tools=tools, extra=extra, idle_s=idle_s, log=log,
                              budget_s=left_s)
        return out["message"], out["usage"]
    return chat


def summary(rep: Report) -> str:
    best = (f"best {rep.best['name']} (numeric score {rep.best['numeric_score']:.3f})"
            if rep.best and rep.best.get("numeric_score") is not None else "no measured attempt")
    lines = [f"agent {rep.model}: {'done' if rep.done else 'NOT done'} after {rep.turns} turns, {rep.evaluations} "
             f"evaluated attempt(s), {best}; {rep.prompt_tokens}+{rep.completion_tokens} tokens"
             + (f" ({rep.cached_tokens} read from the cache, {rep.cache_write_tokens} written to it)"
                if rep.cached_tokens or rep.cache_write_tokens else "")
             + (f", ${rep.cost_usd:.4f}" if rep.cost_usd else "") + f", {rep.wall_s:.0f} s"]
    if rep.error:
        lines.append("error: " + rep.error)
    if rep.summary:
        lines.append(rep.summary)
    return "\n".join(lines)


__all__ = ["run", "chat_for", "tool_schemas", "Tools", "Report", "prune_images", "cap_images", "edit_text", "summary"]
