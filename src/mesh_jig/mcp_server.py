"""mesh-jig as an MCP server: the same tools as the CLI, with the pictures returned inline so the agent sees what it
built in the same turn it built it.

    mesh-jig-mcp                      # stdio; needs the `mcp` extra: pip install -e ".[mcp]" in the checkout

Tools: brief, evaluate, build, preview, paint, judge. Each is a plain function below (`TOOLS`) that returns text and
`Picture`s; `serve` registers them and turns the pictures into image content. They run on the machine the server
runs on, so every path is a local path.
"""

from __future__ import annotations

import functools
import io
import json
from dataclasses import dataclass
from pathlib import Path

from . import __version__, brief as brief_mod, build as build_mod, evaluate as evaluate_mod, glb, judge as judge_mod, llm, paint as paint_mod
from . import project as project_mod, render, silhouette

INSTRUCTIONS = (
    "mesh-jig builds, renders and measures a Blender model script against reference views. Call `brief` first: it "
    "returns the contract, the reference's measured outlines and colour zones, and the reference views. Then write "
    "<project>/attempts/<name>/<script> yourself (one idea per attempt) and call `evaluate` on it: the numbers are "
    "deterministic and free, so evaluate every attempt and read the ledger back. `judge` is a paid model call: use it "
    "only once the numbers have moved, and treat only a DECISIVE verdict as a result. When you report, ask the user "
    "whether they want to see the results in the dashboard (`mesh-jig view <project> --open`, run in a shell). It "
    "serves a local page of every attempt beside the reference until it is stopped, so it is theirs to run, or yours "
    "to start in the background on a yes."
)
PICTURE_MAX_DIM = 1024
PICTURE_QUALITY = 85


@dataclass
class Picture:
    """An image file to show the agent, with the line that says what it is."""
    path: str
    caption: str = ""

    def jpeg(self, max_dim: int = PICTURE_MAX_DIM) -> bytes:
        from PIL import Image
        im = Image.open(self.path).convert("RGB")
        if max(im.size) > max_dim:
            k = max_dim / max(im.size)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=PICTURE_QUALITY)
        return buf.getvalue()


def _pictures(paths: dict[str, str], what: str) -> list:
    out: list = []
    for view, path in paths.items():
        out += [f"{what} {view}: {path}", Picture(path, f"{what} {view}")]
    return out


def _attempt(script: str) -> Path:
    return Path(script).resolve().parent


def brief(project: str) -> list:
    """Read a project before writing anything: what the model must show, how a script is built and what the contract
    checks, the reference's outlines and colour zones as numbers, and the reference views themselves.

    project: the project directory (the one holding jig.json)."""
    proj = project_mod.load(project)
    issues = proj.problems()
    head = ("PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n") if issues else ""
    return [head + brief_mod.text(proj)] + _pictures(proj.references, "reference")


def evaluate(project: str, script: str, note: str = "", name: str = "", paint_plan: str = "", atlas: str = "",
             judge_against: str = "", samples: int = 16) -> list:
    """Build a model script in Blender, hold it to the contract, render it and measure it against the reference views;
    append one line to the project's ledger. Returns the numbers, where the outline is off, and each view beside its
    reference. Deterministic and free unless judge_against is given.

    project: the project directory. script: the model script, in its own attempt directory
    (<project>/attempts/<name>/model.py). note: what this attempt changed, for the ledger. paint_plan / atlas: a
    textured contract's atlas plan (JSON) or a ready PNG. judge_against: another attempt directory to also ask the
    pairwise judge about (a paid model call; leave empty unless the numbers have moved)."""
    proj = project_mod.load(project)
    issues = proj.problems()
    if issues:
        return ["PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues)]
    backend = llm.resolve_backend() if judge_against else None
    out = _attempt(script)
    rec = evaluate_mod.evaluate(proj, Path(script).resolve(), out, atlas=Path(atlas).resolve() if atlas else None,
                                paint_plan=Path(paint_plan).resolve() if paint_plan else None, name=name, note=note,
                                against=Path(judge_against).resolve() if judge_against else None, backend=backend,
                                samples=samples, log=lambda *_: None)
    evaluate_mod.append_ledger(proj.ledger, rec)
    reply: list = [evaluate_mod.feedback(rec, evaluate_mod.standing(proj, out, rec))
                   + f"\n-> {out / evaluate_mod.EVAL_FILE}; ledger {proj.ledger}"]
    if rec.get("ok"):
        reply += _pictures(rec.get("compare") or {}, "reference | your build,")
        others = {v: p for v, p in rec["renders"].items() if v not in rec["compare"]}
        reply += _pictures(others, "your build (no reference view),")
    return reply


def build(project: str, script: str, paint_plan: str = "", atlas: str = "") -> list:
    """Build a model script and run the contract check only: a quick answer to "does it build and fit the contract"
    with no render and no ledger line.

    project: the project directory. script: the model script in its attempt directory."""
    proj = project_mod.load(project)
    out = _attempt(script) / "out"
    bound = Path(atlas).resolve() if atlas else None
    if paint_plan:
        bound = out / glb.atlas_name(proj.spec)
        err = paint_mod.paint_file(paint_plan, bound, proj.palette)
        if err:
            return ["BUILD FAILED:\n" + err]
    err = build_mod.build_variants(script, out, proj.spec, **({"atlas": bound} if bound else {}),
                                   **({"palette": proj.palette} if glb.texture_from(proj.spec) == "script" else {}))
    if err:
        return ["BUILD FAILED:\n" + err]
    report = json.loads((out / glb.REPORT).read_text(encoding="utf-8"))
    lines = [f"{n}: {s['triangles']} triangles, size x/y/z {s['size']} m, {s['distinct_colors']} colours"
             for n, s in report["stats"].items()]
    return ["\n".join(lines + [f"built and in contract -> {out}"])]


def preview(project: str, attempt: str, views: list[str] | None = None) -> list:
    """Render and measure what an attempt has already built, beside the reference: a free look. It gives the numbers
    an evaluation of this build would give (overlap, colour zones, resemblance, the numeric score) and how the
    colours read, with no outline table and no gates. Nothing is added to the ledger and it is not an attempt.

    project: the project directory. attempt: the attempt directory (its out/ holds the GLBs). views: camera names,
    default all; leave it out to get a score."""
    proj = project_mod.load(project)
    rec = evaluate_mod.preview(proj, attempt, views)
    text = evaluate_mod.preview_text(proj, attempt, rec)
    if not rec["ok"]:
        return [text]
    others = {v: p for v, p in rec["renders"].items() if v not in rec["compare"]}
    return ([text] + _pictures(rec["compare"], "reference | your build,")
            + _pictures(others, "your build (no reference view),"))


def paint(project: str, plan: str, out: str) -> list:
    """Paint an atlas plan (JSON: size, background, ops) to a PNG with the project's palette, to see it before a build
    binds it.

    project: the project directory. plan: the plan file. out: the PNG to write."""
    proj = project_mod.load(project)
    err = paint_mod.paint_file(plan, out, proj.palette)
    return [err] if err else [f"painted {out}", Picture(out, "atlas")]


def judge(project: str, attempt_a: str, attempt_b: str = "", samples: int = 16) -> list:
    """Ask the pairwise judge whether attempt A is closer to the reference than attempt B. A paid model call: half the
    samples run with the two swapped, and only a DECISIVE verdict counts. With attempt_b empty, A is judged against
    itself, which measures the judge's own bias (the honest answer is a draw).

    project: the project directory. attempt_a, attempt_b: attempt directories that have been evaluated."""
    proj = project_mod.load(project)
    backend = llm.resolve_backend()
    first = Path(attempt_a).resolve()
    second = Path(attempt_b).resolve() if attempt_b else first
    out = first / (f"pairwise_vs_{second.name}" if attempt_b else "self_check")
    rec = evaluate_mod.pairwise(proj, evaluate_mod.renders_of(first), evaluate_mod.renders_of(second), out,
                                backend=backend, samples=samples)
    (out / "pairwise.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    text = judge_mod.summary(rec, first.name, second.name if attempt_b else "itself")
    if not attempt_b:
        text += ("\nA self-check is two copies of one image: the honest answer is a win rate near 0.5 and NOT decisive. "
                 "Anything else is the judge's own bias.")
    return [text + f"\n-> {out / 'pairwise.json'}"]


TOOLS = (brief, evaluate, build, preview, paint, judge)


def guarded(fn):
    """A tool whose bad request (no project, no judge configured, a missing file) comes back as a message to the
    agent instead of a protocol error."""
    @functools.wraps(fn)
    def run(*args, **kw):
        try:
            return fn(*args, **kw)
        except (project_mod.ProjectError, llm.LLMError, ValueError, OSError) as e:
            return [f"mesh-jig: {e}"]
    return run


def _sdk():
    """(server class, image class) from the `mcp` package: MCPServer in 2.x, FastMCP in 1.x."""
    try:
        from mcp.server.mcpserver import Image, MCPServer
        return MCPServer, Image
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP, Image
        return FastMCP, Image
    except ImportError:
        raise SystemExit('the MCP server needs the `mcp` package: run `pip install -e ".[mcp]"` in the mesh-jig '
                         'checkout') from None


def serve():
    """The server with every tool registered; `.run()` speaks MCP over stdio."""
    Server, Image = _sdk()
    try:
        server = Server("mesh-jig", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:                         # 1.x takes no version
        server = Server("mesh-jig", instructions=INSTRUCTIONS)

    def shown(fn):
        safe = guarded(fn)

        @functools.wraps(fn)
        def run(*args, **kw):
            return [Image(data=x.jpeg(), format="jpeg") if isinstance(x, Picture) else x for x in safe(*args, **kw)]
        return run

    for fn in TOOLS:
        try:
            server.tool(name=fn.__name__, description=fn.__doc__, structured_output=False)(shown(fn))
        except TypeError:                     # an older 1.x: no structured_output argument
            server.tool(name=fn.__name__, description=fn.__doc__)(shown(fn))
    return server


def main() -> None:
    serve().run()


if __name__ == "__main__":
    main()
