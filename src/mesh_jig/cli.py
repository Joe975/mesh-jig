"""mesh-jig: tools that build, render and measure a Blender model script against reference views.

    mesh-jig init <dir> [--name N] [--length 0.08] [--sheet sheet.png]    start a project
    mesh-jig sheet <project>                                              cut refs/ from the design sheet
    mesh-jig refs <project> [--region VIEW X0 Y0 X1 Y1] [--out DIR]       the reference views as the measures see them
    mesh-jig palette <project> [--colors 5] [--write]                     read a palette off the reference views
    mesh-jig brief <project>                                              the contract, outlines and zones, as text
    mesh-jig criteria <project> [--write]                                 what the reference is judged on, and where
                                                                          jig.json states it
    mesh-jig eval <project> --script <attempt>/model.py --note "..."      build + render + measure + ledger line
    mesh-jig show <project> <attempt>                                     what eval said of it, read back
    mesh-jig build <project> --script <attempt>/model.py                  build and contract check only
    mesh-jig check <project> <attempt>                                    the contract check alone
    mesh-jig paint <project> --plan plan.json --out atlas.png             paint an atlas plan
    mesh-jig preview <project> <attempt> [--views hero side]              a quick look at what is built
    mesh-jig judge <project> <attempt A> <attempt B> [--samples 16]       pairwise judge (a model call, costs money)
    mesh-jig streak <project> [--patience 3] [--margin 0.002]             for a run with no attempt cap: keep going, or stop
    mesh-jig agent <project> --model M [--attempts 5] [--judge]           an API model drives the loop (costs money)
    mesh-jig view [<dir> ...] [--skip NAME] [--port 8796] [--open]        a web page of every result, per reference
    mesh-jig view <dir> ... --export <dir> [--best]                       that page as files, for a static host
    mesh-jig doctor [<project>]                                           what is installed and configured

Exit codes: 0 done, 1 the attempt failed (a build error, a contract breach, an empty render), 2 the request itself
was wrong (no project, bad arguments, no judge configured).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, agent, brief as brief_mod, build, criteria, evaluate, glb, judge as judge_mod, llm, paint
from . import palette, project, references, render, sheet, silhouette, streak, viewer

PREVIEW_DIR = "preview"


def _attempt_dir(a) -> Path:
    return (Path(a.out) if a.out else Path(a.script).parent).resolve()


def _atlas_args(sub) -> None:
    sub.add_argument("--atlas", type=Path, help="textured contract: bind this PNG")
    sub.add_argument("--paint", type=Path, help="textured contract: paint this plan first, then bind it")


def cmd_init(a) -> int:
    for p in sheet.init(a.dir, name=a.name, length_m=a.length, sheet=a.sheet):
        print(p)
    print("next: put the reference views in refs/ (hero.png, top.png, side.png, rear.png)"
          + (", or tighten the crop boxes in jig.json and run `mesh-jig sheet`" if a.sheet else "")
          + ", fill in brief.md, state what the reference is judged on in jig.json's `criteria` block (`mesh-jig "
          "criteria` reads it back), then `mesh-jig brief` to read the contract back")
    return 0


def cmd_sheet(a) -> int:
    for p in sheet.write_refs(project.load(a.project)):
        print(p)
    return 0


def cmd_refs(a) -> int:
    proj = project.load(a.project)
    if not proj.references:
        print(f"no reference views under {proj.refs_dir}", file=sys.stderr)
        return 2
    regions = [(view, [float(v) for v in box]) for view, *box in a.region or []]
    print(references.text(proj, regions))
    if a.out:
        print("\nWHAT THE MASK TAKES AS THE SUBJECT (the rest washed out; red box, blue ground line, green centre line):")
        for p in references.write_seen(proj, a.out):
            print(p)
    return 0


def cmd_streak(a) -> int:
    proj = project.load(a.project)
    v = streak.verdict(proj.root, a.patience, a.margin, a.max_attempts)
    print(json.dumps(v, indent=1) if a.json else streak.text(v))
    return 0


def cmd_palette(a) -> int:
    proj = project.load(a.project)
    if a.refine is not None:
        return refine_palette(proj, a.refine, a.write)
    found = palette.suggest(proj.references, a.colors)
    if not found:
        print("no reference views to read a palette from", file=sys.stderr)
        return 2
    print(json.dumps(found, indent=2))
    if a.write:
        if proj.palette:
            print("jig.json already has a palette; not overwritten", file=sys.stderr)
            return 2
        data = dict(proj.data, palette=found)
        (proj.root / project.JIG_FILE).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"written to {proj.root / project.JIG_FILE}: rename c1..c{len(found)} to what the colours are")
    return 0


def refine_palette(proj: project.Project, names: list[str], write: bool) -> int:
    """`palette --refine [NAME ...]`: what each colour of the project's palette would move to, read off the scored
    reference views where they read as it (`palette.refine`). Only the named colours are moved and written: a colour
    that covers little of the subject (an eye, a buckle, a nose) is read off the shaded cells of its neighbours, so
    each is for a person to look at before it moves."""
    if not proj.palette:
        print("jig.json has no palette to refine: `mesh-jig palette <project> --write` reads one first", file=sys.stderr)
        return 2
    unknown = [n for n in names if n not in proj.palette]
    if unknown:
        print(f"no palette colour named {unknown}; the palette has {list(proj.palette)}", file=sys.stderr)
        return 2
    if write and not names:
        print("name the colours to move: `--refine orange red --write`. Without names this prints what each would "
              "move to and writes nothing.", file=sys.stderr)
        return 2
    scored, _looked = brief_mod.scored_views(proj)
    refs = {v: p for v, p in proj.references.items() if v in scored} or proj.references
    found, cells = palette.refine(refs, proj.palette)
    total = sum(cells.values()) or 1
    moved = dict(proj.palette)
    for name, was in proj.palette.items():
        few = cells[name] < palette.REFINE_MIN_CELLS
        said = "too few cells: kept" if few else "moved" if name in names else "not named: kept" if names else "would move"
        print(f"{name}: {was} -> {found[name]}, read off {cells[name]} cells ({cells[name] / total:.0%} of the "
              f"subject); {said}")
        if name in names:
            moved[name] = found[name]
    if not names:
        print("Nothing is written. Look at each colour beside the reference, then name the ones to move: a colour that "
              "covers little of the subject is read off its neighbours' shading and is better left or set by hand.")
    if write:
        data = dict(proj.data, palette=moved)
        (proj.root / project.JIG_FILE).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"written to {proj.root / project.JIG_FILE}. The zones are measured against the palette, so attempts "
              "measured before this are not comparable with attempts measured after it.")
    return 0


def state_views_hint(proj: project.Project) -> str:
    """How to state the scored views, for a project that is on the fallback; "" for one that states them."""
    if proj.scoring_weights is not None:
        return ""
    return (f"The scored views are the fallback. To state them: `mesh-jig criteria {proj.root} --write`, or add "
            '"criteria": {"views": {"<view>": 1, ...}} to jig.json (docs/JIG.md, Criteria).')


def cmd_criteria(a) -> int:
    proj = project.load(a.project)
    issues = proj.problems()
    if issues:
        print("PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues), file=sys.stderr)
        return 2
    print(brief_mod.criteria_facts(proj))
    stated = criteria.rewrite(proj.data, brief_mod.scored_views(proj)[0])
    if stated == proj.data:
        print("jig.json states all of it in its `criteria` block.")
        return 0
    if a.write:
        (proj.root / project.JIG_FILE).write_text(json.dumps(stated, indent=2) + "\n", encoding="utf-8")
        print(f"written to {proj.root / project.JIG_FILE}: the criteria above are now its `criteria` block. The same "
              "views, table, gates and review are in force, so no number moves.")
        return 0
    print("\nAs a `criteria` block (--write puts it in jig.json in place of the older fields; no number moves):")
    print(json.dumps({"criteria": stated["criteria"]}, indent=2))
    return 0


def cmd_brief(a) -> int:
    proj = project.load(a.project)
    issues = proj.problems()
    if issues:
        print("PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues) + "\n")
    print(brief_mod.text(proj))
    return 0


def _painted_atlas(proj: project.Project, a, out: Path) -> tuple[Path | None, str]:
    """The atlas to bind for a build: the one given, or the plan painted into out/. (atlas, error)."""
    if a.paint:
        atlas = out / glb.atlas_name(proj.spec)
        return atlas, paint.paint_file(a.paint, atlas, proj.palette)
    return (a.atlas.resolve() if a.atlas else None), ""


def cmd_build(a) -> int:
    proj = project.load(a.project)
    out = _attempt_dir(a) / "out"
    atlas, err = _painted_atlas(proj, a, out)
    err = err or build.build_variants(a.script, out, proj.spec, **({"atlas": atlas} if atlas else {}),
                                      **({"palette": proj.palette} if glb.texture_from(proj.spec) == "script" else {}))
    if err:
        print("BUILD FAILED:\n" + err)
        return 1
    report = json.loads((out / glb.REPORT).read_text(encoding="utf-8"))
    for name, s in report["stats"].items():
        print(f"{name}: {s['triangles']} triangles, size x/y/z {s['size']} m, {s['distinct_colors']} colours")
    print(f"built and in contract -> {out}")
    return 0


def cmd_check(a) -> int:
    proj = project.load(a.project)
    out = Path(a.attempt)
    out = out / "out" if (out / "out").is_dir() else out
    report = glb.write_report(out, proj.spec)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


def cmd_paint(a) -> int:
    proj = project.load(a.project)
    err = paint.paint_file(a.plan, a.out, proj.palette)
    print(err or f"painted {a.out}")
    return 1 if err else 0


def cmd_preview(a) -> int:
    proj = project.load(a.project)
    rec = evaluate.preview(proj, a.attempt, a.views or None)
    print(evaluate.preview_text(proj, a.attempt, rec))
    if not rec["ok"]:
        return 1
    for view, path in rec["renders"].items():
        if view not in rec["compare"]:
            print(f"{view} (no reference view): {path}")
    return 0


def cmd_eval(a) -> int:
    proj = project.load(a.project)
    issues = proj.problems()
    if issues:
        print("PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues), file=sys.stderr)
        return 2
    if a.atlas and a.paint:
        print("give --atlas or --paint, not both", file=sys.stderr)
        return 2
    backend = None
    if a.judge:
        if not a.against:
            print("--judge needs --against <another attempt>", file=sys.stderr)
            return 2
        try:
            backend = llm.resolve_backend()
        except llm.LLMError as e:
            print(str(e), file=sys.stderr)
            return 2
    out = _attempt_dir(a)
    rec = evaluate.evaluate(proj, Path(a.script).resolve(), out, atlas=a.atlas.resolve() if a.atlas else None,
                            paint_plan=a.paint.resolve() if a.paint else None, name=a.name, note=a.note,
                            against=Path(a.against).resolve() if a.against else None, against_name=a.against_name,
                            backend=backend, samples=a.samples, log=lambda *_: None)
    if not a.no_ledger:
        evaluate.append_ledger(Path(a.ledger).resolve() if a.ledger else proj.ledger, rec)
    print(evaluate.feedback(rec, evaluate.standing(proj, out, rec)))
    print(f"-> {out / evaluate.EVAL_FILE}")
    return 0 if rec.get("ok") else 1


def cmd_show(a) -> int:
    """What `eval` said of an attempt, read back from its eval.json: nothing is built, measured or added to the
    ledger. An evaluation run a second time to read a part of its output again leaves a second ledger line."""
    proj = project.load(a.project)
    attempt = a.attempt if a.attempt.is_dir() else proj.attempts_dir / a.attempt
    if not (attempt / evaluate.EVAL_FILE).is_file():
        print(f"no {evaluate.EVAL_FILE} under {a.attempt}: give an attempt directory, or its name under "
              f"{proj.attempts_dir}, that `mesh-jig eval` has measured", file=sys.stderr)
        return 2
    attempt = attempt.resolve()
    rec = evaluate.read(attempt)
    print(evaluate.feedback(rec, evaluate.standing(proj, attempt, rec)))
    print(f"-> {attempt / evaluate.EVAL_FILE}")
    return 0


def cmd_judge(a) -> int:
    proj = project.load(a.project)
    if not a.self_check and not a.b:
        print("give two attempts, or one with --self-check", file=sys.stderr)
        return 2
    try:
        backend = llm.resolve_backend()
    except llm.LLMError as e:
        print(str(e), file=sys.stderr)
        return 2
    first = Path(a.a).resolve()
    second = first if a.self_check else Path(a.b).resolve()
    out = Path(a.out).resolve() if a.out else first / ("self_check" if a.self_check else f"pairwise_vs_{second.name}")
    try:
        rec = evaluate.pairwise(proj, evaluate.renders_of(first), evaluate.renders_of(second), out, backend=backend,
                                samples=a.samples)
    except ValueError as e:
        print(f"JUDGE FAILED: {e}")
        return 1
    (out / "pairwise.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(judge_mod.summary(rec, first.name, "itself" if a.self_check else second.name))
    if a.self_check:
        print("a self-check is two copies of one image: the honest answer is a win rate near 0.5, mostly ties and NOT "
              "decisive. Anything else is the judge's own bias; do not trust its other verdicts until this passes.")
    print(f"-> {out / 'pairwise.json'}")
    return 0


def call_cap(timeout: float | None, max_wall: float) -> float:
    """The cap on one model call. Under a time budget a call may take what the run has left: cutting a call that is
    still streaming at 900 s only starts it again from nothing."""
    return timeout if timeout is not None else max_wall or 900.0


def cmd_agent(a) -> int:
    if a.max_previews < 0 or a.context_tokens < 0 or a.reply_reserve <= 0 or (a.context_tokens and a.reply_reserve >= a.context_tokens):
        print("max-previews must be nonnegative; context-tokens must be 0 or exceed the positive reply-reserve", file=sys.stderr)
        return 2
    proj = project.load(a.project)
    issues = proj.problems()
    if issues:
        print("PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues), file=sys.stderr)
        return 2
    try:
        backend = llm.resolve_backend(prefix=llm.AGENT_PREFIX, url=a.url, model=a.model)
        backend.timeout_s = call_cap(a.timeout, a.max_wall)
        max_tokens = llm.agent_output_limit(backend, a.max_tokens, log=print)
        if a.context_tokens:
            max_tokens = min(max_tokens, a.reply_reserve) if max_tokens is not None else a.reply_reserve
        cache = llm.wants_cache_marks(backend) if a.cache == "auto" else a.cache == "on"
        chat = agent.chat_for(backend, effort=a.effort, max_tokens=max_tokens, idle_s=a.idle, cache=cache, log=print)
        if a.judge:
            llm.resolve_backend()
    except llm.LLMError as e:
        print(str(e), file=sys.stderr)
        return 2
    out = agent.run_dir(proj, backend.model)
    # a marked cache is lost from the first rewritten message on, so under one the old pictures stay
    keep = a.keep_images if a.keep_images is not None else None if cache else agent.KEEP_IMAGE_TURNS
    print(f"agent: {backend.describe()}, up to {a.attempts} attempts"
          + (f", until {a.patience} in a row have {agent.gained(a.margin)}" if a.patience else "")
          + (", prompt cache marked" if cache else "")
          + (f", cost cap ${a.max_cost:.2f}" if a.max_cost else "")
          + (f", time budget {a.max_wall:.0f} s" if a.max_wall else "") + f" -> {out}")
    print(f"reply output cap: {max_tokens} tokens (reasoning included)" if max_tokens is not None
          else "reply output cap: endpoint default")
    rep = agent.run(proj, chat, attempts=a.attempts, max_turns=a.max_turns, judge=a.judge, patience=a.patience,
                    margin=a.margin, model=backend.model,
                    url=backend.url, out=out, max_cost=a.max_cost, max_wall=a.max_wall, keep_images=keep,
                    max_previews=a.max_previews, require_evaluation=not a.allow_pending_candidates,
                    context_tokens=a.context_tokens, reply_reserve=a.reply_reserve, log=print)
    print(agent.summary(rep))
    print(f"-> {out / 'report.json'}; ledger {proj.ledger}")
    where = f'"{a.project}"' if " " in str(a.project) else a.project
    print(f"see every attempt beside the reference, in a browser: mesh-jig view {where} --open")
    return 0 if rep.done or rep.evaluations else 1


def cmd_view(a) -> int:
    paths = a.paths or [Path(".")]
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        print(f"not a directory: {', '.join(missing)}", file=sys.stderr)
        return 2
    skip = tuple(a.skip or ())
    if a.export:
        try:
            wrote = viewer.export(paths, a.export, skip, best_only=a.best)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"{wrote['projects']} project(s), {wrote['attempts']} attempt(s) -> {a.export}: {wrote['files']} files, "
              f"{wrote['bytes'] / 1e6:.1f} MB. Open index.html through a web server, or put the directory on a static host.")
        return 0
    index = viewer.build_index(paths, skip)
    n = sum(len(r["projects"]) for r in index["references"])
    try:
        server = viewer.serve(paths, a.host, a.port, skip)
    except OSError as e:
        print(f"cannot listen on {a.host}:{a.port}: {e} (try --port)", file=sys.stderr)
        return 2
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{server.server_address[1]}"
    print(f"{len(index['references'])} reference(s), {n} project(s) -> {url}  (Ctrl+C stops it)")
    if a.open:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_doctor(a) -> int:
    import numpy
    import PIL
    print(f"mesh-jig {__version__}, Python {sys.version.split()[0]}, numpy {numpy.__version__}, Pillow {PIL.__version__}")
    blender = build.find_blender()
    print(f"Blender: {blender or 'NOT FOUND (put `blender` on PATH or set ' + build.BLENDER_ENV + ')'}")
    try:
        print(f"judge: {llm.resolve_backend().describe()} (key set)")
    except llm.LLMError as e:
        print(f"judge: not configured ({e}); everything but `judge` works without it")
    try:
        print(f"agent: {llm.resolve_backend(prefix=llm.AGENT_PREFIX).describe()} (key set)")
    except llm.LLMError as e:
        print(f"agent: not configured ({e}); only `agent` needs it")
    ok = bool(blender)
    if a.project:
        proj = project.load(a.project)
        issues = proj.problems()
        print(f"project {proj.name}: {len(proj.references)} reference view(s) {sorted(proj.references)}, "
              f"{len(proj.palette)} palette colour(s), outlines {'measurable' if proj.reference_profile() else 'not measurable'}")
        for i in issues:
            print(f"- {i}")
        for view, path in proj.references.items():
            for w in references.warnings(references.facts(path)):
                print(f"- reference {view}: {w} (`mesh-jig refs {proj.root} --out <dir>` shows it)")
        if not issues:
            print(brief_mod.criteria_facts(proj))
            if state_views_hint(proj):
                print(state_views_hint(proj))
        ok = ok and not issues
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mesh-jig", description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"mesh-jig {__version__}")
    subs = ap.add_subparsers(dest="cmd", required=True)

    s = subs.add_parser("init", help="start a project")
    s.add_argument("dir", type=Path)
    s.add_argument("--name", default="")
    s.add_argument("--length", type=float, default=1.0, help="model length in metres (default 1.0)")
    s.add_argument("--sheet", type=Path, help="a design sheet image to copy in and cut views from")
    s.set_defaults(fn=cmd_init)

    s = subs.add_parser("sheet", help="cut refs/ from the design sheet named in jig.json")
    s.add_argument("project", type=Path)
    s.set_defaults(fn=cmd_sheet)

    s = subs.add_parser("refs", help="the reference views as the measures see them: boxes, ground lines, warnings")
    s.add_argument("project", type=Path)
    s.add_argument("--region", nargs=5, action="append", metavar=("VIEW", "X0", "Y0", "X1", "Y1"),
                   help="a rectangle in that view's pixels, printed as a criteria.review region box (may be repeated)")
    s.add_argument("--out", type=Path, help="also write <view>_seen.png here: what the mask takes as the subject")
    s.set_defaults(fn=cmd_refs)

    s = subs.add_parser("streak", help="for a run with no attempt cap: whether to keep going or stop")
    s.add_argument("project", type=Path)
    s.add_argument("--patience", type=int, default=streak.PATIENCE,
                   help="evaluated attempts in a row with no improvement that end the run (default 3)")
    s.add_argument("--margin", type=float, default=streak.MARGIN,
                   help=f"how much an attempt has to beat the last improvement by to count as one (default {streak.MARGIN:g}; "
                        "0 counts any higher score)")
    s.add_argument("--max-attempts", type=int, default=0, help="a ceiling on evaluated attempts (default none)")
    s.add_argument("--json", action="store_true", help="the verdict as a record")
    s.set_defaults(fn=cmd_streak)

    s = subs.add_parser("palette", help="read a palette off the reference views")
    s.add_argument("project", type=Path)
    s.add_argument("--colors", type=int, default=5)
    s.add_argument("--write", action="store_true", help="write it into jig.json (only when it has no palette)")
    s.add_argument("--refine", nargs="*", metavar="NAME",
                   help="what each colour of jig.json's palette would move to, read off the reference's own paint "
                        "where the scored views read as it; name colours to move them, and add --write to write "
                        "those over the palette")
    s.set_defaults(fn=cmd_palette)

    s = subs.add_parser("brief", help="the contract, measured outlines and colour zones as text")
    s.add_argument("project", type=Path)
    s.set_defaults(fn=cmd_brief)

    s = subs.add_parser("criteria", help="what the reference is judged on, and where jig.json states it")
    s.add_argument("project", type=Path)
    s.add_argument("--write", action="store_true",
                   help="move the criteria into jig.json's `criteria` block, the fallback views stated (no number moves)")
    s.set_defaults(fn=cmd_criteria)

    s = subs.add_parser("eval", help="build, render, measure, append a ledger line")
    s.add_argument("project", type=Path)
    s.add_argument("--script", required=True, type=Path)
    s.add_argument("--out", help="attempt directory (default: the script's directory)")
    s.add_argument("--name", default="", help="attempt name in the ledger (default: the attempt directory's name)")
    s.add_argument("--note", default="", help="what this attempt changed, for the ledger")
    _atlas_args(s)
    s.add_argument("--judge", action="store_true", help="also ask the pairwise judge (a model call, costs money)")
    s.add_argument("--against", help="another attempt directory (or a directory of its renders) to judge against")
    s.add_argument("--against-name", default="")
    s.add_argument("--samples", type=int, default=16)
    s.add_argument("--ledger", help="ledger file (default: <project>/attempts/ledger.md)")
    s.add_argument("--no-ledger", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = subs.add_parser("show", help="what eval said of an attempt, read back: nothing is built or added to the ledger")
    s.add_argument("project", type=Path)
    s.add_argument("attempt", type=Path, help="an attempt directory, or its name under <project>/attempts/")
    s.set_defaults(fn=cmd_show)

    s = subs.add_parser("build", help="build and contract check only")
    s.add_argument("project", type=Path)
    s.add_argument("--script", required=True, type=Path)
    s.add_argument("--out", help="attempt directory (default: the script's directory)")
    _atlas_args(s)
    s.set_defaults(fn=cmd_build)

    s = subs.add_parser("check", help="the contract check on an attempt's GLBs")
    s.add_argument("project", type=Path)
    s.add_argument("attempt", type=Path)
    s.set_defaults(fn=cmd_check)

    s = subs.add_parser("paint", help="paint an atlas plan to a PNG")
    s.add_argument("project", type=Path)
    s.add_argument("--plan", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.set_defaults(fn=cmd_paint)

    s = subs.add_parser("preview", help="render what an attempt has built, beside the reference")
    s.add_argument("project", type=Path)
    s.add_argument("attempt", type=Path)
    s.add_argument("--views", nargs="*")
    s.set_defaults(fn=cmd_preview)

    s = subs.add_parser("judge", help="pairwise judge: is attempt A closer to the reference than attempt B")
    s.add_argument("project", type=Path)
    s.add_argument("a", help="attempt A")
    s.add_argument("b", nargs="?", help="attempt B")
    s.add_argument("--self-check", action="store_true", help="judge A against itself: measures the judge's own bias")
    s.add_argument("--samples", type=int, default=16)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_judge)

    s = subs.add_parser("agent", help="an API model drives the loop with the same tools (model calls, costs money)")
    s.add_argument("project", type=Path)
    s.add_argument("--model", help="model id (default: $MESH_JIG_AGENT_MODEL)")
    s.add_argument("--url", help="OpenAI-compatible endpoint (default: $MESH_JIG_AGENT_URL, else OpenRouter)")
    s.add_argument("--attempts", type=int, default=5,
                   help="evaluated attempts it makes (default 5); with --patience, the most it may make")
    s.add_argument("--patience", type=int, default=0,
                   help="keep going until this many evaluated attempts in a row show no improvement, as `mesh-jig "
                        "streak` counts it, with --attempts as the ceiling (default: off, exactly --attempts)")
    s.add_argument("--margin", type=float, default=streak.MARGIN,
                   help="with --patience: how much an attempt has to beat the last improvement by to count as one "
                        f"(default {streak.MARGIN:g})")
    s.add_argument("--max-turns", type=int, default=0,
                   help=f"turn cap (default {agent.TURNS_PER_ATTEMPT} per attempt)")
    s.add_argument("--max-previews", type=int, default=2,
                   help="successful previews per candidate, persisted across runs (default 2; 0 disables the cap)")
    s.add_argument("--allow-pending-candidates", action="store_true",
                   help="allow working on another candidate before evaluating the last previewed one")
    s.add_argument("--context-tokens", type=int, default=0,
                   help="endpoint context capacity; compact history proactively using a conservative estimate (0 disables)")
    s.add_argument("--reply-reserve", type=int, default=8192,
                   help="tokens reserved for a reply when context budgeting is enabled; also caps output (default 8192)")
    s.add_argument("--max-cost", type=float, default=0.0,
                   help="stop the run before the cost the endpoint reports passes this many USD (default: no cap)")
    s.add_argument("--cache", choices=("auto", "on", "off"), default="auto",
                   help="mark the conversation for the endpoint's prompt cache (auto: Anthropic models on OpenRouter, "
                        "which cache nothing unmarked)")
    s.add_argument("--keep-images", type=int, default=None,
                   help=f"picture-bearing turns that keep their pictures (default {agent.KEEP_IMAGE_TURNS}; all of "
                        "them when the prompt cache is marked)")
    s.add_argument("--effort", choices=llm.EFFORTS, help="reasoning effort (default: the endpoint's own)")
    s.add_argument("--max-tokens", type=int,
                   help="output cap per reply, reasoning included (default: OpenRouter's advertised model maximum; "
                        "endpoint default elsewhere or if metadata is unavailable)")
    s.add_argument("--max-wall", type=float, default=0.0,
                   help="time budget for the whole run, in seconds: a model call gets what is left of it, and the run "
                        "ends there with the attempts it measured (default: no budget)")
    s.add_argument("--timeout", type=float, default=None,
                   help="cap on one model call, in seconds (default 900, or what is left of --max-wall)")
    s.add_argument("--idle", type=float, default=llm.CHAT_IDLE_S,
                   help=f"seconds without a stream chunk before a model call is dropped and retried (default "
                        f"{llm.CHAT_IDLE_S:g}); raise it for an endpoint that thinks in silence")
    s.add_argument("--judge", action="store_true", help="also offer the pairwise judge (more model calls)")
    s.set_defaults(fn=cmd_agent)

    s = subs.add_parser("view", help="a local web page of every result, grouped by reference")
    s.add_argument("paths", nargs="*", type=Path, help="directories to search for projects (default: here)")
    s.add_argument("--skip", action="append", metavar="NAME",
                   help="a directory name not to search, wherever it is (may be given more than once)")
    s.add_argument("--port", type=int, default=viewer.DEFAULT_PORT)
    s.add_argument("--host", default="127.0.0.1", help="address to bind (0.0.0.0 to reach it from another machine)")
    s.add_argument("--open", action="store_true", help="open the page in the browser")
    s.add_argument("--export", type=Path, metavar="DIR",
                   help="write the page and what it shows into DIR as files for a static host, and do not serve")
    s.add_argument("--best", action="store_true", help="with --export: each run's best attempt only")
    s.set_defaults(fn=cmd_view)

    s = subs.add_parser("doctor", help="what is installed and configured")
    s.add_argument("project", nargs="?", type=Path)
    s.set_defaults(fn=cmd_doctor)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = parser().parse_args(argv)
    try:
        return a.fn(a)
    except (project.ProjectError, ValueError) as e:
        print(f"mesh-jig: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
