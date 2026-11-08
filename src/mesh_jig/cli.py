"""mesh-jig: tools that build, render and measure a Blender model script against reference views.

    mesh-jig init <dir> [--name N] [--length 0.08 | --height 1.2] [--sheet sheet.png]  start a project
    mesh-jig sheet <project>                                              cut refs/ from the design sheet
    mesh-jig refs <project> [--region VIEW X0 Y0 X1 Y1] [--out DIR]       the reference views as the measures see them
    mesh-jig palette <project> [--colors 5] [--write]                     read a palette off the reference views
    mesh-jig brief <project>                                              the contract, outlines and zones, as text
    mesh-jig criteria <project> [--write]                                 what the reference is judged on, and where
                                                                          jig.json states it
    mesh-jig new <project> <name> [--from <attempt>]                      a new attempt, its script copied from another
    mesh-jig docs organic|contract|recipe                                 print a file the skill links to
    mesh-jig eval <project> --script <attempt>/model.py --note "..."      build + render + measure + ledger line
    mesh-jig show <project> <attempt>                                     what eval said of it, read back
    mesh-jig build <project> --script <attempt>/model.py                  build and contract check only
    mesh-jig check <project> <attempt>                                    the contract check alone
    mesh-jig paint <project> --plan plan.json --out atlas.png             paint an atlas plan
    mesh-jig preview <project> <attempt> [--views hero side]              a quick look at what is built
    mesh-jig judge <project> <attempt A> <attempt B> [--samples 16]       pairwise judge (a model call, costs money)
    mesh-jig streak <project> [--patience 3] [--margin 0.002]             for a run with no attempt cap: keep going, or stop
    mesh-jig agent <project> --model M [--attempts 5] [--workflow visual|numeric]           an API model drives the loop (costs money)
    mesh-jig sound-cues <animated.glb>                                    when a rigged model's clips make a sound
    mesh-jig sound-check <sound dir> --script sounds.py [--out DIR]       a sound script held to its contract
    mesh-jig agent <sound dir> --workflow sound --model M [--attempts 6]  an API model writes the sound script
    mesh-jig view [<dir> ...] [--skip NAME] [--port 8796] [--open]        a web page of every result, per reference
    mesh-jig view <dir> ... --export <dir> [--best]                       that page as files, for a static host
    mesh-jig doctor [<project>]                                           what is installed and configured

Exit codes: 0 done, 1 the attempt failed (a build error, a contract breach, an empty render), 2 the request itself
was wrong (no project, bad arguments, no judge configured).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__, agent, brief as brief_mod, build, cpu, criteria, evaluate, glb, judge as judge_mod, llm, paint
from . import palette, project, readiness, references, render, rig, selection, sheet, silhouette, streak, target, viewer

PREVIEW_DIR = "preview"
FEEDBACK_FILE = "feedback.txt"


def _attempt_dir(a) -> Path:
    return (Path(a.out) if a.out else Path(a.script).parent).resolve()


def _atlas_args(sub) -> None:
    sub.add_argument("--atlas", type=Path, help="textured contract: bind this PNG")
    sub.add_argument("--paint", type=Path, help="textured contract: paint this plan first, then bind it")


def rig_summary(report: dict, spec: dict, rigged: Path | None) -> list[str]:
    """A rig build as a builder reads it: held or refused and why, each clip's numbers against its gates, and for a
    clip refused on stretch, where it stretched and which bone's weight pulled it (read from the GLB kept)."""
    issues = report.get('issues') or []
    lines = ['CONTRACT: held' if not issues else f'REFUSED: {len(issues)} issue(s): ' + '; '.join(map(str, issues))]
    for name, c in (report.get('clips') or {}).items():
        gate = (spec.get('clips') or {}).get(name) or {}
        said = [f"stretch {c['stretch_area_fraction'] * 100:.2f}%" + (f" of {gate['stretch_area_max'] * 100:g}%"
                                                                      if 'stretch_area_max' in gate else '')]
        if c.get('loop'):
            said.append(f"loop seam {c.get('loop_mm', 0):.2f} mm")
        if 'max_displacement_mm' in c:
            said.append(f"moves {c['max_displacement_mm']:.0f} mm")
        lines.append(f'  {name}: ' + ', '.join(said))
    stretched = [str(i).split(':')[0] for i in issues if 'stretch' in str(i)]
    if stretched and rigged is not None:
        from . import rig_read
        moves = rig_read.motion(rigged, spec)
        for clip in stretched:
            m = moves.get(clip) or {}
            for w in ((m.get('stretch') or {}).get('where') or [])[:3]:
                lines.append(f"  {clip} stretched at {m['stretch']['at_s']:.2f} s: {w['share'] * 100:.2f}% of the "
                             "surface " + (f"in the part {w['bones'][0]} holds, pulled by {w['bones'][1]}'s weight"
                                           if len(w['bones']) == 2 else 'inside ' + w['bones'][0]))
    return lines


def cmd_rig(a) -> int:
    if a.cmd == 'rig-info':
        from . import rig_brief, rig_look
        spec = json.loads(a.contract.read_text(encoding='utf-8')) if a.contract else None
        said = rig_brief.text(a.source, spec) if spec else rig_brief.facts(a.source)
        print(said)
        if a.out:
            # all of it in a file too: at 37 KB a shell cut it short and a builder lost the clip numbers
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / 'info.txt').write_text(said + '\n', encoding='utf-8')
            print(f'(all of the above is also in {a.out / "info.txt"})')
        if a.write_template:
            a.write_template.parent.mkdir(parents=True, exist_ok=True)
            a.write_template.write_text(rig_brief.template(spec, rig_brief.suggest(a.source)), encoding='utf-8')
            print(f'the recipe as a script, its numbers left for you to set: {a.write_template}')
        if a.out:
            model, _ = rig_brief.load(a.source)
            a.out.mkdir(parents=True, exist_ok=True)
            for view, picture in rig_look.ruled_rest(a.source, model['lo'], model['hi']).items():
                picture.save(a.out / f'rest_{view}.png')
                print(a.out / f'rest_{view}.png')
        return 0
    spec = json.loads(a.contract.read_text(encoding='utf-8'))
    if a.cmd != 'rig-build' and not a.rig.is_file():
        # a build refused before it posed anything (a missing bone, a script error) leaves no failed.glb behind
        print(f'no rig at {a.rig}: rig-build wrote none there (a build refused before it posed the model writes no '
              f'failed GLB; its report says why)', file=sys.stderr)
        return 2
    if a.cmd == 'rig-build':
        try:
            report = rig.build(a.source, a.script, spec, a.out, failed=a.failed)
        except rig.ContractError as e:
            report = e.report       # out of contract and not delivered: the numbers of what it built, all the same
        if a.failed and report.get('issues') and not Path(a.failed).is_file():
            print(f'(refused before it posed the model: no {a.failed} was written, so there is nothing to check or '
                  'preview; the issues below say why)')
        # the verdict, a line a clip, and for a stretch refusal where it stretched; the report itself in a file:
        # its JSON on the screen hid the one line that mattered from builders who read the end of the output
        kept = Path(a.out) if Path(a.out).is_file() else Path(a.failed) if a.failed and Path(a.failed).is_file() else None
        print('\n'.join(rig_summary(report, spec, kept)))
        if kept:
            full = kept.with_suffix('.report.json')
            full.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'(the full report: {full})')
        else:
            print(json.dumps(report, indent=2))
        return 1 if report['issues'] else 0
    elif a.cmd == 'rig-check':
        report = rig.check(a.source, a.rig, spec)
        if a.readings:
            # the verdict, the readings, and the full report in a file: its hundred lines of JSON on the screen
            # buried the readings a builder runs this for
            from . import rig_read
            issues = report.get('issues') or []
            print('CONTRACT: ' + ('held' if not issues else f'{len(issues)} issue(s): ' + '; '.join(map(str, issues))))
            print(rig_read.text(a.rig, spec, report))
            full = Path(a.rig).with_suffix('.report.json')
            full.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'(the full report, every number the gates read: {full})')
            return 1 if issues else 0
    elif a.views:
        from . import rig_look
        doc, blob = glb.read_glb(a.rig)
        model, _ = rig.model_data(doc, blob)
        a.out.mkdir(parents=True, exist_ok=True)
        if a.aim:
            # a close-up: the cameras fitted to a box round the point instead of the whole model, to read a knee
            half = a.span / 2
            model = {**model, 'lo': [v - half for v in a.aim], 'hi': [v + half for v in a.aim]}
        rig_look.weights_sheet(a.rig, model['lo'], model['hi']).save(a.out / 'weights.png')
        print(a.out / 'weights.png')
        for clip in ([a.clip] if a.clip else list(spec['clips'])):
            path = a.out / f'{clip}-views.png'
            rig_look.strip(a.rig, clip, spec['clips'][clip]['seconds'], model['lo'], model['hi'], a.views, a.phases).save(path)
            print(path)
        return 0
    else:
        if not a.clip:
            raise ValueError('rig-preview needs --clip, or --views for every clip from cameras by name')
        print(rig.preview(a.rig, a.clip, a.times, a.out, spec))
        return 0
    print(json.dumps(report, indent=2))
    return 1 if report['issues'] else 0


def cmd_sound(a) -> int:
    from . import sound, sound_agent, sound_cues
    if a.cmd == "sound-cues":
        cues = sound_cues.read(a.glb)
        print(json.dumps(cues, indent=1) if a.json else sound_cues.text(cues))
        return 0
    if a.cmd == "sound-info":
        from . import sound_recipe
        spec = sound.load(a.root)
        print(sound_agent.info(a.root, spec))
        if not a.write_template:
            # none of three builders found the option; two pulled the recipe out of this text with a script
            print(f"\nTo start from the recipe as a file: mesh-jig sound-info {a.root} --write-template "
                  f"{Path(a.root) / 'attempts' / '01' / 'sounds.py'} writes it with a sounds(cues) that already holds "
                  "the contract.")
        if a.write_template:
            a.write_template.parent.mkdir(parents=True, exist_ok=True)
            a.write_template.write_text(sound_recipe.template(spec, sound.cues_for(a.root, spec)), encoding="utf-8")
            print(f"\na sounds.py that holds the contract as it stands, to replace one sound at a time: "
                  f"{a.write_template}")
        return 0
    if not Path(a.script).is_file():
        print(f"no script at {a.script}: --script is a path from where you run this (or an absolute one), such as "
              f"{Path(a.root) / 'attempts' / '01' / 'sounds.py'}", file=sys.stderr)
        return 2
    res = sound.check(a.root, a.script, keep=a.out)
    said = sound.text(res)
    readings = ""
    if a.out and res.sounds:
        spec = sound.load(a.root)
        readings = sound_agent.measured(a.out, spec, res.cues, a.out / sound_agent.SHEET)
    # the verdict and what is OFF first, once: every builder grepped seventy lines for them
    lines = said.splitlines()
    verdict, body = lines[-1], lines[:-1]
    if not res.ok and not any(r.get("stats") for r in res.sounds.values()):
        verdict = "SCRIPT ERROR: nothing was made, so nothing was checked (the error is below)."
    print(verdict)
    off = sound_off(readings)
    if readings:
        print("OFF (readings, not gates; where to work next): " + ("; ".join(off) if off else "none"))
    print("\n".join(body))
    if readings:
        print(readings)
        print(a.out / sound_agent.SHEET)
    return 0 if res.ok else 1


def sound_off(readings: str) -> list[str]:
    """The readings that are OFF, each as `sound: what`, from `sound_agent.measured`'s text."""
    out, name = [], ""
    for line in readings.splitlines():
        if line and not line.startswith(" ") and ": " in line and "as it should measure" in line:
            name = line.split(":")[0]
        elif line.strip().startswith("OFF "):
            out.append(f"{name}: {line.strip()[4:]}")
    return out


def cmd_init(a) -> int:
    if a.height is not None and a.length is not None:
        print("give --length (a craft) or --height (a standing subject), not both", file=sys.stderr)
        return 2
    try:
        written = sheet.init(a.dir, name=a.name, length_m=1.0 if a.length is None else a.length, sheet=a.sheet,
                             height_m=a.height)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    for p in written:
        print(p)
    if a.height is not None:
        print("next: put the reference views in refs/ (" + ", ".join(f"{v}.png" for v in sheet.STANDING_VIEWS) + ")"
              + (", or tighten the crop boxes in jig.json and run `mesh-jig sheet`" if a.sheet else "")
              + "; run `mesh-jig refs` and copy the calibration it prints into `criteria.outline` with "
              "\"mode\": \"standing\", \"origin_m\": [0, 0, " + f"{-a.height / 2:g}" + "] and the stations you "
              "name (docs/ORGANIC.md), fill in brief.md, then `mesh-jig brief` to read the contract back")
        return 0
    print("next: put the reference views in refs/ (hero.png, top.png, side.png, rear.png)"
          + (", or tighten the crop boxes in jig.json and run `mesh-jig sheet`" if a.sheet else "")
          + ", fill in brief.md, state what the reference is judged on in jig.json's `criteria` block (`mesh-jig "
          "criteria` reads it back), then `mesh-jig brief` to read the contract back")
    return 0


def cmd_sheet(a) -> int:
    for p in sheet.write_refs(project.load(a.project, getattr(a, "target", None))):
        print(p)
    return 0


def cmd_refs(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
    if not proj.references:
        print(f"no reference views under {proj.refs_dir}", file=sys.stderr)
        return 2
    regions = [(view, [float(v) for v in box]) for view, *box in a.region or []]
    print(references.text(proj, regions))
    centres = {view: float(x) for view, x in a.centre or []}
    for view, x, y in a.probe or []:
        if not (proj.spec or {}).get("height_m"):
            try:
                point, unit = references.probe_thousandths(proj, view, float(x), float(y))
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2
            seen = [f"{axis} {v:+.0f}" for axis, v in zip("XYZ", point) if v]
            print(f"\nPROBE on {view}, pixel ({x}, {y}): " + (", ".join(seen) or "the centre") + f" in thousandths "
                  f"of the subject's extent along {unit}, from the centre of its box, as the brief's outline table "
                  "(the axis the view looks along is not seen)")
            continue
        try:
            (point,), centre = references.articulation_path(proj, view, [(float(x), float(y))], centres.get(view))
        except ValueError as error:
            print(str(error).replace("--articulation", "--probe"), file=sys.stderr)
            return 2
        seen = [f"{axis} {v:+.3f}" for axis, v in zip("XYZ", point) if axis == "Z" or abs(v) > 0]
        print(f"\nPROBE on {view}, pixel ({x}, {y}): " + ", ".join(seen) + " m, Blender axes with Z above the ground "
              f"line (the axis the view looks along is not seen; centred on {centre})")
    for view, *coords in a.articulation or []:
        if len(coords) < 4 or len(coords) % 2:
            print(f"--articulation {view}: give two or more X Y pixel pairs after the view", file=sys.stderr)
            return 2
        points = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
        try:
            path, centre = references.articulation_path(proj, view, points, centres.get(view))
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        print(f"\nARTICULATION on {view}, pixels {points}, as a `review.articulations` entry (the axis the view looks "
              f"along is 0; set radius_m to the part's thickness). Centred on {centre}:")
        print("  " + json.dumps({"name": "<name>", "path_m": path, "radius_m": 0.05, "min_free": 0.5}))
    if a.out:
        print("\nWHAT THE MASK TAKES AS THE SUBJECT (the rest washed out; red box, blue ground line, green centre line):")
        for p in references.write_seen(proj, a.out):
            print(p)
        ruled = references.write_ruled(proj, a.out)
        if ruled:
            print("\nRULED IN BLENDER METRES (where a part of the reference sits in the coordinates a script writes; "
                  "red lines at 0):")
            for p in ruled.values():
                print(p)
    return 0


def cmd_streak(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
    v = streak.verdict(proj.root, a.patience, a.margin, a.max_attempts)
    print(json.dumps(v, indent=1) if a.json else streak.text(v))
    return 0


def cmd_checkpoint(a) -> int:
    proj = project.load(a.project)
    if a.verify:
        if any((a.visual_attempt, a.visual_source, a.reason, a.review_file, a.before, a.through, a.views)):
            raise ValueError("--verify reads an existing checkpoint; omit creation options")
        record = selection.verify(proj.root / selection.DIRECTORY / selection._name(a.name))
    else:
        record = selection.create(proj, a.name, visual_attempt=a.visual_attempt, visual_source=a.visual_source,
                                  reason=a.reason, review_file=a.review_file, before=a.before,
                                  through=a.through, views=a.views)
    print(json.dumps({"checkpoint": str(proj.root / selection.DIRECTORY / a.name),
                      "selections": record["selections"], "successful_count": record["scope"]["successful_count"],
                      "visual_choice": record["visual_choice"], "warnings": record["warnings"]}, indent=2))
    return 0


def cmd_palette(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
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
    proj = project.load(a.project, getattr(a, "target", None))
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
    proj = project.load(a.project, getattr(a, "target", None))
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
    proj = project.load(a.project, getattr(a, "target", None))
    out = _attempt_dir(a) / "out"
    atlas, err = _painted_atlas(proj, a, out)
    err = err or build.build_variants(a.script, out, proj.spec, **({"atlas": atlas} if atlas else {}),
                                      **({"palette": proj.palette} if glb.texture_from(proj.spec) == "script" else {}))
    if err:
        print("BUILD FAILED:\n" + err)
        return 1
    report = json.loads((out / glb.REPORT).read_text(encoding="utf-8"))
    for name, s in report["stats"].items():
        cap = proj.spec.get("max_triangles")
        print(f"{name}: {s['triangles']}" + (f" of {cap}" if cap else "")
              + f" triangles, size in Blender axes X {s['size'][0]:.3f} (width), Y {s['size'][2]:.3f} (depth), Z "
              f"{s['size'][1]:.3f} (height) m, {s['distinct_colors']} colours{glb.by_node(s)}")
        said = off_palette_line(proj, s)
        if said:
            print(said)
    print(f"built and in contract -> {out}")
    return 0


CHECKOUT = Path(__file__).resolve().parents[2]
DOCS = {"organic": "docs/ORGANIC.md", "contract": "examples/character-contract/jig.example.json",
        "recipe": "examples/organic/recipe.py"}


def cmd_docs(a) -> int:
    """The files the skill links to, printed from the checkout mesh-jig runs from: a copy of the skill in a run
    directory sits beside none of them, and a sandboxed builder may read nothing outside that directory."""
    path = CHECKOUT / DOCS[a.name]
    if not path.is_file():
        print(f"{DOCS[a.name]} is not in {CHECKOUT}: mesh-jig is not installed from a checkout (pip install -e .)",
              file=sys.stderr)
        return 2
    print(path.read_text(encoding="utf-8"))
    return 0


COPIED = ("atlas_plan.json", "paint_plan.json")
BAD_NAME = set('/\\:*?"<>|')


def cmd_new(a) -> int:
    """A new attempt directory holding the model script (and an atlas or paint plan) of the attempt it starts from:
    the skill's "copy it to a new directory", with no shell."""
    proj = project.load(a.project, getattr(a, "target", None))
    name = a.name.strip()
    if not name or name in (".", "..") or BAD_NAME & set(name):
        print(f"attempt name {a.name!r}: give a plain directory name, such as a03-wider-head", file=sys.stderr)
        return 2
    dest = proj.attempts_dir / name
    if dest.exists():
        print(f"{dest} already exists: an attempt is never overwritten, choose a new name", file=sys.stderr)
        return 2
    script = glb.script_name(proj.spec)
    src = None
    if a.source:
        src = Path(a.source)
        src = src if src.is_dir() else proj.attempts_dir / a.source
        if not (src / script).is_file():
            print(f"no {script} in {src}: give an attempt directory, or its name under {proj.attempts_dir}",
                  file=sys.stderr)
            return 2
    dest.mkdir(parents=True)
    copied = []
    for filename in (script, *COPIED) if src else ():
        if (src / filename).is_file():
            shutil.copy2(src / filename, dest / filename)
            copied.append(filename)
    print(f"{dest}" + (f": {', '.join(copied)} copied from {src.name}" if copied else f": empty, write {script} here"))
    print(f'then: mesh-jig eval "{a.project}" --script "{dest / script}" --note "<what you changed>"')
    return 0


def cmd_check(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
    out = Path(a.attempt)
    out = out / "out" if (out / "out").is_dir() else out
    report = glb.write_report(out, proj.spec)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


def cmd_ready(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
    out = Path(a.attempt)
    out = out if out.is_dir() else proj.attempts_dir / a.attempt
    out = out / "out" if (out / "out").is_dir() else out
    file = out / glb.judged_glb(proj.spec)
    if not file.is_file():
        print(f"no {file.name} under {out}: build the attempt first", file=sys.stderr)
        return 2
    reading = readiness.measure(file, proj.spec, surface_symmetry=getattr(a, "surface_symmetry", False))
    print(json.dumps(reading, indent=1) if a.json else "\n".join(readiness.lines(reading)))
    return 0 if reading.get("passed", 0) == len(reading.get("checks") or []) else 1


def cmd_paint(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
    err = paint.paint_file(a.plan, a.out, proj.palette)
    print(err or f"painted {a.out}")
    return 1 if err else 0


def off_palette_line(proj: project.Project, stats: dict) -> str:
    """'' or a line naming the vertex colours far from every palette colour (a project with a palette and vertex
    colours): faces a paint step missed."""
    if glb.texture_from(proj.spec) == "script" or proj.spec.get("textured"):
        return ""
    off = glb.off_palette(stats, proj.palette)
    if not off:
        return ""
    srgb = lambda c: "#" + "".join(f"{round(255 * (v / 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)):02x}"  # noqa: E731
                                   for v in c)
    return (f"! {len(off)} vertex colour(s) far from every palette colour: {', '.join(srgb(c) for c in off[:6])}"
            + (" ..." if len(off) > 6 else "") + ". Faces a paint step missed (white is a default) read as a colour "
            "the reference does not have.")


def verdict_block(said: str) -> str:
    """The first block of an evaluation's feedback: its numbers, the mesh line, each gate, where it stands and what
    moved, up to the regions and the views. Builders asked for it after reading a hundred lines an attempt."""
    out = []
    lines = said.splitlines()
    for line in lines:
        if line.startswith(("REGIONS", "Aggregate scoring", "overlap, resemblance")) or (out and ": silhouette" in line):
            break
        out.append(line)
    # and the outline table: for a subject scored on its outline it is what the next attempt is built from, and two
    # mech builders cut it back out of feedback.txt every time
    start = next((i for i, line in enumerate(lines)
                  if line.startswith(("YOUR MODEL AGAINST THE REFERENCE OUTLINES", "STANDING OUTLINE ERRORS"))), None)
    if start is not None:
        out.append("")
        for line in lines[start:]:
            if out[-1] and not line.startswith(("YOUR MODEL", "STANDING", "- ", "    ", "Close the biggest")) \
                    and " min/max/width" not in line:
                break
            out.append(line)
    return "\n".join(out)


def stale(proj: project.Project, attempt: Path) -> str:
    """Why an attempt's build is not of its script as it stands ('' when it is): nothing built, or the script or the
    atlas plan changed since. Three builders previewed a build their edits had already left behind."""
    built = attempt / "out" / glb.judged_glb(proj.spec)
    if not built.is_file():
        return "nothing is built yet"
    for name in (glb.script_name(proj.spec), "atlas_plan.json"):
        source = attempt / name
        if source.is_file() and source.stat().st_mtime > built.stat().st_mtime:
            return f"{name} changed since it was built"
    return ""


def cmd_preview(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
    attempt = a.attempt if a.attempt.is_dir() else proj.attempts_dir / a.attempt
    why = stale(proj, attempt) if (attempt / glb.script_name(proj.spec)).is_file() else ""
    if why:
        print(f"building first: {why}")
        plan = attempt / "atlas_plan.json"
        needs_plan = proj.spec.get("textured") and glb.texture_from(proj.spec) != "script"
        built = cmd_build(argparse.Namespace(project=a.project, script=attempt / glb.script_name(proj.spec), out=None,
                                             atlas=None, paint=plan if needs_plan and plan.is_file() else None,
                                             target=getattr(a, "target", None)))
        if built:
            return built
    rec = evaluate.preview(proj, attempt, a.views or None)
    print(evaluate.preview_text(proj, a.attempt, rec))
    if not rec["ok"]:
        return 1
    for view, path in rec["renders"].items():
        if view not in rec["compare"]:
            print(f"{view} (no reference view): {path}")
    return 0


def cmd_eval(a) -> int:
    proj = project.load(a.project, getattr(a, "target", None))
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
    processed = False

    def postprocess(rec):
        nonlocal processed
        if processed:
            return
        processed = True
        if not a.no_ledger:
            evaluate.append_ledger(Path(a.ledger).resolve() if a.ledger else proj.ledger, rec)
        said = evaluate.feedback(rec, evaluate.standing(proj, out, rec))
        stats_file = out / "out" / glb.REPORT
        if rec.get("ok") and stats_file.is_file():
            for s in json.loads(stats_file.read_text(encoding="utf-8")).get("stats", {}).values():
                warn = off_palette_line(proj, s)
                if warn:                                # under the verdict, where a builder reads first
                    head, *rest = said.split("\n", 1)
                    said = head + "\n" + warn + ("\n" + rest[0] if rest else "")
        print(verdict_block(said) if a.short else said)
        # the same text where a builder whose shell cuts long output can read it in parts
        if out.is_dir():
            (out / FEEDBACK_FILE).write_text(said + "\n", encoding="utf-8")
            print(f"({'the whole of it' if a.short else 'all of the above'} is also in {out / FEEDBACK_FILE})")
        print(f"-> {out / evaluate.EVAL_FILE}")

    rec = evaluate.evaluate(proj, Path(a.script).resolve(), out, atlas=a.atlas.resolve() if a.atlas else None,
                            paint_plan=a.paint.resolve() if a.paint else None, name=a.name, note=a.note,
                            against=Path(a.against).resolve() if a.against else None, against_name=a.against_name,
                            backend=backend, samples=a.samples, log=lambda *_: None, _postprocess=postprocess)
    if not processed:
        postprocess(rec)
    return 0 if rec.get("ok") else 1


def cmd_show(a) -> int:
    """What `eval` said of an attempt, read back from its eval.json: nothing is built, measured or added to the
    ledger. An evaluation run a second time to read a part of its output again leaves a second ledger line."""
    proj = project.load(a.project, getattr(a, "target", None))
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
    proj = project.load(a.project, getattr(a, "target", None))
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
    if a.workflow == "rig":
        return cmd_rig_agent(a)
    if a.workflow == "sound":
        return cmd_sound_agent(a)
    if a.workflow == "visual":
        return cmd_visual_agent(a)
    if a.checks != "none":
        raise ValueError("--checks is for --workflow visual")
    proj = project.load(a.project, getattr(a, "target", None))
    issues = proj.problems()
    if issues:
        print("PROJECT PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues), file=sys.stderr)
        return 2
    try:
        backend = llm.resolve_backend(prefix=llm.AGENT_PREFIX, url=a.url, model=a.model, provider=a.provider)
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


def cmd_rig_agent(a) -> int:
    from . import gauntlet, rig_agent
    issues = rig_agent.problems(a.project.resolve())
    if issues:
        print("RIG DIRECTORY PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues), file=sys.stderr)
        return 2
    try:
        backend = llm.resolve_backend(prefix=llm.AGENT_PREFIX, url=a.url, model=a.model, provider=a.provider)
        if backend.url == gauntlet.URL:
            backend = gauntlet.backend(backend.model)
        backend.timeout_s = call_cap(a.timeout, a.max_wall)
        rep, out = rig_agent.drive(a.project, backend, builds=a.attempts, max_turns=a.max_turns,
                                   max_tokens=a.max_tokens, max_cost=a.max_cost, effort=a.effort, idle_s=a.idle,
                                   reviewer=a.reviewer, log=print)
    except llm.LLMError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(rig_agent.summary(rep))
    print(f"-> {out / 'report.json'}")
    return 0 if rep["final"] else 1


def cmd_sound_agent(a) -> int:
    from . import sound_agent
    issues = sound_agent.problems(a.project.resolve())
    if issues:
        print("SOUND DIRECTORY PROBLEMS:\n" + "\n".join(f"- {i}" for i in issues), file=sys.stderr)
        return 2
    try:
        backend = llm.resolve_backend(prefix=llm.AGENT_PREFIX, url=a.url, model=a.model, provider=a.provider)
        backend.timeout_s = call_cap(a.timeout, a.max_wall)
        rep, out = sound_agent.drive(a.project, backend, builds=a.attempts, max_turns=a.max_turns,
                                     max_tokens=a.max_tokens, effort=a.effort, idle_s=a.idle, log=print)
    except llm.LLMError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(sound_agent.summary(rep))
    print(f"-> {out / 'report.json'}")
    return 0 if rep["final"] else 1


def cmd_visual_agent(a) -> int:
    from . import gauntlet, visual_agent
    if (a.patience or a.judge or a.context_tokens or a.keep_images is not None or a.cache == "on"
            or a.max_previews != 2 or a.allow_pending_candidates or a.reply_reserve != 8192
            or a.margin != streak.MARGIN):
        raise ValueError("preview, context, cache, patience and pairwise-judge options require --workflow numeric; "
                         "visual review uses a fresh reviewer after each evaluation")
    proj = project.load(a.project, getattr(a, "target", None))
    if proj.problems():
        raise ValueError("PROJECT PROBLEMS: " + "; ".join(proj.problems()))
    try:
        backend = llm.resolve_backend(prefix=llm.AGENT_PREFIX, url=a.url, model=a.model, provider=a.provider)
        if backend.url == gauntlet.URL:
            backend = gauntlet.backend(backend.model)
        backend.timeout_s = call_cap(a.timeout, a.max_wall)
        rep, out = visual_agent.run(proj, backend, attempts=a.attempts, checks=a.checks,
                                    max_turns=a.max_turns, max_tokens=a.max_tokens, max_cost=a.max_cost,
                                    max_wall=a.max_wall, effort=a.effort, idle_s=a.idle, log=print,
                                    review_effort=a.review_effort, keep_better=a.keep_better)
    except llm.LLMError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(visual_agent.summary(rep))
    print(f"-> {out / 'report.json'}; ledger {proj.ledger}")
    print(f'see every attempt beside the reference: mesh-jig view "{a.project}" --open')
    return 0 if rep["done"] else 1


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
        print(f"CPU: {cpu.describe(cpu.decide())}")
    except ValueError as e:
        print(f"CPU: not applied: {e}")
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
        proj = project.load(a.project, getattr(a, "target", None))
        issues = proj.problems()
        print(f"project {proj.name}: {len(proj.references)} reference view(s) {sorted(proj.references)}, "
              f"{len(proj.palette)} palette colour(s), outlines {'measurable' if proj.reference_profile() else 'not measurable'}")
        for i in issues:
            print(f"- {i}")
        warned = False
        for view, path in proj.references.items():
            for w in references.warnings(references.facts(path)):
                print(f"- reference {view}: {w} (`mesh-jig refs {proj.root} --out <dir>` shows it)")
                warned = True
        if warned:
            # builders handed a frozen project took these for problems they had to fix before building
            print("  (warnings about the reference pictures, not project problems: a project's owner fixes them in "
                  "the sheet's crops; a builder given the project as it is builds on, and reads those views' numbers "
                  "knowing what is cut off)")
        if not issues:
            said = brief_mod.criteria_facts(proj)
            print(said)
            if state_views_hint(proj):
                print(state_views_hint(proj))
            if "NOT STATED" in said or "WARNING" in said:
                # three builders handed a project they were told not to change read these as work to do first
                print("(The NOT STATED lines and the warning are for the project's owner: a builder handed a project "
                      "it may not change builds on, and every number is measured the same way.)")
        ok = ok and not issues
    return 0 if ok else 1


def _target_arg(s) -> None:
    s.add_argument("--target", help="what the build is for: " + ", ".join(sorted(target.BUILTIN)) + " or one of the "
                   "project's `targets` (default: the project's `target`, else none). It sets the triangle budget "
                   "and the limits the mesh is read against")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mesh-jig", description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"mesh-jig {__version__}")
    subs = ap.add_subparsers(dest="cmd", required=True)
    for command in ('rig-build', 'rig-check', 'rig-preview'):
        s = subs.add_parser(command, help='frozen GLB rigging, contract checks and animated CPU poses')
        s.add_argument('--contract', type=Path, required=True)
        if command != 'rig-preview':
            s.add_argument('--source', type=Path, required=True)
        if command == 'rig-build':
            s.add_argument('--script', type=Path, required=True)
            s.add_argument('--out', type=Path, required=True)
            s.add_argument('--failed', type=Path,
                           help='where to leave a build that fails its contract, with its report beside it, to be '
                                'looked at (rig-check --readings, rig-preview --views); without it such a build is deleted')
        else:
            s.add_argument('--rig', type=Path, required=True)
        if command == 'rig-check':
            s.add_argument('--readings', action='store_true',
                           help='also say what the rig does that no gate reads: what each bone holds, stray '
                                'weights, how each clip moves, where it is against the ground')
        if command == 'rig-preview':
            s.add_argument('--clip', help='the clip to draw (with --views: default every clip)')
            s.add_argument('--times', type=float, nargs='+', default=[0, .25, .5, .75, 1])
            s.add_argument('--views', nargs='+', metavar='VIEW',
                           help='cameras by name, fitted to the model (front back left right top three-quarter): a '
                                'film strip a clip with a row a view, and the weights, in place of the contract shot')
            s.add_argument('--phases', type=int, default=6, help='with --views: frames a strip (default 6)')
            s.add_argument('--aim', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                           help='with --views: a close-up, the cameras fitted to a box round this point (Blender '
                                'metres, as rig-info gives them) in place of the whole model')
            s.add_argument('--span', type=float, default=0.3, help="with --aim: the box's side, metres (default 0.3)")
            s.add_argument('--out', type=Path, required=True)
        s.set_defaults(fn=cmd_rig)
    s = subs.add_parser('rig-info', help='a static GLB as a rig script meets it: nodes, pieces, bounds, slices; '
                                         'with --contract the script rules and the gates too')
    s.add_argument('--source', type=Path, required=True)
    s.add_argument('--contract', type=Path)
    s.add_argument('--out', type=Path, help='also draw the rest pose here, ruled every 0.1 m, from four sides')
    s.add_argument('--write-template', type=Path, metavar='RIG_PY',
                   help="with --contract: write the recipe to this file as a script to start from, its numbers left as "
                        "placeholders to set from the slices and the pictures")
    s.set_defaults(fn=cmd_rig)

    s = subs.add_parser("sound-cues", help="when a rigged model's clips make a sound: footfalls, crouch, takeoff, landing")
    s.add_argument("glb", type=Path)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_sound, cmd="sound-cues")
    s = subs.add_parser("sound-info", help="what a sound author starts from: the brief, the contract, the cues as "
                                          "the script gets them, what each sound should measure as, the recipe")
    s.add_argument("root", type=Path)
    s.add_argument("--write-template", type=Path, metavar="SOUNDS_PY",
                   help="also write the recipe as a sounds.py that holds the contract, to start from")
    s.set_defaults(fn=cmd_sound)

    s = subs.add_parser("sound-check", help="run a sound script twice and hold it to its sound.json")
    s.add_argument("root", type=Path, help="the sound directory (sound.json, brief.md, animated.glb)")
    s.add_argument("--script", type=Path, required=True)
    s.add_argument("--out", type=Path, help="keep the WAVs here, with what each measures as and a spectrogram sheet")
    s.set_defaults(fn=cmd_sound, cmd="sound-check")

    s = subs.add_parser("init", help="start a project")
    s.add_argument("dir", type=Path)
    s.add_argument("--name", default="")
    s.add_argument("--length", type=float, default=None, help="a craft's length in metres (default 1.0)")
    s.add_argument("--height", type=float, default=None,
                   help="a standing subject's height in metres: writes its contract (six views, the front among "
                        "them, held to that height) in place of a craft's")
    s.add_argument("--sheet", type=Path, help="a design sheet image to copy in and cut views from")
    s.set_defaults(fn=cmd_init)

    s = subs.add_parser("sheet", help="cut refs/ from the design sheet named in jig.json")
    s.add_argument("project", type=Path)
    s.set_defaults(fn=cmd_sheet)

    s = subs.add_parser("refs", help="the reference views as the measures see them: boxes, ground lines, warnings")
    s.add_argument("project", type=Path)
    s.add_argument("--region", nargs=5, action="append", metavar=("VIEW", "X0", "Y0", "X1", "Y1"),
                   help="a rectangle in that view's pixels, printed as a criteria.review region box (may be repeated)")
    s.add_argument("--articulation", nargs="+", action="append", metavar="VIEW X Y",
                   help="a part a rig will bend: a view, then two or more pixel points along it, root first; printed "
                        "as a review.articulations entry in Blender metres (may be repeated)")
    s.add_argument("--probe", nargs=3, action="append", metavar=("VIEW", "X", "Y"),
                   help="a pixel of that view in Blender metres, Z above the ground line (may be repeated)")
    s.add_argument("--centre", nargs=2, action="append", metavar=("VIEW", "X"),
                   help="the model's centreline in that view's pixels, for --articulation (default: the calibration's, "
                        "else the middle of the subject's box)")
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

    s = subs.add_parser("checkpoint", help="freeze latest, numeric best and an explicit visual choice; compose saved renders")
    s.add_argument("project", type=Path)
    s.add_argument("name", help="new checkpoint directory name, such as round-05")
    s.add_argument("--visual-attempt", default="", help="successful attempt directory name chosen after inspecting its renders")
    s.add_argument("--visual-source", choices=selection.SOURCES, default="", help="who made the visual choice")
    s.add_argument("--reason", default="", help="why that visual choice was made; required with --visual-attempt")
    s.add_argument("--review-file", default="", help="optional existing evidence file relative to the project")
    s.add_argument("--before", default="", help="earlier checkpoint name for a before/after sheet")
    s.add_argument("--through", default="", help="include successes up to this attempt in evaluation order")
    s.add_argument("--views", nargs="+", help="saved views to show (default hero/front/side where available)")
    s.add_argument("--verify", action="store_true", help="verify the frozen copies without creating or changing files")
    s.set_defaults(fn=cmd_checkpoint)

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
    _target_arg(s)
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
    s.add_argument("--short", action="store_true",
                   help="print the verdict only (numbers, mesh, gates, where it stands); the whole is in feedback.txt")
    _target_arg(s)
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
    _target_arg(s)
    s.set_defaults(fn=cmd_build)

    s = subs.add_parser("docs", help="print a file the skill links to: the organic workflow, the contract template "
                        "or the organic recipe")
    s.add_argument("name", choices=sorted(DOCS))
    s.set_defaults(fn=cmd_docs)

    s = subs.add_parser("new", help="start an attempt directory, copying the script of the attempt it starts from")
    s.add_argument("project", type=Path)
    s.add_argument("name", help="the new attempt's directory name, named for what it changes")
    s.add_argument("--from", dest="source", help="an attempt directory, or its name under <project>/attempts/")
    _target_arg(s)
    s.set_defaults(fn=cmd_new)

    s = subs.add_parser("check", help="the contract check on an attempt's GLBs")
    s.add_argument("project", type=Path)
    s.add_argument("attempt", type=Path)
    _target_arg(s)
    s.set_defaults(fn=cmd_check)

    s = subs.add_parser("ready", help="what a built attempt costs an engine and what a rig or a texture would find in it")
    s.add_argument("project", type=Path)
    s.add_argument("attempt", help="an attempt directory, or its name under <project>/attempts/")
    s.add_argument("--json", action="store_true", help="the reading as JSON")
    s.add_argument("--surface-symmetry", action="store_true",
                   help="also compare sampled geometry across X=0 (approximate diagnostic)")
    _target_arg(s)
    s.set_defaults(fn=cmd_ready)

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

    s = subs.add_parser("agent", help="build, validate, render and get fresh visual feedback (model calls, costs money)")
    s.add_argument("project", type=Path)
    s.add_argument("--workflow", choices=("visual", "numeric", "rig", "sound"), default="visual",
                   help="visual review (default), the legacy numeric tool loop, rig: PROJECT is then a rig "
                        "directory (source.glb, rig.json, brief.md) and --attempts the builds in contract, or sound: "
                        "PROJECT is a sound directory (sound.json, brief.md, animated.glb)")
    s.add_argument("--reviewer", action="store_true",
                   help="rig only: a fresh call of the same model reviews each build's film strips (off by default: "
                        "it costs a call a build and has not been measured to help)")
    s.add_argument("--checks", choices=("none", "builder", "both"), default="none",
                   help="visual workflow: experimental numeric feedback to builder/both (default: blind visual review)")
    s.add_argument("--model", help="model id (default: $MESH_JIG_AGENT_MODEL)")
    _target_arg(s)
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
                   help=f"global implementor turn cap; visual also allows 24 per iteration, numeric defaults to {agent.TURNS_PER_ATTEMPT} per attempt")
    s.add_argument("--max-previews", type=int, default=2,
                   help="numeric only: successful previews per candidate, persisted across runs (default 2; 0 disables the cap)")
    s.add_argument("--allow-pending-candidates", action="store_true",
                   help="allow working on another candidate before evaluating the last previewed one")
    s.add_argument("--context-tokens", type=int, default=0,
                   help="endpoint context capacity; compact history proactively using a conservative estimate (0 disables)")
    s.add_argument("--reply-reserve", type=int, default=8192,
                   help="tokens reserved for a reply when context budgeting is enabled; also caps output (default 8192)")
    s.add_argument("--max-cost", type=float, default=0.0,
                   help="USD cap (default: none); visual reserves worst-case calls on supported direct Anthropic "
                        "5.5 models and holds OpenRouter's reported cost per call; numeric uses the legacy "
                        "endpoint-cost heuristic")
    s.add_argument("--cache", choices=("auto", "on", "off"), default="auto",
                   help="numeric only: mark the conversation for the endpoint's prompt cache (auto: Anthropic models on OpenRouter, "
                        "which cache nothing unmarked)")
    s.add_argument("--keep-images", type=int, default=None,
                   help=f"picture-bearing turns that keep their pictures (default {agent.KEEP_IMAGE_TURNS}; all of "
                        "them when the prompt cache is marked)")
    s.add_argument("--keep-better", action="store_true",
                   help="visual workflow: from round 2, a fresh reviewer call compares each build with the one it "
                        "would replace; a build judged further from the reference is dropped and the next round "
                        "starts again from the one it lost to")
    s.add_argument("--review-effort", choices=llm.EFFORTS,
                   help="visual workflow: the reviewer's own reasoning effort (default: the same as --effort)")
    s.add_argument("--effort", choices=llm.EFFORTS, help="reasoning effort (default: the endpoint's own)")
    s.add_argument("--max-tokens", type=int,
                   help="output cap including reasoning (visual default: 65536; numeric: provider default)")
    s.add_argument("--max-wall", type=float, default=0.0,
                   help="time budget in seconds (default: none); visual checks between model calls, so an "
                        "in-flight call/build may finish after it; numeric also bounds model calls")
    s.add_argument("--timeout", type=float, default=None,
                   help="cap on one model call, in seconds (default 900, or what is left of --max-wall)")
    s.add_argument("--idle", type=float, default=llm.CHAT_IDLE_S,
                   help=f"seconds without a stream chunk before a model call is dropped and retried (default "
                        f"{llm.CHAT_IDLE_S:g}); raise it for an endpoint that thinks in silence")
    s.add_argument("--provider", default=None, metavar="TAG",
                   help="OpenRouter only: send every call to this endpoint, no fallback (a tag from the model's "
                        "endpoints list, e.g. deepinfra/fp8). Default: OpenRouter balances across providers, whose "
                        "prices and quantization differ")
    s.add_argument("--judge", action="store_true", help="numeric only: also offer the pairwise judge (more model calls)")
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
    s.add_argument("--best", action="store_true", help="with --export: each run's highest numeric score only")
    s.set_defaults(fn=cmd_view)

    s = subs.add_parser("doctor", help="what is installed and configured")
    s.add_argument("project", nargs="?", type=Path)
    _target_arg(s)
    s.set_defaults(fn=cmd_doctor)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = parser().parse_args(argv)
    cpu.govern()                                # before any work: everything started later inherits it
    try:
        return a.fn(a)
    except (project.ProjectError, ValueError) as e:
        print(f"mesh-jig: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
