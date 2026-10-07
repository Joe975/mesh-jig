"""One generation of a harness loop: the blind contact sheets, the verdict on them, and the loop's streak.

    python generation.py sheets <review dir> --incumbent <run> [<run> ...] --candidate <run> [<run> ...]
    python generation.py verdict <review dir> [--visual-margin 5] [--numeric-margin 0.02]
    python generation.py status <loop dir> [--stop-after 3]
    python generation.py remeasure <run> <new project dir> [--jig <jig.json>]

A <run> is a mesh-jig project one builder worked in (the attempt it delivered is its best measured one) or one
attempt directory of such a project.

`sheets` writes <review dir>/sheets/: one contact sheet per run (the reference views above, the model's renders
below), named R01.jpg, R02.jpg, ... in a shuffled order with nothing on them that says which side made them, the
brief of each reference, and REVIEW.md, the reviewer's instructions. The key (which sheet is which run, its side and
its measured numbers) goes to <review dir>/key.json, outside what the reviewer is given. The reviewer writes
sheets/scores.json.

`verdict` joins the two and says whether the candidate improved on the incumbent: the visual gain must reach the
margin and the runs' own noise, and the numeric score must not have fallen by more than its margin. It writes
<review dir>/verdict.json.

`status` reads every g<N>/review/verdict.json (and g<N>/confirm/verdict.json) under a loop directory and counts the
generations in a row with no improvement.

`remeasure` copies a run's delivered attempt into a new project and builds and measures it with the code as it is
now, for a generation that changed a measure, the renderer or the cameras.

Exit codes as the CLI's: 0 done (whatever the verdict), 1 a remeasure that failed, 2 the request was wrong.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from mesh_jig import evaluate, project, silhouette, viewer

HERE = Path(__file__).resolve().parent
REVIEW_TEXT = HERE.parent / "review.md"
HEIGHT, GAP, LABEL = 320, 16, 26
SIDES = ("incumbent", "candidate")
KEY_FILE, SCORES_FILE, VERDICT_FILE = "key.json", "scores.json", "verdict.json"
VISUAL_MARGIN = 5.0
NUMERIC_MARGIN = 0.02
MIN_RUNS = 2            # a side's runs on one reference: fewer and the runs' own spread is unknown
GENERATION = re.compile(r"g(\d+)")


class Refused(ValueError):
    """The request cannot be carried out as given."""


def pick(run: str | Path) -> dict:
    """What a run delivered: its project, the attempt (the best measured one, or the one named), the renders of the
    reference's views and the measured numbers."""
    run = Path(run).resolve()
    if (run / project.JIG_FILE).is_file():
        proj, named = project.load(run), None
    elif run.parent.name == project.ATTEMPTS and (run.parent.parent / project.JIG_FILE).is_file():
        proj, named = project.load(run.parent.parent), run.name
    else:
        raise Refused(f"{run} is neither a project (no {project.JIG_FILE}) nor an attempt directory of one")
    refs = {v: Path(p) for v, p in proj.references.items()}
    if not refs:
        raise Refused(f"{proj.root} has no reference views")
    rec = viewer.project_record(proj, proj.name)
    attempt = next((a for a in rec["attempts"] if a["name"] == (named or rec["best"])), None)
    if attempt is None or not attempt["ok"] or attempt["numeric"]["numeric_score"] is None:
        raise Refused(f"{run} has no measured attempt")
    renders = {v: proj.root / w["render"] for v, w in attempt["views"].items() if w.get("render") and v in refs}
    if not renders:
        raise Refused(f"{run}: attempt {attempt['name']} has no renders of the reference's views "
                      f"(`mesh-jig eval` on its script draws them again)")
    n = attempt["numeric"]
    return {"root": proj.root, "attempt": attempt["name"], "refs": refs, "renders": renders, "brief": proj.brief,
            "reference": viewer.reference_key({v: str(p) for v, p in refs.items()}), "name": proj.name,
            "numeric": n["numeric_score"], "overlap": n["ortho_iou"], "resemblance": n["resemblance"],
            "zones": n["zone"]}


def sheet(title: str, refs: dict[str, Path], renders: dict[str, Path], height: int = HEIGHT) -> Image.Image:
    """The reference views above, the model's renders of the same views below, each cropped to its subject and
    scaled to one height, a view to a column."""
    views = [v for v in viewer.ordered_views(refs) if v in renders]
    cols = [(v, silhouette.subject_panel(refs[v], height), silhouette.subject_panel(renders[v], height))
            for v in views]
    widths = [max(r.width, m.width) for _v, r, m in cols]
    out = Image.new("RGB", (sum(widths) + GAP * (len(cols) + 1), 2 * (LABEL + height) + GAP), (255, 255, 255))
    d = ImageDraw.Draw(out)
    x = GAP
    for (v, ref, model), w in zip(cols, widths):
        for row, (label, panel) in enumerate((("REFERENCE", ref), (title, model))):
            y = row * (LABEL + height)
            out.paste(panel, (x + (w - panel.width) // 2, y + LABEL))
            d.text((x + 4, y + 6), f"{label}  {v}", fill=(0, 0, 0))
        x += w + GAP
    return out


def _rel(path: Path, base: Path) -> str:
    """`path` as the key holds it: relative to the review directory, so the record names no machine."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:              # another drive
        return path.name


def make_sheets(review: str | Path, incumbent: list, candidate: list, height: int = HEIGHT) -> dict:
    """Write the sheets and the key for one blind review; the key."""
    review = Path(review).resolve()
    out = review / "sheets"
    if (out / SCORES_FILE).exists():
        raise Refused(f"{out / SCORES_FILE} exists: this review has been scored, a new one needs its own directory")
    runs = [{"side": side, **pick(r)} for side, given in zip(SIDES, (incumbent, candidate)) for r in given]
    seen = set()
    for r in runs:
        if (r["root"], r["attempt"]) in seen:
            raise Refused(f"{r['root']} ({r['attempt']}) is given twice: a run is one sheet, on one side")
        seen.add((r["root"], r["attempt"]))
    groups: dict[str, dict] = {}
    for r in runs:
        g = groups.setdefault(r["reference"], {"name": r["name"], "brief": r["brief"], "incumbent": 0, "candidate": 0})
        g[r["side"]] += 1
    for g in groups.values():
        if min(g["incumbent"], g["candidate"]) < MIN_RUNS or g["incumbent"] != g["candidate"]:
            raise Refused(f"reference {g['name']!r}: {g['incumbent']} incumbent and {g['candidate']} candidate runs. "
                          f"Each side needs the same number, at least {MIN_RUNS}, on the same reference views")
    names = [g["name"] for g in groups.values()]
    for ref, g in groups.items():           # two references under one project name are told apart by their key
        g["label"] = g["name"] if names.count(g["name"]) == 1 else f"{g['name']}-{ref[:6]}"

    for r in runs:
        r["run"] = _rel(r["root"], review)
    runs.sort(key=lambda r: (r["run"], r["attempt"]))
    order = hashlib.sha1("\n".join(f"{r['run']}/{r['attempt']}" for r in runs).encode("utf-8")).hexdigest()
    random.Random(int(order, 16)).shuffle(runs)     # the same runs give the same order; the order says nothing

    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("R*.jpg"), *out.glob("brief-*.md")]:
        old.unlink()
    for g in groups.values():
        (out / f"brief-{g['label']}.md").write_text(g["brief"], encoding="utf-8")
    shutil.copyfile(REVIEW_TEXT, out / "REVIEW.md")
    key = {"sheets": {}}
    for i, r in enumerate(runs, 1):
        rid, label = f"R{i:02d}", groups[r["reference"]]["label"]
        sheet(f"{rid} ({label})", r["refs"], r["renders"], height).save(out / f"{rid}.jpg", quality=90)
        key["sheets"][rid] = {"side": r["side"], "reference": label, "run": r["run"], "attempt": r["attempt"],
                              **{k: r[k] for k in ("numeric", "overlap", "resemblance", "zones")}}
    (review / KEY_FILE).write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    return key


def _side(values: list[float], digits: int) -> dict:
    return {"n": len(values), "mean": round(statistics.fmean(values), digits),
            "values": [round(v, digits) for v in values]}


def decide(key: dict, scores: dict, visual_margin: float = VISUAL_MARGIN,
           numeric_margin: float = NUMERIC_MARGIN) -> dict:
    """Whether the candidate's models improved on the incumbent's, from the key and the reviewer's scores.

    Improved means both: the visual score (0 to 100, mean over a side's sheets, the gain averaged over references)
    rose by at least `visual_margin` and by at least the noise, the standard error of that gain read off the scores'
    own spread inside each side; and the numeric score did not fall by more than `numeric_margin`."""
    sheets = key["sheets"]
    extra = sorted(set(scores) - set(sheets))
    if extra:
        raise Refused(f"scores for sheets the key does not have: {extra}")
    cells: dict[str, dict[str, list]] = {}
    for rid, s in sorted(sheets.items()):
        got = scores.get(rid)
        score = got.get("score") if isinstance(got, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise Refused(f"{rid} needs a `score` from 0 to 100 in {SCORES_FILE}")
        cells.setdefault(s["reference"], {side: [] for side in SIDES})[s["side"]].append((float(score), s["numeric"]))
    gains, changes, squares, freedom, spread = [], [], 0.0, 0, 0.0
    by_reference = {}
    for ref, cell in sorted(cells.items()):
        if min(len(cell[side]) for side in SIDES) < MIN_RUNS:
            raise Refused(f"reference {ref!r} needs at least {MIN_RUNS} sheets a side")
        visual = {side: [v for v, _n in cell[side]] for side in SIDES}
        numeric = {side: [n for _v, n in cell[side]] for side in SIDES}
        gains.append(statistics.fmean(visual["candidate"]) - statistics.fmean(visual["incumbent"]))
        changes.append(statistics.fmean(numeric["candidate"]) - statistics.fmean(numeric["incumbent"]))
        for side in SIDES:
            mean = statistics.fmean(visual[side])
            squares += sum((v - mean) ** 2 for v in visual[side])
            freedom += len(visual[side]) - 1
            spread += 1 / len(visual[side])
        by_reference[ref] = {"visual": {side: _side(visual[side], 1) for side in SIDES},
                             "numeric": {side: _side(numeric[side], 4) for side in SIDES}}
    gain, change = statistics.fmean(gains), statistics.fmean(changes)
    noise = math.sqrt(squares / freedom * spread) / len(cells)
    needed = max(visual_margin, noise)
    reasons = []
    if gain < needed:
        reasons.append(f"the visual gain {gain:+.1f} is under the {needed:.1f} it needs "
                       f"(margin {visual_margin:.1f}, noise {noise:.1f})")
    if change < -numeric_margin:
        reasons.append(f"the numeric score fell {change:+.3f}, more than the {numeric_margin:.3f} allowed")
    return {"improved": not reasons, "reasons": reasons,
            "visual": {"gain": round(gain, 1), "needed": round(needed, 1), "noise": round(noise, 1),
                       "margin": visual_margin},
            "numeric": {"change": round(change, 4), "margin": numeric_margin},
            "references": by_reference}


def verdict_text(v: dict) -> str:
    lines = []
    for ref, r in v["references"].items():
        for what, digits in (("visual", 1), ("numeric", 3)):
            sides = "   ".join(f"{side} {r[what][side]['mean']:.{digits}f} "
                              f"({' '.join(f'{x:.{digits}f}' for x in r[what][side]['values'])})" for side in SIDES)
            lines.append(f"{ref}  {what:8}{sides}")
    vis, num = v["visual"], v["numeric"]
    lines.append(f"visual gain {vis['gain']:+.1f}, needed {vis['needed']:.1f} (margin {vis['margin']:.1f}, "
                 f"noise {vis['noise']:.1f}); numeric change {num['change']:+.3f}, allowed -{num['margin']:.3f}")
    lines.append("IMPROVED" if v["improved"] else "NO IMPROVEMENT: " + "; ".join(v["reasons"]))
    return "\n".join(lines)


def write_verdict(review: str | Path, visual_margin: float = VISUAL_MARGIN,
                  numeric_margin: float = NUMERIC_MARGIN) -> dict:
    review = Path(review)
    try:
        key = json.loads((review / KEY_FILE).read_text(encoding="utf-8"))
        scores = json.loads((review / "sheets" / SCORES_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise Refused(f"{review} needs {KEY_FILE} (from `sheets`) and sheets/{SCORES_FILE} (from the reviewer): {e}")
    v = decide(key, scores, visual_margin, numeric_margin)
    (review / VERDICT_FILE).write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    return v


def _verdict(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def status(loop: str | Path, stop_after: int = 3) -> dict:
    """Every generation under a loop directory, in order, and the streak of generations with no improvement.

    A generation is kept when its review said improved and its confirming review (fresh candidate runs) said so
    again. It showed no improvement when either said no. Until then it is open, and the loop is not done."""
    loop = Path(loop)
    found = sorted((int(m.group(1)), d) for d in loop.iterdir() if d.is_dir() and (m := GENERATION.fullmatch(d.name)))
    generations, streak, open_ = [], 0, []
    for _n, d in found:
        review, confirm = _verdict(d / "review" / VERDICT_FILE), _verdict(d / "confirm" / VERDICT_FILE)
        if review is None:
            state = "baseline" if not generations else "open"
        elif not review["improved"] or (confirm is not None and not confirm["improved"]):
            state = "no improvement"
        else:
            state = "kept" if confirm is not None else "unconfirmed"
        streak = streak + 1 if state == "no improvement" else 0 if state == "kept" else streak
        if state in ("open", "unconfirmed"):
            open_.append(d.name)
        generations.append({"name": d.name, "state": state,
                            "gain": review["visual"]["gain"] if review else None,
                            "confirmed_gain": confirm["visual"]["gain"] if confirm else None,
                            "numeric_change": review["numeric"]["change"] if review else None})
    return {"generations": generations, "streak": streak, "stop_after": stop_after, "open": open_,
            "stop": streak >= stop_after and not open_}


def status_text(s: dict) -> str:
    lines = []
    for g in s["generations"]:
        line = f"{g['name']:5} {g['state']:15}"
        if g["gain"] is not None:
            line += f" visual {g['gain']:+.1f}"
            if g["confirmed_gain"] is not None:
                line += f" (again {g['confirmed_gain']:+.1f})"
            line += f", numeric {g['numeric_change']:+.3f}"
        lines.append(line.rstrip())
    if s["open"]:
        lines.append(f"not decided yet: {', '.join(s['open'])}")
    tail = f"{s['streak']} of {s['stop_after']} generations in a row with no improvement"
    lines.append(f"STOP: {tail}" if s["stop"] else f"CONTINUE: {tail}")
    return "\n".join(lines)


def remeasure(run: str | Path, dest: str | Path, jig: str | Path | None = None, **evaluate_kw) -> dict:
    """A new project at `dest` holding only what `run` delivered, built and measured by the code as it is now (and
    under `jig`, when the generation changed the project file too). The run itself is left as it was."""
    got = pick(run)
    src, dest = got["root"], Path(dest)
    if dest.exists():
        raise Refused(f"{dest} exists: a remeasure needs a new directory")
    source = project.load(src)
    attempt = source.attempts_dir / got["attempt"]
    (dest / project.ATTEMPTS / got["attempt"]).mkdir(parents=True)
    shutil.copyfile(jig or src / project.JIG_FILE, dest / project.JIG_FILE)
    shutil.copytree(source.refs_dir, dest / source.refs_dir.relative_to(src))
    brief = src / str(source.data.get("brief") or "brief.md")
    if brief.is_file():
        shutil.copyfile(brief, dest / brief.relative_to(src))
    for f in attempt.iterdir():
        if f.is_file() and f.suffix in (".py", ".json") and f.name not in (evaluate.EVAL_FILE, evaluate.NUMERIC_FILE,
                                                                             "resemblance.json"):
            shutil.copyfile(f, dest / project.ATTEMPTS / got["attempt"] / f.name)
    proj = project.load(dest)
    out = proj.attempts_dir / got["attempt"]
    plan = out / "atlas_plan.json"
    if proj.spec.get("textured") and plan.is_file():
        evaluate_kw.setdefault("paint_plan", plan)
    rec = evaluate.evaluate(proj, out / proj.spec["script"], out, note=f"remeasured from {src.name}",
                            log=lambda *_: None, **evaluate_kw)
    evaluate.append_ledger(proj.ledger, rec)
    return rec


def expand(paths: list[Path]) -> list[Path]:
    """`runs/*` as a shell would have expanded it, for a shell that hands the pattern over (PowerShell)."""
    out: list[Path] = []
    for p in paths:
        out.extend(sorted(Path(m) for m in glob.glob(str(p))) if any(c in str(p) for c in "*?[") else [p])
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheets", help="blind contact sheets and their key")
    p.add_argument("review", type=Path)
    p.add_argument("--incumbent", nargs="+", required=True, type=Path)
    p.add_argument("--candidate", nargs="+", required=True, type=Path)
    p.add_argument("--height", type=int, default=HEIGHT, help=f"a view's height in pixels (default {HEIGHT})")
    p = sub.add_parser("verdict", help="join the key and the reviewer's scores")
    p.add_argument("review", type=Path)
    p.add_argument("--visual-margin", type=float, default=VISUAL_MARGIN)
    p.add_argument("--numeric-margin", type=float, default=NUMERIC_MARGIN)
    p = sub.add_parser("status", help="the generations so far and the streak")
    p.add_argument("loop", type=Path)
    p.add_argument("--stop-after", type=int, default=3)
    p = sub.add_parser("remeasure", help="a run's delivered attempt, measured again by the code as it is now")
    p.add_argument("run", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--jig", type=Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        if a.cmd == "sheets":
            key = make_sheets(a.review, expand(a.incumbent), expand(a.candidate), a.height)
            print(f"{len(key['sheets'])} sheets -> {a.review / 'sheets'} (give the reviewer that directory and "
                  f"nothing else); key -> {a.review / KEY_FILE}")
        elif a.cmd == "verdict":
            print(verdict_text(write_verdict(a.review, a.visual_margin, a.numeric_margin)))
        elif a.cmd == "status":
            print(status_text(status(a.loop, a.stop_after)))
        else:
            rec = remeasure(a.run, a.dest, a.jig)
            print(evaluate.numbers_text(rec["numeric"]) if rec.get("ok") else f"FAILED at {rec['stage']}: {rec['error']}")
            return 0 if rec.get("ok") else 1
    except (Refused, project.ProjectError) as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
