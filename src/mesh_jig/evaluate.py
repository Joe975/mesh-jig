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

import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

from . import build, criteria, glb, harness, judge as judge_mod, paint, profile, render, resemblance, silhouette, review
from .project import Project

EVAL_FILE = "eval.json"
NUMERIC_FILE = "numeric.json"
ORTHO_VIEWS = silhouette.ORTHO_VIEWS
PATH_FIELDS = ("script", "against")                     # a path each
PATH_MAPS = ("renders", "compare", "overlays", "closeups")      # view -> path; a close-up is region -> path
TEXT_FIELDS = ("error", "pairwise_error")               # what a build or a judge said, which can name a path
LEDGER_HEAD = ("| attempt | build | outline mm | overlap | resemblance | zones | numeric | gates | judge | note |\n"
               "|---|---|---|---|---|---|---|---|---|---|\n")
VIEW_MOVE = 0.005       # `standing_lines` names a view whose overlap or zones moved by this much against another attempt


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


def ledger_line(rec: dict) -> str:
    """One line per attempt, the table an agent reads back: name, verdict, the numbers, the gates, the judge if
    asked. The score is not a mean over the gates, so the line says them beside it."""
    n = rec.get("numeric") or {}
    f = lambda k, d=3: "-" if n.get(k) is None else f"{n[k]:.{d}f}"  # noqa: E731
    pw = rec.get("pairwise")
    verdict = "-" if not pw else f"{pw['win_rate']:.2f}{' decisive' if pw['decisive'] else ''} vs {rec.get('against_name', '?')}"
    status = "ok" if rec.get("ok") else f"FAILED {rec.get('stage', '')}"
    return (f"| {rec.get('name', '')} | {status} | {f('mae_mm', 2)} | {f('ortho_iou')} | {f('resemblance')} | {f('zone', 2)} | "
            f"{f('numeric_score', 4)} | {gate_tally(rec)} | {verdict} | {(rec.get('note') or '').replace('|', '/')} |")


def append_ledger(path: Path, rec: dict) -> None:
    """Append the attempt's line. A ledger begun under other columns gets the header again first, so no row sits
    under a header it does not fit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    heads = [line for line in text.splitlines() if line.startswith("| attempt |")]
    with open(path, "a", encoding="utf-8") as f:
        if not text.strip():
            f.write("# Attempts (mesh-jig eval)\n\n" + LEDGER_HEAD)
        elif not heads or heads[-1] != LEDGER_HEAD.splitlines()[0]:
            f.write(("" if text.endswith("\n") else "\n") + "\n" + LEDGER_HEAD)
        f.write(ledger_line(rec) + "\n")


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


def evaluate(project: Project, script: str | Path, out_dir: str | Path, *, atlas: str | Path | None = None,
             paint_plan: str | Path | None = None, name: str = "", note: str = "", against: str | Path | None = None,
             against_name: str = "", backend=None, samples: int = 16, build_fn: Callable = build.build_variants,
             render_fn: Callable = render.render_views, pairwise_fn: Callable | None = None, log=print) -> dict:
    """Build, render, measure; judge pairwise against another attempt's renders when given a backend. The record,
    also written to out_dir/eval.json with its paths relative to out_dir (`portable`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = project.spec
    rec: dict = {"name": name or out_dir.name, "script": str(script), "note": note, "ok": False, "stage": "",
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "harness": harness.version()}
    t0 = time.time()

    def finish() -> dict:
        rec["wall_s"] = round(time.time() - t0, 1)
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
    sil = silhouette.measure_views(views, refs, compare_dir, weights=weights)
    res = resemblance.measure_views(views, refs, project.zones, out_dir, weights=weights)
    judged = built / glb.judged_glb(spec)
    profile_diff = profile.glb_diff(project.reference_profile(), judged) if judged.is_file() else {}
    marked = review.regions(views, refs, (project.review or {}).get("regions") or [], project.zones)
    numeric = numeric_measures(sil, res, profile_diff, weights, mix=project.score_mix,
                               regions=review.regions_score(marked))
    (out_dir / NUMERIC_FILE).write_text(json.dumps(numeric, indent=1), encoding="utf-8")
    rec.update({"ok": True, "stage": "measured", "renders": shot["renders"], "cropped": shot.get("cropped") or [],
                "compare": render.write_comparisons(views, refs, compare_dir),
                "overlays": {v: str(compare_dir / f"{silhouette.OVERLAY_PREFIX}{v}.png") for v in sil["views"]},
                "numeric": numeric, "silhouette": sil["views"], "resemblance": res["views"],
                "profile_diff": profile_diff, "gates": gates(spec, numeric)})
    if project.zones:
        rec["zone_reading"] = {v: zone_reading(views[v], refs[v], project.zones) for v in sorted(views)}
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
    res = resemblance.measure_views(paired, refs, project.zones, None, weights=weights if whole else None)
    marked = [c for c in review.regions(paired, refs, (project.review or {}).get("regions") or [], project.zones)
              if "error" not in c]              # a region whose view was not drawn is not read, not failed
    rec: dict = {"ok": True, "name": attempt.name, "renders": shot["renders"], "cropped": shot.get("cropped") or [],
                 "compare": render.write_comparisons(paired, refs, out),
                 "overlays": {v: str(out / f"{silhouette.OVERLAY_PREFIX}{v}.png") for v in sil["views"]},
                 "silhouette": sil["views"], "resemblance": res["views"], "gates": [], "regions": marked,
                 "numeric": numeric_measures(sil, res, {}, weights, mix=project.score_mix,
                                             regions=review.regions_score(marked)) if whole and paired else {}}
    if project.zones:
        rec["zone_reading"] = {v: zone_reading(paired[v], refs[v], project.zones) for v in sorted(paired)}
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
        lines += standing_lines(rec, stands, scored) if stands and stands.get("best") else []
    else:
        lines = [f"[{rec['name']}] a free look, not an attempt: not every scored view was drawn, so there is no "
                 "score. Leave `views` out to have one."]
    lines += region_lines(rec)
    lines += view_lines(rec, scored)
    lines += zone_reading_lines(rec.get("zone_reading") or {}, scored)
    lines += picture_lines(rec, scored)
    return "\n".join(lines)


def zone_reading(render_path, reference, zones) -> dict | None:
    """How a render's cells read as palette colours against the reference's, over the cells the zone score counts
    (inside both silhouettes): the share of each colour on each side, and for each reference colour what the render
    shows there. It is `resemblance`'s own classification (`normalised`, `zone_letters`) reported, not a new measure:
    the zone score alone says how many cells disagree, never which paint reads as which colour."""
    ra, ma = resemblance.normalised(render_path)
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
    others = []
    for f in project.attempts_dir.glob(f"*/{EVAL_FILE}"):
        if f.parent.resolve() == here:
            continue
        try:
            other = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        score = (other.get("numeric") or {}).get("numeric_score")
        if other.get("ok") and score is not None:
            others.append((-float(score), str(other.get("started") or ""), f.parent.name, other))
    others.sort(key=lambda o: o[:3])                    # highest first; of two that tie, the one measured first

    def entry(o) -> dict:
        r = o[3]
        sil, res = r.get("silhouette") or {}, r.get("resemblance") or {}
        return {"name": o[2], "score": -o[0], "failed": failed_gates(r), "gates": len(acceptance_gates(r)),
                "views": {v: {"iou": (sil.get(v) or {}).get("iou"), "zone": (res.get(v) or {}).get("zone")}
                          for v in dict.fromkeys(list(sil) + list(res))}}
    passing = [o for o in others if acceptance_gates(o[3]) and not failed_gates(o[3])]
    return {"others": len(others), "best": entry(others[0]) if others else None,
            "clean": entry(passing[0]) if passing else None}


def standing_lines(rec: dict, st: dict | None, scored: list[str]) -> list[str]:
    """`standing` as the agent reads it: how far the attempt is from the best of the others, which gates the two
    differ on, the highest attempt that passes every gate when this one does not, and the scored views that moved.
    The score is not a mean over the gates or the views, so a change that gained a little by trading one of them
    away reads here as a trade."""
    if st is None:
        return []
    best = st.get("best")
    if not best:
        return ["standing: the first measured attempt here."]
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
    clean = st.get("clean")
    if mine and clean and clean["name"] != best["name"]:
        line += f" The highest attempt that passes every gate is {clean['name']} ({clean['score']:.4f})."
    lines = [line]
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
    return ["REGIONS (each marked region read on its own, twice as fine as a whole view; weakest first"
            + mean + (", which the score counts" if counted else "") + "): " + said]


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
        return f"[{name}] FAILED at {rec.get('stage', '?')}:\n{rec.get('error', '')}"
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
    cropped = rec.get("cropped") or []
    cut = (f"the subject touches the frame edge in {cropped}: it is cut off there, so the overlap for that view is "
           "understated")
    if any(v in scored for v in cropped):               # a view the score is taken over: the number above is off
        lines.append(cut)
    lines += gate_lines(rec)
    lines += standing_lines(rec, stands, scored)
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
    for warning in (rec.get("review") or {}).get("warnings", []):
        lines.append("review warning: " + warning)
    if (rec.get("review") or {}).get("diagnostics"):
        lines.append(parts_line(rec["review"]["diagnostics"]))
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
