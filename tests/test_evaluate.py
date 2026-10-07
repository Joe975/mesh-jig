"""mesh_jig.evaluate: the one call an agent makes per attempt, here with a fake build and a fake judge. The render
and every measure are the real ones."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from glbkit import cube, write_cubes
from mesh_jig import build, cli, evaluate, project, render

LENGTH = 0.08
SHOTS = {s["name"]: s for s in render.DEFAULT_SHOTS}
RED, PALE = (0.6, 0.05, 0.03), (0.8, 0.76, 0.66)


def ship(path, span=0.04, nose=RED):
    """A slab 80 mm long along Z and `span` wide, pale, with a coloured nose block on top at the -Z end."""
    plain = {k: PALE for k in ("+x", "-x", "+y", "-y", "+z", "-z")}
    return write_cubes(path, [cube((span, 0.012, LENGTH), colours=plain),
                              cube((span * 0.6, 0.01, 0.02), (0.0, 0.011, -0.03), colours={k: nose for k in plain})])


def make_project(tmp_path: Path, **model) -> project.Project:
    """A project whose reference views are renders of the 40 mm wide ship: a known answer."""
    root = tmp_path / "proj"
    (root / "refs").mkdir(parents=True)
    truth = ship(tmp_path / "truth.glb")
    mesh = render.load(truth)
    for view in ("hero", "top", "side", "rear"):
        render.render(mesh, SHOTS[view], scale=1 / LENGTH).save(root / "refs" / f"{view}.png")
    (root / "brief.md").write_text("A pale slab with a red nose block.\n")
    (root / "jig.json").write_text(json.dumps({
        "name": "ship", "palette": {"red": "#c83c30", "pale": "#e6e0d2"},
        "model": {"script": "ship.py", "glb": "ship.glb", "length_m": LENGTH, "forward": "-Z", **model}}))
    return project.load(root)


def fake_build(span=0.04, nose=RED, error=""):
    def build_fn(script, out_dir, spec, **kw):
        if error:
            return error
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ship(Path(out_dir) / "ship.glb", span, nose)
        return ""
    return build_fn


def script(tmp_path, name="a01") -> Path:
    p = tmp_path / "proj" / "attempts" / name / "ship.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("def build():\n    pass\n")
    return p


def quick_render(spec, glb_dir, out_dir, views=None):
    """The real renderer at a quarter of the pixels: the measures work on masks of at most 512 px either way."""
    return render.render_views(spec, glb_dir, out_dir, views=views, size=render.PREVIEW_SIZE)


def run(proj, tmp_path, name="a01", **kw):
    s = script(tmp_path, name)
    return evaluate.evaluate(proj, s, s.parent, log=lambda *_: None, render_fn=quick_render, **kw)


def test_the_known_answer_measures_as_a_match_and_everything_is_written(tmp_path):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, note="the reference's own geometry", build_fn=fake_build())
    n = rec["numeric"]
    assert rec["ok"] and rec["stage"] == "measured"
    assert n["ortho_iou"] > 0.97 and n["zone"] > 0.97 and n["resemblance"] > 0.97, n
    # the references are perspective renders, so the end stations and a raised part read a little off; the body
    # stations of an exact match read zero
    assert n["mae_mm"] < 2.0, n
    assert all(abs(r["half_width"]) < 0.5 for r in rec["profile_diff"]["top"][1:-1])
    attempt = tmp_path / "proj" / "attempts" / "a01"
    for rel in ("eval.json", "numeric.json", "resemblance.json", "renders/top.png", "compare/top_vs_reference.png",
                "compare/sil_top.png", "compare/silhouette.json"):
        assert (attempt / rel).is_file(), rel
    assert json.loads((attempt / "eval.json").read_text())["numeric"] == n
    text = evaluate.feedback(rec)
    assert "built, in contract, rendered and measured" in text and "top_vs_reference.png" in text
    assert "YOUR MODEL AGAINST THE REFERENCE OUTLINES" in text


def test_the_feedback_says_which_views_the_score_is_taken_on():
    views = dict.fromkeys(("front", "hero", "rear", "side", "top"), {"iou": 0.5, "missing": 0.1, "extra": 0.1})
    assert evaluate.scored_views_lines(views) == [
        "overlap, resemblance, zones and the numeric score are the means over top, side, rear ONLY; front, hero are "
        "compared for you to look at and do not count (the fallback views: jig.json states no `criteria.views`)"]
    assert "hero is compared for you to look at and does not count" in evaluate.scored_views_lines(
        dict.fromkeys(("hero", "top"), {}))[0]
    assert evaluate.scored_views_lines(dict.fromkeys(("top", "side", "rear"), {})) == []       # all of them count
    assert evaluate.scored_views_lines({"hero": {}}) == []                                     # the only view counts
    rec = {"ok": True, "name": "a01", "numeric": {}, "silhouette": views, "resemblance": {}}
    assert "top, side, rear ONLY" in evaluate.feedback(rec)


def test_the_feedback_says_how_the_renders_colours_read_against_the_references(tmp_path):
    proj = make_project(tmp_path)
    same = run(proj, tmp_path, build_fn=fake_build())["zone_reading"]["top"]
    assert same["reference"] == same["yours"] and list(same["reference"]) == ["pale", "red"]
    assert same["read_as"] == {"red": {"red": 1.0}, "pale": {"pale": 1.0}}
    rec = run(proj, tmp_path, "a02", build_fn=fake_build(nose=PALE))            # the red nose block painted pale
    top = rec["zone_reading"]["top"]
    assert top["yours"] == {"pale": 1.0} and top["read_as"]["red"] == {"pale": 1.0} and top["cells"] == same["cells"]
    assert json.loads((tmp_path / "proj" / "attempts" / "a02" / "eval.json").read_text())["zone_reading"]["top"] == top
    text = evaluate.feedback(rec)
    assert "HOW YOUR COLOURS READ" in text and "where the reference is red, yours reads pale 100%" in text
    red = top["reference"]["red"]
    assert f"  top: the reference is pale {1 - red:.0%}, red {red:.0%}; yours reads pale 100%" in text
    assert "  hero:" not in text                                               # only the views the score is taken on


def test_a_wrong_shape_and_a_wrong_colour_each_move_their_own_number(tmp_path):
    proj = make_project(tmp_path)
    right = run(proj, tmp_path, "right", build_fn=fake_build())["numeric"]
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))["numeric"]
    recoloured = run(proj, tmp_path, "pale_nose", build_fn=fake_build(nose=PALE))["numeric"]
    assert wide["ortho_iou"] < right["ortho_iou"] - 0.1 and wide["mae_mm"] > right["mae_mm"] + 2
    assert recoloured["zone"] < right["zone"] - 0.02, "the nose block is the wrong colour: zones see it"
    assert abs(recoloured["ortho_iou"] - right["ortho_iou"]) < 0.01, "and the outline does not"
    assert wide["numeric_score"] < right["numeric_score"] and recoloured["numeric_score"] < right["numeric_score"]


def test_a_build_failure_is_a_failed_record_not_a_crash(tmp_path):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, build_fn=fake_build(error="variant default: JIG_BUILD_ERROR: boom"))
    assert not rec["ok"] and rec["stage"] == "build" and "boom" in rec["error"]
    assert "FAILED build" in evaluate.ledger_line(rec) and "FAILED at build" in evaluate.feedback(rec)
    assert json.loads((tmp_path / "proj" / "attempts" / "a01" / "eval.json").read_text())["ok"] is False


def test_a_model_no_camera_can_see_fails_at_render(tmp_path):
    proj = make_project(tmp_path)

    def tiny(script, out_dir, spec, **kw):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_cubes(Path(out_dir) / "ship.glb", [cube(0.001, (0.0, 5.0, 0.0))])     # a speck, five metres up
        return ""
    rec = run(proj, tmp_path, build_fn=tiny)
    assert not rec["ok"] and rec["stage"] == "render" and "NOTHING VISIBLE" in rec["error"]


def test_the_ledger_gets_one_header_and_one_line_per_attempt(tmp_path):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, note="baseline | first", build_fn=fake_build())
    evaluate.append_ledger(proj.ledger, rec)
    evaluate.append_ledger(proj.ledger, rec)
    text = proj.ledger.read_text()
    assert text.count("| attempt |") == 1 and text.count("| a01 | ok |") == 2
    assert "baseline / first" in text, "a pipe in a note would break the table"


def test_the_contracts_limits_on_the_measures_are_recorded_as_gates(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0, placement_min=0.9)
    good = run(proj, tmp_path, "good", build_fn=fake_build())
    bad = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    assert {g["name"]: g["passed"] for g in good["gates"]} == {"profile_mae_max": True, "placement_min": True}
    assert {g["name"]: g["passed"] for g in bad["gates"]}["profile_mae_max"] is False
    assert bad["ok"], "a failed gate is a measured attempt, reported, not a build failure"
    assert "gate profile_mae_max: FAILED" in evaluate.feedback(bad)
    assert evaluate.gates({}, good["numeric"]) == [], "no limit set: recorded, nothing gated"


def test_the_verdict_comes_first_and_a_failed_gate_says_what_it_needed(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0, placement_min=0.5)
    good = evaluate.feedback(run(proj, tmp_path, "good", build_fn=fake_build())).splitlines()
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    n, said = wide["numeric"], evaluate.feedback(wide).splitlines()
    # four decimals: late in a run two attempts differ in the fourth, and at three they read as a tie
    assert said[0].endswith(f"numeric score {n['numeric_score']:.4f}; 1 of 2 gates FAILED (profile_mae_max)")
    assert said[1] == f"gate profile_mae_max: FAILED - outline error (mm) {n['mae_mm']:.3f} (max 2.0)"
    assert said[2] == f"gates passed: placement_min (colour zones in place {n['zone']:.3f}, min 0.5)"
    assert good[0].endswith("; all 2 gates pass") and good[1].startswith("gates passed: profile_mae_max (outline error")
    # a reader that keeps the first lines has the verdict: the views and the outline table come after it
    first_view = next(i for i, line in enumerate(said) if line.startswith("top: silhouette overlap"))
    table = next(i for i, line in enumerate(said) if line.startswith("YOUR MODEL AGAINST THE REFERENCE OUTLINES"))
    assert 2 < first_view < table and not any(line.startswith("gate") for line in said[3:])
    assert "{" not in "\n".join(said), "no record is printed as a dict"
    plain = run(make_project(tmp_path / "plain"), tmp_path / "plain", build_fn=fake_build())
    assert evaluate.feedback(plain).splitlines()[0].endswith(f"numeric score {plain['numeric']['numeric_score']:.4f}")


def test_each_view_is_one_line_scored_views_first_and_the_pictures_directory_is_named_once(tmp_path):
    proj = make_project(tmp_path)
    text = evaluate.feedback(run(proj, tmp_path, build_fn=fake_build()))
    views = [line.split(":")[0] for line in text.splitlines() if "silhouette overlap" in line]
    assert views == ["top", "side", "rear", "hero (not scored)"], "hero is compared to be looked at, and says so"
    assert all("colour zones agree" in line for line in text.splitlines() if "silhouette overlap" in line)
    compare = str(tmp_path / "proj" / "attempts" / "a01" / "compare")
    assert text.count(compare) == 1
    assert (f"LOOK AT (reference left, your build right) in {compare}: top_vs_reference.png, side_vs_reference.png, "
            "rear_vs_reference.png; not scored: hero_vs_reference.png") in text
    assert ("outline diffs (grey both, red add body, blue remove body), same directory: sil_top.png, sil_side.png, "
            "sil_rear.png; not scored: sil_hero.png") in text


def test_an_attempt_is_told_where_it_stands_among_the_others(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0)
    at = lambda name: tmp_path / "proj" / "attempts" / name    # noqa: E731
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    assert evaluate.standing(proj, at("wide"), wide) == {"others": 0, "best": None, "clean": None}
    assert "standing: the first measured attempt here." in evaluate.feedback(wide, evaluate.standing(proj, at("wide"), wide))
    assert "standing" not in evaluate.feedback(wide), "nothing is claimed of the others when they were not read"
    good = run(proj, tmp_path, "good", build_fn=fake_build())
    gain = good["numeric"]["numeric_score"] - wide["numeric"]["numeric_score"]
    said = evaluate.feedback(good, evaluate.standing(proj, at("good"), good)).splitlines()
    line = next(i for i, x in enumerate(said) if x.startswith("standing:"))
    assert said[line] == (f"standing: the best of the 2 measured attempts, {gain:.4f} above wide "
                          f"({wide['numeric']['numeric_score']:.4f}). It passes profile_mae_max, which wide fails.")
    top = (good["silhouette"]["top"]["iou"], wide["silhouette"]["top"]["iou"])
    assert said[line + 1].startswith(f"  by scored view against wide: top overlap {top[0]:.3f} (was {top[1]:.3f})")
    assert "hero" not in said[line + 1], "only the views the score is taken over"
    back = evaluate.feedback(wide, evaluate.standing(proj, at("wide"), wide))
    assert (f"standing: {gain:.4f} below the best attempt, good ({good['numeric']['numeric_score']:.4f}). "
            "It fails profile_mae_max, which good passes.") in back
    failed = run(proj, tmp_path, "broken", build_fn=fake_build(error="JIG_BUILD_ERROR"))
    assert evaluate.standing(proj, at("broken"), failed) is None
    assert evaluate.standing(proj, at("good"), good)["others"] == 1, "a failed build is not one of the others"


def test_standing_names_the_highest_attempt_that_passes_every_gate(tmp_path):
    proj = project.Project(tmp_path, {"model": {}})

    def put(name, score, failed=(), started="2026-01-02 00:00:00"):
        rec = {"name": name, "ok": True, "started": started, "numeric": {"numeric_score": score},
               "gates": [{"name": g, "passed": g not in failed, "message": ""} for g in ("face", "tail")]}
        (proj.attempts_dir / name).mkdir(parents=True)
        (proj.attempts_dir / name / "eval.json").write_text(json.dumps(rec))
        return rec
    put("a1", 0.60)
    put("a2", 0.70, failed=("face",))
    put("a0", 0.70, failed=("face",), started="2026-01-01 00:00:00")    # ties with a2 and was measured first
    mine = put("a3", 0.69, failed=("tail",))
    st = evaluate.standing(proj, proj.attempts_dir / "a3", mine)
    assert (st["others"], st["best"]["name"], st["clean"]["name"]) == (3, "a0", "a1")
    assert evaluate.standing_lines(mine, st, []) == [
        "standing: 0.0100 below the best attempt, a0 (0.7000). It fails tail, which a0 passes. It passes face, which "
        "a0 fails. The highest attempt that passes every gate is a1 (0.6000)."]
    level = put("a4", 0.70, failed=("face",))
    assert evaluate.standing_lines(level, evaluate.standing(proj, proj.attempts_dir / "a4", level), []) == [
        "standing: level with a0 (0.7000), the best of the other attempts. The highest attempt that passes every "
        "gate is a1 (0.6000)."]


def test_the_ledger_says_the_gates_and_starts_its_table_again_under_older_columns(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0)
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    good = run(proj, tmp_path, "good", build_fn=fake_build())
    assert evaluate.gate_tally(wide) == "0/1 (profile_mae_max)" and evaluate.gate_tally(good) == "1/1"
    assert f"| {wide['numeric']['numeric_score']:.4f} | 0/1 (profile_mae_max) | - |" in evaluate.ledger_line(wide)
    assert evaluate.gate_tally({"ok": False}) == "-" and evaluate.gate_tally({"gates": []}) == "-"
    old = ("# Attempts (mesh-jig eval)\n\n"
           "| attempt | build | outline mm | overlap | resemblance | zones | numeric | judge | note |\n"
           "|---|---|---|---|---|---|---|---|---|\n| a00 | ok | 1.00 | 0.900 | 0.500 | 0.50 | 0.700 | - | before |\n")
    proj.ledger.write_text(old)
    evaluate.append_ledger(proj.ledger, wide)
    evaluate.append_ledger(proj.ledger, good)
    text = proj.ledger.read_text()
    assert text.startswith(old) and text.count(evaluate.LEDGER_HEAD) == 1, "the new columns get their own header, once"
    assert text.index(evaluate.LEDGER_HEAD) < text.index("| wide | ok |") < text.index("| good | ok |")
    assert len(evaluate.ledger_line(good).split("|")) == len(evaluate.LEDGER_HEAD.splitlines()[0].split("|"))


def test_show_reads_an_evaluation_back_without_building_or_touching_the_ledger(tmp_path, capsys, monkeypatch):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, "a01", build_fn=fake_build())
    attempt = tmp_path / "proj" / "attempts" / "a01"
    evaluate.append_ledger(proj.ledger, rec)
    ledger, record = proj.ledger.read_text(), (attempt / "eval.json").read_text()
    monkeypatch.setattr(build, "build_variants", lambda *a, **kw: pytest.fail("show builds nothing"))
    monkeypatch.setattr(render, "render_views", lambda *a, **kw: pytest.fail("show renders nothing"))
    said = evaluate.feedback(rec, evaluate.standing(proj, attempt, rec)).splitlines()
    for named in ("a01", str(attempt)):                       # by its name under attempts/, or by its directory
        assert cli.main(["show", str(proj.root), named]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == said[0] and out[1] == "standing: the first measured attempt here."
        assert any(line.startswith("LOOK AT") for line in out) and out[-1].startswith("-> ")
    assert proj.ledger.read_text() == ledger and (attempt / "eval.json").read_text() == record
    assert cli.main(["show", str(proj.root), "a09"]) == 2 and "no eval.json under a09" in capsys.readouterr().err


def test_the_textured_contract_wants_an_atlas_and_a_flat_one_refuses_it(tmp_path):
    flat = make_project(tmp_path)
    atlas = tmp_path / "atlas.png"
    Image.new("RGB", (8, 8), (200, 60, 50)).save(atlas)
    rec = run(flat, tmp_path, "flat", atlas=atlas, build_fn=fake_build())
    assert rec["stage"] == "build" and "not `textured`" in rec["error"]
    textured = make_project(tmp_path / "t", textured=True)
    s = tmp_path / "t" / "proj" / "attempts" / "a01" / "ship.py"
    s.parent.mkdir(parents=True)
    s.write_text("def build():\n    pass\n")
    rec = evaluate.evaluate(textured, s, s.parent, build_fn=fake_build(), render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "build" and "give an atlas PNG or a paint plan" in rec["error"]
    plan = s.parent / "atlas_plan.json"
    plan.write_text(json.dumps({"size": 16, "ops": [{"op": "blur"}]}))
    rec = evaluate.evaluate(textured, s, s.parent, paint_plan=plan, build_fn=fake_build(), render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "paint" and "PAINT_ERROR" in rec["error"]
    plan.write_text(json.dumps({"size": 16, "ops": [{"op": "rect", "box": [0, 0, 1, 1], "color": "red"}]}))
    seen = {}

    def build_fn(script, out_dir, spec, **kw):
        seen.update(kw)
        return fake_build()(script, out_dir, spec)
    rec = evaluate.evaluate(textured, s, s.parent, paint_plan=plan, build_fn=build_fn, render_fn=quick_render, log=lambda *_: None)
    assert rec["ok"] and Path(seen["atlas"]) == s.parent / "out" / "atlas.png"
    assert Image.open(seen["atlas"]).getpixel((8, 8)) == (0xc8, 0x3c, 0x30), "painted with the project's palette"


def test_the_judge_runs_only_when_given_a_backend_and_only_on_views_the_reference_has(tmp_path):
    proj = make_project(tmp_path)
    prev = run(proj, tmp_path, "a00", build_fn=fake_build(span=0.07))
    Image.new("RGB", (8, 8)).save(Path(prev["renders"]["top"]).parent / "extra.png")     # a view with no reference
    calls = []

    def pairwise_fn(backend, refs, cand, inc, brief, out_dir, samples=8):
        calls.append((sorted(cand), sorted(inc), brief.strip(), samples))
        return {"win_rate": 0.85, "ci": [0.6, 1.0], "decisive": True, "ties": 0, "judgements": 32, "calls": 8,
                "position_bias": 0.5, "views": {"top": 0.9}, "usage": {"cost": 0.02}, "reasons": {"top": ["narrower, as the reference"]}}

    against = tmp_path / "proj" / "attempts" / "a00"
    rec = run(proj, tmp_path, "a01", build_fn=fake_build(), against=against, backend=object(), samples=8, pairwise_fn=pairwise_fn)
    views = ["hero", "rear", "side", "top"]
    assert calls == [(views, views, "A pale slab with a red nose block.", 8)]
    assert rec["better"] and rec["pairwise"]["win_rate"] == 0.85 and rec["pairwise"]["cost"] == 0.02
    assert rec["against_name"] == "a00" and "0.85 decisive vs a00" in evaluate.ledger_line(rec)
    assert "DECISIVE, a01 is closer" in evaluate.feedback(rec) and "narrower, as the reference" in evaluate.feedback(rec)
    # no backend: the numbers are measured, no model is called
    rec2 = run(proj, tmp_path, "a02", build_fn=fake_build(), against=against, pairwise_fn=pairwise_fn)
    assert rec2["ok"] and "pairwise" not in rec2 and len(calls) == 1
    empty = tmp_path / "nothing"
    empty.mkdir()
    rec3 = run(proj, tmp_path, "a03", build_fn=fake_build(), against=empty, backend=object(), pairwise_fn=pairwise_fn)
    assert "no view PNGs" in rec3["pairwise_error"] and len(calls) == 1


def test_eval_json_holds_paths_relative_to_the_attempt_and_read_gives_them_back(tmp_path):
    proj = make_project(tmp_path)
    run(proj, tmp_path, "a00", build_fn=fake_build(span=0.07))
    a00, a01 = (tmp_path / "proj" / "attempts" / n for n in ("a00", "a01"))
    rec = run(proj, tmp_path, "a01", build_fn=fake_build(), against=a00, backend=object(),
              pairwise_fn=lambda *a, **kw: {"win_rate": 0.8, "decisive": True, "ci": [0.6, 0.9]})
    text = (a01 / "eval.json").read_text()
    for spelt in (str(tmp_path), tmp_path.as_posix(), json.dumps(str(tmp_path))[1:-1]):
        assert spelt not in text, "a record names no machine"
    stored = json.loads(text)
    assert stored["script"] == "ship.py" and stored["against"] == "../a00"
    assert stored["renders"]["top"] == "renders/top.png" and stored["compare"]["top"] == "compare/top_vs_reference.png"
    assert stored["overlays"]["rear"] == "compare/sil_rear.png"
    assert {k: v for k, v in stored.items() if k not in evaluate.PATH_FIELDS + evaluate.PATH_MAPS} == {
        k: v for k, v in rec.items() if k not in evaluate.PATH_FIELDS + evaluate.PATH_MAPS}, "only the paths differ"
    # what evaluate returned is absolute, for a caller to open, and `read` is the same record
    assert Path(rec["renders"]["top"]).is_absolute() and Path(rec["renders"]["top"]).is_file()
    back = evaluate.read(a01)
    assert {k: Path(back[k]) for k in evaluate.PATH_FIELDS} == {k: Path(rec[k]) for k in evaluate.PATH_FIELDS}
    maps = [k for k in evaluate.PATH_MAPS if k in rec]          # close-ups are of a project that marks regions
    assert maps == ["renders", "compare", "overlays"]
    assert {k: {v: Path(p) for v, p in back[k].items()} for k in maps} == {
        k: {v: Path(p) for v, p in rec[k].items()} for k in maps}
    # the attempt copied somewhere else still finds its own pictures
    moved = tmp_path / "elsewhere" / "a01"
    moved.parent.mkdir()
    a01.rename(moved)
    assert Path(evaluate.read(moved)["overlays"]["top"]) == moved / "compare" / "sil_top.png"
    assert Path(evaluate.read(moved)["overlays"]["top"]).is_file()


def test_a_path_outside_the_project_is_kept_by_name_and_an_error_text_loses_the_roots(tmp_path):
    proj = make_project(tmp_path)
    attempt = tmp_path / "proj" / "attempts" / "a01"
    outside = tmp_path / "scratch" / "renders"
    rec = {"script": str(attempt / "ship.py"), "against": str(outside), "renders": {"top": str(attempt / "renders" / "top.png")},
           "error": f'File "{attempt / "out" / "ship.py"}", line 3\nno such file: {proj.root / "refs" / "top.png"} in {proj.root.as_posix()}/refs'}
    stored = evaluate.portable(rec, attempt, proj.root)
    assert stored["against"] == "renders" and stored["script"] == "ship.py" and stored["renders"] == {"top": "renders/top.png"}
    assert stored["error"].replace("\\", "/") == 'File "out/ship.py", line 3\nno such file: refs/top.png in refs'
    assert rec["against"] == str(outside), "the record in hand is not changed"
    # an attempt directory outside the project (eval --out) keeps what is inside it, and names the rest
    away = evaluate.portable({"script": str(attempt / "ship.py"), "compare": {"top": str(outside / "top_vs_reference.png")}},
                             outside, proj.root)
    assert away == {"script": "ship.py", "compare": {"top": "top_vs_reference.png"}}


def test_a_record_written_with_absolute_paths_reads_as_it_was(tmp_path):
    attempt = tmp_path / "a01"
    attempt.mkdir()
    old = {"name": "a01", "ok": True, "script": str(tmp_path / "old" / "ship.py"),
           "overlays": {"top": str(tmp_path / "old" / "compare" / "sil_top.png")}}
    (attempt / "eval.json").write_text(json.dumps(old))
    assert evaluate.read(attempt) == old


def test_the_record_names_the_harness_that_measured_it(tmp_path, monkeypatch):
    from mesh_jig import harness
    monkeypatch.setattr(harness, "version", lambda: {"commit": "abc1234", "dirty": True})
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, build_fn=fake_build())
    assert rec["harness"] == {"commit": "abc1234", "dirty": True}
    assert evaluate.read(tmp_path / "proj" / "attempts" / "a01")["harness"] == rec["harness"]
    monkeypatch.setattr(harness, "version", lambda: None)
    failed = run(proj, tmp_path, "a02", build_fn=fake_build(error="JIG_BUILD_ERROR"))
    assert not failed["ok"] and failed["harness"] is None, "a failed attempt records it too; None outside a checkout"


def test_a_stated_score_is_named_with_its_terms_and_the_regions_are_listed_weakest_first():
    n = {"mae_mm": None, "ortho_iou": 0.8, "resemblance": 0.5, "zone": 0.6, "colour": 0.7, "edges": 0.4, "regions": 0.5,
         "mix": {"overlap": 1.0, "colour": 1.0, "edges": 1.0, "regions": 1.0}}
    assert evaluate.numeric_score(n) == 0.6
    assert evaluate.numeric_score({**n, "mix": {"overlap": 3.0, "regions": 1.0}}) == 0.725
    assert evaluate.numeric_score({**n, "regions": None}) == round((0.8 + 0.7 + 0.4) / 3, 4), "a term with no number"
    assert evaluate.numeric_score({k: v for k, v in n.items() if k != "mix"}) == 0.65, "none stated: as it was"
    n["numeric_score"] = 0.6
    assert evaluate.numbers_text(n) == ("overlap 0.800, resemblance 0.500, zones 0.60, colour 0.700, edges 0.400, "
                                        "regions 0.500, numeric score 0.6000 (the mean of overlap, colour, edges and "
                                        "regions)")
    plain = {k: v for k, v in n.items() if k != "mix"}
    assert evaluate.numbers_text(plain) == "overlap 0.800, resemblance 0.500, zones 0.60, numeric score 0.6000"
    checks = [{"name": "tail", "score": 0.7, "colour": 0.8, "edges": 0.6, "passed": True},
              {"name": "face", "score": 0.4, "colour": 0.4, "edges": None, "passed": False},
              {"name": "part:skin", "passed": True}]
    line = ("REGIONS (each marked region read on its own, twice as fine as a whole view; weakest first, their mean "
            "0.500{}): face 0.40 (colours 0.40 close); tail 0.70 (colours 0.80 close, edges in place 0.60)")
    assert evaluate.region_lines({"numeric": n, "review": {"checks": checks}}) == [line.format(", which the score counts")]
    assert evaluate.region_lines({"numeric": plain, "regions": checks}) == [line.format("")], "a preview keeps its own"
    assert evaluate.region_lines({"numeric": n, "review": {"checks": checks[2:]}}) == []
