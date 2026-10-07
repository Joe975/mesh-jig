"""Controlled mutations for opt-in character contracts; no paid judge or image generator."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from glbkit import box, cube, write_cubes, _pack
from mesh_jig import brief, evaluate, glb, profile, project, render, resemblance, review, scoring, silhouette, standing


def shots():
    return [
        {"name": "front", "cam_pos": [0, .5, -3], "cam_target": [0, .5, 0], "ortho": 1.4},
        {"name": "rear", "cam_pos": [0, .5, 3], "cam_target": [0, .5, 0], "ortho": 1.4},
        {"name": "side", "cam_pos": [-3, .5, 0], "cam_target": [0, .5, 0], "ortho": 1.4},
        {"name": "right", "cam_pos": [3, .5, 0], "cam_target": [0, .5, 0], "ortho": 1.4},
        {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, .5, 0], "cam_up": [0, 0, 1], "ortho": 1.4},
    ]


def calibrated(tmp_path):
    refs = {}
    for v, bounds in {"front": (20, 10, 59, 109), "side": (35, 10, 54, 109),
                      "rear": (40, 10, 79, 109), "right": (45, 10, 64, 109)}.items():
        im = Image.new("RGB", (100, 120), "white")
        ImageDraw.Draw(im).rectangle(bounds, fill=(50, 80, 120))
        p = tmp_path / f"{v}.png"
        im.save(p)
        refs[v] = str(p)
    config = {"mode": "standing", "calibration": {"view": "front", "height_px": 100,
              "origins_px": {v: [50, 110] for v in refs}},
              "stations": [{"name": "ankle", "z_m": .1}, {"name": "belly", "z_m": .5},
                           {"name": "head", "z_m": .9}]}
    spec = {"height_m": 1, "shots": shots()}
    return refs, spec, config


def model_box(size=(400, 200, 1000), centre=(100, 50, 500)):
    v, f = box(size, centre)
    return v[f]


def test_height_and_explicit_axes_are_opt_in_and_use_glb_axes(tmp_path):
    path = write_cubes(tmp_path / "standing.glb", [cube((.9, 1.2, .85), (0, .6, 0))])
    stats = glb.stats(path)
    assert glb.check(stats, {}) == []
    assert glb.check(stats, {"height_m": 1.2, "axis_extents_m": {"X": .9, "Z": .85}}) == []
    assert "extent along Y" in glb.check(stats, {"height_m": 1.5})[0]
    assert "extent along X" in glb.check(stats, {"axis_extents_m": {"X": .6}})[0]
    # A tall character is valid; it does not acquire the craft longest-axis heuristic.
    assert glb.check(stats, {"height_m": 1.2}) == []
    assert glb.check(stats, {"height_m": 1.2, "forward": "-Z"}) == []


def test_standing_scale_axes_and_asymmetry_are_preserved(tmp_path):
    refs, spec, config = calibrated(tmp_path)
    reference = standing.measure(refs, spec, config)
    assert reference["mm_per_pixel"] == 10
    for v in ("front", "rear"):
        assert [(r["min_mm"], r["max_mm"]) for r in reference["views"][v]["rows"]] == [(-100, 300)] * 3
    for v in ("side", "right"):
        assert [(r["min_mm"], r["max_mm"]) for r in reference["views"][v]["rows"]] == [(-50, 150)] * 3
    diff = standing.compare(reference, standing.from_mesh(model_box(), reference))
    assert diff["mae_mm"] == 0 and diff["height_error_mm"] == 0
    assert "belly" in profile.text(reference) and "min/max/width" in profile.compare_text(diff)


def test_moved_belly_reports_the_known_station_and_missing_head_cannot_improve_mae(tmp_path):
    refs, spec, config = calibrated(tmp_path)
    reference = standing.measure(refs, spec, config)
    shifted = np.concatenate([model_box(), model_box((200, 100, 100), (350, 50, 500))])
    diff = standing.compare(reference, standing.from_mesh(shifted, reference))
    rows = diff["views"]["front"]["rows"]
    assert rows[1]["max_mm"] == 150 and rows[1]["width_mm"] == 150
    assert rows[0]["max_mm"] == rows[2]["max_mm"] == 0
    short = standing.compare(reference, standing.from_mesh(model_box((400, 200, 600), (100, 50, 300)), reference))
    assert short["height_error_mm"] == -400
    assert "front:head" in short["missing_stations"] and short["mae_mm"] is None


def test_reference_views_share_scale_instead_of_fitting_each_height(tmp_path):
    refs, spec, config = calibrated(tmp_path)
    im = Image.new("RGB", (100, 120), "white")
    ImageDraw.Draw(im).rectangle((35, 30, 54, 109), fill=(50, 80, 120))
    im.save(refs["side"])
    ref = standing.measure(refs, spec, config)
    assert ref["views"]["side"]["rows"][-1]["min_mm"] is None  # shortened drawing stays shortened
    # Internal mask downsampling must retain calibration in ORIGINAL image pixels.
    for v, path in refs.items():
        with Image.open(path) as image:
            image.resize((800, 960), Image.Resampling.NEAREST).save(path)
    config["calibration"]["height_px"] *= 8
    config["calibration"]["origins_px"] = {v: [400, 880] for v in refs}
    enlarged = standing.measure(refs, spec, config)
    # At most one downsampled pixel of quantization; never an arbitrary refit.
    for v in refs:
        for a, b in zip(enlarged["views"][v]["rows"], ref["views"][v]["rows"]):
            if b["min_mm"] is None:
                assert a["min_mm"] is None
            else:
                assert a["min_mm"] == pytest.approx(b["min_mm"], abs=3)
                assert a["max_mm"] == pytest.approx(b["max_mm"], abs=3)


def test_glb_profile_axis_conversion_and_project_brief(tmp_path):
    refs, spec, config = calibrated(tmp_path)
    path = write_cubes(tmp_path / "answer.glb", [cube((.4, 1, .2), (.1, .5, -.05))])
    spec["profile"] = config
    proj = project.Project(tmp_path, {"model": spec, "refs": "."})
    assert not proj.problems()
    diff = profile.glb_diff(proj.reference_profile(), path)
    assert diff["mae_mm"] < 1e-4
    assert "STANDING OUTLINES" in brief.text(proj) and "Standing height" in brief.text(proj)


@pytest.mark.parametrize("mutation", ["perspective", "tilt", "bad_height", "missing_origin"])
def test_standing_requires_calibration(tmp_path, mutation):
    refs, spec, config = calibrated(tmp_path)
    if mutation == "perspective":
        del spec["shots"][0]["ortho"]
    elif mutation == "tilt":
        spec["shots"][0]["cam_pos"][1] += .1
    elif mutation == "bad_height":
        spec["height_m"] = float("nan")
    else:
        del config["calibration"]["origins_px"]["front"]
    with pytest.raises(ValueError):
        standing.measure(refs, spec, config)


def test_character_aggregate_catches_front_right_regression_despite_top_improvement():
    weights = scoring.weights({"preset": "character"})
    before = {v: {"iou": .8, "score": .8, "zone": .8} for v in scoring.CHARACTER_VIEWS}
    after = copy.deepcopy(before)
    after["top"] = dict.fromkeys(("iou", "score", "zone"), 1)
    after["front"] = after["right"] = dict.fromkeys(("iou", "score", "zone"), .3)
    get = lambda views, w: evaluate.numeric_measures({"views": views, "ortho_iou":
        scoring.aggregate(views, "iou", dict.fromkeys(silhouette.ORTHO_VIEWS, 1))},
        {"views": views, "ortho": scoring.aggregate(views, "score", dict.fromkeys(silhouette.ORTHO_VIEWS, 1))}, {}, w)
    assert get(after, None)["numeric_score"] > get(before, None)["numeric_score"]
    a, b = get(before, weights), get(after, weights)
    assert b["numeric_score"] < a["numeric_score"] and b["zone"] == b["resemblance"] == b["ortho_iou"] == .64
    assert scoring.weak_views(after, "iou", weights)[0]["view"] == "front"
    assert scoring.aggregate(after, "iou", {"front": 3, "top": 1}) == .475


def test_measure_functions_use_same_weights_and_keep_hero_separate(tmp_path):
    im = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(im).rectangle((20, 10, 80, 90), fill="red")
    views = {v: im for v in (*scoring.CHARACTER_VIEWS, "hero")}
    candidates = dict(views, right=Image.new("RGB", im.size, "white"))
    weights = {"front": 1, "right": 3}
    sil = silhouette.measure_views(candidates, views, weights=weights)
    res = resemblance.measure_views(candidates, views, weights=weights)
    assert sil["ortho_iou"] == .25
    assert sil["ortho_iou"] == scoring.aggregate(sil["views"], "iou", weights)
    assert res["ortho"] == scoring.aggregate(res["views"], "score", weights)
    assert res["score"] == res["ortho"]
    assert "hero" not in sil["weights"]
    assert silhouette.measure_views(candidates, views)["ortho_iou"] == 1  # legacy excludes right


@pytest.mark.parametrize("config", [{}, {"views": {}}, {"views": {"front": -1}},
    {"views": {"front": float("nan")}}, {"preset": "unknown"}, {"views": {"front": True}}])
def test_invalid_scoring_is_rejected(config):
    with pytest.raises(ValueError):
        scoring.weights(config)


def named_cubes(path, parts):
    write_cubes(path, [cube(size, centre) for name, size, centre in parts])
    doc, blob = glb.read_glb(path)
    for node, (name, _, _) in zip(doc["nodes"], parts):
        node["name"] = name
    doc["buffers"] = []
    return _pack(doc, bytearray(blob), path)


def inspect_parts(tmp_path, parts, config):
    path = named_cubes(tmp_path / "parts.glb", parts)
    return review.measure(path, {"shots": shots()}, config, {}, {})


def test_handedness_uses_actual_imported_geometry_not_metadata_claims(tmp_path):
    cfg = {"parts": {"patch": {"nodes": ["Patch"]}}, "landmarks": [
        {"part": "patch", "bounds_m": [[-.3, .1, .6], [-.1, .3, .9]]}]}
    good = inspect_parts(tmp_path, [("Patch", (.1, .1, .1), (-.2, .75, -.2))], cfg)
    mirrored = inspect_parts(tmp_path, [("Patch", (.1, .1, .1), (.2, .75, -.2))], cfg)
    assert good["checks"][0]["passed"] and not mirrored["checks"][0]["passed"]
    # the check keeps the box it was held to, so a failure says which way the part is out and by how much
    assert mirrored["checks"][0]["allowed_m"] == [[-.3, .1, .6], [-.1, .3, .9]]
    assert review.describe(mirrored["checks"][0]) == "it reaches X +0.250 m, 350 mm past the -0.100 its box allows"
    assert review.describe(good["checks"][0]) == ""
    missing = inspect_parts(tmp_path, [("Eye", (.1, .1, .1), (.2, .75, -.2))], cfg)
    assert not any(c["passed"] for c in missing["checks"])
    assert review.describe(missing["checks"][0]) == "declared part missing from GLB"
    assert review.describe(missing["checks"][1]) == "the part has no geometry"


def test_clearance_detects_torso_through_garment_and_floating_insert(tmp_path):
    cfg = {"parts": {"skin": {"nodes": ["Skin"]}, "vest": {"nodes": ["Vest"]}}, "clearance": [
        {"inner": "skin", "outer": "vest", "min_m": .005, "max_m": .03,
         "rays": [{"origin_m": [0, 0, .5], "direction": [0, 1, 0]},
                  {"origin_m": [0, 0, .5], "direction": [0, -1, 0]}]}]}
    skin = ("Skin", (.4, .6, .4), (0, .5, 0))
    good = inspect_parts(tmp_path, [skin, ("Vest", (.44, .6, .44), (0, .5, 0))], cfg)
    bad = inspect_parts(tmp_path, [skin, ("Vest", (.35, .6, .35), (0, .5, 0))], cfg)
    floating = inspect_parts(tmp_path, [skin, ("Vest", (.6, .6, .6), (0, .5, 0))], cfg)
    assert good["checks"][0]["passed"]
    assert not bad["checks"][0]["passed"] and bad["checks"][0]["samples"][0]["gap_m"] < 0
    assert not floating["checks"][0]["passed"]
    assert review.describe(good["checks"][0]) == "gap 20.0 to 20.0 mm"
    assert review.describe(bad["checks"][0]) == (
        "gap -25.0 mm from skin out to vest on the ray from (0, 0, 0.5) m along +Y, allowed 5 to 30 mm; "
        "gap -25.0 mm from skin out to vest on the ray from (0, 0, 0.5) m along -Y, allowed 5 to 30 mm")
    assert review.describe(floating["checks"][0]).startswith("gap +100.0 mm from skin out to vest on the ray from")
    gone = inspect_parts(tmp_path, [skin, ("Vest", (.1, .1, .1), (0, 2, 0))], cfg)       # a vest no probe meets
    assert review.describe(gone["checks"][0]).startswith("the ray from (0, 0, 0.5) m along +Y meets no vest; ")


def test_front_only_buckle_and_hidden_belt(tmp_path):
    cfg = {"parts": {"buckle": {"nodes": ["Buckle"]}}, "visibility": [
        {"part": "buckle", "view": "front", "min_fraction": .8},
        {"part": "buckle", "view": "rear", "max_fraction": .01}]}
    torso = ("Skin", (.4, .7, .4), (0, .5, 0))
    buckle = ("Buckle", (.14, .08, .03), (0, .35, -.23))
    good = inspect_parts(tmp_path, [torso, buckle], cfg)
    assert all(c["passed"] for c in good["checks"]), good
    assert [review.describe(c) for c in good["checks"]] == ["100% shows", "0% shows"]
    rear = inspect_parts(tmp_path, [torso, ("Buckle", (.14, .08, .03), (0, .35, .23))], cfg)
    assert not any(c["passed"] for c in rear["checks"])
    assert [review.describe(c) for c in rear["checks"]] == [
        "0% of it shows in the front view, needs at least 80%", "100% of it shows in the rear view, at most 1% may"]
    hidden = inspect_parts(tmp_path, [torso, buckle, ("Hem", (.3, .2, .02), (0, .35, -.27))], cfg)
    assert not hidden["checks"][0]["passed"] and hidden["checks"][0]["visible_fraction"] == 0


def test_skin_diagnostics_warn_only_when_requested_and_allow_clothing_islands(tmp_path):
    parts = [("SkinA", (.2, .2, .2), (-.2, .5, 0)), ("SkinB", (.2, .2, .2), (.2, .5, 0))]
    cfg = {"parts": {"skin": {"nodes": ["SkinA", "SkinB"], "smooth_min": .5, "max_components": 1}}}
    rec = inspect_parts(tmp_path, parts, cfg)
    assert rec["diagnostics"]["skin"]["varying_corner_normal_fraction"] == 0
    assert rec["diagnostics"]["skin"]["position_welded_components"] == 2 and len(rec["warnings"]) == 2
    cfg["parts"]["skin"] = {"nodes": ["SkinA", "SkinB"]}
    assert inspect_parts(tmp_path, parts, cfg)["warnings"] == []


def test_missing_paw_and_mirrored_face_regions_are_not_normalised_away():
    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 15, 110, 130), fill="red")
    draw.rectangle((28, 80, 49, 105), fill="red")  # paw, preserves major body bounds
    draw.rectangle((111, 80, 132, 105), fill="red")
    draw.rectangle((65, 30, 75, 40), fill="black")
    missing = image.copy()
    ImageDraw.Draw(missing).rectangle((28, 80, 49, 105), fill="white")
    # Tail-like extreme keeps the normalized bbox fixed after paw removal.
    for im in (image, missing):
        ImageDraw.Draw(im).rectangle((28, 115, 49, 130), fill="red")
    cfg = [{"name": "paw", "view": "front", "box": [.04, .52, .24, .8], "min_iou": .8}]
    assert review.regions({"front": image}, {"front": image}, cfg, ())[0]["passed"]
    lost = review.regions({"front": missing}, {"front": image}, cfg, ())[0]
    assert not lost["passed"] and lost["min_iou"] == .8
    assert review.describe(lost) == f"in its box on the front view: outline overlap {lost['iou']:.3f}, needs 0.8"
    assert review.describe(review.regions({"front": image}, {"front": image}, cfg, ())[0]) == "outline 1.00"
    mirrored = image.copy()
    draw = ImageDraw.Draw(mirrored)
    draw.rectangle((65, 30, 75, 40), fill="red")
    draw.rectangle((85, 30, 95, 40), fill="black")
    cfg = [{"name": "patch", "view": "front", "box": [.35, .12, .5, .25], "min_zone": .95}]
    from mesh_jig.palette import zones
    palette = zones({"red": "#ff0000", "black": "#000000"})
    assert review.regions({"front": image}, {"front": image}, cfg, palette)[0]["passed"]
    wrong = review.regions({"front": mirrored}, {"front": image}, cfg, palette)[0]
    assert not wrong["passed"]
    assert review.describe(wrong) == (f"in its box on the front view: colour zones {wrong['zone']:.3f}, needs 0.95 "
                                      f"(outline overlap {wrong['iou']:.2f})")
    # a record made before the limits were kept in it says what it read, and where the limit is written
    older = {k: v for k, v in wrong.items() if k != "min_zone"}
    assert review.describe(older) == (f"in its box on the front view: outline overlap {wrong['iou']:.3f}, colour zones "
                                      f"{wrong['zone']:.3f} (its limits are in the brief's review block)")




def test_evaluate_serializes_character_profiles_weights_review_and_failures(tmp_path):
    root = tmp_path / "character"
    (root / "refs").mkdir(parents=True)
    geometry = [("Skin", (.4, 1, .2), (0, .5, 0)), ("Patch", (.06, .06, .02), (-.12, .8, -.12))]
    truth = named_cubes(root / "truth.glb", geometry)
    spec = {"glb": "character.glb", "height_m": 1, "shots": shots(), "profile": {
        "mode": "standing", "calibration": {"view": "front", "height_px": 180 / 1.4,
            "origins_px": {v: [160, 90 + .5*180/1.4] for v in ("front", "side", "rear", "right")}},
        "stations": [{"name": "belly", "z_m": .5}, {"name": "head", "z_m": .8}]}}
    mesh = render.load(truth)
    for shot in shots():
        render.render(mesh, shot, width=320, height=180).save(root / "refs" / f"{shot['name']}.png")
    cfg = {"parts": {"patch": {"nodes": ["Patch"]}}, "landmarks": [
        {"part": "patch", "bounds_m": [[-.2, .1, .7], [-.05, .2, .9]]}]}
    proj = project.Project(root, {"model": spec, "scoring": {"preset": "character"}, "review": cfg})
    assert not proj.problems()
    def build_fn(script, out, model):
        out.mkdir(parents=True, exist_ok=True)
        named_cubes(out / model["glb"], geometry)
        return ""
    def render_fn(model, out, dest):
        return render.render_views(model, out, dest, size=(320, 180))
    out = root / "attempts" / "answer"
    rec = evaluate.evaluate(proj, root / "model.py", out, build_fn=build_fn, render_fn=render_fn, log=lambda *_: None)
    assert rec["ok"] and rec["numeric"]["ortho_iou"] == rec["numeric"]["resemblance"] == 1
    assert rec["profile_diff"]["mode"] == "standing" and rec["profile_diff"]["mae_mm"] < 10
    assert rec["scoring"]["weights"] == dict.fromkeys(scoring.CHARACTER_VIEWS, 1)
    assert rec["review"]["checks"][0]["passed"]
    said = evaluate.feedback(rec)
    assert "Aggregate scoring weights: front 1, side 1, rear 1, right 1, top 1; weakest overlap views: " in said
    weakest = [line for line in said.splitlines() if "eakest" in line]
    assert len(weakest) == 2 and not any(c in line for line in weakest for c in "{}[]"), "views as words, not as a list"
    assert json.loads((out / "eval.json").read_text())["review"] == rec["review"]
    # A missing selected camera result must fail rather than renormalize the policy away.
    def missing(model, out, dest):
        result = render_fn(model, out, dest)
        del result["renders"]["right"]
        return result
    fail = evaluate.evaluate(proj, root / "model.py", root / "attempts" / "missing", build_fn=build_fn,
                             render_fn=missing, log=lambda *_: None)
    assert not fail["ok"] and fail["stage"] == "measure" and "right" in fail["error"]
    geometry[1] = ("Patch", (.06, .06, .02), (.12, .8, -.12))
    bad = evaluate.evaluate(proj, root / "model.py", root / "attempts" / "mirror", build_fn=build_fn,
                            render_fn=render_fn, log=lambda *_: None)
    assert bad["ok"] and not next(g for g in bad["gates"] if g["name"] == "landmark:patch")["passed"]
    # the verdict leads with the gate, in words, and says the attempt it is held against passes it
    told = evaluate.feedback(bad, evaluate.standing(proj, root / "attempts" / "mirror", bad)).splitlines()
    assert told[0].endswith("; 1 of 2 gates FAILED (landmark:patch)")
    assert told[1] == "gate landmark:patch: FAILED - it reaches X +0.150 m, 200 mm past the -0.050 its box allows"
    assert told[2] == "gates passed: standing_stations"
    assert told[3].endswith("It fails landmark:patch, which answer passes.") and "below the best attempt, answer" in told[3]
    assert said.splitlines()[0].endswith("; all 2 gates pass") and "gates passed: standing_stations; landmark:patch" in said


@pytest.mark.parametrize("spec", [{"height_m": -1}, {"height_m": float("inf")},
    {"height_m": 1, "axis_extents_m": {"Y": 2}}, {"axis_extents_m": {"W": 1}},
    {"axis_tolerances": {"X": -.1}}, {"height_tolerance": -1}, {"height_tolerance": True}])
def test_project_rejects_invalid_dimensions(tmp_path, spec):
    refs, _, _ = calibrated(tmp_path)
    assert project.Project(tmp_path, {"model": spec, "refs": "."}).problems()


def test_missing_selected_refs_and_invalid_review_are_rejected(tmp_path):
    calibrated(tmp_path)
    proj = project.Project(tmp_path, {"model": {"shots": shots()}, "refs": ".", "scoring": {"preset": "character"}})
    assert "top" in str(proj.problems())
    for cfg in ({"parts": {"skin": {"nodes": ["Skin"]}}}, {"regions": [None]},
                {"parts": {"a": {"nodes": ["Same"]}, "b": {"nodes": ["Same"]}}}):
        with pytest.raises(ValueError):
            review.validate(cfg, {"join": True}, {})


def test_ray_hit_policy_reports_outermost_concave_envelope():
    tris = np.concatenate([model_box((100, 100, 100), (0, 100, 500)),
                           model_box((100, 100, 100), (0, 300, 500))]) / 1000
    assert review.ray_distance(tris, [0, 0, .5], [0, 1, 0]) == pytest.approx(.05)
    assert review.ray_distance(tris, [0, 0, .5], [0, 1, 0], "last") == pytest.approx(.35)


def test_partial_part_metadata_and_missing_scored_view_cannot_silently_pass(tmp_path):
    rec = inspect_parts(tmp_path, [("Eye", (.1, .1, .1), (0, .8, -.2))],
                        {"parts": {"eyes": {"nodes": ["Eye", "MissingEye"]}}})
    assert not rec["checks"][0]["passed"]
    with pytest.raises(ValueError, match="missing"):
        scoring.aggregate({"front": {"iou": 1}}, "iou", {"front": 1, "right": 1})


def test_default_direct_evaluate_can_still_render_without_references(tmp_path):
    proj = project.Project(tmp_path, {"model": {}})
    # The legacy empty-measure path expects an existing comparison directory.
    (tmp_path / "attempt" / "compare").mkdir(parents=True)
    rec = evaluate.evaluate(proj, tmp_path / "model.py", tmp_path / "attempt", build_fn=lambda *args: "",
                            render_fn=lambda *args: {"renders": {}, "error": ""}, log=lambda *_: None)
    assert rec["ok"] and rec["numeric"]["numeric_score"] is None


def test_explicit_ground_origin_supports_frozen_midheight_camera_contracts(tmp_path):
    refs, spec, config = calibrated(tmp_path)
    config["origin_m"] = [.02, -.03, -.5]
    reference = standing.measure(refs, spec, config)
    centered = model_box() + np.array([20, -30, -500])
    assert standing.compare(reference, standing.from_mesh(centered, reference))["mae_mm"] == 0
    with pytest.raises(ValueError, match="origin_m"):
        standing.measure(refs, spec, dict(config, origin_m=[0, 0, float("nan")]))
