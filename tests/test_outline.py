"""mesh_jig.outline: the outline table of any subject, read through the project's cameras, and a GLB's own outline
against it. No view name, no length and no calibration is assumed: only orthographic shots that look along an axis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from glbkit import box, cube, write_cubes
from mesh_jig import evaluate, outline, profile, project, render

PAPER, INK = (235, 234, 230), (40, 44, 50)
# glTF axes: front is -Z, so the front camera stands at -Z and the model's +X is on the picture's LEFT
FRONT = {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.3}
SIDE = {"name": "side", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.3}
REAR = {"name": "rear", "cam_pos": [0, 0, 3], "cam_target": [0, 0, 0], "ortho": 1.3}
TOP = {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "cam_up": [0, 0, 1], "ortho": 1.3}
HERO = {"name": "hero", "cam_pos": [-2.4, 1.35, -3.7], "cam_target": [0, 0, 0], "ortho": 1.3}
SHOTS = [FRONT, SIDE, REAR, TOP, HERO]


def draw(path, w, h, extra=()):
    """A w x h block on paper, plus any (x0, y0, x1, y1) blocks given relative to its top-left corner."""
    blocks = [(0, 0, w, h), *extra]
    left, top = min(b[0] for b in blocks), min(b[1] for b in blocks)
    im = Image.new("RGB", (max(b[2] for b in blocks) - left + 120, max(b[3] for b in blocks) - top + 120), PAPER)
    for x0, y0, x1, y1 in blocks:
        im.paste(INK, (60 + x0 - left, 60 + y0 - top, 60 + x1 - left, 60 + y1 - top))
    im.save(path)
    return str(path)


def box_refs(tmp_path) -> dict:
    """A box 20 across X, 10 deep along Y and 40 tall along Z, drawn at 5 px a millimetre."""
    return {"front": draw(tmp_path / "front.png", 100, 200), "side": draw(tmp_path / "side.png", 50, 200),
            "top": draw(tmp_path / "top.png", 100, 50)}


def slab(size, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A box in model millimetres as triangles, (12, 3, 3)."""
    verts, faces = box(size, centre)
    return verts.astype(float)[faces.astype(int)]


def test_a_view_is_read_by_its_camera_not_by_its_name():
    assert outline.view_axes(FRONT) == ((0, -1.0), (2, 1.0)), "picture-right is -X, picture-up is +Z"
    assert outline.view_axes(SIDE) == ((1, -1.0), (2, 1.0)), "the camera at -X sees the front (+Y) on its left"
    assert outline.view_axes(TOP) == ((0, -1.0), (1, -1.0)), "looking down with the front at the bottom"
    assert outline.view_axes(HERO) is None, "a three-quarter camera looks across two axes"
    assert outline.view_axes({**FRONT, "ortho": 0}) is None and outline.view_axes(render.DEFAULT_SHOTS[1]) is None
    assert outline.view_axes({**FRONT, "name": "anything"}) == outline.view_axes(FRONT)


def test_a_box_measures_in_thousandths_with_no_declared_size_and_millimetres_with_one(tmp_path):
    refs = box_refs(tmp_path)
    p = outline.measure(refs, {"shots": SHOTS}, stations=5)
    assert p["mode"] == "views" and p["unit_axis"] == "Z" and p["unit_mm"] is None
    assert p["extents"] == {"X": 0.5, "Y": 0.25, "Z": 1.0}
    front, top = p["views"]["front"], p["views"]["top"]
    assert (front["along"], front["across"]) == ("Z", "X") and (top["along"], top["across"]) == ("X", "Y")
    assert [r["f"] for r in front["rows"]] == [-0.4, -0.2, 0.0, 0.2, 0.4]
    assert all(r["min"] == -0.25 and r["max"] == 0.25 for r in front["rows"])
    assert all(r["min"] == -0.125 and r["max"] == 0.125 for r in top["rows"])
    body = outline.text(p)
    assert "THOUSANDTHS" in body and "X 500, Y 250, Z 1000" in body and "z+400:-250/+250" in body
    sized = outline.measure(refs, {"shots": SHOTS, "height_m": 0.04}, stations=5)
    assert sized["unit_mm"] == 40.0 and "millimetres" in outline.text(sized) and "z+16:-10/+10" in outline.text(sized)
    # the size may be declared along any axis a view shows; a GLB extent is named in the GLB's axes
    assert outline.measure(refs, {"shots": SHOTS, "axis_extents_m": {"X": 0.02}})["unit_axis"] == "X"
    assert outline.measure(refs, {"shots": [HERO]}) == {}, "no camera looks along an axis: nothing to measure"


def test_a_mesh_is_held_to_the_table_end_by_end(tmp_path):
    refs = box_refs(tmp_path)
    free = outline.measure(refs, {"shots": SHOTS})
    same = outline.compare(free, outline.from_mesh(slab((20, 10, 40)), free))
    assert same["mae_mm"] == 0.0 and same["overall"] == {"X": 0.0, "Y": 0.0, "Z": 0.0}
    doubled = outline.compare(free, outline.from_mesh(slab((40, 20, 80)), free))
    assert doubled["mae_mm"] == 0.0, "with no declared size only the proportions are held"
    sized = outline.measure(refs, {"shots": SHOTS, "height_m": 0.04})
    assert outline.compare(sized, outline.from_mesh(slab((40, 20, 80)), sized))["mae_mm"] > 5
    wide = outline.compare(free, outline.from_mesh(slab((30, 10, 40)), free))
    assert wide["overall"] == {"X": 10.0, "Y": 0.0, "Z": 0.0}
    assert all((r["min_mm"], r["max_mm"], r["width_mm"]) == (-5.0, 5.0, 10.0) for r in wide["views"]["front"]["rows"])
    assert all(r["width_mm"] == 0.0 for r in wide["views"]["side"]["rows"])
    said = outline.compare_text(wide)
    assert "z+0:+5/+5" in said and "overall extents X +10, Y +0, Z +0" in said, "+ is further out at either end"
    off_centre = outline.compare(free, outline.from_mesh(slab((20, 10, 40), (300, -70, 9)), free))
    assert off_centre["mae_mm"] == 0.0, "positions are from the bounding box centre, as the reference's are"


def test_the_ends_are_signed_so_a_mirrored_build_does_not_pass(tmp_path):
    # an arm on the picture's LEFT of the front view: the model's +X
    refs = {"front": draw(tmp_path / "front.png", 100, 200, extra=[(-30, 80, 0, 120)])}
    p = outline.measure(refs, {"shots": SHOTS})
    low = p["views"]["front"]["rows"][0]
    assert (low["min"], low["max"]) == (-0.325, 0.175), "below the arm the body sits towards -X of the centre"
    body = slab((20, 10, 40))
    right = np.concatenate([body, slab((6, 10, 8), (13, 0, 0))])
    mirrored = np.concatenate([body, slab((6, 10, 8), (-13, 0, 0))])
    # the mask's median filter rounds the arm's drawn corners by a pixel, so not exactly zero
    assert outline.compare(p, outline.from_mesh(right, p))["mae_mm"] < 0.1
    wrong = outline.compare(p, outline.from_mesh(mirrored, p))
    assert wrong["mae_mm"] > 1.0
    row = wrong["views"]["front"]["rows"][0]
    assert row["name"] == "z-18" and (row["min_mm"], row["max_mm"]) == (6.0, 6.0), "the body sits 6 mm towards +X"
    assert "z-18:-6/+6" in outline.compare_text(wrong), "short at the -X end, too far out at the +X end"


def test_a_part_left_out_costs_its_width_and_a_part_added_costs_its_own(tmp_path):
    # two blocks with a gap between them: the top view's middle stations hold no body
    refs = {"top": draw(tmp_path / "top.png", 40, 50, extra=[(160, 0, 200, 50)])}
    p = outline.measure(refs, {"shots": SHOTS})
    assert [r["min"] is None for r in p["views"]["top"]["rows"]].count(True) >= 5
    left, right = slab((8, 10, 40), (-16, 0, 0)), slab((8, 10, 40), (16, 0, 0))
    both = outline.compare(p, outline.from_mesh(np.concatenate([left, right]), p))
    assert both["mae_mm"] == 0.0 and not both["missing_stations"] and not both["extra_stations"]
    bridged = outline.compare(p, outline.from_mesh(np.concatenate([left, right, slab((24, 10, 40))]), p))
    assert bridged["extra_stations"] and bridged["mae_mm"] > 3.0
    assert "yours has body and the reference has none" in outline.compare_text(bridged)
    gap_ref = outline.measure({"top": draw(tmp_path / "solid.png", 200, 50)}, {"shots": SHOTS})
    holed = outline.compare(gap_ref, outline.from_mesh(np.concatenate([left, right]), gap_ref))
    assert holed["missing_stations"] and holed["mae_mm"] > 3.0, "a missing station never lowers the error"


# -- through a project and `evaluate`, with the real renderer ---------------------------------------------

def figure(path, arm=1.0) -> Path:
    """A 1 m figure in glTF axes: a body, an arm on +X (`arm` -1 mirrors it) and a pack on its back (+Z)."""
    return write_cubes(path, [cube((0.4, 1.0, 0.25)), cube((0.15, 0.2, 0.2), (arm * 0.275, 0.1, 0.0)),
                              cube((0.2, 0.3, 0.1), (0.0, 0.2, 0.175))])


def make_project(tmp_path: Path, **model) -> project.Project:
    """A project whose reference views are renders of the figure through its own orthographic cameras."""
    root = tmp_path / "proj"
    (root / "refs").mkdir(parents=True)
    mesh = render.load(figure(tmp_path / "truth.glb"))
    for shot in SHOTS:
        render.render(mesh, shot).save(root / "refs" / f"{shot['name']}.png")
    (root / "jig.json").write_text(json.dumps({"name": "figure", "model": {"glb": "figure.glb", "shots": SHOTS, **model}}))
    return project.load(root)


def run(proj, tmp_path, name, arm):
    def build_fn(script, out_dir, spec, **kw):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        figure(Path(out_dir) / "figure.glb", arm)
        return ""
    script = tmp_path / "proj" / "attempts" / name / "model.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("def build():\n    pass\n")
    return evaluate.evaluate(proj, script, script.parent, log=lambda *_: None, build_fn=build_fn,
                             render_fn=lambda spec, g, o: render.render_views(spec, g, o, size=render.PREVIEW_SIZE))


def test_a_project_with_axis_cameras_gets_outlines_without_a_length_or_a_calibration(tmp_path):
    proj = make_project(tmp_path)
    ref = proj.reference_profile()
    assert ref["mode"] == "views" and sorted(ref["views"]) == ["front", "rear", "side", "top"], "hero is not read"
    from mesh_jig import brief
    assert "MEASURED REFERENCE OUTLINES, in THOUSANDTHS" in brief.text(proj)
    truth = run(proj, tmp_path, "a01_truth", 1.0)
    assert truth["ok"] and truth["profile_diff"]["mode"] == "views"
    # one preview pixel is 3.6 mm of this figure: its own geometry reads within a pixel at every end
    assert truth["numeric"]["mae_mm"] < 4.0, truth["numeric"]
    assert evaluate.ledger_line(truth).startswith(f"| a01_truth | ok | {truth['numeric']['mae_mm']:.2f} |")
    mirrored = run(proj, tmp_path, "a02_mirrored", -1.0)
    assert mirrored["numeric"]["mae_mm"] > truth["numeric"]["mae_mm"] + 10
    assert "YOUR MODEL AGAINST THE REFERENCE OUTLINES" in evaluate.feedback(mirrored)
    assert profile.compare_text(mirrored["profile_diff"]) in evaluate.feedback(mirrored)
    # the score itself is untouched: the outline error sits beside it
    assert mirrored["numeric"]["numeric_score"] == evaluate.numeric_score(mirrored["numeric"])


def test_a_declared_height_puts_the_table_in_millimetres(tmp_path):
    proj = make_project(tmp_path, height_m=1.0)
    ref = proj.reference_profile()
    assert ref["unit_axis"] == "Z" and ref["unit_mm"] == 1000.0
    assert abs(ref["extents"]["X"] - 0.55) < 0.01 and abs(ref["extents"]["Y"] - 0.35) < 0.01
    assert "in millimetres" in outline.text(ref) and "Z 1000" in outline.text(ref)
