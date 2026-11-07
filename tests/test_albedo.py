"""`colour_from: "albedo"`: the colour measures read an unlit pass of the build, each pixel the colour its surface
stores, while edges and the pictures keep the lit render. Fixtures are one-colour cubes written in-test: a paint that
is right everywhere reads as right only when the light is taken out of the reading."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from glbkit import cube, write_cubes
from mesh_jig import brief, criteria, evaluate, palette, project, render, resemblance, review
from test_criteria import character, measured

ORANGE = (0.6, 0.18, 0.03)          # linear: sRGB about (203, 118, 49)
HERO = {"name": "hero", "cam_pos": [-2.4, 1.35, -3.7], "cam_target": [0.0, 0.0, 0.0], "ortho": 1.5}


def spec(**more) -> dict:
    return {"glb": "cube.glb", "stage_scale": 1.0, "shots": [HERO], **more}


def drawn(tmp_path, **more) -> dict:
    """The one-colour cube drawn from three-quarters, where three faces at three angles to the light show."""
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    write_cubes(out / "cube.glb", [cube(colours=dict.fromkeys(("+x", "-x", "+y", "-y", "+z", "-z"), ORANGE))])
    return render.render_views(spec(**more), out, tmp_path / "renders", size=(320, 180))


def body(path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=int)
    return img[np.abs(img - np.array(render.BACKGROUND)).max(axis=2) > 0]


def stored() -> np.ndarray:
    return np.rint(render.linear_to_srgb(np.array(ORANGE)) * 255).astype(int)


def zones_for(lit_path) -> tuple:
    """The stored orange, and a darker colour that is exactly the lit render's darkest face: the reading the light
    makes of the paint, so a lit reading has somewhere wrong to put it."""
    dark = min(map(tuple, np.unique(body(lit_path), axis=0)), key=sum)
    return palette.zones({"orange": "#%02x%02x%02x" % tuple(stored()), "shade": "#%02x%02x%02x" % dark})


@pytest.mark.parametrize("light", ["stage", "camera"])
def test_the_unlit_pass_is_the_stored_colour_on_every_face_under_either_light(tmp_path, light):
    shot = drawn(tmp_path, light=light, colour_from="albedo")
    lit, flat = body(shot["renders"]["hero"]), body(shot["albedo"]["hero"])
    assert len(np.unique(lit, axis=0)) >= 2, "the lit pass shades the faces apart"
    assert np.unique(flat, axis=0).tolist() == [stored().tolist()], "the unlit pass is one colour, the stored one"
    assert len(flat) == len(lit), "the same pixels: one rasterisation, two shadings"


def test_without_colour_from_there_is_no_unlit_pass(tmp_path):
    shot = drawn(tmp_path)
    assert "albedo" not in shot and not (tmp_path / "renders" / render.ALBEDO_DIR).exists()


def test_the_colour_measures_read_the_paint_and_the_edges_still_read_the_light(tmp_path):
    """The reference is the cube's own paint. Read from the lit render the stage light turns part of it to the
    darker colour it shows there; read from the unlit pass every cell is the orange it is painted."""
    shot = drawn(tmp_path, colour_from="albedo")
    lit, flat = shot["renders"]["hero"], shot["albedo"]["hero"]
    zones = zones_for(lit)
    before = resemblance.compare(lit, flat, zones)
    after = resemblance.compare(lit, flat, zones, colour=flat)
    assert after["zone"] == 1.0 and after["colour"] == 1.0
    assert before["zone"] < 1.0 and before["colour"] < 1.0, "the light is read as paint"
    assert after["structure"] == before["structure"], "edges are read from the lit render either way"
    both = resemblance.measure_views({"hero": lit}, {"hero": flat}, zones, colours={"hero": flat})
    assert both["views"]["hero"] == after


def test_a_region_reads_its_colour_and_zones_from_the_unlit_pass(tmp_path):
    shot = drawn(tmp_path, colour_from="albedo")
    lit, flat = shot["renders"]["hero"], shot["albedo"]["hero"]
    zones = zones_for(lit)
    marked = [{"name": "all", "view": "hero", "box": [0, 0, 1, 1], "min_zone": 0.9}]
    lit_read = review.regions({"hero": lit}, {"hero": flat}, marked, zones)[0]
    paint_read = review.regions({"hero": lit}, {"hero": flat}, marked, zones, {"hero": flat})[0]
    assert paint_read["zone"] == 1.0 and paint_read["colour"] == 1.0 and paint_read["passed"]
    assert lit_read["zone"] < 0.9 and not lit_read["passed"]
    assert paint_read["edges"] == lit_read["edges"]
    reading = evaluate.zone_reading(lit, flat, zones, flat)
    assert reading["yours"] == {"orange": 1.0}


def test_colour_from_is_a_criterion_folded_named_and_checked(tmp_path):
    data = criteria.rewrite(character(tmp_path))
    plain = project.Project(tmp_path, data)
    assert plain.spec.get("colour_from") is None and "- colour from:" not in brief.criteria_facts(plain)
    assert "unlit drawing" not in brief.reading_facts(plain)
    unlit = project.Project(tmp_path, dict(data, criteria=dict(data["criteria"], colour_from="albedo")))
    assert unlit.problems() == [] and unlit.spec["colour_from"] == "albedo"
    assert "- colour from: albedo [criteria.colour_from]" in brief.criteria_facts(unlit)
    assert "unlit drawing of your build" in brief.reading_facts(unlit)
    assert criteria.block(unlit.data)["colour_from"] == "albedo"
    again = criteria.rewrite(unlit.stated)
    assert again["criteria"]["colour_from"] == "albedo" and "colour_from" not in again["model"]
    wrong = project.Project(tmp_path, dict(data, criteria=dict(data["criteria"], colour_from="paint")))
    assert any("criteria.colour_from must be one of render, albedo" in p for p in wrong.problems())


def test_an_evaluation_under_colour_from_records_the_unlit_pass_and_leaves_overlap_alone(tmp_path):
    data = criteria.rewrite(character(tmp_path))
    plain = measured(tmp_path, data, "plain")
    unlit = measured(tmp_path, dict(data, criteria=dict(data["criteria"], colour_from="albedo")), "unlit")
    assert unlit["numeric"]["ortho_iou"] == plain["numeric"]["ortho_iou"]
    rec = evaluate.read(tmp_path / "attempts" / "unlit")
    assert set(rec["albedo"]) >= {"front", "side", "rear", "right", "top"}
    assert all((tmp_path / "attempts" / "unlit" / "renders" / render.ALBEDO_DIR / f"{v}.png").is_file()
               for v in rec["albedo"])
    assert "albedo" not in evaluate.read(tmp_path / "attempts" / "plain")
