"""mesh_jig.render: the CPU renderer every measure is taken from.

Fixtures are GLBs written in-test (a cube with one colour per face, wound outward), so no Blender and no GPU."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from glbkit import cube, textured_glb, write_cubes
from mesh_jig import render

FRONT = {"name": "front", "cam_pos": [0.0, 0.0, 3.0], "cam_target": [0.0, 0.0, 0.0], "fov": 40}
SPEC = {"script": "cube.py", "glb": "cube.glb", "stage_scale": 1.0, "shots": [dict(FRONT, name="hero")]}


def dominant(img: Image.Image, xy) -> int:
    return int(np.argmax(np.asarray(img.convert("RGB"), dtype=int)[xy[1], xy[0]]))


def test_the_face_toward_the_camera_is_drawn_and_back_faces_are_culled(tmp_path):
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube()]))
    img = render.render(mesh, FRONT)
    assert dominant(img, (320, 180)) == 0, "from +Z the red +Z face is in front; a flipped cull would show green -Z"
    side = render.render(mesh, {"cam_pos": [3.0, 0.0, 0.0], "cam_target": [0, 0, 0], "fov": 40})
    assert dominant(side, (320, 180)) == 2, "from +X the blue +X face"


def test_axes_are_a_look_at_with_plus_x_right_and_plus_y_up(tmp_path):
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube(0.2, (0.6, 0.0, 0.0)), cube(0.2, (0.0, 0.6, 0.0))]))
    ids, _, _ = render.rasterise(mesh, FRONT)
    ys, xs = np.nonzero(ids >= 0)
    right = xs[ids[ys, xs] < 12]      # the first cube's 12 triangles
    up = ys[ids[ys, xs] >= 12]
    assert right.min() > 320 and up.max() < 180


def test_vertical_fov_and_the_scale_set_the_projected_size(tmp_path):
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube(0.5)]))
    ids, _, _ = render.rasterise(mesh, FRONT)
    rows = np.nonzero((ids >= 0).any(axis=1))[0]
    # the +Z face at distance 2.75: 0.5 / 2.75 of the half-height's tan(20 deg) span
    want = 0.5 / 2.75 / np.tan(np.radians(20)) * 180
    assert abs((rows.max() - rows.min() + 1) - want) <= 2
    ids2, _, _ = render.rasterise(mesh, FRONT, scale=0.5)
    rows2 = np.nonzero((ids2 >= 0).any(axis=1))[0]
    assert (rows2.max() - rows2.min()) < 0.6 * (rows.max() - rows.min())


def test_render_is_deterministic_and_an_empty_frame_is_background(tmp_path):
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube()]))
    a, b = render.render(mesh, FRONT), render.render(mesh, FRONT)
    assert a.tobytes() == b.tobytes()
    away = render.render(mesh, {"cam_pos": [0, 0, 3.0], "cam_target": [0, 0, 6.0], "fov": 40})
    assert np.all(np.asarray(away) == np.array(render.BACKGROUND, dtype=np.uint8)), "a camera looking away draws nothing"


def test_the_default_cameras_put_the_nose_where_the_reference_views_have_it(tmp_path):
    """The nose is glTF -Z. top: nose at the bottom of the image; side: nose at the left."""
    nose = cube(0.1, (0.0, 0.1, -0.45))          # a bump on top of the body, at the nose end
    mesh = render.load(write_cubes(tmp_path / "n.glb", [cube((0.2, 0.1, 1.0)), nose]))
    shots = {s["name"]: s for s in render.DEFAULT_SHOTS}
    for view, axis, far_side in (("top", 0, "low"), ("side", 1, "left")):
        ids, _, _ = render.rasterise(mesh, shots[view])
        ys, xs = np.nonzero(ids >= 12)                      # the nose cube's triangles
        body_ys, body_xs = np.nonzero(ids >= 0)
        if far_side == "low":
            assert ys.mean() > body_ys.mean(), "top view: the nose is lower in the image than the body's middle"
        else:
            assert xs.mean() < body_xs.mean(), "side view: the nose is left of the body's middle"


def test_shots_default_to_the_judged_variant_and_show_the_others_from_the_hero_camera():
    spec = {"variants": ["gun", "missile"], "glb": "s_{variant}.glb"}
    shots = render.shots_for(spec)
    assert [s["name"] for s in shots] == ["hero", "top", "side", "rear", "hero_missile"]
    assert {s["variant"] for s in shots[:4]} == {"gun"} and shots[4]["variant"] == "missile"
    listed = render.shots_for(dict(spec, shots=[dict(FRONT, name="front")], judge_variant="missile"))
    assert [(s["name"], s["variant"]) for s in listed] == [("front", "missile")]
    assert render.stage_scale({"length_m": 0.08}) == 12.5 and render.stage_scale({"stage_scale": 2.0}) == 2.0


def test_render_views_writes_one_png_per_shot_and_reports_what_went_wrong(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    res = render.render_views(SPEC, out, tmp_path / "renders")
    assert "not built yet" in res["error"] and res["renders"] == {}
    write_cubes(out / "cube.glb", [cube()])
    res = render.render_views(SPEC, out, tmp_path / "renders")
    assert res["error"] == "" and res["cropped"] == []
    assert Image.open(res["renders"]["hero"]).size == render.RENDER_SIZE
    small = render.render_views(SPEC, out, tmp_path / "small", size=render.PREVIEW_SIZE)
    assert Image.open(small["renders"]["hero"]).size == render.PREVIEW_SIZE
    assert "no shot named ['top']" in render.render_views(SPEC, out, tmp_path / "r2", views=["top"])["error"]
    behind = dict(SPEC, shots=[{"name": "hero", "cam_pos": [0, 0, 3.0], "cam_target": [0, 0, 6.0], "fov": 40}])
    assert "NOTHING VISIBLE" in render.render_views(behind, out, tmp_path / "r3")["error"]
    close = dict(SPEC, shots=[{"name": "hero", "cam_pos": [0, 0, 0.6], "cam_target": [0, 0, 0], "fov": 40}])
    assert render.render_views(close, out, tmp_path / "r4")["cropped"] == ["hero"], "a subject cut off by the frame is named"


def test_comparisons_pair_a_render_with_the_reference_of_the_same_name(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    write_cubes(out / "cube.glb", [cube()])
    renders = render.render_views(SPEC, out, tmp_path / "renders")["renders"]
    made = render.write_comparisons({**renders, "extra": renders["hero"]}, {"hero": renders["hero"]}, tmp_path / "cmp")
    assert list(made) == ["hero"] and made["hero"].endswith("hero_vs_reference.png")
    assert Image.open(made["hero"]).height == 360 + 28


def test_a_textured_contract_draws_the_atlas_and_a_vertex_contract_ignores_it(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 0, 255)).save(buf, format="PNG")
    p = textured_glb(tmp_path / "t.glb", png=buf.getvalue(), size=(0.5, 0.5, 0.5))
    shot = {"cam_pos": [0, 0, 3.0], "cam_target": [0, 0, 0], "fov": 40}
    assert dominant(render.render(render.load(p, textured=True), shot), (320, 180)) == 2, "the blue atlas"
    assert dominant(render.render(render.load(p, textured=False), shot), (320, 180)) == 0, "the red COLOR_0"


def test_an_orthographic_shot_has_no_foreshortening_and_cam_up_orients_a_straight_down_view(tmp_path):
    """Two equal cubes, one nearer the camera: in perspective the near one is drawn bigger, orthographically the same."""
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube(0.2, (-0.4, 0.0, 0.0)), cube(0.2, (0.4, 0.0, 1.0))]))

    def widths(shot):
        ids, _, _ = render.rasterise(mesh, shot)
        ys, xs = np.nonzero(ids >= 0)
        far, near = xs[ids[ys, xs] < 12], xs[ids[ys, xs] >= 12]
        return np.ptp(far) + 1, np.ptp(near) + 1
    far, near = widths(FRONT)
    assert near > 1.3 * far
    far, near = widths({"cam_pos": [0.0, 0.0, 3.0], "cam_target": [0.0, 0.0, 0.0], "ortho": 2.0})
    assert far == near == 36, "0.2 units of a 2.0-unit view 360 px tall"
    # straight down, the image's top is wherever cam_up points: a craft's top view has the tail (+Z) up
    nose = render.load(write_cubes(tmp_path / "n.glb", [cube(0.2, (0.0, 0.0, -0.4))]))
    down = {"cam_pos": [0.0, 2.0, 0.0], "cam_target": [0.0, 0.0, 0.0], "ortho": 2.0, "cam_up": [0, 0, 1]}
    ids, _, _ = render.rasterise(nose, down)
    assert np.nonzero(ids >= 0)[0].mean() > 180, "the nose (-Z) is in the lower half"


def test_camera_light_draws_a_face_that_looks_at_the_camera_in_its_stored_colour_from_every_side(tmp_path):
    """`light: "camera"`: the lights turn with the camera, so the same paint reads the same in every view. Under the
    stage lights the face toward +X (the craft's right) is drawn at a different brightness from the one toward +Z."""
    colour = (0.8, 0.45, 0.15)
    faces = dict.fromkeys(("+x", "-x", "+y", "-y", "+z", "-z"), colour)
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube(colours=faces)]))
    stored = np.rint(render.linear_to_srgb(np.array(colour)) * 255)
    seen = {}
    for name, pos, up in (("front", [0, 0, -3], None), ("rear", [0, 0, 3], None), ("side", [-3, 0, 0], None),
                          ("right", [3, 0, 0], None), ("top", [0, 3, 0], [0, 0, 1])):
        shot = {"cam_pos": pos, "cam_target": [0, 0, 0], "ortho": 1.5, **({"cam_up": up} if up else {})}
        lit = np.asarray(render.render(mesh, shot, spec={"light": "camera"}), dtype=int)[180, 320]
        assert np.abs(lit - stored).max() <= 1, (name, lit, stored)
        seen[name] = np.asarray(render.render(mesh, shot), dtype=int)[180, 320]
    assert np.abs(seen["front"] - seen["top"]).max() > 40, "the stage lights: one paint, two readings"
    assert np.array_equal(seen["front"], np.asarray(render.render(mesh, {"cam_pos": [0, 0, -3], "cam_target": [0, 0, 0],
                          "ortho": 1.5}, spec={"light": "stage"}), dtype=int)[180, 320]), "stage is the default"


def test_camera_light_still_shades_a_face_turned_away_from_the_light(tmp_path):
    """Form has to read: a face turned from the camera is darker than the one that looks at it, and nothing is lit
    past its stored colour by more than a few percent."""
    white = dict.fromkeys(("+x", "-x", "+y", "-y", "+z", "-z"), (0.5, 0.5, 0.5))
    mesh = render.load(write_cubes(tmp_path / "c.glb", [cube(colours=white)]))
    hero = {"cam_pos": [-2.4, 1.35, -3.7], "cam_target": [0, 0, 0], "ortho": 2.0}
    img = np.asarray(render.render(mesh, hero, spec={"light": "camera"}), dtype=int)
    body = img[(np.abs(img - np.array(render.BACKGROUND)).max(axis=2) > 0)]
    levels = np.unique(body[:, 0])
    assert len(levels) >= 3, "three faces in view, three brightnesses"
    ambient, lights = render.lighting({"light": "camera"}, render.camera(hero)[1])
    most = float(ambient[0] + sum(c[0] * render.DIFFUSE for _d, c in lights))
    assert 0.44 < float(ambient[0]) < 0.46 and most < 1.25
