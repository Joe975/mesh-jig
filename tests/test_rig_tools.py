"""What a rig's author is told and shown: the model's facts and the script's rules (rig_brief), cameras by name and
the ruled, weighted and filmed pictures (rig_look), and the readings no gate makes (rig_read). Every fixture is a GLB
written here; no Blender, no model call."""
import json

import numpy as np
import pytest

from glbkit import box, write_glb
from mesh_jig import cli, glb, render, rig, rig_brief, rig_look, rig_read, skin


def part(name, size, centre):
    """A box given in Blender metres (+Z up, +Y front), written in glTF's axes."""
    (sx, sy, sz), (cx, cy, cz) = size, centre
    return (name, box((sx, sz, sy), (cx, cz, -cy)), (0, 0, 0))


def ribbon(name, length=1.0, steps=10, x=0.0):
    """One connected strip standing along Blender Z from 0 to `length`, `steps` quads, welded through its indices."""
    verts, faces = [], []
    for k in range(steps + 1):
        z = length * k / steps
        verts += [(x - 0.05, z, 0.0), (x + 0.05, z, 0.0)]          # glTF: Y is Blender Z
    for k in range(steps):
        a, b, c, d = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        faces += [(a, b, c), (b, d, c)]
    return (name, (np.array(verts, np.float32), np.array(faces, np.uint16)), (0, 0, 0))


FIGURE = [part("Body", (0.4, 0.2, 0.4), (0, 0, 0.2)), part("LegL", (0.1, 0.1, 0.4), (-0.1, 0, -0.2)),
          part("LegR", (0.1, 0.1, 0.4), (0.1, 0, -0.2))]
WALK = '''import math
def skeleton(model):
    return [dict(name="root", head=(0,0,-.4), tail=(0,0,-.3), parent=None),
            dict(name="body", head=(0,0,0), tail=(0,0,.4), parent="root"),
            dict(name="leg.L", head=(-.1,0,0), tail=(-.1,0,-.4), parent="body"),
            dict(name="leg.R", head=(.1,0,0), tail=(.1,0,-.4), parent="body")]
def weights(name, position, piece):
    return {"Body": "body", "LegL": "leg.L", "LegR": "leg.R"}[name]
def pose(clip, t):
    s = 30 * math.sin(2 * math.pi * t)
    return {"leg.L": [("X", s)], "leg.R": [("X", SIGN * s)], "root": {"move": (0, 0, LIFT)}}
'''
SPEC = {"fps": 8, "bones": {"required": ["root", "body", "leg.L", "leg.R"]}, "clips": {"walk": {"seconds": 1, "loop": True}}}


def build(tmp_path, parts, script, spec=SPEC, name="rig"):
    source = write_glb(tmp_path / f"{name}-source.glb", parts)
    path = tmp_path / f"{name}.py"
    path.write_text(script)
    out = tmp_path / f"{name}.glb"
    return source, out, rig.build(source, path, spec, out)


def walk(sign=-1, lift=0.0):
    return WALK.replace("SIGN", str(sign)).replace("LIFT", str(lift))


# -- rig_brief -------------------------------------------------------------------------------------------------------

def test_the_rotation_senses_the_rules_state_are_the_ones_the_runner_gives(tmp_path):
    """The brief says which way each axis turns a hanging and a standing part. A builder goes by it, so each claim is
    held to a bar that hangs from the origin and one that stands on it, turned 90 degrees."""
    script = '''
def skeleton(model):
    return [dict(name="root", head=(0,0,0), tail=(0,0,.1), parent=None), dict(name="b", head=(0,0,0), tail=(0,0,.1), parent="root")]
def weights(name, position, piece):
    return "b"
def pose(clip, t):
    return {"b": [(axis, 90) for axis in clip]}
'''
    spec = {"fps": 1, "bones": {"required": ["root", "b"]}, "clips": {a: {"seconds": 1} for a in ("X", "Y", "Z", "XZ")}}
    parts = [part("Hang", (0.02, 0.02, 0.4), (0, 0, -0.3)), part("Stand", (0.02, 0.02, 0.4), (0, 0, 0.3)),
             part("Front", (0.02, 0.4, 0.02), (0, 0.3, 0))]
    source, out, _ = build(tmp_path, parts, script, spec)
    names = np.repeat(["Hang", "Stand", "Front"], 12)

    def centre(clip, name):
        return rig_look.to_blender(skin.posed(out, clip, 0).tris[names == name].reshape(-1, 3)).mean(axis=0)

    rules = rig_brief.RULES
    assert '("X", +a): what hangs BELOW the joint swings FORWARD (toward +Y), what stands ABOVE it tips BACK' in rules
    np.testing.assert_allclose(centre("X", "Hang"), (0, 0.3, 0), atol=1e-6)
    np.testing.assert_allclose(centre("X", "Stand"), (0, -0.3, 0), atol=1e-6)
    assert '("Y", +a): what hangs below the joint swings toward -X (the model\'s own left); what stands above tips toward +X' in rules
    np.testing.assert_allclose(centre("Y", "Hang"), (-0.3, 0, 0), atol=1e-6)
    np.testing.assert_allclose(centre("Y", "Stand"), (0.3, 0, 0), atol=1e-6)
    assert '("Z", +a): turns the front (+Y) toward -X' in rules
    np.testing.assert_allclose(centre("Z", "Front"), (-0.3, 0, 0), atol=1e-6)
    assert 'the foot going back and up behind the thigh) is ("X", -a) on the shin' in rules
    # a list is applied in order about the parent's axes: forward by X, then that turned about the upright Z
    assert '[("X", 90), ("Z", 90)] swings a hanging part forward and then' in rules and "own left" in rules
    np.testing.assert_allclose(centre("XZ", "Hang"), (-0.3, 0, 0), atol=1e-6)


def test_the_bend_rule_of_thumb_is_about_what_a_blended_band_does(tmp_path):
    """RULES says a limb of radius r bent by a over a blend band h stretches its outside about 1 + r*a/h times, and
    gives 0.08 m, 45 degrees and 0.13 m as staying under 1.5. A strip on the outside of such a bend is measured."""
    r, h, a = 0.08, 0.13, np.radians(45)
    verts = np.array([(-r, -h / 2, -0.01), (-r, -h / 2, 0.01), (-r, h / 2, -0.01), (-r, h / 2, 0.01)], np.float32)   # glTF axes
    source = write_glb(tmp_path / "strip.glb", [("Strip", (verts, np.array([(0, 1, 2), (1, 3, 2)], np.uint16)), (0, 0, 0))])
    script = tmp_path / "bend.py"
    script.write_text('''
def skeleton(model):
    return [dict(name="low", head=(0,0,-1), tail=(0,0,-.9), parent=None), dict(name="high", head=(0,0,0), tail=(0,0,.1), parent="low")]
def weights(name, position, piece):
    up = (position[:, 2] > 0).astype(float)
    return {"low": 1 - up, "high": up}
def pose(clip, t):
    return {"high": [("Y", 45)]}
''')
    spec = {"fps": 1, "bones": {"required": ["low", "high"]}, "clips": {"bend": {"seconds": 1}}}
    report = rig.build(source, script, spec, tmp_path / "bent.glb")
    measured = report["clips"]["bend"]["max_edge_ratio"]
    # and the same strip on the inside of the bend closes up, which the readings say and no gate does
    inside = write_glb(tmp_path / "inside.glb", [("Strip", (verts * np.array([-1, 1, 1], np.float32), np.array([(0, 2, 1), (1, 2, 3)], np.uint16)), (0, 0, 0))])
    script.write_text(script.read_text().replace('("Y", 45)', '("Y", 100)'))
    rig.build(inside, script, spec, tmp_path / "folded.glb")
    stretch = rig_read.motion(tmp_path / "folded.glb", spec)["bend"]["stretch"]
    assert stretch["folded_share"] == 1.0 and stretch["share"] == 0.0
    assert "folded at 0.00 s: 100.00% of the surface" in rig_read.text(tmp_path / "folded.glb", spec)
    assert measured == pytest.approx(1 + r * a / h, rel=0.1) and measured < 1.5
    assert "1 + r*a/h" in rig_brief.RULES and "0.08 m in radius bent 45 degrees: 0.13 m" in rig_brief.RULES


def test_the_recipe_the_brief_hands_out_walks_runs_and_stands_a_figure_on_feet_it_places(tmp_path):
    """The brief gives a builder a whole rig script but for its numbers. It is run here as given, with a jointed
    figure's own: every part lands on its own bone, the legs take turns, an arm goes opposite its leg, and the feet
    are where `legs` was told to put them while the hips sway, roll and turn over them: on the ground and not
    sliding in a walk and an idle, off it between the steps of a run, and never under it."""
    parts = [part("Pelvis", (0.3, 0.2, 0.16), (0, 0, 0.09)), part("Chest", (0.3, 0.2, 0.16), (0, 0, 0.31)),
             part("Head", (0.25, 0.2, 0.2), (0, 0, 0.55))]
    for s in (-1, 1):
        parts += [part(f"Thigh{s}", (0.1, 0.1, 0.18), (s * 0.1, 0, -0.1)), part(f"Shin{s}", (0.1, 0.1, 0.18), (s * 0.1, 0, -0.3)),
                  part(f"Foot{s}", (0.1, 0.25, 0.05), (s * 0.1, 0.05, -0.435)),
                  part(f"Upper{s}", (0.06, 0.06, 0.13), (s * 0.3, 0, 0.275)), part(f"Fore{s}", (0.06, 0.06, 0.13), (s * 0.3, 0, 0.125))]
    script = '''import math
import numpy as np
HIP_Z, KNEE_Z, ANKLE_Z, NECK_Z, WAIST_Z, ELBOW_Z = 0.0, -0.2, -0.4, 0.41, 0.2, 0.2
LEG_X, LEG_Y, LEG_RADIUS, FOOT_FRONT, FOOT_BACK = 0.1, 0.0, 0.08, 0.175, 0.075
SHOULDER, HAND, ARM_RADIUS = (0.3, 0, 0.35), (0.3, 0, 0.05), 0.05
''' + rig_brief.RECIPE_CODE.replace("BAND = 0.06", "BAND = 0.005")
    spec = {"fps": 30, "rigid": True, "bones": {"required": ["root", "hips", "thigh.L", "shin.R", "foot.L", "forearm.R"]},
            "clips": {clip: {"seconds": seconds, "loop": True, "ground_drop_mm_max": 0.01}
                      for clip, seconds in (("walk", 1.0), ("run", 0.6), ("idle", 2.0))}}
    source, out, report = build(tmp_path, parts, script, spec, "recipe")
    assert report["issues"] == [] and rig_read.islands(out) == []       # rigid: each box wholly on one bone
    held = rig_read.holdings(out)
    assert held["thigh.L"]["hi"][0] < 0 < held["thigh.R"]["lo"][0] and held["foot.L"]["corners"] == 36
    assert held["forearm.R"]["hi"][2] == pytest.approx(0.19) and held["head"]["lo"][2] == pytest.approx(0.45)
    moves = rig_read.motion(out, spec)
    leg = 0.4
    for clip in ("walk", "run"):
        m = moves[clip]
        assert m["pairs"]["thigh"] < -0.7 and m["pairs"]["upper_arm"] < -0.9 and m["ground"]["sinks_mm"] < 0.01
        # the hips go from over one foot to over the other, and neither foot goes sideways with them
        assert m["parts"]["hips"]["across_mm"] > 8 and m["parts"]["foot.L"]["across_mm"] < 1
    walk_parts = moves["walk"]["parts"]
    assert moves["walk"]["ground"]["lowest_point_highest_mm"] == pytest.approx(0, abs=0.01)     # a foot always down
    # `sway` each way, less what the hips' roll tips their box (its middle is 0.09 m above their joint) back
    assert walk_parts["hips"]["across_mm"] == pytest.approx(2000 * (0.15 * 0.1 - 0.09 * np.sin(np.radians(2.5))), abs=1)
    # a foot is carried back on the ground by `stride`, and is furthest forward as it lands, at the turn of the loop
    assert walk_parts["foot.L"]["fore_aft_mm"] == pytest.approx(0.8 * leg * 1000, rel=0.1)
    landed = walk_parts["foot.L"]["furthest_forward_s"]
    assert min(landed, 1 - landed) < 0.08
    assert abs(walk_parts["upper_arm.L"]["furthest_forward_s"] - 0.5) < 0.08                    # with the other leg's foot
    # between the steps of a run both feet are off the ground, by about `air`
    assert 0.5 * 0.10 * leg * 1000 < moves["run"]["ground"]["lowest_point_highest_mm"] < 2 * 0.10 * leg * 1000
    assert moves["run"]["parts"]["foot.L"]["fore_aft_mm"] > 1.15 * walk_parts["foot.L"]["fore_aft_mm"]
    idle = moves["idle"]["parts"]
    assert idle["hips"]["across_mm"] == pytest.approx(2000 * (0.2 * 0.1 - 0.09 * np.sin(np.radians(2.0))), abs=1)   # the weight shifts
    assert max(idle["foot.L"][k] for k in ("across_mm", "fore_aft_mm", "up_down_mm")) < 0.5     # over feet that stay
    assert moves["idle"]["ground"]["sinks_mm"] < 0.01 and moves["idle"]["ground"]["share_of_keys_on_ground"] == 1
    # a stride the legs cannot reach lowers the hips and leaves the feet on the ground
    long = build(tmp_path, parts, script.replace("stride=0.8 * LEG", "stride=1.4 * LEG"), spec, "long")[1]
    far = rig_read.motion(long, spec)["walk"]
    assert far["ground"]["lowest_point_highest_mm"] == pytest.approx(0, abs=0.01) and far["ground"]["sinks_mm"] < 0.01
    assert far["parts"]["hips"]["up_down_mm"] > 3 * walk_parts["hips"]["up_down_mm"] > 0
    for said in ("WEIGHTS BY REGION, NEVER BY DISTANCE TO THE BONES", "PLACE THE FEET AND SOLVE THE LEGS", "def gait(p, g)",
                 "def legs(out, move, turn, feet)", "HOW BIG, in leg lengths"):
        assert said in rig_brief.RECIPE
    assert rig_brief.RECIPE in rig_brief.text(source, {**spec, "rigid": False})
    assert "A RECIPE FOR A BODY" not in rig_brief.text(source, spec)   # a rigid model has no blend to get wrong


def test_a_model_in_pieces_is_sliced_whole_and_its_nodes_named_alike_share_a_line(tmp_path):
    """One skin is read through the skin. A model built of eighty-one pieces had its largest node, a hood, sliced
    for it, and the forty largest listed, which left out the sword it was asked to keep whole."""
    pieces = [part("torso", (0.4, 0.2, 0.4), (0, 0, 0.6)), part("leg", (0.1, 0.1, 0.4), (-0.1, 0, 0.2)),
              part("leg.001", (0.1, 0.1, 0.4), (0.1, 0, 0.2))]
    text = rig_brief.facts(write_glb(tmp_path / "pieces.glb", pieces))
    assert "SLICES THROUGH THE WHOLE MODEL (it is in pieces" in text
    assert "- 2 nodes named alike (leg, leg.001): 16 vertices in all, lo (-0.15, -0.05, +0.00) hi (+0.15, +0.05, +0.40)" in text
    assert "- torso: 8 vertices, 1 piece" in text
    assert "the surface is 2 separate parts side by side" in text and "they leave the body at z = +0.40" in text
    # it stands on the origin's own height, and is not told otherwise
    assert "IT STANDS ON z = +0.000: the ground is the lowest point" in text and "NOT on z = 0" not in text


def test_facts_say_where_the_model_stands_and_where_its_parts_separate(tmp_path):
    figure = write_glb(tmp_path / "one.glb", [("Skin", _joined(FIGURE), (0, 0, 0))])
    text = rig_brief.facts(figure)
    assert "IT STANDS ON z = -0.400, NOT on z = 0" in text and "0.40 m above it" in text
    assert "SLICES THROUGH `Skin`" in text
    rows = dict(rig_brief.slices(rig_brief.triangles(*glb.read_glb(figure), "Skin")))
    legs = [runs for z, runs in rows.items() if z < -0.05]
    trunk = [runs for z, runs in rows.items() if z > 0.05]
    assert legs and all(len(r) == 2 for r in legs) and all(len(r) == 1 for r in trunk)
    left, right = legs[0]
    assert left[:2] == pytest.approx((-0.15, -0.05)) and right[:2] == pytest.approx((0.05, 0.15))
    # and the same in a sentence, so the hip's height is read and not assumed
    assert "WHAT THE SLICES SAY" in text
    assert ("from z = +0.00 down to z = -0.40 the surface is 2 separate parts side by side (x -0.15..-0.05, y -0.05..+0.05, "
            "its middle at x -0.10 y +0.00; x +0.05..+0.15, y -0.05..+0.05, its middle at x +0.10 y +0.00). "
            "They reach the ground: the model stands on them; they leave the body at z = +0.00, so the joint each "
            "one turns about is at about that height, a little above it at most") in text


def test_landmarks_name_a_pinch_and_parts_that_hang_clear():
    def row(z, *runs):
        return (z, [(a, b, -0.1, 0.1) for a, b in runs])
    rows = [row(0.55, (-0.2, 0.2)), row(0.45, (-0.25, 0.25)), row(0.35, (-0.12, 0.12)), row(0.25, (-0.3, 0.3)),
            row(0.15, (-0.5, -0.4), (-0.3, 0.3), (0.4, 0.5)), row(0.05, (-0.5, -0.4), (-0.3, 0.3), (0.4, 0.5)),
            row(-0.05, (-0.3, 0.3)), row(-0.15, (-0.25, -0.05), (0.05, 0.25))]
    said = rig_brief.landmarks(rows, ground=-0.175)
    assert len(said) == 3
    assert "the surface is 3 separate parts side by side (x -0.50..-0.40, y -0.10..+0.10, its middle at x -0.45 y +0.00; " in said[0]
    assert "They end above the ground" in said[0]
    # the row above the split is as wide as the middle part alone: the side parts begin where they hang clear
    assert "they leave the body at z = +0.17" in said[0]
    # with the outline widening above the split (arms sloping out from a shoulder), the joint is at the top of that
    sloped = rig_brief.landmarks([row(0.45, (-0.2, 0.2)), row(0.35, (-0.3, 0.3)), row(0.25, (-0.4, 0.4)),
                                  row(0.15, (-0.5, -0.4), (-0.2, 0.2), (0.4, 0.5)), row(0.05, (-0.2, 0.2))], ground=0.0)
    assert ("they hang clear of the body below z = +0.17, and are joined to it from z = +0.38 down" in sloped[0]
            and "about z = +0.38, 0.20 m out from the middle" in sloped[0] and "A part 0.25 m long" in sloped[0])
    assert "2 separate parts" in said[1] and "They reach the ground" in said[1] and "leave the body at z = -0.12" in said[1]
    assert "at z = +0.35 the surface pinches to 0.24 m across" in said[2] and "a neck or a waist" in said[2]
    assert rig_brief.landmarks([row(0.5, (-0.2, 0.2)), row(0.4, (-0.2, 0.2))], ground=0.3) == []
    # one cut where the split has come early on one side is the top of the stretch below and no stretch of its own,
    # and legs that begin at the height the hands hang to still leave the body there
    lopsided = rig_brief.landmarks([row(0.45, (-0.2, 0.2)), row(0.35, (-0.3, 0.3)), row(0.25, (-0.4, 0.2), (0.3, 0.4)),
                                    row(0.15, (-0.5, -0.4), (-0.2, 0.2), (0.4, 0.5)),
                                    row(0.05, (-0.2, -0.02), (0.02, 0.2))], ground=0.0)
    assert len(lopsided) == 2 and "3 separate parts" in lopsided[0] and "joined to it from z = +0.38 down" in lopsided[0]
    assert "2 separate parts" in lopsided[1] and "they leave the body at z = +0.08" in lopsided[1]


def _joined(parts):
    verts, faces, offset = [], [], 0
    for _, (v, f), _ in parts:
        verts.append(v)
        faces.append(f.astype(np.int64) + offset)
        offset += len(v)
    return np.concatenate(verts).astype(np.float32), np.concatenate(faces).astype(np.uint16)


def test_gates_are_said_in_words_with_their_numbers():
    spec = {"fps": 30, "loop_mm_max": 2, "bones": {"required": ["root", "hips"], "max": 40},
            "clips": {"walk": {"seconds": 1, "loop": True, "min_displacement_mm": 60, "ground_drop_mm_max": 15,
                               "stretch_area_max": 0.03}}}
    text = rig_brief.gates(spec)
    assert "required bones (exact names): root, hips" in text
    for said in ("clip `walk`: 1 s; loops", "within 2 mm of the first", "at least 60 mm", "more than 15 mm below",
                 "at most 3% of the surface"):
        assert said in text


# -- rig_look --------------------------------------------------------------------------------------------------------

def test_named_cameras_face_the_way_their_captions_say():
    lo, hi = (-0.5, -0.4, -0.6), (0.5, 0.4, 0.6)
    size = 200

    def at(view, point):
        return rig_look.pixel(rig_look.shot(view, lo, hi), size, rig_look.to_gltf(point))

    centre = size / 2
    assert at("front", (-0.3, 0, 0))[0] > centre > at("front", (0.3, 0, 0))[0]      # its own left on the picture's right
    assert at("back", (-0.3, 0, 0))[0] < centre
    assert at("left", (0, 0.3, 0))[0] < centre < at("left", (0, -0.3, 0))[0]        # it faces the picture's left
    assert at("right", (0, 0.3, 0))[0] > centre
    assert at("top", (0, 0.3, 0))[1] < centre and at("top", (0.3, 0, 0))[0] > centre
    for view in ("front", "back", "left", "right"):
        assert at(view, (0, 0, 0.3))[1] < centre                                   # up is up
    # a ruled view is to scale: 0.1 m is the same number of pixels across and up
    k = size / rig_look.shot("front", lo, hi)["ortho"]
    assert at("front", (0, 0, 0.1))[1] == pytest.approx(centre - 0.1 * k)
    assert at("front", (0.1, 0, 0))[0] == pytest.approx(centre - 0.1 * k)
    with pytest.raises(ValueError, match="no view"):
        rig_look.shot("hero", lo, hi)
    # a close-up: the point aimed at is the picture's middle, and `span` metres fill it
    close = rig_look.shot("left", lo, hi, aim=(0.1, 0.2, -0.4), span=0.5)
    assert rig_look.pixel(close, size, rig_look.to_gltf((0.1, 0.2, -0.4))) == pytest.approx((centre, centre))
    assert rig_look.pixel(close, size, rig_look.to_gltf((0.1, 0.2, -0.15)))[1] == pytest.approx(0)
    ruled = rig_look.rule(rig_look.Image.new("RGB", (size, size), (255, 255, 255)), close, lo, hi)
    column = [ruled.getpixel((int(centre) + 40, y)) for y in range(size)]             # 0.1 m off the middle: z = -0.3
    assert column[int(centre - 0.1 * size / 0.5)] != (255, 255, 255)                  # the grid line of z = -0.3


def test_weights_are_read_in_the_renderers_triangle_order_and_the_pictures_are_the_same_bytes_twice(tmp_path):
    source, out, _ = build(tmp_path, FIGURE, walk())
    doc, blob = glb.read_glb(out)
    weights, names = rig_look.corner_weights(doc, blob), rig_look.joint_names(doc)
    tris = rig_look.to_blender(skin.posed(out).tris)
    assert weights.shape == (36, 3, 4) and names == ["root", "body", "leg.L", "leg.R"]
    strongest = np.array(names)[weights.argmax(axis=2)]
    assert set(strongest[tris[..., 2] > 0.01]) == {"body"}
    assert set(strongest[(tris[..., 2] < -0.01) & (tris[..., 0] < 0)]) == {"leg.L"}
    assert set(strongest[(tris[..., 2] < -0.01) & (tris[..., 0] > 0)]) == {"leg.R"}
    lo, hi = rig_brief.load(source)[0]["lo"], rig_brief.load(source)[0]["hi"]
    sheet = rig_look.weights_sheet(out, lo, hi, size=96)
    strip = rig_look.strip(out, "walk", 1, lo, hi, phases=3, size=64, skeleton=True)
    assert strip.size == (64 * 3, 16 + 64 * 2)
    assert sheet.tobytes() == rig_look.weights_sheet(out, lo, hi, size=96).tobytes()
    assert strip.tobytes() == rig_look.strip(out, "walk", 1, lo, hi, phases=3, size=64, skeleton=True).tobytes()
    # the legs are drawn in their own bones' colours, which differ
    left, right = (tuple((rig_look.bone_colours(4)[k] * 255).astype(int)) for k in (2, 3))
    assert left != right
    rest = rig_look.ruled_rest(source, lo, hi, views=("front",), size=96)["front"]
    assert rest.size == (96, 96) and rest.tobytes() != rig_look.draw(skin.posed(out), rig_look.shot("front", lo, hi), 96).tobytes()


def test_the_quick_rasteriser_gives_the_measured_renderers_picture():
    """rig_look draws forty frames a build and cannot wait two seconds for each; it may not draw a different model
    either. A cloud of small triangles with a few large ones through it, from a ruled view and a perspective one."""
    rng = np.random.default_rng(7)
    centres = rng.uniform(-0.4, 0.4, (4000, 1, 3))
    tris = np.concatenate([centres + rng.normal(0, 0.012, (4000, 3, 3)),                 # a pixel or two each
                           rng.uniform(-0.45, 0.45, (12, 1, 3)) + rng.normal(0, 0.2, (12, 3, 3))])
    mesh = render.Mesh(tris, rng.uniform(0, 1, tris.shape), np.tile(np.array([0.0, 0.0, -1.0]), (len(tris), 3, 1)))
    lo, hi = (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    for view in ("front", "three-quarter"):
        shot = rig_look.shot(view, lo, hi)
        ids, bary, depth = rig_look.rasterise(mesh, shot, 96)
        want_ids, want_bary, want_depth = render.rasterise(mesh, shot, width=96, height=96)
        assert (want_ids >= 0).mean() > 0.05                                             # a picture, not a blank
        assert (ids == want_ids).mean() > 0.999 and ((ids >= 0) == (want_ids >= 0)).all()
        same = ids == want_ids
        np.testing.assert_allclose(depth[same & (ids >= 0)], want_depth[same & (ids >= 0)], rtol=1e-9)
        np.testing.assert_allclose(bary[same], want_bary[same], atol=1e-9)
        assert rig_look.draw(mesh, shot, 96).size == (96, 96)


def test_a_second_strip_of_one_build_renders_only_the_views_the_first_did_not(tmp_path, monkeypatch):
    source, out, _ = build(tmp_path, FIGURE, walk())
    lo, hi = rig_brief.load(source)[0]["lo"], rig_brief.load(source)[0]["hi"]
    whole = rig_look.strip(out, "walk", 1, lo, hi, ("left", "front", "three-quarter"), 3, 64)
    drawn = rig_look.cells(out, "walk", 1, lo, hi, ("left", "front"), 3, 64)
    asked, real = [], rig_look.cells
    monkeypatch.setattr(rig_look, "cells", lambda *a, **k: asked.append(a[5]) or real(*a, **k))
    again = rig_look.strip(out, "walk", 1, lo, hi, ("left", "front", "three-quarter"), 3, 64, drawn=drawn)
    assert asked == [("three-quarter",)] and again.tobytes() == whole.tobytes()
    # frames of another size or count are not the ones asked for: drawn again
    rig_look.strip(out, "walk", 1, lo, hi, ("left",), 2, 64, drawn=drawn)
    assert asked[-1] == ("left",)


# -- rig_read --------------------------------------------------------------------------------------------------------

STRAY = '''import numpy as np
def skeleton(model):
    return [dict(name="a", head=(0,0,0), tail=(0,0,.1), parent=None), dict(name="b", head=(0,0,.5), tail=(0,0,.6), parent="a")]
def weights(name, position, piece):
    z = position[:, 2]
    on_b = TEST
    return {"a": (~on_b).astype(float), "b": on_b.astype(float)}
def pose(clip, t):
    return {"b": [("X", 20 * t)]}
'''
BAR_SPEC = {"fps": 2, "bones": {"required": ["a", "b"]}, "clips": {"bend": {"seconds": 1}}}


def test_a_bones_weight_cut_off_from_its_main_patch_on_one_surface_is_a_stray_weight(tmp_path):
    clean = build(tmp_path, [ribbon("Bar")], STRAY.replace("TEST", "z > 0.45"), BAR_SPEC, "clean")[1]
    assert rig_read.islands(clean) == []
    assert "STRAY WEIGHTS: none found" in rig_read.text(clean, BAR_SPEC)
    # the same test with a second catch at the foot of the bar, as "x > 0.25 is the arm" also caught a foot
    stray = build(tmp_path, [ribbon("Bar")], STRAY.replace("TEST", "(z > 0.45) | (z < 0.15)"), BAR_SPEC, "stray")[1]
    found = rig_read.islands(stray)
    assert [f["bone"] for f in found] == ["b"]
    assert found[0]["vertices"] == 4 and found[0]["hi"][2] == pytest.approx(0.1) and found[0]["main_vertices"] == 12
    said = rig_read.text(stray, BAR_SPEC)
    assert "! b: 4 vertices round" in said and "Tighten that bone's test" in said and "its main patch is 12" in said
    # two bars, each wholly on one bone, are two surfaces: neither is a stray patch of the other
    apart = build(tmp_path, [ribbon("One", x=-0.3), ribbon("Two", x=0.3)], STRAY.replace("TEST", "z > -1"), BAR_SPEC, "apart")[1]
    assert rig_read.islands(apart) == []


def test_a_stretch_is_laid_to_the_bone_that_pulls_and_not_to_the_two_strongest(tmp_path):
    """A strip held nine tenths by two bones that never move, with a faint weight of a third at one end that does.
    The two strong bones are not what stretches it."""
    script = '''import numpy as np
def skeleton(model):
    return [dict(name="low", head=(0,0,0), tail=(0,0,.1), parent=None), dict(name="high", head=(0,0,.5), tail=(0,0,.6), parent="low"),
            dict(name="arm", head=(0,0,1), tail=(0,0,1.1), parent="low")]
def weights(name, position, piece):
    z = position[:, 2]
    arm = 0.3 * (z > 0.85)
    return {"low": (1 - arm) * (z < 0.5), "high": (1 - arm) * (z >= 0.5), "arm": arm}
def pose(clip, t):
    return {"arm": {"move": (0, 0, 0.6)}}
'''
    spec = {"fps": 1, "bones": {"required": ["low", "high", "arm"]}, "clips": {"pull": {"seconds": 1}}}
    out = build(tmp_path, [ribbon("Bar")], script, spec, "pulled")[1]
    where = rig_read.motion(out, spec)["pull"]["stretch"]["where"]
    assert [w["bones"] for w in where] == [["high", "arm"]] and where[0]["centre"][2] == pytest.approx(0.85, abs=0.05)
    assert "in the part high holds, pulled by arm's weight" in rig_read.text(out, spec)


def test_a_bone_that_holds_nothing_is_said(tmp_path):
    out = build(tmp_path, FIGURE, walk())[1]
    held = rig_read.holdings(out)
    assert held["root"] == {"corners": 0} and held["leg.L"]["corners"] == 36
    assert held["leg.L"]["hi"][0] == pytest.approx(-0.05) and held["leg.R"]["lo"][0] == pytest.approx(0.05)
    said = rig_read.text(out, SPEC)
    assert "root: nothing (the root: as it should be)" in said and "holds nothing" not in said
    # a limb's bone that nothing follows is the fault: here the left leg is given to the body
    idle = build(tmp_path, FIGURE, walk().replace('"LegL": "leg.L"', '"LegL": "body"'), name="idle")[1]
    assert "! leg.L holds nothing" in rig_read.text(idle, SPEC)


def test_something_worn_that_follows_another_bone_than_what_it_lies_on_is_said(tmp_path):
    """A belt on the body weighted to a leg leaves the body when the leg swings. No gate reads it, and a builder's
    vest slid 79 mm over its cat's chest in a build that passed every gate."""
    belted = [("Skin", _joined(FIGURE), (0, 0, 0)), part("Band", (0.42, 0.22, 0.05), (0, 0, 0.03)),
              part("Tail", (0.05, 0.4, 0.05), (0, -0.3, 0.2))]                   # joined at one end: a limb, left out

    def walk():
        return WALK.replace("SIGN", "-1").replace("LIFT", "0").replace(
            'return {"Body": "body", "LegL": "leg.L", "LegR": "leg.R"}[name]',
            'if name != "Skin":\n        return {"Tail": "leg.L"}.get(name, "body")\n'
            '    low, x = position[:, 2] < -0.001, position[:, 0]\n'
            '    return {"body": 1.0 * ~low, "leg.L": 1.0 * (low & (x < 0)), "leg.R": 1.0 * (low & (x > 0))}')

    worn = build(tmp_path, belted, walk(), name="worn")[1]
    assert rig_read.motion(worn, SPEC)["walk"]["worn"] == {"Band": {"mm": 0.0, "at_s": 0.0}}
    assert "comes away" not in rig_read.text(worn, SPEC)
    slid = build(tmp_path, belted, walk().replace('{"Tail": "leg.L"}', '{"Tail": "leg.L", "Band": "leg.L"}'), name="slid")[1]
    off = rig_read.motion(slid, SPEC)["walk"]["worn"]["Band"]
    assert off["mm"] > 30 and off["at_s"] in (0.25, 0.75)                        # at the top of the leg's swing
    assert "! comes away from the surface it lies on" in rig_read.text(slid, SPEC) and "Band by" in rig_read.text(slid, SPEC)
    # named in WORN, the band takes the weights of the surface under it whatever `weights` would have said
    kept = build(tmp_path, belted, 'WORN = {"Band": "Skin"}\n' + walk().replace('{"Tail": "leg.L"}', '{"Tail": "leg.L", "Band": "leg.L"}'),
                 name="kept")[1]
    assert rig_read.motion(kept, SPEC)["walk"]["worn"]["Band"]["mm"] == 0.0
    doc, blob = glb.read_glb(kept)
    band = next(n for n in doc["nodes"] if n.get("name") == "Band")
    w = glb.accessor(doc, blob, doc["meshes"][band["mesh"]]["primitives"][0]["attributes"]["WEIGHTS_0"])
    j = glb.accessor(doc, blob, doc["meshes"][band["mesh"]]["primitives"][0]["attributes"]["JOINTS_0"])
    assert set(j[np.arange(len(j)), w.argmax(axis=1)].astype(int)) == {1}             # all on "body", the skin's bone there
    for bad, why in (('{"Band": "Nothing"}', "does not have"), ('{"Band": "Skin", "Skin": "Tail"}', "worn itself"),
                     ('["Band"]', "must be a dictionary")):
        with pytest.raises(ValueError, match=why):
            build(tmp_path, belted, f"WORN = {bad}\n" + walk(), name="bad")
    # the tail swings with the leg too, and touches the body at one end only: a limb, not something worn
    assert set(rig_read.motion(slid, SPEC)["walk"]["worn"]) == {"Band"}


def test_nodes_of_a_model_in_pieces_that_touch_at_rest_and_part_are_said(tmp_path):
    """No edge joins two nodes, so the stretch gate reads nothing between them: a sword in four nodes that bent at
    the hip passed every gate, in three rigs of a model in pieces."""
    sword = FIGURE + [part("blade", (0.02, 0.02, 0.3), (-0.3, 0, -0.25)), part("hilt", (0.02, 0.02, 0.1), (-0.3, 0, -0.05))]
    spec = {**SPEC, "clips": {"walk": {"seconds": 1, "loop": True}, "still": {"seconds": 1, "loop": True}}}

    def script(blade):
        return WALK.replace("SIGN", "-1").replace("LIFT", "0").replace(
            '{"Body": "body", "LegL": "leg.L", "LegR": "leg.R"}', '{"Body": "body", "LegL": "leg.L", "LegR": "leg.R", '
            f'"blade": "{blade}", "hilt": "body"}}').replace("s = 30 *", "s = (clip == 'walk') * 30 *")

    bent = build(tmp_path, sword, script("leg.L"), spec, "bent")[1]
    moves = rig_read.motion(bent, spec)
    assert set(moves["walk"]["parting"]) == {"blade/hilt"} and moves["walk"]["worn"] == {}
    # where they meet is 0.1 m under the joint the blade swings 30 degrees about
    assert moves["walk"]["parting"]["blade/hilt"]["mm"] == pytest.approx(100 * np.sin(np.radians(30)), abs=3)
    assert moves["still"]["parting"]["blade/hilt"]["mm"] == 0 and moves["walk"]["parting"]["blade/hilt"]["at"][2] == -0.1
    said = rig_read.text(bent, spec)
    assert "nodes that touch at rest and part" in said and "blade and hilt by" in said and said.count("blade and hilt by") == 1
    whole = build(tmp_path, sword, script("body"), spec, "whole")[1]
    assert rig_read.motion(whole, spec)["walk"]["parting"]["blade/hilt"]["mm"] == 0
    assert "touch at rest and part" not in rig_read.text(whole, spec)
    # one skin with something on it is read as worn, as before, and not pair by pair
    belted = [("Skin", _joined(FIGURE), (0, 0, 0)), part("Band", (0.42, 0.22, 0.05), (0, 0, 0.03))]
    assert rig_read.touching(*_rest(build(tmp_path, belted, WALK.replace("SIGN", "-1").replace("LIFT", "0").replace(
        '{"Body": "body", "LegL": "leg.L", "LegR": "leg.R"}[name]', '"body"'), name="belted")[1])) == {}


def _rest(rigged):
    return rig_read.to_blender(skin.posed(rigged).tris), rig_read.triangle_nodes(glb.read_glb(rigged)[0])


def test_motion_says_whether_left_and_right_take_turns_and_where_the_ground_is(tmp_path):
    turn = rig_read.motion(build(tmp_path, FIGURE, walk(-1), name="turn")[1], SPEC)["walk"]
    together = rig_read.motion(build(tmp_path, FIGURE, walk(1), name="together")[1], SPEC)["walk"]
    assert turn["pairs"]["leg"] <= -0.99 and together["pairs"]["leg"] >= 0.99
    # a leg 0.4 m long swung 30 degrees each way from the hip: its middle goes 0.2 m fore and aft
    assert turn["parts"]["leg.L"]["fore_aft_mm"] == pytest.approx(2 * 200 * np.sin(np.radians(30)), abs=1)
    assert turn["parts"]["body"]["fore_aft_mm"] == 0
    assert turn["ground"]["share_of_keys_on_ground"] > 0 and turn["ground"]["sinks_mm"] == 0
    hung = build(tmp_path, FIGURE, walk(-1, 0.3), name="hung")[1]
    assert rig_read.motion(hung, SPEC)["walk"]["ground"]["share_of_keys_on_ground"] == 0
    sunk = build(tmp_path, FIGURE, walk(-1, -0.05), name="sunk")[1]
    assert rig_read.motion(sunk, SPEC)["walk"]["ground"]["sinks_mm"] == pytest.approx(50, abs=0.5)
    text = rig_read.text(hung, SPEC)
    assert "! never touches the ground" in text
    assert "left and right take turns (one forward as the other is back): leg" in text
    assert "folded at" not in text                                    # rigid boxes: nothing is squeezed
    assert "left and right move together (both forward at once): leg" in rig_read.text(tmp_path / "together.glb", SPEC)
    assert "! goes 50 mm below the ground" in rig_read.text(sunk, SPEC)


# -- the contract's failure, and the commands --------------------------------------------------------------------

def test_a_build_out_of_contract_says_by_how_much_and_can_be_kept_to_look_at(tmp_path):
    source, out, _ = build(tmp_path, FIGURE, walk())
    delivered = out.read_bytes()
    (tmp_path / "sunk.py").write_text(walk(-1, -0.05))
    gated = {**SPEC, "clips": {"walk": {"seconds": 1, "loop": True, "ground_drop_mm_max": 15}}}
    with pytest.raises(rig.ContractError, match=r"sinks below its rest ground \(50\.0 mm, 15 allowed\)") as failure:
        rig.build(source, tmp_path / "sunk.py", gated, out, failed=tmp_path / "kept.glb")
    assert failure.value.report["clips"]["walk"]["min_y_m"] == pytest.approx(-0.45)
    assert out.read_bytes() == delivered                                  # the delivery is still the one that passed
    assert json.loads((tmp_path / "kept.rig.json").read_text())["issues"] == failure.value.report["issues"]
    assert rig_read.motion(tmp_path / "kept.glb", gated)["walk"]["ground"]["sinks_mm"] == pytest.approx(50, abs=0.5)
    with pytest.raises(rig.ContractError):
        rig.build(source, tmp_path / "sunk.py", gated, out)
    assert not out.with_suffix(".candidate.glb").exists()


def test_rig_info_and_named_views_from_the_command_line(tmp_path, capsys):
    source, out, _ = build(tmp_path, FIGURE, walk())
    spec = tmp_path / "rig.json"
    spec.write_text(json.dumps(SPEC))
    assert cli.main(["rig-info", "--source", str(source), "--contract", str(spec), "--out", str(tmp_path / "rest")]) == 0
    said = capsys.readouterr().out
    assert "IT STANDS ON z = -0.400" in said and "THE RIG SCRIPT" in said and "clip `walk`" in said
    assert (tmp_path / "rest" / "rest_front.png").is_file() and (tmp_path / "rest" / "rest_top.png").is_file()
    assert "IT STANDS ON" in (tmp_path / "rest" / "info.txt").read_text(encoding="utf-8")
    assert cli.main(["rig-preview", "--rig", str(out), "--contract", str(spec), "--views", "left", "front", "--phases",
                     "2", "--out", str(tmp_path / "views")]) == 0
    assert (tmp_path / "views" / "walk-views.png").is_file() and (tmp_path / "views" / "weights.png").is_file()
    assert cli.main(["rig-check", "--source", str(source), "--rig", str(out), "--contract", str(spec), "--readings"]) == 0
    said = capsys.readouterr().out
    assert "\nCONTRACT: held\nWHAT EACH BONE HOLDS" in said and '"issues"' not in said
    assert json.loads(out.with_suffix(".report.json").read_text())["issues"] == []
    # two clips are set beside each other, the larger plain to see
    two = {**SPEC, "clips": {"walk": {"seconds": 1, "loop": True}, "run": {"seconds": 0.5, "loop": True}}}
    script = tmp_path / "two.py"
    script.write_text(walk().replace("s = 30 * math.sin(2 * math.pi * t)",
                                     "s = (30 if clip == 'walk' else 50) * math.sin(2 * math.pi * t / (1 if clip == 'walk' else .5))"))
    rig.build(source, script, two, tmp_path / "two.glb")
    said = rig_read.text(tmp_path / "two.glb", two)
    assert "THE CLIPS BESIDE EACH OTHER" in said and "leg.L: walk 200/27, run 306/71" in said
    assert "THE CLIPS BESIDE EACH OTHER" not in rig_read.text(out, SPEC)          # one clip has nothing beside it
    (tmp_path / "sunk.py").write_text(walk(-1, -0.05))
    spec.write_text(json.dumps({**SPEC, "clips": {"walk": {"seconds": 1, "loop": True, "ground_drop_mm_max": 15}}}))
    assert cli.main(["rig-build", "--source", str(source), "--script", str(tmp_path / "sunk.py"), "--contract", str(spec),
                     "--out", str(tmp_path / "no.glb")]) == 1
    assert '"min_y_m"' in capsys.readouterr().out and not (tmp_path / "no.glb").exists()
    # and with --failed it is kept to be read and drawn, still not delivered
    assert cli.main(["rig-build", "--source", str(source), "--script", str(tmp_path / "sunk.py"), "--contract", str(spec),
                     "--out", str(tmp_path / "no.glb"), "--failed", str(tmp_path / "kept.glb")]) == 1
    assert (tmp_path / "kept.glb").is_file() and not (tmp_path / "no.glb").exists()
    assert cli.main(["rig-check", "--source", str(source), "--rig", str(tmp_path / "kept.glb"), "--contract", str(spec),
                     "--readings"]) == 1
    assert "! goes 50 mm below the ground" in capsys.readouterr().out


def test_the_template_is_the_recipe_with_every_number_a_placeholder_and_the_extra_bones_named(tmp_path, capsys):
    spec = {"bones": {"required": ["root", "hips", "spine", "head", "thigh.L", "tail", "ear.L"]}}
    text = rig_brief.template(spec)
    compile(text, "rig.py", "exec")
    assert text.endswith(rig_brief.RECIPE_CODE)
    for name in rig_brief.NUMBERS:
        assert f'{name} = SET_ME("{name}",' in text and name in rig_brief.RECIPE_CODE
    assert "# The contract also requires ear.L, tail (set TAIL above), which the recipe does not make" in text
    assert "TAIL = None    # a tail:" in text
    with pytest.raises(ValueError, match="set HIP_Z in rig.py: the height of the hip joint"):
        exec(compile(text, "rig.py", "exec"), {})
    source = write_glb(tmp_path / "s.glb", [part("Body", (0.4, 1.0, 0.3), (0, 0.5, 0))])
    contract = tmp_path / "rig.json"
    contract.write_text(json.dumps(SPEC))
    assert cli.main(["rig-info", "--source", str(source), "--contract", str(contract),
                     "--write-template", str(tmp_path / "t" / "rig.py")]) == 0
    assert (tmp_path / "t" / "rig.py").read_text(encoding="utf-8") == rig_brief.template(SPEC, rig_brief.suggest(source))


def test_check_and_preview_of_a_rig_that_was_never_written_say_so_in_one_line(tmp_path, capsys):
    contract = tmp_path / "rig.json"
    contract.write_text(json.dumps(SPEC))
    source = write_glb(tmp_path / "s.glb", [part("Body", (0.4, 1.0, 0.3), (0, 0.5, 0))])
    assert cli.main(["rig-check", "--source", str(source), "--contract", str(contract),
                     "--rig", str(tmp_path / "failed.glb")]) == 2
    assert cli.main(["rig-preview", "--contract", str(contract), "--rig", str(tmp_path / "failed.glb"),
                     "--views", "front", "--out", str(tmp_path / "v")]) == 2
    err = capsys.readouterr().err
    assert err.count("rig-build wrote none there") == 2 and "Traceback" not in err


def test_a_stray_patch_is_told_which_cut_leaves_it_out():
    s = {"bone": "head", "lo": [0.2, -0.05, 0.55], "hi": [0.3, 0.05, 0.62],
         "main_lo": [-0.2, -0.2, 0.75], "main_hi": [0.2, 0.2, 1.2]}
    assert rig_read.separate(s) == "A cut on z leaves it out: head's test needs z > +0.69 (the two are 0.13 m apart on z)."
    inside = dict(s, lo=[-0.1, -0.1, 0.8], hi=[0.1, 0.1, 0.9])
    assert rig_read.separate(inside).startswith("It lies within the main patch's box")
    assert rig_read.separate({"bone": "head"}) == ""


def test_a_sloped_limb_is_given_its_thickness_across_itself_and_its_room_from_the_body():
    # arms 0.10 wide in a slice, sloping 30 degrees out, their middle 0.30 out; the body's side at 0.18
    s = {"joined": 0.6, "core": 0.36, "top": 0.4, "bottom": 0.2,
         "middle": [(-0.35, -0.25, -0.04, 0.04), (-0.18, 0.18, -0.1, 0.1), (0.25, 0.35, -0.04, 0.04)]}
    said = rig_brief.thickness(s)
    assert said.startswith("Each slopes about 22 degrees out from upright, so across itself it is about 0.09 m thick "
                           "where a slice is 0.10 m wide, and its middle line is 0.12 m from the body's side here. A radius")
    crowded = dict(s, middle=[(-0.27, -0.21, -0.04, 0.04), (-0.18, 0.18, -0.1, 0.1), (0.21, 0.27, -0.04, 0.04)],
                   core=0.30)
    assert "within 2 cm of the body" in rig_brief.thickness(crowded)
    assert rig_brief.thickness(dict(s, joined=None)) == ""


def test_parts_that_rise_from_the_body_are_not_read_as_limbs(tmp_path):
    source = write_glb(tmp_path / "eared.glb", [part("Body", (0.4, 0.3, 1.0), (0, 0, 0.5)),
                                                 part("EarL", (0.05, 0.05, 0.2), (-0.12, 0, 1.1)),
                                                 part("EarR", (0.05, 0.05, 0.2), (0.12, 0, 1.1))])
    said = rig_brief.facts(source)
    assert "They rise from the body to the top of the model: ears, horns or a crest" in said
    assert "room for joints" not in said


def test_a_rig_build_says_its_verdict_and_each_clip_in_a_line():
    spec = {"clips": {"run": {"seconds": 0.6, "loop": True, "stretch_area_max": 0.03}}}
    report = {"issues": ["run: too much surface area stretches past 1.5 (3.06%, 3% allowed)"],
              "clips": {"run": {"loop": True, "loop_mm": 0.0, "max_displacement_mm": 412.0,
                                "stretch_area_fraction": 0.0306}}}
    said = cli.rig_summary(report, spec, None)
    assert said[0] == "REFUSED: 1 issue(s): run: too much surface area stretches past 1.5 (3.06%, 3% allowed)"
    assert said[1] == "  run: stretch 3.06% of 3%, loop seam 0.00 mm, moves 412 mm"
    assert cli.rig_summary({"issues": [], "clips": {}}, spec, None) == ["CONTRACT: held"]


def test_a_close_up_fits_the_cameras_to_a_box_round_a_point(tmp_path):
    source, out, _ = build(tmp_path, FIGURE, walk())
    spec = tmp_path / "rig.json"
    spec.write_text(json.dumps(SPEC))
    whole, close = tmp_path / "whole", tmp_path / "close"
    for where, extra in ((whole, []), (close, ["--aim", "0.1", "0", "0.1", "--span", "0.2"])):
        assert cli.main(["rig-preview", "--rig", str(out), "--contract", str(spec), "--views", "front", "--phases", "1",
                         "--out", str(where), *extra]) == 0
    from PIL import Image
    seen = lambda p: (np.asarray(Image.open(p).convert("L")) < 200).mean()      # noqa: E731
    assert seen(close / "walk-views.png") > seen(whole / "walk-views.png"), "the close-up fills more of its frame"


def test_rig_info_lists_a_few_pieces_and_says_when_the_legs_are_short(tmp_path):
    parts = [part("Body", (0.5, 0.4, 1.0), (0, 0, 0.6)),
             part("Legs", (0.12, 0.12, 0.1), (-0.12, 0, 0.05)), part("Legs2", (0.12, 0.12, 0.1), (0.12, 0, 0.05))]
    said = rig_brief.facts(write_glb(tmp_path / "squat.glb", parts))
    assert "THE LEGS ARE SHORT: they stand apart for 0.10 m of a 1.10 m model (9%)" in said
    assert "keep BAND, the blend across a joint, under about 0.02 m" in said
    tall = [part("Body", (0.5, 0.4, 0.6), (0, 0, 0.8)),
            part("Legs", (0.12, 0.12, 0.5), (-0.12, 0, 0.25)), part("Legs2", (0.12, 0.12, 0.5), (0.12, 0, 0.25))]
    assert "THE LEGS ARE SHORT" not in rig_brief.facts(write_glb(tmp_path / "tall.glb", tall))
    (va, fa), (vb, fb) = box((0.4, 0.4, 0.4), (0, 0.5, 0)), box((0.1, 0.1, 0.1), (0.5, 0.5, 0))
    one = (np.concatenate([va, vb]), np.concatenate([fa, fb + len(va)]).astype(np.uint16))
    said = rig_brief.facts(write_glb(tmp_path / "two.glb", [("Skin", one, (0, 0, 0))]))
    assert "its pieces, largest first (`weights` is given each vertex's piece id):" in said
    assert said.count("      piece ") == 2


def test_the_recipes_split_gives_a_limbs_edge_to_its_own_bone_and_the_trunk_only():
    code = rig_brief.RECIPE_CODE
    start = code.index("def split(shares):")
    space = {"np": np}
    exec(code[start:code.index("\ndef ", start + 1)], space)
    edge = space["split"]([(np.array([0.5]), [("forearm", np.array([1.0])), ("upper_arm", 1.0)]),
                           ("hips", np.ones(1))])
    assert edge["forearm"][0] == pytest.approx(0.5) and edge["upper_arm"][0] == pytest.approx(0.0)
    assert edge["hips"][0] == pytest.approx(0.5), "the trunk keeps what the limb does not take"
    elbow = space["split"]([(np.array([1.0]), [("forearm", np.array([0.5])), ("upper_arm", 1.0)]), ("hips", np.ones(1))])
    assert elbow["forearm"][0] == pytest.approx(0.5) and elbow["upper_arm"][0] == pytest.approx(0.5)
    assert elbow["hips"][0] == pytest.approx(0.0)


def test_the_recipe_makes_weights_and_swings_a_tail_when_its_numbers_give_one(tmp_path):
    parts = [part("Pelvis", (0.3, 0.2, 0.16), (0, 0, 0.09)), part("Chest", (0.3, 0.2, 0.16), (0, 0, 0.31)),
             part("Head", (0.25, 0.2, 0.2), (0, 0, 0.55)), part("Tail", (0.05, 0.3, 0.05), (0, -0.3, 0.05))]
    for s in (-1, 1):
        parts += [part(f"Thigh{s}", (0.1, 0.1, 0.18), (s * 0.1, 0, -0.1)), part(f"Shin{s}", (0.1, 0.1, 0.18), (s * 0.1, 0, -0.3)),
                  part(f"Foot{s}", (0.1, 0.25, 0.05), (s * 0.1, 0.05, -0.435)),
                  part(f"Upper{s}", (0.06, 0.06, 0.13), (s * 0.3, 0, 0.275)), part(f"Fore{s}", (0.06, 0.06, 0.13), (s * 0.3, 0, 0.125))]
    script = '''import math
import numpy as np
HIP_Z, KNEE_Z, ANKLE_Z, NECK_Z, WAIST_Z, ELBOW_Z = 0.0, -0.2, -0.4, 0.41, 0.2, 0.2
LEG_X, LEG_Y, LEG_RADIUS, FOOT_FRONT, FOOT_BACK = 0.1, 0.0, 0.08, 0.175, 0.075
SHOULDER, HAND, ARM_RADIUS = (0.3, 0, 0.35), (0.3, 0, 0.05), 0.05
TAIL = ((0, -0.12, 0.05), (0, -0.45, 0.05), 0.05)
''' + rig_brief.RECIPE_CODE.replace("BAND = 0.06", "BAND = 0.005")
    spec = {"fps": 30, "bones": {"required": ["root", "hips", "tail"]},
            "clips": {clip: {"seconds": seconds, "loop": True} for clip, seconds in (("walk", 1.0), ("idle", 2.0))}}
    source, out, report = build(tmp_path, parts, script, spec, "tailed")
    assert report["issues"] == [] and "tail" in report["bones"]
    held = rig_read.holdings(out)["tail"]
    assert held["corners"] == 36 and held["hi"][1] < -0.1, "the tail box, and nothing of the body, is the tail's"
    moves = rig_read.motion(out, spec)
    assert moves["walk"]["parts"]["tail"]["across_mm"] > moves["idle"]["parts"]["tail"]["across_mm"] > 5


RECIPE_NUMBERS = '''import math
import numpy as np
HIP_Z, KNEE_Z, ANKLE_Z, NECK_Z, WAIST_Z, ELBOW_Z = 0.0, -0.2, -0.4, 0.41, 0.2, 0.2
LEG_X, LEG_Y, LEG_RADIUS, FOOT_FRONT, FOOT_BACK = 0.1, 0.0, 0.08, 0.175, 0.075
SHOULDER, HAND, ARM_RADIUS = (0.3, 0, 0.35), (0.3, 0, 0.05), 0.05
'''


def figure_parts(arms=True):
    parts = [part("Pelvis", (0.3, 0.2, 0.16), (0, 0, 0.09)), part("Chest", (0.3, 0.2, 0.16), (0, 0, 0.31)),
             part("Head", (0.25, 0.2, 0.2), (0, 0, 0.55))]
    for s in (-1, 1):
        parts += [part(f"Thigh{s}", (0.1, 0.1, 0.18), (s * 0.1, 0, -0.1)), part(f"Shin{s}", (0.1, 0.1, 0.18), (s * 0.1, 0, -0.3)),
                  part(f"Foot{s}", (0.1, 0.25, 0.05), (s * 0.1, 0.05, -0.435))]
        if arms:
            parts += [part(f"Upper{s}", (0.06, 0.06, 0.13), (s * 0.3, 0, 0.275)),
                      part(f"Fore{s}", (0.06, 0.06, 0.13), (s * 0.3, 0, 0.125))]
    return parts


def test_a_tail_with_a_bend_is_a_chain_of_bones_each_holding_its_stretch(tmp_path):
    # an L-shaped tail: back from the pelvis, then up
    parts = figure_parts() + [part("TailBack", (0.05, 0.25, 0.05), (0, -0.225, 0.05)),
                              part("TailUp", (0.05, 0.05, 0.25), (0, -0.35, 0.2))]
    script = RECIPE_NUMBERS + "TAIL = ((0, -0.1, 0.05), (0, -0.35, 0.05), (0, -0.35, 0.33), 0.05)\n" + \
        rig_brief.RECIPE_CODE.replace("BAND = 0.06", "BAND = 0.005")
    spec = {"fps": 30, "bones": {"required": ["root", "hips", "tail"]},
            "clips": {"walk": {"seconds": 1.0, "loop": True}}}
    _, out, report = build(tmp_path, parts, script, spec, "bent")
    assert report["issues"] == [] and {"tail", "tail.001"} <= set(report["bones"])
    held = rig_read.holdings(out)
    # each stretch is its own bone's, but for the corners at the bend, which blend as a joint does
    assert held["tail"]["corners"] >= 24 and held["tail"]["hi"][2] < 0.1, "the back stretch is the first bone's"
    assert held["tail.001"]["corners"] >= 36 and held["tail.001"]["hi"][2] > 0.3, "the upright stretch the second's"


def test_arms_built_as_pieces_of_one_node_are_weighted_by_piece(tmp_path):
    # the trunk and both arms in one node, the arms as separate pieces of it, far off the ARM line the numbers give
    trunk, arm_l, arm_r = (part("x", (0.3, 0.2, 0.6), (0, 0, 0.3))[1], part("x", (0.06, 0.06, 0.3), (-0.3, 0.1, 0.2))[1],
                           part("x", (0.06, 0.06, 0.3), (0.3, 0.1, 0.2))[1])
    verts = np.concatenate([trunk[0], arm_l[0], arm_r[0]])
    faces = np.concatenate([trunk[1], arm_l[1] + 8, arm_r[1] + 16]).astype(np.uint16)
    parts = figure_parts(arms=False) + [("Skin", (verts, faces), (0, 0, 0))]
    source = write_glb(tmp_path / "pieces-source.glb", parts)
    model, _ = rig_brief.load(source)
    skin = next(n for n in model["nodes"] if n["name"] == "Skin")
    left = next(q["id"] for q in skin["pieces"] if q["hi"][0] < -0.2)
    right = next(q["id"] for q in skin["pieces"] if q["lo"][0] > 0.2)
    script = RECIPE_NUMBERS + f'ARM_PIECES = {{".L": ("Skin", {left}), ".R": ("Skin", {right})}}\n' + \
        rig_brief.RECIPE_CODE.replace("BAND = 0.06", "BAND = 0.005")
    spec = {"fps": 30, "bones": {"required": ["root", "hips"]}, "clips": {"walk": {"seconds": 1.0, "loop": True}}}
    _, out, report = build(tmp_path, parts, script, spec, "pieces")
    assert report["issues"] == []
    held = rig_read.holdings(out)
    arm_corners = held["upper_arm.L"]["corners"] + held["forearm.L"]["corners"]
    assert arm_corners == 36 and held["upper_arm.L"]["hi"][0] < -0.2, "all of the left arm piece, none of the trunk"


def test_rig_info_points_at_pieces_to_one_side_as_possible_limbs(tmp_path):
    (va, fa), (vb, fb) = box((0.4, 1.0, 0.3), (0, 0.5, 0)), box((0.06, 0.3, 0.06), (0.4, 0.6, 0))
    one = (np.concatenate([va, vb]), np.concatenate([fa, fb + len(va)]).astype(np.uint16))
    said = rig_brief.facts(write_glb(tmp_path / "armed.glb", [("Skin", one, (0, 0, 0))]))
    assert "at x > 0) may be limbs built as their own" in said and "ARM_PIECES" in said


def test_the_recipes_numbers_are_suggested_from_the_slices(tmp_path):
    # one skin: a trunk over two legs apart, arms hanging clear, a head on a neck
    (vt, ft), (vh, fh), (vn, fn) = box((0.4, 0.5, 0.3), (0, 0.65, 0)), box((0.3, 0.25, 0.3), (0, 1.15, 0)), \
        box((0.1, 0.1, 0.1), (0, 0.95, 0))
    legs = [box((0.12, 0.4, 0.12), (s * 0.12, 0.2, 0)) for s in (-1, 1)]
    arms = [box((0.08, 0.4, 0.08), (s * 0.3, 0.6, 0)) for s in (-1, 1)]
    pieces = [(vt, ft), (vh, fh), (vn, fn), *legs, *arms]
    verts, faces, at = [], [], 0
    for v, f in pieces:
        verts.append(v)
        faces.append(f + at)
        at += len(v)
    skin = (np.concatenate(verts), np.concatenate(faces).astype(np.uint16))
    got = rig_brief.suggest(write_glb(tmp_path / "fig.glb", [("Skin", skin, (0, 0, 0))]))
    assert abs(float(got["LEG_X"]) - 0.12) < 0.03 and abs(float(got["HIP_Z"]) - 0.4) < 0.06
    shoulder, hand = eval(got["SHOULDER"]), eval(got["HAND"])
    assert hand[0] > 0.25 and hand[2] < shoulder[2], "the hand out to the side and below the shoulder"
    assert "NECK_Z" in got and 0.9 < float(got["NECK_Z"]) < 1.05
    left, right = eval(got["ARM_PIECES"])[".L"], eval(got["ARM_PIECES"])[".R"]
    assert left[0] == right[0] == "Skin" and left[1] != right[1], "the arms are pieces of their own, one a side"


def test_a_chain_gives_each_stretch_of_a_bent_line_to_its_own_bone():
    code = rig_brief.RECIPE_CODE
    space = {"np": np, "BAND": 0.02}
    for fn in ("def ramp(", "def chain("):
        start = code.index(fn)
        exec(code[start:code.index("\ndef ", start + 1)], space)
    points = [(0, 0, 0), (0, -0.3, 0), (0, -0.3, 0.3)]                 # back, then up: an L
    position = np.array([[0, -0.1, 0.02], [0, -0.32, 0.2], [0, -0.33, 0.01]])
    parts = dict(space["chain"](position, points, ["tail", "tail.001"]))
    assert parts["tail.001"][0] == pytest.approx(0.0) and parts["tail.001"][1] == pytest.approx(1.0)
    assert 0.0 < parts["tail.001"][2] < 1.0, "a point at the bend blends the two"


def test_a_tail_nodes_line_is_read_from_the_body_out_to_its_tip():
    # an L tail behind a body centred on the origin: back along -Y, then up
    back = box((0.04, 0.3, 0.04), (0, -0.3, 0.1))               # Blender metres, as rig_brief reads them
    up = box((0.04, 0.04, 0.3), (0, -0.45, 0.25))
    tris = np.concatenate([back[0][back[1]], up[0][up[1]]])
    body = box((0.4, 0.3, 0.6), (0, 0, 0.3))
    said = rig_brief.tail_line(tris, body[0][body[1]])
    line = eval(said)
    points, radius = line[:-1], line[-1]
    assert points[0][1] > -0.2 and points[-1][2] > 0.35, "from the body end out to the upright tip"
    assert 0.02 <= radius < 0.1


def test_the_template_scales_only_the_up_and_down_and_sets_band_for_short_legs(tmp_path):
    parts = [part("Body", (0.5, 0.4, 1.0), (0, 0, 0.6)),
             part("Legs", (0.12, 0.12, 0.1), (-0.12, 0, 0.05)), part("Legs2", (0.12, 0.12, 0.1), (0.12, 0, 0.05))]
    source = write_glb(tmp_path / "squat.glb", parts)
    got = rig_brief.suggest(source)
    text = rig_brief.template(SPEC, got)
    assert "GAIT_SCALE = " in text and text.rstrip().splitlines()[-1].startswith(f"BAND = {got['BAND']}")
    assert 'for k in ("lift", "bounce", "air"):' in rig_brief.RECIPE_CODE, "the stride is not scaled"
