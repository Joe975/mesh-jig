"""mesh_jig.references: a reference view as the measures see it, in the picture's own pixels. The numbers it gives
for a calibration and for a region have to be the ones `standing` and `review` read, so both are held to them."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mesh_jig import cli, project, references, resemblance, review, standing
from test_character import calibrated, shots

GROUND = (235, 234, 230)
BODY, PATCH = (150, 55, 45), (40, 70, 160)


def picture(path, size, *rects):
    im = Image.new("RGB", size, GROUND)
    d = ImageDraw.Draw(im)
    for box, colour in rects:
        d.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=colour)        # box: right and bottom excluded
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def test_the_facts_are_the_subjects_box_ground_and_centre_in_the_pictures_own_pixels(tmp_path):
    f = references.facts(picture(tmp_path / "a.png", (300, 220), ([50, 60, 250, 160], BODY)))
    assert f == {"size": [300, 220], "box": [50, 60, 250, 160], "width_px": 200, "height_px": 100, "centre_x": 150,
                 "ground_y": 160, "pieces": 1, "touches": [], "within_px": 1}
    assert references.warnings(f) == []
    # a picture larger than the mask is read back in its own pixels, and says how far to trust them
    big = references.facts(picture(tmp_path / "big.png", (1536, 1024), ([300, 200, 1200, 900], BODY)))
    assert big["size"] == [1536, 1024] and big["within_px"] == 6
    assert np.allclose(big["box"], [300, 200, 1200, 900], atol=big["within_px"])
    empty = references.facts(picture(tmp_path / "empty.png", (100, 100)))
    assert empty["box"] is None and "no subject found" in references.warnings(empty)[0]


def test_a_cut_off_subject_and_a_stray_piece_are_warned_of(tmp_path):
    cut = references.facts(picture(tmp_path / "cut.png", (200, 200), ([60, 0, 140, 150], BODY)))
    assert cut["touches"] == ["top"] and "reaches the top edge" in references.warnings(cut)[0]
    label = references.facts(picture(tmp_path / "label.png", (300, 200), ([100, 40, 200, 180], BODY),
                                     ([10, 10, 70, 40], (20, 20, 20))))
    assert label["pieces"] == 2 and label["box"] == [10, 10, 200, 180], "the label is in the subject's box"
    assert "2 separate pieces are taken as the subject" in references.warnings(label)[0]
    speck = references.facts(picture(tmp_path / "speck.png", (300, 200), ([100, 40, 200, 180], BODY),
                                     ([10, 10, 22, 20], (20, 20, 20))))
    assert speck["pieces"] == 1 and speck["box"] == [100, 40, 200, 180], "under 2% of the largest: dropped already"


def test_the_ground_line_and_height_are_the_ones_a_standing_calibration_takes(tmp_path):
    refs, spec, config = calibrated(tmp_path)                 # hand calibration: 100 px tall, ground at y 110
    views = {v: references.facts(p) for v, p in refs.items()}
    start = references.calibration(views)
    assert start["view"] == "front" and start["height_px"] == config["calibration"]["height_px"]
    assert {v: o[1] for v, o in start["origins_px"].items()} == \
           {v: o[1] for v, o in config["calibration"]["origins_px"].items()}
    # the fixture's subject is off its centreline on purpose: the box's middle is where to start, not the answer
    assert start["origins_px"]["front"][0] == 40 and config["calibration"]["origins_px"]["front"][0] == 50
    table = standing.measure(refs, spec, {**config, "calibration": start})
    assert table["mm_per_pixel"] == 10
    assert [(r["min_mm"], r["max_mm"]) for r in table["views"]["front"]["rows"]] == [(-200, 200)] * 3
    assert references.calibration({"top": views["front"]}) is None, "no standing view: nothing to calibrate"


def test_a_region_drawn_on_the_reference_is_the_region_the_review_cuts(tmp_path):
    patch = [100, 80, 150, 120]
    ref = picture(tmp_path / "front.png", (300, 220), ([50, 60, 250, 160], BODY), (patch, PATCH))
    box = references.region_box(ref, patch)
    assert box == [0.25, 0.35, 0.5, 0.55], "200x100 subject on a square canvas: 50 px of padding above and below"
    rgb, mask = resemblance.normalised(ref)
    blue = (np.abs(rgb - np.array(PATCH)).sum(-1) < 60) & mask
    x0, y0, x1, y1 = [int(v * resemblance.GRID) for v in box]                 # as review.regions cuts it
    inside = blue[y0:y1, x0:x1]
    assert inside.mean() > 0.9 and blue.sum() - inside.sum() <= 0.1 * blue.sum(), "the patch and nothing else"
    checks = review.regions({"front": str(ref)}, {"front": str(ref)}, [{"name": "patch", "view": "front", "box": box,
                                                                       "min_iou": 0.99}], ())
    assert checks[0]["passed"] and checks[0]["reference_pixels"] == inside.size
    assert references.region_box(ref, [0, 0, 300, 220]) == [0.0, 0.0, 1.0, 1.0], "clamped to the canvas"
    with pytest.raises(ValueError, match="misses the subject"):
        references.region_box(ref, [0, 0, 40, 40])
    with pytest.raises(ValueError, match="x0 < x1"):
        references.region_box(ref, [150, 80, 100, 120])


def make(root, pictures: dict, model: dict) -> project.Project:
    (root / "refs").mkdir(parents=True)
    for name, (size, *rects) in pictures.items():
        picture(root / "refs" / f"{name}.png", size, *rects)
    (root / "jig.json").write_text(json.dumps({"name": "thing", "model": model}))
    return project.load(root)


def test_the_report_names_each_camera_the_scale_between_standing_views_and_the_asked_region(tmp_path, capsys):
    tall, short = ([60, 20, 140, 220], BODY), ([60, 40, 140, 220], BODY)
    proj = make(tmp_path / "p", {"front": ((200, 240), tall), "side": ((200, 240), short),
                                 "top": ((200, 240), ([60, 80, 140, 160], BODY)),
                                 "belly": ((200, 240), tall)}, {"shots": shots()})
    said = references.text(proj, [("front", [60.0, 20.0, 140.0, 70.0])])
    assert "- front: 200x240, subject box [60, 20, 140, 220] = 80 wide, 200 tall; centre x 100, ground y 220" in said
    assert "the camera stands on the +Y side, picture right is -X, picture up is +Z (orthographic, along one axis: " \
           "read for the outline table)" in said
    assert "the camera stands on the +Z side, picture right is -X, picture up is -Y" in said, "top: front points down"
    assert "no camera of this name: never measured" in said
    assert "- subject heights: front 200 px, side 180 px (-10.0%)" in said
    assert "! side differ from front by more than 2%" in said
    assert '"calibration": {"view": "front", "height_px": 200, "origins_px": {"front": [100, 220], "side": [100, 220]}}' in said
    assert '{"name": "<name>", "view": "front", "box": [0.3, 0.0, 0.7, 0.25]}' in said

    assert cli.main(["refs", str(proj.root), "--region", "front", "60", "20", "140", "70", "--out", str(tmp_path / "seen")]) == 0
    out = capsys.readouterr().out
    assert said in out and sorted(p.name for p in (tmp_path / "seen").iterdir()) == [
        "belly_seen.png", "front_seen.png", "side_seen.png", "top_seen.png"]
    assert Image.open(tmp_path / "seen" / "front_seen.png").size == (200, 240)
    assert cli.main(["refs", str(proj.root), "--region", "hero", "0", "0", "9", "9"]) == 2
    assert "no reference view of that name" in capsys.readouterr().err


def test_doctor_says_what_is_wrong_with_a_reference_view_and_refs_needs_some(tmp_path, capsys):
    proj = make(tmp_path / "p", {"front": ((200, 240), ([60, 0, 140, 220], BODY))}, {"shots": shots()})
    cli.main(["doctor", str(proj.root)])
    out = capsys.readouterr().out
    assert "- reference front: the subject reaches the top edge of the picture" in out
    assert "not project problems" in out and "a builder given the project as it is builds on" in out
    bare = make(tmp_path / "q", {}, {})
    assert cli.main(["refs", str(bare.root)]) == 2


def test_a_close_up_shows_the_region_of_each_picture_enlarged_and_the_feedback_names_it(tmp_path):
    patch = [100, 80, 150, 120]
    ref = picture(tmp_path / "refs" / "front.png", (300, 220), ([50, 60, 250, 160], BODY), (patch, PATCH))
    box = references.region_box(ref, patch)
    assert review.pixel_box(ref, box) == tuple(patch), "region_box the other way"
    assert review.pixel_box(ref, [0, 0, 1, 1], margin=0.5) == (0, 0, 300, 220), "held to the picture"
    # a build twice the size with its patch on the other side: the box is the same share of its subject
    built = picture(tmp_path / "renders" / "front.png", (640, 480), ([100, 120, 500, 320], BODY),
                    ([300, 160, 400, 240], PATCH))
    assert review.pixel_box(built, box) == (200, 160, 300, 240)
    regions = [{"name": "patch", "view": "front", "box": box, "min_zone": 0.9},
               {"name": "unseen", "view": "top", "box": box}]
    made = review.closeups({"front": str(built)}, {"front": str(ref)}, regions, tmp_path / "compare")
    assert list(made) == ["patch"], "a region whose view was not drawn has no close-up"
    sheet = Image.open(made["patch"])
    assert sheet.height == review.CLOSEUP_HEIGHT + 28 and sheet.width > 2 * review.CLOSEUP_HEIGHT
    half = sheet.width // 2
    middle = sheet.height // 2 + 14
    assert sheet.getpixel((half // 2, middle)) == PATCH, "the reference's region, on the left"
    assert sheet.getpixel((half + half // 2, middle)) == BODY, "the build has body where the reference has its patch"

    from mesh_jig import evaluate
    compare = str(tmp_path / "compare")
    rec = {"compare": {"front": compare + "/front_vs_reference.png"},
           "closeups": {"tail": compare + "/region_tail.png", "patch": made["patch"]},
           "gates": [{"name": "tail", "passed": True}, {"name": "patch", "passed": False}]}
    assert evaluate.picture_lines(rec, ["front"])[-1] == (
        "close-ups of the marked regions (reference left, your build right, enlarged), same directory: "
        "region_patch.png (its gate FAILED), region_tail.png")
    stored = evaluate.portable({**rec, "name": "a"}, tmp_path, tmp_path)
    assert stored["closeups"] == {"tail": "compare/region_tail.png", "patch": "compare/region_patch.png"}


def test_a_region_is_read_on_its_own_for_its_colour_and_its_edges(tmp_path):
    patch, eye = [100, 80, 150, 120], (20, 20, 20)
    dot = ([115, 92, 135, 108], eye)
    ref = picture(tmp_path / "ref.png", (300, 220), ([50, 60, 250, 160], BODY), (patch, PATCH), dot)
    region = [{"name": "patch", "view": "front", "box": references.region_box(ref, patch)}]

    def read(*rects):
        built = picture(tmp_path / f"b{len(list(tmp_path.iterdir()))}.png", (300, 220), ([50, 60, 250, 160], BODY), *rects)
        return review.regions({"front": str(built)}, {"front": str(ref)}, region, ())[0]

    same = read((patch, PATCH), dot)
    assert (same["colour"], same["edges"], same["score"]) == (1.0, 1.0, 1.0)
    plain = read((patch, PATCH))                                   # the patch without what is drawn on it
    assert plain["colour"] > 0.7 and plain["edges"] < 0.7 and plain["score"] < same["score"]
    bare = read()                                                  # no patch at all: the body's colour, no edge
    assert bare["colour"] < plain["colour"] and bare["edges"] == 0 and bare["score"] < plain["score"]
    assert review.regions_score([same, plain, bare]) == round((1 + plain["score"] + bare["score"]) / 3, 4)
    assert review.regions_score([{"name": "part:skin", "passed": True}]) is None, "only a region has a score"
    # a region the reference has and the build leaves empty has nothing in common with it, and one the reference
    # itself leaves empty is not read
    rgb, mask = resemblance.normalised(ref, review.FINE)
    nothing = (np.zeros_like(rgb), np.zeros_like(mask))
    assert review.region_reading(*nothing, rgb, mask, region[0]["box"]) == {"colour": 0.0, "edges": 0.0, "score": 0.0}
    assert review.region_reading(rgb, mask, *nothing, region[0]["box"])["score"] is None


def test_an_articulation_picked_on_a_view_is_its_path_in_blender_metres(tmp_path):
    from types import SimpleNamespace
    # a 200 px tall subject centred on x 150, standing on y 220, in a 1 m model: 5 mm a pixel
    refs = {v: picture(tmp_path / f"{v}.png", (300, 240), ([100, 20, 200, 220], BODY)) for v in ("front", "side", "top")}
    spec = {"height_m": 1.0, "shots": [{"name": "front", "cam_pos": [0, 0, -3]}, {"name": "side", "cam_pos": [-3, 0, 0]},
                                       {"name": "top", "cam_pos": [0, 3, 0]}]}
    proj = SimpleNamespace(references=refs, spec=spec, data={})
    path, said = references.articulation_path(proj, "front", [(150, 120), (180, 170)])
    # the front camera looks along +Y at the model's front: the picture's right is the model's left, X < 0
    assert path == [[0.0, 0.0, 0.5], [-0.15, 0.0, 0.25]] and "middle of the subject's box" in said
    side, _ = references.articulation_path(proj, "side", [(180, 120)])
    assert side == [[0.0, -0.15, 0.5]], "the side camera sees the front on the left: the picture's right is behind"
    moved, said = references.articulation_path(proj, "front", [(180, 120)], centre_px=170)
    assert moved == [[-0.05, 0.0, 0.5]] and "given" in said
    with pytest.raises(ValueError, match="looks up or down"):
        references.articulation_path(proj, "top", [(150, 120)])
    with pytest.raises(ValueError, match="needs model.height_m"):
        references.articulation_path(SimpleNamespace(references=refs, spec={}, data={}), "front", [(1, 1)])


def test_a_ruled_reference_draws_its_lines_at_the_blender_coordinates_a_pixel_reads_as(tmp_path):
    from types import SimpleNamespace
    # 200 px tall, centred on x 150, on the ground at y 220, in a 1 m model: 5 mm a pixel, lines every 0.1 m (20 px)
    refs = {v: picture(tmp_path / f"{v}.png", (300, 240), ([100, 20, 200, 220], BODY)) for v in ("front", "side", "top")}
    spec = {"height_m": 1.0, "shots": [{"name": "front", "cam_pos": [0, 0, -3]}, {"name": "side", "cam_pos": [-3, 0, 0]},
                                       {"name": "top", "cam_pos": [0, 3, 0]}]}
    data = {"criteria": {"outline": {"origin_m": [0, 0, -0.5]}}}
    proj = SimpleNamespace(references=refs, spec=spec, data=data)
    left, _below = references.RULE_MARGIN
    front = np.asarray(references.ruled(proj, "front"))
    assert tuple(front[5, left + 150]) == (220, 60, 60), "X = 0 is the centreline, in red"
    # the front's picture right is -X: X -0.1 is 20 px right of the centre, and is the line the next one along
    assert tuple(front[5, left + 170]) == (150, 160, 235) and tuple(front[5, left + 160]) != (150, 160, 235)
    # up is Z from the ground at -0.5: Z 0 is 100 px above the ground line, and red; Z -0.4 is 20 px above it
    assert tuple(front[120, 5 + left]) == (220, 60, 60) and tuple(front[200, 5 + left]) == (150, 160, 235)
    side = np.asarray(references.ruled(proj, "side"))
    assert tuple(side[5, left + 170]) == (150, 160, 235), "the side's picture right is -Y, ruled the same"
    assert references.ruled(proj, "top") is None, "a camera looking down is not ruled"
    assert references.ruled(SimpleNamespace(references=refs, spec={}, data={}), "front") is None, "no height, no scale"
    assert set(references.write_ruled(proj, tmp_path / "out")) == {"front", "side"}
    assert [references.rule_step(h) for h in (1.0, 1.2, 0.2, 3.0)] == [0.1, 0.1, 0.02, 0.5]


def test_a_build_is_redrawn_in_the_references_own_pixels_so_one_ruling_reads_both(tmp_path):
    from types import SimpleNamespace
    # the reference: 5 mm a pixel, centreline x 150, ground y 220 at Blender Z -0.5
    refs = {"front": picture(tmp_path / "front.png", (300, 240), ([100, 20, 200, 220], BODY))}
    spec = {"height_m": 1.0, "stage_scale": 1.0,
            "shots": [{"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.12}]}
    proj = SimpleNamespace(references=refs, spec=spec, data={"criteria": {"outline": {"origin_m": [0, 0, -0.5]}}})
    # the render: 560x280 at 250 px a unit, centred on the origin. A patch 0.1 to 0.2 m along the picture's right
    # (the front's right is -X) and Z 0 to 0.2: columns 305 to 330, rows 90 to 140
    render = Image.new("RGB", (560, 280), (220, 220, 215))
    render.paste((200, 30, 30), (305, 90, 330, 140))
    render.save(tmp_path / "render.png")
    seen = np.asarray(references.in_frame(proj, "front", tmp_path / "render.png"))
    assert seen.shape == (240, 300, 3)
    # in the reference's pixels that patch is x 170 to 190 (150 + 0.1 / 0.005 ...) and y 80 to 120 (220 - 0.6 / 0.005)
    assert tuple(seen[100, 180]) == (200, 30, 30) and tuple(seen[100, 160]) == (220, 220, 215)
    assert tuple(seen[70, 180]) == (220, 220, 215) and tuple(seen[125, 180]) == (220, 220, 215)
    pair = references.ruled_pair(proj, "front", tmp_path / "render.png")
    assert pair.width > 2 * 300 and references.write_ruled_pairs(proj, {"front": str(tmp_path / "render.png"),
                                                                       "hero": "x.png"}, tmp_path / "o") == {
        "front": str(tmp_path / "o" / "ruled_front.png")}, "a view with no reference is not drawn"


def test_refs_probe_prints_a_pixel_in_blender_metres(tmp_path, capsys):
    from mesh_jig import cli
    root = tmp_path / "p"
    picture(root / "refs" / "front.png", (300, 240), ([100, 20, 200, 220], BODY))
    (root / "jig.json").write_text(json.dumps({"name": "p", "refs": "refs", "model": {
        "height_m": 1.0, "shots": [{"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0]}]}}))
    assert cli.main(["refs", str(root), "--probe", "front", "180", "120"]) == 0
    assert "PROBE on front, pixel (180, 120): X -0.150, Z +0.500 m" in capsys.readouterr().out
    assert cli.main(["refs", str(root), "--probe", "nowhere", "1", "1"]) == 2
    assert "--probe nowhere: no reference view" in capsys.readouterr().err


def test_refs_probe_on_a_project_with_no_size_answers_in_the_briefs_thousandths(tmp_path, capsys):
    from mesh_jig import cli
    root = tmp_path / "p"
    picture(root / "refs" / "front.png", (300, 240), ([100, 20, 200, 220], BODY))   # 100 wide, 200 tall
    (root / "jig.json").write_text(json.dumps({"name": "p", "refs": "refs", "model": {
        "shots": [{"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.0}]}}))
    assert cli.main(["refs", str(root), "--probe", "front", "150", "20", "--probe", "front", "175", "120"]) == 0
    out = capsys.readouterr().out
    assert "pixel (150, 20): Z +500 in thousandths of the subject's extent along Z" in out
    assert "pixel (175, 120): X -125 in thousandths" in out, "the picture's right is the model's left, half way out"
