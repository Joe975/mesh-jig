"""mesh_jig.hull: the shape carved from every orthographic outline, on outlines with known answers."""
from PIL import Image, ImageDraw

from mesh_jig import hull

GROUND, BODY = (245, 245, 245), (150, 55, 45)
SHOTS = [{"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.0}, {"name": "side", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.0},
         {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "ortho": 1.0}, {"name": "hero", "cam_pos": [-2, 1, -3], "cam_target": [0, 0, 0]}]


def picture(path, *rects, size=(300, 300)):
    im = Image.new("RGB", size, GROUND)
    d = ImageDraw.Draw(im)
    for box in rects:
        d.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=BODY)
    im.save(path)
    return path


def test_outlines_of_one_box_agree_and_the_carved_shape_reaches_them_all(tmp_path):
    # a box 0.6 wide (X), 1.0 tall (Y), 0.4 deep (Z), at 200 px a metre
    refs = {"front": picture(tmp_path / "front.png", (90, 50, 210, 250)),
            "side": picture(tmp_path / "side.png", (110, 50, 190, 250)),
            "top": picture(tmp_path / "top.png", (90, 110, 210, 190)),
            "hero": picture(tmp_path / "hero.png", (50, 50, 250, 250))}
    spec = {"shots": SHOTS}
    assert set(hull.views(spec, refs)) == {"front", "side", "top"}, "a perspective view is neither carved nor measured"
    got = hull.reach(refs, spec)
    assert set(got) == {"front", "side", "top"} and min(got.values()) > 0.97


def test_arms_the_top_view_does_not_allow_keep_a_view_short(tmp_path):
    # the front shows arms out to 0.6 wide; the top view is only the 0.3 wide trunk: no shape has both. Each view is
    # free in scale, as the overlap measure is, so the one whose proportions cannot be met is the one that falls short
    refs = {"front": picture(tmp_path / "front.png", (120, 50, 180, 250), (90, 80, 210, 120)),
            "side": picture(tmp_path / "side.png", (110, 50, 190, 250)),
            "top": picture(tmp_path / "top.png", (120, 110, 180, 190))}
    got = hull.reach(refs, {"shots": SHOTS})
    assert min(got.values()) < 0.9 and max(got.values()) > 0.95


def test_one_view_carves_nothing():
    assert hull.reach({}, {"shots": SHOTS}) == {}
