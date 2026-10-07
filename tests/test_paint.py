"""mesh_jig.paint: an atlas plan painted deterministically, and refused when it would have to be guessed at."""
import json

import numpy as np
import pytest
from PIL import Image

from mesh_jig import paint

PALETTE = {"crimson": "#b53c2e", "ivory": "#e7e1d4", "gunmetal": "#2c3339", "steel": "#687880", "cyan": "#3abddc"}


def test_paint_is_deterministic_and_paints_every_op(tmp_path):
    plan = {"size": 64, "background": "ivory", "ops": [
        {"op": "rect", "box": [0.0, 0.0, 0.5, 0.5], "color": "crimson"},
        {"op": "line", "points": [[0.0, 0.9], [1.0, 0.9]], "color": "gunmetal", "width": 0.05},
        {"op": "polygon", "points": [[0.6, 0.1], [0.9, 0.1], [0.75, 0.4]], "color": "steel"},
        {"op": "ellipse", "box": [0.6, 0.5, 0.9, 0.8], "color": "cyan"},
        {"op": "speckle", "box": [0.0, 0.5, 0.5, 0.85], "color": "gunmetal", "density": 0.1, "size": 0.02, "seed": 3},
        {"op": "stripes", "box": [0.55, 0.85, 1.0, 1.0], "color": "#102030", "count": 3, "width": 0.02},
        {"op": "rect", "box": [0.1, 0.1, 0.2, 0.2], "color": "ivory", "alpha": 0.5}]}
    a = paint.paint(plan, PALETTE)
    b = paint.paint(plan, PALETTE)
    assert a.size == (64, 64) and (np.asarray(a) == np.asarray(b)).all()
    px = np.asarray(a)
    assert tuple(px[2, 2]) == (0xb5, 0x3c, 0x2e)                  # crimson rect at the top-left
    assert tuple(px[58, 32]) == (0x2c, 0x33, 0x39)                # the gunmetal line at v = 0.9
    assert tuple(px[41, 48]) == (0x3a, 0xbd, 0xdc)                # cyan ellipse centre
    assert tuple(px[9, 9]) != (0xb5, 0x3c, 0x2e)                  # the half-alpha ivory over crimson blended
    assert (px[32:54, 0:32] == (0x2c, 0x33, 0x39)).all(axis=-1).any()   # some speckle landed
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert paint.paint_file(tmp_path / "plan.json", tmp_path / "out" / "a.png", PALETTE) == ""
    assert Image.open(tmp_path / "out" / "a.png").size == (64, 64)


def test_the_plans_own_palette_adds_to_the_projects_and_the_default_ground_is_white():
    plan = {"size": 16, "palette": {"moss": "#335522"}, "ops": [{"op": "rect", "box": [0, 0, 0.5, 1], "color": "moss"}]}
    px = np.asarray(paint.paint(plan, PALETTE))
    assert tuple(px[8, 2]) == (0x33, 0x55, 0x22) and tuple(px[8, 14]) == (255, 255, 255)


@pytest.mark.parametrize("bad", [
    {"ops": [{"op": "blur", "box": [0, 0, 1, 1], "color": "ivory"}]},
    {"ops": [{"op": "rect", "box": [0.5, 0, 0.2, 1], "color": "ivory"}]},
    {"ops": [{"op": "rect", "box": [0, 0, 1, 1], "color": "mauve"}]},
    {"ops": [{"op": "line", "points": [[0, 0]], "color": "ivory"}]},
    {"ops": "rect"},
    {"size": 8, "ops": []},
    {"palette": {"x": "red"}, "ops": []},
])
def test_paint_refuses_a_plan_it_would_have_to_guess_at(bad, tmp_path):
    with pytest.raises(ValueError):
        paint.paint(bad, PALETTE)
    (tmp_path / "plan.json").write_text(json.dumps(bad))
    assert paint.paint_file(tmp_path / "plan.json", tmp_path / "a.png", PALETTE).startswith("PAINT_ERROR")
    assert not (tmp_path / "a.png").exists()


def test_a_colour_only_the_project_knows_is_unknown_without_it():
    plan = {"size": 16, "ops": [{"op": "rect", "box": [0, 0, 1, 1], "color": "crimson"}]}
    assert paint.paint(plan, PALETTE).getpixel((8, 8)) == (0xb5, 0x3c, 0x2e)
    with pytest.raises(ValueError, match="unknown colour 'crimson'"):
        paint.paint(plan)
