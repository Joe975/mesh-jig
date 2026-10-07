"""mesh_jig.profile and mesh_jig.palette: the reference outlines and colour zones as numbers, and a GLB's own
outline against them."""
import numpy as np
import pytest
from PIL import Image

from glbkit import box, write_glb
from mesh_jig import palette, profile

PAPER = (235, 234, 230)
ZONES = palette.zones({"crimson": "#96372d", "ivory": "#d7d2c6", "gunmetal": "#2d3238"})


def draw(path, w, h):
    im = Image.new("RGB", (w + 40, h + 40), PAPER)
    im.paste((40, 44, 50), (20, 20, 20 + w, 20 + h))
    im.save(path)
    return str(path)


def box_refs(tmp_path) -> dict:
    """A 40 x 20 x 10 mm box drawn in the three ortho views."""
    return {"top": draw(tmp_path / "top.png", 100, 200),      # span 20 mm across, length 40 mm down
            "side": draw(tmp_path / "side.png", 200, 50),     # length 40 mm across, height 10 mm
            "rear": draw(tmp_path / "rear.png", 100, 50)}


def test_profile_measures_a_synthetic_box(tmp_path):
    p = profile.measure(box_refs(tmp_path), 40.0, stations=5)
    assert p["length_mm"] == 40.0
    assert abs(p["span_mm"] - 20.0) < 1.0
    assert abs(p["height_mm"] - 10.0) < 1.0
    assert all(abs(s["half_width"] - 10.0) < 1.0 for s in p["top"])
    assert all(abs(s["z_top"] - 5.0) < 1.0 and abs(s["z_bottom"] + 5.0) < 1.0 for s in p["side"] if s["z_top"] is not None)
    body = profile.text(p)
    assert "span 20" in body.replace("20.0", "20") and "y+20:10" in body


def test_outlines_need_all_three_ortho_views(tmp_path):
    refs = box_refs(tmp_path)
    assert profile.measurable(refs)
    assert not profile.measurable({k: v for k, v in refs.items() if k != "rear"})


def test_profile_text_marks_empty_stations():
    p = {"length_mm": 10.0, "span_mm": 4.0, "height_mm": 2.0, "top": [{"y": 5.0, "half_width": 2.0}],
         "side": [{"y": 5.0, "z_top": None, "z_bottom": None}], "rear": [{"x": 0.0, "z_top": 1.0, "z_bottom": -1.0}]}
    body = profile.text(p)
    assert "y+5:-/-" in body and "x0:+1/-1" in body


def slab(size, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A box in model millimetres as triangles, (12, 3, 3)."""
    verts, faces = box(size, centre)
    return verts.astype(float)[faces.astype(int)]


def test_a_mesh_profile_and_compare_measure_a_box():
    """A 20 x 40 x 10 mm box measures back at every station, and comparing it with itself is all zeros."""
    m = profile.profile_from_mesh(slab((20, 40, 10)), 40.0, stations=5)
    assert m["span_mm"] == 20.0 and m["height_mm"] == 10.0 and m["length_mm"] == 40.0
    assert [s["half_width"] for s in m["top"]] == [10.0] * 5, "the middle stations too: no vertex sits there"
    assert all(s["z_top"] == 5.0 and s["z_bottom"] == -5.0 for s in m["side"] + m["rear"])
    same = profile.compare(m, m)
    assert same["mae_mm"] == 0.0 and same["overall"] == {"length_mm": 0.0, "span_mm": 0.0, "height_mm": 0.0}
    taller = profile.profile_from_mesh(slab((20, 40, 14)), 40.0, stations=5)
    diff = profile.compare(m, taller)                      # 2 mm taller everywhere reads as +2 on z_top
    assert diff["side"][0]["z_top"] == 2.0 and diff["mae_mm"] > 0
    assert "+2" in profile.compare_text(diff) and "Mean absolute error" in profile.compare_text(diff)


def test_the_outline_is_the_surfaces_not_the_vertices():
    """One box, and the same box built as four end to end: the shape is unchanged, so every number is. A measure
    read off the vertices near each station would differ (it once fell 1 mm when edge loops were added)."""
    one = slab((20, 40, 10))
    four = np.concatenate([slab((20, 10, 10), (0, y, 0)) for y in (-15, -5, 5, 15)])
    assert profile.profile_from_mesh(one, 40.0) == profile.profile_from_mesh(four, 40.0)
    # and a part that ends between two stations is seen only by the stations it reaches
    with_fin = np.concatenate([one, slab((2, 8, 6), (0, -16, 8))])          # a fin on top, y -20..-12
    side = {s["y"]: s["z_top"] for s in profile.profile_from_mesh(with_fin, 40.0, stations=5)["side"]}
    assert side == {20.0: 2.0, 10.0: 2.0, 0.0: 2.0, -10.0: 2.0, -20.0: 8.0}, "z from the box centre: -5 .. 11 is 16 tall"
    wedge = np.array([[[-10, -20, 0], [10, -20, 0], [0, 20, 0]]], dtype=float)   # a flat triangle, nose at +y
    widths = [s["half_width"] for s in profile.profile_from_mesh(wedge, 40.0, stations=5)["top"]]
    assert widths == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0], abs=0.11), "a taper read at the station, not its widest nearby"
    lifted = profile.profile_from_mesh(one + np.array([0.0, 0.0, 5.0]), 40.0)
    assert lifted == profile.profile_from_mesh(one, 40.0), "measured about the bounding box, as the reference is"
    with pytest.raises(ValueError, match="no triangles"):
        profile.profile_from_mesh(np.zeros((0, 3, 3)), 40.0)


def test_a_glb_is_read_in_model_axes_and_millimetres(tmp_path):
    """glTF (x, y up, -z nose) becomes model (x, +y nose, +z up): a GLB 80 mm along Z measures 80 mm long along Y."""
    p = write_glb(tmp_path / "s.glb", [("S", box((0.040, 0.010, 0.080)), (0, 0, 0))])
    tris = profile.glb_triangles(p)
    assert tris.shape == (12, 3, 3)
    assert np.ptp(tris.reshape(-1, 3), axis=0) == pytest.approx([40.0, 80.0, 10.0], abs=1e-3)
    ref = profile.profile_from_mesh(tris, 80.0, stations=5)
    assert profile.glb_diff(ref, p)["mae_mm"] == 0.0
    assert profile.glb_diff({}, p) == {}, "no reference outline, nothing to compare"
    assert profile.glb_diff(ref, tmp_path / "missing.glb") == {}


def test_zones_give_each_palette_colour_a_letter_of_its_own():
    zs = palette.zones({"crimson": "#b53c2e", "cyan": "#3abddc", "cream": "#eeeeee", "42": "#000000"})
    assert [z[0] for z in zs] == ["C", "Y", "R", "A"], "first free letter of the name, else of the alphabet"
    assert zs[0][2] == (0xb5, 0x3c, 0x2e)
    assert palette.zones({}) == () and palette.zones(None) == ()
    with pytest.raises(ValueError, match="#rrggbb"):
        palette.zones({"bad": "red"})


def test_zone_map_reads_the_palette_off_a_view(tmp_path):
    im = Image.new("RGB", (240, 120), PAPER)
    im.paste((150, 55, 45), (20, 20, 120, 100))     # crimson left half of the body
    im.paste((215, 210, 198), (120, 20, 220, 100))  # ivory right half
    im.save(tmp_path / "top.png")
    rows = profile.zone_map(tmp_path / "top.png", ZONES, cols=20)
    assert rows and all(len(r) == 20 for r in rows)
    body = "".join(rows)
    assert body.count("C") > 20 and body.count("I") > 20
    assert rows[len(rows) // 2].startswith("C") and rows[len(rows) // 2].endswith("I")
    assert "C crimson" in profile.zone_text({"top": rows}, ZONES)
    assert "top (nose at the bottom):" in profile.zone_text({"top": rows}, ZONES)
    upright = profile.zone_text({"top": rows}, ZONES, craft=False)
    assert "  top:" in upright and "nose" not in upright, "a subject with no nose is not told where its nose is"
    assert profile.zone_map(tmp_path / "top.png", ()) == [], "no palette, no zone map"


def test_a_palette_is_read_off_the_reference_views(tmp_path):
    im = Image.new("RGB", (240, 120), PAPER)
    im.paste((150, 55, 45), (20, 20, 180, 100))     # mostly crimson
    im.paste((40, 60, 200), (180, 20, 220, 100))    # a blue tail
    im.save(tmp_path / "top.png")
    found = palette.suggest({"top": str(tmp_path / "top.png")}, colors=2)
    assert list(found) == ["c1", "c2"], "largest share first"
    assert palette.hex_rgb(found["c1"]) == (150, 55, 45) and palette.hex_rgb(found["c2"]) == (40, 60, 200)
    assert palette.suggest({"top": str(tmp_path / "top.png")}, colors=2) == found, "deterministic"
    blank = tmp_path / "blank.png"
    Image.new("RGB", (50, 50), PAPER).save(blank)
    assert palette.suggest({"top": str(blank)}) == {}
