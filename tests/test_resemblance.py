"""mesh_jig.resemblance: the offline scalar an inner loop climbs on. Synthetic subjects on a flat ground, so the
checks are about ordering, not absolute values."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mesh_jig import palette, resemblance as R

BG = (221, 219, 214)
CRIMSON, IVORY, GUNMETAL = (150, 55, 45), (215, 210, 198), (45, 50, 56)
ZONES = palette.zones({"crimson": "#96372d", "ivory": "#d7d2c6", "gunmetal": "#2d3238"})


def subject(path: Path, *, stripe_x: int = 60, stripe: bool = True, nose: tuple = CRIMSON, seams: int = 0,
            seam_x: int = 90) -> str:
    """A 200 x 100 ivory wedge with a crimson nose block, an optional gunmetal stripe and optional panel seams."""
    im = Image.new("RGB", (240, 120), BG)
    d = ImageDraw.Draw(im)
    d.polygon([(20, 60), (60, 20), (220, 20), (220, 100), (60, 100)], fill=IVORY)
    d.rectangle([20, 40, 70, 80], fill=nose)
    if stripe:
        d.rectangle([stripe_x, 20, stripe_x + 12, 100], fill=GUNMETAL)
    for i in range(seams):
        x = seam_x + i * 18
        d.line([(x, 24), (x, 96)], fill=GUNMETAL, width=2)
    im.save(path)
    return str(path)


def test_identical_images_score_one(tmp_path):
    a = subject(tmp_path / "a.png")
    r = R.compare(a, a, ZONES)
    assert r["zone"] == 1.0 and r["structure"] == 1.0 and r["score"] == 1.0


def test_a_moved_colour_zone_lowers_zone_agreement(tmp_path):
    ref = subject(tmp_path / "ref.png")
    same = subject(tmp_path / "same.png")
    moved = subject(tmp_path / "moved.png", stripe_x=150)
    assert R.compare(same, ref, ZONES)["zone"] == 1.0
    assert R.compare(moved, ref, ZONES)["zone"] < 0.97


def test_without_a_palette_the_zone_term_is_left_out_not_guessed(tmp_path):
    ref = subject(tmp_path / "ref.png")
    moved = subject(tmp_path / "moved.png", stripe_x=150)
    r = R.compare(moved, ref)
    assert r["zone"] is None and r["score"] == r["structure"]
    lines = R.feedback_lines({"views": {"top": r}})
    assert lines == [f"top: colours {r['colour']:.2f} close, markings/edges in place {r['structure']:.2f}"]


def test_markings_in_the_wrong_place_lower_structure_more_than_no_markings(tmp_path):
    """Displaced detail is worse than none. Seams in place beat no seams; seams shifted half a period score below
    the honest ones."""
    ref = subject(tmp_path / "ref.png", seams=4)
    honest = subject(tmp_path / "honest.png", seams=4)
    none = subject(tmp_path / "none.png", seams=0)
    shifted = subject(tmp_path / "shifted.png", seams=4, seam_x=99)
    s_honest = R.compare(honest, ref, ZONES)["structure"]
    s_none = R.compare(none, ref, ZONES)["structure"]
    s_shifted = R.compare(shifted, ref, ZONES)["structure"]
    assert s_honest > s_none, (s_honest, s_none)
    assert s_shifted < s_honest, (s_shifted, s_honest)


def test_a_different_shape_costs_both_terms(tmp_path):
    ref = subject(tmp_path / "ref.png")
    im = Image.new("RGB", (240, 120), BG)
    ImageDraw.Draw(im).ellipse([40, 30, 200, 90], fill=IVORY)
    im.save(tmp_path / "blob.png")
    r = R.compare(str(tmp_path / "blob.png"), ref, ZONES)
    assert r["zone"] < 0.9 and r["structure"] < 0.9 and r["score"] < 0.9


def test_measure_views_pairs_by_name_writes_the_report_and_means_the_ortho_views(tmp_path):
    ref = {"top": subject(tmp_path / "rt.png"), "side": subject(tmp_path / "rs.png"), "hero": subject(tmp_path / "rh.png")}
    cand = {"top": subject(tmp_path / "ct.png"), "side": subject(tmp_path / "cs.png", stripe_x=150),
            "hero": subject(tmp_path / "ch.png"), "extra": subject(tmp_path / "ce.png")}
    rec = R.measure_views(cand, ref, ZONES, tmp_path / "out")
    assert set(rec["views"]) == {"top", "side", "hero"}
    assert rec["views"]["top"]["score"] == 1.0 and rec["views"]["side"]["score"] < 1.0
    assert rec["ortho"] == round((1.0 + rec["views"]["side"]["score"]) / 2, 4)
    assert (tmp_path / "out" / R.REPORT).is_file()
    assert any("colour zones agree" in line for line in R.feedback_lines(rec))


def test_empty_images_do_not_crash():
    blank = Image.new("RGB", (50, 50), BG)
    assert R.compare(blank, blank, ZONES)["score"] is None


def test_cells_cover_the_canvas():
    c = R._cells(np.zeros((R.GRID, R.GRID)), R.CELL)
    assert c.shape == (R.GRID // R.CELL, R.GRID // R.CELL, R.CELL, R.CELL)


def test_colour_closeness_falls_with_the_shade_where_the_zones_do_not(tmp_path):
    ref = subject(tmp_path / "ref.png", stripe=False)
    assert R.compare(ref, ref, ZONES)["colour"] == 1.0
    # a nose a little darker is still the palette's crimson: every zone agrees, and the colour says it is off
    darker = R.compare(subject(tmp_path / "darker.png", stripe=False, nose=(120, 44, 36)), ref, ZONES)
    assert darker["zone"] == 1.0 and 0.9 < darker["colour"] < 1.0
    # a nose in another paint: the same cells are wrong for both, and the colour falls further than for the shade
    wrong = R.compare(subject(tmp_path / "wrong.png", stripe=False, nose=GUNMETAL), ref, ZONES)
    assert wrong["zone"] < 1.0 and wrong["colour"] < darker["colour"]
    assert R.compare(ref, ref)["colour"] == 1.0, "it needs no palette"
    assert darker["score"] == round((darker["zone"] + darker["structure"]) / 2, 4), "the view's score is as it was"


def test_colour_closeness_is_one_for_the_same_colour_and_nothing_from_the_reach(tmp_path):
    nan = [np.nan] * 3
    a = np.array([[IVORY, CRIMSON, nan]], dtype=float)
    assert R.colour_closeness(a, a) == 1.0
    far = np.array([[GUNMETAL, CRIMSON, IVORY]], dtype=float)               # one cell unrelated, one the same
    d = float(np.linalg.norm(R.oklab(np.array(IVORY)) - R.oklab(np.array(GUNMETAL))))
    assert d > R.COLOUR_REACH and R.colour_closeness(a, far) == 0.5, "the third cell is in one field only"
    near = np.array([[IVORY, GUNMETAL, nan]], dtype=float)                  # crimson to gunmetal: inside the reach
    assert 0.5 < R.colour_closeness(a, near) < 0.6
    assert R.colour_closeness(a, np.full((1, 3, 3), np.nan)) is None
    white, black = R.oklab(np.array([255, 255, 255])), R.oklab(np.array([0, 0, 0]))
    assert np.allclose(white, [1, 0, 0], atol=1e-3) and np.allclose(black, [0, 0, 0], atol=1e-6)
