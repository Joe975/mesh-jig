"""Silhouette overlap per view: masks on a flat render ground and on a painted sheet, holes, labels, normalisation,
the numbers and the overlays. Synthetic images only."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from mesh_jig import silhouette as sil

STAGE = (221, 219, 214)
IVORY_LIT = (214, 212, 205)     # ~10 from the render ground: lost at the sheet threshold


def stage(shapes, size=(320, 180)):
    im = Image.new("RGB", size, STAGE)
    d = ImageDraw.Draw(im)
    for box, colour in shapes:
        d.rectangle(box, fill=colour)
    return im


def sheet(shapes, size=(320, 180), seed=1):
    rng = np.random.default_rng(seed)
    a = np.clip(np.array((214, 211, 206)) + rng.normal(0, 3, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for box, colour in shapes:
        d.rectangle(box, fill=colour)
    return im


def test_render_mask_keeps_a_lit_pale_face_that_a_sheet_threshold_would_drop():
    im = stage([((40, 60, 280, 120), IVORY_LIT)])
    m = sil.mask(im)
    assert sil.bbox(m) == (40, 60, 281, 121)
    assert not sil.mask(im, distance=sil.BG_DISTANCE).any(), "the sheet threshold loses it: why a render gets its own"


def test_sheet_mask_ignores_paper_grain_fills_holes_and_drops_a_small_label():
    im = sheet([((40, 40, 200, 140), (120, 40, 30)), ((300, 160, 304, 164), (40, 40, 40))])
    ImageDraw.Draw(im).rectangle((80, 70, 160, 110), fill=(214, 211, 206))   # a pale panel inside a dark outline
    m = sil.mask(im)
    assert sil.bbox(m) == (40, 40, 201, 141)
    assert m[90, 120], "a hole inside the outline is body"


def test_same_shape_at_another_scale_and_place_overlaps_fully():
    a = stage([((20, 20, 180, 60), (120, 40, 30))])
    b = sheet([((100, 50, 260, 90), (120, 40, 30))], size=(400, 200))
    d = sil.compare(a, b)
    assert d["iou"] > 0.95 and d["missing"] < 0.03 and d["extra"] < 0.03
    assert abs(d["aspect"] - d["reference_aspect"]) < 0.02


def test_a_narrow_candidate_against_a_wide_reference_loses_overlap_and_says_so():
    ref = sheet([((20, 20, 300, 100), (120, 40, 30))])            # wide: height / width 0.29
    cand = stage([((100, 20, 220, 100), (120, 40, 30))])          # narrow: 0.67
    d = sil.compare(cand, ref)
    assert d["iou"] < 0.6 and d["aspect"] > d["reference_aspect"]
    lines = sil.feedback_lines({"views": {"top": d}})
    assert lines[0].startswith("top: silhouette overlap") and "too tall or too narrow" in lines[0]


def test_measure_views_pairs_by_name_writes_overlays_and_ranks_on_the_orthographic_views(tmp_path):
    body = [((20, 20, 180, 60), (120, 40, 30))]
    refs, renders = {}, {}
    for v in ("hero", "top", "side"):
        sheet(body).save(tmp_path / f"ref_{v}.png")
        refs[v] = tmp_path / f"ref_{v}.png"
    for v in ("hero", "top", "side", "hero_missile"):
        stage(body).save(tmp_path / f"{v}.png")
        renders[v] = tmp_path / f"{v}.png"
    out = tmp_path / "hist"
    rec = sil.measure_views(renders, refs, out)
    assert sorted(rec["views"]) == ["hero", "side", "top"], "no reference, no measurement (the other variant's hero)"
    assert rec["ortho_iou"] == round((rec["views"]["top"]["iou"] + rec["views"]["side"]["iou"]) / 2, 4)
    assert json.loads((out / sil.REPORT).read_text())["views"].keys() == rec["views"].keys()
    assert Image.open(out / "sil_top.png").size == (sil.GRID * 2, sil.GRID * 2)


def test_overlay_colours_mean_add_and_remove():
    ref = sheet([((20, 20, 300, 100), (120, 40, 30))])
    cand = stage([((20, 20, 300, 60), (120, 40, 30))])            # the candidate lacks the lower half
    colours = {tuple(c) for c in np.asarray(sil.overlay(cand, ref)).reshape(-1, 3)}
    assert (220, 40, 40) in colours and (150, 150, 150) in colours


def test_void_fraction_tells_a_pierced_body_from_a_solid_one_with_the_same_outline():
    rock = (90, 84, 80)
    solid = stage([((60, 30, 260, 150), rock)])
    painted = stage([((60, 30, 260, 150), rock), ((120, 60, 200, 120), (47, 43, 41))])      # a dark patch is still body
    pierced = stage([((60, 30, 260, 150), rock), ((120, 60, 200, 120), STAGE)])              # the backdrop shows through
    assert sil.compare(solid, pierced)["iou"] == 1.0                                         # the silhouette cannot see it
    assert sil.void_fraction(solid) == 0.0 and sil.void_fraction(painted) == 0.0
    assert abs(sil.void_fraction(pierced) - (81 * 61) / (201 * 121)) < 0.01
    c = sil.compare(painted, pierced)
    assert c["void"] == 0.0 and c["reference_void"] > 0.19


def test_the_subject_panel_crops_to_the_body_and_scales_to_one_height():
    wide = stage([((100, 80, 220, 100), (120, 40, 30))], size=(320, 180))
    panel = sil.subject_panel(wide, 100)
    assert panel.height == 100 and panel.width > 3 * panel.height, "a 120 x 20 body fills the panel, not the 16:9 frame"


def test_the_array_flood_fill_is_the_per_pixel_one():
    """`flood` replaced Pillow's pure-Python floodfill (0.5 s a mask). Same 4-connected region, on blobs with
    holes and specks."""
    rng = np.random.default_rng(7)
    for trial in range(6):
        coarse = rng.random((12, 16)) > 0.45
        open_ = np.kron(coarse, np.ones((5, 5), dtype=bool)) ^ (rng.random((60, 80)) > 0.97)
        ys, xs = np.nonzero(open_)
        seed = (int(ys[trial]), int(xs[trial]))
        im = Image.fromarray(np.where(open_, 255, 0).astype(np.uint8)).copy()
        ImageDraw.floodfill(im, (seed[1], seed[0]), 1)
        assert (sil.flood(open_, seed) == (np.asarray(im) == 1)).all()
    assert not sil.flood(np.zeros((4, 4), dtype=bool), (1, 1)).any(), "a seed on a closed pixel fills nothing"


def test_a_mask_asked_for_by_path_is_kept_until_the_file_changes(tmp_path):
    p = tmp_path / "v.png"
    stage([((40, 60, 280, 120), (120, 40, 30))]).save(p)
    first = sil.mask(p)
    assert sil.mask(p) is first and sil.bbox(first) == (40, 60, 281, 121)
    assert (sil.mask(Image.open(p)) == first).all(), "the same mask as from the image itself"
    stage([((40, 60, 100, 120), (120, 40, 30))]).save(p)
    assert sil.bbox(sil.mask(p)) == (40, 60, 101, 121), "a rewritten file is read again"
