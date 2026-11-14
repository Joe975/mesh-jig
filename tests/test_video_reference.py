"""skill/mesh-jig-video-reference: the deterministic steps (mask, flat white, sheet, view specs) on drawn pictures
with known answers. The segmenter (rembg) and ffmpeg are not run here."""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SKILL = Path(__file__).resolve().parents[1] / "skill" / "mesh-jig-video-reference"
vs = runpy.run_path(str(SKILL / "scripts" / "video_sheet.py"))


def test_label_joins_diagonal_neighbours_and_keeps_separate_pieces_apart():
    m = np.zeros((6, 8), bool)
    m[0, 0] = m[1, 1] = m[2, 2] = True          # one diagonal piece
    m[0, 5:8] = m[1, 7] = True                    # an L
    m[4:6, 0:3] = True                            # a block
    m[5, 5] = True                                # a single pixel
    lab, n = vs["label"](m)
    assert n == 4
    assert len({lab[0, 0], lab[1, 1], lab[2, 2]}) == 1
    assert lab[0, 5] == lab[1, 7]
    assert (lab > 0).sum() == m.sum() and ((lab > 0) == m).all()


def test_label_merges_a_u_shape_whose_arms_meet_only_at_the_bottom():
    m = np.zeros((5, 5), bool)
    m[:, 0] = m[:, 4] = m[4, :] = True
    _, n = vs["label"](m)
    assert n == 1


def test_dilate_is_a_square_that_does_not_wrap_round():
    m = np.zeros((7, 7), bool)
    m[0, 0] = True
    d = vs["dilate"](m, 2)
    assert d[:3, :3].all() and d.sum() == 9, "the corner pixel grows into the picture, not across to the far edge"


def figure():
    """A 200x200 crop: a dark body the segmenter is sure of, a green blade touching it that the segmenter left
    see-through, a dark scrap off to the side (a base remnant), a green blob far away (something in the background)
    and a one-pixel pinhole in the body."""
    rgb = np.full((200, 200, 3), 200, np.uint8)              # grey wall
    alpha = np.zeros((200, 200), np.uint8)
    rgb[60:160, 70:120] = (30, 45, 40)                       # body: dark green-grey, 100x50
    alpha[60:160, 70:120] = 255
    rgb[20:60, 100:106] = (60, 200, 70)                      # blade above the body, touching it
    alpha[20:60, 100:106] = 40
    rgb[175:178, 150:153] = (20, 20, 20)                     # scrap: 9 px, under 2% of the body
    alpha[175:178, 150:153] = 255
    rgb[10:20, 10:20] = (50, 220, 60)                        # green blob, nowhere near the body
    alpha[10:20, 10:20] = 0
    alpha[100, 90] = 0                                       # pinhole
    return rgb, alpha


def test_the_mask_restores_the_see_through_blade_and_drops_scraps_and_strays():
    rgb, alpha = figure()
    m = vs["subject_mask"](alpha, rgb)
    assert m[20:60, 100:106].all(), "the green blade touching the body is subject"
    assert not m[175:178, 150:153].any(), "a scrap under 2% of the largest piece is dropped"
    assert not m[10:20, 10:20].any(), "green away from the body is not subject"
    assert m[100, 90], "a pinhole in the body is filled"
    assert m[60:160, 70:120].all()


def test_without_the_accent_the_blade_is_lost():
    rgb, alpha = figure()
    m = vs["subject_mask"](alpha, rgb, accent=None)
    assert not m[25:55, 100:106].any(), "the segmenter alone keeps nothing it was unsure of"


def test_a_large_enclosed_gap_stays_open():
    rgb = np.full((100, 100, 3), 30, np.uint8)
    alpha = np.zeros((100, 100), np.uint8)
    alpha[10:90, 10:90] = 255
    alpha[30:70, 30:70] = 0                                  # 1600 px gap, like the space between two legs
    m = vs["subject_mask"](alpha, rgb, accent=None)
    assert not m[30:70, 30:70].any()


def test_on_white_is_pure_white_outside_the_subject_and_trimmed_with_a_margin():
    rgb, alpha = figure()
    m = vs["subject_mask"](alpha, rgb)
    im = np.asarray(vs["on_white"](rgb, alpha, m, margin=0.1))
    x0, y0, x1, y1 = vs["tight"](m)
    margin = round(0.1 * max(x1 - x0, y1 - y0))
    assert im.shape[:2] == (y1 - y0 + 2 * margin, x1 - x0 + 2 * margin)
    assert (im[:margin] == 255).all() and (im[:, :margin] == 255).all()
    body = im[margin + (60 - y0) + 5, margin + (70 - x0) + 5]
    assert tuple(body) == (30, 45, 40), "inside the mask the source colour is opaque"
    blade = im[margin + 30 - y0, margin + 102 - x0]
    assert tuple(blade) == (60, 200, 70), "the restored blade is opaque, not washed out by its low alpha"


def test_the_sheet_puts_standing_views_on_one_ground_line_and_keeps_their_scale():
    def view(h, w):
        a = np.zeros((h + 20, w + 20), np.uint8)
        a[10:10 + h, 10:10 + w] = 255
        rgb = np.full((h + 20, w + 20, 3), 40, np.uint8)
        m = a > 0
        return vs["on_white"](rgb, a, m, margin=0.0), m
    views = {"front": view(100, 40), "side": view(80, 60), "rear": view(100, 40),
             "right": view(90, 60), "hero": view(95, 50), "top": view(30, 40)}
    sheet = np.asarray(vs["compose"](views, top_scale=2.0, pad=10, label_h=20))
    dark = sheet.min(axis=-1) < 128
    cell_w = sheet.shape[1] // 3
    first_row_h = 100 + 20                                   # tallest standing view plus padding
    bottoms, heights = [], []
    for c in range(3):
        ys = np.flatnonzero(dark[:first_row_h, c * cell_w + 5:(c + 1) * cell_w - 5].any(axis=1))
        bottoms.append(ys.max()); heights.append(ys.max() - ys.min() + 1)
    assert len(set(bottoms)) == 1, "front, side and rear stand on one line"
    assert heights == [100, 80, 100], "no view is rescaled"
    top_cell = dark[first_row_h + 20:, 2 * cell_w:]
    ys, xs = np.nonzero(top_cell[:-30])                      # leave out the label row
    assert (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) == (60, 80), "top is scaled by top_scale"


@pytest.mark.parametrize("spec, expect", [
    ("front=frames/f_073.jpg:480,150,1260,1000", ("front", "frames/f_073.jpg", (480, 150, 1260, 1000), 0)),
    ("top=C:/clips/f_103.jpg:650,350,1300,850:-90", ("top", "C:/clips/f_103.jpg", (650, 350, 1300, 850), -90)),
])
def test_a_view_spec_keeps_a_drive_letter_and_reads_the_rotation(spec, expect):
    assert vs["parse_view"](spec) == expect


def test_a_view_that_is_not_a_camera_is_refused():
    with pytest.raises(SystemExit):
        vs["parse_view"]("left=f.jpg:0,0,10,10")


def test_photo_cli_produces_cutouts_masks_overlay_and_sheet(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    # Replace only segmentation: the real crop/rotation, mask cleanup and sheet CLI run.
    # The input is a drawn rectangle, with no external video, image or model download.
    def remove(image, **kwargs):
        image = image.convert('RGBA')
        alpha = np.where(np.asarray(image)[..., :3].min(axis=-1) < 128, 255, 0).astype(np.uint8)
        image.putalpha(Image.fromarray(alpha))
        return image
    monkeypatch.setitem(sys.modules, 'rembg', SimpleNamespace(new_session=lambda _: object(), remove=remove))
    image = np.full((80, 100, 3), 255, np.uint8)
    image[20:60, 30:70] = (40, 50, 60)
    photo = tmp_path/'photo.png'
    Image.fromarray(image).save(photo)
    cut, clean = tmp_path/'cut', tmp_path/'clean'
    views = list(vs['VIEWS'])
    specs = [f'{view}={photo.as_posix()}:10,10,90,70' + (':90' if view == 'top' else '') for view in views]
    vs['main'](['cut', str(cut), *specs])
    vs['main'](['clean', str(cut), str(clean), *views, '--accent', 'none'])
    overlay, sheet = tmp_path/'overlay.png', tmp_path/'sheet.png'
    vs['main'](['overlay', str(cut), str(clean), str(overlay), *views, '--height', '80'])
    vs['main'](['sheet', str(clean), str(sheet)])
    assert Image.open(cut/'front_src.png').size == (80, 60)
    assert Image.open(cut/'top_src.png').size == (60, 80)
    for view in views:
        mask = np.asarray(Image.open(clean/f'{view}_mask.png'))
        assert (mask > 0).sum() == 1600
    assert Image.open(overlay).height == 80
    assert Image.open(sheet).width > 0 and Image.open(sheet).height > 0
