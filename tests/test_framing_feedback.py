"""Frame contact is evidence of a presentation limit, not one-direction score bias."""
import copy

import numpy as np
from PIL import Image, ImageDraw
import pytest

from glbkit import box, write_glb
from mesh_jig import evaluate, render, silhouette


def images():
    body = Image.new("RGB", (128, 128), render.BACKGROUND)
    ImageDraw.Draw(body).rectangle((40, 50, 88, 100), fill=(20, 20, 20))
    antenna = body.copy()
    ImageDraw.Draw(antenna).rectangle((60, 10, 68, 50), fill=(20, 20, 20))
    return body, antenna, antenna.crop((0, 50, 128, 128))


@pytest.mark.parametrize("required", [False, True])
def test_cropping_can_raise_or_lower_the_actual_normalized_overlap(required):
    body, antenna, cropped = images()
    reference = antenna if required else body
    full_score = silhouette.compare(antenna, reference)["iou"]
    cut_score = silhouette.compare(cropped, reference)["iou"]
    assert render.touches_edge(cropped)
    assert cut_score < full_score - .1 if required else cut_score > full_score + .1


def test_edge_pixels_can_come_from_geometry_strictly_inside_the_frame(tmp_path):
    size, bottom = 64, -.25
    top = .5 - .1 / size
    path = write_glb(tmp_path / "inside.glb", [
        ("Inside", box((.5, top-bottom, .1), centre=(0, (top+bottom)/2, 0)), (0, 0, 0))])
    mesh = render.load(path)
    shot = {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1}
    eye, basis, _ = render.camera(shot)
    cam = (mesh.tris-eye) @ basis.T
    x, y = size/2 + cam[..., 0]*size, size/2 - cam[..., 1]*size
    assert x.min() > 0 and x.max() < size and y.min() > 0 and y.max() < size
    ids, bary, _ = render.rasterise(mesh, shot, width=size, height=size)
    image = Image.fromarray(render.shade(mesh, ids, bary, basis[2], render.lighting({}, basis)))
    assert (ids[0] >= 0).sum() == 32
    assert render.touches_edge(image)


@pytest.mark.parametrize("view", ["front", "hero"])
def test_feedback_reports_possible_crop_and_unknown_overlap_direction_once(view):
    body, _, _ = images()
    record = {"name": "a01", "ok": True, "cropped": [view], "numeric": {"ortho_iou": .7},
              "silhouette": {"front": silhouette.compare(body, body)}}
    before = copy.deepcopy(record)
    text = evaluate.feedback(record)
    warning = [line for line in text.splitlines() if "frame edge" in line]
    assert len(warning) == 1 and view in warning[0]
    assert "may be cropped" in warning[0]
    assert "raise or lower" in warning[0]
    assert "understated" not in warning[0] and "is cut off there" not in warning[0]
    assert record == before


def test_no_frame_contact_produces_no_framing_warning():
    assert "frame edge" not in evaluate.feedback({"name": "a01", "ok": True, "cropped": []})
