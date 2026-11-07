"""Sampled separation is evidence to inspect, not proof of detachment. No Blender."""
import json

import numpy as np
import pytest

from glbkit import box, cube, write_cubes, write_glb
from mesh_jig import evaluate, review


def subdivide(mesh):
    vertices, faces = mesh
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    triangles = np.concatenate([np.stack(t, axis=1) for t in
                                [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]])
    return triangles.reshape(-1, 3).astype(np.float32), np.arange(len(triangles) * 3, dtype=np.uint16).reshape(-1, 3)


def joined(*meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces).astype(np.uint16)


def ellipsoid(size, centre=(0, 0, 0)):
    # A curved, closed organic lobe, with shared vertices after coordinate welding.
    v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.float32)
    f = np.array([(x, y, z) for x in (0, 1) for y in (2, 3) for z in (4, 5)], np.uint16)
    v, f = subdivide((v, f))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return (v * np.array(size) / 2 + centre).astype(np.float32), f


CASES = ["touching", "detached", "embedded", "intersecting", "retessellated", "reordered",
         "renamed", "joined_node", "detached_touching_pair", "crossed_beams",
         "organic_touching", "organic_detached", "organic_embedded", "organic_continuous"]
# Historical sampled values, not ground-truth distances. The wording fix must not change them.
SAMPLED_GAPS = [.078, .215, .123, .078, None, .078, .078, .078, .213, .078, None, .2, .126, None]


def fixture_mesh(path, case):
    # The first case is the original discovery, retaining its exact GLB construction.
    if case == "touching":
        return write_cubes(path, [cube(1.0), cube(.01, (.06, .06, .505))])
    body = box((1, 1, 1))
    detail = box((.01, .01, .01), (.06, .06, .505))
    names = ["body", "detail"]
    if case == "detached":
        detail = box((.01, .01, .01), (.06, .06, .705))  # true gap .2 m
    elif case == "embedded":
        detail = box((.01, .01, .01), (.06, .06, .4))    # entirely inside the solid
    elif case == "intersecting":
        detail = box((.01, .01, .01), (.06, .06, .5))    # crosses the body's face
    elif case == "retessellated":
        body = subdivide(body)                         # identical surface
    elif case == "reordered":
        v, f = body
        body = v[::-1].copy(), (len(v) - 1 - f[::-1, [1, 2, 0]]).astype(np.uint16)
    elif case == "renamed":
        names = ["petal", "hinge"]                    # names do not supply intent
    elif case == "joined_node":
        return write_glb(path, [("assembly", joined(body, detail), (0, 0, 0))])
    elif case == "detached_touching_pair":
        detail = joined(box((.01, .01, .01), (.06, .06, .705)),
                        box((.01, .01, .01), (.07, .06, .705)))
    elif case == "crossed_beams":
        body = box((1, .01, .01))
        detail = box((.01, 1, .01), (.06, .06, 0))
    elif case.startswith("organic_"):
        body = ellipsoid((.6, 1, .8))
        if case == "organic_continuous":
            return write_glb(path, [("lobe", body, (0, 0, 0))])
        z = {"organic_touching": .45, "organic_detached": .65, "organic_embedded": .2}[case]
        detail = ellipsoid((.08, .12, .1), (0, 0, z))
    return write_glb(path, [(name, mesh, (0, 0, 0)) for name, mesh in zip(names, [body, detail])])


def assert_uncertain(lines):
    for line in lines:
        assert "sampled point separation" in line
        assert "rounded estimate; sampling may miss contact" in line
        assert "contact is unverified" in line
        assert "Inspect actual contact before moving geometry" in line
        assert "if detached and intended to attach" in line
        assert "nothing holds it" not in line
        assert "from the nearest other part" not in line


@pytest.mark.parametrize("case", CASES)
def test_sampled_feedback_does_not_certify_contact_or_intent(tmp_path, case):
    found = review.floating(fixture_mesh(tmp_path / "shape.glb", case), {"height_m": 1.0})
    # Some contacting surfaces still flag; some don't. Neither result certifies contact.
    expected = SAMPLED_GAPS[CASES.index(case)]
    assert [f["gap_m"] for f in found] == ([] if expected is None else [expected, expected])
    if case == "touching":
        assert found == [{"node": "node0", "gap_m": .078, "nearest": "node1"},
                         {"node": "node1", "gap_m": .078, "nearest": "node0"}]
    before = json.dumps(found, allow_nan=False)
    lines = review.floating_lines(found)
    assert len(lines) == len(found)
    assert_uncertain(lines)
    assert json.dumps(found, allow_nan=False) == before


def test_historical_record_feedback_preserves_schema_scores_and_piece_identity():
    found = [{"node": "Tail", "gap_m": .036, "nearest": "Tail",
              "piece": {"triangles": 288, "of": 6, "centre_m": [0., -.4, -.12]}}]
    record = {"ok": True, "name": "historical", "numeric": {"numeric_score": .661},
              "silhouette": {}, "resemblance": {}, "floating": found}
    before = json.dumps(record, allow_nan=False)
    text = evaluate.feedback(record)
    assert "numeric score 0.6610" in text
    assert "FLOATING (sampled separation; inspect contact):" in text
    assert "touching no other" not in text
    assert "Tail (one of its 6 pieces, 288 triangles around (+0.00, -0.40, -0.12))" in text
    assert_uncertain(review.floating_lines(found))
    assert json.dumps(record, allow_nan=False) == before


def test_no_flag_has_no_attachment_claim():
    assert review.floating_lines([]) == []


def test_rounded_sample_distance_is_not_a_certified_upper_bound(tmp_path):
    path = write_cubes(tmp_path / "rounded.glb", [cube(1), cube(1, (0, 0, 1.20049))])
    pieces = review.node_triangles(path)
    tris = review.render.load(path).tris
    true_gap = float(tris[pieces["node1"], :, 2].min() - tris[pieces["node0"], :, 2].max())
    found = review.floating(path, {"height_m": 1})
    assert .20048 < true_gap < .20050
    assert [f["gap_m"] for f in found] == [.2, .2]
    assert all(f["gap_m"] < true_gap for f in found), "rounding can lower a valid unrounded upper bound"
    lines = review.floating_lines(found)
    assert all("rounded estimate" in line for line in lines)
    assert all("sampling may miss contact" in line for line in lines)
    assert all("upper bound on true surface gap" not in line for line in lines)
