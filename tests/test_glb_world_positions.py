"""World-space integrity regression: a NaN-translated box previously passed its
height contract. Validate transformed positions using the existing nonfinite
flag, retaining finite geometry and rejecting the complete artifact. Controls
cover explicit matrices, inherited TRS, indexed and unindexed geometry, and
finite singular transforms (degeneracy is a separate contract failure).
"""
import json

import numpy as np
import pytest

from glbkit import box, write_glb, _pack
from mesh_jig import glb


def transformed(path, kind, value, *, mixed=False, unindexed=False):
    parts = [("Bad", box((2, 2, 2)), (0, 0, 0))]
    if mixed:
        parts.append(("Good", box((1, 2, 3)), (0, 0, 0)))
    write_glb(path, parts)
    doc, blob = glb.read_glb(path)
    node = doc["nodes"][0]
    if kind == "matrix":
        matrix = np.eye(4)
        matrix[0, 3] = value
        node.pop("translation")
        node["matrix"] = matrix.T.ravel().tolist()
    elif kind == "parent":
        doc["scenes"][0]["nodes"][0] = len(doc["nodes"])
        doc["nodes"].append({"children": [0], "translation": [value, 0, 0]})
    elif kind == "overflow":
        # All stored values are finite; composing parent/child scales overflows.
        node["scale"] = [1e200, 1, 1]
        doc["scenes"][0]["nodes"][0] = len(doc["nodes"])
        doc["nodes"].append({"children": [0], "scale": [1e200, 1, 1]})
    elif kind == "position_overflow":
        # Finite world matrix, but multiplying finite vertex coordinates overflows.
        node["scale"] = [value, 1, 1]
        node["translation"] = [value, 0, 0]
    else:
        node[kind] = ([0, value, 0, 1] if kind == "rotation" else [value, 1, 1])
    if unindexed:
        del doc["meshes"][0]["primitives"][0]["indices"]
    doc["buffers"] = []
    return _pack(doc, bytearray(blob), path)


def test_nonfinite_translation_cannot_pass_contract(tmp_path):
    path = write_glb(tmp_path / "model.glb", [("Body", box((1, 2, 3)), (float("nan"), 0, 0))])
    stats = glb.stats(path)
    assert glb.check(stats, {"height_m": 2}), stats
    assert stats["nonfinite"] is True
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("kind,value", [
    ("translation", float("nan")), ("scale", float("inf")),
    ("rotation", float("nan")), ("matrix", float("-inf")),
    ("parent", float("nan")), ("overflow", 0), ("position_overflow", 1e308),
])
@pytest.mark.parametrize("mixed,unindexed", [(False, False), (True, True)])
def test_nonfinite_world_positions_reject_report(tmp_path, kind, value, mixed, unindexed):
    transformed(tmp_path / "model.glb", kind, value, mixed=mixed, unindexed=unindexed)
    (tmp_path / "model.py").write_text("def build(): pass\n")
    with np.errstate(invalid="ignore", over="ignore"):
        report = glb.write_report(tmp_path, {"glb": "model.glb", "axis_extents_m": {"Y": 2}})
    assert report["ok"] is False
    assert any("non-finite vertex positions" in issue for issue in report["issues"])
    stats = report["stats"]["model.glb"]
    assert stats["nonfinite"] is True
    assert stats["triangles"] == (12 if mixed else 0)
    assert stats["node_triangles"][0] == ["Bad", 0]
    assert stats["node_bounds"] == ([["Good", [-0.5, -1.0, -1.5], [0.5, 1.0, 1.5]]] if mixed else [])
    encoded = json.dumps(report, allow_nan=False)
    assert json.loads((tmp_path / glb.REPORT).read_text()) == json.loads(encoded)


@pytest.mark.parametrize("kind,value", [("matrix", 4), ("parent", -4), ("scale", -2)])
def test_finite_transforms_keep_known_bounds(tmp_path, kind, value):
    path = transformed(tmp_path / "model.glb", kind, value)
    stats = glb.stats(path)
    assert stats["nonfinite"] is False
    assert stats["triangles"] == 12
    assert stats["degenerate_triangles"] == 0
    assert stats["size"] == ([4, 2, 2] if kind == "scale" else [2, 2, 2])
    assert stats["centre"] == ([0, 0, 0] if kind == "scale" else [value, 0, 0])
    assert glb.check(stats, {"height_m": 2, "vertex_colors": True}) == []
    json.dumps(stats, allow_nan=False)


def test_finite_singular_transform_is_degenerate_not_nonfinite(tmp_path):
    stats = glb.stats(transformed(tmp_path / "model.glb", "scale", 0))
    assert stats["nonfinite"] is False
    assert stats["triangles"] == 12
    assert stats["zero_area_triangles"] == 8
    issues = glb.check(stats, {})
    assert any("degenerate triangles" in issue for issue in issues)
    assert not any("non-finite" in issue for issue in issues)


def test_missing_geometry_is_not_evidence_of_nonfinite_positions(tmp_path):
    stats = glb.stats(write_glb(tmp_path / "empty.glb", []))
    assert stats["nonfinite"] is False
    assert glb.check(stats, {}) == ["empty.glb: no triangles"]
