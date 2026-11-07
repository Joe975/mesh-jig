"""Direct readiness rejects invalid geometry before producing partial diagnostics.

The original NaN-translated box crashed in mirror-twin integer arithmetic.
World-space validity is separate from absent UVs and finite degeneracy.
"""
import json

import numpy as np
import pytest

from glbkit import _pack, box, write_glb
from mesh_jig import build, cli, glb, readiness
from test_glb_world_positions import transformed


@pytest.mark.parametrize("reader", [readiness.read, readiness.measure])
def test_nan_translated_box_is_explicitly_rejected(tmp_path, reader):
    path = write_glb(tmp_path / "model.glb", [("Body", box((1, 2, 3)), (np.nan, 0, 0))])
    with pytest.raises(ValueError, match="non-finite.*positions"):
        reader(path)


@pytest.mark.parametrize("kind,value", [
    ("translation", np.nan), ("scale", np.inf), ("rotation", np.nan),
    ("matrix", -np.inf), ("parent", np.nan), ("overflow", 0),
    ("position_overflow", 1e308),
])
@pytest.mark.parametrize("mixed,unindexed", [(False, False), (True, True)])
def test_world_positions_rejected_before_any_diagnostics(tmp_path, monkeypatch, kind, value, mixed, unindexed):
    path = transformed(tmp_path / "model.glb", kind, value, mixed=mixed, unindexed=unindexed)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid geometry reached diagnostics")

    monkeypatch.setattr(readiness, "welded", unexpected)
    with np.errstate(invalid="ignore", over="ignore"):
        with pytest.raises(ValueError, match="non-finite.*positions"):
            readiness.measure(path, surface_symmetry=True)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("unused", [False, True])
def test_stored_nonfinite_positions_in_mixed_primitives_reject_whole_file(tmp_path, value, unused):
    verts, faces = box((1, 2, 3))
    if unused:
        verts = np.concatenate([verts, [[value, 0, 0]]]).astype(np.float32)
    else:
        verts[0, 0] = value
    path = write_glb(tmp_path / "mixed.glb", [
        ("Good", box((1, 2, 3)), (0, 0, 0)), ("Bad", (verts, faces), (0, 0, 0)),
    ])
    doc, blob = glb.read_glb(path)
    # Both primitives belong to one node; a valid first primitive cannot mask the second.
    doc["meshes"][0]["primitives"] += doc["meshes"][1]["primitives"]
    doc["scenes"][0]["nodes"] = [0]
    doc["buffers"] = []
    _pack(doc, bytearray(blob), path)
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match="non-finite.*positions"):
            readiness.measure(path)


@pytest.mark.parametrize("kind,value", [("matrix", 4), ("parent", -4), ("scale", -2)])
def test_finite_transforms_match_baked_geometry(tmp_path, kind, value):
    path = transformed(tmp_path / "model.glb", kind, value)
    verts, faces = box((2, 2, 2))
    if kind == "scale":
        verts[:, 0] *= value
    else:
        verts[:, 0] += value
    baked = write_glb(tmp_path / "baked.glb", [("Bad", (verts, faces), (0, 0, 0))])
    np.testing.assert_array_equal(readiness.read(path)["tris"], verts[faces])
    got, expected = readiness.measure(path), readiness.measure(baked)
    for reading in (got, expected):
        reading.pop("file")
        reading["cost"].pop("bytes")
    assert got == expected
    json.dumps(got, allow_nan=False)


def test_finite_translated_box_without_uvs_keeps_known_diagnostics(tmp_path):
    path = write_glb(tmp_path / "model.glb", [("Body", box((1, 2, 3)), (4, 0, 0))])
    assert np.isnan(readiness.read(path)["uvs"]).all()
    got = readiness.measure(path)
    assert got["cost"]["triangles"] == 12
    assert got["surface"] == {"pieces": 1, "pieces_90": 1, "largest_piece": 1.0,
                              "open_edges": 0.0, "slivers": 0.0, "buried": 0.0, "symmetry": 0.0}
    assert got["uv"] is None
    assert got["height_m"] == 2
    json.dumps(got, allow_nan=False)


def test_empty_scene_is_not_nonfinite_geometry(tmp_path):
    path = write_glb(tmp_path / "empty.glb", [])
    got = readiness.measure(path, surface_symmetry=True)
    assert got["cost"]["triangles"] == 0
    assert got["surface"]["pieces"] == 0
    assert got["uv"] is None
    assert got["surface_symmetry"]["status"] == "unavailable"
    json.dumps(got, allow_nan=False)


def test_finite_singular_transform_keeps_degeneracy_diagnostics(tmp_path):
    got = readiness.measure(transformed(tmp_path / "model.glb", "scale", 0))
    assert got["cost"]["triangles"] == 12
    assert got["surface"]["slivers"] > 0
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("options", [[], ["--json"], ["--surface-symmetry"]])
@pytest.mark.parametrize("invalid", [False, True])
def test_ready_cli_reports_error_without_blender_or_partial_output(tmp_path, capsys, monkeypatch, options, invalid):
    def unexpected(*args, **kwargs):
        pytest.fail("ready must not run or locate Blender")

    monkeypatch.setattr(build, "find_blender", unexpected)
    monkeypatch.setattr(build, "build_variants", unexpected)
    (tmp_path / "jig.json").write_text(json.dumps({"name": "test", "model": {"glb": "model.glb"}}))
    out = tmp_path / "attempts" / "a01" / "out"
    out.mkdir(parents=True)
    write_glb(out / "model.glb", [("Body", box((1, 2, 3)), (np.nan if invalid else 0, 0, 0))])
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = cli.main(["ready", str(tmp_path), "a01", *options])
    captured = capsys.readouterr()
    if invalid:
        assert result == 2
        assert captured.out == ""
        assert "non-finite" in captured.err and "positions" in captured.err
        assert "Traceback" not in captured.err
    else:
        assert result == 0
        assert captured.err == ""
        if "--json" in options:
            assert json.loads(captured.out)["cost"]["triangles"] == 12
        else:
            assert "MESH READINGS" in captured.out
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
