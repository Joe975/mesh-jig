import json

from PIL import Image
import pytest

from glbkit import textured_glb
from mesh_jig import cli, glb, material_review, rig, sheet


def decision(channels, **fields):
    return dict(region="shell", cue="polished shell", check="inspect the highlight", nodes=["K"],
                channels=channels, **fields)


def test_new_contract_requires_decision_at_surface_stage(tmp_path):
    sheet.init(tmp_path)
    source = textured_glb(tmp_path / "base.glb")
    args = ["materials", str(tmp_path), str(source)]
    assert cli.main(args) == 2
    contract = tmp_path / "jig.json"
    data = json.loads(contract.read_text())
    data["material_review"] = [decision({})]
    contract.write_text(json.dumps(data))
    assert cli.main(args) == 0
    data["material_review"] = [decision({"roughness": 0.22})]
    contract.write_text(json.dumps(data))
    assert cli.main(args) == 1
    out = tmp_path / "finished.glb"
    assert cli.main(args + ["--out", str(out)]) == 0
    assert cli.main(["materials", str(tmp_path), str(out)]) == 0


def test_apply_checks_bound_images_uvs_and_preserves_geometry(tmp_path):
    source = textured_glb(tmp_path / "base.glb", image=False)
    doc, binary = glb.read_glb(source)
    blob = bytearray(binary)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    pos = glb.accessor(doc, blob, attrs["POSITION"])
    attrs["TEXCOORD_0"] = rig.append_accessor(doc, blob, pos[:, :2], "VEC2")
    doc["nodes"].append(dict(name="Untreated", mesh=0))
    doc["scenes"][0]["nodes"].append(1)
    rig.write_glb(source, doc, blob)
    Image.new("RGB", (4, 4), (120, 160, 240)).save(tmp_path / "map.png")
    maps = {key: "map.png" for key in material_review.MAPS}
    row = decision(dict.fromkeys(maps, True) | {"roughness": 0.3, "emission": True},
                   maps=maps, emission_rgb=[1, 0, 0])
    out = tmp_path / "finish.glb"
    material_review.apply(source, out, tmp_path, [row])
    assert material_review.check(out, [row]) == []
    result, after = glb.read_glb(out)
    assert after.startswith(blob)
    assert len(result["textures"]) == 1
    assert result["meshes"][result["nodes"][1]["mesh"]]["primitives"][0]["material"] == 0
    assert material_review.check(source, [row])
    row["nodes"] = ["absent"]
    before = out.read_bytes()
    with pytest.raises(ValueError, match="expected one mesh node"):
        material_review.apply(source, out, tmp_path, [row])
    assert out.read_bytes() == before


@pytest.mark.parametrize("uvs,reason", [(False, "TEXCOORD_0"), (True, "UVs cover no triangle")])
def test_reject_missing_or_collapsed_uvs(tmp_path, uvs, reason):
    path = textured_glb(tmp_path / "base.glb", uvs=uvs, image=False)
    Image.new("RGB", (4, 4)).save(tmp_path / "normal.png")
    row = decision({"normal_map": True}, maps={"normal_map": "normal.png"})
    with pytest.raises(ValueError, match=reason):
        material_review.apply(path, tmp_path / "finish.glb", tmp_path, [row])
    assert not (tmp_path / "finish.glb").exists()
