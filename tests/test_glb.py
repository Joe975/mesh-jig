"""mesh_jig.glb: the numpy GLB reader and the model contract gate."""
import json

import pytest

from glbkit import box, textured_glb, write_glb
from mesh_jig import glb

SPEC = {"script": "ship.py", "variants": ["gun", "missile"], "glb": "ship_{variant}.glb", "join": True,
        "length_m": 0.08, "length_tolerance": 0.05, "forward": "-Z", "max_triangles": 1500, "min_triangles": 12,
        "single_mesh": True, "vertex_colors": True, "min_colors": 3}
TEX_SPEC = dict(SPEC, variants=["gun"], textured=True, min_colors=1)


def ship(path, **kw):
    return write_glb(path, [("Ship", box((0.066, 0.026, 0.080)), (0, 0, 0))], **kw)


def test_stats_reads_bounds_triangles_colours_and_transforms(tmp_path):
    p = write_glb(tmp_path / "a.glb", [("Hull", box((0.06, 0.02, 0.08)), (0, 0, 0)),
                                       ("Fin", box((0.01, 0.01, 0.01)), (0, 0.02, 0.03))])
    s = glb.stats(p)
    assert s["mesh_nodes"] == ["Hull", "Fin"]
    assert s["triangles"] == 24
    assert s["bounds_max"][1] == pytest.approx(0.025, abs=1e-6)     # the Fin node's translation is applied
    assert s["size"][2] == pytest.approx(0.08, abs=1e-6)
    assert s["distinct_colors"] == 3 and s["primitives_without_color0"] == 0


def test_interleaved_positions_read_like_packed(tmp_path):
    a = glb.stats(ship(tmp_path / "a.glb"))
    b = glb.stats(ship(tmp_path / "b.glb", interleave=True))
    assert a["size"] == b["size"] and a["triangles"] == b["triangles"]


def test_contract_passes_a_conforming_pair_and_writes_the_report(tmp_path):
    (tmp_path / "ship.py").write_text("def build():\n    pass\n")
    ship(tmp_path / "ship_gun.glb")
    ship(tmp_path / "ship_missile.glb")
    issues, detail = glb.check_dir(tmp_path, SPEC)
    assert issues == []
    assert set(detail) == {"ship_gun.glb", "ship_missile.glb"}
    assert glb.write_report(tmp_path, SPEC)["ok"] is True
    assert json.loads((tmp_path / glb.REPORT).read_text())["ok"] is True


@pytest.mark.parametrize("parts, fragment", [
    ([("K", box((0.066, 0.026, 0.120)), (0, 0, 0))], "length along Z"),                     # 120 mm, not 80
    ([("K", box((0.080, 0.026, 0.066)), (0, 0, 0.0))], "length along Z"),                   # lying along X
    ([("K", box((0.050, 0.026, 0.080)), (0, 0, 0)), ("P", box((0.01, 0.01, 0.01)), (0, -0.02, 0))], "mesh nodes"),
    ([("K", box((0.066, 0.026, 0.080)), (0, 0, 0.05))], "from the origin"),
    ([("K", box((0.030, 0.090, 0.080)), (0, 0, 0))], "tallest along Y"),
])
def test_contract_names_each_breach(tmp_path, parts, fragment):
    (tmp_path / "ship.py").write_text("def build():\n    pass\n")
    for v in ("gun", "missile"):
        write_glb(tmp_path / f"ship_{v}.glb", parts)
    issues, _ = glb.check_dir(tmp_path, SPEC)
    assert any(fragment in i for i in issues), issues


def test_a_span_contract_allows_wider_than_long_and_holds_the_span(tmp_path):
    """With a span wider than the length, "wider along X" made every builder shrink its wings to pass."""
    (tmp_path / "ship.py").write_text("def build():\n    pass\n")
    wide = [("K", box((0.094, 0.031, 0.080)), (0, 0, 0))]
    for v in ("gun", "missile"):
        write_glb(tmp_path / f"ship_{v}.glb", wide)
    assert any("wider along X" in i for i in glb.check_dir(tmp_path, SPEC)[0]), "no span in the contract: the heuristic holds"
    assert glb.check_dir(tmp_path, dict(SPEC, span_m=0.094))[0] == []
    narrow = [("K", box((0.078, 0.031, 0.080)), (0, 0, 0))]
    for v in ("gun", "missile"):
        write_glb(tmp_path / f"ship_{v}.glb", narrow)
    issues = glb.check_dir(tmp_path, dict(SPEC, span_m=0.094))[0]
    assert any("span along X is 0.0780 m, contract 0.0940 m +/- 10%" in i for i in issues), issues


def test_missing_colours_script_and_variant_fail(tmp_path):
    ship(tmp_path / "ship_gun.glb", colours=False)
    text = " | ".join(glb.check_dir(tmp_path, SPEC)[0])
    assert "without COLOR_0" in text and "ship.py missing" in text and "ship_missile.glb missing" in text


def test_triangle_budget(tmp_path):
    issues = glb.check(glb.stats(ship(tmp_path / "ship_gun.glb")), dict(SPEC, max_triangles=10))
    assert any("budget 10" in i for i in issues)


def test_not_a_glb(tmp_path):
    (tmp_path / "ship.py").write_text("def build():\n    pass\n")
    (tmp_path / "ship_gun.glb").write_bytes(b"glTF-ish but not" * 4)
    ship(tmp_path / "ship_missile.glb")
    assert any("unreadable GLB" in i for i in glb.check_dir(tmp_path, SPEC)[0])


def test_the_judged_variant_is_the_named_one_else_the_first():
    assert glb.judged_glb(SPEC) == "ship_gun.glb"
    assert glb.judged_glb(dict(SPEC, judge_variant="missile")) == "ship_missile.glb"
    assert glb.judged_glb({"glb": "rock.glb"}) == "rock.glb"


def test_textured_stats_count_images_uvs_and_textured_primitives(tmp_path):
    s = glb.stats(textured_glb(tmp_path / "t.glb"))
    assert s["images"] == 1 and s["textured_primitives"] == 1 and s["primitives_without_texcoord0"] == 0
    s = glb.stats(textured_glb(tmp_path / "u.glb", uvs=False, image=False))
    assert s["images"] == 0 and s["textured_primitives"] == 0 and s["primitives_without_texcoord0"] == 1


def test_textured_contract_passes_a_bound_atlas_and_names_each_breach(tmp_path):
    (tmp_path / "ship.py").write_text("def build():\n    pass\n")
    (tmp_path / "atlas.png").write_bytes(b"png")
    textured_glb(tmp_path / "ship_gun.glb")
    assert glb.check_dir(tmp_path, TEX_SPEC)[0] == []
    assert glb.check_dir(tmp_path, dict(TEX_SPEC, textured=False))[0] == [], "the flat spec does not care about textures"
    textured_glb(tmp_path / "ship_gun.glb", uvs=False)
    assert any("TEXCOORD_0" in i for i in glb.check_dir(tmp_path, TEX_SPEC)[0])
    textured_glb(tmp_path / "ship_gun.glb", image=False)
    issues = glb.check_dir(tmp_path, TEX_SPEC)[0]
    assert any("embeds no image" in i for i in issues) and any("no base colour texture" in i for i in issues)
    textured_glb(tmp_path / "ship_gun.glb")
    (tmp_path / "atlas.png").unlink()
    assert any("atlas atlas.png missing" in i for i in glb.check_dir(tmp_path, TEX_SPEC)[0])
    assert glb.atlas_name({"atlas": "skin.png"}) == "skin.png" and glb.atlas_name({}) == "atlas.png"
    assert glb.texture_mode({}) == "vertex" and glb.texture_mode({"textured": True}) == "atlas"
    assert glb.texture_mode({"textured": True, "texture_mode": "multiply"}) == "multiply"
