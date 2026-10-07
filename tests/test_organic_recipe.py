from pathlib import Path

import pytest

from mesh_jig import build, glb, render, review, script_check

RECIPE = Path(__file__).resolve().parents[1] / "examples" / "organic" / "recipe.py"


def test_recipe_keeps_the_existing_script_import_restrictions():
    source = RECIPE.read_text(encoding="utf-8")
    assert script_check.check_script(source) == []
    assert "import of 'os' not allowed" in ";".join(script_check.check_script(source + "\nimport os\n"))


@pytest.mark.blender
def test_fused_colored_skin_fitted_vest_and_transported_frames_in_real_blender(blender, tmp_path):
    script = tmp_path / "model.py"
    script.write_text(RECIPE.read_text(encoding="utf-8") + '''
recipe_build = build
def build():
    frames = curve_frames([(0,0,0), (0,0,1), (.5,0,1.5), (1,0,1.5)])
    for i in range(len(frames)-1):
        assert frames[i][0].dot(frames[i+1][0]) > .001, "curve frame flipped"
    recipe_build()
''', encoding="utf-8")
    spec = {"script": "model.py", "glb": "organic.glb", "vertex_colors": True, "min_colors": 3,
            "max_triangles": 11000, "join": False}
    out = tmp_path / "out"
    assert build.build_variants(script, out, spec) == ""
    path = out / "organic.glb"
    stats = glb.stats(path)
    assert set(stats["mesh_nodes"]) == {"Skin", "Vest", "Tail"}
    assert not glb.check(stats, spec)
    cfg = {"parts": {"skin": {"nodes": ["Skin"], "smooth_min": .9, "max_components": 1},
                     "vest": {"nodes": ["Vest"]}}, "clearance": [{
           "inner": "skin", "outer": "vest", "inner_hit": "last", "min_m": .002, "max_m": .065,
           "rays": [{"origin_m": [0, 0, z], "direction": d} for z in (.31, .35, .36, .47, .58, .59)
                    for d in ([1, 0, 0], [-1, 0, 0], [0, -1, 0])]}]}
    rec = review.measure(path, spec, cfg, {}, {})
    assert not rec["warnings"], rec
    assert all(c["passed"] for c in rec["checks"]), rec
    assert rec["diagnostics"]["skin"]["triangles"] <= 6000
    assert abs(stats["bounds_min"][1]) < .00001  # recipe grounds final skin; Tail/Vest are above it
    image = render.render(render.load(path), {"cam_pos": [-2, 1.5, -3], "cam_target": [0, .6, 0], "ortho": 1.4})
    image.save(tmp_path / "organic.png")
    assert silhouette_nonempty(image)


def silhouette_nonempty(image):
    from mesh_jig.silhouette import mask
    return mask(image).sum() > 1000
