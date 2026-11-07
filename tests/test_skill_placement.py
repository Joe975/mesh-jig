"""Builder documentation must not override the project's placement frame."""
import json
from pathlib import Path

import pytest

from mesh_jig import brief, project
from test_character import calibrated


ROOT = Path(__file__).resolve().parents[1]


def placement_rules(document):
    path, heading = {"skill": ("skill/mesh-jig/SKILL.md", "## Script rules\n"),
                     "axes": ("docs/JIG.md", "## Axes\n")}[document]
    text = (ROOT / path).read_text(encoding="utf-8")
    return " ".join(text.split(heading, 1)[1].split("\n## ", 1)[0].split())


@pytest.mark.parametrize("origin", [None, [.02, -.03, -.5]])
@pytest.mark.parametrize("location", ["legacy", "criteria"])
@pytest.mark.parametrize("document", ["skill", "axes"])
def test_skill_preserves_the_real_standing_briefs_ground_frame(tmp_path, origin, location, document):
    _, spec, config = calibrated(tmp_path)
    if origin is not None:
        config["origin_m"] = origin
    data = {"model": spec, "refs": "."}
    if location == "legacy":
        spec["profile"] = config
    else:
        data["criteria"] = {"outline": config}
    (tmp_path / "jig.json").write_text(json.dumps(data), encoding="utf-8")
    proj = project.load(tmp_path)
    assert not proj.problems()
    said = brief.text(proj)
    assert "STANDING OUTLINES" in said
    assert f"fixed ground origin {config.get('origin_m', [0, 0, 0])} m in Blender axes" in said
    assert "Do not recenter the model's bounding box." in said
    rules = placement_rules(document)
    assert "Blender metres, +Z up, centred on the origin." not in rules
    assert "Follow the brief's placement instructions" in rules
    assert "calibrated standing model" in rules
    assert "preserve its fixed ground origin" in rules
    assert "do not recenter its bounding box" in rules


@pytest.mark.parametrize("spec", [{}, {"height_m": 1}, {"forward": "-Z", "length_m": .08}])
@pytest.mark.parametrize("document", ["skill", "axes"])
def test_skill_keeps_centring_when_the_brief_has_no_standing_calibration(tmp_path, spec, document):
    proj = project.Project(tmp_path, {"model": spec})
    said = brief.text(proj)
    assert "Centre the model on the origin." in said
    assert "fixed ground origin" not in said
    rules = placement_rules(document)
    assert "centred on the origin" in rules
    assert "A craft's nose points along **+Y** and it mirrors across X = 0" in rules


@pytest.mark.blender
@pytest.mark.parametrize("origin", [[0, 0, 0], [.02, -.03, -.5]])
def test_blender_export_preserves_the_briefs_calibrated_frame(blender, tmp_path, origin):
    from mesh_jig import build, glb, profile, render

    refs, spec, config = calibrated(tmp_path)
    config["origin_m"] = origin
    proj = project.Project(tmp_path, {"model": {**spec, "glb": "box.glb"},
                                     "refs": ".", "criteria": {"outline": config}})
    assert not proj.problems()
    assert f"fixed ground origin {origin} m in Blender axes" in brief.text(proj)
    reference = proj.reference_profile()
    # The same asymmetric box as the CPU fixture, built and exported by Blender.
    for name, centre in (("preserved", [origin[0] + .1, origin[1] + .05, origin[2] + .5]),
                         ("recentered", [0, 0, 0])):
        script = tmp_path / f"{name}.py"
        script.write_text(
            "import bpy\n\ndef build():\n"
            f"    bpy.ops.mesh.primitive_cube_add(size=1, location={tuple(centre)!r})\n"
            "    bpy.context.object.dimensions = (.4, .2, 1)\n", encoding="utf-8")
        out = tmp_path / name
        assert build.build_variants(script, out, proj.spec) == ""
        path = out / "box.glb"
        assert not glb.check(glb.stats(path), proj.spec)
        diff = profile.glb_diff(reference, path)
        render.render(render.load(path), spec["shots"][0]).save(out / "front.png")
        (out / "profile_diff.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
        if name == "preserved":
            assert diff["missing_stations"] == []
            assert diff["mae_mm"] == pytest.approx(0, abs=1e-4)
        elif origin == [0, 0, 0]:
            assert diff["missing_stations"] == [f"{view}:head" for view in refs]
            assert diff["mae_mm"] is None
        else:
            assert diff["missing_stations"] == []
            assert diff["mae_mm"] == pytest.approx(70, abs=1e-4)
