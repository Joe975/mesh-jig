from io import BytesIO

from PIL import Image
import pytest

from glbkit import textured_glb
from mesh_jig import agent, brief, build, cli, gauntlet, glb, production, project, readiness, sheet


def test_agent_decisions_resolve_once_across_both_workflows(tmp_path):
    assert cli.main(["init", str(tmp_path), "--open-ended"]) == 0
    job = project.load(tmp_path)
    before = job.data["criteria"]
    assert production.pending(job.spec["production"]) == ["max_triangles", "colour"]
    assert "pending" in build.build_variants(tmp_path / "unused", tmp_path, job.spec)
    tools = agent.Tools(job, agent.Report(), attempts=1, judge=False)
    args = {"reason": "A single stripe crosses the hull at full view.", "colour": "texture", "max_triangles": 1800}
    text, _ = tools.call("choose_production", args)
    assert "max_triangles: 1800 (agent)" in text and "YOUR SCRIPT PAINTS" in text
    visual = gauntlet.Files(job, tmp_path / "attempts" / "model")
    assert "choose_production" in [s["function"]["name"] for s in visual.schemas]
    assert "colour: texture (agent)" in visual.call("choose_production", args)
    assert "pending" in visual.call("choose_production", dict(args, max_triangles=2000))
    assert project.load(tmp_path).data["criteria"] == before
    assert project.load(tmp_path).spec["max_triangles"] == 1800


@pytest.mark.parametrize("colour,mode,source", [
    ("vertex", "vertex", ""), ("texture", "atlas", "script"),
    ("atlas", "atlas", "plan"), ("multiply", "multiply", "plan")])
def test_user_precedes_preset_and_agent(tmp_path, colour, mode, source):
    assert cli.main(["init", str(tmp_path), "--target", "quest", "--max-triangles", "8000", "--colour", colour]) == 0
    job = project.load(tmp_path)
    assert job.spec["target"]["max_triangles"] == job.spec["max_triangles"] == 8000
    assert glb.texture_mode(job.spec) == mode and glb.texture_from(job.spec) == source
    assert "max_triangles: 8000 (user)" in brief.model_facts(job.spec)
    with pytest.raises(ValueError, match="pending"):
        production.choose(job, "more", max_triangles=9000)


def test_preset_precedes_agent_and_legacy_keeps_fixed_defaults(tmp_path):
    sheet.init(tmp_path / "preset", target="pcvr")
    job = project.load(tmp_path / "preset")
    assert "32000 (preset pcvr)" in brief.model_facts(job.spec)
    assert cli.main(["choose-production", str(job.root), "--colour", "vertex", "--reason", "Solid blocks"]) == 0
    assert job.spec["vertex_colors"]
    sheet.init(tmp_path / "fixed")
    fixed = project.load(tmp_path / "fixed")
    assert fixed.spec["max_triangles"] == 1500 and glb.texture_from(fixed.spec) == "script"
    with pytest.raises(ValueError, match="fixed"):
        production.choose(fixed, "changed", colour="vertex")


def test_cost_reads_dimensions_from_the_export(tmp_path):
    png = BytesIO()
    Image.new("RGB", (64, 32)).save(png, format="PNG")
    mesh = textured_glb(tmp_path / "test.glb", png=png.getvalue())
    cost = readiness.measure(mesh)["cost"]
    assert cost["image_sizes"] == [[64, 32]] and cost["materials"] == 1 and cost["triangles"] == 12


def test_temporary_platform_run_keeps_default_choices_open(tmp_path):
    sheet.init(tmp_path, open_ended=True)
    preset = project.load(tmp_path, "quest")
    tools = agent.Tools(preset, agent.Report(), attempts=1, judge=False)
    text, _ = tools.call("choose_production", {"colour": "vertex", "reason": "Broad solid plates"})
    assert "10000 (preset quest)" in text and "vertex (agent)" in text
    default = project.load(tmp_path)
    assert set(production.pending(default.spec["production"])) == {"colour", "max_triangles"}
    production.choose(default, "Thin painted lines", colour="texture", max_triangles=1700)
    assert preset.spec["vertex_colors"] and default.spec["max_triangles"] == 1700
