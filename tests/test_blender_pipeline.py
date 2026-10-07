"""Real build/render/evaluate behavior against references generated from known geometry."""
import json
from pathlib import Path

import pytest

from mesh_jig import build, cli, evaluate, project, render

pytestmark = pytest.mark.blender
SCRIPT = """import bpy

def build():
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.active_object
    colour = obj.data.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='CORNER')
    for loop in colour.data:
        loop.color = (0.2, 0.4, 0.7, 1.0)
"""


def test_real_pipeline_matches_known_target_and_reports_a_script_failure(blender, tmp_path, capsys):
    spec = {"glb": "block.glb", "shots": [
        {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 2.0},
        {"name": "side", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0], "ortho": 2.0},
        {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "cam_up": [0, 0, 1], "ortho": 2.0},
    ]}
    truth = tmp_path / "truth.py"
    truth.write_text(SCRIPT)
    assert build.build_variants(truth, tmp_path / "truth", spec) == ""
    result = render.render_views(spec, tmp_path / "truth", tmp_path / "refs")
    assert not result["error"]
    config = {"model": spec, "criteria": {"views": {"front": 1, "side": 1, "top": 1}}}
    (tmp_path / "jig.json").write_text(json.dumps(config))
    proj = project.load(tmp_path)
    script = tmp_path / "attempts" / "correct" / "model.py"
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT)
    record = evaluate.evaluate(proj, script, script.parent, log=lambda *_: None)
    assert record["ok"], record.get("error")
    assert record["numeric"]["ortho_iou"] == 1
    assert record["numeric"]["resemblance"] == pytest.approx(1)
    script = tmp_path / "attempts" / "broken" / "model.py"
    script.parent.mkdir()
    script.write_text(SCRIPT.replace("obj.data.color_attributes", "missing.data.color_attributes"))
    assert cli.main(["eval", str(tmp_path), "--script", str(script)]) == 1
    output = capsys.readouterr().out
    assert "FAILED at build" in output and "NameError" in output
    assert "| broken | FAILED build |" in proj.ledger.read_text()
