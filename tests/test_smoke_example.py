"""The documented, key-free first success must build and self-match, without overwriting projects."""
from pathlib import Path
import json
import runpy

import pytest

from mesh_jig import cli, project, script_check

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "smoke"
prepare = runpy.run_path(str(EXAMPLE / "prepare.py"))["prepare"]


def test_smoke_script_obeys_checker():
    assert script_check.check_script((EXAMPLE / "model.py").read_text(encoding="utf-8")) == []


def test_existing_project_is_untouched(tmp_path):
    marker = tmp_path / "do-not-change.txt"
    marker.write_bytes(b"existing content")
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path)
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_bytes() == b"existing content"


@pytest.mark.blender
def test_documented_first_success_builds_and_measures(blender, tmp_path):
    root = tmp_path / "smoke"
    prepare(root)
    proj = project.load(root)
    assert proj.problems() == []
    assert set(proj.references) == {"front", "side", "top"}
    assert not proj.ledger.exists()
    script = root / "attempts" / "a00" / "model.py"
    assert cli.main(["eval", str(root), "--script", str(script)]) == 0
    rec = json.loads((script.parent / "eval.json").read_text(encoding="utf-8"))
    assert rec["ok"], rec.get("error")
    assert rec["numeric"]["ortho_iou"] > 0.99
    assert rec["numeric"]["resemblance"] > 0.99
    assert (script.parent / "out" / "box.glb").is_file()
    assert (script.parent / "compare" / "front_vs_reference.png").is_file()
    assert len([row for row in proj.ledger.read_text(encoding="utf-8").splitlines()
                if row.startswith("| a00 |")]) == 1
