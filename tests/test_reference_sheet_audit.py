"""Observable scale checks and read-only guarantees for the reference skill helper."""
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys

from PIL import Image, ImageDraw
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "skill" / "mesh-jig-reference-sheet" / "scripts" / "audit_sheet.py"
audit = runpy.run_path(str(SCRIPT))["audit"]


def fixture_sheet():
    image = Image.new("RGB", (80, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 10, 13, 29), fill=(220, 100, 30))  # 10 x 20
    draw.rectangle((44, 10, 53, 29), fill=(220, 100, 30))
    draw.rectangle((0, 0, 79, 2), fill="black")  # neutral title, not subject
    config = {"panels": {"front": [0, 0, 40, 40], "rear": [40, 0, 80, 40]},
              "comparisons": [{"name": "height", "axis": "height", "views": ["front", "rear"]}]}
    return image, config


def test_bounds_ignore_background_and_neutral_title_and_are_repeatable():
    image, config = fixture_sheet()
    result = audit(image, config)
    assert result == audit(image, config)
    assert result["views"]["front"]["subject_bbox"] == [4, 10, 14, 30]
    assert result["views"]["rear"]["subject_bbox"] == [44, 10, 54, 30]
    assert result["within_tolerance"] is True
    assert result["comparisons"][0]["spread_pct"] == 0


def test_mismatching_projection_fails_and_cross_axis_comparison_works():
    image, config = fixture_sheet()
    draw = ImageDraw.Draw(image)
    draw.rectangle((44, 10, 53, 29), fill="white")
    draw.rectangle((44, 10, 63, 19), fill=(220, 100, 30))  # top footprint, 20 x 10
    assert audit(image, config)["within_tolerance"] is False
    config["comparisons"] = [{"name": "world depth", "dimensions": [["front", "height"], ["rear", "width"]]}]
    assert audit(image, config)["within_tolerance"] is True
    draw.rectangle((44, 20, 63, 24), fill=(220, 100, 30))
    config["comparisons"] = [{"axis": "height", "views": ["front", "rear"]}]
    result = audit(image, config)
    assert result["comparisons"][0]["spread_pct"] == 25
    config["tolerance_pct"] = 25
    assert audit(image, config)["within_tolerance"] is True
    config["tolerance_pct"] = 24.999
    assert audit(image, config)["within_tolerance"] is False


def test_alpha_measures_neutral_parts_but_not_invisible_colored_pixels():
    image = Image.new("RGBA", (20, 20), (220, 100, 30, 0))
    ImageDraw.Draw(image).rectangle((3, 4, 9, 11), fill=(80, 80, 80, 255))
    config = {"mode": "alpha", "panels": {"top": [0, 0, 20, 20]}}
    result = audit(image, config)
    assert result["views"]["top"]["subject_bbox"] == [3, 4, 10, 12]
    assert result["within_tolerance"] is None  # measured, no comparisons requested
    with pytest.raises(ValueError, match="no foreground"):
        audit(image, {**config, "mode": "chroma"})
    with pytest.raises(ValueError, match="transparent background"):
        audit(image.convert("RGB"), config)


@pytest.mark.parametrize("box", [[0, 0, 100, 40], [4, 4, 4, 10], [0, 0, 40.0, 40], [-1, 0, 40, 40]])
def test_bad_panels_rejected(box):
    image, config = fixture_sheet()
    config["panels"]["front"] = box
    with pytest.raises(ValueError):
        audit(image, config)


def test_empty_mask_unknown_dimension_and_clipping_flag():
    image, config = fixture_sheet()
    with pytest.raises(ValueError, match="no foreground"):
        audit(Image.new("RGB", image.size, "white"), config)
    invalid = copy.deepcopy(config)
    invalid["comparisons"][0]["axis"] = "depth"
    with pytest.raises(ValueError, match="unknown view or dimension"):
        audit(image, invalid)
    config["panels"]["front"] = [4, 10, 14, 30]
    assert audit(image, config)["views"]["front"]["touches_panel_edge"] is True


def test_cli_writes_report_returns_failure_and_preserves_inputs(tmp_path):
    image, config = fixture_sheet()
    ImageDraw.Draw(image).rectangle((44, 30, 53, 34), fill=(220, 100, 30))
    sheet = tmp_path / "sheet.png"
    settings = tmp_path / "config.json"
    output = tmp_path / "report.json"
    image.save(sheet)
    settings.write_text(json.dumps(config), encoding="utf-8")
    before = (sheet.read_bytes(), settings.read_bytes())
    command = [sys.executable, str(SCRIPT), "--sheet", str(sheet), "--config", str(settings), "--out"]
    result = subprocess.run(command + [str(output)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(output.read_text())["within_tolerance"] is False
    assert before == (sheet.read_bytes(), settings.read_bytes())
    for input_path in (sheet, settings):
        result = subprocess.run(command + [str(input_path)], capture_output=True, text=True)
        assert result.returncode == 2
        assert "must not overwrite" in result.stderr
        assert before == (sheet.read_bytes(), settings.read_bytes())
