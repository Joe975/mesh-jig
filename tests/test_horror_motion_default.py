"""Keep fresh horror runs on the published sprint/leap comparison task."""
import json
from pathlib import Path

from mesh_jig import rig


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "six-limbed-horror"


def test_horror_default_requires_current_comparison_clips():
    spec = json.loads((EXAMPLE / "rig.json").read_text(encoding="utf-8"))
    rig.contract(spec)
    assert set(spec["clips"]) == {"sprint", "leap"}
    assert spec["fps"] == 24
    assert spec["bones"] == {"required": ["root", "head"], "max": 48}
    assert spec["max_influences"] == 4
    assert spec["loop_mm_max"] == 2
    for name, seconds, loop, displacement in (("sprint", 1, True, 30), ("leap", 1.5, False, 150)):
        clip = spec["clips"][name]
        assert (clip["seconds"], clip["loop"], clip["min_displacement_mm"]) == (seconds, loop, displacement)
        assert clip["ground_drop_mm_max"] == 8
        assert clip["max_edge_ratio"] == 2.5
        assert clip["stretch_area_max"] == 0.03


def test_horror_motion_brief_explains_travel_support_and_shared_budget():
    brief = (EXAMPLE / "rig-brief.md").read_text(encoding="utf-8")
    brief = " ".join(brief.replace("\u2013", "-").split())
    for requirement in ("`sprint`", "`leap`", "zero net root travel", "Blender +Y",
                        "150 mm", "0.8-1.0 m", "root joint", "all six legs",
                        "three in-contract candidates", "twelve build submissions"):
        assert requirement in brief
    assert all(old not in brief for old in ("`idle`", "`crawl`", "gentle_turn"))
