"""Placement advice must respect the calibrated standing coordinate frame."""
import numpy as np
import pytest

from mesh_jig import brief, project, standing
from test_character import calibrated, model_box


@pytest.mark.parametrize("origin", [None, [.02, -.03, -.5]])
@pytest.mark.parametrize("location", ["legacy", "criteria"])
@pytest.mark.parametrize("craft", [False, True])
def test_standing_brief_preserves_matching_model_placement(tmp_path, origin, location, craft):
    refs, spec, config = calibrated(tmp_path)
    if origin is not None:
        config["origin_m"] = origin
    ground = config.get("origin_m", [0, 0, 0])
    if craft:
        spec["forward"] = "-Z"
    data = {"model": spec, "refs": "."}
    if location == "legacy":
        spec["profile"] = config
    else:
        data["criteria"] = {"outline": config}
    proj = project.Project(tmp_path, data)
    assert not proj.problems()
    reference = proj.reference_profile()
    tris = model_box() + np.asarray(ground) * 1000
    assert standing.compare(reference, standing.from_mesh(tris, reference))["mae_mm"] == 0

    # Following the old instruction changes a known-correct fixed-frame shape.
    points = tris.reshape(-1, 3)
    centered = tris - (points.min(axis=0) + points.max(axis=0)) / 2
    wrong = standing.compare(reference, standing.from_mesh(centered, reference))
    if origin is None:
        assert wrong["missing_stations"] == [f"{view}:head" for view in refs]
        assert wrong["mae_mm"] is None
    else:
        assert wrong["missing_stations"] == []
        assert wrong["mae_mm"] == 70

    said = brief.model_facts(proj.spec)
    assert "Centre the model on the origin." not in said
    assert f"fixed ground origin {ground} m in Blender axes" in said
    assert "Do not recenter the model's bounding box." in said
    assert said in brief.text(proj)


@pytest.mark.parametrize("spec", [{}, {"height_m": 1}, {"forward": "-Z", "length_m": .08}])
def test_no_standing_calibration_does_not_invent_a_ground_origin(spec):
    said = brief.model_facts(spec)
    assert "Centre the model on the origin." in said
    assert "fixed ground origin" not in said


def test_standing_placement_transfers_to_separate_mechanical_columns(tmp_path):
    refs, spec, config = calibrated(tmp_path)
    config["origin_m"] = [.3, -.2, .1]
    for station, name in zip(config["stations"], ["base", "shaft", "cap"]):
        station["name"] = name
    reference = standing.measure(refs, spec, config)
    tris = np.concatenate([
        model_box((100, 200, 1000), (-50, 50, 500)),
        model_box((100, 200, 1000), (250, 50, 500)),
    ]) + np.asarray(config["origin_m"]) * 1000
    assert standing.compare(reference, standing.from_mesh(tris, reference))["mae_mm"] == 0
    said = brief.model_facts({**spec, "profile": config})
    assert "fixed ground origin [0.3, -0.2, 0.1] m in Blender axes" in said
    assert "Centre the model on the origin." not in said
