"""Observation reporting changes no score or frozen evaluation files."""
import copy
import json
import shutil
import subprocess

import pytest

from mesh_jig import cli, evaluate, project, review, streak, viewer
from test_evaluate import fake_build, make_project, run


def configured(tmp_path):
    proj = make_project(tmp_path)
    data = proj.data
    data["criteria"] = {"score": {"regions": 1}, "review": {"regions": [
        {"name": "surface", "view": "top", "box": [0, 0, 1, 1]},
        {"name": "width", "view": "top", "box": [0, 0, 1, 1], "min_iou": 1},
    ]}}
    (proj.root / "jig.json").write_text(json.dumps(data))
    return project.load(proj.root)


def test_observations_and_failed_gates_are_separate_without_changing_score(tmp_path):
    proj = configured(tmp_path)
    rec = run(proj, tmp_path, build_fn=fake_build(span=.07))
    assert rec["ok"]
    observed, gated = rec["review"]["checks"]
    assert observed["passed"] is None and review.is_observation(observed)
    assert gated["passed"] is False and not review.is_observation(gated)
    assert [g["name"] for g in rec["gates"]] == ["width"]
    assert evaluate.gate_tally(rec) == "0/1 (width)"
    said = evaluate.feedback(rec)
    assert "1 of 1 gate FAILED (width)" in said
    assert "observation surface:" in said and "no acceptance limit" in said
    # Adding an acceptance limit changes gate classification, never region measurements or score.
    data = proj.data
    data["criteria"]["review"]["regions"][0]["min_iou"] = 0
    (proj.root / "jig.json").write_text(json.dumps(data))
    other = run(project.load(proj.root), tmp_path, name="a02", build_fn=fake_build(span=.07))
    assert other["numeric"] == rec["numeric"]
    assert not review.is_observation(other["review"]["checks"][0])  # zero is an explicit limit


def test_legacy_observations_are_not_counted_by_ledger_streak_or_viewer(tmp_path):
    proj = configured(tmp_path)
    rec = run(proj, tmp_path, build_fn=fake_build(span=.07))
    legacy = copy.deepcopy(rec)
    c = legacy["review"]["checks"][0]
    c.pop("observation")
    c["passed"] = True
    legacy["gates"].insert(0, {"name": "surface", "passed": True, "message": str(c)})
    attempt = proj.attempts_dir / "a01"
    saved = attempt / "eval.json"
    saved.write_text(json.dumps(legacy))
    before = saved.read_bytes()
    assert evaluate.failed_gates(legacy) == ["width"]
    assert evaluate.gate_tally(legacy) == "0/1 (width)"
    assert streak.attempts(proj.root)[0][0]["gates"] == 1
    # Use the public index, which is also what the browser receives.
    index = viewer.build_index([proj.root])
    item = index["references"][0]["projects"][0]["attempts"][0]
    assert len(item["gates"]) == 1 and item["gates"][0]["name"] == "width"
    assert saved.read_bytes() == before
    assert legacy["numeric"] == rec["numeric"]


def test_missing_region_without_limits_is_an_observation():
    checks = review.regions({}, {}, [{"name": "face", "view": "front", "box": [0, 0, 1, 1]}], ())
    assert checks[0]["passed"] is None and review.is_observation(checks[0])
    assert "missing render" in checks[0]["error"]


def test_doctor_warns_when_regions_have_no_acceptance_limits(tmp_path, capsys):
    proj = configured(tmp_path)
    data = proj.data
    data["criteria"]["review"]["regions"].pop()
    (proj.root / "jig.json").write_text(json.dumps(data))
    cli.main(["doctor", str(proj.root)])
    said = capsys.readouterr().out
    assert "observations, not gates: surface" in said
    assert "no explicit quality acceptance limits" in said


def test_browser_labels_legacy_and_new_observations_without_counting_them_as_gates(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    # Exercise the shipped browser functions with a minimal DOM/fetch fixture.
    code = viewer.PAGE.split("function isObservation(c)", 1)[1].split("function showColours", 1)[0]
    code = "function isObservation(c)" + code
    # Stop after showRecord; intervening helpers, if any, need no evaluation.
    prelude = """
const output = {innerHTML: ''}, state = {pick: 'a'};
const $ = () => output, f = x => String(x), esc = x => String(x), file = () => 'fixture';
const document = {querySelectorAll: () => []};
const checks = [
 {name: 'legacy', view: 'front', iou: 0.3, passed: true},
 {name: 'new', view: 'front', iou: 0.4, observation: true, passed: null},
 {name: 'required', view: 'front', iou: 0.4, min_iou: 0.8, passed: false}
];
const fetch = async () => ({json: async () => ({review: {checks}})});
"""
    script = tmp_path / "checks.mjs"
    script.write_text(prelude + code + "\nawait showRecord({id:'p'}, {eval:'fixture'}, 'a');\nconsole.log(output.innerHTML);", encoding="utf-8")
    result = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "0 of 1 acceptance gates met; 2 observations" in result.stdout
    assert result.stdout.count('>observation</span>') == 2
    assert result.stdout.count('>not met</span>') == 1
