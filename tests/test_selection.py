"""Selection provenance uses saved known artifacts; no Blender or model backend."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from glbkit import cube, write_cubes
from mesh_jig import agent, cli, mcp_server, project, selection


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "project"
    (root / "refs").mkdir(parents=True)
    (root / "jig.json").write_text(json.dumps({"name": "example", "model": {"glb": "model.glb"}}))
    (root / "brief.md").write_text("A synthetic reference for checkpoint tests.\n")
    for view in ["hero", "side"]:
        Image.new("RGB", (40, 30), "ivory").save(root / "refs" / f"{view}.png")
    for i, (score, colour) in enumerate([(0.9, "red"), (0.8, "blue"), (0.9, "green")], 1):
        p = root / "attempts" / f"a0{i}"
        (p / "out").mkdir(parents=True)
        (p / "renders").mkdir()
        (p / "model.py").write_text(f"# Synthetic saved artifact {i}\n")
        write_cubes(p / "out/model.glb", [cube((1, 1, 1))])
        renders = {}
        for view in ["hero", "side"]:
            Image.new("RGB", (40, 30), colour).save(p / "renders" / f"{view}.png")
            renders[view] = f"renders/{view}.png"
        (p / "eval.json").write_text(json.dumps({
            "name": p.name, "script": "model.py", "ok": True, "stage": "measured",
            "started": f"2026-01-01 00:00:0{i}", "numeric": {"numeric_score": score},
            "harness": {"commit": "abc1234", "dirty": False}, "renders": renders}))
    return project.load(root)


def choose(saved, name="round-03", **kw):
    return selection.create(saved, name, visual_attempt="a02", visual_source="self-review",
                            reason="The blue version has the intended shape.", **kw)


def test_three_roles_are_independent_and_ties_keep_earliest(saved):
    before = inventory(saved.root)
    rec = choose(saved)
    assert rec["selections"] == {"latest": "a03", "numeric_best": "a01", "visual": "a02"}
    assert rec["numeric_best_ties"] == ["a01", "a03"]
    assert rec["scope"]["successful_count"] == rec["scope"]["distinct_script_count"] == 3
    assert rec["visual_choice"]["source"] == "self-review"
    assert rec["attempts"]["a02"]["numeric_score"] == 0.8
    assert selection.verify(saved.root / "checkpoints/round-03") == rec
    assert {p: digest(saved.root / p) for p in before} == before, "source bytes were not changed"
    assert all(not Path(a["path"]).is_absolute() for a in rec["references"].values())


def test_omitted_visual_choice_is_unknown_even_with_a_numeric_winner(saved):
    rec = selection.create(saved, "unknown")
    assert rec["selections"]["numeric_best"] == "a01"
    assert rec["selections"]["visual"] is None and rec["visual_choice"] is None
    assert set(rec["attempts"]) == {"a01", "a03"}, "never silently copy numeric best into visual"


@pytest.mark.parametrize("kw", [
    {"visual_attempt": "a02"},
    {"visual_attempt": "a02", "visual_source": "numeric", "reason": "Highest score"},
    {"visual_attempt": "a02", "visual_source": "human", "reason": " "},
    {"visual_source": "human", "reason": "No choice supplied"},
    {"visual_attempt": "not-evaluated", "visual_source": "human", "reason": "Inspected"},
])
def test_visual_selection_requires_a_valid_explicit_choice_and_provenance(saved, kw):
    with pytest.raises(ValueError):
        selection.create(saved, "invalid", **kw)
    assert not (saved.root / "checkpoints/invalid").exists()


def test_failed_and_unfinished_attempts_do_not_become_latest(saved):
    failed = saved.attempts_dir / "a99"
    failed.mkdir()
    (failed / "eval.json").write_text(json.dumps({"ok": False, "stage": "build"}))
    (saved.attempts_dir / "a100").mkdir()
    rec = selection.create(saved, "finished-only")
    assert rec["scope"]["successful_count"] == 3 and rec["selections"]["latest"] == "a03"


def test_checkpoints_are_write_once_and_survive_source_changes(saved):
    choose(saved)
    frozen = saved.root / "checkpoints/round-03"
    original = inventory(frozen)
    (saved.attempts_dir / "a02/model.py").write_text("source changed after checkpoint\n")
    (saved.root / "refs/hero.png").unlink()
    assert selection.verify(frozen)["selections"]["visual"] == "a02"
    with pytest.raises(ValueError, match="already exists"):
        choose(saved)
    assert inventory(frozen) == original


@pytest.mark.parametrize("target", ["attempts/a02/models/model.glb", "comparison.png", "checkpoint.json"])
def test_tampered_frozen_artifacts_are_detected(saved, target):
    choose(saved)
    folder = saved.root / "checkpoints/round-03"
    (folder / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        selection.verify(folder)


def test_missing_renders_are_labelled_without_new_rendering(saved, monkeypatch):
    (saved.attempts_dir / "a02/renders/side.png").unlink()
    from mesh_jig import build, render
    def forbidden(*args, **kw):
        pytest.fail("checkpoint must not build or render")
    monkeypatch.setattr(build, "build_variants", forbidden)
    monkeypatch.setattr(render, "render_views", forbidden)
    rec = choose(saved, views=["side", "not-a-camera"])
    assert any("missing saved render side" in line for line in rec["warnings"])
    assert any("labelled placeholder" in line for line in rec["warnings"])
    assert "side" not in rec["attempts"]["a02"]["artifacts"]["renders"]
    assert (saved.root / "checkpoints/round-03/comparison.png").is_file()


@pytest.mark.parametrize("file", ["out/model.glb", "model.py"])
def test_missing_required_artifacts_fail_before_writing_a_checkpoint(saved, file):
    (saved.attempts_dir / "a02" / file).unlink()
    with pytest.raises(ValueError, match="missing"):
        choose(saved)
    assert not (saved.root / "checkpoints").exists()


def test_invalid_image_and_model_are_rejected(saved):
    file = saved.attempts_dir / "a02/renders/hero.png"
    original = file.read_bytes()
    file.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid saved image"):
        choose(saved)
    file.write_bytes(original)
    (saved.attempts_dir / "a02/out/model.glb").write_bytes(b"not a GLB")
    with pytest.raises(ValueError, match="invalid model"):
        choose(saved)


def test_before_after_uses_frozen_images_and_keeps_both_selections(saved):
    first = selection.create(saved, "round-01", through="a01", visual_attempt="a01",
                             visual_source="human", reason="Inspected the first result.")
    frozen = saved.root / "checkpoints/round-01"
    before = inventory(frozen)
    rec = choose(saved, before="round-01", views=["hero"])
    assert rec["before"] == {"name": "round-01", "checkpoint_sha256": digest(frozen / "checkpoint.json")}
    assert first["selections"]["visual"] == "a01" and rec["selections"]["visual"] == "a02"
    assert inventory(frozen) == before
    with Image.open(saved.root / "checkpoints/round-03/comparison.png") as image:
        assert image.height > image.width / 2 and image.info == {}


def test_before_after_refuses_changed_reference_or_contract(saved):
    selection.create(saved, "before")
    (saved.root / "brief.md").write_text("A changed task.\n")
    with pytest.raises(ValueError, match="different inputs"):
        choose(saved, before="before")


def test_review_evidence_is_copied_and_hashed(saved):
    evidence = saved.attempts_dir / "a02/review.md"
    evidence.write_text("The shape is correct. The red model has a weak side profile.\n")
    rec = choose(saved, review_file="attempts/a02/review.md")
    entry = rec["visual_choice"]["evidence"]
    assert entry["sha256"] == digest(evidence)
    assert (saved.root / "checkpoints/round-03" / entry["path"]).read_bytes() == evidence.read_bytes()


@pytest.mark.parametrize("name", ["../escape", "x/y", "CON", "nul.txt", "trailing."])
def test_output_names_cannot_escape_or_use_windows_devices(saved, name):
    with pytest.raises(ValueError):
        selection.create(saved, name)
    assert not (saved.root / "checkpoints").exists()


def test_render_path_cannot_read_another_attempt_or_external_file(saved):
    file = saved.attempts_dir / "a02/eval.json"
    rec = json.loads(file.read_text())
    rec["renders"]["hero"] = "../../refs/hero.png"
    file.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="escapes"):
        choose(saved)


def test_cli_and_mcp_use_the_same_format(saved, capsys):
    assert cli.main(["checkpoint", str(saved.root), "cli", "--visual-attempt", "a02",
                     "--visual-source", "human", "--reason", "Viewed the renders."]) == 0
    assert json.loads(capsys.readouterr().out)["selections"]["visual"] == "a02"
    reply = mcp_server.checkpoint(str(saved.root), "mcp", visual_attempt="a02", visual_source="human",
                                  reason="Viewed the renders.")
    assert json.loads(reply[0])["selections"]["visual"] == "a02"
    assert isinstance(reply[1], mcp_server.Picture)
    assert cli.main(["checkpoint", str(saved.root), "cli", "--verify"]) == 0
    assert cli.main(["checkpoint", str(saved.root), "cli", "--verify", "--visual-attempt", "a02"]) == 2


def test_api_agent_can_record_a_choice_without_spending_an_evaluation(saved):
    tools = agent.Tools(saved, agent.Report(), attempts=3, judge=False)
    before = tools.rep.evaluations
    text, pictures = tools.call("checkpoint", {"name": "api", "visual_attempt": "a02",
                              "visual_source": "self-review", "reason": "Inspected the saved renders."})
    assert json.loads(text)["selections"]["visual"] == "a02" and len(pictures) == 1
    assert tools.rep.evaluations == before
    assert tools.rep.best["name"] == "a01", "legacy numerical-best state is unchanged"


def test_api_cannot_use_review_evidence_outside_attempts(saved):
    tools = agent.Tools(saved, agent.Report(), attempts=3, judge=False)
    with pytest.raises(agent.Rejected):
        tools.call("checkpoint", {"name": "outside", "visual_attempt": "a02", "visual_source": "human",
                                  "reason": "Inspected", "review_file": "brief.md"})


def test_unscored_successes_do_not_get_an_invented_numeric_winner(saved):
    for file in saved.attempts_dir.glob("*/eval.json"):
        rec = json.loads(file.read_text())
        rec["numeric"] = {}
        file.write_text(json.dumps(rec))
    rec = choose(saved)
    assert rec["selections"] == {"latest": "a03", "numeric_best": None, "visual": "a02"}
    assert rec["numeric_best_ties"] == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.9"])
def test_invalid_scores_are_rejected_without_changing_legacy_scores(saved, score):
    file = saved.attempts_dir / "a02/eval.json"
    rec = json.loads(file.read_text())
    rec["numeric"]["numeric_score"] = score
    file.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="invalid numeric score"):
        choose(saved)


def test_harness_changes_are_reported_without_remeasuring_or_rewriting_scores(saved):
    file = saved.attempts_dir / "a02/eval.json"
    rec = json.loads(file.read_text())
    rec["harness"]["commit"] = "different"
    file.write_text(json.dumps(rec))
    snapshot = choose(saved)
    assert any("identities differ" in warning for warning in snapshot["warnings"])
    assert snapshot["attempts"]["a02"]["numeric_score"] == 0.8


def test_snapshot_inventory_and_declared_identities_are_verified(saved):
    choose(saved)
    root = saved.root / "checkpoints/round-03"
    (root / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="inventory changed"):
        selection.verify(root)
    (root / "extra.txt").unlink()
    file = root / "checkpoint.json"
    rec = json.loads(file.read_text())
    rec["comparison"]["sha256"] = "0" * 64
    file.write_text(json.dumps(rec))
    seal = json.loads((root / "seal.json").read_text())
    seal["files"]["checkpoint.json"] = digest(file)
    (root / "seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="identity disagrees"):
        selection.verify(root)
