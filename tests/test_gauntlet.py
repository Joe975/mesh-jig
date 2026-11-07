import json
from pathlib import Path

import pytest
from PIL import Image

from mesh_jig import agent, gauntlet, project


@pytest.fixture
def seed(tmp_path):
    root = tmp_path / "seed"
    root.mkdir()
    (root / "jig.json").write_text(json.dumps({"name": "test", "model": {}}), encoding="utf-8")
    (root / "brief.md").write_text("A green creature with a long tail.", encoding="utf-8")
    (root / "refs").mkdir()
    Image.new("RGB", (16, 16), "green").save(root / "refs" / "front.png")
    return project.load(root)


def tool(name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def fake_eval(proj, script, attempt, **kwargs):
    assert script.is_file()
    return {"ok": True, "name": attempt.name, "numeric": {"numeric_score": 0.4321},
            "compare": {"front": proj.references["front"]}, "gates": []}


def test_exactly_ten_rounds_feedback_reaches_next_implementor_but_reviewer_is_fresh(seed):
    impl_round, reviewed, writing = 0, 0, False

    def chat(messages, schemas):
        nonlocal impl_round, reviewed, writing
        if not schemas:
            reviewed += 1
            assert len(messages) == 2
            text = json.dumps(agent.strip_images(messages))
            assert "0.4321" not in text
            assert "ADVERSARIAL FEEDBACK" not in text
            assert "Round " not in text
            return {"role": "assistant", "content": f"Fix {reviewed}: lengthen the tail in side view."}, {}
        if not writing:
            impl_round += 1
            writing = True
            text = json.dumps(agent.strip_images(messages))
            assert "0.4321" not in text
            if impl_round > 1:
                assert f"Fix {impl_round - 1}: lengthen the tail" in text
                previous = seed.attempts_dir / f"a{impl_round - 1:02d}" / "model.py"
                assert previous.read_text() == "def build(): pass"
            return tool("write_file", path=f"attempts/a{impl_round:02d}/model.py", text="def build(): pass"), {}
        writing = False
        return tool("submit"), {"prompt_tokens": 10}

    result = gauntlet.run(seed, chat, evaluate_fn=fake_eval, log=lambda s: None)
    assert result["done"] and not result["error"]
    assert result["iterations_completed"] == 10
    assert impl_round == reviewed == 10
    assert len(result["calls"]) == 30
    assert len(list(seed.attempts_dir.glob("*/review.md"))) == 10
    assert "Fix 10" in (seed.attempts_dir / "a10" / "review.md").read_text()
    with pytest.raises(ValueError, match="fresh"):
        gauntlet.run(seed, chat)


def test_failed_build_repaired_without_spending_an_iteration(seed):
    replies = iter([tool("write_file", path="attempts/a01/model.py", text="def build(): pass"),
                    tool("submit"), tool("submit"), {"role": "assistant", "content": "Fix tail."}])
    evaluations = 0

    def evaluate_fn(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 1:
            return {"ok": False, "error": "bad geometry"}
        return fake_eval(*args, **kwargs)

    def chat(messages, schemas):
        if evaluations == 1:
            assert "bad geometry" in json.dumps(messages)
        return next(replies), {}

    result = gauntlet.run(seed, chat, iterations=1, evaluate_fn=evaluate_fn, log=lambda s: None)
    assert result["done"] and result["iterations_completed"] == 1 and evaluations == 2


def test_default_evaluation_ledger_keeps_only_the_success_after_a_failed_submit(seed, monkeypatch):
    """The default locked callback must preserve the conversation's success-only ledger semantics."""
    outcomes = iter([False, True])

    def locked_core(proj, script, attempt, **kwargs):
        ok = next(outcomes)
        return {"ok": ok, "name": attempt.name, "stage": "build",
                "error": "bad geometry" if not ok else "",
                "numeric": {"numeric_score": 0.4321} if ok else None,
                "compare": {"front": proj.references["front"]} if ok else {}, "gates": []}

    monkeypatch.setattr(gauntlet.evaluate, "_evaluate_locked", locked_core)
    builder_turn = 0

    def chat(messages, schemas):
        nonlocal builder_turn
        if not schemas:
            return {"content": "Looks good."}, {}
        builder_turn += 1
        if builder_turn == 1:
            return tool("write_file", path="attempts/a01/model.py", text="def build(): pass"), {}
        return tool("submit"), {}

    result = gauntlet.run(seed, chat, iterations=1, max_turns=4, log=lambda s: None)
    assert result["done"] and result["evaluations"] == 1
    rows = [line for line in seed.ledger.read_text(encoding="utf-8").splitlines()
            if line.startswith("| a01 |")]
    assert len(rows) == 1
    assert "| ok |" in rows[0]
    assert "FAILED" not in rows[0]


def user_text(messages):
    """The round's opening text and every tool result of a conversation, as the builder reads them."""
    return json.dumps(agent.strip_images(messages))


def test_last_turns_are_counted_down_and_a_script_never_submitted_is_submitted_by_the_loop(seed):
    seen, built = [], []

    def chat(messages, schemas):
        if not schemas:
            return {"content": "Fix tail."}, {}
        seen.append(user_text(messages))
        return tool("write_file", path="attempts/a01/model.py", text=f"def build(): return {len(seen)}"), {}

    def evaluate_fn(proj, script, attempt, **kwargs):
        built.append((script.read_text(), kwargs["note"]))
        return fake_eval(proj, script, attempt, **kwargs)

    result = gauntlet.run(seed, chat, iterations=1, max_turns=5, evaluate_fn=evaluate_fn, log=lambda s: None)
    assert len(seen) == 5
    assert "turns are left" not in seen[1]
    assert "3 of this round's 5 turns are left and no build has passed" in seen[2]
    assert "2 of this round's 5 turns are left" in seen[3]
    assert "1 of this round's 5 turns is left" in seen[4] and "Call submit now" in seen[4]
    assert built == [("def build(): return 5", "submitted by the loop when the round's turns ran out")]
    assert result["done"] and result["iterations_completed"] == 1
    assert result["recoveries"] == [{"round": 1, "kind": "loop-submitted", "ok": True}]


def test_loop_submits_a_script_changed_after_its_last_submission_but_not_one_already_built(seed):
    evaluations = []

    def evaluate_fn(proj, script, attempt, **kwargs):
        evaluations.append(script.read_text())
        if script.read_text() != "def build(): good":
            return {"ok": False, "error": "bad geometry"}
        return fake_eval(proj, script, attempt, **kwargs)

    calls = []

    def unchanged(messages, schemas):
        calls.append(1)
        return tool("write_file" if len(calls) == 1 else "submit", path="attempts/a01/model.py", text="bad"), {}

    result = gauntlet.run(seed, unchanged, iterations=1, max_turns=3, evaluate_fn=evaluate_fn, log=lambda s: None)
    assert not result["done"] and "2 conversations of 3 turns" in result["error"]
    # written once, then submitted on every turn left of both conversations; the loop built nothing it had seen fail
    assert evaluations == ["bad"] * 5
    assert [r["kind"] for r in result["recoveries"]] == ["restarted"]

    root = seed.root.with_name("second")
    root.mkdir()
    for name in ("jig.json", "brief.md"):
        (root / name).write_text((seed.root / name).read_text(), encoding="utf-8")
    (root / "refs").mkdir()
    Image.new("RGB", (16, 16), "green").save(root / "refs" / "front.png")
    other, evaluations[:] = project.load(root), []
    replies = iter([tool("write_file", path="attempts/a01/model.py", text="bad"), tool("submit"),
                    tool("write_file", path="attempts/a01/model.py", text="def build(): good"),
                    {"content": "Fix tail."}])
    result = gauntlet.run(other, lambda *a: (next(replies), {}), iterations=1, max_turns=3,
                          evaluate_fn=evaluate_fn, log=lambda s: None)
    assert evaluations == ["bad", "def build(): good"]
    assert result["done"] and result["recoveries"] == [{"round": 1, "kind": "loop-submitted", "ok": True}]


def test_round_with_no_passing_build_starts_once_more_as_a_new_conversation(seed):
    openings, turns = [], 0

    def evaluate_fn(proj, script, attempt, **kwargs):
        if "fixed" not in script.read_text():
            return {"ok": False, "error": "AttributeError: 'BMFace' object has no attribute 'center'"}
        return fake_eval(proj, script, attempt, **kwargs)

    def chat(messages, schemas):
        nonlocal turns
        if not schemas:
            return {"content": "Fix tail."}, {}
        turns += 1
        if len(messages) == 2:
            openings.append(user_text(messages))
        if turns == 1:
            return tool("write_file", path="attempts/a01/model.py", text="def build(): face.center"), {}
        if len(openings) == 2 and len(messages) == 2:
            return tool("write_file", path="attempts/a01/model.py", text="def build(): fixed"), {}
        return tool("submit"), {}

    result = gauntlet.run(seed, chat, iterations=1, max_turns=4, evaluate_fn=evaluate_fn, log=lambda s: None)
    assert result["done"] and result["iterations_completed"] == 1, result["error"]
    assert len(openings) == 2
    assert "started again" not in openings[0] and "Create your first complete model." in openings[0]
    restart = openings[1]
    assert "A green creature with a long tail." in restart and "Reference: front" in restart
    assert "Round 1/1, started again" in restart
    assert "'BMFace' object has no attribute 'center'" in restart
    assert "def build(): face.center" in restart
    assert result["turns"] == turns == 6      # four of the first conversation, two of the second: one total
    assert result["recoveries"] == [{"round": 1, "kind": "restarted", "ok": True}]


def test_run_ends_when_the_second_conversation_fails_too_and_its_turns_count(seed):
    turns = 0

    def chat(messages, schemas):
        nonlocal turns
        turns += 1
        return {"role": "assistant", "content": "thinking about it"}, {}

    result = gauntlet.run(seed, chat, iterations=2, max_turns=3, evaluate_fn=fake_eval, log=lambda s: None)
    assert not result["done"] and result["iterations_completed"] == 0
    assert "round 1 did not produce a successful build in 2 conversations of 3 turns" in result["error"]
    assert turns == result["turns"] == 6
    assert result["recoveries"] == [{"round": 1, "kind": "restarted", "ok": False}]

    capped = seed.root.with_name("capped")
    capped.mkdir()
    for name in ("jig.json", "brief.md"):
        (capped / name).write_text((seed.root / name).read_text(), encoding="utf-8")
    (capped / "refs").mkdir()
    Image.new("RGB", (16, 16), "green").save(capped / "refs" / "front.png")
    turns = 0
    result = gauntlet.run(project.load(capped), chat, iterations=1, max_turns=3, max_total_turns=4,
                          evaluate_fn=fake_eval, log=lambda s: None)
    assert turns == 4 and "total implementor turn cap" in result["error"]


def test_no_critique_means_incomplete_and_saved(seed):
    replies = iter([tool("write_file", path="attempts/a01/model.py", text="def build(): pass"),
                    tool("submit"), {"role": "assistant", "content": ""}])
    result = gauntlet.run(seed, lambda *args: (next(replies), {}), iterations=1,
                          evaluate_fn=fake_eval, log=lambda s: None)
    assert not result["done"] and result["iterations_completed"] == 0
    assert "no critique" in result["error"]
    assert json.loads((seed.root / "gauntlet" / "report.json").read_text())["error"] == result["error"]


@pytest.mark.parametrize("textured,texture_from,expects_plan", [
    (False, "plan", False), (True, "plan", True), (True, "script", False),
])
def test_only_atlas_plan_contracts_forward_a_plan(seed, textured, texture_from, expects_plan):
    seed.data["model"].update(textured=textured, texture_from=texture_from)
    replies = iter([tool("write_file", path="attempts/a01/model.py", text="def build(): pass"),
                    tool("write_file", path="attempts/a01/atlas_plan.json", text='{"ops": []}'),
                    tool("submit"), {"role": "assistant", "content": "Fix tail."}])

    def evaluate_fn(*args, **kwargs):
        assert (kwargs["paint_plan"] is not None) == expects_plan
        return fake_eval(*args, **kwargs)

    result = gauntlet.run(seed, lambda *args: (next(replies), {}), iterations=1,
                          evaluate_fn=evaluate_fn, log=lambda s: None)
    assert result["done"]


def test_files_confine_writes_and_block_metric_reads(seed):
    current = seed.attempts_dir / "a02"
    current.mkdir(parents=True)
    files = gauntlet.Files(seed, current)
    for path in ("jig.json", "../escape.py", "attempts/a01/model.py", "attempts/a02/nested/model.py"):
        with pytest.raises(agent.Rejected):
            files.call("write_file", {"path": path, "text": "bad"})
    for path in ("attempts/a01/eval.json", "attempts/ledger.md", "attempts/a01/gauntlet/report.json"):
        with pytest.raises(agent.Rejected):
            files.call("read_file", {"path": path})
    files.call("write_file", {"path": "attempts/a02/model.py", "text": "def build(): pass"})
    assert (current / "model.py").read_text() == "def build(): pass"


def test_comparison_identical_fresh_seeds_and_same_model_budget(seed, tmp_path):
    (seed.attempts_dir / "old").mkdir(parents=True)
    (seed.attempts_dir / "old" / "model.py").write_text("old secret implementation")
    seen = []

    def gauntlet_fn(proj, chat, **kwargs):
        assert not proj.attempts_dir.exists()
        seen.append((kwargs["model"], kwargs["iterations"]))
        return {"done": True, "iterations_completed": kwargs["iterations"]}

    def control_fn(proj, chat, **kwargs):
        assert not proj.attempts_dir.exists()
        seen.append((kwargs["model"], kwargs["attempts"]))
        assert kwargs["max_turns"] == 240
        return agent.Report(done=True, evaluations=kwargs["attempts"])

    out = tmp_path / "pair"
    result = gauntlet.comparison(seed, out, object(), gauntlet_fn=gauntlet_fn, control_fn=control_fn)
    assert result["status"] == "complete"
    assert result["max_tokens"] == 65536
    assert result["max_turns_per_iteration"] == 24
    assert seen == [(gauntlet.MODEL, 10)] * 2
    for relative, digest in result["seed_sha256"].items():
        assert (out / "gauntlet" / relative).read_bytes() == (out / "mesh-jig" / relative).read_bytes()
    with pytest.raises(ValueError, match="exists"):
        gauntlet.comparison(seed, out, object())


def test_comparison_records_arm_failure_without_losing_control(seed, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("API unavailable")

    result = gauntlet.comparison(seed, tmp_path / "pair", object(), gauntlet_fn=fail,
                                control_fn=lambda *a, **kw: agent.Report(evaluations=10))
    assert result["status"] == "incomplete"
    assert result["arms"]["gauntlet"]["error"] == "API unavailable"
    assert result["arms"]["mesh-jig"]["evaluations"] == 10


def test_backend_uses_only_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MESH_JIG_AGENT_KEY", "different-provider")
    api = gauntlet.backend()
    assert api.key == "anthropic-test" and api.model == "claude-haiku-5-5"
    assert api.url == "https://api.anthropic.com/v1"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(gauntlet.llm.LLMError, match="ANTHROPIC_API_KEY"):
        gauntlet.backend()


def test_visual_brief_preserves_fixed_part_identities_and_fit_contract(seed):
    contract = {"parts": {"skin": {"nodes": ["SkinSurface"]}, "vest": {"nodes": ["VestMesh"]}},
                "clearance": [{"name": "vest-fit", "inner": "skin", "outer": "vest"}]}
    seed.data["review"] = contract
    text = gauntlet.visual_brief(seed)
    assert json.dumps(contract) in text
    assert "preserve exact declared node names" in text
    assert "SkinSurface" in text and "VestMesh" in text
    assert "numeric_score" not in text


def test_a_builder_for_a_target_is_told_its_limits_and_hears_how_its_previous_mesh_read_against_them(seed):
    reading = {"target": "quest", "passed": 1, "checks": [
        {"name": "triangles", "value": 900, "limit": "at most 10000", "passed": True, "what": "the budget", "hard": True},
        {"name": "buried", "value": 0.31, "limit": "at most 15%", "passed": False, "what": "triangles no outside view can see",
         "hard": False}],
        "cost": {"triangles": 900, "vertices": 500, "positions": 480, "mesh_nodes": 3, "primitives": 3, "materials": 0,
                 "bytes": 20480},
        "surface": {"pieces": 3, "pieces_90": 2, "largest_piece": 0.7, "buried": 0.31, "open_edges": 0.0, "slivers": 0.0,
                    "symmetry": 1.0}, "nodes": [], "uv": None, "max_triangles": 10000}

    def run(proj, with_reading):
        openings, reviews, state = [], [], {"writes": 0, "submissions": 0}

        def chat(messages, schemas):
            text = json.dumps(agent.strip_images(messages))
            if not schemas:
                reviews.append(text)
                return {"content": "Lengthen the tail."}, {}
            if len(messages) == 2:
                openings.append(text)
            if state["writes"] == state["submissions"]:
                state["writes"] += 1
                return tool("write_file", path=f"attempts/a{state['writes']:02d}/model.py", text="def build(): pass"), {}
            state["submissions"] += 1
            return tool("submit"), {}

        def evaluate_fn(proj, script, attempt, **kwargs):
            return {**fake_eval(proj, script, attempt), **({"readiness": reading} if with_reading else {})}

        result = gauntlet.run(proj, chat, iterations=2, evaluate_fn=evaluate_fn, log=lambda s: None)
        assert result["done"], result["error"]
        return openings, reviews

    plain, _ = run(seed, with_reading=False)
    assert not any("TARGET" in o or "MESH READINESS" in o for o in plain)

    import shutil
    aimed_root = seed.root.parent / "aimed"
    shutil.copytree(seed.root, aimed_root, ignore=shutil.ignore_patterns("attempts", "gauntlet"))
    openings, reviews = run(project.load(aimed_root, "quest"), with_reading=True)
    assert all("TARGET: quest" in o and "At most 10000 triangles" in o for o in openings)
    assert "MESH READINESS" not in openings[0]                          # round 1 has no build to read
    assert "MESH READINESS OF PREVIOUS BUILD" in openings[1]
    assert "1 of 2 hold; NOT: buried (31%, wants at most 15%" in openings[1]
    assert "mesh: 900 of 10000 triangles (9%)" in openings[1]
    assert not any("MESH READINESS" in r or "TARGET" in r for r in reviews)      # the reviewer judges the pictures


@pytest.mark.parametrize("checks", ["none", "builder", "both"])
def test_check_visibility_is_specific_to_role_and_round(seed, checks):
    writes, submissions, reviews = 0, 0, 0

    def chat(messages, schemas):
        nonlocal writes, submissions, reviews
        text = json.dumps(agent.strip_images(messages))
        if not schemas:
            reviews += 1
            assert ("JIG CHECKS FOR CURRENT BUILD" in text) == (checks == "both")
            assert ("0.4321" in text) == (checks == "both")
            if checks == "both":
                assert "You see no numeric scores" not in text
            return {"content": "Lengthen the tail in side view."}, {"prompt_tokens": 10, "completion_tokens": 5}
        assert ("JIG CHECKS FOR PREVIOUS BUILD" in text) == (submissions > 0 and checks != "none")
        if writes == submissions:
            writes += 1
            return tool("write_file", path=f"attempts/a{writes:02d}/model.py", text="def build(): pass"), {}
        submissions += 1
        return tool("submit"), {}

    result = gauntlet.run(seed, chat, iterations=2, checks=checks, evaluate_fn=fake_eval, log=lambda s: None)
    assert result["done"], result["error"]
    assert reviews == 2
    assert result["rounds"][1]["calls_start"] == 3
    assert result["rounds"][1]["calls_end"] == 6
    assert result["rounds"][1]["elapsed_s"] >= result["rounds"][0]["elapsed_s"]


def test_checkpoint_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"old": true}')
    original = Path.replace
    calls = []

    def replace(path, destination):
        calls.append(destination)
        if len(calls) < 3:
            assert json.loads(target.read_text()) == {"old": True}
            raise PermissionError("temporary reader lock")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(gauntlet.time, "sleep", lambda s: None)
    gauntlet.write_json(target, {"new": True})
    assert len(calls) == 3
    assert json.loads(target.read_text()) == {"new": True}


def test_checkpoint_permanent_lock_is_bounded_and_keeps_previous_result(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"old": true}')
    calls = []

    def replace(*args):
        calls.append(1)
        raise PermissionError("permanent lock")

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(gauntlet.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        gauntlet.write_json(target, {"new": True})
    assert len(calls) == 6
    assert json.loads(target.read_text()) == {"old": True}


def test_a_budget_stop_builds_the_script_left_on_disk_and_each_call_gets_the_time_left(seed, monkeypatch):
    clock, given, built = [0.0], [], []
    monkeypatch.setattr(gauntlet.time, "monotonic", lambda: clock[0])

    def chat(messages, schemas, left_s):
        given.append(left_s)
        clock[0] += 40          # each call thinks for 40 s of a 100 s budget, and never submits
        return tool("write_file", path="attempts/a01/model.py", text=f"def build(): return {len(given)}"), {}

    def evaluate_fn(proj, script, attempt, **kwargs):
        built.append((script.read_text(), kwargs["note"]))
        return fake_eval(proj, script, attempt, **kwargs)

    result = gauntlet.run(seed, chat, iterations=2, max_turns=24, evaluate_fn=evaluate_fn, max_wall=100,
                          log=lambda s: None)
    assert given == [100, 60, 20]
    assert built == [("def build(): return 3", "submitted by the loop when the time budget ran out")]
    assert result["evaluations"] == 1 and not result["done"] and "time budget" in result["error"]
    assert result["recoveries"] == [{"round": 1, "kind": "budget-submitted", "ok": True}]


def test_a_budget_stop_does_not_build_a_script_unchanged_since_the_round_began(seed, monkeypatch):
    clock, built = [0.0], []
    monkeypatch.setattr(gauntlet.time, "monotonic", lambda: clock[0])
    replies = iter([tool("write_file", path="attempts/a01/model.py", text="def build(): pass"), tool("submit")])

    def chat(messages, schemas, left_s):
        if not schemas:
            return {"content": "Fix tail."}, {}
        clock[0] += 30
        return next(replies, tool("read_file", path="attempts/a02/model.py")), {}

    def evaluate_fn(proj, script, attempt, **kwargs):
        built.append(attempt.name)
        return fake_eval(proj, script, attempt, **kwargs)

    result = gauntlet.run(seed, chat, iterations=2, evaluate_fn=evaluate_fn, max_wall=100, log=lambda s: None)
    assert built == ["a01"] and result["evaluations"] == 1
    assert result["recoveries"] == []


def test_a_call_cut_off_by_the_time_budget_is_a_budget_stop_and_the_script_on_disk_is_built(seed, monkeypatch):
    clock, built = [0.0], []
    monkeypatch.setattr(gauntlet.time, "monotonic", lambda: clock[0])

    def chat(messages, schemas, left_s):
        if len(built) == 0 and clock[0] == 0:
            clock[0] = 50
            return tool("write_file", path="attempts/a01/model.py", text="def build(): pass"), {}
        clock[0] += left_s           # the transport gives up when the run's time is gone
        raise TimeoutError("timed out")

    def evaluate_fn(proj, script, attempt, **kwargs):
        built.append(kwargs["note"])
        return fake_eval(proj, script, attempt, **kwargs)

    result = gauntlet.run(seed, chat, iterations=1, evaluate_fn=evaluate_fn, max_wall=100, log=lambda s: None)
    assert built == ["submitted by the loop when the time budget ran out"]
    assert result["evaluations"] == 1 and "time budget" in result["error"]
    assert result["recoveries"] == [{"round": 1, "kind": "budget-submitted", "ok": True}]


def test_a_call_that_fails_with_time_left_is_not_a_budget_stop(seed):
    def chat(messages, schemas, left_s):
        raise TimeoutError("server went away")

    result = gauntlet.run(seed, chat, iterations=1, evaluate_fn=fake_eval, max_wall=100, log=lambda s: None)
    assert result["error"] == "server went away" and result["recoveries"] == []


def test_a_note_left_in_the_project_reaches_the_builder_once_and_never_the_reviewer(seed):
    """steer.md: the person running the loop steers the builder between its turns."""
    (seed.root / gauntlet.STEER).write_text("Make the head bigger.", encoding="utf-8")
    builder_saw, reviewer_saw, turn = [], [], 0

    def chat(messages, schemas):
        nonlocal turn
        if not schemas:
            reviewer_saw.append(user_text(messages))
            return {"content": "Fix tail."}, {}
        turn += 1
        builder_saw.append(user_text(messages))
        if turn == 3:  # left while round 2's builder is working: its next turn reads it
            (seed.root / gauntlet.STEER).write_text("Now the wings.", encoding="utf-8")
        if turn % 2:
            return tool("write_file", path=f"attempts/a{(turn + 1) // 2:02d}/model.py", text="def build(): pass"), {}
        return tool("submit"), {}

    result = gauntlet.run(seed, chat, iterations=2, evaluate_fn=fake_eval, log=lambda s: None)
    assert result["done"]
    assert "Make the head bigger." in builder_saw[0] and "NOTE FROM THE PERSON" in builder_saw[0]
    assert "Make the head bigger." not in builder_saw[2]
    assert "Now the wings." not in builder_saw[2] and "Now the wings." in builder_saw[3]
    assert not any("NOTE FROM THE PERSON" in text for text in reviewer_saw)
    assert not (seed.root / gauntlet.STEER).exists()
    assert (seed.attempts_dir / "a01" / gauntlet.STEER).read_text(encoding="utf-8").strip() == "Make the head bigger."
    assert (seed.attempts_dir / "a02" / gauntlet.STEER).read_text(encoding="utf-8").strip() == "Now the wings."
    assert result["steers"] == [{"round": 1, "turn": 0, "chars": 21}, {"round": 2, "turn": 1, "chars": 14}]


def test_the_builder_reads_references_ruled_in_blender_metres_and_the_reviewer_plain_ones(tmp_path):
    root = tmp_path / "standing"
    (root / "refs").mkdir(parents=True)
    (root / "jig.json").write_text(json.dumps({"name": "s", "model": {
        "height_m": 1.0, "shots": [{"name": "front", "cam_pos": [0, 0, -3]}, {"name": "top", "cam_pos": [0, 3, 0]}]}}))
    (root / "brief.md").write_text("A standing figure.", encoding="utf-8")
    for view in ("front", "top"):
        im = Image.new("RGB", (120, 140), (255, 255, 255))
        im.paste((150, 55, 45), (40, 20, 80, 120))
        im.save(root / "refs" / f"{view}.png")
    proj = project.load(root)
    builder_saw, reviewer_saw = [], []

    def chat(messages, schemas):
        if not schemas:
            reviewer_saw.append(user_text(messages))
            return {"content": "Fix tail."}, {}
        builder_saw.append(user_text(messages))
        if len(builder_saw) == 1:
            return tool("write_file", path="attempts/a01/model.py", text="def build(): pass"), {}
        return tool("submit"), {}

    result = gauntlet.run(proj, chat, iterations=1, evaluate_fn=fake_eval, log=lambda s: None)
    assert result["done"]
    assert json.dumps(gauntlet.RULED_CAPTION + ": front")[1:-1] in builder_saw[0], "a view looking across is ruled"
    assert "Reference: top" in builder_saw[0] and "Reference: front" not in builder_saw[0]
    assert "ruled" not in reviewer_saw[0] and "Reference: front" in reviewer_saw[0]


def test_the_builder_hears_which_objects_of_its_last_build_no_camera_sees(seed):
    hidden = [{"node": "eyes", "views": {"front": 0.0, "side": 0.03}, "behind": "head"}]
    builder_saw = []

    def evaluate_fn(proj, script, attempt, **kwargs):
        return {**fake_eval(proj, script, attempt), "hidden": hidden}

    def chat(messages, schemas):
        if not schemas:
            return {"content": "The eyes are missing."}, {}
        builder_saw.append(user_text(messages))
        n = len(builder_saw)
        return (tool("write_file", path=f"attempts/a{(n + 1) // 2:02d}/model.py", text="def build(): pass")
                if n % 2 else tool("submit")), {}

    assert gauntlet.run(seed, chat, iterations=2, evaluate_fn=evaluate_fn, log=lambda s: None)["done"]
    assert "NO CAMERA SEES MUCH OF" not in builder_saw[0]
    assert "eyes: at most 3% of it shows in any view (front 0%, side 3%); it is inside or behind head" in builder_saw[2]


def test_the_builder_sees_its_last_build_on_the_references_metre_lines_in_place_of_the_plain_pair(seed):
    builder_saw = []

    def evaluate_fn(proj, script, attempt, **kwargs):
        return {**fake_eval(proj, script, attempt), "ruled": {"front": proj.references["front"]},
                "closeups": {"face": proj.references["front"]}}

    reviewer_saw = []

    def chat(messages, schemas):
        if not schemas:
            reviewer_saw.append(user_text(messages))
            return {"content": "Lengthen the tail."}, {}
        builder_saw.append(user_text(messages))
        n = len(builder_saw)
        return (tool("write_file", path=f"attempts/a{(n + 1) // 2:02d}/model.py", text="def build(): pass")
                if n % 2 else tool("submit")), {}

    assert gauntlet.run(seed, chat, iterations=2, evaluate_fn=evaluate_fn, log=lambda s: None)["done"]
    assert json.dumps(gauntlet.RULED_PAIR_CAPTION + ": front")[1:-1] in builder_saw[2]
    assert "at ONE scale on the same fixed lines" in reviewer_saw[0], "the reviewer sees the same pairs"
    assert "a marked region enlarged: reference left, yours right: face" in builder_saw[2], "and the close-ups"
    assert "Previous build: reference left, candidate right: front" not in builder_saw[2]


def test_the_builder_hears_sampled_separation_without_certified_detachment(seed):
    floating = [{"node": "tail", "gap_m": 0.038, "nearest": "body"}]
    builder_saw = []

    def evaluate_fn(proj, script, attempt, **kwargs):
        return {**fake_eval(proj, script, attempt), "floating": floating}

    def chat(messages, schemas):
        if not schemas:
            return {"content": "Attach the tail."}, {}
        builder_saw.append(user_text(messages))
        n = len(builder_saw)
        return (tool("write_file", path=f"attempts/a{(n + 1) // 2:02d}/model.py", text="def build(): pass")
                if n % 2 else tool("submit")), {}

    assert gauntlet.run(seed, chat, iterations=2, evaluate_fn=evaluate_fn, log=lambda s: None)["done"]
    assert "SAMPLED SEPARATION" not in builder_saw[0]
    assert "TOUCH NO OTHER OBJECT" not in builder_saw[2]
    assert "SAMPLED SEPARATION" in builder_saw[2]
    assert "tail: 3.8 cm sampled point separation from body" in builder_saw[2]
    assert "Inspect actual contact before moving geometry" in builder_saw[2]
    assert "if detached and intended to attach" in builder_saw[2]


def test_keep_better_drops_a_build_judged_worse_and_starts_again_from_the_one_it_lost_to(seed):
    """r11 got worse after its first round and every later round built on the worse one."""
    builder_saw, reviews, compared, copied, turn = [], 0, [], [], 0

    def chat(messages, schemas):
        nonlocal reviews, turn
        if not schemas and messages[0]["content"] == gauntlet.COMPARER:
            text = user_text(messages)
            compared.append(text)
            # round 2: the new build is A (even round) and loses; round 3: the new build is B and wins
            return {"content": "B\nB keeps the reference's proportions."}, {}
        if not schemas:
            reviews += 1
            return {"content": f"Review {reviews}."}, {}
        turn += 1
        builder_saw.append(user_text(messages))
        if turn == 5:   # round 3's first turn: what was copied in for it
            copied.append((seed.attempts_dir / "a03" / "model.py").read_text())
        if turn % 2:
            name = f"a{(turn + 1) // 2:02d}"
            return tool("write_file", path=f"attempts/{name}/model.py", text=f"def build(): return {turn}"), {}
        return tool("submit"), {}

    def evaluate_fn(proj, script, attempt, **kwargs):
        return {**fake_eval(proj, script, attempt), "closeups": {"face": proj.references["front"]}}

    result = gauntlet.run(seed, chat, iterations=3, evaluate_fn=evaluate_fn, keep_better=True, log=lambda s: None)
    assert result["done"] and result["evaluations"] == 3 and reviews == 2, "the losing build is not reviewed"
    assert [c["kept"] for c in result["comparisons"]] == ["a01", "a03"]
    assert result["kept"] == {"name": "a03", "numeric_score": 0.4321}, "the build the comparisons kept, last"
    assert [r.get("kept", True) for r in result["rounds"]] == [True, False, True]
    # round 3 starts from round 1's script, with round 1's review and a note saying why
    assert copied == ["def build(): return 1"], "round 3 began from round 1's script, not round 2's"
    assert "judged further from the reference than a01" in builder_saw[4] and "Review 1." in builder_saw[4]
    assert "Review 1." not in builder_saw[4].split("judged further")[1]
    assert (seed.attempts_dir / "a02" / "comparison.md").read_text().startswith("B")
    assert not (seed.attempts_dir / "a02" / "review.md").exists()
    assert "Candidate A" in compared[0] and "Candidate B" in compared[0]
    assert "Candidate A, a marked region enlarged" in compared[0] and "Candidate B, a marked region enlarged" in compared[0]


def test_the_builder_hears_which_objects_of_its_last_build_show_their_backs(seed):
    flipped = [{"node": "body", "views": {"front": 0.55, "side": 0.4}}]
    builder_saw = []

    def evaluate_fn(proj, script, attempt, **kwargs):
        return {**fake_eval(proj, script, attempt), "flipped": flipped}

    def chat(messages, schemas):
        if not schemas:
            return {"content": "The body is patchy."}, {}
        builder_saw.append(user_text(messages))
        n = len(builder_saw)
        return (tool("write_file", path=f"attempts/a{(n + 1) // 2:02d}/model.py", text="def build(): pass")
                if n % 2 else tool("submit")), {}

    assert gauntlet.run(seed, chat, iterations=2, evaluate_fn=evaluate_fn, log=lambda s: None)["done"]
    assert "WOUND THE WRONG WAY" not in builder_saw[0]
    assert "body: the camera sees the back of up to 55% of it (front 55%, side 40%)" in builder_saw[2]
