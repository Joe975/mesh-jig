"""Protocol transfer through real loops and real confined file tools; no model."""
import copy
from dataclasses import asdict
import json

import pytest

from glbkit import write_glb
from mesh_jig import agent, rig_agent, sound_agent
from test_evaluate import make_project
from test_rig_tools import FIGURE, SPEC


@pytest.fixture(params=["numeric", "rig", "sound"])
def workflow(request, tmp_path, monkeypatch):
    kind = request.param
    if kind == "numeric":
        proj = make_project(tmp_path)
        root, script = proj.root, "ship.py"
        def run(chat, turns):
            return asdict(agent.run(proj, chat, attempts=1, max_turns=turns, out=root / "audit"))
    else:
        root, script = tmp_path, "rig.py" if kind == "rig" else "sounds.py"
        (root / "brief.md").write_text("A protocol test.")
        if kind == "rig":
            write_glb(root / "source.glb", FIGURE)
            (root / "rig.json").write_text(json.dumps(SPEC))
            module = rig_agent
        else:
            (root / "sound.json").write_text(json.dumps({"sounds": {"call": {"seconds": [.3, .8]}}}))
            # Spectrogram presentation is irrelevant to tool protocol/side effects.
            monkeypatch.setattr(sound_agent, "example_sheet", lambda _: None)
            module = sound_agent
        def run(chat, turns):
            return module.run(root, chat, builds=1, max_turns=turns, out=root / "audit")
    return kind, root, script, run


def tool(name, arguments, identifier="call"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def assert_history(messages):
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert isinstance(json.loads(call["function"]["arguments"] or "{}"), dict)


@pytest.mark.parametrize("mixed", [False, True])
def test_rejected_batch_preserves_audit_and_usage_without_partial_write(workflow, mixed):
    kind, root, script, run = workflow
    target = root / "attempts/a01" / script
    path = target.relative_to(root).as_posix()
    calls = [tool("write_file", '{"text":"unfinished', "broken")]
    if mixed:
        calls.insert(0, tool("write_file", json.dumps({"path": path, "text": "prefix"}), "prefix"))
    raw = {"content": "Captured response.", "tool_calls": calls,
           "reasoning_details": [{"type": "reasoning.text", "text": "PRIVATE_REASONING_MARKER"}]}
    frozen = copy.deepcopy(raw)
    turns = 0
    def chat(messages, schemas):
        nonlocal turns
        assert_history(messages)
        turns += 1
        if turns == 1:
            return raw, {"prompt_tokens": 100, "completion_tokens": 10, "cost": .001}
        if turns == 2:
            assert not target.exists()
            assert "No call in this batch was executed" in messages[-1]["content"]
            assert "PRIVATE_REASONING_MARKER" not in json.dumps(messages)
            response = tool("write_file", json.dumps({"path": path, "text": "complete"}), "write")
        else:
            assert target.read_text() == "complete"
            response = tool("read_file", json.dumps({"path": path}), "read")
        return {"tool_calls": [response]}, {"prompt_tokens": 100, "completion_tokens": 10, "cost": .001}
    report = run(chat, 3)
    assert raw == frozen
    assert report["turns"] == 3 and report["invalid_tool_batches"] == 1
    assert report["tool_calls"] == 3 + mixed and report["rejected_calls"] == 0
    assert report["prompt_tokens"] == 300 and report["completion_tokens"] == 30
    transcript = [json.loads(line) for line in (root / "audit/transcript.jsonl").read_text().splitlines()]
    captured = next(message for message in transcript if message.get("tool_calls"))
    assert captured["tool_calls"] == frozen["tool_calls"]
    if kind == "numeric":
        assert captured["reasoning_details"] == frozen["reasoning_details"]
    assert any("No call in this batch was executed" in str(m.get("content")) for m in transcript)


def test_persistent_bad_batches_use_existing_turn_limit(workflow):
    _, root, _, run = workflow
    turns = 0
    def chat(messages, schemas):
        nonlocal turns
        assert_history(messages)
        turns += 1
        return {"tool_calls": [tool("write_file", '{"text":')]}, {"completion_tokens": 10}
    report = run(chat, 3)
    assert turns == report["turns"] == report["invalid_tool_batches"] == 3
    assert report["completion_tokens"] == 30 and "turn cap" in report["error"]
    assert not list((root / "attempts").glob("*/*.py"))


def test_valid_batch_keeps_real_file_effects_and_existing_reasoning(workflow):
    kind, root, script, run = workflow
    path = "attempts/a01/" + script
    details = [{"type": "reasoning.text", "text": "VALID_REASONING_MARKER"}]
    turns = 0
    def chat(messages, schemas):
        nonlocal turns
        assert_history(messages)
        turns += 1
        if turns == 1:
            return {"tool_calls": [tool("write_file", json.dumps({"path": path, "text": "complete"}), "write"),
                                    tool("read_file", json.dumps({"path": path}), "read")],
                    "reasoning_details": details}, {}
        assert (root / path).read_text() == "complete"
        if kind == "numeric":
            assert any(m.get("reasoning_details") == details for m in messages)
        return {"content": "No further action."}, {}
    report = run(chat, 3)
    assert report["tool_calls"] == 2 and report.get("invalid_tool_batches", 0) == 0
    assert ("stopped without done" if kind == "numeric" else "turn cap") in report["error"]
