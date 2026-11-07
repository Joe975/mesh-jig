"""Observed transport endings survive real nonvisual loops without policy changes."""
import json

import pytest

from glbkit import write_glb
from mesh_jig import rig_agent
from test_tool_protocol_transfer import workflow, tool
from test_rig_tools import FIGURE, SPEC, walk


def evidence(report, kind):
    if kind == "numeric":
        return [call.get("mesh_jig_completion") for call in report["calls"]]
    return [call["completion"] for call in report["completion_evidence"]]


@pytest.mark.parametrize("completion", [
    {"finish_reason": "stop", "ambiguous": False},
    {"finish_reason": "length", "ambiguous": False},
    {"finish_reason": None, "ambiguous": True},
    {"finish_reason": "custom_provider_reason", "ambiguous": False},
])
def test_real_loop_persists_literal_evidence_before_next_call(workflow, completion):
    kind, root, script, run = workflow
    turns = 0
    def chat(messages, schemas):
        nonlocal turns
        assert "mesh_jig_completion" not in json.dumps(messages)
        turns += 1
        if turns > 1:
            saved = json.loads((root / "audit/report.json").read_text())
            assert evidence(saved, kind) == [completion]
        call = tool("write_file", json.dumps({"path": "attempts/a01/" + script, "text": "complete"}))
        return {"tool_calls": [call]}, {"prompt_tokens": 100, "completion_tokens": 10,
                                        "mesh_jig_completion": completion}
    report = run(chat, 2)
    assert evidence(report, kind) == [completion, completion]
    assert report["turns"] == 2 and report["tool_calls"] == 2
    assert report["prompt_tokens"] == 200 and report["completion_tokens"] == 20
    assert (root / "attempts/a01" / script).read_text() == "complete"


def test_missing_evidence_never_invents_a_finish_reason(workflow):
    kind, root, script, run = workflow
    def chat(messages, schemas):
        call = tool("write_file", json.dumps({"path": "attempts/a01/" + script, "text": "complete"}))
        return {"tool_calls": [call]}, {"prompt_tokens": 100, "completion_tokens": 10}
    report = run(chat, 2)
    assert "finish_reason" not in json.dumps(report)
    if kind == "numeric":
        assert all("mesh_jig_completion" not in call for call in report["calls"])
    elif "completion_evidence" in report:
        assert evidence(report, kind) == [None, None]
    assert report["prompt_tokens"] == 200 and report["completion_tokens"] == 20


def test_reused_provider_dictionary_cannot_rewrite_earlier_evidence(workflow):
    kind, _, script, run = workflow
    completion = {"finish_reason": "length", "ambiguous": False}
    turns = 0
    def chat(messages, schemas):
        nonlocal turns
        turns += 1
        if turns == 2:
            completion["finish_reason"] = "tool_calls"
        call = tool("write_file", json.dumps({"path": "attempts/a01/" + script, "text": "complete"}))
        return {"tool_calls": [call]}, {"mesh_jig_completion": completion}
    report = run(chat, 2)
    assert evidence(report, kind) == [{"finish_reason": "length", "ambiguous": False},
                                      {"finish_reason": "tool_calls", "ambiguous": False}]


def test_rejected_tool_batch_keeps_ending_evidence_without_executing(workflow):
    kind, root, _, run = workflow
    completion = {"finish_reason": "length", "ambiguous": False}
    def chat(messages, schemas):
        return {"tool_calls": [tool("write_file", '{"path":')]}, {
            "completion_tokens": 32768, "mesh_jig_completion": completion}
    report = run(chat, 1)
    assert report["invalid_tool_batches"] == 1 and report["completion_tokens"] == 32768
    assert evidence(report, kind) == [completion]
    assert not list((root / "attempts").glob("*/*.py"))


def test_optional_rig_critic_keeps_its_own_evidence(tmp_path):
    write_glb(tmp_path / "source.glb", FIGURE)
    (tmp_path / "rig.json").write_text(json.dumps(SPEC))
    (tmp_path / "brief.md").write_text("Walk in place.")
    replies = iter([
        tool("write_file", json.dumps({"path": "attempts/a01/rig.py", "text": walk()})),
        tool("build", json.dumps({"script": "attempts/a01/rig.py"})),
        tool("done", json.dumps({"best": "a01", "summary": "walks"})),
    ])
    def chat(messages, schemas):
        completion = {"finish_reason": "tool_calls" if schemas else "stop", "ambiguous": False}
        message = {"tool_calls": [next(replies)]} if schemas else {"content": "Legs alternate."}
        return message, {"prompt_tokens": 10, "completion_tokens": 2, "mesh_jig_completion": completion}
    report = rig_agent.run(tmp_path, chat, builds=1, max_turns=3, reviewer=True)
    assert report["done"] and report["reviews"] == 1
    assert [(c["role"], c["turn"]) for c in report["completion_evidence"]] == [
        ("implementor", 1), ("implementor", 2), ("reviewer", 2), ("implementor", 3)]
    assert [c["finish_reason"] for c in evidence(report, "rig")] == ["tool_calls", "tool_calls", "stop", "tool_calls"]
    assert report["prompt_tokens"] == 40 and report["completion_tokens"] == 8
