"""Returned calls survive interruption at tool/next-request boundaries, offline."""
import json

import pytest

from mesh_jig import agent, rig_agent, sound_agent
from test_completion_evidence_transfer import evidence
from test_tool_protocol_transfer import workflow, tool
from test_evaluate import make_project
from test_rig_agent import root


class Interrupted(BaseException):
    """Bypass loop exception recovery, as an externally stopped worker would."""


def saved(root):
    out = root / "audit"
    assert (out / "report.json").is_file(), "returned usage was not checkpointed"
    return (json.loads((out / "report.json").read_text()),
            [json.loads(line) for line in (out / "transcript.jsonl").read_text().splitlines()])


@pytest.mark.parametrize("completion", [{"finish_reason": "tool_calls", "ambiguous": False},
                                        {"finish_reason": None, "ambiguous": True}, None])
@pytest.mark.parametrize("interrupt_at", ["first_tool", "second_tool", "next_chat"])
def test_returned_reply_and_completed_tools_survive_interruption(workflow, monkeypatch, completion, interrupt_at):
    kind, root, script, run = workflow
    module = {"numeric": agent, "rig": rig_agent, "sound": sound_agent}[kind]
    original = module.Tools.call
    calls = [tool("write_file", json.dumps({"path": "attempts/a01/" + script, "text": "kept"}), "write"),
             tool("list_dir", "{}", "list")]
    reply = {"content": "Returned reply", "tool_calls": calls}
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "cost": .001}
    if completion is not None:
        usage["mesh_jig_completion"] = completion

    def dispatch(self, name, args):
        if interrupt_at == "first_tool" or (interrupt_at == "second_tool" and name == "list_dir"):
            raise Interrupted()
        return original(self, name, args)

    monkeypatch.setattr(module.Tools, "call", dispatch)
    turns = 0

    def chat(messages, schemas):
        nonlocal turns
        turns += 1
        if turns == 2:
            raise Interrupted()
        return reply, usage

    with pytest.raises(Interrupted):
        run(chat, 2)
    report, transcript = saved(root)
    assert (report["prompt_tokens"], report["completion_tokens"]) == (100, 10)
    assert evidence(report, kind) == [completion]
    assert [m for m in transcript if m["role"] == "assistant"] == [dict(reply, role="assistant")]
    results = [m for m in transcript if m["role"] == "tool"]
    assert len(results) == {"first_tool": 0, "second_tool": 1, "next_chat": 2}[interrupt_at]
    assert not report["done"]


@pytest.mark.parametrize("malformed", [False, True])
def test_no_tool_and_rejected_turns_are_saved_before_next_request(workflow, malformed):
    kind, root, _, run = workflow
    reply = {"content": "Still thinking"}
    if malformed:
        reply["tool_calls"] = [tool("write_file", '{"path":')]
    turns = 0

    def chat(messages, schemas):
        nonlocal turns
        turns += 1
        if turns > 1:
            raise Interrupted()
        return reply, {"prompt_tokens": 7, "completion_tokens": 2}

    with pytest.raises(Interrupted):
        run(chat, 3)
    report, transcript = saved(root)
    assert report["prompt_tokens"] == 7 and report["completion_tokens"] == 2
    assert sum(m["role"] == "assistant" for m in transcript) == 1
    assert transcript[-1]["role"] == "user"
    assert report["invalid_tool_batches"] == int(malformed)
    assert not report["done"]


def test_numeric_unpriced_response_is_kept_but_never_executed(tmp_path):
    proj = make_project(tmp_path)
    reply = {"content": "Unpriced", "tool_calls": [
        tool("write_file", json.dumps({"path": "attempts/a01/model.py", "text": "never"}))]}
    rep = agent.run(proj, lambda *_: (reply, {"completion_tokens": 9}), max_cost=1, out=proj.root / "audit")
    report, transcript = saved(proj.root)
    assert "reported no cost" in rep.error
    assert report["completion_tokens"] == 9 and report["tool_calls"] == 0
    assert transcript[-1] == dict(reply, role="assistant")
    assert not (proj.root / "attempts/a01/model.py").exists()


def test_rig_reviewer_reply_saved_before_review_postprocessing(root, monkeypatch):
    reply = {"content": "  Reviewer returned this.  "}
    completion = {"finish_reason": None, "ambiguous": True}

    def build(self, name, args):
        self.rep["builds"] += 1
        self.rep["latest"] = "a01"
        return "BUILT AND IN CONTRACT", []

    def review(chat, *args):
        chat([], [])
        raise Interrupted()

    monkeypatch.setattr(rig_agent.Tools, "call", build)
    monkeypatch.setattr(rig_agent, "review", review)

    def chat(messages, schemas):
        if schemas:
            return {"tool_calls": [tool("build", '{}')]}, {"prompt_tokens": 10, "completion_tokens": 2}
        return reply, {"prompt_tokens": 5, "completion_tokens": 3, "mesh_jig_completion": completion}

    with pytest.raises(Interrupted):
        rig_agent.run(root, chat, reviewer=True, max_turns=1, out=root / "audit")
    report, transcript = saved(root)
    assert (report["prompt_tokens"], report["completion_tokens"]) == (15, 5)
    assert report["reviews"] == 1
    assert report["completion_evidence"][-1] == {
        "role": "reviewer", "turn": 1, "completion": completion, "message": reply}
    assert transcript[-1]["content"] == "BUILT AND IN CONTRACT"
    assert report["latest"] == "a01" and report["final"] is None
