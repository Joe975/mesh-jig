"""Adversarial checks for durable agent state and bounded context tools."""
from __future__ import annotations

import json
import os
import pytest

from mesh_jig import agent, agent_state
from test_mcp_server import proj  # noqa: F401


SCRIPT = "def build():\n    pass\n"


def tools(proj, *, previews=2):
    return agent.Tools(proj, agent.Report(), attempts=8, judge=False, max_previews=previews)


def test_preview_aliases_share_budget_and_pending_state_survives_restart(proj):
    t = tools(proj, previews=2)
    t.call("write_file", {"path": "attempts/a01/model.py", "text": SCRIPT})
    built, _ = t.call("build", {"script": "attempts/a01/model.py"})
    assert "BUILD ERROR" not in built
    first, _ = t.call("preview", {"attempt": "attempts/a01"})
    assert "free look" in first
    assert t.lifecycle.pending == "a01" and t.lifecycle.counts == {"a01": 1}

    resumed = tools(proj, previews=2)
    assert resumed.lifecycle.pending == "a01"
    second, _ = resumed.call("preview", {"attempt": "a01"})
    assert "free look" in second
    blocked, _ = resumed.call("preview", {"attempt": str(proj.root / "attempts" / "a01")})
    assert blocked.startswith("REFUSED:")
    state = json.loads((proj.root / "agent" / "candidate-state.json").read_text())
    assert state["previews"] == {"a01": 2} and state["pending"] == "a01"


def test_failed_evaluation_keeps_candidate_pending_until_a_successful_retry(proj, monkeypatch):
    t = tools(proj)
    t.call("write_file", {"path": "attempts/a01/model.py", "text": SCRIPT})
    t.call("build", {"script": "attempts/a01/model.py"})
    preview, _ = t.call("preview", {"attempt": "attempts/a01"})
    assert "free look" in preview
    t.call("edit_file", {"path": "attempts/a01/model.py", "old": "pass", "new": "bad python"})
    from mesh_jig import evaluate as evaluate_mod
    real_evaluate = evaluate_mod.evaluate
    def fail_eval(project, script, out, **kwargs):
        record = {"ok": False, "name": "a01", "numeric": None}
        (out / "eval.json").write_text(json.dumps(record))
        return record
    monkeypatch.setattr(evaluate_mod, "evaluate", fail_eval)
    failed, _ = t.call("evaluate", {"script": "attempts/a01/model.py"})
    assert "built, in contract" not in failed
    assert t.lifecycle.pending == "a01"
    blocked, _ = t.call("write_file", {"path": "attempts/a02/model.py", "text": SCRIPT})
    assert blocked.startswith("REFUSED:")

    resumed = tools(proj)
    assert resumed.lifecycle.pending == "a01"
    resumed.call("write_file", {"path": "attempts/a01/model.py", "text": SCRIPT})
    monkeypatch.setattr(evaluate_mod, "evaluate", real_evaluate)
    measured, _ = resumed.call("evaluate", {"script": "attempts/a01/model.py"})
    assert agent.MEASURED in measured
    assert resumed.lifecycle.pending is None
    assert json.loads((proj.root / "agent" / "candidate-state.json").read_text())["pending"] is None


def test_new_tools_instance_recovers_project_global_best(proj):
    baseline = tools(proj)
    baseline.call("write_file", {"path": "attempts/a01/model.py", "text": SCRIPT})
    measured, _ = baseline.call("evaluate", {"script": "attempts/a01/model.py"})
    assert agent.MEASURED in measured
    resumed = tools(proj)
    assert resumed.rep.best["name"] == "a01"
    assert resumed.rep.best["numeric_score"] is not None
    assert "a01" in resumed.checkpoint()


def test_bounded_read_ranges_and_search_caps_are_dispatched(proj):
    t = tools(proj)
    file = proj.root / "attempts" / "notes.txt"
    file.parent.mkdir(parents=True)
    file.write_text("".join(f"line-{i} target\n" for i in range(1, 31)))
    excerpt, _ = t.call("read_file", {"path": "attempts/notes.txt", "start_line": 3, "end_line": 6})
    assert "Lines 3-6 of 30" in excerpt
    assert "line-3 target" in excerpt and "line-7 target" not in excerpt
    with pytest.raises(ValueError, match="1-based start_line"):
        t.call("read_file", {"path": "attempts/notes.txt", "start_line": 0, "end_line": 5})
    matches, _ = t.call("search_file", {"path": "attempts/notes.txt", "query": "target"})
    assert matches.count("target") == 20 and "20-match limit" in matches


def test_compaction_keeps_whole_tool_rounds_and_recent_context():
    schemas = []
    opening = [{"role": "system", "content": "contract"}, {"role": "user", "content": "references"}]
    old_round = [{"role": "assistant", "content": None, "tool_calls": [{"id": "old", "function": {"name": "read_file"}}]},
                 {"role": "tool", "tool_call_id": "old", "content": "old result"}]
    recent = [{"role": "assistant", "content": None, "tool_calls": [{"id": "new", "function": {"name": "search_file"}}]},
              {"role": "tool", "tool_call_id": "new", "content": "recent result"},
              {"role": "user", "content": "recent image caption"}]
    msgs = opening + old_round + [{"role": "user", "content": "x" * 3000}] + recent
    compacted, changed = agent_state.compact(msgs, schemas, "checkpoint with best and pending", 1200, 100)
    assert changed
    assert compacted[:2] == opening
    assert "checkpoint with best and pending" in compacted[2]["content"]
    assert compacted[-3:] == recent
    assert not any(m.get("tool_call_id") == "new" for m in compacted[:-2])
    assert agent_state.estimate_tokens(compacted, schemas) <= 1100


def test_full_transcript_retains_messages_removed_from_model_context(proj, tmp_path):
    class Chat:
        def __init__(self):
            self.sent = []
            self.n = 0
        def __call__(self, messages, schemas):
            self.sent.append(json.loads(json.dumps(messages)))
            self.n += 1
            if self.n == 1:
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "large", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "attempts/a01/model.py", "text": "# " + "z" * 10000})}}]}, {"cost": .001}
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "done", "type": "function", "function": {"name": "done", "arguments": '{"summary":"finished"}'}}]}, {"cost": .001}

    chat = Chat()
    out = tmp_path / "run"
    rep = agent.run(proj, chat, attempts=1, max_turns=2, context_tokens=47000, reply_reserve=200, out=out)
    transcript = (out / "transcript.jsonl").read_text()
    assert rep.context_compactions >= 1
    assert "large" in transcript and "wrote" in transcript and "checkpoint" not in transcript.lower()
    assert "large" not in json.dumps(chat.sent[1])  # compact model context, complete persisted transcript


@pytest.mark.parametrize("payload", [
    "not json", "null", "[]", '"state"', '{"version":1,"previews":null}',
    '{"version":1,"previews":[],"pending":null}',
    '{"version":1,"previews":{"a01":-1},"pending":null}',
    '{"version":1,"previews":{"a01":true},"pending":null}',
    '{"version":1,"previews":{},"pending":[]}',
    '{"version":2,"previews":{},"pending":null}',
])
def test_corrupt_lifecycle_state_fails_closed_with_a_value_error(proj, payload):
    state = proj.root / "agent" / "candidate-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(payload)
    with pytest.raises(ValueError, match="candidate-state.json|persisted preview count|persisted candidate"):
        agent_state.Lifecycle(proj.root)


@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_preview_policy_rejects_noninteger_caps(proj, value):
    with pytest.raises(ValueError, match="max_previews"):
        agent_state.Lifecycle(proj.root, max_previews=value)


def test_case_aliases_cannot_reset_a_persisted_preview_count_on_windows(proj):
    attempt = proj.root / "attempts" / "a01"
    attempt.mkdir(parents=True)
    state = proj.root / "agent" / "candidate-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"version": 1, "previews": {"A01": 1}, "pending": "A01"}))
    lifecycle = agent_state.Lifecycle(proj.root, max_previews=2)
    if os.name == "nt":
        assert len(lifecycle.counts) == 1
        assert lifecycle.pending == os.path.normcase(attempt.resolve().name)
        assert lifecycle.refuse("a01", preview=True) == ""
        lifecycle.previewed("a01")
        assert lifecycle.refuse("A01", preview=True).startswith("REFUSED:")


def test_symlinked_candidate_cannot_escape_attempts(proj, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    attempts = proj.root / "attempts"
    attempts.mkdir(exist_ok=True)
    link = attempts / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    state = proj.root / "agent" / "candidate-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text('{"version":1,"previews":{"escape":1},"pending":null}')
    with pytest.raises(ValueError, match="escapes attempts"):
        agent_state.Lifecycle(proj.root)


def test_state_directory_symlink_cannot_redirect_persistent_writes_outside_project(proj, tmp_path):
    outside = tmp_path / "outside-state"
    outside.mkdir()
    link = proj.root / "agent"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match="candidate-state.json path escapes"):
        agent_state.Lifecycle(proj.root)


def test_attempts_root_symlink_cannot_make_candidate_validation_escape_project(proj, tmp_path):
    outside = tmp_path / "outside-attempts"
    outside.mkdir()
    link = proj.root / "attempts"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    state = proj.root / "agent" / "candidate-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text('{"version":1,"previews":{"candidate":1},"pending":null}')
    with pytest.raises(ValueError, match="attempts/ escapes the project"):
        agent_state.Lifecycle(proj.root)


def test_policy_overrides_allow_unbounded_previews_and_disable_pending_lock(proj):
    t = agent.Tools(proj, agent.Report(), attempts=8, judge=False, max_previews=0,
                    require_evaluation=False)
    t.call("write_file", {"path": "attempts/a01/model.py", "text": SCRIPT})
    t.call("build", {"script": "attempts/a01/model.py"})
    for _ in range(3):
        result, _ = t.call("preview", {"attempt": "attempts/a01"})
        assert "free look" in result
    assert t.lifecycle.refuse("a01", preview=True) == ""
    assert t.lifecycle.refuse("a02") == ""
    assert t.lifecycle.counts[os.path.normcase("a01")] == 3
    t.call("write_file", {"path": "attempts/a02/model.py", "text": SCRIPT})
    t.call("build", {"script": "attempts/a02/model.py"})
    result, _ = t.call("preview", {"attempt": "attempts/a02"})
    assert "free look" in result and t.lifecycle.pending == os.path.normcase("a02")


def test_pending_candidate_blocks_other_build_paint_copy_and_edit_dispatch(proj):
    t = tools(proj)
    t.lifecycle.previewed("a01")
    other = proj.root / "attempts" / "a02"
    other.mkdir(parents=True)
    (other / "model.py").write_text(SCRIPT)
    calls = [
        ("build", {"script": "attempts/a02/model.py"}),
        ("paint", {"out": "attempts/a02/atlas.png"}),
        ("copy_attempt", {"source": "attempts/a01", "name": "a02_copy"}),
        ("edit_file", {"path": "attempts/a02/model.py", "old": "pass", "new": "return"}),
    ]
    for name, args in calls:
        result, _ = t.call(name, args)
        assert result.startswith("REFUSED: evaluate pending candidate attempts/a01/"), (name, result)
