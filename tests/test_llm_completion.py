"""Completion evidence remains diagnostic and never enters a provider message."""
import json

import pytest

from mesh_jig import agent, llm
from test_llm_stream import BACKEND, Upstream, sse, stream


def terminal(reason, index=0):
    return {"choices": [{"index": index, "delta": {}, "finish_reason": reason}]}


@pytest.mark.parametrize("reason", ["stop", "tool_calls", "length", "content_filter", "provider-specific"])
def test_literal_primary_completion_reason_survives_later_usage(reason):
    chunks = [{"choices": [{"delta": {"content": "reply"}}]}, terminal(reason),
              {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "cost": .001}}]
    msg, usage = llm.assemble_chunks(chunks)
    assert msg == {"role": "assistant", "content": "reply"}
    assert usage["mesh_jig_completion"] == {"finish_reason": reason, "ambiguous": False}
    assert {k: v for k, v in usage.items() if k != "mesh_jig_completion"} == {
        "prompt_tokens": 7, "completion_tokens": 2, "cost": .001}


@pytest.mark.parametrize("reason", [None, "", " ", False, 4, {}, []])
def test_missing_or_malformed_reason_is_unavailable_not_guessed(reason):
    msg, usage = llm.assemble_chunks([{"choices": [], "usage": {"completion_tokens": 32768}}, terminal(reason)])
    assert usage["mesh_jig_completion"] == {"finish_reason": None, "ambiguous": False}


def test_nonprimary_choice_cannot_supply_or_overwrite_primary_reason():
    _, usage = llm.assemble_chunks([terminal("stop"), terminal("length", 1), terminal("length", False), terminal(None)])
    assert usage["mesh_jig_completion"] == {"finish_reason": "stop", "ambiguous": False}
    _, usage = llm.assemble_chunks([terminal("length", 1)])
    assert usage["mesh_jig_completion"]["finish_reason"] is None


def test_conflicting_primary_reasons_are_ambiguous():
    _, usage = llm.assemble_chunks([terminal("stop"), terminal("length")])
    assert usage["mesh_jig_completion"] == {"finish_reason": None, "ambiguous": True}
    _, usage = llm.assemble_chunks([terminal("stop"), terminal("stop")])
    assert usage["mesh_jig_completion"] == {"finish_reason": "stop", "ambiguous": False}


def test_complete_unterminated_and_length_tool_messages_stay_unchanged():
    tool = {"index": 0, "id": "c0", "function": {"name": "done", "arguments": "{}"}}
    prefix = {"choices": [{"delta": {"tool_calls": [tool]}}]}
    results = [stream(Upstream(sse(prefix, terminal(reason)))) for reason in ("tool_calls", "length", None)]
    assert results[0]["message"] == results[1]["message"] == results[2]["message"]
    assert [r["usage"]["mesh_jig_completion"]["finish_reason"] for r in results] == ["tool_calls", "length", None]
    # Clean HTTP EOF without an SSE terminal remains explicitly unknown. This
    # change deliberately does not invent or change tool-execution policy.
    eof = stream(Upstream([b"data: " + json.dumps(prefix).encode() + b"\n"]))
    assert eof["usage"]["mesh_jig_completion"]["finish_reason"] is None
    assert eof["message"] == results[0]["message"]


def test_reason_from_broken_attempt_does_not_leak_to_retry():
    broken = [b"data: " + json.dumps(terminal("length")).encode() + b"\n", b"data: {broken\n"]
    good = sse({"choices": [{"delta": {"content": "retry reply"}}]})
    result = stream(Upstream(broken, good), sleep=lambda seconds: None)
    assert result["text"] == "retry reply"
    assert result["usage"]["mesh_jig_completion"]["finish_reason"] is None


def test_agent_retains_completion_in_usage_but_provider_message_is_unchanged(monkeypatch):
    message = {"role": "assistant", "content": "kept reply"}
    usage = {"completion_tokens": 4, "mesh_jig_completion": {"finish_reason": "length", "ambiguous": False}}
    monkeypatch.setattr(llm, "chat_stream", lambda *a, **kw: {"message": message, "usage": usage})
    assert agent.chat_for(BACKEND)([], []) == (message, usage)
    assert set(message) == {"role", "content"}
