import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mesh_jig import anthropic_sweep as native, gauntlet_sweep, llm


def test_budget_reserves_concurrent_calls_and_keeps_failed_requests(tmp_path):
    budget = native.Budget(tmp_path / "budget.json", 12)
    def reserve(_):
        try:
            return budget.reserve(65536, "test")
        except llm.LLMError:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert sum(x is not None for x in results) == 2
    assert budget.committed() == pytest.approx(11.42144)
    budget.settle(0, {"input_tokens": 1000, "output_tokens": 1000})
    assert budget.committed() == pytest.approx(5.73472)
    restored = native.Budget(budget.path, 12)
    assert restored.entries[1]["state"] == "reserved"
    with pytest.raises(llm.LLMError, match="missing usage"):
        restored.settle(1, {})
    assert restored.committed() == budget.committed()


def stream_events(events):
    return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events))


def events():
    return [
        {"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 20, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "opaque"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "call1", "name": "read_file", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path":"brief.md"}'}},
        {"type": "message_delta", "usage": {"output_tokens": 100}},
        {"type": "message_stop"},
    ]


def test_stream_empty_tool_delta_preserves_initial_input_and_usage():
    response = events()
    response[3]["content_block"]["name"] = "submit"
    response[4]["delta"]["partial_json"] = ""
    blocks, usage, model, stop = native.read_stream(stream_events(response))
    assert stop is None
    assert blocks[1]["input"] == {}
    assert blocks[0]["signature"] == "opaque"
    assert usage == {"input_tokens": 20, "output_tokens": 100}
    assert model == "claude-opus-5-5"


def test_stream_empty_data_line_is_ignored():
    response = io.BytesIO(b"data: \n\n" + stream_events(events()).getvalue())
    blocks, usage, _, _ = native.read_stream(response)
    assert blocks[1]["input"] == {"path": "brief.md"}
    assert usage["output_tokens"] == 100


@pytest.mark.parametrize("value", [" ", "{broken"])
def test_stream_nonempty_invalid_tool_json_is_rejected(value):
    response = events()
    response[4]["delta"]["partial_json"] = value
    with pytest.raises(llm.LLMError, match="tool input JSON"):
        native.read_stream(stream_events(response))


def test_stream_invalid_event_json_is_rejected():
    with pytest.raises(llm.LLMError, match="event JSON"):
        native.read_stream(io.BytesIO(b"data: broken\n\n"))


def test_native_high_tools_images_thinking_roundtrip_and_usage(tmp_path):
    requests = []
    def opener(req, **kwargs):
        requests.append(json.loads(req.data))
        assert req.full_url == "https://api.anthropic.com/v1/messages"
        return stream_events(events())
    api = SimpleNamespace(model="claude-opus-5-5", url="https://api.anthropic.com/v1", key="test", timeout_s=1800)
    budget = native.Budget(tmp_path / "budget.json", 20)
    chat = native.chat_for(api, budget, "test", opener=opener)
    msgs = [{"role": "system", "content": "rules"}, {"role": "user", "content": [
        {"type": "text", "text": "build"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    schemas = [{"function": {"name": "read_file", "parameters": {"type": "object"}}}]
    msg, usage = chat(msgs, schemas)
    assert usage["prompt_tokens"] == 20 and usage["completion_tokens"] == 100
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"path": "brief.md"}'
    chat(msgs + [msg, {"role": "tool", "tool_call_id": "call1", "content": "brief"},
                 {"role": "user", "content": "continue"}], schemas)
    body = requests[1]
    assert body["output_config"] == {"effort": "high"}
    assert body["thinking"] == {"type": "adaptive"}
    assert body["messages"][0]["content"][1]["source"]["media_type"] == "image/png"
    assert body["messages"][1]["content"][0]["signature"] == "opaque"
    assert [b["type"] for b in body["messages"][2]["content"]] == ["tool_result", "text"]
    assert budget.committed() == pytest.approx(.00416)


def test_incomplete_stream_is_not_retried_and_retains_reservation(tmp_path):
    count = []
    def opener(*a, **kw):
        count.append(1)
        return stream_events(events()[:-1])
    api = SimpleNamespace(model="claude-opus-5-5", url="https://api.anthropic.com/v1", key="test", timeout_s=1800)
    budget = native.Budget(tmp_path / "budget.json", 10)
    chat = native.chat_for(api, budget, "test", opener=opener)
    with pytest.raises(llm.LLMError, match="incomplete"):
        chat([{"role": "user", "content": "hello"}], [])
    assert len(count) == 1
    assert budget.committed() == pytest.approx(5.71072)


def test_opus_rates_and_effort_are_explicit():
    assert gauntlet_sweep.estimate_cost({"prompt_tokens": 100001, "completion_tokens": 1000}, "claude-opus-5-5") == .420004
    assert "output_config.effort=high" in gauntlet_sweep.effort_control("claude-opus-5-5", "high")
    with pytest.raises(ValueError, match="explicit effort"):
        gauntlet_sweep.effort_control("claude-opus-5-5", "medium")


def test_budget_waits_for_inflight_usage_before_refusing(tmp_path, monkeypatch):
    budget = native.Budget(tmp_path / "budget.json", 6)
    ticket = budget.reserve(65536, "first")
    waiting = threading.Event()
    real_wait = budget.changed.wait
    def wait(timeout=None):
        waiting.set()
        return real_wait(timeout)
    monkeypatch.setattr(budget.changed, "wait", wait)
    with ThreadPoolExecutor(max_workers=1) as pool:
        second = pool.submit(budget.reserve, 65536, "second", wait=True)
        assert waiting.wait(2)
        budget.settle(ticket, {"input_tokens": 100, "output_tokens": 100})
        assert second.result(timeout=2) == 1


@pytest.mark.parametrize("model,reserved,settled", [
    ("claude-fable-5-2", 11.42144, .048),
    ("claude-sonnet-5-5", 2.85536, .012),
    ("claude-haiku-5-5", .71384, .0006),
])
def test_model_specific_reservations_and_usage(tmp_path, model, reserved, settled):
    budget = native.Budget(tmp_path / "budget.json", 20, model)
    ticket = budget.reserve(65536, "both roles")
    assert budget.committed() == pytest.approx(reserved)
    budget.settle(ticket, {"input_tokens": 1000, "output_tokens": 1000})
    assert budget.committed() == pytest.approx(settled)
    with pytest.raises(ValueError, match="another model"):
        native.Budget(budget.path, 10, "claude-opus-5-5")


def test_haiku_long_context_settlement_uses_higher_rates(tmp_path):
    budget = native.Budget(tmp_path / "budget.json", 10, "claude-haiku-5-5")
    ticket = budget.reserve(65536, "long context")
    budget.settle(ticket, {"input_tokens": 100001, "output_tokens": 1000})
    assert budget.committed() == pytest.approx(.0525005)


@pytest.mark.parametrize("effort", [None, "low", "medium", "high"])
def test_native_effort_and_uncapped_calls(effort):
    requests = []
    def opener(req, **kwargs):
        requests.append(json.loads(req.data))
        response = events()
        response[0]["message"]["model"] = "claude-sonnet-5-5"
        return stream_events(response)
    api = SimpleNamespace(model="claude-sonnet-5-5", url="https://api.anthropic.com/v1", key="test", timeout_s=900)
    native.chat_for(api, None, "test", effort=effort, opener=opener)([{"role": "user", "content": "hi"}], [])
    assert requests[0].get("output_config") == ({"effort": effort} if effort else None)


def test_unexpected_serving_model_keeps_full_reservation(tmp_path):
    def opener(*a, **kw):
        response = events()
        response[0]["message"]["model"] = "unexpected-model"
        return stream_events(response)
    api = SimpleNamespace(model="claude-opus-5-5", url="https://api.anthropic.com/v1", key="test", timeout_s=900)
    budget = native.Budget(tmp_path / "budget.json", 10)
    with pytest.raises(llm.LLMError, match="unexpected serving"):
        native.chat_for(api, budget, "test", opener=opener)([{"role": "user", "content": "hi"}], [])
    assert budget.committed() == pytest.approx(5.71072)
    assert budget.entries[0]["state"] == "unknown_charge_reserved"


def cut_events():
    """A reply that ran into the output cap in the middle of a tool call: text, then half of a write_file."""
    response = events()
    response[3]["content_block"]["name"] = "write_file"
    response[4]["delta"]["partial_json"] = '{"path": "attempts/a01/model.py", "text": "def build():'
    response[5] = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 65536}}
    response[3:3] = [{"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
                     {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Writing it."}}]
    return response


def test_the_stop_reason_is_read_and_a_reply_that_ended_itself_runs_its_calls():
    response = events()
    response[5] = {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 100}}
    api = SimpleNamespace(model="claude-opus-5-5", url="https://api.anthropic.com/v1", key="test", timeout_s=900)
    msg, usage = native.chat_for(api, None, "test", opener=lambda *a, **kw: stream_events(response))(
        [{"role": "user", "content": "hi"}], [])
    assert usage["mesh_jig_completion"] == {"finish_reason": "tool_use", "ambiguous": False}
    assert not llm.cut_short(usage) and msg["tool_calls"][0]["function"]["name"] == "read_file"


def test_a_reply_cut_at_the_output_cap_runs_no_call_and_its_thinking_is_not_sent_back():
    """Before this the half-written tool input was an error that ended the run, and nothing said the cap was hit."""
    requests = []

    def opener(req, **kwargs):
        requests.append(json.loads(req.data))
        return stream_events(cut_events())
    api = SimpleNamespace(model="claude-opus-5-5", url="https://api.anthropic.com/v1", key="test", timeout_s=900)
    chat = native.chat_for(api, None, "test", opener=opener)
    msgs = [{"role": "user", "content": "build"}]
    msg, usage = chat(msgs, [])
    assert llm.cut_short(usage) and llm.finish_reason(usage) == "max_tokens"
    assert msg == {"role": "assistant", "content": "Writing it."}
    chat(msgs + [msg, {"role": "user", "content": "again"}], [])
    assert requests[1]["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": "Writing it."}]}


def test_broken_tool_input_in_a_reply_that_was_not_cut_is_still_an_error():
    response = cut_events()
    response[7] = {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 100}}
    with pytest.raises(llm.LLMError, match="tool input JSON"):
        native.read_stream(stream_events(response))


@pytest.mark.parametrize("usage,cut", [({"mesh_jig_completion": {"finish_reason": "length"}}, True),
                                       ({"mesh_jig_completion": {"finish_reason": "max_tokens"}}, True),
                                       ({"mesh_jig_completion": {"finish_reason": "stop"}}, False),
                                       ({"mesh_jig_completion": {"finish_reason": None, "ambiguous": True}}, False),
                                       ({}, False), (None, False)])
def test_a_cut_reply_is_named_by_the_endpoints_own_reason_and_never_guessed(usage, cut):
    assert llm.cut_short(usage) is cut
