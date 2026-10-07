"""The streamed chat and its idle watchdog: a hung upstream is dropped after the idle limit and retried, instead of
holding the agent for a whole request timeout. No network: the opener is a fake whose stream can hang."""
import io
import json
import threading
import time
import urllib.error

import pytest

from mesh_jig import agent, cli, llm

IDLE = 0.2      # the idle limit under test; a healthy fake answers in well under it
POLL = 0.005
BACKEND = llm.Backend("openai-compatible", "http://x/v1", "m", "k", timeout_s=30.0)
HELLO = [{"choices": [{"delta": {"role": "assistant", "content": "hel"}}]},
         {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
         {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "cost": 0.001}}]


def sse(*chunks) -> list[bytes]:
    out = []
    for c in chunks:
        out += [b"data: " + json.dumps(c).encode() + b"\n", b"\n"]
    return out + [b"data: [DONE]\n", b"\n"]


class Stream:
    def __init__(self, lines):
        self.lines, self.closed = lines, False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class Upstream:
    """An opener that plays one scripted behaviour per call: lines to stream, an exception to raise, or a callable."""

    def __init__(self, *script):
        self.script, self.requests, self.timeouts = list(script), [], []

    def __call__(self, req, timeout):
        self.requests.append(req)
        self.timeouts.append(timeout)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step() if callable(step) else Stream(step)


@pytest.fixture
def release():
    """Set at teardown, so the threads a test left hanging end with it."""
    ev = threading.Event()
    yield ev
    ev.set()


def no_sleep(_s):
    raise AssertionError("a stalled attempt is retried at once, without a backoff")


def stream(opener, **kw):
    kw.setdefault("sleep", lambda s: None)
    return llm.chat_stream(BACKEND, [{"role": "user", "content": "hi"}], opener=opener, idle_s=IDLE, poll_s=POLL, **kw)


def test_only_data_lines_are_chunks():
    lines = [b": OPENROUTER PROCESSING\n", b"\n", b'data: {"a": 1}\n', b"event: ping\n", b'data: {"b": 2}\r\n',
             b"data: [DONE]\n", b'data: {"after": "the end"}\n']
    assert list(llm.sse_chunks(lines)) == [{"a": 1}, {"b": 2}]
    with pytest.raises(llm.LLMError, match="error in the stream.*overloaded"):
        list(llm.sse_chunks([b'data: {"error": {"message": "overloaded"}}\n']))
    with pytest.raises(llm.LLMError, match="broken stream chunk"):
        list(llm.sse_chunks([b'data: {"choices": [{"de\n']))


def test_chunks_assemble_into_the_blocking_calls_message():
    msg, usage = llm.assemble_chunks(HELLO)
    assert msg == {"role": "assistant", "content": "hello"} and usage["cost"] == 0.001

    def delta(index, **fn):
        return {"choices": [{"delta": {"tool_calls": [{"index": index, **fn}]}}]}
    msg, _ = llm.assemble_chunks([
        {"choices": [{"delta": {"reasoning": "hmm"}}]},
        delta(0, id="c0", function={"name": "write_file", "arguments": '{"path": "a'}),
        delta(1, id="c1", function={"name": "done", "arguments": ""}),
        delta(0, function={"arguments": '.py"}'}),
        delta(1, function={"arguments": "{}"})])
    assert msg["content"] is None
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in msg["tool_calls"]] == [
        ("c0", "write_file", '{"path": "a.py"}'), ("c1", "done", "{}")]


def test_reasoning_details_are_assembled_whole_for_sending_back():
    def thought(**item):
        return {"choices": [{"delta": {"reasoning": "readable", "reasoning_details": [item]}}]}
    msg, _ = llm.assemble_chunks([
        thought(type="reasoning.text", text="the legs ", signature=None, format="anthropic-claude-v1", index=0),
        thought(type="reasoning.text", text="are short", signature=None, format="anthropic-claude-v1", index=0),
        thought(type="reasoning.text", signature="sig", format="anthropic-claude-v1", index=0),
        thought(type="reasoning.encrypted", data="abc", index=1), thought(type="reasoning.encrypted", data="def", index=1),
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "done", "arguments": "{}"}}]}}]}])
    assert msg["reasoning_details"] == [
        {"type": "reasoning.text", "text": "the legs are short", "signature": "sig", "format": "anthropic-claude-v1", "index": 0},
        {"type": "reasoning.encrypted", "data": "abcdef", "index": 1}]
    assert "reasoning" not in msg and msg["tool_calls"][0]["id"] == "c0"
    assert "reasoning_details" not in llm.assemble_chunks(HELLO)[0]


def test_the_stream_request_asks_for_usage_and_sends_no_sampling_defaults():
    up = Upstream(sse(*HELLO))
    out = llm.chat_stream(BACKEND, [], tools=[{"t": 1}], extra={"reasoning_effort": "low"}, opener=up, idle_s=IDLE,
                          poll_s=POLL)
    assert json.loads(up.requests[0].data) == {"model": "m", "messages": [], "tools": [{"t": 1}], "reasoning_effort": "low",
                                               "stream": True, "stream_options": {"include_usage": True}}
    assert up.requests[0].get_header("Authorization") == "Bearer k"
    assert up.timeouts == [IDLE + llm.TRANSPORT_SLACK_S]
    assert out["text"] == "hello" and out["usage"]["prompt_tokens"] == 7


def test_a_hung_upstream_is_dropped_after_the_idle_limit_and_retried_at_once(release):
    def hung():
        release.wait(30)
        raise OSError("closed")
    up, said = Upstream(hung, sse(*HELLO)), []
    t0 = time.monotonic()
    out = stream(up, sleep=no_sleep, log=said.append)
    assert out["message"]["content"] == "hello" and len(up.requests) == 2
    assert time.monotonic() - t0 < 10 * IDLE                 # not the 30 s the hung call would have held it
    assert len(said) == 1 and f"no stream chunk for {IDLE:g} s (0 chunks in)" in said[0]


def test_keep_alive_comments_do_not_count_as_progress(release):
    closed = []

    class Comments(Stream):
        def __iter__(self):
            while not release.is_set():
                time.sleep(0.002)
                yield b": OPENROUTER PROCESSING\n"

        def close(self):
            closed.append(True)

    with pytest.raises(llm.LLMError, match=r"no stream chunk .* \(after 1 attempts; .* longer idle limit\)"):
        stream(Upstream(lambda: Comments([])), retries=0)
    release.wait(0.5)                                        # the close runs in its own thread
    assert closed == [True]


def test_a_stream_that_never_ends_is_cut_at_the_cap(release):
    class Trickle(Stream):
        def __iter__(self):
            while not release.is_set():
                time.sleep(0.002)
                yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n'

    capped = llm.Backend("openai-compatible", "http://x/v1", "m", "k", timeout_s=0.1)
    with pytest.raises(llm.LLMError, match="still streaming at the 0.1 s cap") as e:
        llm.chat_stream(capped, [], opener=Upstream(lambda: Trickle([])), idle_s=IDLE, poll_s=POLL, retries=0)
    assert "idle limit" not in str(e.value)


def test_a_call_gets_what_is_left_of_its_budget_and_is_not_repeated_past_it(release):
    class Trickle(Stream):
        def __iter__(self):
            while not release.is_set():
                time.sleep(0.002)
                yield b'data: {"choices": [{"delta": {"reasoning": "hm"}}]}\n'

    up = Upstream(*[lambda: Trickle([])] * 3)
    t0 = time.monotonic()
    with pytest.raises(llm.OutOfTime, match=r"ran out: still streaming at the 0\.\d s cap"):
        stream(up, budget_s=0.15, sleep=no_sleep)            # BACKEND's own cap is 30 s
    assert len(up.requests) == 1 and time.monotonic() - t0 < 5
    with pytest.raises(llm.OutOfTime):
        stream(up, budget_s=0.0)                             # nothing left: no request is made
    assert len(up.requests) == 1
    assert stream(Upstream(sse(*HELLO)), budget_s=30.0)["text"] == "hello"


def test_a_long_call_says_how_long_it_has_run_and_whether_it_is_still_thinking():
    class Thinks(Stream):
        def __iter__(self):
            for _ in range(60):
                time.sleep(0.002)
                yield b'data: {"choices": [{"delta": {"reasoning": "hm"}}]}\n'
            yield from sse(*HELLO)

    said = []
    assert stream(Upstream(lambda: Thinks([])), beat_s=0.03, log=said.append)["text"] == "hello"
    assert said and all(s.startswith("  the model call is ") for s in said) and "all of them thinking" in said[0]
    assert llm.replies({"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]})
    assert llm.replies(HELLO[0]) and not llm.replies(HELLO[2]) and not llm.replies({"choices": [{"delta": {"reasoning": "x"}}]})


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "nope", {}, io.BytesIO(b'{"error": "nope"}'))


def test_a_refused_request_raises_at_once_and_a_busy_one_backs_off():
    up = Upstream(http_error(401), sse(*HELLO))
    with pytest.raises(llm.LLMError, match="HTTP 401"):
        stream(up)
    assert len(up.requests) == 1
    slept = []
    up = Upstream(http_error(503), urllib.error.URLError("refused"), sse(*HELLO))
    assert stream(up, sleep=slept.append)["text"] == "hello"
    assert len(slept) == 2 and 2 <= slept[0] < 3 and 4 <= slept[1] < 5


def test_a_stream_that_ends_empty_or_broken_is_retried():
    up = Upstream([b"\n"], [b'data: {"choices": [{"de\n'], sse(*HELLO))
    said = []
    assert stream(up, log=said.append)["text"] == "hello"
    assert "ended without a reply" in said[0] and "broken stream chunk" in said[1]
    with pytest.raises(llm.LLMError, match=r"ended without a reply \(after 3 attempts\)"):
        stream(Upstream([], [], []))


def test_the_agents_chat_is_the_streamed_one(monkeypatch):
    seen = {}

    def fake(backend, messages, **kw):
        seen.update(kw, messages=messages)
        return {"text": "", "message": {"role": "assistant", "content": "ok"}, "usage": {"prompt_tokens": 1}}
    monkeypatch.setattr(llm, "chat_stream", fake)
    chat = agent.chat_for(BACKEND, effort="low", idle_s=12.0)
    assert chat([{"role": "user", "content": "x"}], [{"t": 1}]) == ({"role": "assistant", "content": "ok"}, {"prompt_tokens": 1})
    assert seen["idle_s"] == 12.0 and seen["tools"] == [{"t": 1}] and seen["extra"] == {"reasoning_effort": "low"}
    assert seen["max_tokens"] is None


def test_the_agent_command_takes_an_idle_limit():
    assert cli.parser().parse_args(["agent", "p", "--model", "m"]).idle == llm.CHAT_IDLE_S
    assert cli.parser().parse_args(["agent", "p", "--model", "m", "--idle", "240"]).idle == 240.0
