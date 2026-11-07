"""A rejected tool call must not make the next provider request invalid."""
import copy
import json

from PIL import Image
import pytest

from mesh_jig import gauntlet, project


@pytest.fixture
def seed(tmp_path):
    root = tmp_path / "seed"
    root.mkdir()
    (root / "jig.json").write_text(json.dumps({"name": "test", "model": {}}))
    (root / "brief.md").write_text("A green creature.")
    (root / "refs").mkdir()
    Image.new("RGB", (16, 16), "green").save(root / "refs/front.png")
    return project.load(root)


def call(name, arguments, identifier="call"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def evaluate(proj, script, attempt, **kwargs):
    assert script.read_text() == "def build(): pass"
    return {"ok": True, "name": attempt.name, "numeric": {"numeric_score": .4},
            "compare": {"front": proj.references["front"]}, "gates": []}


def assert_provider_history(messages):
    for message in messages:
        for tool in message.get("tool_calls") or []:
            assert isinstance(json.loads(tool["function"]["arguments"] or "{}"), dict)


@pytest.mark.parametrize("bad_arguments", ['{"path":"model.py","text":"unterminated', '[]', '"text"', '42', 'null',
                                          '{"text":NaN}', '{"text":Infinity}', '{"text":-Infinity}',
                                          '{"value":[1e999]}', 0, False, {}, []])
def test_malformed_batch_recovers_without_poisoning_history(seed, bad_arguments):
    turns = 0
    bad = {"role": "assistant", "content": "I will write the model.",
           "tool_calls": [call("write_file", bad_arguments)]}
    frozen = copy.deepcopy(bad)
    def chat(messages, schemas):
        nonlocal turns
        assert_provider_history(messages)
        if not schemas:
            return {"content": "Review."}, {}
        turns += 1
        if turns == 1:
            return bad, {"completion_tokens": 32768}
        if turns == 2:
            assert messages[-1]["role"] == "user"
            assert "No call in this batch was executed" in messages[-1]["content"]
            assert not list(seed.attempts_dir.glob("*/model.py"))
            return {"tool_calls": [call("write_file", json.dumps({
                "path": "attempts/a01/model.py", "text": "def build(): pass"}))]}, {}
        return {"tool_calls": [call("submit", "{}")]}, {}
    report = gauntlet.run(seed, chat, iterations=1, max_turns=3, evaluate_fn=evaluate, log=lambda _: None)
    assert report["done"] and report["turns"] == 3
    assert report["recoveries"] == [{"round": 1, "turn": 1, "kind": "invalid-tool-batch", "ok": False}]
    assert bad == frozen
    event = json.loads((seed.root / "gauntlet/transcript.jsonl").read_text().splitlines()[0])
    assert event["response"] == frozen
    assert event["usage"]["completion_tokens"] == 32768


def test_mixed_batch_executes_no_prefix_write(seed):
    turns = 0
    prefix = call("write_file", json.dumps({"path": "attempts/a01/model.py", "text": "bad"}), "prefix")
    def chat(messages, schemas):
        nonlocal turns
        assert_provider_history(messages)
        turns += 1
        if turns > 1:
            assert not list(seed.attempts_dir.glob("*/model.py"))
        return {"tool_calls": [prefix, call("write_file", '{"text":', "partial")]}, {}
    report = gauntlet.run(seed, chat, iterations=1, max_turns=2, max_total_turns=3, log=lambda _: None)
    assert not report["done"] and report["turns"] == 3
    assert "total implementor turn cap" in report["error"]
    assert len([r for r in report["recoveries"] if r["kind"] == "invalid-tool-batch"]) == 3
    assert not list(seed.attempts_dir.glob("*/model.py"))


def test_valid_batch_keeps_tool_history_and_order(seed):
    valid = [call("write_file", json.dumps({"path": "attempts/a01/model.py", "text": "def build(): pass"}), "write"),
             call("submit", "", "submit")]
    def chat(messages, schemas):
        if schemas:
            return {"tool_calls": valid}, {}
        return {"content": "Review."}, {}
    report = gauntlet.run(seed, chat, iterations=1, max_turns=1, evaluate_fn=evaluate, log=lambda _: None)
    assert report["done"] and report["turns"] == 1 and report["recoveries"] == []


@pytest.mark.parametrize("batch", [[None], [call("write_file", "{}", "")],
                                   [{"id": "x", "function": None}],
                                   [{"id": "x", "function": {"arguments": "{}"}}], "invalid"])
def test_invalid_envelope_rejected(batch):
    with pytest.raises((ValueError, TypeError)):
        gauntlet.parse_tool_batch(batch)
