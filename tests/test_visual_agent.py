import json
import re

import pytest

from mesh_jig import agent, anthropic_sweep, cli, gauntlet, llm, visual_agent
from test_gauntlet import seed, tool, fake_eval


def backend(model="local", url="http://localhost:1234/v1"):
    return llm.Backend("test", url, model, "test")


def install_loop(monkeypatch, scores=(.8, .6)):
    original = gauntlet.run
    values = iter(scores)
    def evaluate(*args, **kwargs):
        rec = fake_eval(*args, **kwargs)
        rec["numeric"]["numeric_score"] = next(values)
        return rec
    monkeypatch.setattr(gauntlet, "run", lambda *a, **kw: original(*a, **kw, evaluate_fn=evaluate))
    def chat(messages, schemas):
        if not schemas:
            assert len(messages) == 2
            text = json.dumps(agent.strip_images(messages))
            assert "numeric_score" not in text and "ADVERSARIAL FEEDBACK" not in text
            return {"content": "Lengthen the tail; preserve the face."}, {}
        if messages[-1]["role"] == "tool":
            return tool("submit"), {}
        text = json.dumps(agent.strip_images(messages))
        path = re.search(r"Current directory: (attempts/[^/]+/)", text)[1]
        return tool("write_file", path=path + "model.py", text="def build(): pass"), {}
    monkeypatch.setattr(agent, "chat_for", lambda *a, **kw: chat)


@pytest.mark.parametrize("model", ["local", "claude-sonnet-5-5"])
def test_runs_preserve_candidates_and_distinguish_numeric_best_from_latest(seed, monkeypatch, model):
    install_loop(monkeypatch, (.8, .6, .7))
    result, out = visual_agent.run(seed, backend(model), attempts=2, log=lambda s: None)
    assert result["done"] and result["evaluations"] == 2
    assert result["best_numeric"]["numeric_score"] == .8
    assert result["latest"]["numeric_score"] == .6
    assert result["cost_usd"] is None and result["cost_kind"] == "unpriced"
    before = {p: p.read_bytes() for p in seed.attempts_dir.glob("*/*") if p.is_file()}
    again, other = visual_agent.run(seed, backend(model), attempts=1, log=lambda s: None)
    assert again["done"] and out != other
    assert all(p.read_bytes() == value for p, value in before.items())
    assert len(list(seed.attempts_dir.glob("*/model.py"))) == 3
    assert json.loads((out / "report.json").read_text())["latest"] == result["latest"]


def test_default_cli_routes_to_visual_with_five_iterations(seed, monkeypatch):
    given = {}
    monkeypatch.setattr(llm, "resolve_backend", lambda **kw: backend())
    def run(proj, api, **kwargs):
        given.update(kwargs)
        return {"done": True, "evaluations": 0, "iterations_completed": 0}, proj.root / "agent"
    monkeypatch.setattr(visual_agent, "run", run)
    assert cli.main(["agent", str(seed.root)]) == 0
    assert given["attempts"] == 5 and given["checks"] == "none"


def test_the_cli_hands_a_pinned_provider_to_the_backend_and_the_report(seed, monkeypatch):
    install_loop(monkeypatch, (.8, .6))
    given = {}

    def resolve(**kwargs):
        given.update(kwargs)
        return llm.Backend("openrouter", llm.DEFAULT_URL, "m", "k", provider=kwargs["provider"])
    monkeypatch.setattr(llm, "resolve_backend", resolve)
    monkeypatch.setattr(llm, "agent_output_limit", lambda backend, requested, **kw: requested)
    assert cli.main(["agent", str(seed.root), "--attempts", "1", "--provider", "deepinfra/fp8"]) == 0
    assert given["provider"] == "deepinfra/fp8"
    report = next(seed.root.glob("agent/*/report.json"))
    assert json.loads(report.read_text())["provider"] == "deepinfra/fp8"


@pytest.mark.parametrize("flags", [["--patience", "3"], ["--cache", "on"], ["--judge"],
                                   ["--context-tokens", "65536"]])
def test_visual_rejects_numeric_only_flags(seed, flags):
    assert cli.main(["agent", str(seed.root), *flags]) == 2


@pytest.mark.parametrize("kwargs", [{"max_cost": 1}, {"max_cost": float("nan")},
                                    {"max_wall": -1}, {"attempts": 0}, {"max_tokens": 0}])
def test_invalid_limits_or_unverified_pricing_fail_before_api(seed, monkeypatch, kwargs):
    monkeypatch.setattr(agent, "chat_for", lambda *a, **kw: pytest.fail("must not call API"))
    with pytest.raises(ValueError):
        visual_agent.run(seed, backend(), **kwargs)
    assert not (seed.root / "agent").exists()


def test_native_budget_blocks_before_http_and_records_stop(seed, monkeypatch):
    def factory(api, budget, label, **kwargs):
        def chat(*a):
            budget.reserve(kwargs["max_tokens"], label)
            pytest.fail("tiny cap must block dispatch")
        return chat
    monkeypatch.setattr(anthropic_sweep, "chat_for", factory)
    result, out = visual_agent.run(seed, backend("claude-sonnet-5-5", gauntlet.URL),
                                   max_cost=.1, effort="high", log=lambda s: None)
    assert not result["done"] and "budget stop" in result["error"]
    assert result["budget"]["charged_or_reserved_usd"] == 0
    assert (out / "report.json").is_file()


def priced(monkeypatch, cost):
    """install_loop's chat, each reply reporting `cost` as OpenRouter does; returns the list of calls made."""
    looped, calls = agent.chat_for, []

    def chat_for(api, **kwargs):
        inner = looped(api, **kwargs)

        def chat(messages, schemas, *rest):
            message, usage = inner(messages, schemas, *rest)
            calls.append(schemas)
            return message, {**usage, "prompt_tokens": 10, "completion_tokens": 5,
                             **({"cost": cost} if cost is not None else {})}
        return chat
    monkeypatch.setattr(agent, "chat_for", chat_for)
    return calls


def test_openrouter_cap_holds_the_reported_cost(seed, monkeypatch):
    install_loop(monkeypatch, (.8, .6, .7))
    calls = priced(monkeypatch, .4)
    result, out = visual_agent.run(seed, llm.Backend("openrouter", "https://openrouter.ai/api/v1", "deepseek/x", "test"),
                                   attempts=3, max_cost=1, log=lambda s: None)
    assert len(calls) == 2, "after two calls of $0.40, a third could cross $1"
    assert not result["done"] and "cost cap" in result["error"]
    assert result["cost_usd"] == pytest.approx(.8) and result["cost_kind"] == "reported"
    assert (out / "report.json").is_file()


def test_openrouter_cap_stops_when_a_reply_reports_no_cost(seed, monkeypatch):
    install_loop(monkeypatch)
    calls = priced(monkeypatch, None)
    result, _ = visual_agent.run(seed, llm.Backend("openrouter", "https://openrouter.ai/api/v1", "deepseek/x", "test"),
                                 max_cost=1, log=lambda s: None)
    assert len(calls) == 1 and "reported no cost" in result["error"]


def test_openrouter_cost_is_reported_without_a_cap(seed, monkeypatch):
    install_loop(monkeypatch, (.8, .6))
    priced(monkeypatch, .01)
    result, _ = visual_agent.run(seed, llm.Backend("openrouter", "https://openrouter.ai/api/v1", "deepseek/x", "test"),
                                 attempts=1, log=lambda s: None)
    assert result["done"] and result["cost_kind"] == "reported" and result["cost_usd"] > 0


def test_global_turn_cap_stops_before_next_call(seed, monkeypatch):
    install_loop(monkeypatch)
    result, _ = visual_agent.run(seed, backend(), max_turns=2, log=lambda s: None)
    assert not result["done"] and result["evaluations"] == 1
    assert result["iterations_completed"] == 1 and result["turns"] == 2
    assert "turn cap" in result["error"]


def test_review_timeout_keeps_evaluated_candidate(seed, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(gauntlet.time, "monotonic", lambda: clock[0])
    def evaluate(*args, **kwargs):
        clock[0] = 10
        return fake_eval(*args, **kwargs)
    replies = iter([tool("write_file", path="attempts/a01/model.py", text="def build(): pass"), tool("submit")])
    result = gauntlet.run(seed, lambda *a: (next(replies), {}), iterations=1,
                          evaluate_fn=evaluate, max_wall=5, log=lambda s: None)
    assert not result["done"] and "time budget" in result["error"]
    assert result["evaluations"] == 1 and result["iterations_completed"] == 0
    assert result["latest"] == result["best_numeric"]
    assert result["attempts"][0]["reviewed"] is False


def test_the_reviewer_can_think_at_its_own_effort(seed, monkeypatch):
    install_loop(monkeypatch, (.8, .6, .7))
    looped = agent.chat_for                      # the loop's chat, which install_loop put in place
    calls = []

    def chat_for(api, *, effort=None, **kwargs):
        inner = looped(api, effort=effort, **kwargs)

        def chat(messages, schemas, *rest):
            calls.append(("reviewer" if not schemas else "builder", effort))
            return inner(messages, schemas, *rest)
        return chat
    monkeypatch.setattr(agent, "chat_for", chat_for)
    result, _ = visual_agent.run(seed, backend(), attempts=2, effort="low", review_effort="high", log=lambda s: None)
    assert result["done"] and result["effort"] == "low" and result["review_effort"] == "high"
    assert {c for c in calls} == {("builder", "low"), ("reviewer", "high")}
    calls.clear()
    visual_agent.run(seed, backend(), attempts=1, effort="low", log=lambda s: None)
    assert {c for c in calls} == {("builder", "low"), ("reviewer", "low")}, "unset, the reviewer thinks as the builder"
    assert cli.parser().parse_args(["agent", "p", "--review-effort", "high"]).review_effort == "high"
