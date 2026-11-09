import json
from types import SimpleNamespace

import pytest

from mesh_jig import agent, gauntlet_sweep, project
from mesh_jig.gauntlet_sweep import Meter, estimate_cost


def test_price_threshold_applies_to_entire_request_and_reasoning_is_not_added_twice():
    assert estimate_cost({"prompt_tokens": 100000, "completion_tokens": 1000}) == .0105
    assert estimate_cost({"prompt_tokens": 100001, "completion_tokens": 1000}) == .0525005
    assert estimate_cost({"prompt_tokens": 100000, "completion_tokens": 1000,
                          "completion_tokens_details": {"reasoning_tokens": 900}}) == .0105
    assert estimate_cost({"prompt_tokens": 100000, "completion_tokens": 1000,
                          "prompt_tokens_details": {"cached_tokens": 50000}}) == .006


def test_missing_usage_and_unknown_cache_write_ttl_are_not_free():
    assert estimate_cost({}) is None
    assert estimate_cost({"prompt_tokens": 10}) is None
    assert estimate_cost({"prompt_tokens": 10, "completion_tokens": 10,
                          "prompt_tokens_details": {"cache_write_tokens": 5}}) is None


def test_meter_checkpoints_success_and_failure_without_fabricating_cost(tmp_path):
    def chat(messages, schemas):
        if not schemas:
            raise RuntimeError("offline failure")
        return {"content": "ok"}, {"prompt_tokens": 100000, "completion_tokens": 1000}

    meter = Meter(chat, tmp_path)
    meter([], [{}])
    with pytest.raises(RuntimeError):
        meter([], [])
    account = json.loads((tmp_path / "accounting.json").read_text())
    assert account["estimated_usd"] == .0105
    assert account["unpriced_calls"] == 1
    assert [c["role"] for c in account["calls"]] == ["implementor", "reviewer"]
    assert account["api_wall_s"] >= 0


@pytest.mark.parametrize("model,effort", [("claude-haiku-5-5", None), ("claude-sonnet-5-5", "high"),
                                         ("claude-opus-5-5", "high")])
def test_sweep_clones_identical_seeds_routes_arms_and_keeps_partial_results(tmp_path, monkeypatch, model, effort):
    from PIL import Image
    root = tmp_path / "seed"
    root.mkdir()
    (root / "jig.json").write_text(json.dumps({"name": "test", "model": {}}))
    (root / "brief.md").write_text("Test model")
    (root / "refs").mkdir()
    Image.new("RGB", (8, 8), "white").save(root / "refs" / "front.png")
    seen = {}
    def backend(selected):
        assert selected == model
        return SimpleNamespace(timeout_s=600)
    monkeypatch.setattr(gauntlet_sweep.gauntlet, "backend", backend)
    def chat_for(api, **kwargs):
        assert api.timeout_s == 1800
        assert kwargs["idle_s"] == 900
        assert kwargs["max_tokens"] == gauntlet_sweep.gauntlet.MAX_TOKENS     # the compatibility endpoint's, unchanged
        return lambda *a: ({"content": "ok"}, {"prompt_tokens": 10, "completion_tokens": 5})
    monkeypatch.setattr(agent, "chat_for", chat_for)
    if model == "claude-opus-5-5":
        from mesh_jig import anthropic_sweep
        def native_chat_for(api, budget, label, **kwargs):
            assert budget.limit == 44
            assert label in gauntlet_sweep.ARMS
            # the native transport gets the model's whole reply: at 65,536 a visual run's fourth round ended on a
            # write cut at the cap
            assert kwargs["max_tokens"] == anthropic_sweep.MAX_OUTPUT_TOKENS == 128_000
            return lambda *a: ({"content": "ok"}, {"prompt_tokens": 10, "completion_tokens": 5})
        monkeypatch.setattr(anthropic_sweep, "chat_for", native_chat_for)
        def wrong_transport(*a, **kw):
            pytest.fail("Opus high must use the native transport")
        monkeypatch.setattr(agent, "chat_for", wrong_transport)

    cap = 128_000 if model == "claude-opus-5-5" else gauntlet_sweep.gauntlet.MAX_TOKENS

    def fake_run(proj, chat, **kwargs):
        assert kwargs["model"] == model
        assert kwargs.get("max_tokens", cap) == cap      # the visual arms are told the cap their replies really have
        seen[proj.root.name] = kwargs.get("checks", "control")
        assert not proj.attempts_dir.exists()
        chat([], [{}])
        if proj.root.name == "checks-both":
            raise RuntimeError("one arm failed")
        attempt = proj.attempts_dir / "a01"
        attempt.mkdir(parents=True)
        (attempt / "eval.json").write_text(json.dumps({"ok": True, "numeric": {"numeric_score": .5}}))
        common = {"done": True, "wall_s": 1.0}
        if proj.root.name == "mesh-jig":
            return agent.Report(**common, evaluations=1, attempts=[{"name": "a01", "ok": True}])
        return {**common, "rounds": [{"name": "a01", "calls_end": 1, "elapsed_s": 1, "wall_s": 1}]}

    monkeypatch.setattr(gauntlet_sweep.gauntlet, "run", fake_run)
    monkeypatch.setattr(agent, "run", fake_run)
    out = tmp_path / "sweep"
    result = gauntlet_sweep.sweep(project.load(root), out, iterations=1, workers=2, model=model, effort=effort,
                                  idle_s=900, call_timeout_s=1800,
                                  budget_usd=44 if model == "claude-opus-5-5" else None)
    assert result["idle_s"] == 900 and result["call_timeout_s"] == 1800
    assert result["max_tokens"] == cap
    assert result["model"] == model
    assert result["effort"] == (effort or "provider default")
    assert result["status"] == "incomplete"
    assert seen == {"gauntlet": "none", "mesh-jig": "control", "checks-builder": "builder", "checks-both": "both"}
    assert len(result["arms"]["mesh-jig"]["trajectory"]) == 1
    assert result["arms"]["checks-both"]["error"] == "one arm failed"
    assert len(result["seed_sha256"]) == 3
    for arm in seen:
        assert (out / arm / "brief.md").read_bytes() == (root / "brief.md").read_bytes()
        assert (out / arm / "accounting.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        gauntlet_sweep.sweep(project.load(root), out, model=model, effort=effort,
                             budget_usd=44 if model == "claude-opus-5-5" else None)


def test_sonnet_pricing_has_no_haiku_long_context_multiplier():
    model = "claude-sonnet-5-5"
    assert estimate_cost({"prompt_tokens": 100000, "completion_tokens": 1000}, model) == .21
    assert estimate_cost({"prompt_tokens": 100001, "completion_tokens": 1000}, model) == .210002
    assert estimate_cost({"prompt_tokens": 100000, "completion_tokens": 1000,
                          "prompt_tokens_details": {"cached_tokens": 50000}}, model) == .115
    assert estimate_cost({"prompt_tokens": 10, "completion_tokens": 10}, "unknown") is None


def test_explicit_effort_cannot_be_silently_ignored():
    assert "default high" in gauntlet_sweep.effort_control("claude-sonnet-5-5", "high")
    with pytest.raises(ValueError, match="explicit effort"):
        gauntlet_sweep.effort_control("claude-haiku-5-5", "high")
    with pytest.raises(ValueError, match="explicit effort"):
        gauntlet_sweep.effort_control("claude-sonnet-5-5", "low")
