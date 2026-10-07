"""The pairwise judge: swap mapping, ties, position bias, the interval, and the composite the model is shown.

The swap mapping is the part worth testing hardest - a sign error there silently inverts every result, and the
live numbers would still look plausible.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
from PIL import Image

from mesh_jig import judge, llm


def _png(path, colour, size=(80, 60)):
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[:, :] = (221, 219, 214)                      # the render ground, so the silhouette crop finds a subject
    a[10:size[1] - 10, 10:size[0] - 10] = colour
    Image.fromarray(a).save(path)
    return str(path)


def _sample(verdicts, swapped=False):
    return {"views": {v: {"closer": c, "why": ""} for v, c in verdicts.items()}, "swapped": swapped}


def test_unswapped_a_verdict_is_a_win_for_the_first_candidate():
    t = judge.tally([_sample({"hero": "A"})], ["hero"])
    assert t["win_rate"] == 1.0 and t["wins"] == 1 and t["losses"] == 0


def test_swapped_a_verdict_is_a_LOSS_for_the_first_candidate():
    """When swapped, the first candidate sat in slot B, so 'A is closer' means the OTHER one won."""
    t = judge.tally([_sample({"hero": "A"}, swapped=True)], ["hero"])
    assert t["win_rate"] == 0.0 and t["wins"] == 0 and t["losses"] == 1


def test_swapped_b_verdict_is_a_win_for_the_first_candidate():
    t = judge.tally([_sample({"hero": "B"}, swapped=True)], ["hero"])
    assert t["win_rate"] == 1.0 and t["wins"] == 1


def test_a_pure_slot_preference_cancels_to_a_draw():
    """A judge that always says 'A', shown both orders equally, must come out 50/50 - that is the whole point."""
    t = judge.tally([_sample({"hero": "A"}), _sample({"hero": "A"}, swapped=True)], ["hero"])
    assert t["win_rate"] == 0.5
    assert t["position_bias"] == 1.0          # and the bias itself is still reported, not hidden


def test_ties_count_half_to_each_side():
    t = judge.tally([_sample({"hero": "TIE"}), _sample({"hero": "TIE"})], ["hero"])
    assert t["win_rate"] == 0.5 and t["ties"] == 2 and t["wins"] == 0


def test_ties_do_not_count_toward_position_bias():
    t = judge.tally([_sample({"hero": "TIE"}), _sample({"hero": "A"})], ["hero"])
    assert t["position_bias"] == 1.0          # one decided judgement, slot A won it


def test_every_view_is_its_own_judgement():
    t = judge.tally([_sample({"hero": "A", "side": "B", "top": "TIE"})], ["hero", "side", "top"])
    assert t["judgements"] == 3
    assert t["views"] == {"hero": 1.0, "side": 0.0, "top": 0.5}
    assert t["win_rate"] == 0.5               # one win, one loss, one tie


def test_margin_is_symmetric_around_a_draw():
    assert judge.tally([_sample({"hero": "A"})], ["hero"])["margin"] == 1.0
    assert judge.tally([_sample({"hero": "B"})], ["hero"])["margin"] == -1.0
    assert judge.tally([_sample({"hero": "TIE"})], ["hero"])["margin"] == 0.0


def test_parse_rejects_a_missing_view():
    with pytest.raises(ValueError, match="no verdict for view"):
        judge.parse({"views": {"hero": {"closer": "A"}}}, ["hero", "side"])


def test_parse_rejects_a_verdict_that_is_not_a_b_or_tie():
    with pytest.raises(ValueError, match="not A, B or tie"):
        judge.parse({"views": {"hero": {"closer": "the left one"}}}, ["hero"])


def test_parse_accepts_lowercase_and_keeps_the_reason():
    out = judge.parse({"views": {"Hero": {"closer": "b", "why": "wings too short"}}}, ["hero"])
    assert out["views"]["hero"] == {"closer": "B", "why": "wings too short"}


def test_composite_puts_the_reference_between_the_candidates(tmp_path):
    ref = _png(tmp_path / "r.png", (200, 40, 40))
    a = _png(tmp_path / "a.png", (40, 200, 40))
    b = _png(tmp_path / "b.png", (40, 40, 200))
    im = judge.composite(ref, a, b, "hero", height=60)
    assert im.height == 60 + 28
    assert im.width > 3 * 60                     # three panels side by side, not one
    thirds = [np.asarray(im.crop((i * im.width // 3, 30, (i + 1) * im.width // 3, im.height)), float).reshape(-1, 3)
              for i in range(3)]
    dominant = [t[np.abs(t - np.array([221, 219, 214])).sum(1) > 60].mean(0) for t in thirds]
    assert dominant[0][1] > dominant[0][0]       # A is green-dominant, on the left
    assert dominant[1][0] > dominant[1][2]       # the REFERENCE is red-dominant, in the middle
    assert dominant[2][2] > dominant[2][0]       # B is blue-dominant, on the right


def _backend():
    return llm.Backend(name="fake", url="http://fake", model="fake-1")


def test_compare_runs_both_orders_and_cancels_a_slot_a_judge(tmp_path):
    """End to end with a fake model that always answers 'A': the aggregate must be a draw."""
    ref = {"hero": _png(tmp_path / "r.png", (200, 40, 40))}
    a = {"hero": _png(tmp_path / "a.png", (40, 200, 40))}
    b = {"hero": _png(tmp_path / "b.png", (40, 40, 200))}
    seen = []

    def fake_chat(backend, msgs, **kw):
        seen.append(msgs)
        return {"text": '{"views": {"hero": {"why": "x", "closer": "A"}}}', "usage": {"total_tokens": 1}}

    res = judge.compare(_backend(), ref, a, b, "", tmp_path / "out", samples=8, chat_fn=fake_chat,
                        encoder=lambda p: "data:,")
    assert len(seen) == 8
    assert res["win_rate"] == 0.5 and res["position_bias"] == 1.0 and res["samples"] == 8
    assert not judge.better(res)
    assert "not decisive" in judge.summary(res, "a01", "a00")


def test_a_reply_that_is_not_the_asked_json_is_retried_then_dropped(tmp_path):
    ref = {"hero": _png(tmp_path / "r.png", (200, 40, 40))}
    a = {"hero": _png(tmp_path / "a.png", (40, 200, 40))}
    replies = iter(["I think A", '{"views": {"hero": {"closer": "B", "why": "y"}}}'] * 4)
    res = judge.compare(_backend(), ref, a, a, "", tmp_path / "out", samples=4,
                        chat_fn=lambda *a, **k: {"text": next(replies), "usage": {}}, encoder=lambda p: "data:,")
    assert res["samples"] == 4 and res["errors"] == [], "each bad first reply was recovered by its retry"
    with pytest.raises(ValueError, match="every sample failed"):
        judge.compare(_backend(), ref, a, a, "", tmp_path / "out2", samples=2,
                      chat_fn=lambda *a, **k: {"text": "no json here", "usage": {}}, encoder=lambda p: "data:,")
    with pytest.raises(ValueError, match="no view is present"):
        judge.compare(_backend(), ref, {"top": a["hero"]}, a, "", tmp_path / "out3", chat_fn=None)


def test_a_draw_is_not_decisive_however_many_judgements():
    many = [_sample({"hero": "A"}), _sample({"hero": "A"}, swapped=True)] * 40
    t = judge.tally(many, ["hero"])
    assert t["win_rate"] == 0.5 and t["decisive"] is False
    assert t["ci"][0] < 0.5 < t["ci"][1]


def test_two_calls_are_never_decisive_however_one_sided():
    """Two calls agreeing has zero between-call spread; the interval must not read that as certainty."""
    t = judge.tally([_sample({"hero": "A", "side": "A"}), _sample({"hero": "B", "side": "B"}, swapped=True)],
                    ["hero", "side"])
    assert t["win_rate"] == 1.0 and t["calls"] == 2
    assert t["decisive"] is False


def test_a_consistent_result_over_many_calls_is_decisive():
    many = [_sample({"hero": "A", "side": "A"}), _sample({"hero": "B", "side": "B"}, swapped=True)] * 20
    t = judge.tally(many, ["hero", "side"])
    assert t["win_rate"] == 1.0 and t["decisive"] is True and t["ci"][0] > 0.5


def test_a_lean_over_few_calls_is_not_decisive():
    """6 of 10 calls one way is a lean, not a result, and must not be reported as one."""
    lop = [_sample({"hero": "A", "side": "A"})] * 6 + [_sample({"hero": "B", "side": "B"})] * 4
    t = judge.tally(lop, ["hero", "side"])
    assert t["win_rate"] == 0.6 and t["decisive"] is False, t


def test_the_same_lean_over_many_calls_does_become_decisive():
    lop = ([_sample({"hero": "A", "side": "A"})] * 6 + [_sample({"hero": "B", "side": "B"})] * 4) * 10
    t = judge.tally(lop, ["hero", "side"])
    assert t["win_rate"] == 0.6 and t["decisive"] is True, t


def test_the_interval_narrows_as_calls_accumulate():
    def width(reps):
        t = judge.tally([_sample({"hero": "A"}), _sample({"hero": "B"})] * reps, ["hero"])
        return t["ci"][1] - t["ci"][0]
    assert width(40) < width(10) < width(2)


def test_interval_refuses_to_narrow_on_a_single_call():
    assert judge.interval([]) == (0.0, 1.0)
    assert judge.interval([1.0]) == (0.0, 1.0)


def test_correlated_views_do_not_fake_precision():
    """Four views that always agree carry one call's worth of evidence, not four."""
    agree = [_sample({"a": "A", "b": "A", "c": "A", "d": "A"}),
             _sample({"a": "B", "b": "B", "c": "B", "d": "B"})] * 6
    split = [_sample({"a": "A", "b": "B", "c": "A", "d": "B"})] * 12
    wide = judge.tally(agree, ["a", "b", "c", "d"])
    narrow = judge.tally(split, ["a", "b", "c", "d"])
    assert wide["judgements"] == narrow["judgements"] == 48
    assert (wide["ci"][1] - wide["ci"][0]) > (narrow["ci"][1] - narrow["ci"][0])


@pytest.mark.parametrize("res, promoted", [
    ({"decisive": True, "win_rate": 0.9}, True),
    ({"decisive": False, "win_rate": 0.72}, False),      # a win rate inside its own interval is a lean, not a result
    ({"decisive": True, "win_rate": 0.1}, False),        # decisive is not enough: the candidate has to be the winner
])
def test_only_a_decisive_win_promotes(res, promoted):
    assert judge.better(res) is promoted


# --- the backend ------------------------------------------------------------------------------------------

def test_the_backend_comes_from_the_environment_and_refuses_a_hosted_endpoint_without_a_key():
    with pytest.raises(llm.LLMError, match="needs an API key"):
        llm.resolve_backend({}, load_file=False)
    b = llm.resolve_backend({"OPENROUTER_API_KEY": "k"}, load_file=False)
    assert (b.name, b.url, b.model, b.key) == ("openrouter", llm.DEFAULT_URL, llm.DEFAULT_MODEL, "k")
    assert llm.reasoning_extra(b) == {"reasoning": {"effort": "low"}}
    local = llm.resolve_backend({"MESH_JIG_JUDGE_URL": "http://localhost:8080/v1/", "MESH_JIG_JUDGE_MODEL": "m"}, load_file=False)
    assert (local.name, local.url, local.key) == ("local", "http://localhost:8080/v1", None)
    assert llm.reasoning_extra(local) == {}
    other = llm.resolve_backend({"MESH_JIG_JUDGE_URL": "https://api.example.com/v1", "MESH_JIG_JUDGE_KEY": "x",
                                 "OPENROUTER_API_KEY": "not this one"}, load_file=False)
    assert other.key == "x" and other.name == "openai-compatible"


def test_an_env_file_fills_only_unset_variables(tmp_path):
    f = tmp_path / "llm.env"
    f.write_text("# keys\nOPENROUTER_API_KEY=from-file\nEMPTY=\nKEPT='quoted'\n")
    env = {"OPENROUTER_API_KEY": "already-set"}
    assert llm.load_env_file(str(f), env) == ["KEPT"]
    assert env == {"OPENROUTER_API_KEY": "already-set", "KEPT": "quoted"}
    assert llm.load_env_file(str(tmp_path / "missing.env"), {}) == []
    assert llm.resolve_backend({"MESH_JIG_ENV_FILE": str(f)}).key == "from-file"


def test_the_default_env_file_is_dot_env_where_it_is_run_and_never_one_in_the_home(tmp_path, monkeypatch):
    home, work = tmp_path / "home", tmp_path / "work"
    (home / ".config").mkdir(parents=True)
    work.mkdir()
    (home / ".config" / "llm.env").write_text("OPENROUTER_API_KEY=from-home\n")
    (home / ".env").write_text("OPENROUTER_API_KEY=from-home\n")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.chdir(work)
    # conftest points llm's default at nothing; this is the module as it loads, with that home in place
    spec = importlib.util.spec_from_file_location("llm_as_loaded", llm.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "llm_as_loaded", fresh)
    spec.loader.exec_module(fresh)
    assert fresh.DEFAULT_ENV_FILE == ".env"
    assert fresh.load_env_file(env={}) == [], "a key file in the home directory is not looked for"
    (work / ".env").write_text("OPENROUTER_API_KEY=from-project\n")
    assert fresh.resolve_backend({}).key == "from-project"
    named = tmp_path / "named.env"
    named.write_text("OPENROUTER_API_KEY=from-named\n")
    assert fresh.resolve_backend({"MESH_JIG_ENV_FILE": str(named)}).key == "from-named", "the named file wins"


def test_chat_shapes_the_payload_and_reads_the_reply():
    sent = {}

    def poster(url, payload, key, timeout_s):
        sent.update(url=url, payload=payload, key=key)
        return {"choices": [{"message": {"content": "```json\n{\"a\": 1}\n```"}}], "usage": {"cost": 0.01}}

    out = llm.chat(llm.Backend("openrouter", "http://x/v1", "m", "k"), [{"role": "user", "content": "hi"}],
                   json_mode=True, extra={"reasoning": {"effort": "low"}}, poster=poster)
    assert sent["url"] == "http://x/v1/chat/completions" and sent["key"] == "k"
    assert sent["payload"]["response_format"] == {"type": "json_object"} and sent["payload"]["max_tokens"] == 4096
    assert llm.extract_json(out["text"]) == {"a": 1} and out["usage"]["cost"] == 0.01
    assert "temperature" not in llm.shape_payload(llm.Backend("o", "u", "gpt-5.5"), [], 0.3, 10)
    with pytest.raises(llm.LLMError, match="malformed"):
        llm.chat(llm.Backend("o", "u", "m"), [], poster=lambda *a: {"error": "nope"})
    with pytest.raises(llm.LLMError, match="no JSON object"):
        llm.extract_json("sorry")
