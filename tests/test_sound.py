"""The sound stage: cues read from a rigged GLB (sound_cues), a sound script held to its contract (sound), the
measurer (sound_measure) and the agent workflow over a scripted chat (sound_agent). Fixtures are GLBs and scripts
written here; no model call."""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from glbkit import box, write_glb
from mesh_jig import cli, rig, sound, sound_agent, sound_cues, sound_measure

RATE = sound.RATE


def part(name, size, centre):
    (sx, sy, sz), (cx, cy, cz) = size, centre
    return (name, box((sx, sz, sy), (cx, cz, -cy)), (0, 0, 0))


FIGURE = [part("Body", (0.4, 0.2, 0.4), (0, 0, 0.6)), part("LegL", (0.1, 0.1, 0.4), (-0.1, 0, 0.2)),
          part("LegR", (0.1, 0.1, 0.4), (0.1, 0, 0.2))]
# step: each leg lifts 50 mm in its own half of the cycle (L in the first, R in the second), so each foot comes down
# at the end of its half. hop: a crouch to 0.15 s, then the whole figure leaves the ground from 0.3 s to 0.7 s.
SCRIPT = '''import math
def skeleton(model):
    return [dict(name="root", head=(0,0,0), tail=(0,0,.1), parent=None),
            dict(name="body", head=(0,0,.4), tail=(0,0,.8), parent="root"),
            dict(name="leg.L", head=(-.1,0,.4), tail=(-.1,0,0), parent="body"),
            dict(name="leg.R", head=(.1,0,.4), tail=(.1,0,0), parent="body")]
def weights(name, position, piece):
    return {"Body": "body", "LegL": "leg.L", "LegR": "leg.R"}[name]
def pose(clip, t):
    if clip == "step":
        s = math.sin(2 * math.pi * t)
        return {"leg.L": {"move": (0, 0, .05 * max(s, 0))}, "leg.R": {"move": (0, 0, .05 * max(-s, 0))}}
    if t < .3:
        return {"body": {"move": (0, 0, -.05 * math.sin(math.pi * t / .3))}}
    if t < .7:
        return {"root": {"move": (0, 0, .3 * math.sin(math.pi * (t - .3) / .4))}}
    return {}
'''
SPEC = {"fps": 60, "bones": {"required": ["root", "body", "leg.L", "leg.R"]},
        "clips": {"step": {"seconds": 1, "loop": True}, "hop": {"seconds": 1, "loop": False}}}


@pytest.fixture(scope="module")
def animated(tmp_path_factory):
    d = tmp_path_factory.mktemp("rig")
    source = write_glb(d / "source.glb", FIGURE)
    (d / "rig.py").write_text(SCRIPT)
    out = d / "animated.glb"
    rig.build(source, d / "rig.py", SPEC, out)
    return out


def test_cues_are_where_the_feet_meet_the_ground_and_the_figure_leaves_it(animated):
    cues = sound_cues.read(animated, {"step": None, "hop": False})
    step, hop = cues["step"], cues["hop"]
    assert step["loop"] is True and hop["loop"] is False
    # a foot 50 mm up comes down within 5 mm of the ground: sin(2 pi t) = 0.1 short of the half
    lands = 0.5 - math.asin(0.1) / (2 * math.pi)
    assert [s for _, s in step["footfalls"]] == ["L", "R"]
    assert step["footfalls"][0][0] == pytest.approx(lands, abs=0.005)
    assert step["footfalls"][1][0] == pytest.approx(lands + 0.5, abs=0.005)
    assert "takeoff" not in step
    rise = 0.4 * math.asin(0.005 / 0.3) / math.pi
    assert hop["takeoff"] == pytest.approx(0.3 + rise, abs=0.005)
    assert hop["landing"] == pytest.approx(0.7 - rise, abs=0.005)
    assert hop["apex"] == pytest.approx(0.5, abs=0.01)
    assert hop["crouch"] == pytest.approx(0.15, abs=0.01)
    assert "footfalls 0.48" in sound_cues.text(cues)


def test_a_swing_foot_that_only_scuffs_the_ground_is_not_a_footfall():
    times = np.arange(20) / 20
    low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 2, 2, 30, 30, 30, 30, 30, 30, 30], float)
    down = low <= 5
    # down at 0.55 for two frames (a scuff), and down again from the wrap round 0.95 -> 0.00 for nine
    assert sound_cues._crossings(down, times, low, 5, True) == [pytest.approx(0.545, abs=0.003), pytest.approx(0.992, abs=0.003)]
    assert sound_cues._crossings(down, times, low, 5, True, stance=0.4) == [pytest.approx(0.992, abs=0.003)]


def _burst(n, at, length=0.03, seed=0):
    x = np.zeros(n)
    i = int(at * RATE)
    k = np.arange(int(length * RATE))
    b = np.random.default_rng(seed).standard_normal(len(k)) * np.exp(-k / (0.006 * RATE))
    x[i:i + len(k)] += b[:max(0, min(len(k), n - i))]
    return x


def test_onsets_are_found_where_sounds_start_wrapped_round_for_a_loop():
    n = RATE
    x = _burst(n, 0.2) + _burst(n, 0.6, seed=1) * 0.6
    found = sound.onsets(x)
    strong = [t for t, w in found if w >= 0.25]
    assert len(strong) == 2
    assert strong[0] == pytest.approx(0.2, abs=0.006) and strong[1] == pytest.approx(0.6, abs=0.006)
    # a burst across a loop's seam starts at its end, not at its start
    seam = np.roll(_burst(n, 0.5), int(0.49 * RATE))         # starts at 0.99 s and runs over the end
    looped = [t for t, w in sound.onsets(seam, loop=True) if w >= 0.25]
    assert any(abs(t - 0.99) < 0.006 for t in looped) and not any(t < 0.05 for t in looped)
    # read as a sound played once, what runs over the end is a second sound starting from silence at 0
    assert any(t < 0.01 for t, w in sound.onsets(seam, loop=False) if w >= 0.25)


GOOD = '''import math
import numpy as np

R = 44100


def burst(n, at, rng, length=0.04):
    x = np.zeros(n)
    k = np.arange(int(length * R))
    b = rng.standard_normal(len(k)) * np.exp(-k / (0.006 * R))
    for j, v in zip((int(at * R) + k) % n, b):
        x[j] += v
    return x


def shape(x):
    x = x - x.mean()
    x = 0.9 * x / np.abs(x).max()
    f = int(0.005 * R)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    return x


def sounds(cues):
    rng = np.random.default_rng(7)
    t = np.arange(int(0.5 * R)) / R
    out = {"call": shape(np.sin(2 * math.pi * (500 + 300 * t) * t) * np.sin(math.pi * t / 0.5))}
    c = cues["steps"]
    n = int(round(c["seconds"] * R))
    x = sum(burst(n, at, rng) for at, side in c["footfalls"])
    out["steps"] = 0.9 * (x - x.mean()) / np.abs(x - x.mean()).max()
    c = cues["leap"]
    n = int(round(c["seconds"] * R))
    x = 0.3 * np.sin(2 * math.pi * 110 * np.arange(n) / R) + burst(n, c["takeoff"], rng) + burst(n, c["landing"], rng)
    out["leap"] = shape(x)
    return out
'''


@pytest.fixture
def sound_dir(tmp_path, animated):
    (tmp_path / "animated.glb").write_bytes(animated.read_bytes())
    (tmp_path / "brief.md").write_text("Three test sounds.")
    (tmp_path / "sound.json").write_text(json.dumps({"sounds": {
        "call": {"seconds": [0.3, 0.8]},
        "steps": {"clip": "step", "on": "footfalls"},
        "leap": {"clip": "hop", "loop": False, "on": ["takeoff", "landing"]}}}))
    return tmp_path


def _script(tmp_path, text, name="sounds.py"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_a_script_that_fits_its_clips_is_in_contract_and_its_runs_agree(sound_dir, tmp_path):
    keep = tmp_path / "kept"
    res = sound.check(sound_dir, _script(tmp_path, GOOD), keep=keep)
    assert res.ok, sound.text(res)
    assert sorted(p.name for p in keep.glob("*.wav")) == ["call.wav", "leap.wav", "steps.wav"]
    assert res.sounds["steps"]["stats"]["samples"] == RATE
    assert [m["cue"] for m in res.sounds["steps"]["cues"]] == ["footfall L", "footfall R"]
    assert all(abs(m["off_ms"]) <= 6 for s in ("steps", "leap") for m in res.sounds[s]["cues"])
    assert "IN CONTRACT" in sound.text(res)
    assert res.cues["steps"]["footfalls"] and res.cues["call"]["seconds"] is None


def test_a_sound_off_its_cue_wrong_in_length_or_unfaded_is_out_of_contract(sound_dir, tmp_path):
    late = GOOD.replace('burst(n, c["landing"], rng)', 'burst(n, c["landing"] - 0.08, rng)')
    res = sound.check(sound_dir, _script(tmp_path, late))
    assert any("landing at" in p and "nothing starts within 35 ms" in p for p in res.sounds["leap"]["problems"])
    short = GOOD.replace('n = int(round(c["seconds"] * R))\n    x = sum', 'n = int(round(c["seconds"] * R)) - 10\n    x = sum')
    res = sound.check(sound_dir, _script(tmp_path, short, "short.py"))
    assert any("44090 samples" in p and "exactly 44100" in p for p in res.sounds["steps"]["problems"])
    unfaded = GOOD.replace("x[-f:] *= np.linspace(1, 0, f)", "x[-1] = 0.5")
    res = sound.check(sound_dir, _script(tmp_path, unfaded, "unfaded.py"))
    assert any("the last sample is +0.500" in p for p in res.sounds["call"]["problems"])
    assert not res.ok and "OUT OF CONTRACT" in sound.text(res)


def test_a_script_is_refused_before_it_runs_and_a_run_that_differs_is_said(sound_dir, tmp_path):
    res = sound.check(sound_dir, _script(tmp_path, "import os\n" + GOOD))
    assert res.problems and all(p.startswith("refused before running") for p in res.problems)
    assert any("'os'" in p for p in res.problems) and not res.sounds
    res = sound.check(sound_dir, _script(tmp_path, GOOD.replace("default_rng(7)", "default_rng()"), "unseeded.py"))
    assert any("differs between two runs" in p for p in res.problems)
    res = sound.check(sound_dir, _script(tmp_path, GOOD.replace('out = {"call"', 'out = {"cal"'), "typo.py"))
    assert "call is missing" in res.problems and any("'cal' is not a sound of the contract" in p for p in res.problems)
    res = sound.check(sound_dir, _script(tmp_path, GOOD.replace("def sounds(cues):", "def sounds(cues):\n    1 / 0"), "err.py"))
    assert any("ZeroDivisionError" in p and "line" in p for p in res.problems)


def test_the_contract_is_said_with_the_clips_own_numbers(sound_dir, animated):
    spec = sound.load(sound_dir)
    text = sound.contract_text(spec, sound.cues_for(sound_dir, spec))
    assert "steps (required): goes with the clip `step`; exactly 44100 samples (1 s), and it loops" in text
    assert "leap (required): goes with the clip `hop`; exactly 44100 samples (1 s), played once" in text
    assert "call (required): 0.30 to 0.80 s, played once" in text
    (sound_dir / "sound.json").write_text(json.dumps({"sounds": {"x": {"clip": "step", "on": "toes"}}}))
    with pytest.raises(sound.ContractError, match="no cue 'toes'"):
        sound.load(sound_dir)


def test_the_measurer_reads_a_glide_as_a_rising_pitch_and_noise_as_unvoiced(tmp_path):
    t = np.arange(int(0.6 * RATE)) / RATE
    f = 400 + 400 * t / 0.6
    glide = 0.8 * np.sin(2 * np.pi * np.cumsum(f) / RATE)
    d = sound_measure.describe(glide, RATE)
    f0 = d["f0_hz_at_10_30_50_70_90_percent"]
    assert all(a < b for a, b in zip(f0, f0[1:])) and f0[0] == pytest.approx(440, rel=0.05)
    noise = np.random.default_rng(0).standard_normal(len(t)) * 0.3
    assert sound_measure.describe(noise, RATE)["voiced_fraction"] < 0.2
    path = sound_measure.write_sheet([("glide", RATE, glide)], tmp_path / "s.png", {"glide": [(0.3, "takeoff")]})
    assert path.is_file()


def test_the_sound_agent_writes_builds_and_delivers_over_a_scripted_chat(sound_dir):
    replies = iter([
        {"tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
            {"path": "attempts/a01/sounds.py", "text": GOOD.replace("default_rng(7)", "default_rng()")})}}]},
        {"tool_calls": [{"id": "2", "function": {"name": "build", "arguments": json.dumps({"script": "attempts/a01/sounds.py"})}}]},
        {"tool_calls": [{"id": "3", "function": {"name": "write_file", "arguments": json.dumps(
            {"path": "attempts/a02/sounds.py", "text": GOOD})}}]},
        {"tool_calls": [{"id": "4", "function": {"name": "build", "arguments": json.dumps({"script": "attempts/a02/sounds.py"})}}]},
        {"tool_calls": [{"id": "5", "function": {"name": "edit_file", "arguments": json.dumps(
            {"path": "attempts/a02/sounds.py", "edits": [{"old": "default_rng(7)", "new": "default_rng(8)"}]})}}]},
        {"tool_calls": [{"id": "6", "function": {"name": "done", "arguments": json.dumps({"best": "a02", "summary": "ok"})}}]},
    ])
    seen = []

    def chat(messages, tools):
        seen.append(json.loads(json.dumps(messages[-1])))      # as sent: old pictures are pruned in place later
        return {"role": "assistant", **next(replies)}, {"prompt_tokens": 10, "completion_tokens": 5}

    rep = sound_agent.run(sound_dir, chat, builds=1, out=sound_dir / "out")
    assert rep["done"] and rep["final"] == "a02" and rep["builds"] == 1 and rep["failed_builds"] == 1
    said = [json.loads(line) for line in (sound_dir / "out" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    tool_results = [m for m in said if m.get("role") == "tool"]
    assert "OUT OF CONTRACT" in tool_results[1]["content"] and "differs between two runs" in tool_results[1]["content"]
    assert "BUILT AND IN CONTRACT" in tool_results[3]["content"] and "WHAT EACH SOUND MEASURES AS" in tool_results[3]["content"]
    assert "built in contract and stays as it was built" in tool_results[4]["content"]
    pictures = [m for m in seen if m.get("role") == "user" and isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])]
    builds = [m for m in pictures if "spectrograms (60 Hz" in m["content"][0].get("text", "")]
    assert builds and builds[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "example_call, drawn as your builds will be drawn" in json.dumps(pictures[0]) and (sound_dir / "rest" / "example_call.png").is_file()
    assert sorted(p.name for p in (sound_dir / "delivered").iterdir()) == ["call.wav", "leap.wav", "sounds.py", "steps.wav"]
    assert json.loads((sound_dir / "out" / "report.json").read_text())["workflow"] == "sound"


def test_sound_commands(sound_dir, tmp_path, capsys):
    assert cli.main(["sound-cues", str(sound_dir / "animated.glb")]) == 0
    assert "step: 1 s, loops; footfalls" in capsys.readouterr().out
    script = _script(tmp_path, GOOD)
    assert cli.main(["sound-check", str(sound_dir), "--script", str(script), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "IN CONTRACT" in out and "pitch" in out and (tmp_path / "o" / "spectrograms.png").is_file()


def test_a_sound_is_read_against_what_it_should_measure_as_and_the_opening_says_what_is_wanted(sound_dir):
    t = np.arange(int(0.6 * RATE)) / RATE
    f = 500 + 300 * np.sin(np.pi * t / 0.6)                     # up and back down
    d = sound_measure.describe(0.8 * np.sin(2 * np.pi * np.cumsum(f) / RATE), RATE)
    assert sound_measure.pitch_shape(d) == "rise-fall"
    rows = sound_measure.against(d, {"pitch_hz": [500, 900], "pitch_shape": "fall", "periodicity": [0.5, 0.9]})
    assert [ok for ok, _ in rows] == [True, False, False]
    assert "pitch shape rise-fall" in rows[1][1] and "fall wanted" in rows[1][1]
    text = sound_measure.format_against("call", d, {"pitch_hz": [500, 900], "pitch_shape": "fall"})
    assert text.startswith("call: 1 of 2 as it should measure") and "OFF pitch shape" in text
    spec = json.loads((sound_dir / "sound.json").read_text())
    spec["sounds"]["call"]["expect"] = {"pitch_hz": [500, 900], "pitch_shape": "rise-fall"}
    (sound_dir / "sound.json").write_text(json.dumps(spec))
    opened = sound_agent.opening(sound_dir, sound.load(sound_dir), 3)[0]["text"]
    assert "WHAT EACH SHOULD MEASURE AS" in opened and "- call: pitch (median of the voiced points) 500 to 900 Hz; pitch shape rise-fall" in opened
    spec["sounds"]["call"]["expect"] = {"pitch_shape": "wobble"}
    (sound_dir / "sound.json").write_text(json.dumps(spec))
    with pytest.raises(sound.ContractError, match="pitch_shape is one of"):
        sound.load(sound_dir)


def test_the_recipe_handed_to_an_author_passes_the_script_check_and_makes_a_call():
    from mesh_jig import script_check, sound_recipe
    assert script_check.check_script(sound_recipe.RECIPE, required=(), allowed_imports=sound.ALLOWED_IMPORTS) == []
    ns = {}
    exec(sound_recipe.RECIPE, ns)
    call = ns["example_call"](np.random.default_rng(1))
    assert abs(np.abs(call).max() - 0.9) < 1e-9 and abs(call[0]) < 0.02 and abs(call[-1]) < 0.02
    d = sound_measure.describe(call, RATE)
    assert sound_measure.pitch_shape(d) == "rise-fall" and d["voiced_fraction"] > 0.9
    loop = np.zeros(RATE)
    ns["place"](loop, ns["thud"](np.random.default_rng(2), 0.1, 300, 0.02), 0.97)
    assert np.abs(loop[:int(0.05 * RATE)]).max() > 0                          # what ran past the end wrapped round


def test_the_opening_hands_the_recipe_over_as_it_is(sound_dir):
    from mesh_jig import sound_recipe
    opened = sound_agent.opening(sound_dir, sound.load(sound_dir), 3)[0]["text"]
    assert "A RECIPE you may start from" in opened and sound_recipe.RECIPE.strip() in opened


def test_a_loop_whose_noise_runs_across_its_seam_is_not_a_click_but_a_step_there_is():
    spec, c = {"clip": "step", "on": []}, {"seconds": 1.0, "loop": True}
    rng = np.random.default_rng(0)
    x = np.zeros(RATE)
    k = np.arange(int(0.05 * RATE))
    burst = rng.standard_normal(len(k)) * np.exp(-k / (0.02 * RATE))
    x[(int(0.98 * RATE) + k) % RATE] += burst                # loud noise across the seam, wrapped round
    x = 0.9 * (x - x.mean()) / np.abs(x - x.mean()).max()
    assert abs(x[-1] - x[0]) > sound.SEAM                     # the old fixed limit called this a click
    assert not any("clicks" in p for p in sound.check_sound("steps", x, spec, c)["problems"])
    stepped = np.sin(2 * np.pi * 3 * np.arange(RATE) / RATE) * 0.5
    stepped[-200:] += 0.3                                     # quiet and smooth either side, a step at the seam
    assert any("clicks where it repeats" in p for p in sound.check_sound("steps", stepped, spec, c)["problems"])


def test_the_recipes_growl_pulses_and_its_break_up_keeps_the_length():
    from mesh_jig import sound_recipe
    ns = {}
    exec(sound_recipe.RECIPE, ns)
    g = ns["growl"](np.random.default_rng(1), 0.4)
    d = sound_measure.describe(ns["finish"](g), RATE)
    assert len(g) == int(0.4 * RATE) and d["flutter_depth"] > 0.3 and 20 <= d["flutter_hz"] <= 45
    call = ns["example_call"](np.random.default_rng(1))
    broken = ns["break_up"](np.random.default_rng(2), call)
    assert len(broken) == len(call) and not np.array_equal(broken, call)
    assert np.allclose(broken[:1000], call[:1000], atol=2.0 ** -11)   # its first grain is the sound, to 12 bits


def test_a_build_with_every_sound_as_it_should_measure_may_be_delivered_before_the_builds_are_used(sound_dir, monkeypatch):
    spec = json.loads((sound_dir / "sound.json").read_text())
    spec["sounds"]["call"]["expect"] = {"voiced": [0.5, 1.0]}
    (sound_dir / "sound.json").write_text(json.dumps(spec))
    replies = iter([
        {"tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
            {"path": "attempts/a01/sounds.py", "text": GOOD})}}]},
        {"tool_calls": [{"id": "2", "function": {"name": "build", "arguments": json.dumps({"script": "attempts/a01/sounds.py"})}}]},
        {"tool_calls": [{"id": "3", "function": {"name": "done", "arguments": json.dumps({"best": "a01", "summary": "ok"})}}]},
    ])
    script = list(replies)
    refused = iter(script)
    rep = sound_agent.run(sound_dir, lambda m, t: ({"role": "assistant", **next(refused)}, {}), builds=4, max_turns=20)
    assert not rep["done"] and rep["builds"] == 1                  # one build is too early to settle on (g11)
    monkeypatch.setattr(sound_agent, "SETTLE_AFTER", 1)
    shutil.rmtree(sound_dir / "attempts")
    replies = iter(script)
    rep = sound_agent.run(sound_dir, lambda m, t: ({"role": "assistant", **next(replies)}, {}), builds=4,
                          out=sound_dir / "out")
    assert rep["done"] and rep["final"] == "a01" and rep["builds"] == 1
    said = (sound_dir / "out" / "transcript.jsonl").read_text(encoding="utf-8")
    assert "Every sound of a01 measures as it should" in said
    assert sound_agent.settled("AGAINST WHAT EACH SHOULD MEASURE AS\ncall: 1 of 1\n  ok  x")
    assert not sound_agent.settled("AGAINST WHAT EACH SHOULD MEASURE AS\ncall: 0 of 1\n  OFF x")
    assert not sound_agent.settled("no expectations")




def test_a_run_the_harness_breaks_still_delivers_its_last_build(sound_dir):
    calls = iter([
        {"tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
            {"path": "attempts/a01/sounds.py", "text": GOOD})}}]},
        {"tool_calls": [{"id": "2", "function": {"name": "build", "arguments": json.dumps({"script": "attempts/a01/sounds.py"})}}]},
    ])

    def chat(messages, tools):
        try:
            return {"role": "assistant", **next(calls)}, {}
        except StopIteration:
            raise RuntimeError("the harness broke") from None

    rep = sound_agent.run(sound_dir, chat, builds=3)
    assert rep["final"] == "a01" and "RuntimeError: the harness broke" in rep["error"]
    assert (sound_dir / "delivered" / "call.wav").is_file()


def test_a_contract_naming_a_cue_its_clip_lacks_is_refused_before_a_script_runs(sound_dir):
    spec = json.loads((sound_dir / "sound.json").read_text())
    spec["sounds"]["steps"]["on"] = ["takeoff"]                # the stepping clip never leaves the ground
    (sound_dir / "sound.json").write_text(json.dumps(spec))
    with pytest.raises(sound.ContractError, match="steps starts sounds on takeoff, and the clip step has none"):
        sound.cues_for(sound_dir, sound.load(sound_dir))


def test_a_stretch_between_two_cues_is_read_against_its_own_expectations(sound_dir):
    spec = json.loads((sound_dir / "sound.json").read_text())
    spec["sounds"]["leap"]["between"] = {"takeoff-landing": {"voiced": [0.9, 1.0]}, "start-takeoff": {"centroid_hz": [50, 400]}}
    (sound_dir / "sound.json").write_text(json.dumps(spec))
    loaded = sound.load(sound_dir)
    cues = sound.cues_for(sound_dir, loaded)
    x = np.arange(RATE) / RATE
    part = sound.stretch(x, "takeoff-landing", cues["leap"])
    assert len(part) == int(round(cues["leap"]["landing"] * RATE)) - int(round(cues["leap"]["takeoff"] * RATE))
    assert part[0] == pytest.approx(cues["leap"]["takeoff"], abs=1 / RATE)
    opened = sound_agent.opening(sound_dir, loaded, 3)[0]["text"]
    assert "- leap from its takeoff to its landing: voiced share of the loud frames 0.9 to 1" in opened
    spec["sounds"]["leap"]["between"] = {"takeoff-toes": {}}
    (sound_dir / "sound.json").write_text(json.dumps(spec))
    with pytest.raises(sound.ContractError, match="a stretch is two of start, end"):
        sound.load(sound_dir)




def test_sound_off_reads_the_off_lines_by_sound():
    from mesh_jig import cli
    readings = "meow: 6 of 7 as it should measure\n  ok  pitch 600 Hz\n  OFF centroid 912 Hz, 1000 to 3000 wanted"
    assert cli.sound_off(readings) == ["meow: centroid 912 Hz, 1000 to 3000 wanted"]


