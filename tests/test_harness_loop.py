"""skill/mesh-jig-harness-loop: the blind sheets, the verdict rule, the streak and the remeasure, on projects measured
for real with a fake build (test_evaluate's ship: 40 mm wide is the reference, 70 mm is a worse model)."""
from __future__ import annotations

import json
import re
import runpy
import shlex
import shutil
from pathlib import Path

import pytest
from PIL import Image

from test_evaluate import fake_build, make_project, quick_render, run

SKILL = Path(__file__).resolve().parents[1] / "skill" / "mesh-jig-harness-loop"
gen = runpy.run_path(str(SKILL / "scripts" / "generation.py"))
Refused = gen["Refused"]


def make_run(base: Path, attempts: dict[str, float]) -> Path:
    """One builder's run under `base`: an attempt per name, each a ship of that span."""
    proj = make_project(base)
    for name, span in attempts.items():
        run(proj, base, name, build_fn=fake_build(span=span))
    return proj.root


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    """old1, old2: runs whose best attempt is the wide ship. new1, new2: runs that reached the reference's shape."""
    seed = make_run(tmp_path_factory.mktemp("seed"), {})
    root = tmp_path_factory.mktemp("loop")
    out = {}
    for name, attempts in (("old1", {"a01": 0.07}), ("old2", {"a01": 0.075, "a02": 0.07}),
                           ("new1", {"a01": 0.07, "a02": 0.04}), ("new2", {"a01": 0.04})):
        base = root / ("g00" if name.startswith("old") else "g01") / "runs" / name
        base.mkdir(parents=True)
        shutil.copytree(seed, base / "proj")
        proj = gen["project"].load(base / "proj")
        for attempt, span in attempts.items():
            run(proj, base, attempt, build_fn=fake_build(span=span))
        out[name] = base / "proj"
    out["root"] = root
    return out


def sides(runs):
    return [runs["old1"], runs["old2"]], [runs["new1"], runs["new2"]]


def test_a_run_delivers_its_best_measured_attempt_unless_one_is_named(runs):
    assert gen["pick"](runs["new1"])["attempt"] == "a02"
    named = gen["pick"](runs["new1"] / "attempts" / "a01")
    assert named["attempt"] == "a01" and named["numeric"] < gen["pick"](runs["new1"])["numeric"]
    with pytest.raises(Refused, match="neither a project"):
        gen["pick"](runs["new1"] / "refs")


def test_sheets_say_nothing_about_which_side_made_them_and_the_key_is_outside(runs, tmp_path):
    review = runs["root"] / "g01" / "review"
    key = gen["make_sheets"](review, *sides(runs))
    names = sorted(p.name for p in (review / "sheets").iterdir())
    assert names == ["R01.jpg", "R02.jpg", "R03.jpg", "R04.jpg", "REVIEW.md", "brief-ship.md"]
    assert json.loads((review / "key.json").read_text()) == key
    assert sorted(s["side"] for s in key["sheets"].values()) == ["candidate"] * 2 + ["incumbent"] * 2
    told = "".join((review / "sheets" / n).read_text(encoding="utf-8") for n in ("REVIEW.md", "brief-ship.md")).lower()
    assert not any(word in told for word in ("incumbent", "candidate", "old1", "new1", "g00", "g01", "key.json"))
    # reference row above, model row below, one column per view
    sheet = Image.open(review / "sheets" / "R01.jpg")
    assert sheet.height == 2 * (gen["LABEL"] + gen["HEIGHT"]) + gen["GAP"] and sheet.width > 4 * gen["GAP"]
    # the record names no machine: runs are relative to the review directory
    assert {s["run"] for s in key["sheets"].values()} == {"../../g00/runs/old1/proj", "../../g00/runs/old2/proj",
                                                         "../runs/new1/proj", "../runs/new2/proj"}
    assert str(runs["root"]) not in (review / "key.json").read_text()
    by_run = {s["run"]: s for s in key["sheets"].values()}
    assert by_run["../runs/new1/proj"]["attempt"] == "a02" and by_run["../../g00/runs/old2/proj"]["attempt"] == "a02"
    assert by_run["../runs/new1/proj"]["numeric"] > by_run["../../g00/runs/old1/proj"]["numeric"]


def test_the_same_runs_give_the_same_sheets_whatever_order_they_were_named_in(runs):
    old, new = sides(runs)
    first = gen["make_sheets"](runs["root"] / "g01" / "again", old, new)
    second = gen["make_sheets"](runs["root"] / "g01" / "third", old[::-1], new[::-1])
    assert first == second
    for rid in first["sheets"]:
        assert ((runs["root"] / "g01" / "again" / "sheets" / f"{rid}.jpg").read_bytes()
                == (runs["root"] / "g01" / "third" / "sheets" / f"{rid}.jpg").read_bytes())


def test_sheets_refuse_sides_that_cannot_be_compared(runs, tmp_path):
    old, new = sides(runs)
    with pytest.raises(Refused, match="same number, at least 2"):
        gen["make_sheets"](tmp_path / "one", old[:1], new[:1])
    with pytest.raises(Refused, match="same number, at least 2"):
        gen["make_sheets"](tmp_path / "uneven", old, new + [runs["new1"] / "attempts" / "a01"])
    with pytest.raises(Refused, match="given twice"):
        gen["make_sheets"](tmp_path / "twice", old, [runs["new1"], runs["new1"]])
    # another reference: the task changed, so the sides are not runs on one seed
    other = tmp_path / "other"
    shutil.copytree(runs["new2"], other)
    top = Image.open(other / "refs" / "top.png").convert("RGB")
    top.putpixel((0, 0), (1, 2, 3))
    top.save(other / "refs" / "top.png")
    with pytest.raises(Refused, match="same number"):
        gen["make_sheets"](tmp_path / "refs", old, [runs["new1"], other])
    assert not (tmp_path / "refs" / "key.json").exists()


def test_a_scored_review_is_not_written_over(runs, tmp_path):
    gen["make_sheets"](tmp_path / "review", *sides(runs))
    (tmp_path / "review" / "sheets" / "scores.json").write_text("{}")
    with pytest.raises(Refused, match="has been scored"):
        gen["make_sheets"](tmp_path / "review", *sides(runs))


def key_of(incumbent, candidate, reference="ship", start=1):
    """A key for (visual, numeric) pairs a side; the scores that go with it."""
    sheets, scores = {}, {}
    for side, pairs in (("incumbent", incumbent), ("candidate", candidate)):
        for visual, numeric in pairs:
            rid = f"R{start + len(sheets):02d}"
            sheets[rid] = {"side": side, "reference": reference, "numeric": numeric}
            scores[rid] = {"score": visual, "note": ""}
    return {"sheets": sheets}, scores


def decide(incumbent, candidate, **kw):
    return gen["decide"](*key_of([(v, 0.55) for v in incumbent], [(v, 0.55) for v in candidate]), **kw)


def test_a_clear_gain_is_an_improvement_and_one_under_the_margin_is_not():
    v = decide([20, 24, 22], [40, 44, 42])
    assert v["improved"] and v["visual"]["gain"] == 20.0 and v["reasons"] == []
    v = decide([20, 24, 22], [24, 28, 26])
    assert not v["improved"] and v["visual"]["gain"] == 4.0 and v["visual"]["needed"] == 5.0
    assert "under the 5.0 it needs" in v["reasons"][0]
    assert decide([20, 24, 22], [24, 28, 26], visual_margin=3)["improved"]


def test_a_gain_inside_the_runs_own_spread_is_not_an_improvement():
    v = decide([10, 40, 25], [20, 50, 35])                 # +10, but each side's runs are 15 apart
    assert not v["improved"] and v["visual"]["gain"] == 10.0
    assert v["visual"]["noise"] == pytest.approx(12.2, abs=0.05) and v["visual"]["needed"] == v["visual"]["noise"]
    six = decide([10, 40, 25] * 2, [22, 52, 37] * 2)       # the same spread over twice the runs is less noise
    assert six["visual"]["noise"] < 9 and six["improved"]


def test_models_that_look_better_and_measure_worse_are_not_an_improvement():
    key, scores = key_of([(20, 0.55), (24, 0.56)], [(40, 0.50), (44, 0.51)])
    v = gen["decide"](key, scores)
    assert not v["improved"] and v["numeric"]["change"] == -0.05 and "numeric score fell" in v["reasons"][0]
    key, scores = key_of([(20, 0.55), (24, 0.56)], [(40, 0.54), (44, 0.55)])
    assert gen["decide"](key, scores)["improved"], "a fall inside what two runs differ by is not a fall"


def test_two_references_are_compared_each_with_its_own_and_then_averaged():
    mech, scores = key_of([(20, 0.5), (22, 0.5)], [(40, 0.5), (42, 0.5)], "mech")
    cat, more = key_of([(70, 0.6), (72, 0.6)], [(70, 0.6), (72, 0.6)], "cat", start=5)
    v = gen["decide"]({"sheets": {**mech["sheets"], **cat["sheets"]}}, {**scores, **more})
    assert v["visual"]["gain"] == 10.0 and v["improved"]
    assert v["visual"]["noise"] < 2, "the gap between two subjects is not run-to-run noise"
    assert v["references"]["cat"]["visual"]["candidate"] == {"n": 2, "mean": 71.0, "values": [70.0, 72.0]}


def test_a_verdict_needs_a_score_for_every_sheet_and_no_others():
    key, scores = key_of([(20, 0.5), (22, 0.5)], [(40, 0.5), (42, 0.5)])
    for bad in ({**scores, "R01": {"note": "forgot"}}, {**scores, "R01": {"score": 140}},
                {**scores, "R01": {"score": True}}, {k: v for k, v in scores.items() if k != "R04"}):
        with pytest.raises(Refused, match="needs a `score` from 0 to 100"):
            gen["decide"](key, bad)
    with pytest.raises(Refused, match="the key does not have"):
        gen["decide"](key, {**scores, "R09": {"score": 50}})


def test_the_verdict_command_joins_the_key_and_the_scores(runs, tmp_path, capsys):
    review = tmp_path / "review"
    assert gen["main"](["sheets", str(review), "--incumbent", str(runs["root"] / "g00" / "runs" / "*" / "proj"),
                        "--candidate", str(runs["new1"]), str(runs["new2"])]) == 0, "a pattern is expanded here"
    assert gen["main"](["verdict", str(review)]) == 2, "no scores yet"
    key = json.loads((review / "key.json").read_text())["sheets"]
    scores = {rid: {"score": 70 if s["side"] == "candidate" else 30} for rid, s in key.items()}
    (review / "sheets" / "scores.json").write_text(json.dumps(scores))
    capsys.readouterr()
    assert gen["main"](["verdict", str(review)]) == 0
    said = capsys.readouterr().out
    v = json.loads((review / "verdict.json").read_text())
    assert v["improved"] and v["visual"]["gain"] == 40.0 and v["numeric"]["change"] > 0
    assert said.splitlines()[-1] == "IMPROVED" and "visual gain +40.0" in said
    # the same sheets scored the other way round
    (review / "sheets" / "scores.json").write_text(json.dumps({r: {"score": 100 - s["score"]} for r, s in scores.items()}))
    assert gen["main"](["verdict", str(review)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("NO IMPROVEMENT: the visual gain -40.0")


def write(loop: Path, generation: str, review=None, confirm=None):
    (loop / generation).mkdir(parents=True, exist_ok=True)
    for name, improved in (("review", review), ("confirm", confirm)):
        if improved is not None:
            (loop / generation / name).mkdir(exist_ok=True)
            (loop / generation / name / "verdict.json").write_text(json.dumps(
                {"improved": improved, "visual": {"gain": 9.0 if improved else 1.0}, "numeric": {"change": 0.004}}))


def test_the_streak_counts_generations_in_a_row_with_no_improvement(tmp_path):
    loop = tmp_path / "loop"
    write(loop, "g00")
    write(loop, "g01", review=False)
    write(loop, "g02", review=True, confirm=True)
    write(loop, "g03", review=False)
    write(loop, "g04", review=True, confirm=False)        # a gain that did not survive being measured again
    (loop / "final").mkdir()
    s = gen["status"](loop)
    assert [g["state"] for g in s["generations"]] == ["baseline", "no improvement", "kept", "no improvement",
                                                      "no improvement"]
    assert (s["streak"], s["stop"]) == (2, False)
    assert gen["status_text"](s).splitlines()[-1] == "CONTINUE: 2 of 3 generations in a row with no improvement"
    assert gen["status"](loop, stop_after=2)["stop"]
    for late in ("g9", "g10"):                            # in number order, not in the order of their names
        write(loop, late, review=False)
    s = gen["status"](loop)
    assert [g["name"] for g in s["generations"]][-2:] == ["g9", "g10"]
    assert (s["streak"], s["stop"]) == (4, True)
    assert gen["status_text"](s).splitlines()[-1] == "STOP: 4 of 3 generations in a row with no improvement"


def test_a_gain_not_yet_measured_again_stops_nothing_and_a_kept_one_starts_the_count_over(tmp_path):
    loop = tmp_path / "loop"
    write(loop, "g00")
    for g in ("g01", "g02", "g03"):
        write(loop, g, review=False)
    write(loop, "g04", review=True)
    s = gen["status"](loop)
    assert s["generations"][-1]["state"] == "unconfirmed" and s["open"] == ["g04"] and not s["stop"]
    assert "not decided yet: g04" in gen["status_text"](s)
    write(loop, "g04", confirm=True)
    assert (gen["status"](loop)["streak"], gen["status"](loop)["stop"]) == (0, False)
    write(loop, "g05")                                     # runs under way, no review yet
    assert gen["status"](loop)["open"] == ["g05"]


def test_remeasure_measures_what_a_run_delivered_in_a_new_project_and_leaves_the_run_alone(runs, tmp_path):
    before = sorted(p.relative_to(runs["new1"]).as_posix() for p in runs["new1"].rglob("*"))
    dest = tmp_path / "incumbent" / "new1"
    rec = gen["remeasure"](runs["new1"], dest, build_fn=fake_build(span=0.04), render_fn=quick_render)
    assert rec["ok"] and rec["numeric"]["numeric_score"] == gen["pick"](runs["new1"])["numeric"]
    assert sorted(p.name for p in (dest / "attempts").iterdir()) == ["a02", "ledger.md"], "only the delivered attempt"
    again = gen["pick"](dest)
    assert again["attempt"] == "a02" and again["reference"] == gen["pick"](runs["new1"])["reference"]
    assert before == sorted(p.relative_to(runs["new1"]).as_posix() for p in runs["new1"].rglob("*"))
    with pytest.raises(Refused, match="needs a new directory"):
        gen["remeasure"](runs["new1"], dest)
    # the generation's own project file, when it changed one
    jig = json.loads((runs["new1"] / "jig.json").read_text())
    jig["palette"]["pale"] = "#e0e0e0"
    (tmp_path / "jig.json").write_text(json.dumps(jig))
    gen["remeasure"](runs["new1"], tmp_path / "rejigged", jig=tmp_path / "jig.json", build_fn=fake_build(),
                     render_fn=quick_render)
    assert json.loads((tmp_path / "rejigged" / "jig.json").read_text())["palette"]["pale"] == "#e0e0e0"


def fenced(text: str) -> list[str]:
    out, inside = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside = not inside
        elif inside:
            out.append(line.strip())
    return out


def test_every_command_the_skill_hands_over_parses():
    """docs/LESSONS.md section 7: the instructions are a program."""
    from mesh_jig import cli
    lines = fenced((SKILL / "SKILL.md").read_text(encoding="utf-8"))
    ours = [c for c in lines if c.startswith("python skill/mesh-jig-harness-loop/scripts/generation.py ")]
    assert {shlex.split(c)[2] for c in ours} == {"sheets", "verdict", "status", "remeasure"}
    for command in ours:
        gen["parser"]().parse_args(shlex.split(command)[2:])
    theirs = [c for c in lines if c.startswith("mesh-jig ")]
    assert theirs
    for command in theirs:
        cli.parser().parse_args(shlex.split(command)[1:])


def test_the_skills_links_resolve_and_its_copied_files_are_there():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    targets = [t for t in re.findall(r"\]\(([^)]+)\)", text) if not t.startswith(("http", "#"))]
    assert targets and not [t for t in targets if not (SKILL / t).exists()]
    assert (SKILL / "review.md").is_file() and (SKILL / "loop.gitignore").is_file()
    assert text.startswith("---\nname: mesh-jig-harness-loop\ndescription: ")
