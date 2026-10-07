"""Production stopping policy and CLI, using synthetic records."""
import json
from pathlib import Path
from mesh_jig import streak

SKILL = Path(__file__).resolve().parents[1] / "skill" / "mesh-jig-keep-going" / "SKILL.md"


def measured(project, name, minute, score, gates=(), ok=True):
    d = project / "attempts" / name
    d.mkdir(parents=True)
    numeric = {} if score is None else {"numeric_score": score}
    d.joinpath("eval.json").write_text(json.dumps(
        {"ok": ok, "started": f"2026-10-05 09:{minute:02d}:00", "numeric": numeric,
         "gates": [{"name": n, "passed": p} for n, p in gates]}))


def test_with_a_margin_an_attempt_improves_only_when_it_beats_the_last_improvement_by_more_than_it():
    creep = [.7000, .7010, .7015, .7019, .7040]
    assert streak.rule_held_at(creep) is None, "with no margin, every one beats the best before it"
    assert streak.rule_held_at(creep, margin=.002) == 4, "three in a row within .002 of .700"
    assert streak.flat_streak(creep, margin=.002) == 0, "and .704 would have been an improvement"
    assert streak.flat_streak([.7000, .7010, .7025, .7030], margin=.002) == 1, "measured from .7025, the last that counted"
    assert streak.flat_streak([.5, .75], margin=.25) == 1, "by exactly the margin is not by more than it"
    assert streak.rule_held_at([.5, .6, .55, .58, .59], patience=2) == 4


def test_the_verdict_counts_measured_attempts_in_the_order_made_and_names_the_best_that_passes_every_gate(tmp_path):
    both, one = (("face", True), ("tail", True)), (("face", False), ("tail", True))
    measured(tmp_path, "z_first", 1, .60, both)
    measured(tmp_path, "a_broken", 2, None, ok=False)
    measured(tmp_path, "b_up", 3, .65, both)
    v = streak.verdict(tmp_path)
    assert not v["stop"] and (v["made"], v["left_out"], v["streak"]) == (2, 1, 0)
    assert v["margin"] == streak.MARGIN == .002, "what a builder's run counts with unless it is told otherwise"
    assert streak.text(v).splitlines()[-1] == "KEEP GOING: 0 in a row have not gained more than 0.002; stop at 3."
    measured(tmp_path, "c_higher_and_breaks_a_gate", 4, .70, one)
    for i, score in enumerate((.69, .70, .68)):
        assert not streak.verdict(tmp_path)["stop"]
        measured(tmp_path, f"d{i}", 5 + i, score, both)
    v = streak.verdict(tmp_path)
    assert v["stop"] and v["best"]["name"] == "c_higher_and_breaks_a_gate" and v["streak"] == 3
    assert [a["name"] for a in v["since"]] == ["d0", "d1", "d2"], "a tie with the best is not an improvement"
    assert v["clean"]["name"] == "d1", "the highest attempt that fails no gate"
    lines = streak.text(v).splitlines()
    assert lines[0] == "evaluated attempts: 6 (1 failed build not counted)"
    assert "gates: the best fails 1 of 2 (face)" in lines and "highest that passes every gate: d1 0.7000" in lines
    assert "the last 3 have not gained more than 0.002: d0 0.6900, d1 0.7000, d2 0.6800" in lines
    assert lines[-1] == ("STOP: 3 evaluated attempts in a row have not gained more than 0.002. "
                         "Report c_higher_and_breaks_a_gate, and d1 as the highest that passes every gate.")
    # a ceiling stops a run the streak has not; a wider patience keeps this one going
    assert streak.verdict(tmp_path, patience=4)["stop"] is False
    assert streak.verdict(tmp_path, patience=4, max_attempts=6)["why"] == "the ceiling of 6 evaluated attempts is reached"
    assert streak.verdict(tmp_path / "nothing")["why"] == "nothing has been measured yet"


def test_a_run_that_creeps_up_by_less_than_the_margin_stops_and_with_no_margin_does_not(tmp_path):
    for i, score in enumerate((.6900, .6910, .6915, .6919)):
        measured(tmp_path, f"a{i}", i, score)
    v = streak.verdict(tmp_path)
    assert v["stop"] and v["streak"] == 3 and v["best"]["name"] == "a3", "the best is still the highest score"
    assert "the last 3 have not gained more than 0.002: a1 0.6910, a2 0.6915, a3 0.6919" in streak.text(v)
    plain = streak.verdict(tmp_path, margin=0.0)
    assert not plain["stop"] and plain["streak"] == 0
    assert streak.text(plain).splitlines()[-1] == "KEEP GOING: 0 in a row have not improved on the best; stop at 3."


def test_the_command_prints_the_verdict_for_a_project_and_refuses_a_directory_that_is_not_one(tmp_path, capsys):
    from mesh_jig import cli
    from test_evaluate import make_project
    proj = make_project(tmp_path)
    for i, score in enumerate((.5, .6, .6, .59, .6)):
        measured(proj.root, f"a{i}", i, score)
    assert cli.main(["streak", str(proj.root)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "best: a1 0.6000" and out[-1].startswith("STOP: 3 evaluated attempts in a row")
    assert cli.main(["streak", str(proj.root), "--patience", "4", "--json"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["stop"] is False and said["margin"] == .002, "the command's default margin is the module's"
    assert cli.main(["streak", str(proj.root), "--margin", "0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["margin"] == 0
    assert cli.main(["streak", str(tmp_path / "nowhere")]) == 2


def test_every_command_the_keep_going_skill_hands_over_parses_and_its_links_resolve():
    """docs/LESSONS.md section 7: the instructions are a program."""
    import re
    import shlex
    from mesh_jig import cli
    text = SKILL.read_text(encoding="utf-8")
    blocks = re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
    commands = [line.strip() for block in blocks for line in block.splitlines() if line.strip().startswith("mesh-jig ")]
    assert len(commands) >= 2 and all(c.split()[1] == "streak" for c in commands)
    for command in commands:
        cli.parser().parse_args(shlex.split(command)[1:])
    assert "--max-attempts 30" in " ".join(commands) and "--margin 0 " in " ".join(commands), "the options are shown in use"
    assert f"| {streak.MARGIN:g} |" in text and f"no more than {streak.MARGIN:g}" in text, "the skill states the command's own margin"
    targets = [t for t in re.findall(r"\]\(([^)]+)\)", text) if not t.startswith(("http", "#"))]
    assert targets and not [t for t in targets if not (SKILL.parent / t).exists()]
