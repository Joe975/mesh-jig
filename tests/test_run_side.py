"""skill/mesh-jig-harness-loop/scripts/run_side.py: when a side's builders start, which runs are run again, and what
the summary holds, with a stand-in for the builder and a clock the test moves."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from mesh_jig import evaluate
from test_evaluate import fake_build, make_project, quick_render

SCRIPT = Path(__file__).resolve().parents[1] / "skill" / "mesh-jig-harness-loop" / "scripts" / "run_side.py"
spec = importlib.util.spec_from_file_location("run_side", SCRIPT)
run_side = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_side)


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Builder:
    """A builder that ends `lasts` seconds after it was started."""

    def __init__(self, clock, lasts, code=0):
        self.clock, self.started, self.lasts, self.code, self.killed = clock, clock.t, lasts, code, False

    def poll(self):
        return self.code if self.clock.t - self.started >= self.lasts else None

    def kill(self):
        self.killed = True

    def wait(self):
        return -9


def side(lasts: dict, dead=(), **kw):
    """Schedule stand-in builders: {name: seconds it runs}. Returns the result, each start time and the most at once."""
    clock, starts, going, most = Clock(), {}, [], [0]

    def launch(name):
        starts[name] = clock.t
        builder = Builder(clock, lasts[name])
        going.append(builder)
        most[0] = max(most[0], sum(b.poll() is None for b in going))
        return builder

    result = run_side.schedule(kw.pop("count", 4), launch, lambda name: name not in dead, now=clock.now,
                               sleep=clock.sleep, poll_s=10, **kw)
    return result, starts, most[0]


def test_two_builders_are_kept_going_and_the_second_of_a_pair_starts_ten_minutes_after_the_first():
    result, starts, most = side({"r01": 1500, "r02": 1800, "r03": 1200, "r04": 1200})
    assert starts == {"r01": 0, "r02": 600, "r03": 1500, "r04": 2400}
    assert most == 2
    assert [r["name"] for r in result["runs"]] == ["r01", "r02", "r03", "r04"]
    assert [r["wall_s"] for r in result["runs"]] == [1500, 1800, 1200, 1200]
    assert all(r["delivered"] and r["rerun_of"] is None and not r["timed_out"] for r in result["runs"])
    assert result["stopped"] == ""


def test_a_start_waits_for_the_stagger_only_while_the_run_before_it_is_still_going():
    # r01 is over in 100 s, so r02 need not wait for it; r03 waits ten minutes on r02, which is still writing
    _result, starts, _most = side({"r01": 100, "r02": 3000, "r03": 100, "r04": 100})
    assert starts == {"r01": 0, "r02": 100, "r03": 700, "r04": 800}


def test_one_at_a_time_and_no_stagger_are_honoured():
    _result, starts, most = side({"r01": 50, "r02": 50, "r03": 50}, count=3, at_a_time=1)
    assert starts == {"r01": 0, "r02": 50, "r03": 100} and most == 1
    _result, starts, most = side({"r01": 50, "r02": 50, "r03": 50}, count=3, at_a_time=3, stagger=0)
    assert starts == {"r01": 0, "r02": 0, "r03": 0} and most == 3


def test_a_dead_run_is_run_again_under_a_new_name_up_to_the_cap():
    lasts = {f"r{i:02d}": 100 for i in range(1, 8)}
    result, starts, _most = side(lasts, dead={"r01", "r02", "r05"})
    rows = {r["name"]: r for r in result["runs"]}
    assert set(starts) == {"r01", "r02", "r03", "r04", "r05", "r06"}       # two re-runs, and r05's death buys no third
    assert rows["r05"]["rerun_of"] == "r01" and rows["r06"]["rerun_of"] == "r02"
    assert [n for n, r in rows.items() if not r["delivered"]] == ["r01", "r02", "r05"]
    assert sum(r["delivered"] for r in rows.values()) == 3

    result, starts, _most = side(lasts, dead={"r01"}, reruns=0)
    assert set(starts) == {"r01", "r02", "r03", "r04"}


def test_a_builder_past_the_timeout_is_killed_and_the_side_still_ends():
    clock, made = Clock(), []

    def launch(name):
        made.append(Builder(clock, 10 ** 9))
        return made[-1]

    result = run_side.schedule(1, launch, lambda name: False, reruns=1, run_timeout=500, now=clock.now,
                               sleep=clock.sleep, poll_s=10)
    assert len(made) == 2 and all(b.killed for b in made)
    assert [(r["name"], r["timed_out"], r["exit"], r["rerun_of"]) for r in result["runs"]] == [
        ("r01", True, -9, None), ("r02", True, -9, "r01")]


def test_a_server_that_does_not_start_twice_in_a_row_ends_the_side():
    asked = []

    def down():
        asked.append(1)
        return False

    result, starts, _most = side({}, ensure_server=down)
    assert starts == {} and len(asked) == 2
    assert result == {"runs": [], "stopped": "the server did not start, 2 times in a row"}

    answers = iter([True, False, True, False, False])
    result, starts, _most = side({"r01": 100, "r02": 100, "r03": 100, "r04": 100}, ensure_server=lambda: next(answers))
    assert sorted(starts) == ["r01", "r02"]            # one failed start is retried; two in a row end it
    assert "did not start" in result["stopped"] and len(result["runs"]) == 2


def test_a_run_is_a_fresh_copy_of_the_seed_with_what_built_it_written_down(tmp_path):
    seed = make_project(tmp_path).root
    cut = []
    run_side.prepare(seed, tmp_path / "runs" / "r01", ["--model", "a model", "--attempts", "3"], cut=cut.append)
    made = tmp_path / "runs" / "r01"
    assert cut == [made] and (made / "jig.json").read_bytes() == (seed / "jig.json").read_bytes()
    commit, command = (made / "harness-commit.txt").read_text().splitlines()
    assert command == "mesh-jig agent <run> --model 'a model' --attempts 3"
    assert commit and str(tmp_path) not in commit
    with pytest.raises(run_side.Refused, match="fresh"):
        run_side.prepare(seed, made, [], cut=cut.append)


def test_a_side_builds_its_runs_and_the_summary_says_what_each_delivered(tmp_path, monkeypatch):
    seed = make_project(tmp_path).root
    runs = tmp_path / "g00" / "runs"

    class Stand:
        """`mesh-jig agent <run>`: r02 dies with nothing measured, every other run measures one attempt."""

        def __init__(self, root, builder_args, out):
            assert builder_args == ["--attempts", "1"]
            out.write(b"builder output\n")
            report = {"done": root.name != "r02", "error": "" if root.name != "r02" else "round 1 did not build",
                      "attempts": [], "turns": 3, "recoveries": [],
                      "calls": [{"usage": {"prompt_tokens": 100, "completion_tokens": 900}},
                                {"usage": {"prompt_tokens": 400, "completion_tokens": 50}}]}
            if root.name != "r02":
                script = root / "attempts" / "a01" / "ship.py"
                script.parent.mkdir(parents=True)
                script.write_text("def build():\n    pass\n")
                rec = evaluate.evaluate(run_side.project.load(root), script, script.parent, log=lambda *_: None,
                                        render_fn=quick_render, build_fn=fake_build(span=0.05))
                report["attempts"] = [{"name": "a01", "numeric_score": rec["numeric"]["numeric_score"]}]
            (root / "agent" / "one").mkdir(parents=True)
            (root / "agent" / "one" / "report.json").write_text(json.dumps(report))

        def poll(self):
            return 0

    monkeypatch.setattr(run_side, "builder", Stand)
    monkeypatch.setattr(run_side.time, "sleep", lambda s: None)
    said = []
    summary = run_side.side(runs, seed, 2, ["--attempts", "1"], stagger=0, log=said.append)
    assert summary == json.loads((runs / "summary.json").read_text())
    assert (summary["complete"], summary["asked"], summary["delivered"], summary["died"], summary["reruns"]) == (
        True, 2, 2, 1, 1)
    rows = {r["name"]: r for r in summary["runs"]}
    assert sorted(rows) == ["r01", "r02", "r03"] and rows["r03"]["rerun_of"] == "r02"
    assert not rows["r02"]["delivered"] and rows["r02"]["error"] == "round 1 did not build"
    assert rows["r01"]["finished"] and 0 < rows["r01"]["attempts"][0]["numeric_score"] < 1
    assert rows["r01"]["largest_prompt_and_reply"] == 1000 and rows["r01"]["output_tokens"] == 950
    assert (runs / "r01.log").read_bytes() == b"builder output\n"
    assert (runs / "r03" / "harness-commit.txt").is_file()
    assert str(tmp_path) not in (runs / "summary.json").read_text()
    assert any("DIED" in line and "r02" in line for line in said)
    with pytest.raises(run_side.Refused, match="not empty"):
        run_side.side(runs, seed, 2, [])
    shutil.rmtree(runs)
    with pytest.raises(run_side.Refused, match="needs --server-url"):
        run_side.side(runs, seed, 2, [], server_command=tmp_path / "command.txt")


def test_what_follows_the_two_dashes_is_the_builders_and_the_rest_is_the_sides():
    a, builder_args = run_side.arguments(["g00/runs", "--seed", "seeds/cat", "--count", "4", "--stagger", "0", "--",
                                          "--model", "m", "--attempts", "3", "--", "--effort", "low"])
    assert (a.runs, a.seed, a.count, a.stagger, a.at_a_time) == (Path("g00/runs"), Path("seeds/cat"), 4, 0.0, 2)
    assert builder_args == ["--model", "m", "--attempts", "3", "--", "--effort", "low"]
    a, builder_args = run_side.arguments(["g00/runs", "--seed", "seeds/cat", "--count", "4"])
    assert a.count == 4 and builder_args == []
    assert run_side.main(["g00/runs", "--seed", "no such seed", "--count", "1", "--", "--model", "m"]) == 2


def test_the_context_peak_is_read_from_the_metrics_page(monkeypatch):
    class Page:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"# HELP x\nllamacpp:n_tokens_max 57423\nllamacpp:requests_processing 2\n"

    monkeypatch.setattr(run_side.urllib.request, "urlopen", lambda url, timeout: Page())
    assert run_side.context_peak("http://localhost:1") == "context peak 57423, processing 2"

    def refuse(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr(run_side.urllib.request, "urlopen", refuse)
    assert run_side.context_peak("http://localhost:1") == "context ?"
    assert not run_side.answers("http://localhost:1")
