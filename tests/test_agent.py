"""mesh_jig.agent: the tool loop, driven by a scripted model. The tools are the real MCP functions on the
known-answer project (Blender replaced by a builder that writes the right GLB, as in test_mcp_server)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh_jig import agent, cli, llm, mcp_server, project as project_mod
from test_mcp_server import proj  # noqa: F401  (the fixture)


def call(tool, i=0, /, **args):
    return {"id": f"c{tool}{i}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}


class Scripted:
    """A chat function that replies with the given turns in order and keeps what it was sent."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.sent: list[list[dict]] = []
        self.tools: list[dict] = []

    def __call__(self, messages, tools):
        self.sent.append(json.loads(json.dumps(messages)))
        self.tools = tools
        turn = self.turns.pop(0) if self.turns else []
        msg = {"role": "assistant", "content": turn} if isinstance(turn, str) else \
              {"role": "assistant", "content": "", "tool_calls": turn}
        return msg, {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001}


SCRIPT = "def build():\n    pass\n"


def tool_results(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


def images(content):
    return [c for c in content if isinstance(c, dict) and c.get("type") == "image_url"] if isinstance(content, list) else []


def test_a_scripted_model_writes_evaluates_and_finishes(proj, tmp_path):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", script="attempts/a01/ship.py", note="baseline")],
                    [call("done", summary="a01 will do")], [call("done", summary="a01 is the best")])
    out = tmp_path / "run"
    rep = agent.run(proj, chat, attempts=3, max_turns=6, model="m", out=out)
    assert rep.done and rep.summary == "a01 is the best" and not rep.error
    assert rep.evaluations == 1 and rep.best["name"] == "a01" and rep.best["numeric_score"] > 0.9
    assert (rep.turns, rep.prompt_tokens, rep.cost_usd) == (4, 400, 0.004)
    # stopping with attempts unused is refused until the last turns, where it is all that is left to do
    assert tool_results(chat.sent[3])[-1].startswith("NOT YET: 2 of the 3 evaluated attempts are unused")
    assert "make exactly 3 evaluated attempts" in chat.sent[0][0]["content"]
    assert chat.sent[0][1]["content"][-1]["text"].startswith("Make exactly 3 successfully evaluated attempts")
    assert "| a01 | ok |" in proj.ledger.read_text() and "baseline" in proj.ledger.read_text()
    # the opening message carries the brief and every reference view; evaluate's side-by-sides follow its tool result
    opening = chat.sent[0][1]["content"]
    assert "PROJECT ship" in opening[0]["text"] and len(images(opening)) == 4
    after = chat.sent[2]
    assert agent.MEASURED in tool_results(after)[-1] and "2 evaluated attempts left" in tool_results(after)[-1]
    # ... then the outline diff of each scored view (top, side, rear), and no path carries the project's own
    shown = after[-1]["content"]
    assert after[-1]["role"] == "user" and len(images(shown)) == 4 + 3
    assert sorted(c["text"].split(" (")[0] for c in shown if c.get("text", "").startswith("outline diff")) == [
        "outline diff, rear", "outline diff, side", "outline diff, top"]
    said = "".join(tool_results(after)) + json.dumps(opening[0])
    assert str(proj.root) not in said and proj.root.as_posix() not in said and proj.root.name not in said
    assert ("LOOK AT (reference left, your build right) in attempts/a01/compare: top_vs_reference.png"
            in tool_results(after)[-1].replace("\\", "/"))
    assert "standing: the first measured attempt here." in tool_results(after)[-1]
    report = json.loads((out / "report.json").read_text())
    assert report["evaluations"] == 1 and report["done"]
    transcript = (out / "transcript.jsonl").read_text()
    assert "base64" not in transcript and '{"type": "image"}' in transcript
    # what a run leaves behind is published: the transcript, the report and eval.json name no directory above the project
    left = transcript + (out / "report.json").read_text() + (proj.root / "attempts" / "a01" / "eval.json").read_text()
    for spelt in (str(proj.root), proj.root.as_posix(), json.dumps(str(proj.root))[1:-1], json.dumps(json.dumps(str(proj.root)))[2:-2]):
        assert spelt not in left
    assert report["project"] == "ship" and str(proj.root) not in chat.sent[0][0]["content"]


def test_the_model_cannot_write_outside_an_attempt_or_reach_outside_the_project(proj, tmp_path):
    jig = (proj.root / "jig.json").read_text()
    outside = tmp_path / "outside.py"
    outside.write_text(SCRIPT)
    chat = Scripted([call("write_file", 1, path="jig.json", text="{}"),
                     call("write_file", 2, path="attempts/ledger.md", text="forged"),
                     call("write_file", 3, path="attempts/../brief.md", text="easier"),
                     call("read_file", 4, path="../outside.py"),
                     call("evaluate", 5, script=str(outside)),
                     call("build", 6, script="refs/top.png"),
                     call("evaluate", 7, script="attempts/a01/ship.py", judge_against="attempts/a00"),
                     call("evaluate", 8, script="attempts/a01/ship.py", project="/elsewhere")],
                    [call("read_file", path="brief.md")], [call("done", summary="x")])
    rep = agent.run(proj, chat, max_turns=5)
    results = tool_results(chat.sent[1])
    assert len(results) == 8 and all(r.startswith("REJECTED") for r in results), results
    assert rep.rejected_calls == 8 and rep.done
    assert (proj.root / "jig.json").read_text() == jig and not proj.ledger.exists()
    assert "red nose block" in tool_results(chat.sent[2])[-1], "reading inside the project works"


def test_file_tools_edit_list_and_refuse_binaries(proj):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text="a = 1\nb = 2\n")],
                    [call("edit_file", path="attempts/a01/ship.py", edits=[{"old": "a = 1", "new": "a = 3"},
                                                                          {"old": "b = 2", "new": "b = 4"}])],
                    [call("edit_file", path="attempts/a01/ship.py", edits=[{"old": "a = 3", "new": "a = 5"},
                                                                          {"old": "nowhere", "new": ""}])],
                    [call("read_file", 1, path="refs/top.png"), call("list_dir", 2, path="attempts"),
                     call("view_image", 3, paths=["refs/top.png", "jig.json"])],
                    [call("done", summary="x")])
    agent.run(proj, chat, max_turns=5)
    assert (proj.root / "attempts" / "a01" / "ship.py").read_text() == "a = 3\nb = 4\n", "a failed edit applies nothing"
    last = chat.sent[-1]
    binary, listing, viewed = tool_results(last)[-3:]
    assert binary.startswith("BINARY") and listing == "a01/" and "1 picture(s) follow" in viewed and "jig.json" in viewed
    assert len(images(last[-1]["content"])) == 1


def test_write_file_without_text_writes_nothing_and_content_is_accepted(proj):
    chat = Scripted([call("write_file", 1, path="attempts/a01/ship.py"),
                     call("write_file", 2, path="attempts/a01/other.py", content="x = 1\n")],
                    [call("done", summary="x")])
    agent.run(proj, chat)
    refused, wrote = tool_results(chat.sent[1])
    assert refused.startswith("ERROR: write_file needs `text`") and not (proj.root / "attempts" / "a01" / "ship.py").exists()
    assert wrote.startswith("wrote 6 chars") and (proj.root / "attempts" / "a01" / "other.py").read_text() == "x = 1\n"


@pytest.mark.parametrize("edits, message", [
    ([{"old": "x", "new": "y"}], "matches 2 places"),
    ([{"old": "", "new": "y"}], "is empty"),
    ([], "non-empty list"),
])
def test_an_edit_must_name_one_place(edits, message):
    with pytest.raises(ValueError, match=message):
        agent.edit_text("x\nx\n", edits)
    assert agent.edit_text("a\nb\nc\n", [{"old": "c", "new": "C"}, {"old": "a\n", "new": ""}]) == ("b\nC\n", [3, 1])


def test_the_attempt_budget_refuses_the_next_evaluate(proj):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", script="attempts/a01/ship.py")],
                    [call("evaluate", script="attempts/a01/ship.py")],
                    [call("done", summary="x")])
    rep = agent.run(proj, chat, attempts=1)
    assert "last evaluated attempt in the budget" in tool_results(chat.sent[2])[-1]
    assert tool_results(chat.sent[3])[-1].startswith("REFUSED") and rep.evaluations == 1


def test_done_is_taken_at_once_when_the_budget_is_used(proj):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", script="attempts/a01/ship.py")], [call("done", summary="a01")])
    rep = agent.run(proj, chat, attempts=1)
    assert rep.done and rep.turns == 3 and rep.summary == "a01"


def test_a_failed_build_is_recorded_but_does_not_use_up_an_attempt(proj, monkeypatch):
    def failed(project, script, out, **kw):
        rec = {"name": Path(out).name, "ok": False, "stage": "build", "error": "JIG_BUILD_ERROR: line 2"}
        (Path(out) / "eval.json").write_text(json.dumps(rec))
        return rec
    monkeypatch.setattr(mcp_server.evaluate_mod, "evaluate", failed)
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", script="attempts/a01/ship.py")], [call("done", summary="x")])
    rep = agent.run(proj, chat, attempts=1)
    assert "FAILED at build" in tool_results(chat.sent[2])[-1]
    assert rep.evaluations == 0 and rep.attempts == [{"name": "a01", "ok": False, "numeric_score": None}]


def test_the_turn_cap_ends_the_run_and_is_announced(proj):
    chat = Scripted(*[[call("list_dir", i)] for i in range(5)])
    rep = agent.run(proj, chat, max_turns=3)
    assert rep.turns == 3 and "turn cap (3)" in rep.error and not rep.done
    assert "[2 turns left." in tool_results(chat.sent[1])[-1]


def test_the_cost_cap_stops_the_run_while_the_dearest_call_still_fits(proj, tmp_path):
    chat = Scripted(*[[call("list_dir", i)] for i in range(10)])          # 0.001 USD a call
    rep = agent.run(proj, chat, max_cost=0.0035, out=tmp_path / "run")
    assert (rep.turns, len(chat.sent), rep.cost_usd) == (3, 3, 0.003) and not rep.done
    assert "cost cap" in rep.error and "$0.0030 spent" in rep.error
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["max_cost_usd"] == 0.0035 and report["cost_usd"] <= 0.0035 and report["error"] == rep.error
    assert cli.parser().parse_args(["agent", "p", "--model", "m", "--max-cost", "1.5"]).max_cost == 1.5
    assert cli.parser().parse_args(["agent", "p", "--model", "m"]).max_cost == 0.0


def test_a_cost_cap_cannot_be_held_against_an_endpoint_that_reports_no_cost(proj):
    def free(messages, tools):
        return {"role": "assistant", "content": "", "tool_calls": [call("list_dir")]}, {"prompt_tokens": 100}
    rep = agent.run(proj, free, max_cost=1.0)
    assert rep.turns == 1 and "reported no cost" in rep.error
    assert not agent.run(proj, free, max_turns=2).error.startswith("stopped")     # no cap, nothing to hold


def test_the_time_budget_is_one_clock_and_each_call_is_handed_what_is_left(proj, tmp_path):
    now, lefts = [1000.0], []

    def slow(messages, tools, left_s):                       # every model call takes 40 s
        lefts.append(left_s)
        now[0] += 40
        return {"role": "assistant", "content": "", "tool_calls": [call("list_dir")]}, {"prompt_tokens": 100}
    rep = agent.run(proj, slow, max_wall=100, clock=lambda: now[0], out=tmp_path / "run")
    assert lefts == [100, 60, 20] and rep.turns == 3 and not rep.done
    assert rep.error == "stopped at the time budget: 120 s of 100 s used" and rep.wall_s == 120
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["max_wall_s"] == 100 and report["error"] == rep.error
    assert agent.over_wall(99.9, 100) == "" and agent.over_wall(5000, 0) == ""      # 0 is no budget


def test_a_call_that_outlasts_the_budget_ends_the_run_as_out_of_time_not_as_a_failure(proj):
    def cut(messages, tools, left_s):
        raise llm.OutOfTime(f"the {left_s:.0f} s left for the model call ran out")
    rep = agent.run(proj, cut, max_wall=30, clock=lambda: 0.0)
    assert rep.error.startswith("stopped at the time budget (30 s) during a model call") and "chat failed" not in rep.error


def test_the_command_takes_a_time_budget_and_lets_a_call_use_all_of_it():
    args = cli.parser().parse_args(["agent", "p", "--model", "m", "--max-wall", "1800"])
    assert args.max_wall == 1800.0 and cli.call_cap(args.timeout, args.max_wall) == 1800.0
    args = cli.parser().parse_args(["agent", "p", "--model", "m"])
    assert args.max_wall == 0.0 and cli.call_cap(args.timeout, args.max_wall) == 900.0       # as it was
    assert cli.call_cap(300.0, 1800.0) == 300.0                                             # a given cap is kept


def test_a_reply_without_tools_is_nudged_once_then_ends_the_run(proj):
    chat = Scripted("thinking out loud", "still talking")
    rep = agent.run(proj, chat)
    assert rep.error == "stopped without done()" and rep.summary == "still talking" and rep.turns == 2
    assert chat.sent[1][-1] == {"role": "user", "content": "Use the tools. Call done(summary) when you are finished."}


def test_a_chat_failure_ends_the_run_and_still_writes_the_report(proj, tmp_path):
    def broken(messages, tools):
        raise llm.LLMError("HTTP 401 from x")
    rep = agent.run(proj, broken, out=tmp_path / "run")
    assert "HTTP 401" in rep.error and json.loads((tmp_path / "run" / "report.json").read_text())["error"] == rep.error


def test_old_pictures_become_captions_and_the_references_stay():
    pic = {"type": "image_url", "image_url": {"url": "data:"}}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "brief"}, pic]}]
    for i in range(4):
        msgs += [{"role": "tool", "content": "r"}, {"role": "user", "content": [{"type": "text", "text": f"p{i}"}, pic]}]
    assert agent.prune_images(msgs, keep=2) == 2
    assert [len(images(m["content"])) for m in msgs if m["role"] == "user"] == [1, 0, 0, 1, 1]
    assert msgs[3]["content"][1] == {"type": "text", "text": agent.DROPPED}
    assert agent.prune_images(msgs, keep=2) == 0


def three_attempts(prefix):
    """A scripted model that measures three attempts, the second and third copied from the first."""
    turns = [[call("write_file", path=f"attempts/{prefix}1/ship.py", text=SCRIPT)],
             [call("evaluate", 1, script=f"attempts/{prefix}1/ship.py")]]
    turns += [[call("copy_attempt", n, source=f"attempts/{prefix}1", name=f"{prefix}{n}"),
               call("evaluate", n, script=f"attempts/{prefix}{n}/ship.py")] for n in (2, 3)]
    return Scripted(*turns, [call("done", summary="x")])


def test_under_a_marked_cache_every_turns_pictures_stay_up_to_the_request_limit(proj):
    chat = three_attempts("a")
    assert agent.run(proj, chat, attempts=3, keep_images=None).evaluations == 3
    assert [len(images(m["content"])) for m in chat.sent[-1] if m["role"] == "user"] == [4, 7, 7, 7]
    pruned = three_attempts("b")
    agent.run(proj, pruned, attempts=3)
    assert [len(images(m["content"])) for m in pruned.sent[-1] if m["role"] == "user"] == [4, 0, 0, 7, 7]
    # over the limit the oldest turns go, never the references
    msgs = [m for m in chat.sent[-1]]
    assert agent.cap_images(msgs, limit=25) == 0 and agent.cap_images(msgs, limit=20) == 7
    assert [len(images(m["content"])) for m in msgs if m["role"] == "user"] == [4, 0, 7, 7]


def test_an_attempt_below_the_best_is_said_to_be_and_the_next_starts_from_the_best(proj, monkeypatch):
    scores = iter([0.50, 0.45, 0.55])

    def scored(project, script, out, **kw):
        rec = {"name": Path(out).name, "ok": True, "numeric": {"numeric_score": next(scores)}, "renders": {}, "compare": {}}
        (Path(out) / "eval.json").write_text(json.dumps(rec))
        return rec
    monkeypatch.setattr(mcp_server.evaluate_mod, "evaluate", scored)
    chat = three_attempts("a")
    rep = agent.run(proj, chat, attempts=3)
    first, second, third = (tool_results(chat.sent[i])[-1] for i in (2, 3, 4))
    assert "best" not in first and "[The best attempt so far.]" in third and rep.best["name"] == "a3"
    assert "[Below a1 (0.500), the best so far: this change lost score. Start the next attempt from a1" in second


def scores_in_turn(monkeypatch, *scores):
    """`evaluate` replaced by one that gives the next of `scores` to each attempt it is asked to measure."""
    left = iter(scores)

    def scored(project, script, out, **kw):
        rec = {"name": Path(out).name, "ok": True, "numeric": {"numeric_score": next(left)}, "renders": {}, "compare": {}}
        (Path(out) / "eval.json").write_text(json.dumps(rec))
        return rec
    monkeypatch.setattr(mcp_server.evaluate_mod, "evaluate", scored)


def another(n):
    return [call("copy_attempt", n, source="attempts/a1", name=f"a{n}"), call("evaluate", n, script=f"attempts/a{n}/ship.py")]


def test_under_patience_the_run_goes_on_until_that_many_in_a_row_have_not_improved(proj, monkeypatch, tmp_path):
    """`--patience`: the count `mesh-jig streak` gives ends every evaluation, done() is refused while it says KEEP
    GOING, and an evaluate after it says STOP is refused."""
    scores_in_turn(monkeypatch, 0.50, 0.55, 0.551, 0.549, 0.60)
    chat = Scripted([call("write_file", path="attempts/a1/ship.py", text=SCRIPT)],
                    [call("evaluate", 1, script="attempts/a1/ship.py")], [call("done", summary="enough")],
                    another(2), another(3), another(4), another(5), [call("done", summary="a3")])
    rep = agent.run(proj, chat, attempts=10, patience=2, out=tmp_path / "run")
    prompt, task = chat.sent[0][0]["content"], chat.sent[0][1]["content"][-1]["text"]
    assert "until 2 in a row have not gained more than 0.002 on the best numeric score before them, or 10 are made" in prompt
    assert "make exactly" not in prompt and "A LONG RUN:" in prompt and "one of your 10 attempts" not in prompt
    assert task.startswith("Make successfully evaluated attempts") and "exactly" not in task
    assert "Keep going until 2 attempts in a row show no improvement" in task and "Stop early at: 10 attempts." in task
    first, early, second, third, fourth, fifth = (tool_results(chat.sent[i])[-1] for i in (2, 3, 4, 5, 6, 7))
    assert "KEEP GOING: 0 in a row have not gained more than 0.002; stop at 2. 9 more at most: the ceiling is 10.]" in first
    assert early.startswith("NOT YET: the task is to keep going until 2 evaluated attempts in a row have not gained")
    assert "[The best attempt so far.]" in second and "KEEP GOING: 0 in a row" in second
    # 0.551 is the best and is not an improvement: it is within the margin of 0.55
    assert "best: a3 0.5510" in third and "KEEP GOING: 1 in a row has not gained more than 0.002" in third
    assert "Start the next attempt from a3 (copy_attempt)" not in third and "[Below a3 (0.551)" in fourth
    assert fourth.endswith("STOP: 2 evaluated attempts in a row have not gained more than 0.002. Report a3. Make no more "
                           "attempts: call done(summary).]")
    assert "Start the next attempt" not in fourth and "left in the budget" not in first + fourth
    assert fifth.startswith("REFUSED: 2 evaluated attempts in a row have not gained more than 0.002, so the run is over")
    assert rep.done and rep.summary == "a3" and rep.evaluations == 4 and rep.best["name"] == "a3"
    assert (rep.patience, rep.margin, rep.stopped) == (2, 0.002, "2 evaluated attempts in a row have not gained more than 0.002")
    assert json.loads((tmp_path / "run" / "report.json").read_text())["stopped"] == rep.stopped


def test_under_patience_the_attempts_are_a_ceiling_and_any_higher_score_counts_at_no_margin(proj, monkeypatch):
    scores_in_turn(monkeypatch, 0.50, 0.501, 0.502)
    chat = Scripted([call("write_file", path="attempts/a1/ship.py", text=SCRIPT)],
                    [call("evaluate", 1, script="attempts/a1/ship.py")], another(2), another(3), [call("done", summary="a3")])
    rep = agent.run(proj, chat, attempts=3, patience=2, margin=0)
    assert "until 2 in a row have not improved on the best numeric score before them, or 3 are made" in chat.sent[0][0]["content"]
    assert "KEEP GOING: 0 in a row have not improved on the best; stop at 2. 1 more at most" in tool_results(chat.sent[3])[-1]
    assert "last evaluated attempt in the budget" in tool_results(chat.sent[4])[-1]
    assert rep.done and rep.evaluations == 3 and not rep.stopped and (rep.patience, rep.margin) == (2, 0.0)


def test_a_run_given_no_patience_is_told_a_fixed_number_and_records_no_rule(proj):
    chat = Scripted([call("done", summary="x")])
    rep = agent.run(proj, chat, attempts=3, max_turns=1)
    assert "A LONG RUN" not in chat.sent[0][0]["content"] and "KEEP GOING" not in chat.sent[0][0]["content"]
    assert (rep.patience, rep.margin, rep.stopped) == (0, 0.0, "")


def test_the_command_hands_the_rule_to_the_run(proj, capsys, monkeypatch):
    given = {}
    monkeypatch.setattr(llm, "resolve_backend", lambda **kw: llm.Backend("openai-compatible", "http://x/v1", "m", "k"))
    monkeypatch.setattr(agent, "chat_for", lambda *args, **kw: None)
    monkeypatch.setattr(agent, "run", lambda *args, **kw: given.update(kw) or agent.Report(model="m", done=True))
    assert cli.main(["agent", str(proj.root), "--attempts", "60", "--patience", "5"]) == 0
    assert (given["attempts"], given["patience"], given["margin"]) == (60, 5, 0.002)
    assert "up to 60 attempts, until 5 in a row have not gained more than 0.002" in capsys.readouterr().out
    assert cli.main(["agent", str(proj.root)]) == 0 and (given["attempts"], given["patience"]) == (5, 0)


def test_the_projects_own_path_comes_off_a_tool_result_before_it_is_clipped(proj, monkeypatch):
    """A result that only the project's path, spelt out on every line, pushes past the limit keeps its middle."""
    root = str(proj.root)
    lines = agent.MAX_TOOL_OUTPUT // len(root) + 20
    monkeypatch.setattr(mcp_server.evaluate_mod, "feedback",
                        lambda rec, stands=None: "\n".join(str(proj.root / f"x{i}") for i in range(lines)))
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", script="attempts/a01/ship.py")], [call("done", summary="x")])
    agent.run(proj, chat, attempts=1)
    said = tool_results(chat.sent[2])[-1]
    assert lines * len(root) > agent.MAX_TOOL_OUTPUT, "as the tool gave it, the text is over the limit"
    assert "clipped" not in said and said.startswith("x0\nx1\n") and f"\nx{lines // 2}\n" in said and root not in said


def test_a_path_that_starts_with_an_attempts_name_is_taken_as_under_attempts(proj):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", 1, script="a01/ship.py"), call("copy_attempt", 2, source="a01", name="a02"),
                     call("read_file", 3, path="a01/ship.py"), call("build", 4, script="refs/top.png"),
                     call("write_file", 5, path="a09/ship.py", text=SCRIPT)],
                    [call("done", summary="x")])
    rep = agent.run(proj, chat, attempts=1)
    measured, copied, read, refs, new = tool_results(chat.sent[2])[-5:]
    assert agent.MEASURED in measured and copied.startswith("copied ship.py") and rep.evaluations == 1
    assert read.startswith("ERROR: no such file")            # reads are relative to the project, as before
    assert refs.startswith("REJECTED") and new.startswith("REJECTED") and rep.rejected_calls == 2


def test_a_preview_is_a_free_look_held_against_the_best_measured_attempt(proj):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("build", 1, script="attempts/a01/ship.py"), call("preview", 2, attempt="attempts/a01")],
                    [call("evaluate", script="attempts/a01/ship.py")],
                    [call("copy_attempt", 1, source="attempts/a01", name="a02"),
                     call("build", 2, script="attempts/a02/ship.py"),
                     call("preview", 3, attempt="attempts/a02", views=["top", "side"])],
                    [call("done", summary="x")])
    rep = agent.run(proj, chat, attempts=1)
    first = tool_results(chat.sent[2])[-1]
    assert "[A free look, not an attempt: overlap 1.000 over top, side, rear. Change" in first
    shown = chat.sent[2][-1]["content"]
    assert len(images(shown)) == 4 + 3 and sum(c.get("text", "").startswith("outline diff") for c in shown) == 3
    second = tool_results(chat.sent[4])[-1]
    assert "overlap 1.000 over top, side; your best measured attempt, a01, has 1.000" in second
    assert rep.evaluations == 1 and [a["name"] for a in rep.attempts] == ["a01"]      # the looks are not attempts
    assert "`preview` is a free look" in chat.sent[0][0]["content"]


def test_a_project_that_names_its_scoring_views_is_shown_those_and_not_the_orthographic_three(proj):
    jig = json.loads((proj.root / "jig.json").read_text())
    jig["scoring"] = {"views": {"hero": 3, "top": 1}}
    (proj.root / "jig.json").write_text(json.dumps(jig))
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("build", 1, script="attempts/a01/ship.py"), call("preview", 2, attempt="attempts/a01")],
                    [call("evaluate", script="attempts/a01/ship.py")], [call("done", summary="x")])
    agent.run(project_mod.load(proj.root), chat, attempts=1)
    assert "overlap 1.000 over hero, top. Change" in tool_results(chat.sent[2])[-1]
    for sent in (chat.sent[2], chat.sent[3]):                # the look, then the measured attempt
        diffs = [c["text"].split(" (")[0] for c in sent[-1]["content"] if c.get("text", "").startswith("outline diff")]
        assert diffs == ["outline diff, hero", "outline diff, top"]
    measured = tool_results(chat.sent[3])[-1]
    assert "Aggregate scoring weights: hero 3, top 1; weakest overlap views: " in measured and "ONLY" not in measured
    assert "{" not in measured.split("Aggregate scoring weights")[1].split("\n")[0], "settings as words, not as a dict"


def test_a_measured_attempt_is_frozen_and_the_next_one_starts_as_a_copy(proj):
    chat = Scripted([call("write_file", path="attempts/a01/ship.py", text=SCRIPT)],
                    [call("evaluate", script="attempts/a01/ship.py")],
                    [call("edit_file", 1, path="attempts/a01/ship.py", old="pass", new="return"),
                     call("write_file", 2, path="attempts/a01/ship.py", text="x = 1\n"),
                     call("evaluate", 3, script="attempts/a01/ship.py", note="again"),
                     call("evaluate", 4, script="attempts/a01/ship.py", name="a02")],
                    [call("copy_attempt", 1, source="attempts/a01", name="attempts/a01"),
                     call("copy_attempt", 2, source="attempts/a01/ship.py", name="a02_return"),
                     call("copy_attempt", 3, source="attempts/a01", name="../a03"),
                     call("edit_file", 4, path="attempts/a02_return/ship.py", old="pass", new="return"),   # one edit, flat
                     call("evaluate", 5, script="attempts/a02_return/ship.py")],
                    [call("done", summary="x")])
    rep = agent.run(proj, chat, attempts=2)
    frozen = tool_results(chat.sent[3])[-4:]
    assert all("already measured" in r and "copy_attempt" in r for r in frozen[:3])
    assert frozen[0].startswith("ERROR") and frozen[2].startswith("REFUSED") and frozen[3].startswith("REJECTED")
    exists, copied, bad, edited, measured = tool_results(chat.sent[4])[-5:]
    assert "exists already" in exists and copied.startswith("copied ship.py (3 lines) to attempts/a02_return/")
    assert bad.startswith("ERROR: `name`") and edited.startswith("edited ship.py") and agent.MEASURED in measured
    assert (proj.root / "attempts/a01/ship.py").read_text() == SCRIPT
    assert (proj.root / "attempts/a02_return/ship.py").read_text() == SCRIPT.replace("pass", "return")
    assert [a["name"] for a in rep.attempts] == ["a01", "a02_return"] and rep.evaluations == 2 and rep.done


def test_cache_marks_sit_on_the_opening_the_last_request_and_this_one():
    pic = {"type": "image_url", "image_url": {"url": "data:"}}
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "brief"}, pic, {"type": "text", "text": "start"}]},
            {"role": "assistant", "content": None, "tool_calls": [call("list_dir")]},
            {"role": "tool", "tool_call_id": "c", "content": "a01/"},
            {"role": "assistant", "content": None, "tool_calls": [call("evaluate")]},
            {"role": "tool", "tool_call_id": "c", "content": "measured"},
            {"role": "user", "content": [{"type": "text", "text": "side"}, pic]},
            {"role": "assistant", "content": None, "tool_calls": [call("list_dir")]},
            {"role": "tool", "tool_call_id": "c", "content": "a01/"}]
    before = json.dumps(msgs)
    out = llm.cache_marks(msgs)
    assert json.dumps(msgs) == before                                    # a copy: the conversation itself is unmarked
    marked = [i for i, m in enumerate(out) if "cache_control" in json.dumps(m)]
    assert marked == [1, 6, 8]
    assert out[1]["content"][2] == {"type": "text", "text": "start", "cache_control": {"type": "ephemeral"}}
    assert out[6]["content"][0]["cache_control"] and "cache_control" not in out[6]["content"][1]   # the caption, not the image
    assert out[8]["content"] == [{"type": "text", "text": "a01/", "cache_control": {"type": "ephemeral"}}]
    # the first request has one mark; the second marks the first request's end again, where the cache was written
    assert [i for i, m in enumerate(llm.cache_marks(msgs[:2])) if "cache_control" in json.dumps(m)] == [1]
    assert [i for i, m in enumerate(llm.cache_marks(msgs[:4])) if "cache_control" in json.dumps(m)] == [1, 3]


def test_the_cache_is_marked_for_anthropic_on_openrouter_and_the_report_counts_what_it_read(proj, monkeypatch):
    claude = llm.Backend("openrouter", llm.DEFAULT_URL, "anthropic/claude-x", "k")
    assert llm.wants_cache_marks(claude)
    assert not llm.wants_cache_marks(llm.Backend("openrouter", llm.DEFAULT_URL, "deepseek/deepseek-v4.1-flash", "k"))
    assert not llm.wants_cache_marks(llm.Backend("openai-compatible", "https://api.anthropic.com/v1", "anthropic/x", "k"))
    sent = []
    usage = {"prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.002,
             "prompt_tokens_details": {"cached_tokens": 900, "cache_write_tokens": 80},
             "completion_tokens_details": {"reasoning_tokens": 20}}

    def fake(backend, messages, **kw):
        sent.append(messages)
        return {"text": "", "message": {"role": "assistant", "content": "", "tool_calls": [call("done", summary="x")]},
                "usage": usage}
    monkeypatch.setattr(llm, "chat_stream", fake)
    rep = agent.run(proj, agent.chat_for(claude, cache=True), keep_images=None, attempts=0, max_turns=3)
    assert "cache_control" in json.dumps(sent[0][1]) and "cache_control" not in json.dumps(sent[0][0])
    assert (rep.prompt_tokens, rep.cached_tokens, rep.cache_write_tokens, rep.reasoning_tokens) == (1000, 900, 80, 20)
    assert rep.calls == [{"turn": 1, "evaluations": 0, "prompt_tokens": 1000, "cached_tokens": 900, "cache_write_tokens": 80,
                          "completion_tokens": 50, "reasoning_tokens": 20, "cost_usd": 0.002}]
    assert "900 read from the cache" in agent.summary(rep)
    agent.run(proj, agent.chat_for(claude), keep_images=None)
    assert "cache_control" not in json.dumps(sent[1])
    args = cli.parser().parse_args(["agent", "p", "--model", "m"])
    assert (args.cache, args.keep_images) == ("auto", None)


def test_a_models_reasoning_blocks_go_back_with_the_tool_results(proj):
    thought = [{"type": "reasoning.text", "text": "the legs are short", "signature": "sig", "index": 0}]

    def chat(messages, tools):
        chat.sent.append(json.loads(json.dumps(messages)))
        name = "done" if len(chat.sent) > 1 else "list_dir"
        return {"role": "assistant", "content": None, "reasoning": "dropped", "reasoning_details": thought,
                "tool_calls": [call(name, summary="x")]}, {}
    chat.sent = []
    agent.run(proj, chat)
    assert chat.sent[1][2] == {"role": "assistant", "content": None, "reasoning_details": thought,
                               "tool_calls": [call("list_dir", summary="x")]}


def test_the_offered_tools_are_the_mcp_tools_minus_the_bound_project():
    plain = {t["function"]["name"]: t["function"] for t in agent.tool_schemas()}
    assert set(plain) == {"evaluate", "build", "preview", "paint", "read_file", "write_file", "edit_file", "list_dir",
                          "view_image", "done", "copy_attempt", "search_file"}
    assert "name" not in plain["evaluate"]["parameters"]["properties"]      # an attempt is named by its directory
    assert all("project" not in f["parameters"]["properties"] for f in plain.values())
    ev = plain["evaluate"]["parameters"]
    assert ev["required"] == ["script"] and "judge_against" not in ev["properties"]
    assert plain["preview"]["parameters"]["properties"]["views"] == {"type": "array", "items": {"type": "string"}}
    judged = {t["function"]["name"]: t["function"] for t in agent.tool_schemas(judge=True)}
    assert "judge" in judged and "judge_against" in judged["evaluate"]["parameters"]["properties"]
    assert judged["evaluate"]["parameters"]["properties"]["samples"] == {"type": "integer"}


def test_a_new_mcp_parameter_must_be_classified(monkeypatch):
    def sneaky(project: str, target: str) -> list:
        """Writes wherever it is told."""
        return []
    monkeypatch.setattr(mcp_server, "TOOLS", mcp_server.TOOLS + (sneaky,))
    with pytest.raises(ValueError, match="neither a path nor a text"):
        agent.tool_schemas()


# --- the backend --------------------------------------------------------------------------------------------------

def test_the_agent_backend_has_its_own_variables_and_no_default_model():
    with pytest.raises(llm.LLMError, match="needs a model"):
        llm.resolve_backend({"OPENROUTER_API_KEY": "k"}, load_file=False, prefix=llm.AGENT_PREFIX)
    b = llm.resolve_backend({"MESH_JIG_AGENT_MODEL": "x/y", "OPENROUTER_API_KEY": "k", "MESH_JIG_JUDGE_MODEL": "no"},
                            load_file=False, prefix=llm.AGENT_PREFIX)
    assert (b.name, b.model, b.key) == ("openrouter", "x/y", "k")
    direct = llm.resolve_backend({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}, load_file=False,
                                 prefix=llm.AGENT_PREFIX, url="https://api.anthropic.com/v1/", model="m")
    assert (direct.url, direct.key, direct.name) == ("https://api.anthropic.com/v1", "a", "openai-compatible")
    with pytest.raises(llm.LLMError, match="MESH_JIG_AGENT_KEY \\(or GEMINI_API_KEY\\)"):
        llm.resolve_backend({}, load_file=False, prefix=llm.AGENT_PREFIX, model="m",
                            url="https://generativelanguage.googleapis.com/v1beta/openai")
    assert llm.effort_extra(b, "high") == {"reasoning": {"effort": "high"}}
    assert llm.effort_extra(direct, "low") == {"reasoning_effort": "low"} and llm.effort_extra(direct, None) == {}


def test_no_thinking_is_an_explicit_effort_not_an_omitted_default():
    args = cli.parser().parse_args(["agent", "p", "--model", "m", "--effort", "none"])
    assert args.effort == "none"
    router = llm.Backend("openrouter", llm.DEFAULT_URL, "m", "k")
    direct = llm.Backend("openai-compatible", "http://x/v1", "m", "k")
    assert llm.effort_extra(router, args.effort) == {"reasoning": {"effort": "none"}}
    assert llm.effort_extra(direct, args.effort) == {"reasoning_effort": "none"}
    assert llm.effort_extra(router, None) == {}


def test_the_agent_chat_sends_tools_and_no_sampling_defaults():
    sent = {}

    def poster(url, payload, key, timeout_s):
        sent.update(payload)
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call("done")]}}],
                "usage": {"prompt_tokens": 5}}

    backend = llm.Backend("openai-compatible", "http://x/v1", "m", "k")
    out = llm.chat(backend, [], temperature=None, max_tokens=None, tools=[{"t": 1}], extra={"reasoning_effort": "low"},
                   poster=poster)
    assert sent == {"model": "m", "messages": [], "tools": [{"t": 1}], "reasoning_effort": "low"}
    assert out["message"]["tool_calls"][0]["function"]["name"] == "done" and out["text"] == ""


@pytest.mark.parametrize("capacity", [262144, 65536, 32768])
def test_agent_default_uses_the_selected_models_advertised_output_capacity(capacity):
    import io
    backend = llm.Backend("openrouter", llm.DEFAULT_URL, "chosen/model", "k")
    requests = []

    def metadata(url, timeout):
        requests.append((url, timeout))
        return io.BytesIO(json.dumps({"data": [
            {"id": "other/model", "top_provider": {"max_completion_tokens": 999999}},
            {"id": "chosen/model", "top_provider": {"max_completion_tokens": capacity}},
        ]}).encode())

    assert llm.agent_output_limit(backend, opener=metadata) == capacity
    assert requests == [(llm.DEFAULT_URL + "/models", 10)]


def test_agent_explicit_output_cap_and_other_endpoints_need_no_catalog():
    def forbidden(*args, **kwargs):
        pytest.fail("metadata should not be fetched")

    router = llm.Backend("openrouter", llm.DEFAULT_URL, "m", "k")
    local = llm.Backend("local", "http://localhost:1234/v1", "m")
    assert llm.agent_output_limit(router, 8192, opener=forbidden) == 8192
    assert llm.agent_output_limit(local, opener=forbidden) is None


@pytest.mark.parametrize("document", [b'{', b'{}', b'{"data":[]}',
    b'{"data":[{"id":"m","top_provider":{"max_completion_tokens":null}}]}'])
def test_agent_output_capacity_missing_or_bad_metadata_keeps_endpoint_default(document):
    import io
    backend = llm.Backend("openrouter", llm.DEFAULT_URL, "m", "k")
    assert llm.agent_output_limit(backend, opener=lambda *a, **kw: io.BytesIO(document)) is None


def test_agent_output_capacity_network_failure_is_nonfatal():
    def offline(*args, **kwargs):
        raise OSError("offline")
    assert llm.agent_output_limit(llm.Backend("openrouter", llm.DEFAULT_URL, "m", "k"), opener=offline) is None


def test_agent_command_sends_resolved_capacity_to_the_real_chat_payload(proj, monkeypatch):
    backend = llm.Backend("openrouter", llm.DEFAULT_URL, "m", "k")
    monkeypatch.setattr(llm, "resolve_backend", lambda **kw: backend)
    limits = []
    monkeypatch.setattr(llm, "agent_output_limit", lambda b, requested, **kw: limits.append((b, requested)) or 262144)
    payloads = []

    def fake_stream(b, messages, **kw):
        payloads.append(llm.shape_payload(b, messages, None, kw["max_tokens"]))
        return {"message": {"role": "assistant", "content": None, "tool_calls": [call("done", summary="x")]},
                "usage": {}}

    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    assert cli.main(["agent", str(proj.root)]) == 0
    assert limits == [(backend, None)]
    assert payloads[0]["max_tokens"] == 262144


def test_the_command_refuses_to_start_without_a_model(proj, capsys):
    assert cli.main(["agent", str(proj.root)]) == 2
    assert "needs a model" in capsys.readouterr().err
    assert not (Path(proj.root) / agent.RUNS_DIR).exists()


def test_the_command_ends_by_saying_how_to_see_the_results(proj, capsys, monkeypatch):
    monkeypatch.setattr(llm, "resolve_backend", lambda **kw: llm.Backend("openai-compatible", "http://x/v1", "m", "k"))
    monkeypatch.setattr(agent, "chat_for", lambda *args, **kw: None)
    monkeypatch.setattr(agent, "run", lambda *args, **kw: agent.Report(model="m", done=True, evaluations=1))
    assert cli.main(["agent", str(proj.root)]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    where = f'"{proj.root}"' if " " in str(proj.root) else proj.root
    assert last.endswith(f"mesh-jig view {where} --open")
