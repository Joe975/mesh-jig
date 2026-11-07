"""`mesh-jig agent --workflow rig` driven by a scripted chat function: no model call, no Blender."""
import copy
import json

import pytest

from glbkit import write_glb
from mesh_jig import cli, rig_agent
from test_rig_tools import FIGURE, SPEC, walk

GATED = {**SPEC, "clips": {"walk": {"seconds": 1, "loop": True, "ground_drop_mm_max": 15}}}


@pytest.fixture
def root(tmp_path):
    write_glb(tmp_path / "source.glb", FIGURE)
    (tmp_path / "rig.json").write_text(json.dumps(GATED))
    (tmp_path / "brief.md").write_text("Make the figure walk in place: the legs take turns.")
    return tmp_path


def tool(name, /, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def scripted(replies, seen=None):
    replies = iter(replies)

    def chat(messages, schemas):
        if seen is not None:
            seen.append((copy.deepcopy(messages), schemas))    # the loop goes on changing its own list
        if not schemas:                                    # the fresh reviewer: no tools
            return {"role": "assistant", "content": "REVIEW: the legs take turns; keep it."}, {"prompt_tokens": 7, "completion_tokens": 3}
        return next(replies), {"prompt_tokens": 100, "completion_tokens": 10}
    return chat


def last_tool_text(messages):
    return next(m["content"] for m in reversed(messages) if m["role"] == "tool")


def test_a_run_from_a_failed_build_to_a_delivered_rig(root):
    seen = []
    chat = scripted([
        tool("write_file", path="attempts/a01/rig.py", text=walk(-1, -0.05)),
        tool("build", script="attempts/a01/rig.py"),                                   # sinks 50 mm: out of contract
        tool("edit_file", path="attempts/a01/rig.py", edits=[{"old": '(0, 0, -0.05)', "new": "(0, 0, 0)"}]),
        tool("build", script="attempts/a01/rig.py", note="feet back on the ground"),
        tool("edit_file", path="attempts/a01/rig.py", edits=[{"old": "30 *", "new": "40 *"}]),   # built: frozen
        tool("look", attempt="attempts/a01", clip="walk", times=[0, 0.25], views=["left"], skeleton=True),
        tool("look", attempt="attempts/a01", clip="walk", times=[0.5, 2.5]),                       # 2.5 s of a 1 s clip
        tool("look", attempt="attempts/a01", clip="walk", times=[0.25], views=["left"], aim=[-0.1, 0, -0.2], span=0.4),
        tool("done", best="a01", summary="walks"),                                                   # a build is unused
        tool("copy_attempt", source="a01", name="a02"),
        tool("build", script="attempts/a02/rig.py"),
        tool("done", best="a01", summary="walks"),                                                   # the first, not the last
    ], seen)
    rep = rig_agent.run(root, chat, builds=2, out=root / "agent" / "run", reviewer=True, log=lambda s: None)
    assert rep["done"] and rep["final"] == "a01" and rep["builds"] == 2 and rep["failed_builds"] == 1
    assert [a["ok"] for a in rep["attempts"]] == [False, True, True]
    assert (root / "animated.glb").read_bytes() == (root / "attempts" / "a01" / "animated.glb").read_bytes()
    assert (root / "rig.py").read_text() == (root / "attempts" / "a01" / "rig.py").read_text()
    saved = json.loads((root / "agent" / "run" / "report.json").read_text())
    assert "harness" in saved and saved["final"] == "a01" and saved["prompt_tokens"] == 1214 and (root / "agent" / "run" / "transcript.jsonl").is_file()

    builder = [messages for messages, schemas in seen if schemas]
    opening = builder[0][1]["content"]
    said = " ".join(p["text"] for p in opening if p["type"] == "text")
    for needed in ("Make the figure walk in place", "IT STANDS ON z = -0.400", "ROTATIONS, worked through once",
                   "clip `walk`", "more than 15 mm below", "You have 2 builds"):
        assert needed in said
    assert sum(p["type"] == "image_url" for p in opening) == 2                        # the ruled rest pose, four views

    failed = last_tool_text(builder[2])                                               # what the first build answered
    assert failed.startswith("OUT OF CONTRACT") and "sinks below its rest ground (50.0 mm, 15 allowed)" in failed
    assert "! goes 50 mm below the ground" in failed and "left and right take turns" in failed
    assert "INDEPENDENT REVIEW" not in failed                                         # only a build in contract is reviewed
    pictures = builder[2][-1]
    assert pictures["role"] == "user" and sum(p["type"] == "image_url" for p in pictures["content"]) == 2
    assert not (root / "attempts" / "a01" / "failed.glb").exists()                    # cleared by the build that passed

    passed = last_tool_text(builder[4])
    assert passed.startswith("BUILT AND IN CONTRACT") and "[1 build left.]" in passed
    assert "AN INDEPENDENT REVIEW" in passed and "REVIEW: the legs take turns" in passed
    assert (root / "attempts" / "a01" / "review.md").read_text().startswith("REVIEW:")
    reviewer = [messages for messages, schemas in seen if not schemas]
    assert len(reviewer) == 2 and len(reviewer[0]) == 2                               # fresh: the system prompt and one message
    shown = reviewer[0][1]["content"]
    assert "Make the figure walk in place" in shown[0]["text"] and sum(p["type"] == "image_url" for p in shown) == 1
    assert "rig.py" not in json.dumps([p for p in shown if p["type"] == "text"])      # no script, no numbers

    assert "built in contract and stays as it was built" in last_tool_text(builder[5])
    assert "walk at 0, 0.25 s" in next(m["content"] for m in builder[6] if m["role"] == "tool" and "walk at" in m["content"])
    assert (root / "attempts" / "a01" / "look_01_walk.png").is_file()
    assert "walk is 1 s long and `times` are seconds, not a phase: 2.5 is outside 0..1" in last_tool_text(builder[7])
    assert "walk at 0.25 s, 0.4 m across, centred on [-0.1, 0, -0.2]" in "".join(
        m["content"] for m in builder[8] if m["role"] == "tool")
    assert (root / "attempts" / "a01" / "look_02_walk.png").is_file() and rep["looks"] == 2
    refused = last_tool_text(builder[9])
    assert refused.startswith("NOT YET: 1 of the 2 builds are unused") and "need not be the last" in refused
    assert "That was the last build" in "".join(m["content"] for m in builder[11] if m["role"] == "tool")


def test_writes_are_confined_and_the_builds_are_a_budget(root):
    seen = []
    chat = scripted([
        tool("write_file", path="rig.json", text="{}"),
        tool("write_file", path="attempts/a01/other.py", text="x"),
        tool("write_file", path="../escape/rig.py", text="x"),
        tool("done", best="a01", summary="nothing yet"),
        tool("write_file", path="attempts/a01/rig.py", text="import os\n" + walk()),
        tool("build", script="attempts/a01/rig.py"),
        tool("write_file", path="attempts/a01/rig.py", text=walk().replace("30 *", "undefined *")),
        tool("build", script="attempts/a01/rig.py"),
        tool("write_file", path="attempts/a01/rig.py", text=walk()),
        tool("build", script="attempts/a01/rig.py"),
        tool("copy_attempt", source="attempts/a01", name="a02"),
        tool("build", script="attempts/a02/rig.py"),
        tool("done", best="attempts/a02", summary="unused"),
    ], seen)
    rep = rig_agent.run(root, chat, builds=1, max_turns=13, reviewer=False, log=lambda s: None)
    texts = [last_tool_text(m) for m, s in seen[1:]]
    assert texts[0].startswith("REJECTED") and texts[1].startswith("REJECTED") and texts[2].startswith("REJECTED")
    assert rep["rejected_calls"] == 3 and json.loads((root / "rig.json").read_text()) == GATED
    assert texts[3].startswith("NOT YET: nothing has been built")
    assert texts[5].startswith("BUILD FAILED") and "rig script refused" in texts[5]
    assert texts[7].startswith("BUILD FAILED") and "NameError" in texts[7] and "rig.py line" in texts[7] and "in pose" in texts[7]
    assert texts[9].startswith("BUILT AND IN CONTRACT") and "That was the last build" in texts[9]
    assert texts[11].startswith("REFUSED: the 1 builds are used")
    assert rep["builds"] == 1 and rep["failed_builds"] == 2 and rep["error"].startswith("hit the turn cap")
    assert rep["done"] is False and "a02" not in (rep["final"] or "")                 # a02 was never built in contract
    assert rep["final"] == "a01" and (root / "animated.glb").is_file()                # the last build is delivered


def test_no_build_is_reviewed_unless_the_run_asks_for_a_reviewer(root):
    """Three runs with the fresh reviewer and three without came out the same, the ones with it slower
    (docs/KNOWN_GAPS.md): it is asked for, not given."""
    seen = []
    chat = scripted([tool("write_file", path="attempts/a01/rig.py", text=walk()), tool("build", script="attempts/a01/rig.py"),
                     tool("done", best="a01", summary="")], seen)
    rep = rig_agent.run(root, chat, builds=1, log=lambda s: None)
    assert rep["done"] and rep["reviewer"] is False and rep["reviews"] == 0
    assert all(schemas for messages, schemas in seen)                                  # no call without tools
    built = last_tool_text(seen[2][0])
    assert built.startswith("BUILT AND IN CONTRACT") and "INDEPENDENT REVIEW" not in built
    assert not (root / "attempts" / "a01" / "review.md").exists()
    assert "review" not in rig_agent.SYSTEM.lower()                                    # nor is the builder promised one
    parser = cli.parser()
    assert parser.parse_args(["agent", "x", "--workflow", "rig"]).reviewer is False
    assert parser.parse_args(["agent", "x", "--workflow", "rig", "--reviewer"]).reviewer is True


def test_a_builder_stuck_on_one_gate_is_told_to_change_its_approach(root):
    seen = []
    sunk = [tool("build", script="attempts/a01/rig.py")] * 3
    chat = scripted([tool("write_file", path="attempts/a01/rig.py", text=walk(-1, -0.05))] + sunk + [
        tool("write_file", path="attempts/a01/rig.py", text=walk()), tool("build", script="attempts/a01/rig.py"),
        tool("done", best="a01", summary="")], seen)
    rep = rig_agent.run(root, chat, builds=1, reviewer=False, log=lambda s: None)
    answers = [last_tool_text(m) for m, s in seen[2:]]
    assert "builds in a row out of contract" not in answers[0] + answers[1]
    assert "[3 builds in a row out of contract." in answers[2] and "the approach is wrong and not its numbers" in answers[2]
    assert answers[4].startswith("BUILT AND IN CONTRACT") and "in a row" not in answers[4]
    assert rep["failed_builds"] == 3 and rep["builds"] == 1 and rep["done"]


def test_history_is_cut_on_what_the_endpoint_counted_and_not_on_a_guess(root, monkeypatch):
    """agent_state's own estimate counts 4096 a picture. On it, a run whose calls were 20,000 tokens was cut back to
    its checkpoint in twelve turns of 21 and forgot each build as it made the next."""
    limits, real = [], rig_agent.agent_state.compact
    monkeypatch.setattr(rig_agent.agent_state, "compact", lambda m, s, c, limit, r: limits.append(limit) or real(m, s, c, limit, r))
    sizes = iter([100, 100, 100, 85_000, 100])

    def chat(messages, schemas):
        return next(replies), {"prompt_tokens": next(sizes), "completion_tokens": 1}

    replies = iter([tool("write_file", path="attempts/a01/rig.py", text=walk()), tool("build", script="attempts/a01/rig.py"),
                    tool("read_file", path="attempts/a01/rig.py"), tool("read_file", path="attempts/a01/rig.py"),
                    tool("done", best="a01", summary="")])
    rep = rig_agent.run(root, chat, builds=1, reviewer=False, context_tokens=90_000, reply_reserve=8192, log=lambda s: None)
    assert limits == [0, 0, 0, 0, 90_000] and rep["done"] and rep["context_compactions"] == 0


def test_a_reply_with_no_tool_call_is_answered_with_what_is_left_and_three_in_a_row_end_the_run(root):
    talk = {"role": "assistant", "content": "The next step is a softer shoulder blend."}
    seen = []
    chat = scripted([tool("write_file", path="attempts/a01/rig.py", text=walk()), talk, tool("build", script="attempts/a01/rig.py"),
                     talk, talk, tool("copy_attempt", source="a01", name="a02"), talk, talk, talk], seen)
    rep = rig_agent.run(root, chat, builds=2, reviewer=False, log=lambda s: None)
    told = [m[-1]["content"] for m, s in seen if m[-1]["role"] == "user" and isinstance(m[-1]["content"], str)]
    assert told[0].startswith("Nothing was called. 2 of the 2 builds are unused: do what you just described now")
    assert told[1].startswith("Nothing was called. 1 of the 2 builds are unused")
    assert rep["turns"] == 9 and rep["error"] == "stopped without done()" and rep["summary"] == talk["content"]
    assert rep["final"] == "a01"                                       # what it had built is still delivered


def test_a_dropped_call_is_made_again_and_a_run_of_them_ends_the_run_with_its_last_build_delivered(root, monkeypatch):
    monkeypatch.setattr(rig_agent, "RETRY_WAIT_S", 0)
    replies = iter([tool("write_file", path="attempts/a01/rig.py", text=walk()), ConnectionResetError("reset"),
                    tool("build", script="attempts/a01/rig.py"), TimeoutError("slow"), OSError("down"), OSError("down")])

    def chat(messages, schemas):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"prompt_tokens": 1, "completion_tokens": 1}

    rep = rig_agent.run(root, chat, builds=3, reviewer=False, log=lambda s: None)
    assert rep["dropped_calls"] == 4 and rep["builds"] == 1 and rep["turns"] == 3
    assert rep["error"].startswith("chat failed: 3 calls in a row were dropped") and not rep["done"]
    assert rep["final"] == "a01" and (root / "animated.glb").is_file()


def test_a_directory_that_is_not_a_rig_directory_is_refused(tmp_path, capsys):
    (tmp_path / "rig.json").write_text(json.dumps(GATED))
    assert rig_agent.problems(tmp_path) == ["no source.glb in " + tmp_path.name + "/", "no brief.md in " + tmp_path.name + "/"]
    with pytest.raises(ValueError, match="no source.glb"):
        rig_agent.run(tmp_path, scripted([]))
    assert cli.main(["agent", str(tmp_path), "--workflow", "rig", "--model", "anything"]) == 2
    assert "RIG DIRECTORY PROBLEMS" in capsys.readouterr().err
    write_glb(tmp_path / "source.glb", FIGURE)
    (tmp_path / "brief.md").write_text("walk")
    (tmp_path / "rig.json").write_text(json.dumps({"clips": {}}))
    assert rig_agent.problems(tmp_path) == ["rig.json: rig contract needs clips"]


def test_the_prompt_says_what_the_first_rigs_got_wrong():
    """Each line is something a builder with a shell found out the slow way (docs/KNOWN_GAPS.md); the readings and
    the brief carry the rest."""
    for said in ("t is in SECONDS", "A parent's rotation carries its children", 'every line starting with "!"',
                 "skeleton and weights first", "its right is X>0", "Where the brief has A RECIPE, start from its",
                 "is failed by weights and pivots, not by motion"):
        assert said in rig_agent.SYSTEM
    names = [t["function"]["name"] for t in rig_agent.TOOLS]
    assert names == ["write_file", "edit_file", "read_file", "copy_attempt", "build", "look", "done"]


def test_a_run_the_harness_breaks_still_delivers_its_last_build(root):
    """A sound run of the same loop lost its builds to an error the loop raised itself (a compaction that could not
    fit); a rig run would have too."""
    replies = iter([tool("write_file", path="attempts/a01/rig.py", text=walk()), tool("build", script="attempts/a01/rig.py"),
                    RuntimeError("the harness broke")])

    def chat(messages, schemas):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"prompt_tokens": 1, "completion_tokens": 1}

    rep = rig_agent.run(root, chat, builds=3, reviewer=False, log=lambda s: None)
    assert rep["error"] == "RuntimeError: the harness broke" and rep["final"] == "a01"
    assert (root / "animated.glb").is_file()
