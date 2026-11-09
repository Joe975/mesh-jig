"""An isolated visual critique loop and a paired control. Run with python -m mesh_jig.gauntlet."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from . import agent, articulation, brief, cpu, evaluate, glb, llm, project, readiness, references
from . import review as review_mod

MODEL = "claude-haiku-5-5"
URL = "https://api.anthropic.com/v1"
MAX_TOKENS = 65536  # adaptive thinking and visible tool arguments share this budget
IMPLEMENTOR = """You are the implementor in a tenacious 3D modelling gauntlet. Build a complete model
matching ALL reference views. Prioritize silhouette, anatomy, proportions, recognizable features and coherent
surfaces, then materials and small details. A separate adversarial reviewer examines every submitted build.
Use its concrete visual feedback to revise your previous script. The reviewer can misread a picture: before acting
on a fix, find what it describes in the comparison images, and leave a part it calls wrong that matches the
reference. Preserve what works. You receive the previous
script and review each round; do not restart from primitives unless that solves a stated structural problem.
Use the file tools to write or edit the current attempt, then call submit. A failed build may be repaired within
this round. submit is the only build operation; a successful submission ends your round automatically.
Only the current attempt's model script and atlas_plan.json may be changed. The only files you can read are the
model scripts and atlas plans under attempts/, earlier rounds' included: there is no ledger or evaluation record
to read, and all you are given about the previous build is in the round's opening message. No shell or external
files.
The harness exports and renders your model. Follow the supplied script contract exactly."""
REVIEWER = """You are an independent adversarial art reviewer. Compare the supplied reference views against
the candidate renders: comparison images have REFERENCE LEFT, CANDIDATE RIGHT. Inspect every view and close-up.
Find visible mismatches in silhouette, proportions, anatomy, pose, recognizable parts, colour placement and
surface quality. Do not invent defects or reward complexity. Distinguish uncertain perspective or lighting
differences from modelling errors. Be candid about what is still wrong and what should be preserved.
Give a short verdict, then at most five prioritized fixes. Each fix must name the affected view and part,
describe the observed mismatch and give an actionable geometric or paint change, including direction and
approximate relative magnitude when supported. Finish with features to preserve. You see no numeric scores,
builder reasoning, earlier reviews or iteration number. Judge only the images and design brief."""
SUBMIT = agent._fn("submit", "Build, render and submit the current attempt for adversarial review.",
                   {"note": agent.S}, [])
COMPARER = """You are an independent judge. Two candidate 3D builds, A and B, of the same subject are each shown
beside the same reference views. Decide which is closer to the reference as a whole, as a viewer would see it:
silhouette and proportions in every view first, then the parts a viewer would name and where they sit, then colour
placement. Do not reward detail for its own sake. Answer with A or B alone on the first line, then at most three
sentences of why."""
TURNS_LEFT_NOTE = 3         # a conversation's last turns with no build that passed are counted down to the builder
CONVERSATIONS = 2           # a round with no build that passed starts once more as a new conversation
RULED_CAPTION = ("Reference, ruled in Blender metres (the lines are round coordinates along the axes labelled in "
                 "the margins, red at 0: read where a part sits and compare it with the numbers in your script)")
RULED_PAIR_CAPTION = ("Previous build beside the reference, both at one scale on the same Blender-metre lines "
                      "(reference left, yours right): where a part of yours ends against a line, compare where the "
                      "reference's ends")
# A reply that ran into the output cap is left out of the conversation and the builder is told why. Thinking and
# the reply share the cap: one first call spent 32,768 tokens thinking and wrote nothing, and a second-round rewrite
# took 63,234 of 65,536.
CUT_NOTE = ("[Your last reply was cut off at the output limit of {cap} tokens, which your thinking and your reply "
            "share. Nothing in it was kept or run. Settle on the next step with less deliberation and act on it; "
            "send a long script in parts, a write_file of the first part and edit_file calls that add the rest.]")
STEER = "steer.md"          # a note from the person running the loop, in the project directory
STEER_HEAD = ("NOTE FROM THE PERSON RUNNING THIS BUILD (they are watching your builds; where this note and the "
              "reviewer disagree, follow this note):\n")


def take_steer(root: Path, attempt: Path) -> str:
    """The person's note, if they left one at <project>/steer.md: kept in the attempt it was given to and taken
    away, so each note reaches the builder once. The builder cannot write it: its writes stay in its attempt."""
    path = root / STEER
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    try:
        path.unlink()
    except OSError:
        return ""  # still being written or held open: the next turn takes it
    if text:
        with (attempt / STEER).open("a", encoding="utf-8") as f:
            f.write(text + "\n\n")
    return text


def tell(messages: list, text: str) -> None:
    """Add text to what the builder reads next: the last user or tool message, so no turn order changes."""
    last = messages[-1]
    if isinstance(last.get("content"), list):
        last["content"].append({"type": "text", "text": text})
    else:
        last["content"] = (last.get("content") or "") + "\n\n" + text


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    # Windows readers/virus scanners can briefly deny replacing an open file.
    # Keep the old checkpoint intact and retry the atomic replacement, bounded.
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (attempt + 1))


def backend(model: str = MODEL) -> llm.Backend:
    # Deliberately bypass MESH_JIG_AGENT_KEY: an OpenRouter override must never be sent to Anthropic.
    llm.load_env_file()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise llm.LLMError("Set ANTHROPIC_API_KEY in the environment or .env; no provider fallback is used")
    return llm.Backend("anthropic", URL, model, key, timeout_s=600)


def pictures(paths: dict, caption: str) -> list[dict]:
    parts = []
    for name, path in sorted(paths.items()):
        parts += [{"type": "text", "text": f"{caption}: {name}"},
                  {"type": "image_url", "image_url": {"url": llm.image_to_data_url(str(path), max_dim=1536)}}]
    return parts


def visual_brief(proj: project.Project) -> str:
    return (proj.brief + "\n\n" + brief.model_facts(proj.spec) + "\nPALETTE (sRGB): "
            + json.dumps(proj.palette)
            + ("\n" + brief.linear_palette(proj.palette) if proj.palette and proj.spec.get("vertex_colors") else "")
            + ("\nFIXED REGIONS / PARTS / VISIBILITY / CLEARANCE CONTRACT (Blender metres; "
               "preserve exact declared node names):\n" + json.dumps(proj.review)
               if proj.review else "")
            + ("\n" + articulation.brief_text(proj.review) if articulation.brief_text(proj.review) else ""))


# What a reviewer is shown of a build, after the plain references: the key of the evaluation's record and the caption.
# The ruled pairs are the same views at one scale on fixed lines: where a part ends, which the comparison pictures
# (each scaled to its own subject) cannot show. The close-ups are the marked regions enlarged.
REVIEW_PICTURES = (
    ("compare", "Reference left, candidate right"),
    ("ruled", "Reference left, candidate right, at ONE scale on the same fixed lines: compare where each part starts "
              "and ends against the lines"),
    ("closeups", "Region close-up: reference left, candidate right"))


def review_pictures(proj: project.Project, rec: dict) -> list[tuple[str, dict]]:
    """Every picture a reviewer of this build is shown, in order, as (caption, name -> path). One list for the loop's
    own reviewer and for `review_brief`, so a reviewer started any other way is shown what the loop's is."""
    return [("Reference", dict(proj.references))] + [(caption, rec.get(key) or {}) for key, caption in REVIEW_PICTURES]


def review_brief(proj: project.Project, rec: dict, attempt: Path) -> str:
    """The loop's reviewer for a builder that is not in the loop: the reviewer's instructions, the design brief and
    the pictures to open, as text to hand to a fresh reviewer (a coding agent's subagent, a person). Builders with a
    shell were told only to "obtain a fresh visual critique", and several runs reviewed their own work or left no
    review. ValueError for an attempt with no evaluation that passed."""
    if not rec.get("ok") or not rec.get("compare"):
        raise ValueError(f"{attempt.name} has no evaluation that built and rendered: there is nothing to review")
    shown = [f"- {caption}: {name}: {path}" for caption, paths in review_pictures(proj, rec)
             for name, path in sorted(paths.items())]
    return "\n".join([
        f"REVIEW BRIEF for {attempt.name}. Give a fresh reviewer everything below the line and the pictures it lists, "
        "and nothing else: no scores, no notes of yours, no earlier review, no other attempt. A reviewer that has "
        "seen your reasoning repeats it. This is what `mesh-jig agent` gives its own reviewer.",
        f"Save the reviewer's answer, as it gave it, in {attempt / 'review.md'}.",
        "-" * 78,
        REVIEWER,
        "",
        "DESIGN BRIEF:",
        proj.brief.strip(),
        "",
        "PICTURES (open every one, in this order):",
        *shown])


class Files:
    """Reuse existing file confinement, with a strict script/plan allowlist and one writable attempt."""

    def __init__(self, proj, current: Path):
        self.proj, self.current = proj, current
        self.tools = agent.Tools(proj, agent.Report(), attempts=1, judge=False)
        self.names = {glb.script_name(proj.spec), "atlas_plan.json"}
        self.schemas = [t for t in agent.FILE_TOOLS
                        if t["function"]["name"] in {"read_file", "write_file", "edit_file"}] + [SUBMIT]

    def call(self, name, args):
        if name not in {"read_file", "write_file", "edit_file"}:
            raise agent.Rejected(f"unknown file tool {name}")
        path = agent.in_attempt(self.proj.root, args.get("path", ""), file=True)
        if path.name not in self.names or path.parent.parent != self.proj.attempts_dir:
            raise agent.Rejected("only attempt model scripts and atlas_plan.json are readable")
        if name != "read_file" and path.parent != self.current:
            raise agent.Rejected("only the current attempt may be changed")
        return getattr(self.tools, "t_" + name)(args)[0]


class BudgetStop(RuntimeError):
    """The run's time budget is spent: no further model call is made."""


# Compatibility for callers of the original visual-loop helper.
parse_tool_batch = llm.parse_tool_batch


def run(proj: project.Project, chat_fn, *, iterations=10, max_turns=24, model=MODEL,
        max_tokens=MAX_TOKENS, evaluate_fn=None, log=print, checks="none",
        out=None, url=URL, max_total_turns=0, max_wall=0, keep_better=False) -> dict:
    if checks not in ("none", "builder", "both"):
        raise ValueError("checks must be none, builder or both")
    if iterations < 1 or max_turns < 1:
        raise ValueError("iterations and max-turns must be positive")
    standalone = out is None
    out = Path(out) if out is not None else proj.root / "gauntlet"
    if out.exists() or (standalone and any(proj.attempts_dir.glob("*/" + glb.script_name(proj.spec)))):
        raise ValueError("gauntlet needs a fresh project; existing runs are never overwritten")
    out.mkdir(parents=True)
    report = {"workflow": "visual-gauntlet", "checks": checks, "model": model, "url": url,
              "max_tokens": max_tokens, "max_turns_per_iteration": max_turns,
              "iterations_requested": iterations, "iterations_completed": 0, "done": False,
              "error": "", "rounds": [], "calls": [], "wall_s": 0,
              "evaluations": 0, "attempts": [], "best": None, "best_numeric": None,
              "latest": None, "turns": 0, "recoveries": [], "steers": [], "keep_better": keep_better, "kept": None,
              "comparisons": []}
    start = time.monotonic()
    previous, feedback, incumbent_review = None, "", ""
    script_file = glb.script_name(proj.spec)
    ruled = references.write_ruled(proj, out / "ruled")    # the builder's references; the reviewer's stay plain

    def save():
        report["wall_s"] = round(time.monotonic() - start, 1)
        write_json(out / "report.json", report)

    def call(role, messages, schemas):
        left_s = max_wall - (time.monotonic() - start) if max_wall else None
        if left_s is not None and left_s <= 0:
            raise BudgetStop("stopped at time budget before next model call")
        if role == "implementor":
            if max_total_turns and report["turns"] >= max_total_turns:
                raise RuntimeError("stopped at total implementor turn cap")
            report["turns"] += 1
        call_start = time.monotonic()
        # under a budget the call gets what the run has left, so one slow call cannot run on past it
        try:
            message, usage = chat_fn(messages, schemas) if left_s is None else chat_fn(messages, schemas, left_s)
        except Exception as error:
            # a call cut short because the run's time ran out is the budget stop, whatever the transport raised
            if max_wall and time.monotonic() - start >= max_wall - 1:
                raise BudgetStop(f"stopped at time budget during a model call ({error})") from error
            raise
        report["calls"].append({"role": role, "usage": usage,
                                "wall_s": round(time.monotonic() - call_start, 3)})
        with (out / "transcript.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "messages": agent.strip_images(messages),
                                "response": message, "usage": usage}) + "\n")
        save()
        return message

    def build(attempt, note):
        plan = attempt / "atlas_plan.json"
        needs_plan = proj.spec.get("textured") and glb.texture_from(proj.spec) != "script"
        kwargs = {"paint_plan": plan if needs_plan and plan.is_file() else None, "note": note, "log": log}
        if evaluate_fn is not None:
            return evaluate_fn(proj, attempt / script_file, attempt, **kwargs)

        def append_successful_ledger(rec):
            if rec.get("ok"):
                evaluate.append_ledger(proj.ledger, rec)

        return evaluate.evaluate(proj, attempt / script_file, attempt,
                                 **kwargs, _postprocess=append_successful_ledger)

    def on_disk(attempt):
        """What a submission would build: the script's and the plan's bytes."""
        digest = hashlib.sha256()
        for filename in (script_file, "atlas_plan.json"):
            path = attempt / filename
            digest.update(path.read_bytes() if path.is_file() else b"\0")
        return digest.hexdigest()

    def conversation(index, attempt, files, parts):
        """One conversation of a round: the build that passed (or None) and the last build error."""
        messages = [{"role": "system", "content": IMPLEMENTOR}, {"role": "user", "content": parts}]
        rec, error, built = None, "", None
        given = on_disk(attempt)
        for turn in range(max_turns):
            note = take_steer(proj.root, attempt)
            if note:
                tell(messages, STEER_HEAD + note)
                report["steers"].append({"round": index, "turn": turn, "chars": len(note)})
                log(f"gauntlet round {index}: the person's note ({len(note)} characters) given to the builder")
            try:
                message = call("implementor", messages, files.schemas)
            except BudgetStop:
                # the script written so far is built before the run stops: a build costs seconds, the
                # thinking that wrote it cost the budget
                if (attempt / script_file).is_file() and on_disk(attempt) not in (built, given):
                    log(f"gauntlet round {index}: the run's budget ran out with the script on disk not submitted; "
                        "submitting it")
                    try:
                        rec = build(attempt, "submitted by the loop when the time budget ran out")
                    except (ValueError, OSError, TypeError):
                        rec = None
                    report["recoveries"].append({"round": index, "kind": "budget-submitted",
                                                 "ok": bool(rec and rec.get("ok"))})
                    if rec and rec.get("ok"):
                        return rec, ""
                raise
            usage = report["calls"][-1]["usage"]
            if llm.finish_reason(usage) == "refusal":
                raise RuntimeError("the model declined the request (stop reason: refusal); nothing was built from it")
            if llm.cut_short(usage):
                # the turn stays out of the conversation (the transcript has it): a tool call in it may have lost
                # its end, and thinking that was cut cannot be sent back
                tell(messages, CUT_NOTE.format(cap=max_tokens))
                report["recoveries"].append({"round": index, "turn": turn + 1, "kind": "reply-cut", "ok": False})
                log(f"gauntlet round {index}: the builder's reply was cut at the {max_tokens}-token output cap; "
                    "it is told so and asked again")
                save()
                continue
            message = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
            message["role"] = "assistant"
            try:
                calls = message.get("tool_calls")
                parsed = parse_tool_batch([] if calls is None else calls)
            except (ValueError, TypeError) as failure:
                # A partial JSON call is not valid provider history. Keep the
                # original response in the transcript, but execute none of its
                # batch and do not echo invalid calls into the next request.
                messages.extend(llm.rejected_tool_turn(message, failure))
                report["recoveries"].append({"round": index, "turn": turn + 1,
                                             "kind": "invalid-tool-batch", "ok": False})
                save()
                continue
            messages.append(message)
            if not parsed:
                messages.append({"role": "user", "content": "Use the file tools, then submit the current model."})
            for tool, fn, args in parsed:
                try:
                    if fn.get("name") == "submit":
                        built = on_disk(attempt)
                        rec = build(attempt, str(args.get("note") or ""))
                        if rec.get("ok"):
                            result = "Submitted successfully. This round is complete."
                        else:
                            error = str(rec.get("error"))
                            result = "BUILD FAILED; repair the current script and submit again:\n" + error
                    else:
                        result = files.call(fn.get("name"), args)
                except (agent.Rejected, ValueError, OSError, TypeError) as failure:
                    result = f"ERROR: {failure}"
                messages.append({"role": "tool", "tool_call_id": tool["id"], "content": result})
                if rec and rec.get("ok"):
                    return rec, ""
            left = max_turns - turn - 1
            if 0 < left <= TURNS_LEFT_NOTE:
                messages[-1]["content"] += (
                    f"\n\n[{left} of this round's {max_turns} turns {'is' if left == 1 else 'are'} left and no build "
                    "has passed. Call submit now: a round that ends with no passing build delivers nothing. "
                    "Spend what is left on the smallest repair that makes the build pass.]")
        if (attempt / script_file).is_file() and built != on_disk(attempt):
            log(f"gauntlet round {index}: the turns ran out with the script on disk not submitted; submitting it")
            try:
                rec = build(attempt, "submitted by the loop when the round's turns ran out")
                error = "" if rec.get("ok") else str(rec.get("error"))
            except (ValueError, OSError, TypeError) as failure:
                rec, error = None, str(failure)
            report["recoveries"].append({"round": index, "kind": "loop-submitted", "ok": bool(rec and rec.get("ok"))})
            if rec and rec.get("ok"):
                return rec, ""
        return None, error

    def better_than(index, name, rec, incumbent) -> bool:
        """Whether a fresh call of the reviewer, shown both builds beside the references, picks the new one. The
        order alternates by round (the new build is A in even rounds), against a preference for a position; an answer
        that names neither keeps the new build."""
        new_first = index % 2 == 0
        first, second = (rec, incumbent) if new_first else (incumbent, rec)
        parts = [{"type": "text", "text": proj.brief}] + pictures(proj.references, "Reference")
        parts += pictures(first.get("compare") or {}, "Candidate A: reference left, candidate A right")
        parts += pictures(second.get("compare") or {}, "Candidate B: reference left, candidate B right")
        # the marked regions enlarged for both, so a face's eyes and glasses weigh as they do in the review
        parts += pictures(first.get("closeups") or {}, "Candidate A, a marked region enlarged: reference left, A right")
        parts += pictures(second.get("closeups") or {}, "Candidate B, a marked region enlarged: reference left, B right")
        log(f"gauntlet round {index}/{iterations}: comparing with {incumbent['name']}")
        said = str(call("reviewer", [{"role": "system", "content": COMPARER},
                                     {"role": "user", "content": parts}], []).get("content") or "").strip()
        letter = said[:1].upper()
        keep_new = letter not in ("A", "B") or (letter == "A") == new_first
        report["comparisons"].append({"round": index, "new": name, "incumbent": incumbent["name"],
                                      "kept": name if keep_new else incumbent["name"], "said": said[:600]})
        (proj.attempts_dir / name / "comparison.md").write_text(said + "\n", encoding="utf-8")
        return keep_new

    def restart_text(index, name, attempt, error):
        script = attempt / script_file
        text = (f"Round {index}/{iterations}, started again. Current directory: attempts/{name}/. An earlier "
                f"conversation used this round's {max_turns} turns with no build that passed. This is a new "
                "conversation and holds none of its messages. ")
        if not script.is_file():
            return text + "It left no script on disk. Write a complete model plain enough to build, then submit."
        source = script.read_text(encoding="utf-8", errors="replace")
        text += ("The script it left is on disk. Make the smallest repair that lets it build, then submit; improve "
                 "the model only after a build has passed.\nLAST BUILD ERROR:\n"
                 + (error or "none recorded: the script was never built") + "\n")
        if len(source) > agent.MAX_FILE_READ:
            return text + f"Read attempts/{name}/{script_file} before changing it."
        return text + f"SCRIPT ON DISK (attempts/{name}/{script_file}):\n{source}"

    try:
        for index in range(1, iterations + 1):
            round_start = time.monotonic()
            calls_before = len(report["calls"])
            name = f"a{index:02d}" if standalone else f"{out.name}-a{index:02d}"
            attempt = proj.attempts_dir / name
            attempt.mkdir(parents=True, exist_ok=False)
            if previous:
                for filename in (glb.script_name(proj.spec), "atlas_plan.json"):
                    source = proj.attempts_dir / previous["name"] / filename
                    if source.is_file():
                        shutil.copy2(source, attempt / filename)
            files = Files(proj, attempt)

            def carried():
                """The files copied in from the previous round, in full. Left to read_file, which gives 200 lines a
                call, a 693-line script took three turns to read before a line of it was changed."""
                said = []
                for filename in (script_file, "atlas_plan.json"):
                    path = attempt / filename
                    if not path.is_file():
                        continue
                    source = path.read_text(encoding="utf-8", errors="replace")
                    where = f"attempts/{name}/{filename}"
                    said.append(f"{where} is too long to show here: read it before you change it."
                                if len(source) > agent.MAX_FILE_READ else
                                f"{where}, in full as it is on disk now (a copy of attempts/{previous['name']}/"
                                f"{filename}):\n{source}")
                return said

            def opening(text, files_shown=()):
                parts = [{"type": "text", "text": visual_brief(proj)}] + pictures(
                    {v: p for v, p in proj.references.items() if v not in ruled}, "Reference")
                parts += pictures(ruled, RULED_CAPTION)
                parts += [{"type": "text", "text": text + ("\nADVERSARIAL FEEDBACK:\n" + feedback if previous else "")}]
                parts += [{"type": "text", "text": shown} for shown in files_shown]
                if previous:
                    metric = previous.get("ruled") or {}
                    parts += pictures({v: p for v, p in (previous.get("compare") or {}).items() if v not in metric},
                                      "Previous build: reference left, candidate right")
                    parts += pictures(metric, RULED_PAIR_CAPTION)
                    # the marked regions enlarged, as the reviewer saw them: a rim half sunk in a face is a few
                    # pixels of a whole view
                    parts += pictures(previous.get("closeups") or {},
                                      "Previous build, a marked region enlarged: reference left, yours right")
                    if checks != "none":
                        parts += [{"type": "text", "text": "JIG CHECKS FOR PREVIOUS BUILD:\n"
                                   + evaluate.feedback(evaluate.portable(previous, proj.attempts_dir / previous["name"], proj.root))}]
                    elif (previous.get("readiness") or {}).get("checks"):
                        # a project built for a target: the builder hears how its mesh read against it, whoever
                        # else is shown the jig's checks
                        parts += [{"type": "text", "text": "MESH READINESS OF PREVIOUS BUILD (the target's limits; "
                                   "read beside the review, not part of it):\n"
                                   + "\n".join(([readiness.summary(previous["readiness"])]
                                                if readiness.summary(previous["readiness"]) else [])
                                               + readiness.lines(previous["readiness"]))}]
                    if previous.get("hidden"):
                        # nor why a part the reviewer calls missing is missing: it is inside another
                        parts += [{"type": "text", "text": "OBJECTS OF PREVIOUS BUILD THAT NO CAMERA SEES MUCH OF "
                                   "(the share of each that shows; read beside the review):\n"
                                   + "\n".join(review_mod.hidden_lines(previous["hidden"]))}]
                    if previous.get("floating"):
                        parts += [{"type": "text", "text": "SAMPLED SEPARATION IN PREVIOUS BUILD "
                                   "(read beside the review):\n"
                                   + "\n".join(review_mod.floating_lines(previous["floating"]))}]
                    if previous.get("flipped"):
                        parts += [{"type": "text", "text": "OBJECTS OF PREVIOUS BUILD WITH TRIANGLES WOUND THE WRONG "
                                   "WAY (read beside the review):\n"
                                   + "\n".join(review_mod.flipped_lines(previous["flipped"]))}]
                    failed = articulation.failed_lines((previous.get("review") or {}).get("checks") or [])
                    if checks == "none" and failed:
                        # the one kind of gate a blind builder hears: whether a rig can bend what it built is not
                        # in the pictures the reviewer judges
                        parts += [{"type": "text", "text": "ARTICULATIONS OF PREVIOUS BUILD:\n" + "\n".join(failed)}]
                return parts

            log(f"gauntlet round {index}/{iterations}: implementor")
            shown = carried() if previous else []
            rec, error = conversation(index, attempt, files, opening(
                f"Round {index}/{iterations}. Current directory: attempts/{name}/. "
                + (("Your previous script and plan are" if len(shown) > 1 else "Your previous script is")
                   + " already copied here and shown below, after the feedback. Revise "
                   + ("them." if len(shown) > 1 else "it.")
                   if previous else "Create your first complete model."), shown))
            for _again in range(1, CONVERSATIONS):
                if rec:
                    break
                log(f"gauntlet round {index}/{iterations}: no build passed in {max_turns} turns; a new conversation")
                rec, error = conversation(index, attempt, files, opening(restart_text(index, name, attempt, error)))
                report["recoveries"].append({"round": index, "kind": "restarted", "ok": bool(rec)})
            if not rec:
                raise RuntimeError(f"round {index} did not produce a successful build in {CONVERSATIONS} "
                                   f"conversations of {max_turns} turns")
            if evaluate_fn is not None:
                evaluate.append_ledger(proj.ledger, rec)
            candidate = {"name": name, "numeric_score": rec["numeric"]["numeric_score"]}
            report["evaluations"] += 1
            report["attempts"].append({**candidate, "ok": True, "reviewed": False})
            report["latest"] = candidate
            if report["best_numeric"] is None or candidate["numeric_score"] > report["best_numeric"]["numeric_score"]:
                report["best_numeric"] = candidate
                report["best"] = candidate  # compatibility for older viewer/report readers
            save()  # Preserve a measured candidate even if its review hits a limit or fails.
            if keep_better and previous is not None and not better_than(index, name, rec, previous):
                # judged further from the reference: the next round starts again from the one it was to replace
                feedback = (incumbent_review + f"\n\nNOTE: your round-{index} build was judged further from the "
                            f"reference than {previous['name']}, so this round starts again from {previous['name']}'s "
                            "script and the review above is of that build. Make a different change.")
                report["attempts"][-1]["kept"] = False
                report["rounds"].append({"name": name, "numeric_score": rec["numeric"]["numeric_score"],
                                         "kept": False, "wall_s": round(time.monotonic() - round_start, 3),
                                         "elapsed_s": round(time.monotonic() - start, 3),
                                         "calls_start": calls_before, "calls_end": len(report["calls"]),
                                         "failed_gates": evaluate.failed_gates(rec)})
                report["iterations_completed"] = index
                save()
                continue
            log(f"gauntlet round {index}/{iterations}: adversarial reviewer")
            review_parts = [{"type": "text", "text": proj.brief}]
            for caption, paths in review_pictures(proj, rec):
                review_parts += pictures(paths, caption)
            reviewer_prompt = REVIEWER
            if checks == "both":
                review_parts += [{"type": "text", "text": "JIG CHECKS FOR CURRENT BUILD:\n"
                                  + evaluate.feedback(evaluate.portable(rec, attempt, proj.root))}]
                reviewer_prompt = REVIEWER.replace(
                    "You see no numeric scores,\nbuilder reasoning, earlier reviews or iteration number. Judge only the images and design brief.",
                    "You also receive deterministic jig checks. Use them as diagnostic evidence alongside the images; "
                    "a higher score is not proof of visual quality. Explain conflicts with the images. "
                    "You see no builder reasoning, earlier reviews or iteration number.")
            if not rec.get("compare"):
                raise RuntimeError("no comparison images available for the reviewer")
            review_messages = [{"role": "system", "content": reviewer_prompt}, {"role": "user", "content": review_parts}]
            response = call("reviewer", review_messages, [])
            feedback = response.get("content") or ""
            if llm.cut_short(report["calls"][-1]["usage"]):
                # what there is of it still goes to the builder; the report says it is not the whole review
                report["recoveries"].append({"round": index, "kind": "review-cut",
                                             "ok": bool(isinstance(feedback, str) and feedback.strip())})
                log(f"gauntlet round {index}: the review was cut at the {max_tokens}-token output cap")
            if not isinstance(feedback, str) or not feedback.strip():
                raise RuntimeError("reviewer returned no critique; refusing to advance without feedback")
            (attempt / "review.md").write_text(feedback + "\n", encoding="utf-8")
            incumbent_review = feedback
            report["attempts"][-1]["reviewed"] = True
            report["rounds"].append({"name": name, "numeric_score": rec["numeric"]["numeric_score"],
                                     "wall_s": round(time.monotonic() - round_start, 3),
                                     "elapsed_s": round(time.monotonic() - start, 3),
                                     "calls_start": calls_before, "calls_end": len(report["calls"]),
                                     "failed_gates": evaluate.failed_gates(rec),
                                     "review": f"attempts/{name}/review.md"})
            report["iterations_completed"] = index
            previous = rec
            if keep_better:          # the build every comparison so far has kept: the run's pick by eye
                report["kept"] = candidate
            save()
        report["done"] = True
    except Exception as error:
        report["error"] = str(error)
        log(f"gauntlet stopped: {error}")
    finally:
        save()
    return report


def seed_files(proj: project.Project) -> list[Path]:
    paths = [proj.root / "jig.json", *map(Path, proj.references.values())]
    for relative in (proj.data.get("brief") or "brief.md", (proj.data.get("sheet") or {}).get("source")):
        if relative:
            candidate = agent.inside(proj.root, str(relative))
            if candidate.is_file():
                paths.append(candidate)
    for path in paths:
        if not path.resolve().is_relative_to(proj.root):
            raise ValueError("seed input paths must stay inside the source project")
    return sorted(set(paths))


def clone_seed(proj: project.Project, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in seed_files(proj):
        relative = source.relative_to(proj.root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def comparison(source: project.Project, out: Path, chat_fn, *, iterations=10, max_turns=24,
               model=MODEL, max_tokens=MAX_TOKENS, log=print, gauntlet_fn=run, control_fn=agent.run) -> dict:
    if iterations < 1 or max_turns < 1:
        raise ValueError("iterations and max-turns must be positive")
    if source.problems():
        raise ValueError("; ".join(source.problems()))
    if out.exists():
        raise ValueError("comparison output exists; choose a fresh directory")
    seed_files(source)  # validate before creating either arm
    out.mkdir(parents=True)
    hashes = clone_seed(source, out / "gauntlet")
    if hashes != clone_seed(source, out / "mesh-jig"):
        raise ValueError("comparison seeds differ")
    manifest = {"model": model, "url": URL, "iterations": iterations, "seed_sha256": hashes,
                "max_tokens": max_tokens, "max_turns_per_iteration": max_turns,
                "status": "running", "arms": {}}
    write_json(out / "comparison.json", manifest)

    def control():
        proj = project.load(out / "mesh-jig")
        return asdict(control_fn(proj, chat_fn, attempts=iterations, max_turns=iterations * max_turns,
                                 model=model, url=URL, out=proj.root / "agent" / "control", log=log))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(gauntlet_fn, project.load(out / "gauntlet"), chat_fn,
                               iterations=iterations, max_turns=max_turns, model=model,
                               max_tokens=max_tokens, log=log): "gauntlet",
                   pool.submit(control): "mesh-jig"}
        for future in as_completed(futures):
            arm = futures[future]
            try:
                manifest["arms"][arm] = future.result()
            except Exception as error:
                manifest["arms"][arm] = {"done": False, "error": str(error)}
            write_json(out / "comparison.json", manifest)
    manifest["status"] = "complete" if all(
        manifest["arms"][arm].get("iterations_completed", manifest["arms"][arm].get("evaluations", 0)) == iterations
        and not manifest["arms"][arm].get("error") for arm in ("gauntlet", "mesh-jig")) else "incomplete"
    write_json(out / "comparison.json", manifest)
    return manifest


def main(argv=None) -> int:
    cpu.govern()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "compare"))
    parser.add_argument("project", type=Path)
    parser.add_argument("--out", type=Path, help="fresh output directory (required for compare)")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=24, help="implementor turns per iteration; also control budget")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.max_turns < 1 or args.max_tokens < 1:
        parser.error("iteration, turn and token limits must be positive")
    if args.mode == "compare" and args.out is None:
        parser.error("compare requires --out")
    try:
        proj = project.load(args.project.resolve())
        if proj.problems():
            raise ValueError("; ".join(proj.problems()))
        api = backend(args.model)
        chat = agent.chat_for(api, max_tokens=args.max_tokens, idle_s=180, log=lambda s: print(s, flush=True))
        log = lambda s: print(s, flush=True)
        if args.mode == "compare":
            result = comparison(proj, args.out.resolve(), chat, iterations=args.iterations,
                                max_turns=args.max_turns, model=args.model, max_tokens=args.max_tokens, log=log)
            return 0 if result["status"] == "complete" else 1
        result = run(proj, chat, iterations=args.iterations, max_turns=args.max_turns,
                     model=args.model, max_tokens=args.max_tokens, log=log)
        return 0 if result["done"] else 1
    except (ValueError, OSError, llm.LLMError) as error:
        print(f"gauntlet: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
