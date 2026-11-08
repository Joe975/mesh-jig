"""mesh_jig.evaluate: the one call an agent makes per attempt, here with a fake build and a fake judge. The render
and every measure are the real ones."""
from __future__ import annotations

import errno
import json
import multiprocessing
import os
import tempfile
import threading
from pathlib import Path

import pytest
from PIL import Image

from glbkit import cube, write_cubes
from mesh_jig import build, cli, evaluate, project, render

LENGTH = 0.08
SHOTS = {s["name"]: s for s in render.DEFAULT_SHOTS}
RED, PALE = (0.6, 0.05, 0.03), (0.8, 0.76, 0.66)


def _append_ledger_process(path, name, entered=None, release=None, started=None, done=None, temp_tmp=None):
    """Child helper for exercising the public append path across real processes."""
    if temp_tmp is not None:
        os.environ["TEMP"], os.environ["TMP"] = temp_tmp
    from mesh_jig import evaluate as child_evaluate
    if entered is not None:
        original = child_evaluate.ledger_line
        def held_line(rec):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("parent did not release the held ledger writer")
            return original(rec)
        child_evaluate.ledger_line = held_line
    if started is not None:
        started.set()
    child_evaluate.append_ledger(Path(path), {"name": name, "ok": True})
    if done is not None:
        done.set()


def ship(path, span=0.04, nose=RED):
    """A slab 80 mm long along Z and `span` wide, pale, with a coloured nose block on top at the -Z end."""
    plain = {k: PALE for k in ("+x", "-x", "+y", "-y", "+z", "-z")}
    return write_cubes(path, [cube((span, 0.012, LENGTH), colours=plain),
                              cube((span * 0.6, 0.01, 0.02), (0.0, 0.011, -0.03), colours={k: nose for k in plain})])


def make_project(tmp_path: Path, **model) -> project.Project:
    """A project whose reference views are renders of the 40 mm wide ship: a known answer."""
    root = tmp_path / "proj"
    (root / "refs").mkdir(parents=True)
    truth = ship(tmp_path / "truth.glb")
    mesh = render.load(truth)
    for view in ("hero", "top", "side", "rear"):
        render.render(mesh, SHOTS[view], scale=1 / LENGTH).save(root / "refs" / f"{view}.png")
    (root / "brief.md").write_text("A pale slab with a red nose block.\n")
    (root / "jig.json").write_text(json.dumps({
        "name": "ship", "palette": {"red": "#c83c30", "pale": "#e6e0d2"},
        "model": {"script": "ship.py", "glb": "ship.glb", "length_m": LENGTH, "forward": "-Z", **model}}))
    return project.load(root)


def fake_build(span=0.04, nose=RED, error=""):
    def build_fn(script, out_dir, spec, **kw):
        if error:
            return error
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ship(Path(out_dir) / "ship.glb", span, nose)
        return ""
    return build_fn


def script(tmp_path, name="a01") -> Path:
    p = tmp_path / "proj" / "attempts" / name / "ship.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("def build():\n    pass\n")
    return p


def quick_render(spec, glb_dir, out_dir, views=None):
    """The real renderer at a quarter of the pixels: the measures work on masks of at most 512 px either way."""
    return render.render_views(spec, glb_dir, out_dir, views=views, size=render.PREVIEW_SIZE)


def run(proj, tmp_path, name="a01", **kw):
    s = script(tmp_path, name)
    return evaluate.evaluate(proj, s, s.parent, log=lambda *_: None, render_fn=quick_render, **kw)


def test_the_known_answer_measures_as_a_match_and_everything_is_written(tmp_path):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, note="the reference's own geometry", build_fn=fake_build())
    n = rec["numeric"]
    assert rec["ok"] and rec["stage"] == "measured"
    assert n["ortho_iou"] > 0.97 and n["zone"] > 0.97 and n["resemblance"] > 0.97, n
    # the references are perspective renders, so the end stations and a raised part read a little off; the body
    # stations of an exact match read zero
    assert n["mae_mm"] < 2.0, n
    assert all(abs(r["half_width"]) < 0.5 for r in rec["profile_diff"]["top"][1:-1])
    attempt = tmp_path / "proj" / "attempts" / "a01"
    for rel in ("eval.json", "numeric.json", "resemblance.json", "renders/top.png", "compare/top_vs_reference.png",
                "compare/sil_top.png", "compare/silhouette.json"):
        assert (attempt / rel).is_file(), rel
    assert json.loads((attempt / "eval.json").read_text())["numeric"] == n
    text = evaluate.feedback(rec)
    assert "built, in contract, rendered and measured" in text and "top_vs_reference.png" in text
    assert "YOUR MODEL AGAINST THE REFERENCE OUTLINES" in text


def test_the_feedback_says_which_views_the_score_is_taken_on():
    views = dict.fromkeys(("front", "hero", "rear", "side", "top"), {"iou": 0.5, "missing": 0.1, "extra": 0.1})
    assert evaluate.scored_views_lines(views) == [
        "overlap, resemblance, zones and the numeric score are the means over top, side, rear ONLY; front, hero are "
        "compared for you to look at and do not count (the fallback views: jig.json states no `criteria.views`)"]
    assert "hero is compared for you to look at and does not count" in evaluate.scored_views_lines(
        dict.fromkeys(("hero", "top"), {}))[0]
    assert evaluate.scored_views_lines(dict.fromkeys(("top", "side", "rear"), {})) == []       # all of them count
    assert evaluate.scored_views_lines({"hero": {}}) == []                                     # the only view counts
    rec = {"ok": True, "name": "a01", "numeric": {}, "silhouette": views, "resemblance": {}}
    assert "top, side, rear ONLY" in evaluate.feedback(rec)


def test_the_feedback_says_how_the_renders_colours_read_against_the_references(tmp_path):
    proj = make_project(tmp_path)
    same = run(proj, tmp_path, build_fn=fake_build())["zone_reading"]["top"]
    assert same["reference"] == same["yours"] and list(same["reference"]) == ["pale", "red"]
    assert same["read_as"] == {"red": {"red": 1.0}, "pale": {"pale": 1.0}}
    rec = run(proj, tmp_path, "a02", build_fn=fake_build(nose=PALE))            # the red nose block painted pale
    top = rec["zone_reading"]["top"]
    assert top["yours"] == {"pale": 1.0} and top["read_as"]["red"] == {"pale": 1.0} and top["cells"] == same["cells"]
    assert json.loads((tmp_path / "proj" / "attempts" / "a02" / "eval.json").read_text())["zone_reading"]["top"] == top
    text = evaluate.feedback(rec)
    assert "HOW YOUR COLOURS READ" in text and "where the reference is red, yours reads pale 100%" in text
    red = top["reference"]["red"]
    assert f"  top: the reference is pale {1 - red:.0%}, red {red:.0%}; yours reads pale 100%" in text
    assert "  hero:" not in text                                               # only the views the score is taken on


def test_a_wrong_shape_and_a_wrong_colour_each_move_their_own_number(tmp_path):
    proj = make_project(tmp_path)
    right = run(proj, tmp_path, "right", build_fn=fake_build())["numeric"]
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))["numeric"]
    recoloured = run(proj, tmp_path, "pale_nose", build_fn=fake_build(nose=PALE))["numeric"]
    assert wide["ortho_iou"] < right["ortho_iou"] - 0.1 and wide["mae_mm"] > right["mae_mm"] + 2
    assert recoloured["zone"] < right["zone"] - 0.02, "the nose block is the wrong colour: zones see it"
    assert abs(recoloured["ortho_iou"] - right["ortho_iou"]) < 0.01, "and the outline does not"
    assert wide["numeric_score"] < right["numeric_score"] and recoloured["numeric_score"] < right["numeric_score"]


def test_a_build_failure_is_a_failed_record_not_a_crash(tmp_path):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, build_fn=fake_build(error="variant default: JIG_BUILD_ERROR: boom"))
    assert not rec["ok"] and rec["stage"] == "build" and "boom" in rec["error"]
    assert "FAILED build" in evaluate.ledger_line(rec) and "FAILED at build" in evaluate.feedback(rec)
    assert json.loads((tmp_path / "proj" / "attempts" / "a01" / "eval.json").read_text())["ok"] is False


def test_failed_evaluations_keep_immutable_source_record_and_observed_outputs(tmp_path):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    ledger = proj.ledger
    snapshots = []

    def failed_build(span, message, mutate_source=False):
        def build_fn(_script, build_dir, _spec, **_kw):
            Path(build_dir).mkdir(parents=True, exist_ok=True)
            ship(Path(build_dir) / "ship.glb", span)
            if mutate_source:
                model.write_bytes(b"def build():\n    pass  # changed during build\n")
            return message
        return build_fn

    for index, (source, span, message) in enumerate(((b"def build():\n    pass  # first\n", 0.04, "first boom"),
                                                     (b"def build():\n    pass  # second\n", 0.06, "second boom"))):
        model.write_bytes(source)
        rec = evaluate.evaluate(proj, model, out, build_fn=failed_build(span, message, mutate_source=index == 0),
                                 render_fn=quick_render, log=lambda *_: None)
        evaluate.append_ledger(ledger, rec)
        assert not rec["ok"] and rec["stage"] == "build"
        snap = out / rec["failure_preservation"]["snapshot"]
        snapshots.append(snap)

    saved = []
    for snapshot, expected_source, expected_error in zip(
            snapshots, (b"def build():\n    pass  # first\n", b"def build():\n    pass  # second\n"),
            ("first boom", "second boom")):
        record = json.loads((snapshot / "eval.json").read_text(encoding="utf-8"))
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        source_bytes = (snapshot / "source.py").read_bytes()
        artifact = snapshot / "artifacts" / "out" / "ship.glb"
        artifact_bytes = artifact.read_bytes()
        row = next(item for item in manifest["outputs"]["files"] if item["observed_path"] == "out/ship.glb")
        eval_row = manifest["evaluation_record"]
        assert source_bytes == expected_source and record["error"].endswith(expected_error)
        assert manifest["source"]["sha256"] == __import__("hashlib").sha256(source_bytes).hexdigest()
        assert eval_row["sha256"] == __import__("hashlib").sha256((snapshot / "eval.json").read_bytes()).hexdigest()
        assert row["sha256"] == __import__("hashlib").sha256(artifact_bytes).hexdigest()
        assert row["status"] == "observed" and row["changed_since_entry"] is True
        assert manifest["outputs"]["certified_produced_by_failing_call"] == []
        saved.append((source_bytes, (snapshot / "eval.json").read_bytes(), artifact_bytes,
                      (snapshot / "manifest.json").read_bytes()))

    model.write_text("def build():\n    pass  # repaired\n", encoding="utf-8")
    success = evaluate.evaluate(proj, model, out, build_fn=fake_build(), render_fn=quick_render,
                                log=lambda *_: None)
    evaluate.append_ledger(ledger, success)
    assert success["ok"] and success["stage"] == "measured"
    latest = json.loads((out / "eval.json").read_text(encoding="utf-8"))
    assert latest["ok"] and latest["stage"] == "measured"
    assert evaluate.standing(proj, out, success)["others"] == 0
    from mesh_jig import streak
    verdict = streak.verdict(proj.root)
    assert verdict["made"] == 1 and verdict["left_out"] == 0 and verdict["measured_total"] == 1
    assert ledger.read_text(encoding="utf-8").count("| a01 | FAILED build |") == 2
    assert ledger.read_text(encoding="utf-8").count("| a01 | ok |") == 1
    for snapshot, original in zip(snapshots, saved):
        assert ((snapshot / "source.py").read_bytes(), (snapshot / "eval.json").read_bytes(),
                (snapshot / "artifacts" / "out" / "ship.glb").read_bytes(),
                (snapshot / "manifest.json").read_bytes()) == original


def test_early_failure_marks_old_outputs_as_observed_stale_state(tmp_path, monkeypatch):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    glb_path = out / "out" / "ship.glb"
    glb_path.parent.mkdir(parents=True)
    ship(glb_path)
    old_hash = __import__("hashlib").sha256(glb_path.read_bytes()).hexdigest()
    proj.data["criteria"] = {}
    monkeypatch.setattr(proj, "problems", lambda: ["fixture project error"])
    rec = evaluate.evaluate(proj, model, out, build_fn=lambda *_a, **_kw: pytest.fail("early failure built"),
                            render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "project" and rec["failure_preservation"]["status"] == "preserved"
    snapshot = out / rec["failure_preservation"]["snapshot"]
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    row = next(item for item in manifest["outputs"]["files"] if item["observed_path"] == "out/ship.glb")
    assert row["sha256"] == old_hash and row["changed_since_entry"] is False
    assert manifest["outputs"]["certified_produced_by_failing_call"] == []


def test_snapshot_io_failure_is_written_and_reported_without_masking_failure(tmp_path, monkeypatch):
    proj = make_project(tmp_path)
    model = script(tmp_path)

    def failed_snapshot(*_args, **_kwargs):
        raise OSError("snapshot write probe")

    monkeypatch.setattr(evaluate, "_preserve_failed_evaluation", failed_snapshot)
    rec = evaluate.evaluate(proj, model, model.parent, build_fn=fake_build(error="original build error"),
                            render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "build" and rec["error"].endswith("original build error")
    assert rec["failure_preservation"] == {"status": "failed", "error": "OSError: snapshot write probe"}
    assert "failure snapshot could not be saved" in evaluate.feedback(rec)
    latest = evaluate.read(model.parent)
    assert latest["stage"] == "build" and latest["failure_preservation"]["status"] == "failed"


def test_snapshot_excludes_symlink_targets_from_owned_output_tree(tmp_path):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    external = tmp_path / "outside-output.txt"
    external.write_text("must not be copied", encoding="utf-8")
    owned_out = out / "out"
    owned_out.mkdir()
    link = owned_out / "external-link.txt"
    try:
        os.symlink(external, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")
    rec = evaluate.evaluate(proj, model, out, build_fn=fake_build(error="link probe failed"),
                            render_fn=quick_render, log=lambda *_: None)
    assert rec["failure_preservation"]["status"] == "preserved"
    snapshot = out / rec["failure_preservation"]["snapshot"]
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert "out/external-link.txt" in manifest["outputs"]["excluded_symlinks"]
    assert not (snapshot / "artifacts" / "out" / "external-link.txt").exists()
    assert not any(path.read_text(encoding="utf-8") == "must not be copied"
                   for path in (snapshot / "artifacts").rglob("*.txt"))


def test_snapshot_refuses_a_symlinked_history_root(tmp_path):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    outside = tmp_path / "outside-history"
    outside.mkdir()
    try:
        os.symlink(outside, out / evaluate.FAILURE_SNAPSHOTS, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    rec = evaluate.evaluate(proj, model, out, build_fn=fake_build(error="preserve root probe"),
                            render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "build" and rec["failure_preservation"]["status"] == "failed"
    assert "not a plain directory" in rec["failure_preservation"]["error"]
    assert list(outside.iterdir()) == []


def test_snapshot_records_an_unreadable_source_without_claiming_it_was_saved(tmp_path):
    proj = make_project(tmp_path)
    missing = tmp_path / "project-secret" / "missing.py"
    out = tmp_path / "project" / "attempts" / "missing-source"
    rec = evaluate.evaluate(proj, missing, out, build_fn=lambda *_a, **_kw: "build did not run",
                            render_fn=quick_render, log=lambda *_: None)
    manifest = json.loads((out / rec["failure_preservation"]["snapshot"] / "manifest.json").read_text())
    assert manifest["source"]["path"] is None and "unreadable" in manifest["source"]["error"]


def test_portable_redaction_does_not_mutate_nested_failure_metadata(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    rec = {"failure_preservation": {"status": "failed", "error": f"denied at {attempt}\\secret"}}
    stored = evaluate.portable(rec, attempt, tmp_path)
    assert str(attempt) in rec["failure_preservation"]["error"]
    assert str(attempt) not in stored["failure_preservation"]["error"]


def test_a_model_no_camera_can_see_fails_at_render(tmp_path):
    proj = make_project(tmp_path)

    def tiny(script, out_dir, spec, **kw):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_cubes(Path(out_dir) / "ship.glb", [cube(0.001, (0.0, 5.0, 0.0))])     # a speck, five metres up
        return ""
    rec = run(proj, tmp_path, build_fn=tiny)
    assert not rec["ok"] and rec["stage"] == "render" and "NOTHING VISIBLE" in rec["error"]


def test_the_ledger_gets_one_header_and_one_line_per_attempt(tmp_path):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, note="baseline | first", build_fn=fake_build())
    evaluate.append_ledger(proj.ledger, rec)
    evaluate.append_ledger(proj.ledger, rec)
    text = proj.ledger.read_text()
    assert text.count("| attempt |") == 1 and text.count("| a01 | ok |") == 2
    assert "baseline / first" in text, "a pipe in a note would break the table"


def test_concurrent_process_writers_share_one_header_and_keep_both_rows(tmp_path):
    """The second process must wait until the first finishes its header decision and append."""
    path = tmp_path / "shared ledger.md"
    path.touch()
    ctx = multiprocessing.get_context("spawn")
    entered, release = ctx.Event(), ctx.Event()
    started, done = ctx.Event(), ctx.Event()
    temp_a, temp_b = tmp_path / "temp-a", tmp_path / "temp-b"
    temp_a.mkdir()
    temp_b.mkdir()
    first = ctx.Process(target=_append_ledger_process,
                        args=(str(path), "first", entered, release, None, None, (str(temp_a), str(temp_a))))
    second = ctx.Process(target=_append_ledger_process,
                         args=(str(path), "second", None, None, started, done, (str(temp_b), str(temp_b))))
    first.start()
    try:
        assert entered.wait(10), "first writer did not reach the header/append section"
        second.start()
        assert started.wait(10), "second writer did not start"
        assert not done.wait(0.2), "second writer entered while the first held ledger ownership"
    finally:
        release.set()
        first.join(10)
        if second.pid is not None:
            second.join(10)
    assert first.exitcode == 0 and second.exitcode == 0
    text = path.read_text(encoding="utf-8")
    assert text.count("| attempt |") == 1
    assert text.count("| first |") == 1 and text.count("| second |") == 1


def test_held_ledger_writer_times_out_without_changing_bytes_then_append_succeeds(tmp_path, monkeypatch):
    path = tmp_path / "shared ledger.md"
    original = "existing ledger bytes\n"
    path.write_text(original, encoding="utf-8")
    original_bytes = path.read_bytes()
    ctx = multiprocessing.get_context("spawn")
    entered, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_append_ledger_process, args=(str(path), "held", entered, release))
    child.start()
    try:
        assert entered.wait(10), "child writer did not reach its held critical section"
        monkeypatch.setattr(evaluate, "_LEDGER_LOCK_TIMEOUT_SECONDS", 0.1)
        with pytest.raises(TimeoutError, match="ledger busy; wait and retry"):
            evaluate.append_ledger(path, {"name": "timed-out", "ok": True})
        assert path.read_bytes() == original_bytes, "a timed-out writer changed the ledger"
    finally:
        release.set()
        child.join(10)
    assert not child.is_alive() and child.exitcode == 0
    evaluate.append_ledger(path, {"name": "after-release", "ok": True})
    text = path.read_text(encoding="utf-8")
    assert "| held |" in text and "| after-release |" in text
    assert "timed-out" not in text


def test_posix_lock_retries_contention_and_propagates_other_errors(monkeypatch):
    class FakeFcntl:
        LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4

        def __init__(self, errors):
            self.errors = iter(errors)
            self.calls = []

        def flock(self, fd, operation):
            self.calls.append((fd, operation))
            error = next(self.errors, None)
            if error is not None:
                raise error

    class FakeFile:
        def fileno(self):
            return 17

    now = [0.0]
    monkeypatch.setattr(evaluate.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(evaluate.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))
    monkeypatch.setattr(evaluate, "_LEDGER_LOCK_TIMEOUT_SECONDS", 0.2)
    busy = OSError(errno.EAGAIN, "busy")
    fcntl = FakeFcntl([busy, OSError(errno.EACCES, "busy"), None])
    evaluate._acquire_posix_ledger_lock(FakeFile(), fcntl)
    assert [operation for _, operation in fcntl.calls] == [3, 3, 3]  # LOCK_EX | LOCK_NB

    unexpected = OSError(errno.EIO, "disk error")
    with pytest.raises(OSError) as caught:
        evaluate._acquire_posix_ledger_lock(FakeFile(), FakeFcntl([unexpected]))
    assert caught.value is unexpected

    monkeypatch.setattr(evaluate, "_LEDGER_LOCK_TIMEOUT_SECONDS", 0.0)
    with pytest.raises(TimeoutError, match="ledger busy; wait and retry"):
        evaluate._acquire_posix_ledger_lock(FakeFile(), FakeFcntl([busy]))


def test_concurrent_thread_writers_share_one_header_and_keep_both_rows(tmp_path, monkeypatch):
    path = tmp_path / "shared ledger.md"
    path.touch()
    entered, release, started, done = (threading.Event() for _ in range(4))
    original = evaluate.ledger_line
    def held_line(rec):
        if rec["name"] == "first":
            entered.set()
            assert release.wait(10), "test did not release the held thread writer"
        return original(rec)
    monkeypatch.setattr(evaluate, "ledger_line", held_line)
    first = threading.Thread(target=evaluate.append_ledger, args=(path, {"name": "first", "ok": True}))
    def append_second():
        started.set()
        evaluate.append_ledger(path, {"name": "second", "ok": True})
        done.set()
    second = threading.Thread(target=append_second)
    first.start()
    try:
        assert entered.wait(10), "first writer did not reach the header/append section"
        second.start()
        assert started.wait(10), "second writer did not start"
        assert not done.wait(0.2), "second thread entered while the first held ledger ownership"
    finally:
        release.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
    assert not first.is_alive() and not second.is_alive()
    text = path.read_text(encoding="utf-8")
    assert text.count("| attempt |") == 1
    assert text.count("| first |") == 1 and text.count("| second |") == 1


def test_ledger_writer_exception_releases_ownership(tmp_path, monkeypatch):
    path = tmp_path / "shared ledger.md"
    def fail(_rec):
        raise RuntimeError("line failure")
    monkeypatch.setattr(evaluate, "ledger_line", fail)
    with pytest.raises(RuntimeError, match="line failure"):
        evaluate.append_ledger(path, {"name": "broken", "ok": True})
    monkeypatch.undo()
    evaluate.append_ledger(path, {"name": "after", "ok": True})
    text = path.read_text(encoding="utf-8")
    assert text.count("| attempt |") == 1 and "| after |" in text


def test_process_exit_releases_ledger_writer_and_unlocked_reader_stays_usable(tmp_path):
    path = tmp_path / "shared ledger.md"
    path.write_text("existing ledger bytes\n", encoding="utf-8")
    ctx = multiprocessing.get_context("spawn")
    entered, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_append_ledger_process, args=(str(path), "held", entered, release))
    child.start()
    try:
        assert entered.wait(10), "child writer did not reach its held critical section"
        assert path.read_text(encoding="utf-8") == "existing ledger bytes\n", "ordinary reader could not read while writer was held"
    finally:
        child.terminate()
        child.join(10)
    assert not child.is_alive()
    evaluate.append_ledger(path, {"name": "after-crash", "ok": True})
    assert "| after-crash |" in path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "nt", reason="Windows legal-path regression for sidecar-free ledger ownership")
def test_ledger_writer_does_not_need_a_long_sidecar_path(tmp_path):
    with tempfile.TemporaryDirectory(prefix="lw-") as temporary:
        scratch = Path(temporary)
        # Fit both parent components and the filename to the actual temp root.
        desired_path_length = 250
        filename = "l" * 45 + ".md"
        parent_component_budget = desired_path_length - len(str(scratch)) - len(filename) - 3
        first_component_length = parent_component_budget // 2
        second_component_length = parent_component_budget - first_component_length
        assert 1 <= first_component_length <= 120
        assert 1 <= second_component_length <= 120
        parent = scratch / ("p" * first_component_length) / ("q" * second_component_length)
        parent.mkdir(parents=True)
        path = parent / filename
        assert len(str(path)) < 260
        assert len(str(path)) >= 245
        assert len(str(path) + ".mesh-jig-ledger.lock") > 260
        evaluate.append_ledger(path, {"name": "long-path", "ok": True})
        assert "| long-path |" in path.read_text(encoding="utf-8")


def test_the_contracts_limits_on_the_measures_are_recorded_as_gates(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0, placement_min=0.9)
    good = run(proj, tmp_path, "good", build_fn=fake_build())
    bad = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    assert {g["name"]: g["passed"] for g in good["gates"]} == {"profile_mae_max": True, "placement_min": True}
    assert {g["name"]: g["passed"] for g in bad["gates"]}["profile_mae_max"] is False
    assert bad["ok"], "a failed gate is a measured attempt, reported, not a build failure"
    assert "gate profile_mae_max: FAILED" in evaluate.feedback(bad)
    assert evaluate.gates({}, good["numeric"]) == [], "no limit set: recorded, nothing gated"


def test_the_verdict_comes_first_and_a_failed_gate_says_what_it_needed(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0, placement_min=0.5)
    good = evaluate.feedback(run(proj, tmp_path, "good", build_fn=fake_build())).splitlines()
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    n, said = wide["numeric"], evaluate.feedback(wide).splitlines()
    # four decimals: late in a run two attempts differ in the fourth, and at three they read as a tie
    assert said[0].endswith(f"numeric score {n['numeric_score']:.4f}; 1 of 2 gates FAILED (profile_mae_max)")
    assert said[1] == f"gate profile_mae_max: FAILED - outline error (mm) {n['mae_mm']:.3f} (max 2.0)"
    assert said[2] == f"gates passed: placement_min (colour zones in place {n['zone']:.3f}, min 0.5)"
    assert good[0].endswith("; all 2 gates pass") and good[1].startswith("gates passed: profile_mae_max (outline error")
    # a reader that keeps the first lines has the verdict: the views and the outline table come after it
    first_view = next(i for i, line in enumerate(said) if line.startswith("top: silhouette overlap"))
    table = next(i for i, line in enumerate(said) if line.startswith("YOUR MODEL AGAINST THE REFERENCE OUTLINES"))
    assert 2 < first_view < table and not any(line.startswith("gate") for line in said[3:])
    assert "{" not in "\n".join(said), "no record is printed as a dict"
    plain = run(make_project(tmp_path / "plain"), tmp_path / "plain", build_fn=fake_build())
    assert evaluate.feedback(plain).splitlines()[0].endswith(f"numeric score {plain['numeric']['numeric_score']:.4f}")


def test_opt_in_surface_summary_follows_gate_feedback_without_moving_legacy_prefix(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0, placement_min=0.5)
    rec = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    legacy_reading = {k: v for k, v in rec["readiness"].items() if k != "surface_symmetry"}
    legacy = dict(rec, readiness=legacy_reading)
    surface = {"status": "available", "left_to_right": 1.0, "right_to_left": 1.0, "agreement": 1.0,
               "sample_count": 4096, "tolerance_mm": 1.0}
    opted_in = dict(rec, readiness={**legacy_reading, "surface_symmetry": surface})

    legacy_lines = evaluate.feedback(legacy).splitlines()
    opted_lines = evaluate.feedback(opted_in).splitlines()
    surface_line = next(i for i, line in enumerate(opted_lines) if "sampled mirrored-surface agreement" in line)
    failed_gate = next(i for i, line in enumerate(opted_lines) if line.startswith("gate profile_mae_max:"))
    passed_gate = next(i for i, line in enumerate(opted_lines) if line.startswith("gates passed:"))
    assert failed_gate < passed_gate < surface_line
    assert opted_lines[:surface_line] == legacy_lines[:surface_line]


def test_each_view_is_one_line_scored_views_first_and_the_pictures_directory_is_named_once(tmp_path):
    proj = make_project(tmp_path)
    text = evaluate.feedback(run(proj, tmp_path, build_fn=fake_build()))
    views = [line.split(":")[0] for line in text.splitlines() if "silhouette overlap" in line]
    assert views == ["top", "side", "rear", "hero (not scored)"], "hero is compared to be looked at, and says so"
    assert all("colour zones agree" in line for line in text.splitlines() if "silhouette overlap" in line)
    compare = str(tmp_path / "proj" / "attempts" / "a01" / "compare")
    assert text.count(compare) == 1
    assert (f"LOOK AT (reference left, your build right) in {compare}: top_vs_reference.png, side_vs_reference.png, "
            "rear_vs_reference.png; not scored: hero_vs_reference.png") in text
    assert ("outline diffs (grey both, red add body, blue remove body), same directory: sil_top.png, sil_side.png, "
            "sil_rear.png; not scored: sil_hero.png") in text


def test_an_attempt_is_told_where_it_stands_among_the_others(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0)
    at = lambda name: tmp_path / "proj" / "attempts" / name    # noqa: E731
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    assert evaluate.standing(proj, at("wide"), wide) == {"others": 0, "best": None, "clean": None}
    assert "standing: the first measured attempt here." in evaluate.feedback(wide, evaluate.standing(proj, at("wide"), wide))
    assert "standing" not in evaluate.feedback(wide), "nothing is claimed of the others when they were not read"
    good = run(proj, tmp_path, "good", build_fn=fake_build())
    gain = good["numeric"]["numeric_score"] - wide["numeric"]["numeric_score"]
    said = evaluate.feedback(good, evaluate.standing(proj, at("good"), good)).splitlines()
    line = next(i for i, x in enumerate(said) if x.startswith("standing:"))
    assert said[line] == (f"standing: the best of the 2 measured attempts, {gain:.4f} above wide "
                          f"({wide['numeric']['numeric_score']:.4f}). It passes profile_mae_max, which wide fails.")
    top = (good["silhouette"]["top"]["iou"], wide["silhouette"]["top"]["iou"])
    assert said[line + 1].startswith("  the score's terms against wide: overlap ")
    assert said[line + 2].startswith(f"  by scored view against wide: top overlap {top[0]:.3f} (was {top[1]:.3f})")
    assert "hero" not in said[line + 1], "only the views the score is taken over"
    back = evaluate.feedback(wide, evaluate.standing(proj, at("wide"), wide))
    assert (f"standing: {gain:.4f} below the best attempt, good ({good['numeric']['numeric_score']:.4f}). "
            "It fails profile_mae_max, which good passes.") in back
    failed = run(proj, tmp_path, "broken", build_fn=fake_build(error="JIG_BUILD_ERROR"))
    assert evaluate.standing(proj, at("broken"), failed) is None
    assert evaluate.standing(proj, at("good"), good)["others"] == 1, "a failed build is not one of the others"


def test_standing_names_the_highest_attempt_that_passes_every_gate(tmp_path):
    proj = project.Project(tmp_path, {"model": {}})

    def put(name, score, failed=(), started="2026-01-02 00:00:00"):
        rec = {"name": name, "ok": True, "started": started, "numeric": {"numeric_score": score},
               "gates": [{"name": g, "passed": g not in failed, "message": ""} for g in ("face", "tail")]}
        (proj.attempts_dir / name).mkdir(parents=True)
        (proj.attempts_dir / name / "eval.json").write_text(json.dumps(rec))
        return rec
    put("a1", 0.60)
    put("a2", 0.70, failed=("face",))
    put("a0", 0.70, failed=("face",), started="2026-01-01 00:00:00")    # ties with a2 and was measured first
    mine = put("a3", 0.69, failed=("tail",))
    st = evaluate.standing(proj, proj.attempts_dir / "a3", mine)
    assert (st["others"], st["best"]["name"], st["clean"]["name"]) == (3, "a0", "a1")
    assert evaluate.standing_lines(mine, st, []) == [
        "standing: 0.0100 below the best attempt, a0 (0.7000). It fails tail, which a0 passes. It passes face, which "
        "a0 fails. The highest attempt that passes every gate is a1 (0.6000).",
        "  historical full evaluation provenance unavailable for legacy records."]
    level = put("a4", 0.70, failed=("face",))
    assert evaluate.standing_lines(level, evaluate.standing(proj, proj.attempts_dir / "a4", level), []) == [
        "standing: level with a0 (0.7000), the best of the other attempts. The highest attempt that passes every "
        "gate is a1 (0.6000).", "  historical full evaluation provenance unavailable for legacy records."]


def test_the_ledger_says_the_gates_and_starts_its_table_again_under_older_columns(tmp_path):
    proj = make_project(tmp_path, profile_mae_max=2.0)
    wide = run(proj, tmp_path, "wide", build_fn=fake_build(span=0.07))
    good = run(proj, tmp_path, "good", build_fn=fake_build())
    assert evaluate.gate_tally(wide) == "0/1 (profile_mae_max)" and evaluate.gate_tally(good) == "1/1"
    assert (f"| {wide['numeric']['numeric_score']:.4f} | 0/1 (profile_mae_max) | {evaluate.triangles_text(wide)} | - |"
            in evaluate.ledger_line(wide))
    assert evaluate.triangles_text({"readiness": {"cost": {"triangles": 1980}, "max_triangles": 2000}}) == "1980/2000"
    assert evaluate.triangles_text({"readiness": {"cost": {"triangles": 1980}}}) == "1980"
    assert evaluate.triangles_text({"ok": False}) == "-"
    assert evaluate.gate_tally({"ok": False}) == "-" and evaluate.gate_tally({"gates": []}) == "-"
    old = ("# Attempts (mesh-jig eval)\n\n"
           "| attempt | build | outline mm | overlap | resemblance | zones | numeric | judge | note |\n"
           "|---|---|---|---|---|---|---|---|---|\n| a00 | ok | 1.00 | 0.900 | 0.500 | 0.50 | 0.700 | - | before |\n")
    proj.ledger.write_text(old)
    evaluate.append_ledger(proj.ledger, wide)
    evaluate.append_ledger(proj.ledger, good)
    text = proj.ledger.read_text()
    assert text.startswith(old) and text.count(evaluate.LEDGER_HEAD) == 1, "the new columns get their own header, once"
    assert text.index(evaluate.LEDGER_HEAD) < text.index("| wide | ok |") < text.index("| good | ok |")
    assert len(evaluate.ledger_line(good).split("|")) == len(evaluate.LEDGER_HEAD.splitlines()[0].split("|"))


def test_show_reads_an_evaluation_back_without_building_or_touching_the_ledger(tmp_path, capsys, monkeypatch):
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, "a01", build_fn=fake_build())
    attempt = tmp_path / "proj" / "attempts" / "a01"
    evaluate.append_ledger(proj.ledger, rec)
    ledger, record = proj.ledger.read_text(), (attempt / "eval.json").read_text()
    monkeypatch.setattr(build, "build_variants", lambda *a, **kw: pytest.fail("show builds nothing"))
    monkeypatch.setattr(render, "render_views", lambda *a, **kw: pytest.fail("show renders nothing"))
    said = evaluate.feedback(rec, evaluate.standing(proj, attempt, rec)).splitlines()
    for named in ("a01", str(attempt)):                       # by its name under attempts/, or by its directory
        assert cli.main(["show", str(proj.root), named]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == said[0] and out[1] == "standing: the first measured attempt here."
        assert any(line.startswith("LOOK AT") for line in out) and out[-1].startswith("-> ")
    assert proj.ledger.read_text() == ledger and (attempt / "eval.json").read_text() == record
    assert cli.main(["show", str(proj.root), "a09"]) == 2 and "no eval.json under a09" in capsys.readouterr().err


def test_the_textured_contract_wants_an_atlas_and_a_flat_one_refuses_it(tmp_path):
    flat = make_project(tmp_path)
    atlas = tmp_path / "atlas.png"
    Image.new("RGB", (8, 8), (200, 60, 50)).save(atlas)
    rec = run(flat, tmp_path, "flat", atlas=atlas, build_fn=fake_build())
    assert rec["stage"] == "build" and "not `textured`" in rec["error"]
    textured = make_project(tmp_path / "t", textured=True)
    s = tmp_path / "t" / "proj" / "attempts" / "a01" / "ship.py"
    s.parent.mkdir(parents=True)
    s.write_text("def build():\n    pass\n")
    rec = evaluate.evaluate(textured, s, s.parent, build_fn=fake_build(), render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "build" and "give an atlas PNG or a paint plan" in rec["error"]
    plan = s.parent / "atlas_plan.json"
    plan.write_text(json.dumps({"size": 16, "ops": [{"op": "blur"}]}))
    rec = evaluate.evaluate(textured, s, s.parent, paint_plan=plan, build_fn=fake_build(), render_fn=quick_render, log=lambda *_: None)
    assert rec["stage"] == "paint" and "PAINT_ERROR" in rec["error"]
    plan.write_text(json.dumps({"size": 16, "ops": [{"op": "rect", "box": [0, 0, 1, 1], "color": "red"}]}))
    seen = {}

    def build_fn(script, out_dir, spec, **kw):
        seen.update(kw)
        return fake_build()(script, out_dir, spec)
    rec = evaluate.evaluate(textured, s, s.parent, paint_plan=plan, build_fn=build_fn, render_fn=quick_render, log=lambda *_: None)
    assert rec["ok"] and Path(seen["atlas"]) == s.parent / "out" / "atlas.png"
    assert Image.open(seen["atlas"]).getpixel((8, 8)) == (0xc8, 0x3c, 0x30), "painted with the project's palette"


def test_the_judge_runs_only_when_given_a_backend_and_only_on_views_the_reference_has(tmp_path):
    proj = make_project(tmp_path)
    prev = run(proj, tmp_path, "a00", build_fn=fake_build(span=0.07))
    Image.new("RGB", (8, 8)).save(Path(prev["renders"]["top"]).parent / "extra.png")     # a view with no reference
    calls = []

    def pairwise_fn(backend, refs, cand, inc, brief, out_dir, samples=8):
        calls.append((sorted(cand), sorted(inc), brief.strip(), samples))
        return {"win_rate": 0.85, "ci": [0.6, 1.0], "decisive": True, "ties": 0, "judgements": 32, "calls": 8,
                "position_bias": 0.5, "views": {"top": 0.9}, "usage": {"cost": 0.02}, "reasons": {"top": ["narrower, as the reference"]}}

    against = tmp_path / "proj" / "attempts" / "a00"
    rec = run(proj, tmp_path, "a01", build_fn=fake_build(), against=against, backend=object(), samples=8, pairwise_fn=pairwise_fn)
    views = ["hero", "rear", "side", "top"]
    assert calls == [(views, views, "A pale slab with a red nose block.", 8)]
    assert rec["better"] and rec["pairwise"]["win_rate"] == 0.85 and rec["pairwise"]["cost"] == 0.02
    assert rec["against_name"] == "a00" and "0.85 decisive vs a00" in evaluate.ledger_line(rec)
    assert "DECISIVE, a01 is closer" in evaluate.feedback(rec) and "narrower, as the reference" in evaluate.feedback(rec)
    # no backend: the numbers are measured, no model is called
    rec2 = run(proj, tmp_path, "a02", build_fn=fake_build(), against=against, pairwise_fn=pairwise_fn)
    assert rec2["ok"] and "pairwise" not in rec2 and len(calls) == 1
    empty = tmp_path / "nothing"
    empty.mkdir()
    rec3 = run(proj, tmp_path, "a03", build_fn=fake_build(), against=empty, backend=object(), pairwise_fn=pairwise_fn)
    assert "no view PNGs" in rec3["pairwise_error"] and len(calls) == 1


def test_eval_json_holds_paths_relative_to_the_attempt_and_read_gives_them_back(tmp_path):
    proj = make_project(tmp_path)
    run(proj, tmp_path, "a00", build_fn=fake_build(span=0.07))
    a00, a01 = (tmp_path / "proj" / "attempts" / n for n in ("a00", "a01"))
    rec = run(proj, tmp_path, "a01", build_fn=fake_build(), against=a00, backend=object(),
              pairwise_fn=lambda *a, **kw: {"win_rate": 0.8, "decisive": True, "ci": [0.6, 0.9]})
    text = (a01 / "eval.json").read_text()
    for spelt in (str(tmp_path), tmp_path.as_posix(), json.dumps(str(tmp_path))[1:-1]):
        assert spelt not in text, "a record names no machine"
    stored = json.loads(text)
    assert stored["script"] == "ship.py" and stored["against"] == "../a00"
    assert stored["renders"]["top"] == "renders/top.png" and stored["compare"]["top"] == "compare/top_vs_reference.png"
    assert stored["overlays"]["rear"] == "compare/sil_rear.png"
    assert {k: v for k, v in stored.items() if k not in evaluate.PATH_FIELDS + evaluate.PATH_MAPS} == {
        k: v for k, v in rec.items() if k not in evaluate.PATH_FIELDS + evaluate.PATH_MAPS}, "only the paths differ"
    # what evaluate returned is absolute, for a caller to open, and `read` is the same record
    assert Path(rec["renders"]["top"]).is_absolute() and Path(rec["renders"]["top"]).is_file()
    back = evaluate.read(a01)
    assert {k: Path(back[k]) for k in evaluate.PATH_FIELDS} == {k: Path(rec[k]) for k in evaluate.PATH_FIELDS}
    maps = [k for k in evaluate.PATH_MAPS if k in rec]          # close-ups are of a project that marks regions
    assert maps == ["renders", "compare", "ruled", "overlays", "zone_maps"]
    assert {k: {v: Path(p) for v, p in back[k].items()} for k in maps} == {
        k: {v: Path(p) for v, p in rec[k].items()} for k in maps}
    # the attempt copied somewhere else still finds its own pictures
    moved = tmp_path / "elsewhere" / "a01"
    moved.parent.mkdir()
    a01.rename(moved)
    assert Path(evaluate.read(moved)["overlays"]["top"]) == moved / "compare" / "sil_top.png"
    assert Path(evaluate.read(moved)["overlays"]["top"]).is_file()


def test_a_path_outside_the_project_is_kept_by_name_and_an_error_text_loses_the_roots(tmp_path):
    proj = make_project(tmp_path)
    attempt = tmp_path / "proj" / "attempts" / "a01"
    outside = tmp_path / "scratch" / "renders"
    rec = {"script": str(attempt / "ship.py"), "against": str(outside), "renders": {"top": str(attempt / "renders" / "top.png")},
           "error": f'File "{attempt / "out" / "ship.py"}", line 3\nno such file: {proj.root / "refs" / "top.png"} in {proj.root.as_posix()}/refs'}
    stored = evaluate.portable(rec, attempt, proj.root)
    assert stored["against"] == "renders" and stored["script"] == "ship.py" and stored["renders"] == {"top": "renders/top.png"}
    assert stored["error"].replace("\\", "/") == 'File "out/ship.py", line 3\nno such file: refs/top.png in refs'
    assert rec["against"] == str(outside), "the record in hand is not changed"
    # an attempt directory outside the project (eval --out) keeps what is inside it, and names the rest
    away = evaluate.portable({"script": str(attempt / "ship.py"), "compare": {"top": str(outside / "top_vs_reference.png")}},
                             outside, proj.root)
    assert away == {"script": "ship.py", "compare": {"top": "top_vs_reference.png"}}


def test_a_record_written_with_absolute_paths_reads_as_it_was(tmp_path):
    attempt = tmp_path / "a01"
    attempt.mkdir()
    old = {"name": "a01", "ok": True, "script": str(tmp_path / "old" / "ship.py"),
           "overlays": {"top": str(tmp_path / "old" / "compare" / "sil_top.png")}}
    (attempt / "eval.json").write_text(json.dumps(old))
    assert evaluate.read(attempt) == old


def test_the_record_names_the_harness_that_measured_it(tmp_path, monkeypatch):
    from mesh_jig import harness
    monkeypatch.setattr(harness, "version", lambda: {"commit": "abc1234", "dirty": True})
    proj = make_project(tmp_path)
    rec = run(proj, tmp_path, build_fn=fake_build())
    assert rec["harness"] == {"commit": "abc1234", "dirty": True}
    assert evaluate.read(tmp_path / "proj" / "attempts" / "a01")["harness"] == rec["harness"]
    monkeypatch.setattr(harness, "version", lambda: None)
    failed = run(proj, tmp_path, "a02", build_fn=fake_build(error="JIG_BUILD_ERROR"))
    assert not failed["ok"] and failed["harness"] is None, "a failed attempt records it too; None outside a checkout"


def test_a_stated_score_is_named_with_its_terms_and_the_regions_are_listed_weakest_first():
    n = {"mae_mm": None, "ortho_iou": 0.8, "resemblance": 0.5, "zone": 0.6, "colour": 0.7, "edges": 0.4, "regions": 0.5,
         "mix": {"overlap": 1.0, "colour": 1.0, "edges": 1.0, "regions": 1.0}}
    assert evaluate.numeric_score(n) == 0.6
    assert evaluate.numeric_score({**n, "mix": {"overlap": 3.0, "regions": 1.0}}) == 0.725
    assert evaluate.numeric_score({**n, "regions": None}) == round((0.8 + 0.7 + 0.4) / 3, 4), "a term with no number"
    assert evaluate.numeric_score({k: v for k, v in n.items() if k != "mix"}) == 0.65, "none stated: as it was"
    n["numeric_score"] = 0.6
    assert evaluate.numbers_text(n) == ("overlap 0.800, resemblance 0.500, zones 0.60, colour 0.700, edges 0.400, "
                                        "regions 0.500, numeric score 0.6000 (the mean of overlap, colour, edges and "
                                        "regions)")
    plain = {k: v for k, v in n.items() if k != "mix"}
    assert evaluate.numbers_text(plain) == "overlap 0.800, resemblance 0.500, zones 0.60, numeric score 0.6000"
    checks = [{"name": "tail", "score": 0.7, "colour": 0.8, "edges": 0.6, "passed": True},
              {"name": "face", "score": 0.4, "colour": 0.4, "edges": None, "passed": False},
              {"name": "part:skin", "passed": True}]
    line = ("REGIONS (each marked region read on its own, twice as fine as a whole view; weakest first, their mean "
            "0.500{}): face 0.40 (colours 0.40 close); tail 0.70 (colours 0.80 close, edges in place 0.60)")
    assert evaluate.region_lines({"numeric": n, "review": {"checks": checks}}) == [line.format(", which the score counts")]
    assert evaluate.region_lines({"numeric": plain, "regions": checks}) == [line.format("")], "a preview keeps its own"
    assert evaluate.region_lines({"numeric": n, "review": {"checks": checks[2:]}}) == []


def test_standing_says_when_the_arms_no_longer_hang_clear_of_the_body():
    rec = {"numeric": {"numeric_score": 0.5}, "readiness": {"limbs": [
        {"parts": 2, "clear_m": 0.4, "joined_m": 0.0, "from_m": 0.4, "on_ground": True}], "height_m": 1.0}}
    best = {"name": "a03", "score": 0.6, "failed": [], "gates": 0, "views": {}, "arms_clear": 0.15}
    said = evaluate.standing_lines(rec, {"others": 1, "best": best, "clean": None}, [])[0]
    assert said.endswith("No arm hangs clear of the body, where a03's hang clear for 0.15 m.")
    kept = dict(rec, readiness={"limbs": rec["readiness"]["limbs"] + [
        {"parts": 3, "clear_m": 0.1, "joined_m": 0.0, "from_m": 0.8, "on_ground": False}], "height_m": 1.0})
    assert "No arm hangs clear" not in evaluate.standing_lines(kept, {"others": 1, "best": best, "clean": None}, [])[0]


def test_a_failed_region_gate_says_which_reference_colours_the_build_reads_wrong():
    import numpy as np
    from mesh_jig import review
    names = ["orange", "cream", "black"]
    # a 4x4 region: the reference is cream on the left half and orange on the right; the build is orange throughout
    cb = np.array([[1, 1, 0, 0]] * 4)
    ca = np.zeros((4, 4), dtype=int)
    mb = np.ones((4, 4), dtype=bool)
    ma = mb.copy()
    ma[0] = False                       # and leaves the top row empty
    read = review.region_colours(ca, ma, cb, mb, names)
    assert list(read) == ["orange", "cream"]
    assert read["cream"] == {"share": 0.5, "yours": {"orange": 0.75, "nothing": 0.25}}
    check = {"name": "face", "view": "front", "passed": False, "score": 0.4, "colour": 0.5, "edges": 0.3,
             "read_as": read}
    lines = evaluate.region_lines({"review": {"checks": [check]}})
    assert lines[1] == ("  face (its gate FAILED; front view, the zone colours as rendered): where the reference is "
                        "orange (50% of it), yours reads orange 75%, nothing 25%; where the reference is cream (50% of "
                        "it), yours reads orange 75%, nothing 25%")
    assert len(evaluate.region_lines({"review": {"checks": [dict(check, passed=True)]}})) == 1


def test_the_zone_picture_marks_where_the_colours_disagree(tmp_path):
    import numpy as np
    from PIL import Image as Im
    from mesh_jig.palette import Zone   # noqa: F401  (zones are (letter, name, rgb) as the project folds them)
    zones = (("O", "orange", (230, 120, 30)), ("C", "cream", (240, 220, 180)))
    ref = Im.new("RGB", (200, 200), (255, 255, 255))
    Im.Image.paste(ref, (230, 120, 30), (50, 50, 150, 150))
    mine = ref.copy()
    Im.Image.paste(mine, (240, 220, 180), (50, 50, 100, 150))          # the left half cream where it should be orange
    ref.save(tmp_path / "ref.png")
    mine.save(tmp_path / "mine.png")
    out = evaluate.zone_picture(tmp_path / "mine.png", tmp_path / "ref.png", zones, tmp_path / "zones_front.png")
    sheet = np.asarray(Im.open(out).convert("RGB"))
    red = (sheet[..., 0] > 200) & (sheet[..., 1] < 60) & (sheet[..., 2] < 60)
    third = sheet.shape[1] // 3
    assert red[:, 2 * third:].sum() > 0 and red[:, :2 * third].sum() == 0, "differences only on the third panel"
    left, right = red[:, 2 * third:2 * third + third // 2].sum(), red[:, 2 * third + third // 2:].sum()
    assert left > 5 * max(right, 1), "the disagreement is on the left half, where the build is cream"



def test_a_regions_zone_picture_is_cut_to_its_box(tmp_path):
    import numpy as np
    from PIL import Image as Im
    zones = (("O", "orange", (230, 120, 30)), ("C", "cream", (240, 220, 180)))
    ref = Im.new("RGB", (200, 200), (255, 255, 255))
    Im.Image.paste(ref, (230, 120, 30), (50, 50, 150, 150))
    Im.Image.paste(ref, (240, 220, 180), (50, 50, 100, 150))            # the subject's left half cream
    ref.save(tmp_path / "ref.png")
    ref.save(tmp_path / "mine.png")
    face = np.asarray(Im.open(evaluate.zone_picture(tmp_path / "mine.png", tmp_path / "ref.png", zones,
                                                    tmp_path / "f.png", box=[0.05, 0.2, 0.4, 0.8])).convert("RGB"))
    panel = face[30:, : face.shape[1] // 3 - 10]                        # the reference's zones, under its label
    orange = (np.abs(panel.astype(int) - (230, 120, 30)).sum(-1) < 30).sum()
    cream = (np.abs(panel.astype(int) - (240, 220, 180)).sum(-1) < 30).sum()
    assert cream > 0 and orange == 0, "a box on the cream half shows only cream"


def test_a_preview_of_a_build_the_contract_refuses_says_so_and_does_not_promise_its_numbers(tmp_path):
    """A builder previewed a build its contract had just refused (X extent out of tolerance) and was told an
    evaluation of it would give those numbers: a preview read the GLBs and never the contract."""
    proj = make_project(tmp_path)
    s = script(tmp_path)
    evaluate.evaluate(proj, s, s.parent, build_fn=fake_build(), render_fn=quick_render, log=lambda *_: None)
    (s.parent / "out" / s.name).write_text(s.read_text())        # a real build leaves the script beside its GLB
    fine = evaluate.preview(proj, s.parent, render_fn=quick_render)
    assert fine["contract"] == [] and "An evaluation of this build gives" in evaluate.preview_text(proj, s.parent, fine)
    strict = project.Project(proj.root, {**proj.data, "model": {**proj.data["model"], "max_triangles": 2}})
    refused = evaluate.preview(strict, s.parent, render_fn=quick_render)
    said = evaluate.preview_text(strict, s.parent, refused)
    assert refused["contract"] and said.startswith("OUT OF CONTRACT: an evaluation would refuse this build")
    assert "triangles" in said and "An evaluation of this build gives" not in said
