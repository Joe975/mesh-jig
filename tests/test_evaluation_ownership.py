"""One live evaluation owns its attempt outputs until it returns or its process exits."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from mesh_jig import cli, evaluate
from test_evaluate import fake_build, make_project, quick_render, script


def _error_render(*_args, **_kwargs):
    return {"error": "stop after the deterministic build probe"}


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(root.rglob("*")) if path.name != ".mjlock"}


def test_same_output_rejects_overlap_before_build_or_writes(tmp_path, monkeypatch, capsys):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    entered, release = threading.Event(), threading.Event()
    calls, errors = [], []

    def build_fn(_script, build_dir, _spec, **_kw):
        calls.append(threading.current_thread().name)
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        (Path(build_dir).parent / "active-marker").write_text("owned")
        entered.set()
        assert release.wait(10), "probe owner was not released"
        return ""

    def owner():
        try:
            evaluate.evaluate(proj, model, out, build_fn=build_fn, render_fn=_error_render, log=lambda *_: None)
        except BaseException as error:  # propagate worker failures to the test thread
            errors.append(error)

    worker = threading.Thread(target=owner, name="owner")
    worker.start()
    assert entered.wait(10), "owner did not enter the build stage"
    before = _tree_snapshot(out)
    ledger_before = "existing ledger contents\n"
    proj.ledger.write_text(ledger_before)
    monkeypatch.setattr(evaluate, "_evaluate_locked",
                        lambda *_args, **_kwargs: pytest.fail("CLI bypassed ownership and entered evaluation"))
    status = cli.main(["eval", str(proj.root), "--script", str(model), "--out", str(out),
                       "--ledger", str(proj.ledger)])
    cli_error = capsys.readouterr().err
    assert status == 2 and "already running" in cli_error and "wait for that command" in cli_error
    assert proj.ledger.read_text() == ledger_before
    with pytest.raises(ValueError, match="already running.*wait for that command"):
        evaluate.evaluate(proj, model, out,
                          build_fn=lambda *_args, **_kwargs: pytest.fail("rejected evaluation reached build"),
                          render_fn=_error_render, log=lambda *_: None)
    after = _tree_snapshot(out)
    assert after == before
    assert calls == ["owner"]
    release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert not errors
    assert (out / evaluate.EVAL_FILE).is_file()


def test_failed_and_successful_evaluations_release_for_sequential_reruns(tmp_path):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent

    def raises(*_args, **_kwargs):
        raise RuntimeError("raised build failure")

    with pytest.raises(RuntimeError, match="raised build failure"):
        evaluate.evaluate(proj, model, out, build_fn=raises, render_fn=_error_render, log=lambda *_: None)
    failed = evaluate.evaluate(proj, model, out, build_fn=lambda *_a, **_kw: "fixture build error",
                              render_fn=_error_render, log=lambda *_: None)
    assert failed["stage"] == "build"
    passed_to_render = evaluate.evaluate(proj, model, out, build_fn=lambda *_a, **_kw: "",
                                         render_fn=_error_render, log=lambda *_: None)
    assert passed_to_render["stage"] == "render"
    assert "stop after" in passed_to_render["error"]
    measured = evaluate.evaluate(proj, model, out, build_fn=fake_build(), render_fn=quick_render,
                                 log=lambda *_: None)
    assert measured["ok"] and measured["stage"] == "measured"


def test_different_output_directories_can_evaluate_at_the_same_time(tmp_path):
    proj = make_project(tmp_path)
    models = [script(tmp_path, "a01"), script(tmp_path, "a02")]
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    results, errors = [], []

    def worker(index):
        def build_fn(*_args, **_kwargs):
            entered[index].set()
            assert release.wait(10), "probe evaluations were not released"
            return ""
        try:
            results.append(evaluate.evaluate(proj, models[index], models[index].parent, build_fn=build_fn,
                                             render_fn=_error_render, log=lambda *_: None))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,), name=f"output-{i}") for i in range(2)]
    for thread in threads:
        thread.start()
    assert all(event.wait(10) for event in entered), "independent output did not enter build concurrently"
    release.set()
    for thread in threads:
        thread.join(10)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert [result["stage"] for result in results] == ["render", "render"]


def test_cli_holds_ownership_through_ledger_and_feedback_postprocessing(tmp_path, monkeypatch, capsys):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    ledger = out.parent / "ledger.md"
    ledger.write_text("existing ledger contents\n")
    (out / cli.FEEDBACK_FILE).write_text("existing feedback\n")
    core = evaluate._evaluate_locked
    build_calls, errors, results = [], [], []
    postprocessing, release = threading.Event(), threading.Event()
    append = evaluate.append_ledger

    def core_with_fakes(*args, **kwargs):
        def fake_build(*_args, **_kwargs):
            build_calls.append(threading.current_thread().name)
            return ""
        kwargs.update(build_fn=fake_build, render_fn=_error_render)
        return core(*args, **kwargs)

    def held_append(path, rec):
        if threading.current_thread().name == "first-cli":
            postprocessing.set()
            assert release.wait(10), "CLI post-processing owner was not released"
        return append(path, rec)

    monkeypatch.setattr(evaluate, "_evaluate_locked", core_with_fakes)
    monkeypatch.setattr(evaluate, "append_ledger", held_append)
    args = ["eval", str(proj.root), "--script", str(model), "--out", str(out), "--ledger", str(ledger)]

    def first_cli():
        try:
            results.append(cli.main(args))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=first_cli, name="first-cli")
    worker.start()
    assert postprocessing.wait(10), "first CLI did not reach its ledger post-processing"
    before_output = _tree_snapshot(out)
    before_ledger = ledger.read_bytes()
    stderr = capsys.readouterr().err
    assert cli.main(args) == 2
    second_stderr = capsys.readouterr().err
    assert "already running" in second_stderr and "wait for that command" in second_stderr
    assert _tree_snapshot(out) == before_output
    assert ledger.read_bytes() == before_ledger
    assert build_calls == ["first-cli"]
    release.set()
    worker.join(10)
    assert not worker.is_alive() and not errors
    assert results == [1]
    assert "FAILED render" in ledger.read_text()
    assert (out / cli.FEEDBACK_FILE).read_text() != "existing feedback\n"


def test_process_exit_releases_real_evaluation_ownership(tmp_path):
    proj = make_project(tmp_path)
    model = script(tmp_path)
    out = model.parent
    child_code = r'''\
import os
import sys
from pathlib import Path
from mesh_jig import evaluate, project
proj = project.load(sys.argv[1])
def build_fn(_script, build_dir, _spec, **_kw):
    marker = Path(build_dir).parent / "child-active-marker"
    marker.write_text("owned")
    print("OWNED", flush=True)
    sys.stdin.readline()
    os._exit(23)
evaluate.evaluate(proj, sys.argv[2], sys.argv[3], build_fn=build_fn,
                  render_fn=lambda *_a, **_kw: {"error": "stop"}, log=lambda *_: None)
'''
    env = dict(os.environ)
    alternate_temp = out.parent / "alternate-temp"
    alternate_temp.mkdir()
    assert os.path.normcase(str(alternate_temp)) != os.path.normcase(tempfile.gettempdir())
    env["TEMP"] = str(alternate_temp)
    env["TMP"] = str(alternate_temp)
    env.pop("TMPDIR", None)
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen([sys.executable, "-u", "-c", child_code, str(proj.root), str(model), str(out)],
                             cwd=Path(__file__).resolve().parents[1], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line, read_error = [], []
    ready = threading.Event()

    def read_ready():
        try:
            line.append(child.stdout.readline())
        except BaseException as error:
            read_error.append(error)
        finally:
            ready.set()

    reader = threading.Thread(target=read_ready, daemon=True)
    reader.start()
    try:
        assert ready.wait(10), "child evaluation did not reach its build callback"
        assert not read_error and line == ["OWNED\n"]
        marker = out / "child-active-marker"
        before = marker.read_bytes()
        with pytest.raises(ValueError, match="already running"):
            evaluate.evaluate(proj, model, out,
                              build_fn=lambda *_a, **_kw: pytest.fail("second process lock was ignored"),
                              render_fn=_error_render, log=lambda *_: None)
        assert marker.read_bytes() == before
    finally:
        child.stdin.write("crash\n")
        child.stdin.flush()
        child.wait(timeout=10)
        reader.join(10)
        assert child.returncode == 23
    # The OS releases advisory ownership when the holder exits, including an abrupt exit mid-build.
    after_crash = evaluate.evaluate(proj, model, out, build_fn=lambda *_a, **_kw: "",
                                    render_fn=_error_render, log=lambda *_: None)
    assert after_crash["stage"] == "render"


@pytest.mark.skipif(os.name != "nt", reason="regresses Windows MAX_PATH sidecar naming")
def test_valid_long_output_path_uses_a_short_deterministic_lock_name(tmp_path):
    with tempfile.TemporaryDirectory(prefix="h07-") as temporary:
        scratch = Path(temporary)
        proj = make_project(tmp_path)
        model = script(tmp_path)
        # The actual absolute path, including the temp prefix, determines the name.
        # Windows must create this path as a directory; keep it near MAX_PATH
        # while under the directory API's slightly shorter practical ceiling.
        desired_path_length = 245
        output_name_length = desired_path_length - len(str(scratch)) - 1
        assert 1 <= output_name_length <= 240
        out = scratch / ("o" * output_name_length)
        former_sidecar = out.parent / f".{out.name}.mesh-jig-eval.lock"
        assert len(out.name) < 255 and len(str(out)) < 260
        assert len(str(out)) >= 240
        assert len(str(former_sidecar)) > 260, "the previous sidecar spelling must exceed MAX_PATH"
        rec = evaluate.evaluate(proj, model, out, build_fn=lambda *_a, **_kw: "stop after ownership",
                                render_fn=_error_render, log=lambda *_: None)
        assert rec["stage"] == "build" and (out / evaluate.EVAL_FILE).is_file()
        assert rec["failure_preservation"]["status"] == "preserved"
        deep_component_budget = desired_path_length - len(str(scratch)) - 3
        first_component_length = deep_component_budget // 2
        second_component_length = deep_component_budget - first_component_length
        assert 1 <= first_component_length <= 120
        assert 1 <= second_component_length <= 120
        deep_out = scratch / ("p" * first_component_length) / ("q" * second_component_length) / "o"
        former_deep_sidecar = deep_out.parent / f".{deep_out.name}.mesh-jig-eval.lock"
        assert len(str(deep_out)) < 260 and len(str(former_deep_sidecar)) > 260
        assert len(str(deep_out)) >= 240
        deep_rec = evaluate.evaluate(proj, model, deep_out, build_fn=lambda *_a, **_kw: "stop after ownership",
                                     render_fn=_error_render, log=lambda *_: None)
        assert deep_rec["stage"] == "build" and (deep_out / evaluate.EVAL_FILE).is_file()
        assert deep_rec["failure_preservation"]["status"] == "preserved"
        lock_files = [out / ".mjlock", deep_out / ".mjlock"]
        assert all(path.is_file() and len(path.name) == len(".mjlock") for path in lock_files)
        assert all(len(str(path)) < 260 for path in lock_files)
        # The snapshot lives beyond MAX_PATH below these otherwise legal output paths.
        import shutil
        shutil.rmtree(Path("\\\\?\\" + str(scratch)))
