"""Checkpoint fault injection through all three offline workflow loops."""
import builtins
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

from mesh_jig import agent
from test_tool_protocol_transfer import workflow


class BrokenWrite:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        self.stream.write(text[:13])
        self.stream.flush()
        raise OSError("injected partial checkpoint write")


def fail_writes(monkeypatch, out, filename):
    # Cover the old direct opens and the replacement's private descriptor.
    selected_fds = set()
    mkstemp = tempfile.mkstemp

    def create(*args, **kwargs):
        fd, name = mkstemp(*args, **kwargs)
        selected_fds.discard(fd)
        if Path(name).name.startswith("." + filename + "."):
            selected_fds.add(fd)
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", create)
    def wrap(original):
        def opened(file, mode="r", *args, **kwargs):
            stream = original(file, mode, *args, **kwargs)
            if "w" in mode and ((isinstance(file, int) and file in selected_fds) or
                    (not isinstance(file, int) and Path(file).parent == out and
                     (Path(file).name == filename or Path(file).name.startswith("." + filename + ".")))):
                return BrokenWrite(stream)
            return stream
        return opened
    for module, name in [(builtins, "open"), (io, "open"), (os, "fdopen")]:
        monkeypatch.setattr(module, name, wrap(getattr(module, name)))


def response(text):
    return {"content": text}, {"prompt_tokens": 7, "completion_tokens": 2}


@pytest.mark.parametrize("filename", ["report.json", "transcript.jsonl"])
def test_partial_write_preserves_published_file(workflow, monkeypatch, filename):
    _, root, _, run = workflow
    run(lambda *_: response("previous"), 1)
    out = root / "audit"
    before = (out / filename).read_bytes()
    fail_writes(monkeypatch, out, filename)
    with pytest.raises(OSError, match="injected partial"):
        run(lambda *_: response("replacement"), 1)
    assert (out / filename).read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "transcript.jsonl"]


@pytest.mark.parametrize("filename", ["report.json", "transcript.jsonl"])
def test_serialization_failure_preserves_published_file(workflow, monkeypatch, filename):
    _, root, _, run = workflow
    run(lambda *_: response("previous"), 1)
    out = root / "audit"
    before = (out / filename).read_bytes()
    original = json.dumps

    def dumps(value, *args, **kwargs):
        selected = (isinstance(value, dict) and
                    ((filename == "report.json" and "turns" in value) or
                     (filename == "transcript.jsonl" and value.get("role") == "assistant")))
        if selected:
            raise TypeError("injected unserializable checkpoint")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", dumps)
    with pytest.raises(TypeError, match="injected unserializable"):
        run(lambda *_: response("replacement"), 1)
    assert (out / filename).read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "transcript.jsonl"]


def test_repeated_unicode_checkpoints_preserve_schema_and_strip_images(workflow):
    _, root, _, run = workflow
    out = root / "audit"
    for text in ["first", "Māori 日本語 café", ""]:
        reply = {"content": [{"type": "text", "text": text},
                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}}]}
        rep = run(lambda *_: (reply, {"prompt_tokens": 7, "completion_tokens": 2}), 1)
        assert json.loads((out / "report.json").read_text(encoding="utf-8")) == rep
        transcript = [json.loads(line) for line in (out / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
        saved = next(m for m in transcript if m["role"] == "assistant")
        assert saved["content"] == [{"type": "text", "text": text}, {"type": "image"}]
        assert reply["content"][1]["type"] == "image_url"
        assert sorted(p.name for p in out.iterdir()) == ["report.json", "transcript.jsonl"]

@pytest.mark.parametrize("filename", ["report.json", "transcript.jsonl"])
def test_replace_failure_preserves_published_file(workflow, monkeypatch, filename):
    _, root, _, run = workflow
    run(lambda *_: response("previous"), 1)
    out = root / "audit"
    before = (out / filename).read_bytes()
    original = os.replace
    observed = []

    def replace(source, destination):
        if Path(destination) == out / filename:
            assert Path(source).parent == out
            assert Path(source) != Path(destination)
            assert (out / filename).read_bytes() == before
            text = Path(source).read_text(encoding="utf-8")
            if filename == "report.json":
                json.loads(text)
            else:
                assert all(isinstance(json.loads(line), dict) for line in text.splitlines())
            observed.append(source)
            raise PermissionError("injected replacement denied")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(PermissionError, match="injected replacement denied"):
        run(lambda *_: response("replacement"), 1)
    assert observed
    assert (out / filename).read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "transcript.jsonl"]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", ["create", "open", "serialize", "write", "close", "replace"])
def test_helper_preparation_failures_clean_up(tmp_path, monkeypatch, existing, fault):
    from mesh_jig.checkpoint import atomic_write_text

    destination = tmp_path / "report.json"
    unrelated = tmp_path / ".report.json.unrelated.tmp"
    unrelated.write_text("another writer's temporary file")
    if existing:
        destination.write_text("previous", encoding="utf-8")
    before = destination.read_bytes() if existing else None
    descriptor = []
    create = tempfile.mkstemp

    def fail(*args, **kwargs):
        raise OSError("injected " + fault)

    def create_recorded(*args, **kwargs):
        result = create(*args, **kwargs)
        descriptor.append(result[0])
        return result

    monkeypatch.setattr(tempfile, "mkstemp", fail if fault == "create" else create_recorded)
    if fault == "open":
        monkeypatch.setattr(os, "fdopen", fail)
    elif fault == "write":
        fail_writes(monkeypatch, tmp_path, "report.json")
    elif fault == "close":
        original = os.fdopen

        class BrokenClose(BrokenWrite):
            def write(self, text):
                return self.stream.write(text)

            def __exit__(self, *args):
                self.stream.close()
                raise OSError("injected close")

        monkeypatch.setattr(os, "fdopen", lambda *a, **kw: BrokenClose(original(*a, **kw)))
    elif fault == "replace":
        monkeypatch.setattr(os, "replace", fail)

    def chunks():
        yield "complete first line\n"
        if fault == "serialize":
            raise TypeError("injected serialize")
        yield "second line\n"

    with pytest.raises((OSError, TypeError), match="injected"):
        atomic_write_text(destination, chunks())
    assert (destination.read_bytes() if destination.exists() else None) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [unrelated.name] + ([destination.name] if existing else []))
    for fd in descriptor:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_helper_unicode_empty_and_byte_parity(tmp_path):
    from mesh_jig.checkpoint import atomic_write_text

    destination = tmp_path / "checkpoint.txt"
    control = tmp_path / "control.txt"
    for text in ["Māori 日本語 café\nnext\n", "short", ""]:
        control.write_text(text, encoding="utf-8")
        atomic_write_text(destination, text)
        assert destination.read_bytes() == control.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.txt", "control.txt"]


@pytest.mark.parametrize("kind", ["rig", "sound"])
@pytest.mark.parametrize("fault", [None, "write", "replace"])
def test_final_accounting_report_uses_safe_writer(tmp_path, monkeypatch, kind, fault):
    from mesh_jig import llm, rig_agent, sound_agent

    module = rig_agent if kind == "rig" else sound_agent
    saved = {}

    def no_chat(*args, **kwargs):
        raise AssertionError("No model call allowed")

    monkeypatch.setattr(agent, "chat_for", lambda *a, **kw: no_chat)

    def run(root, chat, *, out, **kwargs):
        saved["out"] = out
        report = {"workflow": kind, "turns": 1, "final": None}
        (out / "report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
        saved["before"] = (out / "report.json").read_bytes()
        if fault == "write":
            fail_writes(monkeypatch, out, "report.json")
        elif fault == "replace":
            def denied(*args):
                raise PermissionError("injected replace")
            monkeypatch.setattr(os, "replace", denied)
        return report

    monkeypatch.setattr(module, "run", run)
    backend = llm.Backend("offline", "https://offline.invalid", "mock")
    if fault:
        with pytest.raises(OSError, match="injected"):
            module.drive(tmp_path, backend, effort="low", log=lambda _: None)
        assert (saved["out"] / "report.json").read_bytes() == saved["before"]
    else:
        rep, out = module.drive(tmp_path, backend, effort="low", log=lambda _: None)
        assert rep["effort"] == "low"
        assert json.loads((out / "report.json").read_text(encoding="utf-8")) == rep
    assert [p.name for p in saved["out"].iterdir()] == ["report.json"]


def test_files_are_individual_replacements_not_a_pair_transaction(tmp_path):
    rep = agent.Report()
    agent.write_run(tmp_path, rep, [{"role": "assistant", "content": "previous"}])
    old_transcript = (tmp_path / "transcript.jsonl").read_bytes()
    rep.turns = 2
    with pytest.raises(TypeError):
        agent.write_run(tmp_path, rep, [{"role": "assistant", "content": object()}])
    assert json.loads((tmp_path / "report.json").read_text())["turns"] == 2
    assert (tmp_path / "transcript.jsonl").read_bytes() == old_transcript
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json", "transcript.jsonl"]
