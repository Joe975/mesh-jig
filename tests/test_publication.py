"""Privacy guard must reject ignored ancestry, renamed private files and unsafe metadata."""
import importlib.util
from pathlib import Path
import subprocess
import json
import io
from types import SimpleNamespace

import pytest

from glbkit import cube, write_cubes
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check_publication.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "publication/check_publication.py"
spec = importlib.util.spec_from_file_location("publication_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def git(root, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + root.as_posix(), *args], cwd=root)


def commit(root, message="safe"):
    git(root, "add", "--all")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@users.noreply.github.com", "commit", "-qm", message)


def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "PUBLIC_FILES.txt").write_text("PUBLIC_FILES.txt\nsafe.py\n")
    (tmp_path / "safe.py").write_text("pass\n")
    commit(tmp_path)
    return tmp_path


def test_clean_history_passes_and_untracked_files_block(tmp_path):
    root = repo(tmp_path)
    assert guard.audit(root, ["HEAD"]) == []
    (root / "unexpected.txt").write_text("private notes")
    assert any("outside public inventory" in f for f in guard.audit(root, ["HEAD"]))


def test_deleted_secret_still_blocks_history_without_disclosing_value(tmp_path):
    root = repo(tmp_path)
    secret = "sk-" + "Z" * 30
    (root / "safe.py").write_text("KEY = " + repr(secret))
    commit(root)
    (root / "safe.py").write_text("pass\n")
    commit(root)
    failures = guard.audit(root, ["HEAD"])
    assert any("credential" in f for f in failures)
    assert secret not in "\n".join(failures)


def test_duplicate_blob_under_private_name_cannot_evade_path_checks(tmp_path):
    root = repo(tmp_path)
    (root / "experiments").mkdir()
    (root / "experiments/result.py").write_text("pass\n")
    commit(root)
    assert any("experiments/result.py" in f for f in guard.audit(root, ["HEAD"]))


def test_private_email_and_home_directory_block(tmp_path):
    root = repo(tmp_path)
    (root / "safe.py").write_text(repr("C:" + "/Users/" + "test/private"))
    git(root, "add", "--all")
    git(root, "-c", "user.name=Test", "-c", "user.email=private@example.com", "commit", "-qm", "safe")
    failures = guard.audit(root, ["HEAD"])
    assert any("email" in f for f in failures) and any("home directory" in f for f in failures)


def test_inventory_cannot_opt_in_binary_results_or_environment_secrets():
    for name in ("experiments/results.md", ".env.production", "private/model.py", "demo.glb", "shot.png", "../escape", "link/.git/config"):
        assert guard.forbidden_path(name)
        assert guard.scan(name, b"safe text", {name})
    assert guard.scan("safe.py", b"\0binary", {"safe.py"})


def selected_showcase(root):
    package = root / "showcase/example"
    report = root.parent / (root.name + "-raw-report.json")
    report.write_text(json.dumps({"model": "example/model", "cost_usd": .01, "wall_s": 2, "evaluations": 1}))
    image = root.parent / (root.name + "-image.png")
    Image.new("RGB", (3, 3), "red").save(image)
    model = write_cubes(root.parent / (root.name + "-model.glb"), [cube()])
    guard.curator.prepare(report, package, title="Example", driver="API", scope="One evaluation", cost_kind="reported", images=[image], models=[model])
    inventory = root / "PUBLIC_FILES.txt"
    names = set(inventory.read_text().splitlines())
    names.update("showcase/example/" + p.name for p in package.iterdir())
    names.add("showcase/example/release.json")
    inventory.write_text("\n".join(sorted(names)) + "\n")
    return package


def test_only_valid_sealed_selected_images_and_models_can_pass(tmp_path):
    root = repo(tmp_path)
    package = selected_showcase(root)
    assert any("unsealed" in f for f in guard.audit(root, ["HEAD"]))
    guard.curator.seal(package)
    assert guard.audit(root, ["HEAD"]) == []
    commit(root)
    assert guard.audit(root, ["HEAD"]) == []
    Image.new("RGB", (3, 3), "blue").save(package / "image-01.png")
    assert any("altered" in f for f in guard.audit(root, ["HEAD"]))


def test_later_sealing_does_not_hide_an_unsealed_historical_commit(tmp_path):
    root = repo(tmp_path)
    package = selected_showcase(root)
    commit(root)
    guard.curator.seal(package)
    commit(root)
    assert any("unsealed" in f for f in guard.audit(root, ["HEAD"]))


def test_each_valid_historical_selection_uses_its_own_hashes(tmp_path):
    root = repo(tmp_path)
    package = selected_showcase(root)
    guard.curator.seal(package)
    commit(root)
    Image.new("RGB", (3, 3), "blue").save(package / "image-01.png")
    (package / "release.json").unlink()
    guard.curator.seal(package)
    commit(root)
    assert guard.audit(root, ["HEAD"]) == []


def legacy_audit(root, refs, worktree=True):
    """Pre-optimization traversal, retained as an independent failure-set oracle."""
    allowed = set((root / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines())
    failures = [name + ": forbidden inventory entry" for name in allowed if guard.forbidden_path(name)]
    seen_blobs = set()
    commits = git(root, "rev-list", *refs).decode().splitlines()
    for revision in commits:
        metadata = git(root, "show", "-s", "--format=%ae%n%ce%n%B", revision)
        lines = metadata.decode().splitlines()
        if any(not line.endswith("@users.noreply.github.com") for line in lines[:2]):
            failures.append(revision[:12] + ": author/committer email is not GitHub noreply")
        for label, pattern in guard.PATTERNS.items():
            if pattern.search(metadata):
                failures.append(revision[:12] + ": commit metadata contains " + label)
        entries = {}
        for row in git(root, "ls-tree", "-rz", revision).split(b"\0"):
            if not row:
                continue
            header, name_bytes = row.split(b"\t", 1)
            mode, kind, oid = header.split()
            name = name_bytes.decode("utf-8")
            if mode not in {b"100644", b"100755"} or kind != b"blob":
                failures.append(name + ": symlink or submodule requires review")
                continue
            entries[name] = oid
        selected = {name: git(root, "cat-file", "blob", oid.decode())
                    for name, oid in entries.items() if name.startswith("showcase/")}
        released, invalid = guard.showcases(selected)
        failures.extend(revision[:12] + ": " + failure for failure in invalid)
        for name, oid in entries.items():
            if name not in allowed or guard.forbidden_path(name):
                failures.append(revision[:12] + ": " + name + ": outside public inventory")
            identity = (name, oid, name in released)
            if identity not in seen_blobs:
                seen_blobs.add(identity)
                data = selected[name] if name in selected else git(root, "cat-file", "blob", oid.decode())
                failures.extend(guard.scan(name, data, allowed, released))
    if worktree:
        names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
        contents = {}
        for name in filter(None, names):
            path = root / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                failures.append(name + ": unsupported filesystem entry")
            elif path.is_file():
                contents[name] = path.read_bytes()
            else:
                failures.append(name + ": missing tracked file")
        released, invalid = guard.showcases(contents)
        failures.extend(invalid)
        for name, data in contents.items():
            failures.extend(guard.scan(name, data, allowed, released))
    return sorted(set(failures))


def assert_parity(root, refs=None, worktree=True):
    refs = refs or ["HEAD"]
    expected = legacy_audit(root, refs, worktree)
    for cache, batch in [(True, True), (False, True), (True, False), (False, False)]:
        assert guard.audit(root, refs, worktree, cache=cache, batch=batch) == expected
    return expected


@pytest.mark.parametrize("case", ["unchanged", "unsealed", "altered", "extra", "deleted-secret", "metadata", "renamed", "missing"])
def test_cache_matches_legacy_failures_for_adversarial_history(tmp_path, case):
    root = repo(tmp_path)
    package = selected_showcase(root)
    if case == "unsealed":
        commit(root)
    guard.curator.seal(package)
    commit(root)
    good = {p.name: p.read_bytes() for p in package.iterdir()}
    if case == "altered":
        Image.new("RGB", (3, 3), "blue").save(package / "image-01.png")
    elif case == "extra":
        (package / "raw-report.json").write_text("{}")
    elif case == "deleted-secret":
        (root / "safe.py").write_text("KEY = " + repr("sk-" + "Z" * 30))
    elif case == "renamed":
        (root / "experiments").mkdir()
        (root / "experiments/result.py").write_bytes((root / "safe.py").read_bytes())
    elif case == "missing":
        (package / "release.json").unlink()
    else:
        (root / "safe.py").write_text("pass # a distinct commit\n")
    if case == "metadata":
        git(root, "add", "--all")
        git(root, "-c", "user.name=Test", "-c", "user.email=private@example.com", "commit", "-qm", "sk-" + "Z" * 30)
    else:
        commit(root)
    bad_revision = git(root, "rev-parse", "HEAD").decode().strip()
    for p in package.iterdir():
        if p.name not in good:
            p.unlink()
    for name, data in good.items():
        (package / name).write_bytes(data)
    (root / "safe.py").write_text("pass # restored\n")
    if case == "renamed":
        (root / "experiments/result.py").unlink()
    commit(root)
    failures = assert_parity(root)
    assert bool(failures) == (case != "unchanged")
    if case in {"altered", "extra", "missing", "metadata"}:
        assert any(bad_revision[:12] in f for f in failures)
    assert not any("Z" * 30 in f for f in failures)


def test_snapshot_cache_uses_complete_mapping_and_mode(tmp_path, monkeypatch):
    root = repo(tmp_path)
    package = selected_showcase(root)
    guard.curator.seal(package)
    commit(root)
    for i in range(3):
        (root / "safe.py").write_text(f"pass # {i}\n")
        commit(root)
    calls = []
    validate = guard.curator.validate_files
    def record(files):
        calls.append(set(files))
        return validate(files)
    monkeypatch.setattr(guard.curator, "validate_files", record)
    assert guard.audit(root, ["HEAD"], worktree=False) == []
    assert len(calls) == 1
    calls.clear()
    assert guard.audit(root, ["HEAD"], worktree=False, cache=False) == []
    assert len(calls) == 4
    # Same blobs and manifest, different file mode: distinct validation identity.
    git(root, "update-index", "--chmod=+x", "showcase/example/README.md")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@users.noreply.github.com", "commit", "-qm", "mode change")
    calls.clear()
    assert guard.audit(root, ["HEAD"], worktree=False) == []
    assert len(calls) == 2


def test_cache_preserves_mode_rejection_and_ref_union(tmp_path):
    root = repo(tmp_path)
    package = selected_showcase(root)
    guard.curator.seal(package)
    commit(root)
    good = git(root, "rev-parse", "HEAD").decode().strip()
    oid = git(root, "rev-parse", "HEAD:showcase/example/README.md").decode().strip()
    git(root, "update-index", "--cacheinfo", "120000," + oid + ",showcase/example/README.md")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@users.noreply.github.com", "commit", "-qm", "unsafe mode")
    bad = git(root, "rev-parse", "HEAD").decode().strip()
    failures = assert_parity(root, [good, bad], worktree=False)
    assert any("symlink or submodule" in f for f in failures)
    assert any("unsealed" in f for f in failures)


def test_worktree_and_inventory_are_never_reused_between_audits(tmp_path):
    root = repo(tmp_path)
    package = selected_showcase(root)
    guard.curator.seal(package)
    commit(root)
    assert guard.audit(root, ["HEAD"]) == []
    Image.new("RGB", (3, 3), "blue").save(package / "image-01.png")
    assert any("altered" in f for f in assert_parity(root))
    (root / "unexpected.txt").write_text("private")
    assert any("unexpected.txt" in f for f in assert_parity(root))
    inventory = root / "PUBLIC_FILES.txt"
    inventory.write_text(inventory.read_text().replace("safe.py\n", ""))
    assert any("safe.py" in f and "inventory" in f for f in assert_parity(root))


def test_git_batch_is_binary_safe_and_cache_is_bounded(tmp_path):
    root = repo(tmp_path)
    data = b"a\n\0b\r\n" * 10
    path = root / "binary.dat"
    path.write_bytes(data)
    oid = git(root, "hash-object", "-w", str(path)).strip()
    with guard.BlobReader(root, cache_bytes=8) as reader:
        assert reader.read(oid) == data
        assert reader.cache == {} and reader.cached_bytes == 0
        assert reader.read(oid) == data
    assert reader.process is None
    with guard.BlobReader(root, cache_bytes=100) as reader:
        assert reader.read(oid) == data
        assert reader.cached_bytes == len(data)
        for i in range(5):
            path.write_bytes(bytes([i]) * 60)
            next_oid = git(root, "hash-object", "-w", str(path)).strip()
            reader.read(next_oid)
            assert reader.cached_bytes <= 100
    with guard.BlobReader(root, cache_bytes=0) as reader:
        reader.read(oid)
        assert not reader.cache


@pytest.mark.parametrize("response", [
    b"missing\n", b"F" * 40 + b" blob 1\nx\n", b"a" * 40 + b" tree 1\nx\n",
    b"a" * 40 + b" blob -1\n", b"a" * 40 + b" blob nope\n", b"a" * 40 + b" blob 01\nx\n",
    b"a" * 40 + b" blob 5\nab", b"a" * 40 + b" blob 1\nx!", b"a" * 40 + b" blob 1",
    b"a" * 40 + b" blob " + b"9" * 200 + b"\n",
])
def test_malformed_git_batch_response_fails_closed(tmp_path, response):
    reader = guard.BlobReader(tmp_path)
    reader.process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(response))
    with pytest.raises(ValueError):
        reader.read(b"a" * 40)


def test_missing_git_blob_fails_closed_and_child_exits(tmp_path):
    root = repo(tmp_path)
    reader = guard.BlobReader(root)
    with pytest.raises(ValueError):
        with reader:
            process = reader.process
            reader.read(b"0" * 40)
    assert reader.process is None and process.poll() is not None
    with pytest.raises(ValueError):
        with guard.BlobReader(root) as reader:
            reader.read(b"HEAD\nHEAD")


def test_divergent_second_ref_cannot_hide_unpublished_secret(tmp_path):
    root = repo(tmp_path)
    base = git(root, "rev-parse", "HEAD").decode().strip()
    (root / "safe.py").write_text("pass # clean branch\n")
    commit(root)
    clean = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "checkout", "-qb", "unsafe-fixture", base)
    (root / "safe.py").write_text("KEY = " + repr("sk-" + "Q" * 30))
    commit(root)
    unsafe = git(root, "rev-parse", "HEAD").decode().strip()
    assert guard.audit(root, [clean], worktree=False) == []
    assert any("credential" in f for f in assert_parity(root, [clean, unsafe], worktree=False))


def test_invalid_snapshot_is_cached_without_losing_commit_failures(tmp_path, monkeypatch):
    root = repo(tmp_path)
    selected_showcase(root)  # Deliberately unsealed in every snapshot.
    revisions = []
    for i in range(3):
        (root / "safe.py").write_text(f"pass # {i}\n")
        commit(root)
        revisions.append(git(root, "rev-parse", "HEAD").decode().strip())
    expected = legacy_audit(root, ["HEAD"], worktree=False)
    calls = []
    validate = guard.curator.validate_files
    def record(files):
        calls.append(set(files))
        return validate(files)
    monkeypatch.setattr(guard.curator, "validate_files", record)
    actual = guard.audit(root, ["HEAD"], worktree=False)
    assert actual == expected and len(calls) == 1
    assert all(any(revision[:12] in f and "unsealed" in f for f in actual) for revision in revisions)


def test_exactly_one_git_batch_process_per_audit(tmp_path, monkeypatch):
    root = repo(tmp_path)
    package = selected_showcase(root)
    guard.curator.seal(package)
    commit(root)
    original = subprocess.Popen
    batches = []
    def record(args, *positional, **kwargs):
        if "--batch" in args:
            batches.append(args)
        return original(args, *positional, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", record)
    assert guard.audit(root, ["HEAD"]) == []
    assert len(batches) == 1


@pytest.mark.parametrize("problem", ["stdin-close", "nonzero", "timeout"])
def test_git_batch_cleanup_fails_closed_and_closes_both_streams(tmp_path, problem):
    class BrokenClose(io.BytesIO):
        def close(self):
            super().close()
            raise OSError("synthetic close failure")
    class Child:
        def __init__(self):
            self.stdin = BrokenClose() if problem == "stdin-close" else io.BytesIO()
            self.stdout = io.BytesIO()
            self.kills = 0
            self.waits = 0
        def kill(self):
            self.kills += 1
        def wait(self, timeout=None):
            self.waits += 1
            if problem == "timeout" and self.waits == 1:
                raise subprocess.TimeoutExpired("git", timeout)
            return 1 if problem == "nonzero" else 0
    child = Child()
    reader = guard.BlobReader(tmp_path)
    reader.process = child
    with pytest.raises(ValueError, match="finish cleanly"):
        reader.__exit__(None, None, None)
    assert child.stdin.closed and child.stdout.closed and reader.process is None
    assert child.waits >= 1
    assert child.kills == (problem != "nonzero")
