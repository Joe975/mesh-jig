"""Privacy guard must reject ignored ancestry, renamed private files and unsafe metadata."""
import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publication_guard", ROOT / "tools/check_publication.py")
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
