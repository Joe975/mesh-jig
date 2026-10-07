"""Which mesh-jig made a record: the checkout's commit, whether the harness had moved on from it, and its web address."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mesh_jig import harness

git = pytest.mark.skipif(not shutil.which("git"), reason="git is not installed")


def run(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A repository shaped like this one: src/pkg (the package), skill/, and a README that is not the harness."""
    root = tmp_path / "repo"
    package = root / "src" / "pkg"
    package.mkdir(parents=True)
    (root / "skill").mkdir()
    (package / "__init__.py").write_text("")
    (root / "skill" / "SKILL.md").write_text("guidance\n")
    (root / "README.md").write_text("readme\n")
    run(tmp_path, "init", "-q", str(root))
    run(root, "add", "-A")
    run(root, "commit", "-q", "-m", "first")
    return root, package


@git
def test_the_version_is_the_checkouts_commit_and_clean_until_the_harness_changes(checkout):
    root, package = checkout
    head = run(root, "rev-parse", "--short", "HEAD")
    assert harness.version(package) == {"commit": head, "dirty": False}
    (root / "README.md").write_text("a new readme\n")
    (root / "notes.txt").write_text("not the harness\n")
    assert harness.version(package) == {"commit": head, "dirty": False}, "only src/ and skill/ are the harness"
    (root / "skill" / "SKILL.md").write_text("better guidance\n")
    assert harness.version(package) == {"commit": head, "dirty": True}, "guidance alone is a change to the harness"
    run(root, "commit", "-q", "-am", "guidance")
    assert harness.version(package) == {"commit": run(root, "rev-parse", "--short", "HEAD"), "dirty": False}
    (package / "new_module.py").write_text("")
    assert harness.version(package)["dirty"], "a new file in the package counts before it is added"


@git
def test_a_copy_installed_inside_some_other_repository_has_no_version(checkout, tmp_path):
    root, _ = checkout
    installed = root / "venv" / "site-packages" / "pkg"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    assert harness.version(installed) is None and harness.repo_url(installed) is None
    outside = tmp_path / "nowhere" / "pkg"
    outside.mkdir(parents=True)
    assert harness.version(outside) is None


@git
def test_a_commit_is_in_history_only_when_head_descends_from_it(checkout):
    root, package = checkout
    first = run(root, "rev-parse", "--short", "HEAD")
    assert harness.in_history(first, package)
    (root / "skill" / "SKILL.md").write_text("better guidance\n")
    run(root, "commit", "-q", "-am", "guidance")
    assert harness.in_history(first, package), "an ancestor of HEAD"
    assert harness.in_history(run(root, "rev-parse", "HEAD"), package), "the full hash of HEAD"
    # the same tree as one commit with no parent, which is what a squashed history is: the old commits are still
    # in the object store and no longer in the history
    squashed = run(root, "commit-tree", "HEAD^{tree}", "-m", "one commit")
    run(root, "reset", "-q", "--soft", squashed)
    assert not harness.in_history(first, package)
    assert not harness.in_history("0123abc", package), "a commit this checkout has never had"
    assert not harness.in_history(first, root / "venv" / "pkg"), "no checkout, no history"


@git
def test_the_repo_address_is_the_origin_remote_without_credentials(checkout):
    root, package = checkout
    assert harness.repo_url(package) is None
    run(root, "remote", "add", "origin", "https://someone:s3cret@github.com/owner/repo.git")
    assert harness.repo_url(package) == "https://github.com/owner/repo"


@pytest.mark.parametrize("remote, page", [
    ("https://github.com/owner/repo.git", "https://github.com/owner/repo"),
    ("https://github.com/owner/repo", "https://github.com/owner/repo"),
    ("https://user:token@github.com/owner/repo.git", "https://github.com/owner/repo"),
    ("git@github.com:owner/repo.git", "https://github.com/owner/repo"),
    ("ssh://git@gitlab.example.com:2222/group/sub/repo.git", "https://gitlab.example.com/group/sub/repo"),
    ("/srv/git/repo.git", None),
    ("C:/work/repo", None),
    ("file:///srv/git/repo.git", None),
    ("", None),
])
def test_a_remote_becomes_its_web_address(remote, page):
    assert harness.web_url(remote) == page
