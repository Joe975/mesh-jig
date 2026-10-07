"""Which mesh-jig made a record: the commit of the checkout this package runs from.

A result is only comparable to another made by the same harness, and a change to nothing but the guidance (the
skill, the brief's wording, the agent's prompt) is a change to the harness. So every evaluation records the commit,
and whether the working tree had moved on from it: `mesh-jig agent` and the CLI run the working tree, not the commit.

A record holds the commit and nothing about the machine or the remote (a remote's URL can carry a token). The page
that shows a record makes its link from the remote of the checkout it is served from (`repo_url`).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
WATCHED = ("src", "skill")          # what the harness is: the package and the skills. A change elsewhere is not one


def _git(cwd: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _checkout(package: Path) -> Path | None:
    """The git checkout the package is tracked in; None for an installed copy (site-packages inside some other
    repository is not that repository's code) or without git."""
    top = _git(package, "rev-parse", "--show-toplevel")
    if not top or _git(package, "ls-files", "--error-unmatch", "__init__.py") is None:
        return None
    return Path(top)


def version(package: Path = PACKAGE) -> dict | None:
    """{"commit": the short hash, "dirty": whether a file of the harness differs from it}, or None when the package
    is not in a checkout."""
    root = _checkout(package)
    commit = _git(root, "rev-parse", "--short", "HEAD") if root else None
    if not commit:
        return None
    changed = _git(root, "status", "--porcelain", "--", *WATCHED)
    return {"commit": commit, "dirty": bool(changed)}


def in_history(commit: str, package: Path = PACKAGE) -> bool:
    """Whether a recorded commit is in the checkout's own history (HEAD or an ancestor of it). A record made on
    another history names a commit the remote's page does not have, and a link to it leads nowhere."""
    root = _checkout(package)
    return bool(root) and _git(root, "merge-base", "--is-ancestor", f"{commit}^{{commit}}", "HEAD") is not None


def web_url(remote: str) -> str | None:
    """A remote's URL as the https address of its page, without credentials: git@host:owner/repo.git,
    ssh://git@host/owner/repo and https://user:token@host/owner/repo.git all give https://host/owner/repo.
    None for a path on disk or anything else."""
    remote = (remote or "").strip()
    m = (re.fullmatch(r"(?:https?|ssh|git)://(?:[^@/]+@)?([^/:]+)(?::\d+)?/(.+?)", remote)
         or re.fullmatch(r"[^@/\s]+@([^:/\s]+):(.+?)", remote))
    if not m or "." not in m.group(1):
        return None
    path = re.sub(r"\.git$", "", m.group(2).strip("/"))
    return f"https://{m.group(1)}/{path}" if path else None


def repo_url(package: Path = PACKAGE) -> str | None:
    """Where the checkout's commits can be looked at: its `origin` remote as a web address, or None."""
    root = _checkout(package)
    return web_url(_git(root, "remote", "get-url", "origin") or "") if root else None
