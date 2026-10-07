"""Fail closed on unreviewed files or recognizable private data in publication history.

Run using your virtual environment: python tools/check_publication.py HEAD
Install as a pre-push hook using --pre-push; matching values are never printed.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("showcase_curator", ROOT / "skill/mesh-jig-publish-results/scripts/curate.py")
curator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curator)
PATTERNS = {
    "home directory": re.compile(rb"(?:[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b)[\w.-]+|/(?:home|Users)/[\w.-]+)"),
    "machine checkout": re.compile(rb"[A-Za-z]:[\\/]+Repos[\\/]+[\w.-]+", re.I),
    "credential": re.compile(rb"(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"),
}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-c", "safe.directory=" + root.as_posix(), *args], cwd=root)


def forbidden_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.parts[:1] == ("showcase",):
        if len(path.parts) != 3 or not curator.SLUG.fullmatch(path.parts[1]):
            return True
        if path.name not in {"README.md", "summary.json", "release.json"} and not curator.MEDIA.fullmatch(path.name):
            return True
    return (path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name
            or any(part in {"experiments", "local-work", "private", "attempts", "agent", "overnight", ".git", ".venv"} for part in path.parts)
            or (path.suffix.lower() in {".blend", ".blend1", ".glb", ".png", ".jpg", ".jpeg", ".webp", ".jsonl", ".log", ".out"}
                and not (path.parts[:1] == ("showcase",) and len(path.parts) == 3 and curator.MEDIA.fullmatch(path.name)))
            or (path.name.startswith(".env") and path.name != ".env.example"))


def scan(name: str, data: bytes, allowed: set[str], released: set[str] | None = None) -> list[str]:
    failures = []
    if name not in allowed or forbidden_path(name):
        failures.append(name + ": outside public inventory")
    if name.startswith("showcase/") and name not in (released or set()):
        failures.append(name + ": not part of a valid sealed showcase")
    if b"\0" in data[:8192] and name not in (released or set()):
        failures.append(name + ": binary content requires review")
    if name in (released or set()) and curator.MEDIA.fullmatch(PurePosixPath(name).name):
        return failures  # structural/metadata checks and byte hashes already passed
    for label, pattern in PATTERNS.items():
        if pattern.search(data):
            failures.append(name + ": " + label)
    return failures


def showcases(files: dict[str, bytes]) -> tuple[set[str], list[str]]:
    groups = {}
    failures = []
    for name, data in files.items():
        if not name.startswith("showcase/"):
            continue
        if forbidden_path(name):
            failures.append(name + ": invalid showcase path")
            continue
        parent, _, filename = name.rpartition("/")
        groups.setdefault(parent, {})[filename] = data
    released = set()
    for parent, contents in groups.items():
        try:
            curator.validate_files(contents)
        except (ValueError, TypeError, KeyError, OSError, struct.error):
            failures.append(parent + ": unsealed, altered or unsupported showcase content")
        else:
            released.update(parent + "/" + name for name in contents)
    return released, failures


def audit(root: Path, refs: list[str], worktree: bool = True) -> list[str]:
    allowed = set((root / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines())
    failures = [name + ": forbidden inventory entry" for name in allowed if forbidden_path(name)]
    seen_blobs = set()
    commits = git(root, "rev-list", *refs).decode().splitlines()
    for commit in commits:
        metadata = git(root, "show", "-s", "--format=%ae%n%ce%n%B", commit)
        lines = metadata.decode().splitlines()
        if any(not line.endswith("@users.noreply.github.com") for line in lines[:2]):
            failures.append(commit[:12] + ": author/committer email is not GitHub noreply")
        for label, pattern in PATTERNS.items():
            if pattern.search(metadata):
                failures.append(commit[:12] + ": commit metadata contains " + label)
        entries = {}
        for row in git(root, "ls-tree", "-rz", commit).split(b"\0"):
            if not row:
                continue
            header, name_bytes = row.split(b"\t", 1)
            mode, kind, oid = header.split()
            name = name_bytes.decode("utf-8")
            if mode not in {b"100644", b"100755"} or kind != b"blob":
                failures.append(name + ": symlink or submodule requires review")
                continue
            entries[name] = oid
        selected = {name: git(root, "cat-file", "blob", oid.decode()) for name, oid in entries.items() if name.startswith("showcase/")}
        released, invalid = showcases(selected)
        failures.extend(commit[:12] + ": " + failure for failure in invalid)
        for name, oid in entries.items():
            # Check every path, including two different names for the same blob.
            if name not in allowed or forbidden_path(name):
                failures.append(commit[:12] + ": " + name + ": outside public inventory")
            identity = (name, oid, name in released)
            if identity not in seen_blobs:
                seen_blobs.add(identity)
                failures.extend(scan(name, selected[name] if name in selected else git(root, "cat-file", "blob", oid.decode()), allowed, released))
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
        released, invalid = showcases(contents)
        failures.extend(invalid)
        for name, data in contents.items():
            failures.extend(scan(name, data, allowed, released))
    return sorted(set(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="*", default=["HEAD"])
    parser.add_argument("--pre-push", action="store_true")
    args = parser.parse_args(argv)
    refs = args.refs
    if args.pre_push:
        rows = [line.split() for line in sys.stdin if line.strip()]
        if any(len(row) != 4 for row in rows):
            parser.error("invalid pre-push update")
        refs = [row[1] for row in rows if set(row[1]) != {"0"}]
        if not refs:
            return 0
    try:
        failures = audit(ROOT, refs)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("Publication check could not complete: " + type(exc).__name__, file=sys.stderr)
        return 1
    if failures:
        print("Publication blocked:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("Publication check passed: reviewed inventory, history, metadata and working files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
