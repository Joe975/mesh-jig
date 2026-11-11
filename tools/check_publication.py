"""Fail closed on unreviewed files or recognizable private data in publication history.

Run using your virtual environment: python tools/check_publication.py HEAD
Install as a pre-push hook using --pre-push; matching values are never printed.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
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


class BlobReader:
    """One binary Git pipe per audit, with a bounded, invocation-local object cache."""

    def __init__(self, root: Path, *, batch: bool = True, cache_bytes: int = 32 * 1024 * 1024):
        self.root = root
        self.batch = batch
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.cached_bytes = 0
        self.process = None

    def __enter__(self):
        if self.batch:
            self.process = subprocess.Popen(
                ["git", "-c", "safe.directory=" + self.root.as_posix(), "cat-file", "--batch"],
                cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process is None:
            return
        process = self.process
        failed = False
        try:
            # A broken stdin must not prevent closing stdout or reaping our child.
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    failed = True
            if failed:
                process.kill()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()  # only the pipe this reader started, never another publisher
                process.wait()
                failed = True
                code = None
        finally:
            self.process = None
        if exc_type is None and (failed or code):
            raise ValueError("Git object reader did not finish cleanly")

    def read(self, oid: bytes) -> bytes:
        if not re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid):
            raise ValueError("invalid Git object ID")
        if oid in self.cache:
            self.cache.move_to_end(oid)
            return self.cache[oid]
        if not self.batch:
            data = git(self.root, "cat-file", "blob", oid.decode("ascii"))
        else:
            if self.process is None:
                raise ValueError("Git object reader is not open")
            process = self.process
            process.stdin.write(oid + b"\n")
            process.stdin.flush()
            header = process.stdout.readline(256)
            parts = header.rstrip(b"\n").split(b" ")
            if (not header.endswith(b"\n") or len(parts) != 3 or parts[0] != oid
                    or parts[1] != b"blob" or not re.fullmatch(rb"0|[1-9][0-9]*", parts[2])):
                raise ValueError("invalid or missing Git blob response")
            size = int(parts[2])
            if size > sys.maxsize:
                raise ValueError("invalid Git blob size")
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise ValueError("truncated Git blob response")
        if self.cache_bytes > 0 and len(data) <= self.cache_bytes:
            while self.cache and self.cached_bytes + len(data) > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cached_bytes -= len(old)
            self.cache[oid] = data
            self.cached_bytes += len(data)
        return data


def historical_showcases(entries, reader, cache):
    """Reuse validation only for the complete identical package in this invocation.

    Keys include the path, filenames, modes and blob IDs, not just the manifest.
    Cache failures too, but attach the current commit prefix in audit as before.
    """
    groups = {}
    failures = []
    for name, (mode, oid) in entries.items():
        if not name.startswith("showcase/"):
            continue
        if forbidden_path(name):
            failures.append(name + ": invalid showcase path")
            continue
        parent, _, filename = name.rpartition("/")
        groups.setdefault(parent, {})[filename] = (mode, oid)
    released = set()
    for parent, contents in groups.items():
        key = (parent, tuple(sorted((name, mode, oid) for name, (mode, oid) in contents.items())))
        result = cache.get(key) if cache is not None else None
        if result is None:
            files = {parent + "/" + name: reader.read(oid) for name, (_, oid) in contents.items()}
            selected, invalid = showcases(files)
            result = (frozenset(selected), tuple(invalid))
            if cache is not None:
                cache[key] = result
        selected, invalid = result
        released.update(selected)
        failures.extend(invalid)
    return released, failures


def audit(root: Path, refs: list[str], worktree: bool = True, *, cache: bool = True, batch: bool = True) -> list[str]:
    allowed = set((root / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines())
    failures = [name + ": forbidden inventory entry" for name in allowed if forbidden_path(name)]
    seen_blobs = set()
    snapshot_cache = {} if cache else None
    commits = git(root, "rev-list", *refs).decode().splitlines()
    with BlobReader(root, batch=batch, cache_bytes=32 * 1024 * 1024 if cache else 0) as reader:
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
                entries[name] = (mode, oid)
            released, invalid = historical_showcases(entries, reader, snapshot_cache)
            failures.extend(commit[:12] + ": " + failure for failure in invalid)
            for name, (_, oid) in entries.items():
                # Check every path, including two different names for the same blob.
                if name not in allowed or forbidden_path(name):
                    failures.append(commit[:12] + ": " + name + ": outside public inventory")
                identity = (name, oid, name in released)
                if identity not in seen_blobs:
                    seen_blobs.add(identity)
                    failures.extend(scan(name, reader.read(oid), allowed, released))
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
        # Working files are mutable: always read and validate them afresh.
        released, invalid = showcases(contents)
        failures.extend(invalid)
        for name, data in contents.items():
            failures.extend(scan(name, data, allowed, released))
    return sorted(set(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="*", default=["HEAD"])
    parser.add_argument("--pre-push", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="repeat validation without invocation-local caches")
    parser.add_argument("--no-batch", action="store_true", help="read each Git blob in a separate process")
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
        failures = audit(ROOT, refs, cache=not args.no_cache, batch=not args.no_batch)
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
