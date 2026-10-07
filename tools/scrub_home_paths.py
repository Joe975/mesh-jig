"""Take machine paths out of run records, so a run can be published: home directories, the login name, and the
directory the checkout or the run was in.

    python experiments/scrub_home_paths.py [--check] [--name LOGIN] [--root DIR ...] [<dir or file> ...]
                                                                              (default: experiments examples)

`eval.json`, agent transcripts and coding-agent event logs hold absolute paths. A run made under a home directory
(a scratch directory, a checkout in ~/src) therefore carries the login name of whoever made it, in every spelling a
log can hold: the Windows profile directory with its backslashes single, doubled by JSON or doubled twice, the same
with forward slashes, its Git Bash and WSL forms, and the Linux and macOS home directories. Each becomes `~`, and the
bare login name (an `ls -l` owner column) becomes `user`.

A record also says where on the machine the checkout was. Each `--root` (default: the checkout this script is in) is
taken off the front of every path under it, in the same spellings, so `<root>/experiments/x.png` becomes
`experiments/x.png` and the root alone becomes `.`. A root may be a run's own directory outside the checkout, and
may be written from `~`. The longest root is taken first.

Text files only, rewritten in place; `--check` changes nothing and exits 1 when anything would change.
`tests/test_home_paths.py` holds every tracked file to the same patterns, and has an example of each spelling.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "build", "__pycache__", ".pytest_cache", "node_modules"}
NAME = r"[A-Za-z0-9_.-]+"
HOME = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/]+[Uu]sers[\\/]+(?!Public\b)" + NAME        # the Windows profile directory, at any escape depth
    + r"|(?<![\w.~])/(?:[a-z]|mnt/[a-z])/Users/" + NAME                      # Git Bash and WSL spellings of the same
    + r"|(?<![\w.~/])/(?:home|Users)/" + NAME)                               # Linux and macOS
HOME_MARK, NAME_MARK, ROOT_MARK = "~", "user", "."
SEP = r"[\\/]+"
INTO = r"(?=[^\s\"'`\\/)\]}>,;:|])"         # what follows a separator when the path goes on into the root


def root_patterns(root: str, same_separator: bool = True) -> list[re.Pattern]:
    r"""One directory in each spelling a log can hold it in: with a drive letter (any slash, at any escape depth), as
    Git Bash and WSL write that drive, from `/`, and from `~` when it is under a home directory. Each pattern also
    takes the separator after the root when a path goes on from there (group `into`).

    Within one path every separator is spelt the same way, and `same_separator` holds a match to that: in JSON a
    path's separators are two backslashes, so the `\n` after one is a line break and not a directory `n`. Without it
    any run of slashes will do, which is what a test that looks for leftovers wants."""
    parts = [p for p in re.split(SEP, str(root).strip()) if p]
    if len(parts) < 2:
        raise ValueError(f"a root needs a directory under a drive, `/` or `~`: {root!r}")

    def spelt(head: str, names: list[str], sep: str | None = None) -> re.Pattern:
        first, later = ((sep, sep) if sep else (rf"(?P<s>{SEP})", r"(?P=s)") if same_separator else (SEP, SEP))
        body = head + first + later.join(re.escape(n) for n in names) + r"(?![\w.-])"
        return re.compile(body + rf"(?:(?P<into>{later}){INTO}|(?:{later})?)")

    drive = re.fullmatch(r"([A-Za-z]):", parts[0])
    if drive:
        d = drive.group(1)
        out = [spelt(rf"(?<![A-Za-z])[{d.lower()}{d.upper()}]:", parts[1:]),
               spelt(rf"(?<![\w.~])/(?:mnt/)?{d.lower()}", parts[1:], "/")]
    elif parts[0] == "~":
        return [spelt("~", parts[1:])]
    else:
        out = [spelt(r"(?<![\w.~:/])", parts, "/")]
    home = HOME.match(str(root).strip())
    under = [p for p in re.split(SEP, str(root).strip()[home.end():]) if p] if home else []
    return out + ([spelt("~", under)] if under else [])


def checkouts(roots) -> tuple[re.Pattern, ...]:
    """The patterns of several roots, the longest root first: a run directory inside a checkout goes before it."""
    return tuple(p for root in sorted(map(str, roots), key=len, reverse=True) for p in root_patterns(root))


def scrub(text: str, names: tuple[str, ...] = (), roots: tuple[re.Pattern, ...] = ()) -> tuple[str, int]:
    """`text` with every path under a root (`checkouts`) relative to it, every home directory as `~` and every bare
    login name in `names` as `user`, and how many replacements that took."""
    count = 0
    for pattern in roots:
        text, n = pattern.subn(lambda m: "" if m.group("into") else ROOT_MARK, text)
        count += n
    text, n = HOME.subn(HOME_MARK, text)
    count += n
    for name in names:
        if name:
            text, n = re.subn(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", NAME_MARK, text)
            count += n
    return text, count


def text_of(path: Path) -> str | None:
    """A file's text, or None for a binary file (a NUL byte, or bytes that are not UTF-8)."""
    raw = path.read_bytes()
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def files(paths: list[Path]):
    for root in paths:
        if root.is_file():
            yield root
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts):
                yield p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path, default=[Path("experiments"), Path("examples")])
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 when a file would change")
    ap.add_argument("--name", action="append", help="a login name to replace where it stands alone "
                                                    "(default: this machine's home directory name)")
    ap.add_argument("--root", action="append", help="a directory to take off the front of every path under it: a "
                                                    "checkout, or a run's own directory (default: this checkout)")
    a = ap.parse_args(argv)
    names = tuple(a.name or [Path.home().name])
    roots = checkouts(a.root or [CHECKOUT])
    changed = total = 0
    for path in files(a.paths):
        text = text_of(path)
        if text is None:
            continue
        new, n = scrub(text, names, roots)
        if not n:
            continue
        changed, total = changed + 1, total + n
        if a.check:
            print(f"{n:6d}  {path.as_posix()}")
        else:
            path.write_bytes(new.encode("utf-8"))
    print(f"{total} replacement(s) in {changed} file(s){' needed' if a.check else ' made'}")
    return 1 if a.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
