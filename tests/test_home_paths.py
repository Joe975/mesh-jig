"""No tracked file names a home directory or the checkout it is in: run records are published, a path under a home
directory carries the login name of whoever made the run, and a checkout's path says where on their machine it was.
`tools/scrub_home_paths.py` is what takes them out."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "scrub_home_paths.py"

spec = importlib.util.spec_from_file_location("scrub_home_paths", SCRIPT)
scrubber = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scrubber)

# spelled in pieces, so this file passes its own test
WIN, NIX, MAC = "C:" + "\\Users\\" + "sam", "/home" + "/sam", "/Users" + "/sam"


@pytest.mark.parametrize("text, expected", [
    (WIN + "\\AppData\\Temp\\run", "~\\AppData\\Temp\\run"),
    (WIN.replace("\\", "\\\\") + "\\\\x.png", "~\\\\x.png"),                       # JSON
    (WIN.replace("\\", "\\\\\\\\") + "\\\\\\\\x.png", "~\\\\\\\\x.png"),             # JSON inside JSON
    (WIN.replace("\\", "/") + "/x.png", "~/x.png"),
    ("/c" + MAC + "/x.png", "~/x.png"),                                            # Git Bash
    ("/mnt/c" + MAC + "/x.png", "~/x.png"),                                        # WSL
    ("cd " + NIX + "/src && ls", "cd ~/src && ls"),
    ('"cwd": "' + MAC + '/src"', '"cwd": "~/src"'),
])
def test_a_home_directory_becomes_a_tilde_in_every_spelling(text, expected):
    assert scrubber.scrub(text) == (expected, 1)


@pytest.mark.parametrize("text", [
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    "C:" + r"\Users\Public\Documents",
    "https://example.com" + NIX + "/page",
    "~" + NIX[:5] + "/still/relative",
    "PROJECT PROBLEMS:\n- none",
    "a/b" + MAC,
])
def test_other_paths_are_left_alone(text):
    assert scrubber.scrub(text) == (text, 0)


def test_a_bare_login_name_is_replaced_only_where_it_stands_alone():
    listing = "drwxr-xr-x 1 sam 197609 0 Oct  4 09:39 refs\nsample.png samuel sam-low"
    assert scrubber.scrub(listing, ("sam",)) == ("drwxr-xr-x 1 user 197609 0 Oct  4 09:39 refs\nsample.png samuel sam-low", 1)


def test_check_reports_and_changes_nothing_then_the_run_rewrites(tmp_path, capsys):
    f = tmp_path / "run" / "eval.json"
    f.parent.mkdir()
    before = '{"script": "' + WIN.replace("\\", "\\\\") + '\\\\model.py"}\r\n'
    f.write_bytes(before.encode("utf-8"))
    (tmp_path / "run" / "render.png").write_bytes(b"\x89PNG\0" + WIN.encode("utf-8"))
    assert scrubber.main(["--check", "--name", "sam", str(tmp_path)]) == 1
    assert f.read_bytes() == before.encode("utf-8")
    assert scrubber.main(["--name", "sam", str(tmp_path)]) == 0
    assert f.read_bytes() == b'{"script": "~\\\\model.py"}\r\n'
    assert (tmp_path / "run" / "render.png").read_bytes().endswith(WIN.encode("utf-8"))     # binaries are not touched
    assert scrubber.main(["--check", "--name", "sam", str(tmp_path)]) == 0
    assert "1 replacement(s) in 1 file(s) made" in capsys.readouterr().out


def test_no_blend_file_is_tracked():
    """A .blend holds the directory it was saved in, compressed where `text_of` cannot read it: the test below would
    pass over one. So none is tracked."""
    try:
        listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    blends = [n for n in listed.stdout.decode("utf-8").split("\0") if n.endswith((".blend", ".blend1"))]
    assert not blends, f"tracked .blend files (see .gitignore): {blends}"


JIG = r"D:\work\jig"                # a checkout, for the root tests


@pytest.mark.parametrize("text, expected", [
    (JIG + r"\experiments\x.png", r"experiments\x.png"),
    (r"d:\work\jig\.venv\Scripts\python.exe", r".venv\Scripts\python.exe"),
    (r'"script": "D:\\work\\jig\\attempts\\a1\\model.py"', r'"script": "attempts\\a1\\model.py"'),       # JSON
    (r"D:\\\\work\\\\jig\\\\x.png", "x.png"),                                                          # JSON inside JSON
    ("D:/work/jig/experiments/x.png", "experiments/x.png"),
    ("/d/work/jig/x.png", "x.png"),                                                                    # Git Bash
    ("/mnt/d/work/jig/x.png", "x.png"),                                                                # WSL
    (r'"cwd": "D:\\work\\jig"', '"cwd": "."'),                                                         # the root alone
    ("cd " + JIG + " && ls", "cd . && ls"),
    (r"built in D:\\work\\jig\nthen", r"built in .\nthen"),             # a JSON line break is not a directory `n`
    (JIG + "/attempts", "./attempts"),                                  # a second spelling of the slash is left
])
def test_a_path_under_a_root_becomes_relative_to_it_in_every_spelling(text, expected):
    assert scrubber.scrub(text, roots=scrubber.checkouts([JIG])) == (expected, 1)


@pytest.mark.parametrize("text", [
    JIG + r"-old\x.png",
    r"D:\work\jigsaw",
    r"E:\work\jig\x.png",
    "docs/d/work/jig/x.png",
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
])
def test_a_path_that_only_starts_like_a_root_is_left_alone(text):
    assert scrubber.scrub(text, roots=scrubber.checkouts([JIG])) == (text, 0)


def test_a_root_can_be_a_run_directory_under_home_or_from_slash_and_the_longest_goes_first():
    run = scrubber.checkouts([JIG, JIG + r"\runs\r1"])
    assert scrubber.scrub(JIG + r"\runs\r1\attempts\a1 " + JIG + r"\src", roots=run) == (r"attempts\a1 src", 2)
    worktree = scrubber.checkouts([WIN + r"\wt\jig"])
    assert scrubber.scrub(WIN + r"\wt\jig\src\x.py", roots=worktree) == (r"src\x.py", 1)
    assert scrubber.scrub("~/wt/jig/src/x.py", roots=worktree) == ("src/x.py", 1)       # scrubbed for home before
    assert scrubber.scrub(r"~\scratch\run1\attempts\a1", roots=scrubber.checkouts(["~/scratch/run1"])) == (r"attempts\a1", 1)
    posix = scrubber.checkouts(["/srv/ci/jig"])
    assert scrubber.scrub("/srv/ci/jig/a.png", roots=posix) == ("a.png", 1)
    assert scrubber.scrub("https://example.com/srv/ci/jig/a.png", roots=posix)[1] == 0
    with pytest.raises(ValueError):
        scrubber.checkouts(["D:"])


def test_the_run_takes_a_root_off_and_the_default_root_is_this_checkout(tmp_path, capsys):
    f = tmp_path / "eval.json"
    f.write_text('{"script": "' + str(ROOT / "attempts" / "model.py").replace("\\", "\\\\") + '", "out": "'
                 + JIG.replace("\\", "\\\\") + '\\\\out"}', encoding="utf-8")
    assert scrubber.main(["--check", "--name", "sam", str(f)]) == 1
    assert scrubber.main(["--name", "sam", str(f)]) == 0
    assert json.loads(f.read_text(encoding="utf-8"))["script"] in ("attempts\\model.py", "attempts/model.py")
    assert scrubber.main(["--name", "sam", "--root", JIG, str(f)]) == 0
    assert json.loads(f.read_text(encoding="utf-8"))["out"] == "out"
    assert capsys.readouterr().out.count("1 replacement(s) in 1 file(s)") == 3


def tracked_text():
    """(name, text) of every tracked text file."""
    try:
        listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    for name in filter(None, listed.stdout.decode("utf-8").split("\0")):
        path = ROOT / name
        text = scrubber.text_of(path) if path.is_file() else None
        if text:
            yield name, text


def test_no_tracked_file_names_a_home_directory():
    found = []
    for name, text in tracked_text():
        hits = scrubber.HOME.findall(text)
        if hits:
            found.append(f"{name}: {len(hits)} (first: {hits[0]})")
    assert not found, ("home directories in tracked files; run `python tools/scrub_home_paths.py`:\n"
                       + "\n".join(found[:20]))


def test_no_tracked_file_names_the_checkout_it_is_in():
    """A run made in this checkout writes this checkout's path into its records. Any run of slashes counts here,
    which is looser than what the scrubber takes out, so a spelling it misses shows up."""
    try:
        leaks = scrubber.root_patterns(str(ROOT), same_separator=False)
    except ValueError:
        pytest.skip("the checkout is at the top of a drive")
    found = []
    for name, text in tracked_text():
        hits = [m.group(0) for p in leaks for m in p.finditer(text)]
        if hits:
            found.append(f"{name}: {len(hits)} (first: {hits[0]})")
    assert not found, ("this checkout's path in tracked files; run `python tools/scrub_home_paths.py .`:\n"
                       + "\n".join(found[:20]))
