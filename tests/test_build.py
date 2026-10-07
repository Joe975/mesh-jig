"""mesh_jig.build with a fake Blender: what is run, what is copied, and what comes back as the error."""
import subprocess

from glbkit import box, write_glb
from mesh_jig import build

SPEC = {"script": "ship.py", "variants": ["gun", "missile"], "glb": "ship_{variant}.glb", "join": True,
        "length_m": 0.08, "forward": "-Z", "single_mesh": True, "vertex_colors": True}
GOOD = "import bpy\n\n\ndef build():\n    pass\n"


def fake_blender(calls, size=(0.066, 0.026, 0.080), fail=""):
    """A runner that 'builds' a box GLB at --out and prints the runner's OK mark, or fails the named variant."""
    def run(argv, **kw):
        calls.append(argv)
        opts = dict(zip(argv[argv.index("--") + 1::2], argv[argv.index("--") + 2::2]))
        if fail and opts["--variant"] == fail:
            return subprocess.CompletedProcess(argv, 1, stdout='File "ship.py", line 4\nJIG_BUILD_ERROR: NameError: wing', stderr="")
        write_glb(opts["--out"], [("Ship", box(size), (0, 0, 0))])
        return subprocess.CompletedProcess(argv, 0, stdout=f"JIG_BUILD_OK objects=1 out={opts['--out']}", stderr="")
    return run


def test_every_variant_is_built_through_the_runner_and_gated(tmp_path):
    script = tmp_path / "a01" / "draft.py"
    script.parent.mkdir()
    script.write_text(GOOD)
    calls = []
    out = tmp_path / "a01" / "out"
    assert build.build_variants(script, out, SPEC, blender="blender-x", runner=fake_blender(calls)) == ""
    assert (out / "ship.py").read_text() == GOOD, "the script is held under the contract's name beside its GLBs"
    assert (out / "glb_stats.json").is_file()
    assert sorted(a[a.index("--variant") + 1] for a in calls) == ["gun", "missile"]
    argv = calls[0]
    assert argv[:5] == ["blender-x", "--background", "--factory-startup", "--python", str(build.RUNNER)]
    assert argv[-2:] == ["--join", "1"] and "--atlas" not in argv
    assert build.RUNNER.is_file()


def test_the_atlas_is_copied_beside_the_script_and_handed_to_the_runner(tmp_path):
    script = tmp_path / "ship.py"
    script.write_text(GOOD)
    atlas = tmp_path / "painted.png"
    atlas.write_bytes(b"png")
    calls = []
    out = tmp_path / "out"
    # the fake GLB has no texture, so the textured gate refuses it: the error is the contract's, not the build's
    err = build.build_variants(script, out, dict(SPEC, textured=True, variants=["gun"]), blender="b",
                               runner=fake_blender(calls), atlas=atlas)
    assert (out / "atlas.png").read_bytes() == b"png"
    assert calls[0][calls[0].index("--atlas") + 1] == str(out / "atlas.png")
    assert err.startswith("contract: ") and "embeds no image" in err


def test_a_failed_variant_returns_the_tail_of_blenders_log(tmp_path):
    script = tmp_path / "ship.py"
    script.write_text(GOOD)
    err = build.build_variants(script, tmp_path / "out", SPEC, blender="b", runner=fake_blender([], fail="missile"))
    assert err.startswith("variant missile: ") and "NameError: wing" in err and "line 4" in err


def test_a_contract_breach_is_the_error_even_when_blender_succeeded(tmp_path):
    script = tmp_path / "ship.py"
    script.write_text(GOOD)
    err = build.build_variants(script, tmp_path / "out", SPEC, blender="b", runner=fake_blender([], size=(0.066, 0.026, 0.120)))
    assert err.startswith("contract: ") and "length along Z is 0.1200 m" in err


def test_a_script_that_reaches_outside_is_refused_before_blender_runs(tmp_path):
    script = tmp_path / "ship.py"
    script.write_text("import os\n\n\ndef build():\n    os.remove('x')\n")
    calls = []
    err = build.build_variants(script, tmp_path / "out", SPEC, blender="b", runner=fake_blender(calls))
    assert err.startswith("model script refused: ") and "import of 'os'" in err and calls == []
    assert "does not exist" in build.build_variants(tmp_path / "nope.py", tmp_path / "out", SPEC, blender="b")


def test_a_missing_or_hung_blender_is_an_error_message_not_a_crash(tmp_path, monkeypatch):
    script = tmp_path / "ship.py"
    script.write_text(GOOD)
    monkeypatch.setattr(build, "find_blender", lambda: None)
    assert "Blender not found" in build.build_variants(script, tmp_path / "out", SPEC)

    def hung(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    assert "timed out" in build.build_variants(script, tmp_path / "out", SPEC, blender="b", runner=hung)

    def absent(argv, **kw):
        raise FileNotFoundError("no such file")
    assert "could not run Blender at b" in build.build_variants(script, tmp_path / "out", SPEC, blender="b", runner=absent)


def test_blender_is_found_from_the_environment_first(monkeypatch):
    assert build.find_blender({"MESH_JIG_BLENDER": "/opt/b/blender"}) == "/opt/b/blender"
    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/blender")
    assert build.find_blender({}) == "/usr/bin/blender"
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build, "_candidates", lambda: ["/nowhere/blender"])
    assert build.find_blender({}) is None
