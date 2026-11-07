"""mesh-jig new: an attempt started from another with no shell, the skill's "copy it to a new directory"."""
import json

import pytest

from mesh_jig import cli, project


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "p"
    (root / "attempts" / "a01-base").mkdir(parents=True)
    (root / "jig.json").write_text(json.dumps({"name": "test", "model": {}}), encoding="utf-8")
    (root / "attempts" / "a01-base" / "model.py").write_text("def build(): pass\n", encoding="utf-8")
    (root / "attempts" / "a01-base" / "atlas_plan.json").write_text("{}", encoding="utf-8")
    (root / "attempts" / "a01-base" / "eval.json").write_text("{}", encoding="utf-8")
    return project.load(root)


def test_new_copies_the_script_and_plan_of_the_attempt_it_starts_from(proj, capsys):
    assert cli.main(["new", str(proj.root), "a02-wider", "--from", "a01-base"]) == 0
    made = proj.attempts_dir / "a02-wider"
    assert (made / "model.py").read_text(encoding="utf-8") == "def build(): pass\n"
    assert (made / "atlas_plan.json").is_file() and not (made / "eval.json").exists()
    assert "model.py, atlas_plan.json copied from a01-base" in capsys.readouterr().out
    # an attempt directory given as a path works the same
    assert cli.main(["new", str(proj.root), "a03", "--from", str(proj.attempts_dir / "a02-wider")]) == 0


def test_new_with_no_source_makes_an_empty_attempt(proj, capsys):
    assert cli.main(["new", str(proj.root), "a01-fresh"]) == 0
    assert list((proj.attempts_dir / "a01-fresh").iterdir()) == []
    assert "empty, write model.py here" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["a01-base"], ["../escape"], ["a/b"], [".."], ["a02", "--from", "nowhere"]])
def test_new_never_overwrites_and_refuses_a_bad_name_or_source(proj, args):
    assert cli.main(["new", str(proj.root), *args]) == 2
    assert sorted(p.name for p in proj.attempts_dir.iterdir()) == ["a01-base"]


@pytest.mark.parametrize("name, starts", [("organic", "#"), ("recipe", ""), ("contract", "{")])
def test_docs_prints_each_file_the_skill_links_to(name, starts, capsys):
    assert cli.main(["docs", name]) == 0
    out = capsys.readouterr().out
    assert out.strip() and out.lstrip().startswith(starts)
    # the skill's links and the command name the same files
    skill = (cli.CHECKOUT / "skill" / "mesh-jig" / "SKILL.md").read_text(encoding="utf-8")
    assert f"(../../{cli.DOCS[name]})" in skill and f"mesh-jig docs {name}" in skill


def test_a_build_is_stale_when_its_script_changed_after_it(proj):
    import os
    from mesh_jig import glb
    attempt = proj.attempts_dir / "a01-base"
    assert cli.stale(proj, attempt) == "nothing is built yet"
    built = attempt / "out" / glb.judged_glb(proj.spec)
    built.parent.mkdir()
    built.write_bytes(b"glb")
    os.utime(attempt / "model.py", (1000, 1000))
    os.utime(attempt / "atlas_plan.json", (1000, 1000))
    os.utime(built, (2000, 2000))
    assert cli.stale(proj, attempt) == ""
    os.utime(attempt / "model.py", (3000, 3000))
    assert cli.stale(proj, attempt) == "model.py changed since it was built"


def test_build_says_its_size_in_blender_axes(proj, monkeypatch, capsys):
    import json as js
    from mesh_jig import build, glb

    def fake(script, out, spec, **kw):
        out.mkdir(parents=True, exist_ok=True)
        (out / glb.REPORT).write_text(js.dumps({"stats": {"m.glb": {
            "triangles": 10, "size": [0.6, 1.2, 0.4], "distinct_colors": 3, "node_triangles": [["Body", 10]]}}}))
        return ""
    monkeypatch.setattr(build, "build_variants", fake)
    script = proj.attempts_dir / "a01-base" / "model.py"
    assert cli.main(["build", str(proj.root), "--script", str(script)]) == 0
    assert "size in Blender axes X 0.600 (width), Y 0.400 (depth), Z 1.200 (height) m" in capsys.readouterr().out


def test_the_verdict_block_is_the_feedback_up_to_the_regions_and_views():
    said = ("[a01] built, in contract, rendered and measured: overlap 0.800\nmesh: 1980 of 2000 triangles (99%)\n"
            "gate face: FAILED - zones 0.40\nstanding: 0.0100 below the best attempt, a00 (0.6000).\n"
            "  the score's terms against a00: overlap 0.800 (was 0.790, +0.010)\n"
            "REGIONS (each marked region ...): face 0.45\nfront: silhouette overlap 0.82")
    block = cli.verdict_block(said)
    assert block.endswith("(was 0.790, +0.010)") and "REGIONS" not in block and "silhouette" not in block


def test_the_verdict_block_keeps_the_outline_table():
    said = ("[a01] built, in contract, rendered and measured: overlap 0.800\nstanding: the first measured attempt here.\n"
            "front: silhouette overlap 0.82\n"
            "YOUR MODEL AGAINST THE REFERENCE OUTLINES (millimetres; + means ...):\n"
            "- front view, stations along Z, error at the -X end/the +X end:\n    z-454 (z-18 mm):-6/+6 (-150/+150)\n"
            "Close the biggest of these and the outline matches the reference.\nHOW YOUR COLOURS READ (...):")
    block = cli.verdict_block(said)
    assert "z-454 (z-18 mm):-6/+6" in block and "Close the biggest" in block
    assert "silhouette overlap" not in block and "HOW YOUR COLOURS READ" not in block
