"""A project on disk: jig.json, its defaults and problems, the sheet cutter, the brief an agent reads, and the CLI
commands that need neither Blender nor a judge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mesh_jig import brief, cli, project, sheet


def make_sheet(path: Path) -> Path:
    a = np.full((200, 300, 3), 220, dtype=np.uint8)
    a[:, :4] = 210                               # the background sample column
    a[10:30, 10:120] = 20                        # title text
    a[60:140, 60:240] = (150, 40, 30)            # the model
    Image.fromarray(a).save(path)
    return path


def write_jig(root: Path, **over) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data = {"name": "ship", "model": {"length_m": 0.08, "forward": "-Z", "vertex_colors": True, "min_colors": 2}}
    data.update(over)
    (root / "jig.json").write_text(json.dumps(data))
    return root


def view(path: Path, w: int, h: int) -> None:
    im = Image.new("RGB", (w + 40, h + 40), (235, 234, 230))
    im.paste((150, 55, 45), (20, 20, 20 + w, 20 + h))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def test_the_model_block_gets_its_defaults_and_a_missing_project_says_how_to_start_one(tmp_path):
    proj = project.load(write_jig(tmp_path / "p"))
    assert proj.spec["script"] == "model.py" and proj.spec["glb"] == "ship.glb" and proj.length_mm == 80.0
    variants = project.load(write_jig(tmp_path / "v", model={"variants": ["gun", "missile"]}))
    assert variants.spec["glb"] == "ship_{variant}.glb"
    assert project.load(tmp_path / "p" / "jig.json").root == proj.root, "the file or its directory"
    with pytest.raises(project.ProjectError, match="mesh-jig init"):
        project.load(tmp_path / "nowhere")
    (tmp_path / "p" / "jig.json").write_text("{not json")
    with pytest.raises(project.ProjectError, match="not JSON"):
        project.load(tmp_path / "p")


def test_problems_name_what_would_stop_a_tool(tmp_path):
    assert "needs a `model` block" in project.load(write_jig(tmp_path / "a", model=None)).problems()[0]
    bad = project.load(write_jig(tmp_path / "b", palette={"red": "crimson"},
                                 model={"variants": ["a", "b"], "glb": "one.glb", "judge_variant": "c", "length_m": -1,
                                        "shots": [{"name": "front"}]}))
    text = " | ".join(bad.problems())
    for fragment in ("must contain {variant}", "judge_variant 'c'", "length_m must be a positive", "needs name, cam_pos",
                     "palette:", "no reference views"):
        assert fragment in text, fragment
    good = write_jig(tmp_path / "c", palette={"red": "#96372d"})
    view(good / "refs" / "top.png", 100, 200)
    assert project.load(good).problems() == []


def test_references_are_the_images_in_refs_and_outlines_need_a_length_and_three_views(tmp_path):
    root = write_jig(tmp_path / "p")
    view(root / "refs" / "top.png", 100, 200)
    view(root / "refs" / "side.png", 200, 50)
    (root / "refs" / "notes.txt").write_text("x")
    view(root / "refs" / "sil_top.png", 10, 10)          # an outline diff left in the folder is not a reference
    proj = project.load(root)
    assert sorted(proj.references) == ["side", "top"] and proj.reference_profile() == {}
    view(root / "refs" / "rear.png", 100, 50)
    measured = project.load(root).reference_profile()
    assert measured["length_mm"] == 80.0 and abs(measured["span_mm"] - 40.0) < 2.0
    no_length = project.load(write_jig(tmp_path / "q", model={}, refs="../p/refs"))
    assert sorted(no_length.references) == ["rear", "side", "top"] and no_length.reference_profile() == {}


def test_the_sheet_is_cut_into_masked_views(tmp_path):
    root = write_jig(tmp_path / "p", sheet={"source": "s.png", "mask_sample_x": 2, "crops": {
        "hero": {"box": [0, 0, 300, 200], "mask": [[5, 5, 130, 35]]}, "side": {"box": [50, 50, 250, 150]}}})
    make_sheet(root / "s.png")
    written = sheet.write_refs(project.load(root))
    assert [p.name for p in written] == ["hero.png", "side.png"]
    hero = np.asarray(Image.open(root / "refs" / "hero.png"))
    assert (hero[10:30, 10:120] == 210).all(), "title text painted with the sheet's background column"
    assert tuple(hero[100, 150]) == (150, 40, 30), "the model is untouched"
    assert Image.open(root / "refs" / "side.png").size == (200, 100)


def test_the_sheet_refuses_a_box_outside_the_image_and_a_project_without_a_sheet(tmp_path):
    root = write_jig(tmp_path / "p", sheet={"source": "s.png", "crops": {"hero": {"box": [0, 0, 400, 200]}}})
    make_sheet(root / "s.png")
    with pytest.raises(ValueError, match="sheet.crops.hero: crop box .* is outside"):
        sheet.write_refs(project.load(root))
    with pytest.raises(ValueError, match="no `sheet` block"):
        sheet.write_refs(project.load(write_jig(tmp_path / "q")))
    with pytest.raises(ValueError, match="does not exist"):
        sheet.write_refs(project.load(write_jig(tmp_path / "r", sheet={"source": "gone.png", "crops": {"hero": {"box": [0, 0, 1, 1]}}})))


def test_init_writes_a_project_the_loader_accepts_and_never_overwrites_one(tmp_path):
    src = make_sheet(tmp_path / "concept.png")
    written = sheet.init(tmp_path / "rover", length_m=0.5, sheet=src)
    assert {p.name for p in written} == {"sheet.png", "jig.json", "brief.md"}
    proj = project.load(tmp_path / "rover")
    assert proj.name == "rover" and proj.spec["length_m"] == 0.5 and proj.spec["glb"] == "rover.glb"
    assert proj.data["sheet"]["crops"]["top"]["box"] == [0, 0, 300, 200], "one full-frame box per view, to tighten"
    assert [p.name for p in sheet.write_refs(proj)] == ["hero.png", "top.png", "side.png", "rear.png"]
    with pytest.raises(ValueError, match="already exists"):
        sheet.init(tmp_path / "rover")


def test_the_brief_carries_the_rules_the_contract_the_outlines_and_the_zones(tmp_path):
    root = write_jig(tmp_path / "p", palette={"crimson": "#96372d", "ivory": "#d7d2c6"},
                     model={"length_m": 0.08, "span_m": 0.04, "forward": "-Z", "vertex_colors": True, "min_colors": 2,
                            "variants": ["gun", "missile"], "join": True, "max_triangles": 1500, "min_triangles": 300})
    (root / "brief.md").write_text("A crimson slab.\n")
    view(root / "refs" / "top.png", 100, 200)
    view(root / "refs" / "side.png", 200, 50)
    view(root / "refs" / "rear.png", 100, 50)
    view(root / "refs" / "belly.png", 100, 200)
    text = brief.text(project.load(root))
    for fragment in ("A crimson slab.", "def build():", "NOSE POINTS ALONG BLENDER +Y", "VARIANT", "'gun', 'missile'",
                     "ship_gun.glb, ship_missile.glb", "Length nose-to-tail along Blender Y: 0.08 m",
                     "Wingspan across Blender X: 0.04 m", "Triangles per GLB: 300 to 1500", "COLOUR IS VERTEX COLOUR",
                     "Fewer than 2 distinct colours", "MEASURED REFERENCE OUTLINES", "COLOUR ZONES READ OFF",
                     "C crimson, I ivory", "hero_<variant>", "crimson #96372d", "mesh-jig eval",
                     "no camera of the same name (never measured): ['belly']"):
        assert fragment in text, fragment
    assert "TEXTURED CONTRACT" not in text
    textured = brief.model_facts({"textured": True, "glb": "rock.glb", "shots": [{"name": "front"}]})
    assert "TEXTURED CONTRACT" in textured and "atlas.png" in textured and "fixed cameras: front" in textured
    assert "NOSE" not in textured and "the brief says which way the model faces" in textured


def test_the_brief_says_where_a_projects_own_cameras_stand_in_the_scripts_axes():
    # jig.json gives cameras in glTF axes (Y up, -Z forward); the script is written in Blender's (Z up, +Y forward)
    assert [brief.blender_axis(v) for v in ([0, 0, -1], [0, 1, 0], [-1, 0, 0], [-2.4, 1.35, -3.7])] == [
        "+Y", "+Z", "-X", "-X +Y +Z"]
    shots = [{"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.1},
             {"name": "side", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0]},
             {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "cam_up": [0, 0, 1]},
             {"name": "unplaced"}]
    said = brief.camera_facts({"shots": shots})
    assert "front stands on the +Y side (picture right is -X, picture up is +Z)" in said
    assert "side stands on the -X side (picture right is -Y, picture up is +Z)" in said
    assert "top stands on the +Z side (picture right is -X, picture up is -Y)" in said and "unplaced" not in said
    assert said in brief.model_facts({"glb": "mech.glb", "shots": shots})
    assert brief.camera_facts({}) == "" and "Where each camera stands" not in brief.model_facts({"length_m": 1, "forward": "-Z"})


def test_the_cli_starts_a_project_reads_it_back_and_reports_bad_requests(tmp_path, capsys):
    root = tmp_path / "rover"
    assert cli.main(["init", str(root), "--length", "0.5", "--sheet", str(make_sheet(tmp_path / "concept.png"))]) == 0
    assert cli.main(["sheet", str(root)]) == 0
    assert cli.main(["palette", str(root), "--colors", "2", "--write"]) == 0
    assert list(project.load(root).palette) == ["c1", "c2"]
    assert cli.main(["palette", str(root), "--write"]) == 2, "a palette already there is not overwritten"
    capsys.readouterr()
    assert cli.main(["brief", str(root)]) == 0
    out = capsys.readouterr().out
    assert "PROJECT rover" in out and "PALETTE" in out and "REFERENCE VIEWS" in out
    assert cli.main(["brief", str(tmp_path / "nowhere")]) == 2
    assert "mesh-jig init" in capsys.readouterr().err
    assert cli.main(["eval", str(root), "--script", str(root / "x.py"), "--judge"]) == 2
    assert "--judge needs --against" in capsys.readouterr().err
    assert cli.main(["eval", str(root), "--script", str(root / "x.py"), "--judge", "--against", str(root)]) == 2
    assert "needs an API key" in capsys.readouterr().err
    assert cli.main(["judge", str(root), str(root)]) == 2
