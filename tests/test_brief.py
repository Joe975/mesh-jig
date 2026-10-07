"""mesh_jig.brief.reading_facts: which views count, which outline table a project has and how a row of it reads are
written from the project, so the skill and the agent's system prompt never state them for a subject they do not fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mesh_jig import agent, brief, project, render

FRONT = {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.3}
SIDE = {"name": "side", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.3}
REAR = {"name": "rear", "cam_pos": [0, 0, 3], "cam_target": [0, 0, 0], "ortho": 1.3}
RIGHT = {"name": "right", "cam_pos": [3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.3}
TOP = {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "cam_up": [0, 0, 1], "ortho": 1.3}
HERO = {"name": "hero", "cam_pos": [-2.4, 1.35, -3.7], "cam_target": [0, 0, 0], "ortho": 1.3}
SKILL = Path(__file__).resolve().parents[1] / "skill" / "mesh-jig" / "SKILL.md"
# what was true of the first subject (a craft scored on three named views) and was told to every builder since
CRAFT_ONLY = ("wing", "top, side and rear", "length, span and height", "ahead of centre")


def make(root: Path, views: dict[str, tuple[int, int]], model: dict, **over) -> project.Project:
    (root / "refs").mkdir(parents=True)
    for name, (w, h) in views.items():
        im = Image.new("RGB", (w + 40, h + 40), (235, 234, 230))
        im.paste((150, 55, 45), (20, 20, 20 + w, 20 + h))
        im.save(root / "refs" / f"{name}.png")
    (root / "jig.json").write_text(json.dumps({"name": "thing", "model": model, **over}))
    return project.load(root)


def test_a_craft_is_told_its_three_views_and_its_station_table(tmp_path):
    proj = make(tmp_path / "p", {"hero": (120, 80), "top": (100, 200), "side": (200, 50), "rear": (100, 50)},
                {"length_m": 0.08, "forward": "-Z"}, palette={"red": "#96372d"})
    assert brief.scored_views(proj) == (["top", "side", "rear"], ["hero"])
    said = brief.reading_facts(proj)
    assert "taken over top, side, rear ONLY. hero is compared for you to look at and does not count." in said
    assert "against the craft table above" in said and "`y+40:+7`" in said
    assert "only gets the overall length, span and height right" in said
    assert "- zones: the share of cells" in said and said in brief.text(proj)


def test_a_project_with_its_own_scoring_is_told_those_views_and_never_the_craft_ones(tmp_path):
    tall = (100, 200)
    views = {"front": tall, "side": tall, "rear": tall, "right": tall, "top": (100, 100), "hero": tall}
    proj = make(tmp_path / "p", views, {"shots": [FRONT, SIDE, REAR, RIGHT, HERO, TOP]},
                scoring={"preset": "character"})
    assert brief.scored_views(proj) == (["front", "side", "rear", "right", "top"], ["hero"])
    said = brief.reading_facts(proj)
    assert "taken over front, side, rear, right, top ONLY. hero is compared" in said
    assert "which is read through the orthographic cameras" in said and "`z+364:+149/+177`" in said
    assert "the overall extents the outline table opens with" in said
    assert "zones: not measured" in said, "no palette: the zone term is left out and the brief says so"
    assert not any(phrase in said for phrase in CRAFT_ONLY)
    weighted = make(tmp_path / "q", views, {"shots": [FRONT, SIDE, HERO]}, scoring={"views": {"front": 2, "side": 1}})
    assert "taken over front, side (weights: front 2, side 1) ONLY. hero is" in brief.reading_facts(weighted)
    assert "AGGREGATE SCORING VIEW WEIGHTS: front 2, side 1. Hero only" in brief.text(weighted)


def test_a_project_with_no_outline_table_and_one_view_is_told_so(tmp_path):
    proj = make(tmp_path / "p", {"hero": (120, 80)}, {})
    assert proj.reference_profile() == {} and brief.scored_views(proj) == (["hero"], [])
    said = brief.reading_facts(proj)
    assert "taken over hero." in said and "ONLY" not in said, "one perspective view: it is the measure"
    assert "outline error (mm): none." in said and "the overall proportions of the reference views" in said
    assert "against the outline numbers" not in said
    assert brief.reading_facts(make(tmp_path / "q", {}, {})) == "", "nothing is compared: nothing to say"


def test_the_agents_prompt_and_the_skill_leave_the_projects_facts_to_the_brief(tmp_path):
    proj = make(tmp_path / "p", {"front": (100, 200)}, {"shots": [FRONT]})
    prompt = agent.system_prompt(proj, attempts=5, max_turns=40, judge=False)
    loop = SKILL.read_text(encoding="utf-8").split("## Script rules")[0]
    for text in (prompt, loop):
        assert "HOW THIS PROJECT IS MEASURED" in text
        assert not any(phrase in text for phrase in CRAFT_ONLY), [p for p in CRAFT_ONLY if p in text]


def test_the_agents_prompt_gives_the_renderers_conversion_where_a_palette_is_painted_on_corners(tmp_path):
    views, model = {"front": (100, 200)}, {"shots": [FRONT]}
    painted = agent.system_prompt(make(tmp_path / "p", views, model, palette={"red": "#96372d"}), 5, 40, False)
    formula = agent.TO_LINEAR
    assert formula in painted and "the conversion is settled, not something to tune" in painted
    assert "type='FLOAT_COLOR', domain='CORNER'" in painted, "the attribute the linear values go through"
    for c in (0.0, 0.03, 0.04045, 0.2, 0.59, 1.0):       # the formula quoted is the one the renders are drawn with
        assert eval(formula, {"c": c}) == pytest.approx(float(render.srgb_to_linear(np.array(c))))
    # the line that reads a hex colour reads it the same with its '#' and without: two first scripts of the loop
    # sliced one character off
    assert agent.HEX_CHANNELS in painted and "a slice one character off is the usual cause" in painted
    for h in ("#96372d", "96372d"):
        scope = {"h": h}
        exec(agent.HEX_CHANNELS, scope)
        assert [round(scope[k] * 255) for k in "rgb"] == [0x96, 0x37, 0x2d]
    plain = agent.system_prompt(make(tmp_path / "q", views, model), 5, 40, False)
    atlas = agent.system_prompt(make(tmp_path / "r", views, {**model, "textured": True}, palette={"red": "#96372d"}),
                                5, 40, False)
    assert "COLOUR:" not in plain and "COLOUR:" not in atlas, "no palette, or colour from a painted atlas"
    for text in (painted, plain, atlas):
        assert "An evaluation is one of your 5 attempts and is for a whole model meant to beat your best" in text


def test_the_agents_prompt_says_what_a_station_spans_and_to_write_the_script_once(tmp_path):
    prompt = agent.system_prompt(make(tmp_path / "p", {"front": (100, 200)}, {"shots": [FRONT]}), 5, 40, False)
    assert "a station is the whole silhouette at that position" in prompt and "and not the trunk" in prompt
    assert "WRITE THE SCRIPT ONCE" in prompt and "write_file the script for the first attempt only" in prompt
    # Record losses as plateau evidence, under the same preview policy the tools enforce.
    assert "copying the BEST measured attempt and editing one idea" in prompt
    assert "leave `views` out" in prompt and "configured numeric score and its components" in prompt
    assert "Evaluate even a losing preview" in prompt
    assert "at most 2 successful previews per candidate, across restarts" in prompt
    assert "0.02 of overlap" not in prompt and "up to four previews" not in prompt
    assert "Plan briefly" in prompt
    assert "KEEP EDITS SMALL" in prompt and "in named constants or one table near the top of the script" in prompt
    assert "At most four edits a call" in prompt
    assert not any(phrase in prompt for phrase in CRAFT_ONLY)


def test_a_contract_that_checks_no_size_says_what_size_the_cameras_are_set_for(tmp_path):
    tall = (100, 200)
    views = {"front": tall, "side": tall}
    unsized = make(tmp_path / "p", views, {"shots": [FRONT, SIDE], "stage_scale": 0.25})
    assert ("- The contract checks no size. Renders are drawn at stage_scale 0.25, so a model 4 m long or tall is one "
            "unit to the cameras") in brief.text(unsized)
    for n, sized in enumerate(({"height_m": 4.0, "stage_scale": 0.25}, {"axis_extents_m": {"X": 0.9, "Z": 0.85}})):
        said = brief.text(make(tmp_path / f"s{n}", views, {"shots": [FRONT, SIDE], **sized}))
        assert "checks no size" not in said and "{" not in said.split("HOW A MODEL IS BUILT")[1].split("BLENDER")[0]
    craft = make(tmp_path / "c", {"top": (100, 200), "side": (200, 50), "rear": (100, 50)},
                 {"length_m": 0.08, "forward": "-Z"})
    assert "checks no size" not in brief.text(craft)


def test_a_contract_whose_script_paints_its_texture_is_told_the_function_it_owes(tmp_path):
    """`texture_from: "script"`: the brief gives `paint`'s signature, what it is handed and that PALETTE needs no
    conversion, names no atlas plan in its loop, and a project that misstates the contract is told."""
    from mesh_jig import project
    model = {"glb": "cat.glb", "textured": True, "texture_from": "script", "texture_size": 512, "blend": True}
    said = brief.model_facts(dict(model, script="model.py"))
    assert "YOUR SCRIPT PAINTS THE TEXTURE" in said and "def texture(name, position, normal):" in said
    assert "one 512 px atlas" in said and "with NO conversion" in said and "Author no UVs" in said
    assert "write a paint PLAN" not in said and "COLOUR IS VERTEX COLOUR" not in said
    (tmp_path / "refs").mkdir()
    proj = project.Project(tmp_path, {"model": model})
    assert "atlas_plan.json" not in brief.text(proj) and "--paint" not in brief.text(proj)
    plan = project.Project(tmp_path, {"model": {"glb": "cat.glb", "textured": True}})
    assert "write a paint PLAN" in brief.text(plan) and "--paint" in brief.text(plan)

    def problems(**change):
        return project.Project(tmp_path, {"model": {**model, **change}}).problems()
    assert any("texture_from must be one of plan, script" in p for p in problems(texture_from="bake"))
    assert any("texture_from needs `textured`" in p for p in problems(textured=False))
    assert any("texture_size must be a whole number of pixels from 256 to 4096" in p for p in problems(texture_size=100))
    assert any('texture_mode "multiply" needs vertex colours' in p for p in problems(texture_mode="multiply"))
    assert not any("texture" in p for p in problems())
