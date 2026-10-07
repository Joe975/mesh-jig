"""jig.json's `criteria` block: what a reference is judged on (scored views, a calibrated outline table, gates, review)
stated in one place beside it. It reads as the older fields did, says where each criterion is stated, and a project
that states no views is told it is on the fallback."""
from __future__ import annotations

import json

import pytest

from mesh_jig import brief, cli, criteria, evaluate, project, render, sheet
from test_character import named_cubes, shots
from test_evaluate import fake_build, make_project, run

GEOMETRY = [("Skin", (.4, 1, .2), (0, .5, 0)), ("Patch", (.06, .06, .02), (-.12, .8, -.12))]
VIEWS = ("front", "side", "rear", "right", "top")


def character(root) -> dict:
    """A standing character written the older way: every criterion where it was first put. Its references are
    renders of the geometry every attempt builds, so it measures as a match."""
    (root / "refs").mkdir(parents=True)
    mesh = render.load(named_cubes(root / "truth.glb", GEOMETRY))
    for shot in shots():
        render.render(mesh, shot, width=320, height=180).save(root / "refs" / f"{shot['name']}.png")
    table = {"mode": "standing", "calibration": {"view": "front", "height_px": 180 / 1.4,
             "origins_px": {v: [160, 90 + .5 * 180 / 1.4] for v in ("front", "side", "rear", "right")}},
             "stations": [{"name": "belly", "z_m": .5}, {"name": "head", "z_m": .8}]}
    return {"model": {"glb": "character.glb", "height_m": 1, "shots": shots(), "profile": table,
                      "profile_mae_max": 0.001, "resemblance_min": 0.5},
            "scoring": {"preset": "character"},
            "review": {"parts": {"patch": {"nodes": ["Patch"]}},
                       "landmarks": [{"part": "patch", "bounds_m": [[-.2, .1, .7], [-.05, .2, .9]]}]}}


def measured(root, data: dict, name: str) -> dict:
    def build_fn(script, out, model):
        out.mkdir(parents=True, exist_ok=True)
        named_cubes(out / model["glb"], GEOMETRY)
        return ""

    def render_fn(model, out, dest):
        return render.render_views(model, out, dest, size=(320, 180))

    proj = project.Project(root, data)
    assert proj.problems() == []
    rec = evaluate.evaluate(proj, root / "model.py", root / "attempts" / name, build_fn=build_fn, render_fn=render_fn,
                            log=lambda *_: None)
    assert rec["ok"], rec
    return {k: rec[k] for k in ("numeric", "gates", "profile_diff", "scoring", "review")}


def test_the_block_holds_every_criterion_and_measures_as_the_older_fields_did(tmp_path):
    older = character(tmp_path)
    stated = criteria.rewrite(older)
    assert list(stated["criteria"]) == ["views", "outline", "gates", "review"]
    assert stated["criteria"]["views"] == dict.fromkeys(VIEWS, 1), "a preset is written out as the views it selects"
    assert stated["criteria"]["gates"] == {"outline_mm_max": 0.001, "resemblance_min": 0.5}
    assert stated["criteria"]["outline"] == older["model"]["profile"] and stated["criteria"]["review"] == older["review"]
    assert not {"scoring", "review"} & set(stated)
    assert not {"profile", "profile_mae_max", "resemblance_min"} & set(stated["model"]), "none left where it was"
    assert stated["model"]["height_m"] == 1, "a size is the build contract's: it stays in model"
    assert criteria.rewrite(stated) == stated

    a, b = measured(tmp_path, older, "older"), measured(tmp_path, stated, "stated")
    assert a == b
    assert a["scoring"]["weights"] == dict.fromkeys(VIEWS, 1.0) and a["profile_diff"]["mode"] == "standing"
    names = {g["name"]: g["passed"] for g in a["gates"]}
    assert names["resemblance_min"] and not names["profile_mae_max"], "a gate stated in the block gates"
    assert names["landmark:patch"] and names["standing_stations"]

    proj = project.Project(tmp_path, stated)
    assert criteria.sources(stated) == {k: f"criteria.{k}" for k in ("views", "outline", "gates", "review")}
    assert criteria.sources(older) == {"views": "scoring", "outline": "model.profile", "gates": "model", "review": "review"}
    said = brief.criteria_facts(proj)
    assert "- views: front x1, side x1, rear x1, right x1, top x1 [criteria.views]" in said
    assert "- outline: the calibrated standing table, at 2 named heights (belly, head) [criteria.outline]" in said
    assert "- gates: outline error at most 0.001 mm, resemblance at least 0.5 [criteria.gates]" in said
    assert "- review: 1 parts, 1 landmarks [criteria.review]" in said and "- zones: not measured [no palette]" in said
    assert "[scoring]" in brief.criteria_facts(project.Project(tmp_path, older))
    reading = brief.reading_facts(proj)
    assert "gates this reference sets" in reading and "outline error at most 0.001 mm, resemblance at least 0.5." in reading
    assert "fallback" not in reading


@pytest.mark.parametrize("change, problem", [
    ({"criteria": []}, "criteria must be an object with any of views, outline, gates, review"),
    ({"criteria": {"weights": {"front": 1}}}, "criteria: unknown key(s) ['weights']"),
    ({"criteria": {"views": {}}}, "criteria.views must be a nonempty mapping of view names to positive weights"),
    ({"criteria": {"views": {"front": 0}}}, "criteria.views must be a nonempty mapping"),
    ({"criteria": {"views": {"front": 1, "hero": 1}}}, "criteria.views: selected views lack a reference/camera: ['hero']"),
    ({"criteria": {"gates": {"score_min": 0.5}}}, "criteria.gates: unknown gate 'score_min'; it takes outline_mm_max"),
    ({"criteria": {"gates": {"zones_min": "half"}}}, "criteria.gates.zones_min must be a number"),
    ({"criteria": {"gates": []}}, "criteria.gates must be an object"),
    ({"criteria": {"outline": {"mode": "upright"}}}, "criteria.outline.mode must be standing"),
    ({"criteria": {"review": {"regions": [None]}}}, "criteria.review.regions must be a list of objects"),
    ({"criteria": {"views": {"front": 1}}, "scoring": {"preset": "character"}},
     "criteria.views is also stated as scoring: state it once, in criteria"),
    ({"criteria": {"review": {}}, "review": {}}, "criteria.review is also stated as review"),
    ({"criteria": {"gates": {"zones_min": 0.5}}, "model": {"placement_min": 0.4}},
     "criteria.gates.zones_min is also stated as model.placement_min"),
    ({"criteria": {"outline": {"mode": "standing"}}, "model": {"profile": {"mode": "standing"}}},
     "criteria.outline is also stated as model.profile"),
])
def test_a_block_that_is_wrong_is_a_problem_named_where_it_is_stated(tmp_path, change, problem):
    data = character(tmp_path)
    plain = {"model": {k: data["model"][k] for k in ("glb", "height_m", "shots")}}
    plain["model"].update(change.pop("model", {}))
    problems = project.Project(tmp_path, {**plain, **change}).problems()
    assert any(p.startswith(problem) for p in problems), problems


def test_the_older_fields_keep_their_own_messages(tmp_path):
    data = character(tmp_path)
    data["model"]["profile"] = {"mode": "upright"}
    data["scoring"] = {"views": {"hero": 1}}
    data["review"] = {"regions": [None]}
    assert project.Project(tmp_path, data).problems() == [
        "scoring: selected views lack a reference/camera: ['hero']", "profile.mode must be standing",
        "review.regions must be a list of objects"]


def test_a_project_that_states_no_views_is_told_it_is_on_the_fallback_and_how_to_state_them(tmp_path, capsys):
    proj = make_project(tmp_path, profile_mae_max=5.0)
    jig = proj.root / "jig.json"
    written = jig.read_text()
    assert "taken over top, side, rear ONLY. hero is compared for you to look at and does not count. These are the " \
           "fallback views: jig.json states no `criteria.views`." in brief.reading_facts(proj)
    said = brief.criteria_facts(proj)
    assert "- views: top, side, rear [NOT STATED: the fallback, top, side and rear when the project has them, else " \
           "every compared view]\n  compared but not scored: hero" in said
    assert "- outline: the craft table" in said and "- gates: outline error at most 5.0 mm [model]" in said
    assert "- zones: measured against the palette's 2 colour(s) [palette]" in said and "- review: none stated" in said
    before = run(proj, tmp_path, "a01", build_fn=fake_build(span=0.05))
    assert "do not count (the fallback views: jig.json states no `criteria.views`)" in evaluate.feedback(before) or \
           "does not count (the fallback views: jig.json states no `criteria.views`)" in evaluate.feedback(before)

    cli.main(["doctor", str(proj.root)])
    out = capsys.readouterr().out
    assert "CRITERIA (what this reference is judged on" in out
    assert f"To state them: `mesh-jig criteria {proj.root} --write`" in out

    assert cli.main(["criteria", str(proj.root)]) == 0
    out = capsys.readouterr().out
    assert "NOT STATED: the fallback" in out and "As a `criteria` block" in out
    assert json.loads(out[out.index("{"):]) == {"criteria": {"views": {"top": 1, "side": 1, "rear": 1},
                                                             "gates": {"outline_mm_max": 5.0}}}
    assert jig.read_text() == written, "without --write the file is not touched"

    assert cli.main(["criteria", str(proj.root), "--write"]) == 0
    capsys.readouterr()
    data = json.loads(jig.read_text())
    assert data["criteria"] == {"views": {"top": 1, "side": 1, "rear": 1}, "gates": {"outline_mm_max": 5.0}}
    assert "profile_mae_max" not in data["model"] and data["palette"] and data["model"]["length_m"]
    stated = project.load(proj.root)
    assert stated.problems() == [] and "fallback" not in brief.reading_facts(stated)
    after = run(stated, tmp_path, "a02", build_fn=fake_build(span=0.05))
    assert after["numeric"] == before["numeric"] and after["gates"] == before["gates"], "stating the fallback moves nothing"
    assert "fallback" not in evaluate.feedback(after)
    assert cli.main(["criteria", str(proj.root)]) == 0
    assert "jig.json states all of it in its `criteria` block." in capsys.readouterr().out
    cli.main(["doctor", str(proj.root)])
    assert "To state them" not in capsys.readouterr().out


def test_the_criteria_command_refuses_a_project_with_problems(tmp_path, capsys):
    proj = make_project(tmp_path)
    data = json.loads((proj.root / "jig.json").read_text())
    (proj.root / "jig.json").write_text(json.dumps({**data, "criteria": {"views": {"belly": 1}}}))
    assert cli.main(["criteria", str(proj.root), "--write"]) == 2
    assert "criteria.views: selected views lack a reference/camera: ['belly']" in capsys.readouterr().err
    assert json.loads((proj.root / "jig.json").read_text())["criteria"] == {"views": {"belly": 1}}


def test_init_states_the_views_its_craft_is_scored_on(tmp_path):
    sheet.init(tmp_path / "rover", length_m=0.5)
    proj = project.load(tmp_path / "rover")
    assert proj.data["criteria"] == {"views": {"top": 1, "side": 1, "rear": 1}}
    assert proj.scoring_weights == {"top": 1.0, "side": 1.0, "rear": 1.0}


def test_the_light_a_build_is_drawn_under_is_a_criterion_of_the_reference(tmp_path):
    """`criteria.light` is read where every tool reads it (`model.light`), is named in the brief, and a project that
    states none is drawn under the stage lights as it always was."""
    data = criteria.rewrite(character(tmp_path))
    plain = project.Project(tmp_path, data)
    assert plain.spec.get("light") is None and "- light:" not in brief.reading_facts(plain)
    lit = project.Project(tmp_path, dict(data, criteria=dict(data["criteria"], light="camera")))
    assert lit.problems() == [] and lit.spec["light"] == "camera"
    assert "lights that turn with its camera" in brief.reading_facts(lit)
    assert "- light: camera [criteria.light]" in brief.criteria_facts(lit)
    assert criteria.block(lit.data)["light"] == "camera"
    again = criteria.rewrite(lit.stated)
    assert again["criteria"]["light"] == "camera" and "light" not in again["model"], "rewritten into the block"
    wrong = project.Project(tmp_path, dict(data, criteria=dict(data["criteria"], light="noon")))
    assert any("criteria.light must be one of stage, camera" in p for p in wrong.problems())
    twice = dict(data, model=dict(data["model"], light="stage"), criteria=dict(data["criteria"], light="camera"))
    assert any("criteria.light is also stated as model.light" in p for p in project.Project(tmp_path, twice).problems())
    assert any("model.light must be one of" in p
               for p in project.Project(tmp_path, dict(data, model=dict(data["model"], light="noon"))).problems())


def test_a_project_states_what_its_score_is_made_of_and_one_that_states_none_is_scored_as_before(tmp_path):
    region = {"name": "patch", "view": "front", "box": [.3, .1, .7, .4]}
    data = character(tmp_path / "proj")
    data["review"]["regions"] = [region]
    (tmp_path / "proj" / "jig.json").write_text(json.dumps(data), encoding="utf-8")
    plain = project.load(tmp_path / "proj")
    assert plain.problems() == [] and plain.score_mix is None
    assert "- score: overlap and resemblance in equal parts [NOT STATED" in brief.criteria_facts(plain)
    said = brief.reading_facts(plain)
    assert "numeric score: the mean of overlap and resemblance" in said and "before touching colour" in said
    assert "- regions: the 1 marked region (patch), each read on its own" in said

    data["criteria"] = {"score": {"overlap": 1, "colour": 1, "edges": 1, "regions": 1, "zones": 0}}
    (tmp_path / "proj" / "jig.json").write_text(json.dumps(data), encoding="utf-8")
    proj = project.load(tmp_path / "proj")
    assert proj.problems() == []
    assert proj.score_mix == {"overlap": 1.0, "colour": 1.0, "edges": 1.0, "regions": 1.0}, "a weight of 0 is left out"
    assert "- score: the mean of overlap, colour, edges and regions [criteria.score]" in brief.criteria_facts(proj)
    assert criteria.block(proj.data)["score"] == {"overlap": 1, "colour": 1, "edges": 1, "regions": 1}
    assert criteria.rewrite(proj.data)["criteria"]["score"] == criteria.block(proj.data)["score"]
    said = brief.reading_facts(proj)
    assert ("- numeric score: the mean of overlap, colour, edges and regions; rank attempts on it. Overlap is 25% of "
            "it and the marked regions 25%.") in said
    assert "it is 25% of this score" in said and "before touching colour" not in said
    assert criteria.score_text({"overlap": 2, "regions": 1}) == "the weighted mean of overlap x2 and regions x1"

    old = measured(tmp_path / "proj", {k: v for k, v in data.items() if k != "criteria"}, "old")["numeric"]
    new = measured(tmp_path / "proj", data, "new")["numeric"]
    assert old["numeric_score"] == round((old["ortho_iou"] + old["resemblance"]) / 2, 4) and "mix" not in old
    assert {k: new[k] for k in ("ortho_iou", "resemblance", "zone", "colour", "edges", "regions")} == {
        k: old.get(k) for k in ("ortho_iou", "resemblance", "zone", "colour", "edges", "regions")}, "one set of measures"
    assert new["mix"] == proj.score_mix
    assert new["numeric_score"] == round((new["ortho_iou"] + new["colour"] + new["edges"] + new["regions"]) / 4, 4)


@pytest.mark.parametrize("score, problem", [
    ({"overlap": 1, "likeness": 1}, "criteria.score must be a mapping of any of overlap, zones, colour, edges, regions"),
    ({"overlap": -1}, "criteria.score must be a mapping"),
    ({"overlap": 0}, "criteria.score must be a mapping"),
    ("overlap", "criteria.score must be a mapping"),
    ({"overlap": 1, "regions": 1}, "criteria.score counts `regions` and the project marks none"),
    ({"overlap": 1, "zones": 1}, "criteria.score counts `zones` and the project has no palette"),
])
def test_a_score_that_cannot_be_taken_is_a_project_problem(tmp_path, score, problem):
    data = character(tmp_path / "proj")
    data["criteria"] = {"score": score}
    (tmp_path / "proj" / "jig.json").write_text(json.dumps(data), encoding="utf-8")
    proj = project.load(tmp_path / "proj")
    assert [p for p in proj.problems() if p.startswith(problem)], proj.problems()
    # and a `score` written outside the block is not a way to state one
    loose = {k: v for k, v in data.items() if k != "criteria"} | {"score": {"overlap": 1}}
    assert criteria.fold(loose)[0].get("score") is None
