"""readiness: what a built GLB costs and what a rig or a texture would find in it, on shapes with known answers;
target: the budgets a build is held to and read against, and how a project and a command choose one."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mesh_jig import brief, cli, evaluate, glb, project, readiness, target
from glbkit import box, textured_glb, write_glb
from test_evaluate import fake_build, make_project, quick_render, script


def sphere(radius=0.5, centre=(0.0, 0.0, 0.0), rings=16, around=24):
    """A closed lat-long sphere with shared vertices, wound outward, mirrored across X = 0 when centred there."""
    verts = [(0.0, radius, 0.0)]
    for i in range(1, rings):
        polar = np.pi * i / rings
        for j in range(around):
            turn = 2 * np.pi * j / around
            verts.append((radius * np.sin(polar) * np.sin(turn), radius * np.cos(polar), radius * np.sin(polar) * np.cos(turn)))
    verts.append((0.0, -radius, 0.0))
    faces, last = [], len(verts) - 1
    ring = lambda i, j: 1 + (i - 1) * around + j % around      # noqa: E731
    for j in range(around):
        faces.append((0, ring(1, j), ring(1, j + 1)))
        faces.append((last, ring(rings - 1, j + 1), ring(rings - 1, j)))
        for i in range(1, rings - 1):
            faces += [(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)), (ring(i, j), ring(i + 1, j + 1), ring(i, j + 1))]
    return (np.array(verts, np.float32) + np.array(centre, np.float32)), np.array(faces, np.uint16)


def reading(tmp_path, parts, **kw):
    return readiness.measure(write_glb(tmp_path / "m.glb", [(f"p{i}", mesh, (0, 0, 0)) for i, mesh in enumerate(parts)]), **kw)


def test_one_closed_sphere_is_one_piece_with_nothing_buried_open_or_lopsided(tmp_path):
    r = reading(tmp_path, [sphere()])
    assert r["cost"] == {"triangles": 720, "vertices": 362, "positions": 362, "mesh_nodes": 1, "primitives": 1,
                         "materials": 0, "images": 0, "bytes": (tmp_path / "m.glb").stat().st_size}
    assert r["surface"] == {"pieces": 1, "pieces_90": 1, "largest_piece": 1.0, "open_edges": 0.0, "slivers": 0.0,
                            "buried": 0.0, "symmetry": 1.0}
    assert r["uv"] is None and r["rig"] == {"skins": 0, "animations": 0}
    assert "checks" not in r and "target" not in r                  # no target, no verdict
    assert readiness.measure(tmp_path / "m.glb") == r               # the same bytes, the same reading


def test_a_part_inside_another_is_buried_and_two_that_overlap_bury_what_is_inside_each(tmp_path):
    big, small = sphere(0.5), sphere(0.2, rings=8, around=12)
    inside = reading(tmp_path, [big, small])["surface"]
    assert inside["buried"] == pytest.approx(len(small[1]) / (len(big[1]) + len(small[1])), abs=1e-4)
    assert inside["pieces"] == 2 and inside["pieces_90"] == 2          # the small one is a seventh of the area
    # centres about one radius apart: each sphere has a cap inside the other. A triangle is buried when all of it is
    left, right = sphere(0.5, (-0.26, 0, 0)), sphere(0.5, (0.26, 0, 0))
    halves = reading(tmp_path, [left, right])["surface"]
    whole = sum(int((np.linalg.norm(v[f] - other, axis=2) < 0.5).all(axis=1).sum())
                for (v, f), other in ((left, (0.26, 0, 0)), (right, (-0.26, 0, 0))))
    assert halves["buried"] == pytest.approx(whole / 1440, abs=0.02) and 0.1 < halves["buried"] < 0.2
    assert halves["pieces"] == 2 and halves["pieces_90"] == 2 and halves["largest_piece"] == 0.5
    # apart, nothing is hidden: a hollow a direction looks into is not buried
    assert reading(tmp_path, [sphere(0.5, (-0.6, 0, 0)), sphere(0.5, (0.6, 0, 0))])["surface"]["buried"] == 0.0


def test_holes_fins_slivers_and_lost_symmetry_are_each_read(tmp_path):
    verts, faces = sphere()
    holed = reading(tmp_path, [(verts, faces[:-48])])["surface"]            # a cap of faces taken off
    assert 0 < holed["open_edges"] < 0.1
    fin = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1]], np.float32)
    three = reading(tmp_path, [(fin, np.array([(0, 1, 2), (0, 1, 3), (0, 1, 4)], np.uint16))])["surface"]
    assert three["open_edges"] == 1.0                                       # six open edges and one shared by three
    needle = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.01, 0], [0.5, 1, 0]], np.float32)
    thin = reading(tmp_path, [(needle, np.array([(0, 1, 2), (0, 1, 3)], np.uint16))])["surface"]
    assert thin["slivers"] == 0.5
    assert reading(tmp_path, [sphere(0.5, (0.123, 0, 0))])["surface"]["symmetry"] < 0.1
    pair = reading(tmp_path, [sphere(0.3, (-0.4, 0, 0)), sphere(0.3, (0.4, 0, 0))])["surface"]
    assert pair["symmetry"] == 1.0


def test_pieces_are_counted_by_what_holds_the_surface(tmp_path):
    bits = [box((0.02, 0.02, 0.02), (0.8 + 0.05 * i, 0, 0)) for i in range(10)]
    s = reading(tmp_path, [sphere()] + bits)["surface"]
    assert s["pieces"] == 11 and s["pieces_90"] == 1 and s["largest_piece"] > 0.95
    nodes = readiness.measure(tmp_path / "m.glb")["nodes"]
    assert nodes[0] == {"name": "p0", "triangles": 720, "primitives": 1} and len(nodes) == 11


def test_uvs_are_read_when_a_primitive_has_them(tmp_path):
    flat = readiness.measure(textured_glb(tmp_path / "t.glb"))["uv"]         # every UV at the origin
    assert flat == {"with_uv": 1.0, "outside": 0.0, "collapsed": 1.0, "texel_spread": 0.0, "used": 0.0}
    tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [2, 0, 0], [0, 2, 0]]], float)
    uvs = np.array([[[0, 0], [0.5, 0], [0, 0.5]], [[0, 0], [1.5, 0], [0, 0.5]]])
    got = readiness.uv_reading(tris, uvs)
    assert got["with_uv"] == 1.0 and got["outside"] == 0.5 and got["collapsed"] == 0.0
    assert got["used"] == pytest.approx(0.125 + 0.375) and got["texel_spread"] > 1
    assert readiness.uv_reading(tris, np.full((2, 3, 2), np.nan)) is None


def test_a_target_holds_each_reading_to_its_limit_and_counts_what_holds(tmp_path):
    quest = target.resolve({}, "quest")
    assert (quest["name"], quest["max_triangles"], quest["max_materials"], quest["texture_px"]) == ("quest", 10000, 1, 1024)
    pcvr = target.resolve({}, "pcvr")
    assert (pcvr["max_triangles"], pcvr["max_materials"], pcvr["texture_px"]) == (32000, 2, 2048)
    assert pcvr["max_buried"] > quest["max_buried"]
    good = reading(tmp_path, [sphere()], target=quest)
    assert good["target"] == "quest" and good["passed"] == len(good["checks"]) == 7
    assert [c["name"] for c in good["checks"]] == ["triangles", "buried", "few pieces", "closed", "slivers", "symmetry",
                                                   "materials"]
    assert [c["hard"] for c in good["checks"]] == [True] + [False] * 6
    tight = {**quest, "max_triangles": 700, "max_pieces": 1}
    bad = reading(tmp_path, [sphere(0.5, (0.123, 0, 0)), sphere(0.2, (0.123, 0, 0), rings=8, around=12)] +
                  [box((0.3, 0.3, 0.3), (2 + i, 0, 0)) for i in range(4)], target=tight)
    failed = {c["name"]: c for c in bad["checks"] if not c["passed"]}
    assert set(failed) == {"triangles", "buried", "few pieces", "symmetry"} and bad["passed"] == 3
    assert failed["triangles"]["value"] == 720 + 168 + 48 and failed["triangles"]["limit"] == "at most 700"
    text = "\n".join(readiness.lines(bad))
    assert text.startswith("MESH READINESS for quest: 3 of 7 hold; NOT: triangles (936, wants at most 700")
    assert "buried" in text and "mesh nodes" in text and "uv: none" in text
    assert readiness.lines(good)[0] == "MESH READINESS for quest: 7 of 7 hold"
    assert "no target is set" in readiness.lines(reading(tmp_path, [sphere()]))[0]
    textured = readiness.measure(textured_glb(tmp_path / "t.glb"), {"textured": True}, quest)
    assert [c for c in textured["checks"] if c["name"] == "uv"][0]["passed"] is False


def test_a_project_states_its_target_and_its_own_limits_and_a_command_can_choose_another(tmp_path):
    proj = make_project(tmp_path, max_triangles=65000, min_triangles=100)
    assert proj.target is None and "target" not in proj.spec and proj.spec["max_triangles"] == 65000
    assert "TARGET" not in brief.text(proj) and proj.problems() == []
    root = proj.root
    quest = project.load(root, "quest")
    assert quest.spec["max_triangles"] == 10000 and quest.spec["target"]["source"] == "the package's"
    told = brief.text(quest)
    assert "TARGET: quest" in told and "At most 10000 triangles per GLB" in told and "buried triangles at most 15%" in told
    assert "Triangles per GLB: 100 to 10000" in told
    # the contract's own check now holds a build to the target's ceiling
    stats = {**glb.stats(write_glb(tmp_path / "b.glb", [("a", sphere(), (0, 0, 0))])), "triangles": 12000}
    assert any("12000 triangles > budget 10000" in i for i in glb.check(stats, quest.spec))
    assert not any("budget" in i for i in glb.check(stats, project.load(root, "pcvr").spec))

    data = json.loads((root / "jig.json").read_text())
    data.update({"target": "prop", "targets": {"prop": {"max_triangles": 1500, "for": "a held prop"},
                                               "quest": {"max_buried": 0.3}}})
    (root / "jig.json").write_text(json.dumps(data))
    own = project.load(root)
    assert own.target["name"] == "prop" and own.spec["max_triangles"] == 1500 and own.target["source"] == "the project's"
    assert own.spec.get("min_triangles") == 100 and "a held prop" in brief.text(own)
    changed = project.load(root, "quest").target
    assert changed["max_buried"] == 0.3 and changed["max_triangles"] == 10000
    assert changed["source"] == "the package's, with the project's own max_buried"
    assert target.names(data) == ["pcvr", "prop", "quest"]
    with pytest.raises(project.ProjectError, match="not a target of this project: pcvr, prop, quest"):
        project.load(root, "console")

    data.update({"target": "nowhere", "targets": {"prop": {"max_buried": 2, "colour": 1}, "quest": {"max_triangles": 0}}})
    (root / "jig.json").write_text(json.dumps(data))
    wrong = project.load(root)
    assert wrong.target is None and wrong.spec["max_triangles"] == 65000        # nothing is applied from a wrong block
    said = "\n".join(wrong.problems())
    for part in ("targets.prop.max_buried must be a share from 0 to 1", "targets.prop.colour is not a limit",
                 "targets.prop needs max_triangles", "targets.quest.max_triangles must be a whole number above 0",
                 "target 'nowhere' is not a target"):
        assert part in said
    floor = target.apply({"min_triangles": 5000, "max_triangles": 65000}, {"name": "t", "max_triangles": 1500})
    assert (floor["min_triangles"], floor["max_triangles"]) == (0, 1500)


def test_an_evaluation_records_the_reading_and_says_it_and_the_command_reads_a_built_attempt(tmp_path, capsys):
    proj = make_project(tmp_path)
    s = script(tmp_path)
    rec = evaluate.evaluate(proj, s, s.parent, log=lambda *_: None, render_fn=quick_render, build_fn=fake_build())
    assert rec["ok"] and "checks" not in rec["readiness"]
    assert rec["readiness"]["cost"]["triangles"] == glb.stats(s.parent / "out" / "ship.glb")["triangles"]
    assert json.loads((s.parent / "eval.json").read_text())["readiness"] == rec["readiness"]
    assert "MESH READINGS (no target is set" in evaluate.feedback(rec)

    aimed = project.load(proj.root, "quest")
    s2 = script(tmp_path, "a02")
    rec = evaluate.evaluate(aimed, s2, s2.parent, log=lambda *_: None, render_fn=quick_render, build_fn=fake_build())
    assert rec["readiness"]["target"] == "quest" and rec["readiness"]["checks"][0]["name"] == "triangles"
    assert "MESH READINESS for quest:" in evaluate.feedback(rec)
    assert "MESH READ" not in evaluate.feedback({k: v for k, v in rec.items() if k != "readiness"})

    code = cli.main(["ready", str(proj.root), "a02", "--target", "quest"])
    out = capsys.readouterr().out
    assert out.startswith("MESH READINESS for quest:") and code == (0 if "NOT:" not in out else 1)
    assert cli.main(["ready", str(proj.root), "a02", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cost"] == rec["readiness"]["cost"]
    assert cli.main(["ready", str(proj.root), "a09"]) == 2
    assert cli.main(["brief", str(proj.root), "--target", "console"]) == 2


def test_surface_symmetry_is_an_opt_in_glb_reading_and_ready_cli_preserves_default_schema(tmp_path, capsys):
    proj = make_project(tmp_path)
    out = proj.attempts_dir / "a01" / "out"
    out.mkdir(parents=True)
    left = box((0.2, 0.2, 0.2))
    file = write_glb(out / "ship.glb", [("left", left, (-0.3, 0, 0)), ("right", left, (0.3, 0, 0))])

    default = readiness.measure(file, proj.spec)
    opted_in = readiness.measure(file, proj.spec, surface_symmetry=True)
    assert set(opted_in) - set(default) == {"surface_symmetry"}
    assert {k: v for k, v in opted_in.items() if k != "surface_symmetry"} == default
    surface = opted_in["surface_symmetry"]
    assert surface["status"] == "available" and surface["agreement"] == 1.0
    assert surface["left_to_right"] == surface["right_to_left"] == 1.0
    assert surface["sample_count"] == 4096 and surface["tolerance_mm"] == 1.0
    assert readiness.summary(default) == ""  # existing default feedback remains byte-for-byte unchanged
    assert readiness.summary(opted_in) == ("mesh: sampled mirrored-surface agreement 100% "
                                           "(approximate; 4096 samples at 1 mm; vertex correspondence is separate)")
    said = "\n".join(readiness.lines(opted_in))
    assert "mirrored surface across X=0" in said and "approximate" in said
    assert "retessellation" in said and "no construction technique is required" in said

    assert cli.main(["ready", str(proj.root), "a01", "--json"]) == 0
    default_cli = json.loads(capsys.readouterr().out)
    assert default_cli == default and "surface_symmetry" not in default_cli
    assert cli.main(["ready", str(proj.root), "a01", "--json", "--surface-symmetry"]) == 0
    assert json.loads(capsys.readouterr().out) == opted_in


def test_surface_symmetry_unavailable_reason_reaches_ready_cli(tmp_path, capsys):
    proj = make_project(tmp_path)
    out = proj.attempts_dir / "a01" / "out"
    out.mkdir(parents=True)
    one_side = box((0.2, 0.2, 0.2))
    file = write_glb(out / "ship.glb", [("right", one_side, (0.3, 0, 0))])
    result = readiness.measure(file, proj.spec, surface_symmetry=True)
    assert result["surface_symmetry"]["status"] == "unavailable"
    assert result["surface_symmetry"]["reason"] == "missing_left_area"
    assert "unavailable (missing_left_area)" in "\n".join(readiness.lines(result))

    assert cli.main(["ready", str(proj.root), "a01", "--json", "--surface-symmetry"]) == 0
    assert json.loads(capsys.readouterr().out)["surface_symmetry"] == result["surface_symmetry"]
    assert cli.main(["ready", str(proj.root), "a01", "--surface-symmetry"]) == 0
    assert "unavailable (missing_left_area)" in capsys.readouterr().out


def test_eval_short_surfaces_opt_in_surface_feedback_but_keeps_default_absent(tmp_path, monkeypatch, capsys):
    proj = project.load(make_project(tmp_path).root, "quest")
    real_evaluate = evaluate.evaluate

    def injected_evaluate(*args, **kwargs):
        kwargs.setdefault("build_fn", fake_build())
        kwargs.setdefault("render_fn", quick_render)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(evaluate, "evaluate", injected_evaluate)
    measure = readiness.measure

    def with_surface_symmetry(*args, **kwargs):
        kwargs["surface_symmetry"] = True
        return measure(*args, **kwargs)

    trial_script = script(tmp_path, "trial-short")
    with monkeypatch.context() as patch:
        patch.setattr(readiness, "measure", with_surface_symmetry)
        assert cli.main(["eval", str(proj.root), "--script", str(trial_script), "--short", "--no-ledger"]) == 0
    trial_output = capsys.readouterr().out
    assert "sampled mirrored-surface agreement 100%" in trial_output
    assert "4096 samples at 1 mm" in trial_output and "vertex correspondence is separate" in trial_output
    full_trial = (trial_script.parent / cli.FEEDBACK_FILE).read_text(encoding="utf-8")
    assert "not visible-surface or union coverage" in full_trial

    def without_surface_symmetry(*args, **kwargs):
        kwargs["surface_symmetry"] = False
        return measure(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(readiness, "measure", without_surface_symmetry)
        default_script = script(tmp_path, "default-short")
        assert cli.main(["eval", str(proj.root), "--script", str(default_script), "--short", "--no-ledger"]) == 0
        default_output = capsys.readouterr().out
    assert "sampled mirrored-surface agreement" not in default_output


def figure(tmp_path, arm_x):
    """A body on two legs with an arm each side, in glTF axes (+Y up), standing on y = 0: legs 0.4 m, the body
    above them to 1.0 m, arms 0.4 m long hanging from 0.9 m with their middles `arm_x` out."""
    parts = [box((0.4, 0.6, 0.2), (0, 0.7, 0)), box((0.1, 0.4, 0.1), (-0.1, 0.2, 0)), box((0.1, 0.4, 0.1), (0.1, 0.2, 0)),
             box((0.08, 0.4, 0.08), (-arm_x, 0.7, 0)), box((0.08, 0.4, 0.08), (arm_x, 0.7, 0))]
    return reading(tmp_path, parts)


def test_limbs_say_how_long_arms_and_legs_hang_clear_of_the_body(tmp_path):
    r = figure(tmp_path, arm_x=0.3)                 # 6 cm between arm and flank
    assert r["height_m"] == pytest.approx(1.0)
    assert r["limbs"] == [{"parts": 3, "clear_m": pytest.approx(0.4), "joined_m": 0.0, "from_m": pytest.approx(0.9), "on_ground": False},
                          {"parts": 2, "clear_m": pytest.approx(0.4), "joined_m": 0.0, "from_m": pytest.approx(0.4), "on_ground": True}]
    said = readiness.lines(r)[-1]
    assert said.startswith("  limbs (a rig bends a limb only where it hangs clear of the body")
    assert ("arms (or a tail): 3 parts side by side, the body among them, from 0.90 m down to 0.50 m above the ground "
            "(0.40 m, 40% of the height)") in said
    assert "legs: 2 stand apart from 0.40 m down to 0.00 m above the ground (0.40 m, 40% of the height) and reach the ground" in said
    assert "no arm hangs clear" not in said
    assert readiness.arms_clear(r) == pytest.approx(0.4)
    assert readiness.summary(r) == ""
    assert readiness.summary(dict(r, max_triangles=100)) == f"mesh: {r['cost']['triangles']} of 100 triangles ({r['cost']['triangles'] / 100:.0%})"


def test_limbs_say_when_arms_are_joined_to_the_body_all_the_way_down(tmp_path):
    r = figure(tmp_path, arm_x=0.22)                # the arms sink 2 cm into the flank
    assert [limb["parts"] for limb in r["limbs"]] == [2]
    said = readiness.lines(r)[-1]
    assert "legs: 2 stand apart from 0.40 m down to 0.00 m above the ground" in said
    assert ("no arm hangs clear of the body: an arm, if there is one, is joined to it or nearer it than 3 cm all the way "
            "down. Leave a gap") in said
    assert readiness.arms_clear(r) == 0
    assert readiness.summary(r) == "mesh: no arm hangs clear of the body (see limbs below)"


def test_limbs_say_when_legs_part_but_join_again_at_the_feet():
    # a stretch of two parts that ends 0.1 m above the ground of a 1 m model, and nothing reaching the ground apart
    reading = {"cost": {"triangles": 10}, "height_m": 1.0,
               "limbs": [{"parts": 2, "clear_m": 0.3, "joined_m": 0.0, "from_m": 0.4, "on_ground": False}]}
    said = readiness.lines(dict(reading, surface={}, nodes=[], uv=None) | {"surface": {
        "pieces": 1, "pieces_90": 1, "largest_piece": 1.0, "buried": 0, "open_edges": 0, "slivers": 0, "symmetry": 1},
        "cost": {"triangles": 10, "vertices": 0, "positions": 0, "mesh_nodes": 1, "primitives": 1, "materials": 1,
                 "bytes": 0}})[-1]
    assert "legs: 2 parts stand apart from 0.40 m down to 0.10 m above the ground" in said and "feet fused" in said
    assert readiness.summary(reading) == ("mesh: no arm hangs clear of the body (see limbs below); the legs do not "
                                          "stand apart down to the ground (see limbs below)")
    assert readiness.arms_clear({"limbs": []}) is None


def test_limbs_read_an_arm_sloping_out_from_the_shoulder_as_joined_above_where_it_hangs_clear():
    def slab(x0, x1, y0, y1):          # a thin flat-faced box in glTF axes, as its 12 triangles
        v, f = box((x1 - x0, y1 - y0, 0.1), ((x0 + x1) / 2, (y0 + y1) / 2, 0))
        return v[f.astype(int)]
    # a trunk, legs, and below the shoulder a single wide run (arms joined to the flank) before they part
    tris = np.concatenate([slab(-0.2, 0.2, 0.4, 1.0), slab(-0.15, -0.05, 0.0, 0.4), slab(0.05, 0.15, 0.0, 0.4),
                           slab(-0.35, 0.35, 0.6, 0.9), slab(-0.35, -0.27, 0.3, 0.6), slab(0.27, 0.35, 0.3, 0.6)])
    arms = readiness.limbs(tris)[0]
    assert arms["parts"] == 3 and arms["clear_m"] == pytest.approx(0.2) and arms["joined_m"] == pytest.approx(0.3)


def test_buried_slivers_and_unmirrored_triangles_are_counted_by_node(tmp_path):
    # p0 a sphere, p1 a small sphere sunk inside it, p2 a box off to one side (no mirror twin)
    small = sphere(0.2)
    r = reading(tmp_path, [sphere(), small, box((0.1, 0.1, 0.1), (0.9, 0, 0))])
    rows = {n["name"]: n for n in r["by_node"]}
    n = len(small[1])
    assert rows["p1"]["buried"] == n and rows["p0"]["buried"] == 0 and rows["p2"]["buried"] == 0
    assert rows["p2"]["unmirrored"] == 12 and rows["p0"]["unmirrored"] == 0
    assert readiness.worst_nodes(r, "buried") == f"p1 {n} of {n}"
    assert f"  buried by node (triangles): p1 {n} of {n}" in "\n".join(readiness.lines(r))
