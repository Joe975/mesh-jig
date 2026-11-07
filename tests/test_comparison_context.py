from __future__ import annotations

import json
import shutil

from mesh_jig import comparison_context, evaluate, project, streak


def _project(root, *, target="quest", max_triangles=None, criteria=None):
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "front.png").write_bytes(b"reference pixels")
    (root / "brief.md").write_text("a brief", encoding="utf-8")
    data = {"model": {"shots": [{"name": "front", "cam_pos": [1, 2, 3]}]},
            "target": target, "criteria": criteria or {}}
    if max_triangles is not None:
        data["targets"] = {target: {"max_triangles": max_triangles}}
    return project.Project(root, data)


def _put(root, name, score, started, *, context=None, readiness=None, ok=True):
    path = root / "attempts" / name
    path.mkdir(parents=True, exist_ok=True)
    rec = {"ok": ok, "started": started, "numeric": {"numeric_score": score} if score is not None else {},
           "gates": []}
    if context is not None:
        rec[comparison_context.FIELD] = context
    if readiness is not None:
        rec["readiness"] = readiness
    (path / "eval.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_signature_tracks_all_effective_inputs_and_ignores_relocation(tmp_path):
    base = _project(tmp_path / "a")
    original = comparison_context.signature(base)
    relocated_root = tmp_path / "other-parent" / "a"
    relocated_root.parent.mkdir()
    relocated_root.mkdir()
    shutil.copytree(base.root / "refs", relocated_root / "refs")
    shutil.copyfile(base.root / "brief.md", relocated_root / "brief.md")
    relocated = project.Project(relocated_root, base.data)
    assert comparison_context.signature(relocated) == original
    assert set(original) == {"schema", "digest", "target"}
    assert str(tmp_path) not in json.dumps(original)

    changed = []
    changed.append(comparison_context.signature(_project(tmp_path / "target", target="pcvr")))
    changed.append(comparison_context.signature(_project(tmp_path / "cap", max_triangles=5000)))
    changed.append(comparison_context.signature(_project(tmp_path / "criteria", criteria={"score": {"overlap": 2}})))
    brief_project = _project(tmp_path / "brief")
    (brief_project.root / "brief.md").write_text("different brief", encoding="utf-8")
    changed.append(comparison_context.signature(brief_project))
    refs_project = _project(tmp_path / "refs")
    (refs_project.root / "refs" / "front.png").write_bytes(b"changed pixels")
    changed.append(comparison_context.signature(refs_project))
    contract_project = _project(tmp_path / "contract")
    contract_project.data["model"]["shots"][0]["cam_pos"] = [4, 5, 6]
    changed.append(comparison_context.signature(contract_project))
    model_project = _project(tmp_path / "model-contract")
    model_project.data["model"]["length_m"] = 1.25
    changed.append(comparison_context.signature(model_project))
    assert all(c["digest"] != original["digest"] for c in changed)

    override = _project(tmp_path / "override")
    override.chosen_target = "pcvr"
    assert comparison_context.signature(override)["digest"] != comparison_context.signature(override.__class__(
        override.root, override.data)).get("digest")


def test_implementation_fingerprint_is_relative_and_part_of_signature(tmp_path):
    proj = _project(tmp_path / "project")
    source = tmp_path / "package" / "mesh_jig"
    source.mkdir(parents=True)
    (source / "measure.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = comparison_context.signature(proj, package_root=source)
    relocated_source = tmp_path / "other" / "mesh_jig"
    relocated_source.mkdir(parents=True)
    shutil.copyfile(source / "measure.py", relocated_source / "measure.py")
    assert comparison_context.source_fingerprint(source) == comparison_context.source_fingerprint(relocated_source)
    (source / "measure.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = comparison_context.signature(proj, package_root=source)
    assert first["digest"] != second["digest"]
    assert first["schema"] == second["schema"] == 1


def test_legacy_buckets_separate_changed_caps_unknown_and_signed_records(tmp_path):
    old = {"ok": True, "readiness": {"target": "handheld", "max_triangles": 10000}}
    same = {"ok": True, "readiness": {"target": "handheld", "max_triangles": 10000}}
    changed_cap = {"ok": True, "readiness": {"target": "handheld", "max_triangles": 5000}}
    unknown = {"ok": True, "readiness": {"target": "handheld"}}
    signed = {"ok": True, comparison_context.FIELD: {"schema": 1, "digest": "a" * 64}}
    assert comparison_context.compatible(old, same)
    assert not comparison_context.compatible(old, changed_cap)
    assert not comparison_context.compatible(old, unknown)
    assert not comparison_context.compatible(
        {"readiness": {"target": "quest"}}, {"readiness": {"target": "pcvr"}})
    assert not comparison_context.compatible(
        {"readiness": []}, {"readiness": []})
    assert not comparison_context.compatible(old, signed)
    assert not comparison_context.compatible(signed, {comparison_context.FIELD: {"schema": 77, "digest": "a" * 64}})
    assert not comparison_context.compatible(signed, {comparison_context.FIELD: {"schema": 1, "digest": "bad"}})
    for schema in (True, 1.0, "1"):
        assert comparison_context.classify({comparison_context.FIELD: {"schema": schema, "digest": "a" * 64}})[0] == "invalid"
    assert comparison_context.classify(None)[0] == "invalid"


def test_standing_and_streak_exclude_other_signed_contexts_and_failed_attempts(tmp_path):
    a = {"schema": 1, "digest": "a" * 64}
    b = {"schema": 1, "digest": "b" * 64}
    _put(tmp_path, "a_old", .80, "2026-10-01 00:00", context=a)
    _put(tmp_path, "b_current", .78, "2026-10-02 00:00", context=b)
    _put(tmp_path, "c_current", .78, "2026-10-03 00:00", context=b)
    _put(tmp_path, "d_current", .78, "2026-10-04 00:00", context=b)
    _put(tmp_path, "e_failed", None, "2026-10-05 00:00", context=a, ok=False)

    latest_dir = tmp_path / "attempts" / "d_current"
    current = json.loads((latest_dir / "eval.json").read_text(encoding="utf-8"))
    standing = evaluate.standing(project.Project(tmp_path, {"model": {}}), latest_dir, current)
    assert standing["others"] == 2 and standing["best"]["name"] == "b_current"
    assert standing["excluded_incompatible"] == 1
    assert "contexts differ" in standing["compatibility_note"]
    text = streak.text(streak.verdict(tmp_path, patience=3))
    assert "evaluated attempts: 3 (1 failed build not counted)" in text
    assert "1 measured attempt excluded" in text
    assert "STOP:" not in text


def test_legacy_streak_keeps_unknown_bucket_and_targetless_ranking(tmp_path):
    _put(tmp_path, "a", .80, "2026-10-01 00:00")
    _put(tmp_path, "b", .78, "2026-10-02 00:00")
    _put(tmp_path, "c", .78, "2026-10-03 00:00")
    _put(tmp_path, "d", .78, "2026-10-04 00:00")
    v = streak.verdict(tmp_path, patience=3)
    assert v["made"] == 4 and v["streak"] == 3 and v["stop"]
    assert "excluded_incompatible" not in v
    assert "provenance unavailable" in streak.text(v)

    target_root = tmp_path / "target-bucket"
    _put(target_root, "a_old", .80, "2026-10-01 00:00", readiness={"target": "handheld", "max_triangles": 10000})
    _put(target_root, "b_new", .70, "2026-10-02 00:00", readiness={"target": "handheld", "max_triangles": 5000})
    _put(target_root, "c_new", .70, "2026-10-03 00:00", readiness={"target": "handheld", "max_triangles": 5000})
    v = streak.verdict(target_root)
    assert v["made"] == 2 and v["excluded_incompatible"] == 1
    assert "readiness target/cap" in streak.text(v)
    newest = target_root / "attempts" / "c_new"
    current = json.loads((newest / "eval.json").read_text(encoding="utf-8"))
    standing = evaluate.standing(project.Project(target_root, {"model": {}}), newest, current)
    assert "provenance unavailable" in standing["provenance_note"]
    assert "provenance unavailable" in " ".join(evaluate.standing_lines(current, standing, []))


def test_max_attempt_ceiling_counts_all_successful_contexts(tmp_path):
    for i in range(4):
        _put(tmp_path, f"a{i}", .7 - i / 100, f"2026-10-0{i + 1} 00:00",
             context={"schema": 1, "digest": f"{i + 1:x}" * 64})
    v = streak.verdict(tmp_path, max_attempts=4)
    assert v["made"] == 1 and v["measured_total"] == 4
    assert v["excluded_incompatible"] == 3 and v["stop"]
    assert v["why"] == "the ceiling of 4 evaluated attempts is reached"


def test_invalid_latest_signature_reports_comparison_unavailable(tmp_path):
    _put(tmp_path, "a_valid", .7, "2026-10-01 00:00", context={"schema": 1, "digest": "a" * 64})
    _put(tmp_path, "b_invalid", .6, "2026-10-02 00:00", context={"schema": True, "digest": "b" * 64})
    v = streak.verdict(tmp_path)
    assert v["made"] == 0 and v["measured_total"] == 2
    assert "comparisons unavailable" in v["why"]
    assert "comparisons unavailable" in streak.text(v)
    newest = tmp_path / "attempts" / "b_invalid"
    current = json.loads((newest / "eval.json").read_text(encoding="utf-8"))
    standing = evaluate.standing(project.Project(tmp_path, {"model": {}}), newest, current)
    assert "comparison unavailable" in " ".join(evaluate.standing_lines(current, standing, []))


def test_new_evaluation_roundtrips_signature_and_standing_same_context(tmp_path):
    from test_evaluate import fake_build, make_project, quick_render, script

    proj = make_project(tmp_path)
    records = []
    for name in ("a", "b"):
        path = script(tmp_path, name)
        rec = evaluate.evaluate(proj, path, path.parent, build_fn=fake_build(), render_fn=quick_render,
                                log=lambda *_: None)
        assert rec["ok"] and rec[comparison_context.FIELD]["schema"] == 1
        stored = json.loads((path.parent / "eval.json").read_text(encoding="utf-8"))
        assert stored[comparison_context.FIELD] == rec[comparison_context.FIELD]
        records.append((path.parent, rec))
    standing = evaluate.standing(proj, *records[1])
    assert standing["others"] == 1 and "excluded_incompatible" not in standing
    preview = evaluate.preview(proj, records[1][0], render_fn=quick_render)
    assert preview[comparison_context.FIELD] == records[1][1][comparison_context.FIELD]
    partial = evaluate.preview(proj, records[1][0], views=["top"], render_fn=quick_render)
    assert "numeric_score" not in partial["numeric"] and comparison_context.FIELD not in partial
