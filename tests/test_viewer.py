"""mesh_jig.viewer: the index a results page is drawn from, and the server that hands it out. The projects are the
known-answer one from test_evaluate, measured for real with a fake build, then copied the way a comparison is made."""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import glbkit
from mesh_jig import cli, project, render, viewer
from test_evaluate import LENGTH, fake_build, make_project, run, ship


@pytest.fixture
def scan(tmp_path):
    """scan/alpha (two attempts, one failed, and an agent run), scan/beta and scan/deep/alpha (copies with no
    attempts), scan/other (another reference), scan/empty (a jig.json with no reference views)."""
    proj = make_project(tmp_path)
    run(proj, tmp_path, "a01", note="the right shape", build_fn=fake_build())
    run(proj, tmp_path, "a02", note="too wide", build_fn=fake_build(span=0.07))
    run(proj, tmp_path, "a03", build_fn=fake_build(error="JIG_BUILD_ERROR: line 3"))
    report = proj.root / "agent" / "20261003-000000_m"
    report.mkdir(parents=True)
    (report / "report.json").write_text(json.dumps({"model": "m", "turns": 4, "evaluations": 2, "done": True,
                                                    "best": {"name": "a01", "numeric_score": 0.99}, "cost_usd": 0.01}))
    (report / "transcript.jsonl").write_text('{"role": "user", "content": "hi"}\n')
    root = tmp_path / "scan"
    shutil.copytree(proj.root, root / "alpha")
    for bare in (root / "beta", root / "deep" / "alpha"):
        shutil.copytree(proj.root, bare, ignore=shutil.ignore_patterns("attempts", "agent"))
    shutil.copytree(root / "beta", root / "other")
    top = Image.open(root / "other" / "refs" / "top.png").convert("RGB")
    top.putpixel((0, 0), (1, 2, 3))
    top.save(root / "other" / "refs" / "top.png")
    (root / "empty").mkdir()
    (root / "empty" / "jig.json").write_text(json.dumps({"name": "empty", "model": {}}))
    shutil.rmtree(proj.root)          # the copies must stand on their own: eval.json still names this directory
    return root


def test_projects_with_the_same_reference_views_are_one_reference(scan):
    index = viewer.build_index([scan])
    same, other = sorted(index["references"], key=lambda r: -len(r["projects"]))
    assert (len(same["projects"]), len(other["projects"])) == (3, 1)
    assert same["key"] != other["key"] and same["name"] == "ship"
    assert list(same["views"]) == ["hero", "side", "rear", "top"], "views come in a fixed order"
    # a run is named by its directory, and by its path when two directories share a name
    assert sorted(p["label"] for p in same["projects"]) == ["alpha", "beta", "deep/alpha"]
    assert [s["why"] for s in index["skipped"]] == ["no reference views"]
    assert json.loads(json.dumps(index)) == index, "the index is plain JSON"


def test_an_attempt_is_read_from_its_own_directory_after_the_project_was_copied(scan):
    same = max(viewer.build_index([scan])["references"], key=lambda r: len(r["projects"]))
    alpha = next(p for p in same["projects"] if p["attempts"])
    by_name = {a["name"]: a for a in alpha["attempts"]}
    assert list(by_name) == ["a01", "a02", "a03"] and alpha["best"] == "a01"
    a01 = by_name["a01"]
    assert a01["ok"] and a01["note"] == "the right shape" and a01["numeric"]["numeric_score"] > 0.9
    assert a01["numeric"]["numeric_score"] > by_name["a02"]["numeric"]["numeric_score"]
    top = a01["views"]["top"]
    assert (top["compare"], top["overlay"], top["render"]) == (
        "attempts/a01/compare/top_vs_reference.png", "attempts/a01/compare/sil_top.png", "attempts/a01/renders/top.png")
    assert top["iou"] > 0.97 and a01["script"] == "attempts/a01/ship.py" and a01["eval"] == "attempts/a01/eval.json"
    failed = by_name["a03"]
    assert not failed["ok"] and failed["stage"] == "build" and "line 3" in failed["error"] and failed["views"] == {}
    assert failed["numeric"]["numeric_score"] is None
    (r,) = alpha["runs"]
    assert (r["model"], r["turns"], r["best"]["name"]) == ("m", 4, "a01") and r["transcript"].endswith("transcript.jsonl")
    bare = next(p for p in same["projects"] if p["label"] == "beta")
    assert bare["attempts"] == [] and bare["best"] is None and bare["runs"] == []


def test_an_attempt_carries_every_measure_per_view_and_names_the_views_its_numbers_are_means_over(scan):
    alpha = next(p for p in viewer.build_index([scan / "alpha"])["references"][0]["projects"])
    assert alpha["palette"] == {"red": "#c83c30", "pale": "#e6e0d2"}
    a02 = next(a for a in alpha["attempts"] if a["name"] == "a02")
    assert list(a02["views"]) == ["hero", "side", "rear", "top"] and a02["counted"] == ["side", "rear", "top"]
    for name, view in a02["views"].items():
        for key in ("iou", "missing", "extra", "aspect", "reference_aspect", "zone", "structure", "resemblance"):
            assert view[key] is not None, (name, key)
        assert view["resemblance"] == pytest.approx((view["zone"] + view["structure"]) / 2, abs=1e-3)
    top = a02["views"]["top"]
    assert top["extra"] > top["missing"] and top["aspect"] < top["reference_aspect"], "too wide, and it says so"

    def mean(key):
        return sum(a02["views"][v][key] for v in a02["counted"]) / 3
    # the attempt's numbers are the means over the counted views, the hero left out
    assert a02["numeric"]["ortho_iou"] == pytest.approx(mean("iou"), abs=1e-3)
    assert a02["numeric"]["zone"] == pytest.approx(mean("zone"), abs=1e-3)
    assert a02["structure"] == pytest.approx(mean("structure"), abs=1e-3)
    assert a02["numeric"]["resemblance"] == pytest.approx((a02["numeric"]["zone"] + a02["structure"]) / 2, abs=1e-3)
    assert a02["gates"] == [] and a02["wall_s"] is not None and a02["colours"] is True, "a palette: a colour reading"
    failed = next(a for a in alpha["attempts"] if a["name"] == "a03")
    assert failed["counted"] == [] and failed["structure"] is None
    # an attempt measured on none of top, side and rear is measured on the views it has
    path = scan / "alpha" / "attempts" / "a02" / "eval.json"
    rec = json.loads(path.read_text())
    rec["silhouette"] = {"hero": rec["silhouette"]["hero"]}
    rec["gates"] = [{"name": "resemblance_min", "passed": False, "message": "resemblance 0.400 (min 0.5)"}]
    del rec["zone_reading"]
    path.write_text(json.dumps(rec))
    a02 = viewer.attempt_record(project.load(scan / "alpha"), path.parent)
    assert a02["counted"] == ["hero"] and a02["structure"] == a02["views"]["hero"]["structure"]
    assert a02["colours"] is False and a02["weights"] is None
    assert a02["gates"] == [{"name": "resemblance_min", "passed": False, "message": "resemblance 0.400 (min 0.5)"}]
    # a project that names its scoring views is measured on those, by their weights
    rec["scoring"] = {"weights": {"hero": 3.0, "top": 1.0}}
    path.write_text(json.dumps(rec))
    a02 = viewer.attempt_record(project.load(scan / "alpha"), path.parent)
    hero, top = a02["views"]["hero"]["structure"], a02["views"]["top"]["structure"]
    assert a02["counted"] == ["hero", "top"] and a02["weights"] == {"hero": 3.0, "top": 1.0}
    assert a02["structure"] == pytest.approx((3 * hero + top) / 4, abs=1e-4) and hero != top
    # the page reads these per view, under the names the index gives them
    script = page_script()
    for key in re.findall(r'\["(\w+)", "[^"]+"\]', script[script.index("const MEASURES"):script.index("const MEANING")]):
        assert key in a02["views"]["hero"], key


def test_a_picture_missing_from_disk_is_not_linked(scan):
    (scan / "alpha" / "attempts" / "a01" / "compare" / "top_vs_reference.png").unlink()
    alpha = next(p for p in viewer.build_index([scan / "alpha"])["references"][0]["projects"])
    assert alpha["label"] == "alpha", "a path that is itself the project is labelled by its name"
    assert alpha["attempts"][0]["views"]["top"]["compare"] is None
    assert alpha["attempts"][0]["views"]["top"]["render"] == "attempts/a01/renders/top.png"


def test_a_run_says_the_group_it_was_found_in_and_a_skipped_directory_is_not_searched(scan):
    shutil.copytree(scan / "alpha", scan / "archive" / "runs" / "old")

    def groups(index):
        return {p["label"]: p["group"] for r in index["references"] for p in r["projects"]}

    # the first directory below the searched one; the searched directory's own name for a project directly in it
    assert groups(viewer.build_index([scan])) == {"alpha": "scan", "beta": "scan", "deep/alpha": "deep", "old": "archive",
                                                  "other": "scan"}
    index = viewer.build_index([scan], skip=("archive",))
    assert "old" not in groups(index) and index["skip"] == ["archive"]
    assert viewer.project_id(scan / "archive" / "runs" / "old") not in viewer.roots_of([scan], ("archive",))
    assert viewer.project_id(scan / "archive" / "runs" / "old") in viewer.roots_of([scan])
    # a searched directory that is itself the project is of the directory it is in
    assert groups(viewer.build_index([scan / "beta", scan / "deep"])) == {"beta": "scan", "alpha": "deep"}
    server = viewer.serve([scan], port=0, skip=["archive"])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/index", timeout=10) as r:
            assert "old" not in groups(json.loads(r.read()))
    finally:
        server.shutdown()
        server.server_close()
    assert cli.parser().parse_args(["view", "a", "--skip", "archive", "--skip", "tmp"]).skip == ["archive", "tmp"]


def test_a_port_another_server_holds_is_refused_not_shared(scan, capsys):
    first = viewer.serve([scan], port=0)
    port = first.server_address[1]
    try:
        with pytest.raises(OSError):
            viewer.serve([scan], port=port)
        assert cli.main(["view", str(scan), "--port", str(port)]) == 2
        assert "cannot listen" in capsys.readouterr().err
    finally:
        first.server_close()


def test_only_pictures_and_text_inside_a_found_project_are_served(scan, tmp_path):
    roots = viewer.roots_of([scan])
    pid = viewer.project_id(scan / "alpha")
    (tmp_path / "secret.json").write_text("{}")
    assert viewer.safe_file(roots, pid, "refs/top.png") == (scan / "alpha" / "refs" / "top.png").resolve()
    assert viewer.safe_file(roots, pid, "../../secret.json") is None
    assert viewer.safe_file(roots, pid, "attempts/a01/out/ship.glb") == (scan / "alpha/attempts/a01/out/ship.glb").resolve()
    (scan / "alpha" / "attempts" / "a01" / "out" / "ship.blend").write_bytes(b"BLENDER")
    assert viewer.safe_file(roots, pid, "attempts/a01/out/ship.blend") is None, "not a servable type"
    assert viewer.safe_file(roots, pid, "refs/nope.png") is None
    assert viewer.safe_file(roots, "0000000000", "refs/top.png") is None
    assert viewer.safe_file(roots, pid, "") is None


def test_the_server_hands_out_the_page_the_index_and_the_pictures(scan):
    server = viewer.serve([scan], port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers["Content-Type"], r.read()
    try:
        status, ctype, body = get("/")
        assert status == 200 and ctype.startswith("text/html") and b"mesh-jig results" in body
        index = json.loads(get("/api/index")[2])
        assert index == viewer.build_index([scan])
        pid = viewer.project_id(scan / "alpha")
        status, ctype, body = get(f"/file/{pid}/attempts/a01/compare/top_vs_reference.png")
        assert ctype == "image/png" and body[:4] == b"\x89PNG"
        status, ctype, body = get(f"/panel/{pid}/refs/side.png?h=64")
        assert ctype == "image/jpeg" and Image.open(io.BytesIO(body)).height == 64
        assert get(f"/file/{pid}/attempts/a01/ship.py")[1].startswith("text/plain")
        status, ctype, body = get(f"/file/{pid}/attempts/a01/out/ship.glb")
        assert ctype == "model/gltf-binary" and body[:4] == b"glTF", "the build, for the page to draw in 3D"
        (scan / "alpha" / "attempts" / "a01" / "out" / "ship.blend").write_bytes(b"BLENDER")
        for bad in (f"/file/{pid}/%2e%2e/%2e%2e/secret.json", f"/file/{pid}/attempts/a01/out/ship.blend", "/nope",
                    f"/panel/{pid}/jig.json"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(bad)
            assert e.value.code == 404, bad
    finally:
        server.shutdown()
        server.server_close()


def test_a_tile_is_one_size_whatever_the_subject_and_stands_on_the_pictures_own_ground(scan):
    side = scan / "alpha" / "refs" / "side.png"
    free = Image.open(io.BytesIO(viewer.panel_jpeg(side, 96)))
    assert free.height == 96 and free.width != 96, "without a width the panel is as wide as the subject makes it"
    tile = Image.open(io.BytesIO(viewer.panel_jpeg(side, 96, 96))).convert("RGB")
    assert tile.size == (96, 96)
    ground = Image.open(side).convert("RGB").getpixel((0, 0))
    for corner in ((0, 0), (95, 0), (0, 95), (95, 95)):
        assert max(abs(a - b) for a, b in zip(tile.getpixel(corner), ground)) <= 6, "padded with the picture's ground"
    assert Image.open(io.BytesIO(viewer.panel_jpeg(side, 96, 40))).size == (40, 96), "a subject too wide is scaled to fit"


def test_a_picture_is_sent_once_and_the_directories_are_not_searched_for_each_one(scan, monkeypatch):
    searches = []
    roots_of = viewer.roots_of
    monkeypatch.setattr(viewer, "roots_of", lambda *args: searches.append(1) or roots_of(*args))
    server = viewer.serve([scan], port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path, tag=None):
        req = urllib.request.Request(base + path, headers={"If-None-Match": tag} if tag else {})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.headers["ETag"], r.read()
    try:
        pid = viewer.project_id(scan / "alpha")
        urls = [f"/file/{pid}/refs/top.png", f"/panel/{pid}/refs/top.png?h=64&w=64", f"/file/{pid}/attempts/a01/eval.json"]
        tags = {u: get(u)[0] for u in urls}
        assert len(searches) == 1 and all(tags.values()), "one search finds the project, the next pictures reuse it"
        for u in urls:
            with pytest.raises(urllib.error.HTTPError) as e:
                get(u, tags[u])
            assert e.value.code == 304, "the browser's copy is still the file on disk"
        assert get(f"/panel/{pid}/refs/top.png?h=64")[0] != tags[urls[1]], "another size is another picture"
        top = scan / "alpha" / "refs" / "top.png"
        Image.new("RGB", (40, 30), (9, 9, 9)).save(top)      # the file is written again, as evaluate does
        tag, body = get(urls[0], tags[urls[0]])
        assert tag != tags[urls[0]] and body == top.read_bytes()
        # a project copied in while the page is open is found without restarting
        shutil.copytree(scan / "beta", scan / "late")
        assert get(f"/file/{viewer.project_id(scan / 'late')}/refs/side.png")[1][:4] == b"\x89PNG"
        assert len(searches) == 2
    finally:
        server.shutdown()
        server.server_close()


def page_script() -> str:
    return re.search(r"<script>(.*)</script>", viewer.PAGE, re.S).group(1)


def test_the_page_script_finds_every_element_it_asks_for():
    ids = set(re.findall(r'\$\("([^"]+)"\)', page_script()))
    assert ids and not {i for i in ids if f'id="{i}"' not in viewer.PAGE}
    # and reads only what the index holds for an attempt and a run
    assert set(re.findall(r"numeric\.(\w+)", page_script())) <= set(viewer.NUMERIC_KEYS)


def test_the_page_script_parses(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    (tmp_path / "page.mjs").write_text(page_script(), encoding="utf-8")
    done = subprocess.run([node, "--check", str(tmp_path / "page.mjs")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_command_refuses_a_path_that_is_not_a_directory(tmp_path, capsys):
    assert cli.main(["view", str(tmp_path / "nowhere")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_an_attempt_names_the_commit_that_measured_it_and_the_index_says_where_to_look_it_up(scan, monkeypatch):
    def attempts(project="alpha"):
        return {a["name"]: a["harness"] for a in viewer.build_index([scan / project])["references"][0]["projects"][0]["attempts"]}

    def record(name, harness):
        f = scan / "alpha" / "attempts" / name / "eval.json"
        f.write_text(json.dumps({**json.loads(f.read_text()), "harness": harness}))

    record("a01", {"commit": "abc1234", "dirty": True})
    record("a02", None)
    record("a03", None)
    assert attempts() == {"a01": {"commit": "abc1234", "dirty": True}, "a02": None, "a03": None}
    # a run script's own note of the commit stands in for an attempt that recorded none
    (scan / "alpha" / "harness-commit.txt").write_text("e377098\nmesh-jig agent --model m --effort low\n")
    assert attempts() == {"a01": {"commit": "abc1234", "dirty": True}, "a02": {"commit": "e377098", "dirty": None},
                          "a03": {"commit": "e377098", "dirty": None}}
    # what goes into a link is a commit or nothing
    record("a01", {"commit": 'x" onclick="alert(1)', "dirty": False})
    (scan / "alpha" / "harness-commit.txt").write_text("<script>\n")
    assert set(attempts().values()) == {None}
    monkeypatch.setattr(viewer.harness_mod, "repo_url", lambda: "https://example.com/owner/repo")
    assert viewer.build_index([scan])["repo"] == "https://example.com/owner/repo"
    # a commit is a link only when it is in the checkout's history: a record made on another history is plain text
    record("a01", {"commit": "abc1234", "dirty": False})
    (scan / "alpha" / "harness-commit.txt").write_text("e377098\n")
    monkeypatch.setattr(viewer.harness_mod, "in_history", lambda commit: commit == "abc1234")
    assert viewer.build_index([scan])["linked"] == ["abc1234"]
    monkeypatch.setattr(viewer.harness_mod, "repo_url", lambda: None)
    assert viewer.build_index([scan])["linked"] == [], "nowhere to link to"


def test_the_page_puts_the_commit_under_every_score():
    script = page_script()
    assert "${esc(data.repo)}/commit/${esc(h.commit)}" in script
    assert "data.repo && (data.linked || []).includes(h.commit) ?" in script, "a link only for a commit in the history"
    # the best of each run, every attempt in the list, each model in 3D, an agent run's best, and the picked attempt
    assert script.count("commit(") == 6, "the function and its five uses"
    for after_the_score in ('<span class="score">${f(a.numeric.numeric_score)}</span>${commit(a)}',
                            '<span>score</span>${commit(a)}</div>'):
        assert after_the_score in script


def test_a_run_says_what_drove_it_from_what_it_recorded(scan):
    def driver(project="alpha"):
        (p,) = viewer.build_index([scan / project])["references"][0]["projects"]
        return p["driver"]

    def report(**fields):
        f = scan / "alpha" / "agent" / "20261003-000000_m" / "report.json"
        f.write_text(json.dumps({**json.loads(f.read_text()), **fields}))

    # a report with no `harness` is mesh-jig agent's own; nothing says the host or the effort
    assert driver() == {"kind": "agent", "name": "API", "detail": "", "model": "m", "effort": None}
    assert driver("beta") is None, "no report and no command: not guessed from the directory's name"
    report(url="https://openrouter.ai/api/v1", model="anthropic/claude-opus-5.5")
    (scan / "alpha" / "harness-commit.txt").write_text("e377098\nmesh-jig agent --model x --attempts 5 --effort high --max-cost 8 \n")
    assert driver() == {"kind": "agent", "name": "API", "detail": "openrouter.ai", "model": "anthropic/claude-opus-5.5",
                        "effort": "high"}
    # a run brought in from a coding agent's log names the tool, whatever the command file says
    report(harness="Claude Code 2.1.289", model="claude-opus-5-5")
    assert driver() == {"kind": "claude-code", "name": "Claude Code", "detail": "2.1.289", "model": "claude-opus-5-5",
                        "effort": "high"}
    report(harness="Codex 0.98")
    assert (driver()["kind"], driver()["name"], driver()["detail"]) == ("codex", "Codex", "0.98")
    report(harness="Some Other Tool 3")
    assert (driver()["kind"], driver()["name"]) == ("other", "Some Other Tool 3")
    # with no report, the command the run script noted says
    for command, kind, model, effort in (("claude -p --model opus --effort low (run_claude_code.sh cat)", "claude-code", "opus", "low"),
                                         ("codex exec --model gpt-6-astra", "codex", "gpt-6-astra", None),
                                         ("mesh-jig agent --model=q --url http://localhost:1234/v1", "agent", "q", None),
                                         ("claudette --model m", None, None, None), ("", None, None, None)):
        (scan / "beta" / "harness-commit.txt").write_text(f"e377098\n{command}\n")
        d = driver("beta")
        assert (d and (d["kind"], d["model"], d["effort"])) == (kind and (kind, model, effort)), command
    # the page names a run by these fields and narrows the runs to one of these kinds
    script = page_script()
    assert set(re.findall(r"\bd\.(\w+)", script[script.index("function badge"):script.index("function bestRuns")])) <= set(driver())
    assert {k for k, _, _ in viewer.DRIVERS} | {"other", "none"} == set(json.loads(re.search(r"const KINDS = (\[.*?\]);", script).group(1)))


def test_an_attempt_links_its_glb_and_a_run_that_kept_only_its_best_build_links_that(scan):
    def glbs(project="alpha"):
        (p,) = viewer.build_index([scan / project])["references"][0]["projects"]
        return {a["name"]: a["glb"] for a in p["attempts"]}

    assert glbs() == {"a01": "attempts/a01/out/ship.glb", "a02": "attempts/a02/out/ship.glb", "a03": None}
    # the sweep's runs: attempts/*/out is not kept, and best/ holds the best attempt's GLB with best.json naming it
    best = scan / "alpha" / "best"
    best.mkdir()
    shutil.move(scan / "alpha" / "attempts" / "a02" / "out" / "ship.glb", best / "ship.glb")
    shutil.rmtree(scan / "alpha" / "attempts" / "a01" / "out")
    assert glbs() == {"a01": None, "a02": None, "a03": None}, "a GLB in best/ belongs to no attempt until best.json says"
    (best / "best.json").write_text(json.dumps({"attempt": "a02", "numeric_score": 0.5}))
    assert glbs() == {"a01": None, "a02": "best/ship.glb", "a03": None}
    (best / "best.json").write_text("[]")
    assert glbs()["a02"] is None


def test_the_index_carries_the_renderers_lights_and_each_projects_cameras(scan):
    index = viewer.build_index([scan / "alpha"])
    assert index["light"]["background"] == pytest.approx([c / 255 for c in render.BACKGROUND])
    assert index["light"]["ambient"] == pytest.approx(list(render.AMBIENT))
    assert len(index["light"]["lights"]) == len(render.LIGHTS)
    for said, (direction, colour) in zip(index["light"]["lights"], render.LIGHTS):
        assert np.linalg.norm(said["direction"]) == pytest.approx(1.0)
        assert np.cross(said["direction"], direction) == pytest.approx([0, 0, 0], abs=1e-9)
        assert said["colour"] == pytest.approx(list(colour * render.DIFFUSE))
    stage = index["references"][0]["projects"][0]["stage"]
    assert stage["scale"] == pytest.approx(1 / LENGTH) and stage["colour"] == "vertex"
    assert [s["name"] for s in stage["shots"]] == [s["name"] for s in render.DEFAULT_SHOTS]
    for said, shot in zip(stage["shots"], render.DEFAULT_SHOTS):
        eye, basis, tan = render.camera(shot)
        assert (said["eye"], said["target"], said["tan"], said["ortho"]) == (list(eye), shot["cam_target"], tan, 0.0)
        assert said["forward"] == list(basis[2]) and said["up"] == list(basis[1])
    # the judged variant's shots only, a shot that is not a camera left out, and the texture rule named
    spec = {"glb": "k_{variant}.glb", "variants": ["a", "b"], "stage_scale": 2.0, "textured": True, "texture_mode": "multiply",
            "shots": [{"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.1},
                      {"name": "other", "cam_pos": [0, 0, 3], "cam_target": [0, 0, 0], "variant": "b"},
                      {"name": "broken", "cam_pos": [0, 0, 3]}, {"name": "nowhere", "cam_pos": [1, 1, 1], "cam_target": [1, 1, 1]}]}
    stage = viewer.stage_of(spec)
    assert (stage["scale"], stage["colour"], [(s["name"], s["ortho"]) for s in stage["shots"]]) == (2.0, "multiply", [("front", 1.1)])


needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
# the part of the page that reads a GLB and places the camera, made to print what it read
NODE_TAIL = r"""
import {readFileSync} from "node:fs";
const [glb, stageFile] = process.argv.slice(2), stage = JSON.parse(readFileSync(stageFile, "utf8")), bytes = readFileSync(glb);
const m = readModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), stage.scale, stage.colour);
const corner = (values, width) => Array.from(m.idx, i => Array.from(values.subarray(i * width, i * width + width)));
const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(i => [0, 1, 2].map(r => (i >> r & 1 ? m.max : m.min)[r]));
const picture = cam => {
  const {vp, unit} = matrices(cam, reachOf(cam, corners), stage.aspect, 1);
  return {unit, seen: corners.map(c => [0, 1, 2].map(r => (vp[r] * c[0] + vp[4 + r] * c[1] + vp[8 + r] * c[2] + vp[12 + r])
                                                          / (vp[3] * c[0] + vp[7] * c[1] + vp[11] * c[2] + vp[15])))};
};
const shots = stage.shots.map(shot => {
  const cam = cameraOf(shot), {f, u} = basisOf(cam);
  const turned = [[0.9, 0.6], [2.5, -1.2], [-1.6, 1.5], [4.0, 0.0]].map(([yaw, pitch]) => picture({...cam, yaw, pitch}));
  return {name: shot.name, f, u, ...picture(cam), turned};
});
console.log(JSON.stringify({triangles: m.triangles, pos: corner(m.pos, 3), col: corner(m.col, 3), nrm: corner(m.nrm, 3),
  uv: corner(m.uv, 2), groups: m.groups, images: m.images.map(i => i.bytes.length), shots}));
"""


# the whole page script with no page under it: enough of a browser for it to load, then the attempts list as it
# would be drawn for the reference with the most runs
NO_PAGE = r"""
const nothing = () => ({});
Object.assign(globalThis, {matchMedia: nothing, location: {hash: ""}, history: {replaceState() {}},
  document: {addEventListener() {}, getElementById: nothing, querySelectorAll: () => []}});
"""
LIST_TAIL = r"""
import {readFileSync} from "node:fs";
data = JSON.parse(readFileSync(process.argv[2], "utf8"));
const ref = data.references.reduce((a, b) => b.projects.length > a.projects.length ? b : a);
const id = label => ref.projects.find(p => p.label === label).id;
const listedRows = () => [...attempts(ref).matchAll(/<tr class="row[^"]*?( under)? *" tabindex="0" data-pick="([^"]*)"/g)].map(m =>
  ref.projects.find(p => m[2].startsWith(p.id)).label + "/" + m[2].split("/").pop() + (m[1] ? " under" : ""));
const out = {};
state.ref = ref.key;
for (const p of ref.projects) state.groups.add(p.group);          // every section open: this is about a run's rows
out.best = listedRows();
state.open.add(id("alpha")); out.opened = listedRows(); state.open.clear();
state.pick = id("gamma") + "/a03"; out.picked = listedRows();
state.pick = id("alpha") + "/a02"; out.hidden = listedRows(); state.pick = null;
state.every = true; out.every = listedRows();
console.log(JSON.stringify(out));
"""


@needs_node
def test_the_attempts_list_is_one_row_a_run_until_a_run_is_opened_or_every_attempt_is_asked_for(scan, tmp_path):
    shutil.copytree(scan / "alpha", scan / "gamma")
    for name in ("a01", "a02"):          # a run with nothing measured: its last attempt stands for it
        shutil.rmtree(scan / "gamma" / "attempts" / name)
    shutil.copytree(scan / "alpha" / "attempts" / "a03", scan / "gamma" / "attempts" / "a00")
    (tmp_path / "index.json").write_text(json.dumps(viewer.build_index([scan])))
    (tmp_path / "list.mjs").write_text(NO_PAGE + page_script() + LIST_TAIL, encoding="utf-8")
    done = subprocess.run([shutil.which("node"), str(tmp_path / "list.mjs"), str(tmp_path / "index.json")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got["best"] == ["alpha/a01", "gamma/a03"], "the best attempt of a run; the last of a run that measured none"
    assert got["opened"] == ["alpha/a01", "alpha/a02 under", "alpha/a03 under", "gamma/a03"]
    assert got["picked"] == ["alpha/a01", "gamma/a03"], "picking the row that is shown opens nothing"
    assert got["hidden"] == got["opened"], "a picked attempt that has no row of its own opens its run"
    assert got["every"] == ["alpha/a01", "alpha/a02", "alpha/a03", "gamma/a00", "gamma/a03"], "ranked together, failures last"


SECTION_TAIL = r"""
import {readFileSync} from "node:fs";
data = JSON.parse(readFileSync(process.argv[2], "utf8"));
const ref = data.references.reduce((a, b) => b.projects.length > a.projects.length ? b : a);
const lone = data.references.find(r => r !== ref);
const of = id => ref.projects.find(p => id.startsWith(p.id)).label + "/" + id.split("/").pop();
const id = label => ref.projects.find(p => p.label === label).id;
const picks = html => [...html.matchAll(/<tr class="row[^"]*" tabindex="0" data-pick="([^"]*)"/g)].map(m => of(m[1]));
const heads = html => [...html.matchAll(/class="fold"(?: data-group="[^"]*" aria-expanded="(true|false)")?[^>]*>\s*<i>[^<]*<\/i><b>([^<]*)<\/b>/g)]
  .map(m => m[2] + (m[1] === "true" ? " open" : ""));
drawBoard = () => {}; drawInspector = () => {};          // toggleGroup draws the page again: there is none here
const out = {};
state.ref = ref.key;
out.heads = heads(attempts(ref));
out.buttons = [...attempts(ref).matchAll(/<button class="fold" data-group="([^"]*)"/g)].map(m => m[1]);
out.closed = picks(attempts(ref));
out.grid = picks(grid(ref));
out.gridHeads = heads(grid(ref));
out.lineup = listed(ref).map(p => p.label);
state.groups.add("exp"); out.opened = picks(attempts(ref)); out.openedGrid = picks(grid(ref)); state.groups.clear();
state.pick = id("one") + "/a02"; out.picked = heads(attempts(ref)); out.pickedRows = picks(attempts(ref));
toggleGroup("exp"); out.afterClose = [of(state.pick), ...heads(attempts(ref))];
toggleGroup("exp"); out.afterOpen = heads(attempts(ref)); toggleGroup("exp");
state.every = true; out.every = picks(attempts(ref)).length; state.every = false;
state.ref = lone.key; out.lone = [sectioned(lone), heads(grid(lone)).length, listed(lone).length];
console.log(JSON.stringify(out));
"""


@needs_node
def test_the_runs_of_a_reference_are_listed_in_sections_closed_to_their_top_scoring_run(scan, tmp_path):
    shutil.copytree(scan / "alpha", scan / "exp" / "runs" / "one")
    shutil.rmtree(scan / "exp" / "runs" / "one" / "attempts" / "a01")          # its best is a02, under two's a01
    shutil.copytree(scan / "alpha", scan / "exp" / "runs" / "two")
    (tmp_path / "index.json").write_text(json.dumps(viewer.build_index([scan])))
    (tmp_path / "sections.mjs").write_text(NO_PAGE + page_script() + SECTION_TAIL, encoding="utf-8")
    done = subprocess.run([shutil.which("node"), str(tmp_path / "sections.mjs"), str(tmp_path / "index.json")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    # a section a group, by its top score; each closed, with its top scoring run alone under it
    assert got["heads"] == got["gridHeads"] == ["scan", "exp", "deep"]
    assert got["buttons"] == ["scan", "exp"], "a section of one run has nothing to open"
    assert got["closed"] == got["grid"] == ["alpha/a01", "two/a01"], "deep has a run and nothing measured"
    assert got["lineup"] == ["alpha", "two", "deep/alpha"], "in 3D too: one run a closed section"
    assert got["opened"] == got["openedGrid"] == ["alpha/a01", "two/a01", "one/a02"]
    # an attempt picked in a run that a closed section would hide opens the section
    assert got["picked"] == ["scan", "exp open", "deep"] and got["pickedRows"] == got["opened"]
    assert got["afterClose"] == ["two/a01", "scan", "exp", "deep"], "closing it hands the pick to its top run"
    assert got["afterOpen"] == ["scan", "exp open", "deep"]
    assert got["every"] == 8, "ranked together, every attempt is a row whatever section its run is in"
    assert got["lone"] == [False, 0, 1], "a reference whose runs are all of one group has no sections"


LEFT_OUT_TAIL = r"""
console.log(JSON.stringify(leftOut(JSON.parse(process.argv[2]))));
"""


@needs_node
def test_the_footer_counts_the_projects_left_out_and_names_a_few_not_every_path(tmp_path):
    records = [{"path": rf"C:\somewhere\deep\runs\run-{n}", "why": "no reference views"} for n in range(26)]
    broken = [{"path": "/somewhere/else/broken", "why": "jig.json: no model block"}]
    (tmp_path / "left.mjs").write_text(NO_PAGE + page_script() + LEFT_OUT_TAIL, encoding="utf-8")

    def said(skipped) -> str:
        done = subprocess.run([shutil.which("node"), str(tmp_path / "left.mjs"), json.dumps(skipped)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    assert said([]) == "", "nothing left out, nothing said"
    assert said(records[:2]) == "2 (no reference views): run-0, run-1"
    # one reason is one count however many there are, and a project that failed to read keeps its own reason
    assert said(records + broken) == ("26 (no reference views): run-0, run-1, run-2 and 23 more; "
                                      "1 (jig.json: no model block): broken")


def page_reads(tmp_path, glb_path, stage) -> dict:
    """What the page's own GLB reader and camera make of a file, run under node."""
    script = page_script()
    part = script[script.index("// -- 3D: reading a GLB"):script.index("// -- 3D: the stage")]
    (tmp_path / "read.mjs").write_text(part + NODE_TAIL, encoding="utf-8")
    (tmp_path / "stage.json").write_text(json.dumps({"shots": [], "aspect": 1.0, **stage}))
    done = subprocess.run([shutil.which("node"), str(tmp_path / "read.mjs"), str(glb_path), str(tmp_path / "stage.json")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def turned_glb(path):
    """One cube three times: under a parent that is turned, stretched and moved, then moved again; and mirrored by a
    matrix. Colours are normalised shorts, as Blender writes them."""
    gltf, blob = glbkit._empty(), bytearray()
    add = glbkit._adder(gltf, blob)
    pos, col, nrm, idx = glbkit.cube((0.2, 0.1, 0.3), (0.05, 0.0, 0.0))
    prim = {"attributes": {"POSITION": add(pos, "VEC3", 5126), "NORMAL": add(nrm, "VEC3", 5126),
                           "COLOR_0": add((col * 65535 + 0.5).astype(np.uint16), "VEC4", 5123)}, "indices": add(idx, "SCALAR", 5123)}
    gltf["accessors"][prim["attributes"]["COLOR_0"]]["normalized"] = True
    gltf["meshes"].append({"primitives": [prim]})
    axis, half = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0), np.radians(40.0) / 2
    gltf["nodes"] += [{"rotation": [*(axis * np.sin(half)), float(np.cos(half))], "scale": [1, 2, 0.5],
                       "translation": [0.1, 0.2, -0.3], "children": [1]},
                      {"mesh": 0, "translation": [0, 0.05, 0]},
                      {"mesh": 0, "matrix": [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0, 0, 1]}]
    gltf["scenes"][0]["nodes"] = [0, 2]
    return glbkit._pack(gltf, blob, path)


@needs_node
def test_the_page_reads_a_glb_as_the_renderer_does(tmp_path):
    def same(got, mesh, scale=1.0, normals=True):
        assert got["triangles"] == len(mesh.tris) > 0
        assert np.array(got["pos"]).reshape(-1, 3, 3) == pytest.approx(mesh.tris * scale, abs=1e-5)
        assert np.array(got["col"]).reshape(-1, 3, 3) == pytest.approx(mesh.colors, abs=1e-4)
        if normals:
            assert np.array(got["nrm"]).reshape(-1, 3, 3) == pytest.approx(mesh.normals, abs=1e-5)

    turned = turned_glb(tmp_path / "turned.glb")
    got = page_reads(tmp_path, turned, {"scale": 2.5, "colour": "vertex"})
    same(got, render.load(turned), 2.5)
    assert [(g["tex"], g["tint"], g["start"], g["count"]) for g in got["groups"]] == [(-1, [1, 1, 1], 0, 72)], "one draw"
    # positions interleaved in their buffer, nodes moved, and no normals: those are left for the shader
    plain = glbkit.write_glb(tmp_path / "plain.glb", [("a", glbkit.box((0.1, 0.2, 0.3)), (0.5, 0, 0)),
                                                      ("b", glbkit.box((0.3, 0.1, 0.1)), (0, 0, -0.4))], interleave=True)
    got = page_reads(tmp_path, plain, {"scale": 1.0, "colour": "vertex"})
    same(got, render.load(plain), normals=False)
    assert not np.any(got["nrm"])
    # a textured contract: the texture alone, or the vertex colour times it; a vertex contract does not read it
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (200, 30, 30)).save(buf, format="PNG")
    atlas = glbkit.textured_glb(tmp_path / "atlas.glb", png=buf.getvalue())
    for colour, kw in (("atlas", {"textured": True}), ("multiply", {"textured": True, "multiply": True}), ("vertex", {})):
        got = page_reads(tmp_path, atlas, {"scale": 1.0, "colour": colour})
        mesh = render.load(atlas, **kw)
        same(got, mesh, normals=False)
        assert (got["images"], [g["tex"] for g in got["groups"]]) == (([len(buf.getvalue())], [0]) if mesh.textures else ([], [-1])), colour
    bare = glbkit.textured_glb(tmp_path / "bare.glb", textured_material=False, png=buf.getvalue())
    got = page_reads(tmp_path, bare, {"scale": 1.0, "colour": "atlas"})
    same(got, render.load(bare, textured=True), normals=False)
    assert got["images"] == [] and got["groups"][0]["tex"] == -1, "a material with no texture is its factor alone"


@needs_node
def test_the_page_camera_stands_where_the_renderers_does_and_shows_the_whole_model(tmp_path):
    shots = render.DEFAULT_SHOTS + [
        {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.12},
        {"name": "across", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.1},
        {"name": "down", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "ortho": 1.1},
        {"name": "down_turned", "cam_pos": [0, 3, 0], "cam_target": [0, 0, 0], "cam_up": [0, 0, 1], "ortho": 1.1},
        {"name": "up", "cam_pos": [0.2, -3, 0.1], "cam_target": [0.2, 0, 0.1], "cam_up": [1, 0, 0], "fov": 30}]
    stage = viewer.stage_of({"length_m": LENGTH, "shots": shots})
    assert len(stage["shots"]) == len(shots)
    for aspect in (0.8, 1.6):
        got = page_reads(tmp_path, ship(tmp_path / "ship.glb"), {**stage, "aspect": aspect})
        for said, shot in zip(got["shots"], stage["shots"]):
            assert said["f"] == pytest.approx(shot["forward"], abs=1e-6), shot["name"]
            assert said["u"] == pytest.approx(shot["up"], abs=1e-6), shot["name"]
            assert np.abs(said["seen"]).max() <= 1.0, f"{shot['name']}: a corner of the model is outside the picture"
            # turned any way from there, the model is still whole and the lens has not moved: it keeps its size
            for turned in said["turned"]:
                assert turned["unit"] == pytest.approx(said["unit"], rel=1e-9), shot["name"]
                assert np.abs(turned["seen"]).max() <= 1.0, shot["name"]
        # and the lens is not much looser than the model: seen along its length it nearly fills the picture
        assert max(np.abs(np.array(s["seen"])[:, :2]).max() for s in got["shots"]) > 0.75


# -- the page as files (`mesh-jig view --export`) ----------------------------------------------------------------------

def test_an_export_is_the_page_its_index_and_its_pictures_and_names_no_directory(scan, tmp_path):
    out = tmp_path / "site"
    wrote = viewer.export([scan], out)
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "const STATIC = true;" in page and viewer.STATIC_FLAG not in page
    assert viewer.PAGE.count(viewer.STATIC_FLAG) == 1, "the served page is not a saved copy"
    text = (out / "index.json").read_text(encoding="utf-8")
    for here in (str(tmp_path), tmp_path.as_posix(), json.dumps(str(tmp_path))[1:-1]):
        assert here not in text, "a saved copy says nothing of where it was made"
    index = json.loads(text)
    live = viewer.build_index([scan])
    assert index["static"] is True and index["paths"] == [] and [s["path"] for s in index["skipped"]] == ["empty"]
    assert [[p["id"] for p in r["projects"]] for r in index["references"]] == [[p["id"] for p in r["projects"]] for r in live["references"]]
    alpha = next(p for r in index["references"] for p in r["projects"] if p["label"] == "alpha")
    assert alpha["path"] == "alpha" and [a["name"] for a in alpha["attempts"]] == ["a01", "a02", "a03"]
    # an agent run keeps its numbers and loses its transcript: the raw log is not published
    assert alpha["runs"][0]["model"] == "m" and alpha["runs"][0]["transcript"] is None
    assert not list(out.rglob("*.jsonl"))
    pid = alpha["id"]
    for kept in ("refs/top.png", "attempts/a01/compare/top_vs_reference.png", "attempts/a01/renders/top.png",
                 "attempts/a01/eval.json", "attempts/a01/ship.py", "attempts/a01/out/ship.glb"):
        assert (out / "file" / pid / kept).read_bytes() == (scan / "alpha" / kept).read_bytes(), kept
    for px in viewer.TILE_SIZES:
        for cut in ("refs/top.png", "attempts/a02/renders/side.png"):
            assert Image.open(out / "panel" / str(px) / pid / f"{cut}.jpg").size == (px, px), (px, cut)
    on_disk = [f for f in out.rglob("*") if f.is_file()]
    assert wrote == {"projects": 4, "attempts": 3, "files": len(on_disk), "bytes": sum(f.stat().st_size for f in on_disk)}
    assert (out / ".nojekyll").is_file(), "a static host that runs Jekyll would leave some directories out"


def test_an_export_of_the_best_attempts_replaces_an_earlier_one_and_nothing_else(scan, tmp_path, capsys):
    out = tmp_path / "site"
    viewer.export([scan], out)
    pid = viewer.project_id(scan / "alpha")
    assert (out / "file" / pid / "attempts" / "a02").is_dir()
    assert cli.main(["view", str(scan), "--export", str(out), "--best"]) == 0
    assert "4 project(s), 1 attempt(s)" in capsys.readouterr().out
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    alpha = next(p for r in index["references"] for p in r["projects"] if p["label"] == "alpha")
    assert [a["name"] for a in alpha["attempts"]] == ["a01"] and alpha["best"] == "a01"
    assert sorted(d.name for d in (out / "file" / pid / "attempts").iterdir()) == ["a01"], "the earlier copy's files are gone"
    assert not (out / "panel" / "240" / pid / "attempts" / "a02").exists()
    # a directory that holds something else is left alone
    other = tmp_path / "notes"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    with pytest.raises(ValueError, match="not an earlier export"):
        viewer.export([scan], other)
    assert cli.main(["view", str(scan), "--export", str(other)]) == 2 and "not an earlier export" in capsys.readouterr().err
    assert [f.name for f in other.iterdir()] == ["keep.txt"]


def test_an_export_cuts_every_size_of_tile_the_page_asks_for():
    asked = {int(px) for px in re.findall(r"tile\([^()]*, (\d+)\)", page_script())}
    assert asked == set(viewer.TILE_SIZES)


# the saved page with no page under it, made to print every address it would ask its host for
EXPORT_TAIL = r"""
import {readFileSync} from "node:fs";
data = JSON.parse(readFileSync(process.argv[2], "utf8"));
const sizes = JSON.parse(process.argv[3]), asked = [];
for (const ref of data.references) {
  for (const v of Object.values(ref.views)) asked.push(file(v.project, v.path), ...sizes.map(px => tile(v.project, v.path, px)));
  for (const p of ref.projects) for (const a of p.attempts) {
    for (const k of ["script", "eval", "glb"]) if (a[k]) asked.push(file(p.id, a[k]));
    for (const w of Object.values(a.views)) {
      for (const k of ["compare", "overlay", "render"]) if (w[k]) asked.push(file(p.id, w[k]));
      if (w.render) asked.push(...sizes.map(px => tile(p.id, w.render, px)));
    }
  }
}
console.log(JSON.stringify(asked));
"""


@needs_node
def test_every_address_the_saved_page_asks_for_is_a_file_beside_it(scan, tmp_path):
    out = tmp_path / "site"
    viewer.export([scan], out)
    script = re.search(r"<script>(.*)</script>", (out / "index.html").read_text(encoding="utf-8"), re.S).group(1)
    (tmp_path / "asked.mjs").write_text(NO_PAGE + script + EXPORT_TAIL, encoding="utf-8")
    done = subprocess.run([shutil.which("node"), str(tmp_path / "asked.mjs"), str(out / "index.json"),
                           json.dumps(list(viewer.TILE_SIZES))], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    asked = json.loads(done.stdout)
    assert len(asked) > 40 and not [a for a in asked if a.startswith("/") or "?" in a], "relative, and no query to answer"
    missing = [a for a in asked if not (out / urllib.parse.unquote(a)).is_file()]
    assert not missing, missing[:5]


# -- the built model, to take into another program ---------------------------------------------------------------------

def post(base: str, path: str, origin: str | None = None) -> int:
    req = urllib.request.Request(base + path, data=b"", method="POST", headers={"Origin": origin} if origin else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_a_built_model_is_shown_in_its_folder_for_a_page_open_on_this_machine_and_no_other(scan, tmp_path):
    shown, broken = [], []

    def show(path):
        if broken:
            raise OSError("no file manager")
        shown.append(path)
    server = viewer.serve([scan], port=0, show=show)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    base, pid = f"http://127.0.0.1:{port}", viewer.project_id(scan / "alpha")
    glb = f"/reveal/{pid}/attempts/a01/out/ship.glb"
    (tmp_path / "secret.glb").write_bytes(b"glTF")
    try:
        assert post(base, glb) == 200 and post(base, glb, f"http://localhost:{port}") == 200
        assert shown == [(scan / "alpha/attempts/a01/out/ship.glb").resolve()] * 2
        assert post(base, glb, "https://example.com") == 403, "another site's page posting here"
        assert post(base, glb, "null") == 403 and post(base, glb, f"http://192.168.1.20:{port}") == 403
        for bad in (f"/reveal/{pid}/attempts/a01/eval.json", f"/reveal/{pid}/%2e%2e/%2e%2e/secret.glb", f"/reveal/{pid}",
                    f"/reveal/{pid}/attempts/a03/out/ship.glb", "/reveal/0000000000/attempts/a01/out/ship.glb",
                    f"/file/{pid}/attempts/a01/out/ship.glb", "/api/index"):
            assert post(base, bad) == 404, bad
        with pytest.raises(urllib.error.HTTPError) as e:        # asking is not reading: a link followed opens nothing
            urllib.request.urlopen(base + glb, timeout=10)
        assert e.value.code == 404 and len(shown) == 2
        broken.append(True)
        assert post(base, glb) == 500, "a machine with no command for it says so"
    finally:
        server.shutdown()
        server.server_close()
    assert viewer.on_this_machine("::1", "http://[::1]:8796") and viewer.on_this_machine("127.0.0.1", None)
    assert not viewer.on_this_machine("192.168.1.20", None), "the window would open here, not where the page is"
    assert not viewer.on_this_machine("", None)


def test_the_folder_command_selects_the_file_where_the_file_manager_can():
    from pathlib import PurePosixPath, PureWindowsPath
    spaced = PureWindowsPath(r"D:\runs\my mech\attempts\a01\out\mech.glb")
    assert viewer.reveal_command(spaced, "win32") == r'explorer /select,"D:\runs\my mech\attempts\a01\out\mech.glb"'
    posix = PurePosixPath("/runs/my mech/attempts/a01/out/mech.glb")
    assert viewer.reveal_command(posix, "darwin") == ["open", "-R", "/runs/my mech/attempts/a01/out/mech.glb"]
    assert viewer.reveal_command(posix, "linux") == ["xdg-open", "/runs/my mech/attempts/a01/out"]


# what the page puts under a drawn attempt, for a page read from each of these names
TAKE_TAIL = r"""
import {readFileSync} from "node:fs";
data = JSON.parse(readFileSync(process.argv[2], "utf8"));
const p = data.references.flatMap(r => r.projects).find(p => p.label === "alpha");
const [built, failed] = ["a01", "a03"].map(name => p.attempts.find(a => a.name === name));
const first = (html, re) => (html.match(re) || [])[1] || null;
const out = {};
for (const host of ["localhost", "127.0.0.1", "192.168.1.20"]) {
  location.hostname = host;
  const html = take(p, built);
  out[host] = {reveal: first(html, /data-reveal="([^"]*)"/), download: first(html, /<a download href="([^"]*)"/)};
}
location.hostname = "localhost";
out.failed = take(p, failed);
console.log(JSON.stringify(out));
"""


@needs_node
def test_the_page_offers_the_folder_only_where_the_server_can_open_it_and_the_download_everywhere(scan, tmp_path):
    def taken(script: str, index: Path) -> dict:
        (tmp_path / "take.mjs").write_text(NO_PAGE + script + TAKE_TAIL, encoding="utf-8")
        done = subprocess.run([shutil.which("node"), str(tmp_path / "take.mjs"), str(index)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    pid, glb = viewer.project_id(scan / "alpha"), "attempts/a01/out/ship.glb"
    (tmp_path / "index.json").write_text(json.dumps(viewer.build_index([scan])))
    live = taken(page_script(), tmp_path / "index.json")
    assert live["localhost"] == live["127.0.0.1"] == {"reveal": f"/reveal/{pid}/{glb}", "download": f"/file/{pid}/{glb}"}
    assert live["192.168.1.20"] == {"reveal": None, "download": f"/file/{pid}/{glb}"}, "read from another machine"
    assert live["failed"] == "", "an attempt with no GLB has nothing to take"
    # the address the page asks is one the server answers
    shown = []
    server = viewer.serve([scan], port=0, show=shown.append)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert post(f"http://127.0.0.1:{server.server_address[1]}", live["localhost"]["reveal"]) == 200
    finally:
        server.shutdown()
        server.server_close()
    assert shown == [(scan / "alpha" / glb).resolve()]
    # a saved copy has no server to ask: the file beside it, and nothing else
    out = tmp_path / "site"
    viewer.export([scan], out)
    script = re.search(r"<script>(.*)</script>", (out / "index.html").read_text(encoding="utf-8"), re.S).group(1)
    saved = taken(script, out / "index.json")
    assert saved["localhost"] == {"reveal": None, "download": f"file/{pid}/{glb}"}
    assert (out / saved["localhost"]["download"]).read_bytes() == (scan / "alpha" / glb).read_bytes()


# -- the builder is told to offer the page ---------------------------------------------------------------------------

def test_the_skills_and_the_mcp_server_have_a_builder_ask_whether_to_open_the_page_when_it_reports():
    """docs/LESSONS.md section 7: the instructions are a program. Nothing else tells a first user the page exists."""
    import shlex
    from mesh_jig import mcp_server
    skills = Path(__file__).resolve().parents[1] / "skill"
    said = (skills / "mesh-jig" / "SKILL.md").read_text(encoding="utf-8").split("## When you report")[1].split("\n## ")[0]
    fenced = [line.strip() for block in re.findall(r"```[a-z]*\n(.*?)```", said, re.S) for line in block.splitlines()]
    (command,) = [line for line in fenced if line.startswith("mesh-jig ")]
    args = cli.parser().parse_args(shlex.split(command)[1:])
    assert args.cmd == "view" and args.open and not args.export
    assert "asking the user whether" in said and f"localhost:{viewer.DEFAULT_PORT}" in said
    assert "background" in said, "the command never returns: a builder that waits on it hangs"
    asks = re.compile(r"\bask the user whether[^.]*`mesh-jig view <project> --open`", re.I)
    for other in ((skills / "mesh-jig-keep-going" / "SKILL.md").read_text(encoding="utf-8"), mcp_server.INSTRUCTIONS):
        assert asks.search(" ".join(other.split())) and "background" in other
