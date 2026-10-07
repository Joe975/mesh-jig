"""Selected public bytes preserve metrics and visuals without exporting raw run data."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from glbkit import cube, textured_glb, write_cubes, _pack
from mesh_jig import glb, render

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skill/mesh-jig-publish-results/scripts/curate.py"
spec = importlib.util.spec_from_file_location("result_curator", SCRIPT)
curator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curator)


def metadata(**changes):
    return dict(title="Example result", driver="API agent", scope="Five successful evaluations, one run",
                cost_kind="reported", effort="low", **changes)


def report():
    return {"model": "example/model", "wall_s": 12.5, "cost_usd": .025, "evaluations": 5, "done": True,
            "prompt_tokens": 100, "cached_tokens": 70, "completion_tokens": 20, "reasoning_tokens": 8,
            "calls": [{"private": "raw model reply"}, {}], "best": {"numeric_score": .6}}


def png(private=False):
    output = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if private:
        info.add_text("private", "sk-" + "Z" * 30)
    Image.new("RGBA", (8, 6), (10, 20, 30, 100)).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def draft(tmp_path, assets=True):
    source = tmp_path / "raw-report.json"
    source.write_text(json.dumps(report()))
    image = tmp_path / "private-image.png"
    image.write_bytes(png(private=True))
    model = write_cubes(tmp_path / "private-model.glb", [cube()])
    dest = tmp_path / "draft"
    curator.prepare(source, dest, images=[image] if assets else [], models=[model] if assets else [], **metadata())
    return dest


def git(root, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + root.as_posix(), *args], cwd=root)


def commit(root):
    git(root, "add", "--all")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@users.noreply.github.com", "commit", "-qm", "test")


def test_only_selected_numeric_fields_leave_a_report_and_unknown_is_not_zero():
    raw = report()
    private = "C:" + "/Users/" + "test/private"
    raw.update(project=private, url="https://private.invalid", summary=private, error=private, usage={"raw": private})
    result = curator.summary(raw, **metadata())
    encoded = curator.encode(result)
    assert private.encode() not in encoded and b"private.invalid" not in encoded and b"raw model reply" not in encoded
    assert result["metrics"]["runtime_seconds"] == 12.5
    assert result["metrics"]["completion_tokens"] == 20 and result["metrics"]["reasoning_tokens"] == 8
    assert result["metrics"]["model_calls"] == 2
    assert result["metrics"]["driver_turns"] is None and result["metrics"]["cache_write_tokens"] is None
    assert result["cost"] == {"kind": "reported", "amount_usd": .025}
    result = curator.summary({"model": "example/model"}, title="Example", driver="subscription", scope="One run", cost_kind="unavailable")
    assert result["cost"]["amount_usd"] is None and result["metrics"]["runtime_seconds"] is None


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), "12"])
def test_invalid_metric_cannot_be_published(bad):
    raw = report(); raw["wall_s"] = bad
    with pytest.raises(ValueError):
        curator.summary(raw, **metadata())


def test_estimated_cost_and_gate_observations_stay_distinct():
    args = metadata(); args["cost_kind"] = "estimated"
    result = curator.summary(report(), evaluation={"ok": True, "numeric": {"numeric_score": .7}, "gates": [
        {"passed": True}, {"passed": False}, {"passed": None, "observation": True}, {"passed": True, "observation": True}]}, **args)
    assert result["result"] == {"numeric_score": .7, "gates_passed": 1, "gates_total": 2}
    assert result["cost"]["kind"] == "estimated"
    with pytest.raises(ValueError):
        curator.summary({"model": "example/model"}, **metadata())


def test_visible_prose_rejects_private_contact_paths_and_credentials():
    for bad in ["C:" + "/Users/" + "test/notes", "user" + "@personal.invalid", "sk-" + "Z" * 30]:
        with pytest.raises(ValueError):
            curator.summary(report(), notes=[bad], **metadata())


def test_image_metadata_is_removed_without_changing_alpha_or_pixels():
    before = png(private=True)
    with pytest.raises(ValueError, match="metadata"):
        curator.validate_png(before)
    after = curator.image_bytes(before)
    curator.validate_png(after)
    assert b"private" not in after and b"sk-" not in after
    with Image.open(io.BytesIO(before)) as a, Image.open(io.BytesIO(after)) as b:
        assert a.mode == b.mode and a.tobytes() == b.tobytes()


def test_png_cannot_hide_data_after_its_compressed_pixel_stream():
    raw = curator.image_bytes(png())
    offset = 8
    while raw[offset + 4:offset + 8] != b"IDAT":
        offset += 12 + struct.unpack_from(">I", raw, offset)[0]
    length = struct.unpack_from(">I", raw, offset)[0]
    payload = raw[offset + 8:offset + 8 + length] + b"private trailing data"
    chunk = b"IDAT" + payload
    modified = raw[:offset] + struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk)) + raw[offset + 12 + length:]
    with pytest.raises(ValueError, match="trailing"):
        curator.validate_png(modified)


def test_glb_metadata_and_unused_buffer_bytes_are_removed_with_exact_geometry(tmp_path):
    original = write_cubes(tmp_path / "base.glb", [cube()])
    doc, blob = glb.read_glb(original)
    private = "C:" + "/Users/" + "test/private"
    doc["asset"]["generator"] = private
    doc["nodes"][0].update(name=private, extras={"token": "sk-" + "Z" * 30})
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(private)})
    doc["buffers"] = []
    before = _pack(doc, bytearray(blob + private.encode()), tmp_path / "raw.glb")
    sanitized = curator.glb_bytes(before.read_bytes())
    after = tmp_path / "clean.glb"; after.write_bytes(sanitized)
    assert private.encode() not in sanitized and b"sk-" not in sanitized
    assert curator.glb_bytes(sanitized) == sanitized
    a, b = render.load(before), render.load(after)
    np.testing.assert_array_equal(a.tris, b.tris)
    np.testing.assert_array_equal(a.colors, b.colors)
    assert glb.stats(before)["triangles"] == glb.stats(after)["triangles"]


def test_embedded_texture_metadata_is_removed_and_texture_pixels_survive(tmp_path):
    before = textured_glb(tmp_path / "texture.glb", png=png(private=True))
    after = tmp_path / "clean.glb"; after.write_bytes(curator.glb_bytes(before.read_bytes()))
    doc, binary = glb.read_glb(after)
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    normalized = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    curator.validate_png(normalized)
    assert b"private" not in normalized
    with Image.open(io.BytesIO(normalized)) as a, Image.open(io.BytesIO(png())) as b:
        assert a.tobytes() == b.tobytes()
    for shot in [{"name": "front", "cam_pos": [0, 0, -1], "cam_target": [0, 0, 0], "ortho": .12}]:
        a = render.render(render.load(before, textured=True), shot, width=64, height=64)
        b = render.render(render.load(after, textured=True), shot, width=64, height=64)
        np.testing.assert_array_equal(a, b)


def test_shared_image_geometry_view_is_refused_instead_of_corrupting_geometry(tmp_path):
    path = textured_glb(tmp_path / "base.glb", png=png())
    doc, blob = glb.read_glb(path)
    doc["images"][0]["bufferView"] = doc["accessors"][0]["bufferView"]
    doc["buffers"] = []
    path = _pack(doc, bytearray(blob), path)
    with pytest.raises(ValueError, match="share a buffer view"):
        curator.glb_bytes(path.read_bytes())


@pytest.mark.parametrize("mutate", ["uri", "extension", "unknown", "trailing"])
def test_unsupported_or_external_glb_content_is_refused(tmp_path, mutate):
    path = write_cubes(tmp_path / "base.glb", [cube()])
    doc, blob = glb.read_glb(path)
    if mutate == "uri": doc["buffers"][0]["uri"] = "external.bin"
    if mutate == "extension": doc["extensionsUsed"] = ["UNKNOWN_extension"]
    if mutate == "unknown": doc["customPrivateData"] = "secret"
    doc["buffers"][0]["byteLength"] = len(blob)
    payload = json.dumps(doc).encode(); payload += b" " * (-len(payload) % 4)
    raw = struct.pack("<III", glb.GLB_MAGIC, 2, 28 + len(payload) + len(blob)) + struct.pack("<II", len(payload), glb.CHUNK_JSON) + payload + struct.pack("<II", len(blob), glb.CHUNK_BIN) + blob
    if mutate == "trailing": raw += b"private"
    with pytest.raises(ValueError):
        curator.glb_bytes(raw)


def test_package_is_explicit_sealed_and_changes_invalidate_it(tmp_path):
    package = draft(tmp_path)
    original = (tmp_path / "private-image.png").read_bytes()
    with pytest.raises(ValueError, match="unsealed"):
        curator.validate_files(curator.package_files(package))
    curator.seal(package)
    assert set(curator.package_files(package)) == {"README.md", "summary.json", "release.json", "image-01.png", "model-01.glb"}
    curator.validate_files(curator.package_files(package))
    assert (tmp_path / "private-image.png").read_bytes() == original
    (package / "image-01.png").write_bytes(curator.image_bytes(png()))
    # The pixels may be the same, so change one visible pixel before checking the digest.
    im = Image.new("RGB", (8, 6), "red"); im.save(package / "image-01.png")
    with pytest.raises(ValueError, match="reviewed bytes changed"):
        curator.validate_files(curator.package_files(package))


def test_raw_or_extra_package_files_are_refused(tmp_path):
    package = draft(tmp_path, assets=False)
    (package / "raw-report.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        curator.seal(package)


def test_duplicate_json_fields_cannot_hide_raw_content_in_a_summary(tmp_path):
    package = draft(tmp_path, assets=False)
    path = package / "summary.json"
    path.write_bytes(b'{"notes":["hidden raw content"],' + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="noncanonical"):
        curator.seal(package)


def test_install_updates_only_selected_paths_and_keeps_raw_runs_ignored(tmp_path):
    package = draft(tmp_path)
    curator.seal(package)
    repo = tmp_path / "public"; repo.mkdir()
    git(repo, "init", "-q")
    script = repo / "skill/mesh-jig-publish-results/scripts/curate.py"; script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    guard = repo / "tools/check_publication.py"; guard.parent.mkdir()
    source_guard = ROOT / "publication/check_publication.py"
    if not source_guard.exists(): source_guard = ROOT / "tools/check_publication.py"
    shutil.copyfile(source_guard, guard)
    source_ignore = ROOT / "publication/gitignore"
    if not source_ignore.exists(): source_ignore = ROOT / ".gitignore"
    shutil.copyfile(source_ignore, repo / ".gitignore")
    names = [".gitignore", "PUBLIC_FILES.txt", str(script.relative_to(repo)).replace("\\", "/"), "tools/check_publication.py"]
    (repo / "PUBLIC_FILES.txt").write_text("\n".join(sorted(names)) + "\n")
    commit(repo)
    hook = curator.protect(repo)
    assert "--pre-push" in hook.read_text()
    raw = repo / "local-work/raw-run"; raw.mkdir(parents=True)
    (raw / "transcript.jsonl").write_text("private log")
    assert not git(repo, "status", "--porcelain").strip()
    installed = curator.install(package, repo, "example-first-five")
    assert len(list(installed.iterdir())) == 5
    assert b"showcase/example-first-five/model-01.glb" in git(repo, "ls-files", "--others", "--exclude-standard")
    assert b"local-work" not in git(repo, "status", "--porcelain")
    check = subprocess.run([sys.executable, str(guard), "HEAD"], cwd=repo, capture_output=True)
    assert check.returncode == 0, check.stderr.decode()
    commit(repo)
    check = subprocess.run([sys.executable, str(guard), "HEAD"], cwd=repo, capture_output=True)
    assert check.returncode == 0, check.stderr.decode()
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-q", str(remote))
    pushed = subprocess.run(["git", "-c", "safe.directory=" + repo.as_posix(), "push", "--dry-run", str(remote), "HEAD:refs/heads/main"], cwd=repo, capture_output=True)
    assert pushed.returncode == 0, pushed.stderr.decode()
    assert b"Publication check passed" in pushed.stdout
    with pytest.raises(ValueError): curator.install(package, repo, "example-first-five")


def test_protect_preserves_a_custom_hook_and_quotes_paths_with_spaces(tmp_path):
    repo = tmp_path / "public checkout"; repo.mkdir()
    git(repo, "init", "-q")
    guard = repo / "tools/check_publication.py"; guard.parent.mkdir(); guard.write_text("pass\n")
    (repo / "PUBLIC_FILES.txt").write_text("tools/check_publication.py\n")
    hook = curator.protect(repo)
    assert "'" in hook.read_text() and "public checkout" in hook.read_text()
    hook.write_text("#!/bin/sh\necho existing\n")
    with pytest.raises(ValueError, match="existing pre-push hook"):
        curator.protect(repo)
    assert "echo existing" in hook.read_text()
