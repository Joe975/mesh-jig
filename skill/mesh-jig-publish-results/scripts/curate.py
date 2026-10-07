"""Build and seal a small explicit selection of results; raw run records never leave local storage."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import zlib

from PIL import Image

SCHEMA = 1
MEDIA = re.compile(r"(?:image-[0-9]{2}\.png|model-[0-9]{2}\.glb)\Z")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
PRIVATE = re.compile(
    r"[A-Za-z]:[\\/]|/(?:home|Users)/|/(?:mnt/)?[a-z]/Users/|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|://|"
    r"sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|BEGIN .*PRIVATE KEY")
METRICS = {
    "evaluations": "evaluations", "runtime_seconds": "wall_s", "driver_turns": "turns",
    "prompt_tokens": "prompt_tokens", "cached_prompt_tokens": "cached_tokens",
    "cache_write_tokens": "cache_write_tokens", "completion_tokens": "completion_tokens",
    "reasoning_tokens": "reasoning_tokens",
}
GLTF_KEYS = set("""accessors animations asset bufferViews buffers cameras images materials meshes nodes samplers
scene scenes skins textures version minVersion bufferView byteOffset componentType normalized count type max min
sparse indices values byteLength byteStride target channels sampler input output interpolation node path
camera mesh skin matrix translation rotation scale weights children inverseBindMatrices skeleton joints
perspective orthographic aspectRatio yfov zfar znear xmag ymag primitives attributes material mode targets
POSITION NORMAL TANGENT pbrMetallicRoughness baseColorFactor baseColorTexture metallicFactor roughnessFactor
metallicRoughnessTexture normalTexture occlusionTexture emissiveTexture emissiveFactor alphaMode alphaCutoff
doubleSided index texCoord strength magFilter minFilter wrapS wrapT source mimeType buffer""".split())
GLTF_KEYS.update(f"{kind}_{i}" for kind in ("TEXCOORD", "COLOR", "JOINTS", "WEIGHTS") for i in range(8))
STRING_VALUES = {
    "version": {"2.0"}, "minVersion": {"2.0"}, "mimeType": {"image/png", "image/jpeg"},
    "type": {"SCALAR", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4", "perspective", "orthographic"},
    "alphaMode": {"OPAQUE", "MASK", "BLEND"}, "interpolation": {"LINEAR", "STEP", "CUBICSPLINE"},
    "path": {"translation", "rotation", "scale", "weights"},
}


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text(value, label, limit=500):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(label + " must be short public text")
    if PRIVATE.search(value) or any(ord(c) < 32 or c in "<>[]`|" for c in value):
        raise ValueError(label + " contains private paths, credentials, contact details or markup")
    return value.strip()


def number(value, label, integer=False, unit=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(label + " must be a finite nonnegative number or null")
    if integer and (not isinstance(value, int)):
        raise ValueError(label + " must be an integer or null")
    if unit and value > 1:
        raise ValueError(label + " must be at most one")
    return value


def summary(report, *, title, driver, scope, cost_kind, effort=None, notes=(), evaluation=None):
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    if cost_kind not in {"reported", "estimated", "unavailable"}:
        raise ValueError("select cost provenance")
    cost = number(report.get("cost_usd"), "cost")
    if cost_kind == "unavailable":
        cost = None
    elif cost is None:
        raise ValueError("cost is missing; use unavailable")
    metrics = {dest: number(report.get(source), dest, integer=dest != "runtime_seconds")
               for dest, source in METRICS.items()}
    calls = report.get("calls")
    metrics["model_calls"] = len(calls) if isinstance(calls, list) else number(report.get("model_calls"), "model_calls", integer=True)
    best = report.get("best") or {}
    if not isinstance(best, dict):
        raise ValueError("best must be an object or null")
    result = {"numeric_score": number(best.get("numeric_score"), "numeric score", unit=True),
              "gates_passed": None, "gates_total": None}
    if evaluation is not None:
        if not isinstance(evaluation, dict) or evaluation.get("ok") is not True:
            raise ValueError("selected evaluation must have succeeded")
        result["numeric_score"] = number((evaluation.get("numeric") or {}).get("numeric_score"), "numeric score", unit=True)
        gates = evaluation.get("gates")
        if isinstance(gates, list):
            acceptance = [g for g in gates if isinstance(g, dict) and g.get("observation") is not True and isinstance(g.get("passed"), bool)]
            result["gates_passed"] = sum(g["passed"] for g in acceptance)
            result["gates_total"] = len(acceptance)
    done = report.get("done")
    if done is not None and not isinstance(done, bool):
        raise ValueError("done must be boolean or null")
    return {"schema_version": SCHEMA, "title": text(title, "title", 120), "model": text(report.get("model"), "model", 120),
            "driver": text(driver, "driver", 120), "scope": text(scope, "scope"),
            "effort": text(effort, "effort", 40) if effort else None, "completed": done,
            "metrics": metrics, "cost": {"amount_usd": cost, "kind": cost_kind}, "result": result,
            "notes": [text(note, "note") for note in notes]}


def validate_summary(doc):
    if not isinstance(doc, dict) or set(doc) != {"schema_version", "title", "model", "driver", "scope", "effort", "completed", "metrics", "cost", "result", "notes"}:
        raise ValueError("unexpected summary fields")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != SCHEMA or not isinstance(doc["metrics"], dict) or set(doc["metrics"]) != {*METRICS, "model_calls"}:
        raise ValueError("unexpected metrics schema")
    if not isinstance(doc["cost"], dict) or set(doc["cost"]) != {"amount_usd", "kind"}:
        raise ValueError("unexpected cost fields")
    if not isinstance(doc["result"], dict) or set(doc["result"]) != {"numeric_score", "gates_passed", "gates_total"}:
        raise ValueError("unexpected result fields")
    if not isinstance(doc["notes"], list):
        raise ValueError("notes must be an array")
    report = {source: doc["metrics"][dest] for dest, source in METRICS.items()}
    report.update(model=doc["model"], done=doc["completed"], model_calls=doc["metrics"]["model_calls"],
                  cost_usd=doc["cost"]["amount_usd"], best={"numeric_score": doc["result"]["numeric_score"]})
    rebuilt = summary(report, title=doc["title"], driver=doc["driver"], scope=doc["scope"],
                      cost_kind=doc["cost"]["kind"], effort=doc["effort"], notes=doc["notes"])
    passed = number(doc["result"]["gates_passed"], "gates passed", integer=True)
    total = number(doc["result"]["gates_total"], "gates total", integer=True)
    if (passed is None) != (total is None) or (passed is not None and passed > total):
        raise ValueError("invalid gate counts")
    rebuilt["result"].update(gates_passed=passed, gates_total=total)
    if rebuilt != doc:
        raise ValueError("summary is not canonical")


def image_bytes(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
        fresh = Image.frombytes(mode, image.size, image.convert(mode).tobytes())
        output = io.BytesIO()
        fresh.save(output, format="PNG")
        return output.getvalue()


def validate_png(raw):
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("expected normalized PNG")
    offset, kinds, compressed, header = 8, [], bytearray(), None
    while offset + 12 <= len(raw):
        length, = struct.unpack_from(">I", raw, offset)
        end = offset + 12 + length
        if end > len(raw):
            raise ValueError("truncated PNG")
        kind = raw[offset + 4:offset + 8]
        if kind not in {b"IHDR", b"IDAT", b"IEND"}:
            raise ValueError("PNG contains metadata or unsupported chunks")
        if zlib.crc32(raw[offset + 4:end - 4]) != struct.unpack_from(">I", raw, end - 4)[0]:
            raise ValueError("PNG checksum mismatch")
        kinds.append(kind)
        if kind == b"IHDR":
            header = raw[offset + 8:end - 4]
        elif kind == b"IDAT":
            compressed.extend(raw[offset + 8:end - 4])
        elif length:
            raise ValueError("invalid PNG end chunk")
        offset = end
    if offset != len(raw) or kinds[:1] != [b"IHDR"] or kinds[-1:] != [b"IEND"] or kinds.count(b"IHDR") != 1 or kinds.count(b"IEND") != 1:
        raise ValueError("invalid PNG structure")
    if header is None or len(header) != 13:
        raise ValueError("invalid PNG header")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
    if depth != 8 or color not in {2, 6} or compression or filtering or interlace:
        raise ValueError("PNG is not a normalized raster")
    stream = zlib.decompressobj()
    try:
        pixels = stream.decompress(compressed) + stream.flush()
    except zlib.error:
        raise ValueError("invalid PNG pixel stream") from None
    if not stream.eof or stream.unused_data or stream.unconsumed_tail or len(pixels) != height * (width * (3 if color == 2 else 4) + 1):
        raise ValueError("PNG contains trailing or invalid compressed data")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError("PNG must be RGB or RGBA")


def glb_bytes(raw):
    if len(raw) < 20 or struct.unpack_from("<III", raw) != (0x46546C67, 2, len(raw)):
        raise ValueError("expected a complete GLB 2 file")
    chunks, offset = [], 12
    while offset + 8 <= len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        end = offset + 8 + length
        if length % 4 or end > len(raw):
            raise ValueError("invalid GLB chunk length")
        chunks.append((kind, raw[offset + 8:end])); offset = end
    if offset != len(raw) or [kind for kind, _ in chunks] not in ([0x4E4F534A], [0x4E4F534A, 0x004E4942]):
        raise ValueError("unsupported GLB chunks")
    doc = json.loads(chunks[0][1])
    binary = chunks[1][1] if len(chunks) == 2 else b""

    def clean(value, key=""):
        if isinstance(value, dict):
            out = {}
            for field, child in value.items():
                if field in {"name", "extras", "generator", "copyright"}:
                    continue
                if field in {"extensions", "extensionsUsed", "extensionsRequired"} and not child:
                    continue
                if field not in GLTF_KEYS:
                    raise ValueError("unsupported GLB field or external resource")
                out[field] = clean(child, field)
            return out
        if isinstance(value, list):
            return [clean(v, key) for v in value]
        if isinstance(value, str) and value not in STRING_VALUES.get(key, set()):
            raise ValueError("unrecognized string in GLB")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite GLB value")
        return value

    doc = clean(doc)
    if not isinstance(doc, dict) or doc.get("asset", {}).get("version") != "2.0":
        raise ValueError("invalid GLB asset")
    buffers, views = doc.get("buffers", []), doc.get("bufferViews", [])
    if len(buffers) != 1 or type(buffers[0].get("byteLength")) is not int or not 0 <= buffers[0]["byteLength"] <= len(binary):
        raise ValueError("expected one embedded GLB buffer")
    referenced = set()

    def visit(value, remap=None):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "bufferView":
                    if isinstance(child, bool) or not isinstance(child, int) or not 0 <= child < len(views):
                        raise ValueError("invalid GLB buffer view reference")
                    referenced.add(child)
                    if remap is not None:
                        value[key] = remap[child]
                else:
                    visit(child, remap)
        elif isinstance(value, list):
            for child in value:
                visit(child, remap)

    visit(doc)
    geometry_views = set()

    def geometry(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "bufferView":
                    geometry_views.add(child)
                else:
                    geometry(child)
        elif isinstance(value, list):
            for child in value:
                geometry(child)

    geometry(doc.get("accessors", []))
    image_views = {}
    for image in doc.get("images", []):
        if "bufferView" not in image or image.get("mimeType") not in {"image/png", "image/jpeg"}:
            raise ValueError("expected embedded GLB image")
        if image["bufferView"] in geometry_views:
            raise ValueError("image data cannot share a buffer view with geometry")
        image_views[image["bufferView"]] = image
    new_binary, new_views, remap = bytearray(), [], {}
    for index in sorted(referenced):
        view = views[index]
        start, length = view.get("byteOffset", 0), view.get("byteLength")
        if view.get("buffer") != 0 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (start, length)) or start + length > buffers[0]["byteLength"]:
            raise ValueError("invalid GLB buffer range")
        chunk = binary[start:start + length]
        if index in image_views:
            chunk = image_bytes(chunk)
        new_binary.extend(b"\0" * (-len(new_binary) % 4))
        new_views.append(dict(view, byteOffset=len(new_binary), byteLength=len(chunk)))
        remap[index] = len(new_views) - 1
        new_binary.extend(chunk)
    visit(doc, remap)
    for image in doc.get("images", []):
        image["mimeType"] = "image/png"
    doc["bufferViews"] = new_views
    doc["buffers"] = [{"byteLength": len(new_binary)}]
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload += b" " * (-len(payload) % 4)
    new_binary.extend(b"\0" * (-len(new_binary) % 4))
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(payload) + len(new_binary))
            + struct.pack("<II", len(payload), 0x4E4F534A) + payload
            + struct.pack("<II", len(new_binary), 0x004E4942) + new_binary)


def readme(doc, media):
    value = lambda v: "unknown" if v is None else str(v).lower() if isinstance(v, bool) else str(v)
    rows = [("Model", doc["model"]), ("Driver", doc["driver"]), ("Effort", doc["effort"]), ("Scope", doc["scope"]),
            ("Completed", doc["completed"]), *[(k.replace("_", " ").capitalize(), doc["metrics"][k]) for k in (*METRICS, "model_calls")],
            ("Cost USD (" + doc["cost"]["kind"] + ")", doc["cost"]["amount_usd"]),
            *[(k.replace("_", " ").capitalize(), doc["result"][k]) for k in ("numeric_score", "gates_passed", "gates_total")]]
    output = ["# " + doc["title"], "", "| Measure | Value |", "|---|---|"]
    output += [f"| {key} | {value(val)} |" for key, val in rows]
    output += ["", "Token counts use the report's provider conventions. Reasoning is included in completion tokens; cache counters are not extra prompt tokens. Estimated cost is not an invoiced charge.", ""]
    output += [note + "\n" for note in doc["notes"]]
    for name in sorted(media):
        output += [(f"![Selected output]({name})" if name.endswith(".png") else f"[Download selected model]({name})"), ""]
    return ("\n".join(output).rstrip() + "\n").encode()


def prepare(report_path, destination, *, images=(), models=(), evaluation_path=None, **metadata):
    if destination.exists():
        raise ValueError("destination already exists; choose a fresh draft")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path else None
    doc = summary(report, evaluation=evaluation, **metadata)
    files = {"summary.json": encode(doc)}
    for kind, paths, normalize, suffix in (("image", images, image_bytes, "png"), ("model", models, glb_bytes, "glb")):
        if len(paths) > 99:
            raise ValueError("too many selected assets")
        for index, path in enumerate(paths, 1):
            files[f"{kind}-{index:02d}.{suffix}"] = normalize(path.read_bytes())
    files["README.md"] = readme(doc, set(files) - {"summary.json"})
    validate_files(files, sealed=False)
    destination.mkdir(parents=True)
    for name, raw in files.items():
        (destination / name).write_bytes(raw)
    return files


def validate_files(files, sealed=True):
    if not {"README.md", "summary.json"} <= set(files) or any(name not in {"README.md", "summary.json", "release.json"} and not MEDIA.fullmatch(name) for name in files):
        raise ValueError("unexpected or missing showcase file")
    doc = json.loads(files["summary.json"])
    validate_summary(doc)
    if files["summary.json"] != encode(doc):
        raise ValueError("summary contains noncanonical or hidden fields")
    media = set(files) - {"README.md", "summary.json", "release.json"}
    if files["README.md"] != readme(doc, media):
        raise ValueError("README differs from the reviewed summary and selected assets")
    for name in media:
        if name.endswith(".png"):
            validate_png(files[name])
        elif glb_bytes(files[name]) != files[name]:
            raise ValueError("GLB contains metadata or is not normalized")
    if sealed:
        manifest = json.loads(files.get("release.json", b"null"))
        expected = {"schema_version": SCHEMA, "files": {name: digest(raw) for name, raw in files.items() if name != "release.json"}}
        if manifest != expected or files.get("release.json") != encode(expected):
            raise ValueError("showcase is unsealed or its reviewed bytes changed")
    return doc


def package_files(directory):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("package must be an ordinary directory")
    paths = list(directory.iterdir())
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("package contains directories or links")
    return {p.name: p.read_bytes() for p in paths}


def seal(directory):
    files = package_files(directory)
    if "release.json" in files:
        raise ValueError("already sealed; prepare a new package for changed results")
    validate_files(files, sealed=False)
    manifest = {"schema_version": SCHEMA, "files": {name: digest(raw) for name, raw in sorted(files.items())}}
    (directory / "release.json").write_bytes(encode(manifest))


def git(root, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + root.as_posix(), *args], cwd=root)


def install(directory, repo, slug):
    if not SLUG.fullmatch(slug):
        raise ValueError("use a lowercase showcase slug")
    files = package_files(directory)
    validate_files(files)
    inventory = repo / "PUBLIC_FILES.txt"
    guard = repo / "tools/check_publication.py"
    if not inventory.is_file() or not guard.is_file():
        raise ValueError("destination must be a public checkout with a publication inventory and guard")
    if git(repo, "status", "--porcelain").strip():
        raise ValueError("public checkout must be clean before installing a showcase")
    subprocess.run([sys.executable, str(guard), "HEAD"], cwd=repo, check=True)
    parent = repo / "showcase"
    target = parent / slug
    if parent.is_symlink() or target.exists():
        raise ValueError("showcase already exists or destination is a link")
    names = set(inventory.read_text(encoding="utf-8").splitlines())
    names.update(f"showcase/{slug}/{name}" for name in files)
    target.mkdir(parents=True)
    for name, raw in files.items():
        (target / name).write_bytes(raw)
    inventory.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8", newline="\n")
    return target


def protect(repo):
    if sys.prefix == sys.base_prefix:
        raise ValueError("run from the checkout virtual environment")
    guard = repo / "tools/check_publication.py"
    if not guard.is_file() or not (repo / "PUBLIC_FILES.txt").is_file():
        raise ValueError("destination must be a public checkout with a publication guard")
    hook = Path(git(repo, "rev-parse", "--git-path", "hooks/pre-push").decode().strip())
    if not hook.is_absolute():
        hook = repo / hook
    if hook.is_symlink() or not hook.resolve().is_relative_to(repo.resolve()):
        raise ValueError("hook location is outside this checkout or is a link")
    marker = "# mesh-jig publication guard"
    if hook.exists() and marker not in hook.read_text(encoding="utf-8"):
        raise ValueError("an existing pre-push hook must be integrated deliberately")
    command = " ".join(shlex.quote(path) for path in (Path(sys.executable).resolve().as_posix(), guard.resolve().as_posix()))
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\n" + marker + "\nexec " + command + " --pre-push\n", encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    return hook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--report", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    for field in ("title", "driver", "scope"):
        make.add_argument("--" + field, required=True)
    make.add_argument("--cost-kind", choices=("reported", "estimated", "unavailable"), required=True)
    make.add_argument("--effort")
    make.add_argument("--note", action="append", default=[])
    make.add_argument("--evaluation", type=Path)
    make.add_argument("--image", type=Path, action="append", default=[])
    make.add_argument("--glb", type=Path, action="append", default=[])
    stamp = sub.add_parser("seal")
    stamp.add_argument("--package", type=Path, required=True)
    copy = sub.add_parser("install")
    copy.add_argument("--package", type=Path, required=True)
    copy.add_argument("--repo", type=Path, required=True)
    copy.add_argument("--slug", required=True)
    hook = sub.add_parser("protect")
    hook.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [v for v in vars(args).values() if isinstance(v, Path)] + getattr(args, "image", []) + getattr(args, "glb", [])
    if any(not path.is_absolute() for path in paths):
        parser.error("use absolute paths")
    try:
        if args.command == "prepare":
            prepare(args.report, args.out, title=args.title, driver=args.driver, scope=args.scope, cost_kind=args.cost_kind,
                    effort=args.effort, notes=args.note, evaluation_path=args.evaluation, images=args.image, models=args.glb)
            print("Prepared selected metrics and assets. Review visible content before sealing.")
        elif args.command == "seal":
            seal(args.package)
            print("Sealed the exact selected bytes. No files committed or pushed.")
        elif args.command == "install":
            install(args.package, args.repo, args.slug)
            print("Installed the selected showcase and updated the public inventory. Review and commit explicit paths.")
        else:
            protect(args.repo)
            print("Installed the local pre-push privacy guard using this virtual environment.")
    except (OSError, ValueError, TypeError, KeyError, struct.error, subprocess.SubprocessError) as exc:
        print("Showcase operation failed: " + type(exc).__name__ + ": " + (str(exc) if isinstance(exc, ValueError) else "check the selected inputs locally"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
