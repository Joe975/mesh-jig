"""GLB stats and the model contract gate, numpy only.

No trimesh: a GLB is a JSON chunk and a binary chunk, and the gate needs positions, indices, COLOR_0, TEXCOORD_0
and the node transforms, nothing else.

The `model` block of jig.json is the contract a build is held to:
    script            model script name (defines build(), reads VARIANT); default model.py
    variants          fit names, e.g. ["gun", "missile"]; [] or absent = one GLB
    glb               output name pattern, "{variant}" substituted, e.g. "ship_{variant}.glb"
    join              the runner joins every mesh into one object (single-mesh games)
    length_m, length_tolerance   extent along the forward axis, relative tolerance (0.05 = 5 %)
    span_m, span_tolerance       extent along X, relative tolerance (default 0.10); when set it replaces the
                      "no wider than long" orientation check (some craft are wider than they are long)
    forward           "-Z" (the GLB's forward); without span_m the longest horizontal extent must lie along Z
    max_triangles, min_triangles
    single_mesh       exactly one mesh node
    vertex_colors     every primitive carries COLOR_0
    min_colors        distinct vertex colours at least this many (a palette, not one flat colour)
    textured          the textured contract: every primitive carries TEXCOORD_0 and a material with a base colour
                      texture, the GLB embeds at least one image, and the atlas file (`atlas`, default atlas.png)
                      sits beside the script
    texture_mode      "multiply" (with textured + vertex_colors): albedo is COLOR_0 x the texture; absent: the
                      texture alone
    judge_variant     the variant that is rendered and measured (default: the first)
    shots             cameras, replacing render.DEFAULT_SHOTS
    placement_min, resemblance_min, profile_mae_max   optional gates on the measures (a project states them as
                      `criteria.gates`, see mesh_jig.criteria, which writes them here)
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
MAX_DEGENERATE_FRACTION = 0.01
MAX_ORIGIN_OFFSET = 0.25     # bounds centre within this fraction of the length from the node origin
REPORT = "glb_stats.json"


def read_glb(path: str | Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError("file too short for a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB (bad magic)")
    if version != 2:
        raise ValueError(f"GLB version {version}, want 2")
    off, gltf, blob = 12, None, b""
    while off + 8 <= min(length, len(data)):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == CHUNK_JSON:
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == CHUNK_BIN:
            blob = chunk
        off += 8 + clen          # chunk lengths are already padded to 4 bytes
    if gltf is None:
        raise ValueError("GLB has no JSON chunk")
    return gltf, blob


def accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    n = NCOMP[acc["type"]]
    dtype = np.dtype(COMPONENT[acc["componentType"]])
    count = acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, n), dtype=np.float64)
    view = gltf["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * n
    if stride == dtype.itemsize * n:
        rows = np.frombuffer(blob, dtype=dtype, count=count * n, offset=start).reshape(count, n)
    else:   # interleaved: a record per vertex, one field per component
        rec = np.dtype({"names": [f"c{i}" for i in range(n)], "formats": [dtype] * n,
                        "offsets": [i * dtype.itemsize for i in range(n)], "itemsize": stride})
        rows = np.frombuffer(blob, dtype=rec, count=count, offset=start)
        rows = np.stack([rows[f"c{i}"] for i in range(n)], axis=1)
    out = rows.astype(np.float64)
    if acc.get("normalized") and dtype.kind in "iu":
        out /= float(np.iinfo(dtype).max)
    return out


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T   # glTF is column-major
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(node.get("rotation", [0, 0, 0, 1])) * np.array(node.get("scale", [1, 1, 1]))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def mesh_nodes(gltf: dict):
    """(node index, world matrix) for every node carrying a mesh, walking the default scene."""
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes") or [{"nodes": list(range(len(nodes)))}]
    roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    stack = [(i, np.eye(4)) for i in reversed(roots)]
    while stack:   # depth-first in scene order
        i, parent = stack.pop()
        world = parent @ node_matrix(nodes[i])
        if "mesh" in nodes[i]:
            yield i, world
        stack += [(c, world) for c in reversed(nodes[i].get("children", []))]


def stats(path: str | Path) -> dict:
    gltf, blob = read_glb(path)
    names, tris, degenerate, prims, colours = [], 0, 0, 0, set()
    zero_area, near_degenerate = 0, 0
    all_pts, missing_colour, nonfinite = [], 0, False
    textured_prims, missing_uv = 0, 0
    mats = gltf.get("materials", [])
    for ni, world in mesh_nodes(gltf):
        node = gltf["nodes"][ni]
        mesh = gltf["meshes"][node["mesh"]]
        names.append(node.get("name") or mesh.get("name") or f"node{ni}")
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            prims += 1
            pos = accessor(gltf, blob, prim["attributes"]["POSITION"])
            if not np.isfinite(pos).all():
                nonfinite = True
                continue
            pts = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3]
            all_pts.append(pts)
            idx = accessor(gltf, blob, prim["indices"]).astype(np.int64).ravel() if "indices" in prim else np.arange(len(pos))
            tri = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            tris += len(tri)
            if len(tri):
                a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
                area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
                scale = max(float(np.ptp(pts, axis=0).max()), 1e-9)
                small = area < (scale * 1e-5) ** 2
                degenerate += int(small.sum())
                zero_area += int((area == 0).sum())
                near_degenerate += int((small & (area > 0)).sum())
            if "COLOR_0" in prim["attributes"]:
                col = accessor(gltf, blob, prim["attributes"]["COLOR_0"])[:, :3]
                colours.update(map(tuple, np.round(col, 2)))
            else:
                missing_colour += 1
            if "TEXCOORD_0" not in prim["attributes"]:
                missing_uv += 1
            mi = prim.get("material")
            if mi is not None and mi < len(mats) and "baseColorTexture" in (mats[mi].get("pbrMetallicRoughness") or {}):
                textured_prims += 1
    pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))
    lo = pts.min(axis=0).tolist() if len(pts) else [0, 0, 0]
    hi = pts.max(axis=0).tolist() if len(pts) else [0, 0, 0]
    return {"file": Path(path).name, "mesh_nodes": names, "primitives": prims, "triangles": tris,
            "degenerate_triangles": degenerate, "nonfinite": nonfinite,
            "zero_area_triangles": zero_area, "near_degenerate_triangles": near_degenerate,
            "bounds_min": [round(v, 6) for v in lo], "bounds_max": [round(v, 6) for v in hi],
            "size": [round(h - l, 6) for l, h in zip(lo, hi)],
            "centre": [round((h + l) / 2, 6) for l, h in zip(lo, hi)],
            "primitives_without_color0": missing_colour, "distinct_colors": len(colours),
            "materials": len(mats), "images": len(gltf.get("images", [])),
            "primitives_without_texcoord0": missing_uv, "textured_primitives": textured_prims}


def check(s: dict, spec: dict) -> list[str]:
    """Contract issues for one GLB's stats; [] when it passes."""
    issues = []
    name = s["file"]
    if s["triangles"] == 0:
        return [f"{name}: no triangles"]
    if s["nonfinite"]:
        issues.append(f"{name}: non-finite vertex positions")
    if spec.get("single_mesh") and len(s["mesh_nodes"]) != 1:
        issues.append(f"{name}: {len(s['mesh_nodes'])} mesh nodes {s['mesh_nodes'][:6]}, the contract wants ONE mesh (join the parts)")
    if spec.get("max_triangles") and s["triangles"] > spec["max_triangles"]:
        issues.append(f"{name}: {s['triangles']} triangles > budget {spec['max_triangles']}")
    if spec.get("min_triangles") and s["triangles"] < spec["min_triangles"]:
        issues.append(f"{name}: only {s['triangles']} triangles (< {spec['min_triangles']}); too crude to read as the reference")
    if s["degenerate_triangles"] > MAX_DEGENERATE_FRACTION * s["triangles"]:
        detail = (f" ({s['zero_area_triangles']} zero-area, {s['near_degenerate_triangles']} tiny positive-area)"
                  if "zero_area_triangles" in s and "near_degenerate_triangles" in s else "")
        issues.append(f"{name}: {s['degenerate_triangles']} degenerate triangles of {s['triangles']}{detail}")
    sx, sy, sz = s["size"]
    dimensions = dict(spec.get("axis_extents_m") or {})
    if spec.get("height_m"):
        dimensions["Y"] = spec["height_m"]
    for axis, target in dimensions.items():
        extent = s["size"]["XYZ".index(axis)]
        tol = float(spec.get("height_tolerance", 0.05) if axis == "Y" and spec.get("height_m")
                    else (spec.get("axis_tolerances") or {}).get(axis, 0.05))
        if abs(extent - target) > tol * target:
            issues.append(f"{name}: extent along {axis} is {extent:.4f} m, contract {target:.4f} m +/- {tol:.0%}")
    if spec.get("length_m"):
        want, tol = float(spec["length_m"]), float(spec.get("length_tolerance", 0.05))
        if abs(sz - want) > tol * want:
            issues.append(f"{name}: length along Z is {sz:.4f} m, contract {want:.4f} m +/- {tol:.0%} "
                          f"(size x/y/z = {sx:.4f}/{sy:.4f}/{sz:.4f})")
        off = math.hypot(s["centre"][0], s["centre"][2])
        if off > MAX_ORIGIN_OFFSET * want:
            issues.append(f"{name}: bounds centre {s['centre']} is {off:.4f} m from the origin; keep the model centred on it")
    if spec.get("span_m"):
        want, tol = float(spec["span_m"]), float(spec.get("span_tolerance", 0.10))
        if abs(sx - want) > tol * want:
            issues.append(f"{name}: span along X is {sx:.4f} m, contract {want:.4f} m +/- {tol:.0%} "
                          f"(size x/y/z = {sx:.4f}/{sy:.4f}/{sz:.4f})")
    if spec.get("forward") in ("-Z", "+Z") and not spec.get("height_m"):
        # with a span wider than the length every builder shrank its wings to pass the heuristic below; length_m
        # already catches a model lying along X, so an explicit span replaces it
        if not spec.get("span_m") and sx > sz * 1.02:
            issues.append(f"{name}: wider along X ({sx:.4f}) than long along Z ({sz:.4f}); the nose must point along -Z")
        if sy > max(sx, sz):
            issues.append(f"{name}: tallest along Y ({sy:.4f}); Y is up, the model lies along Z")
    if spec.get("vertex_colors") and s["primitives_without_color0"]:
        issues.append(f"{name}: {s['primitives_without_color0']} primitive(s) without COLOR_0; colour comes from the vertex "
                      "colour attribute, not materials")
    if spec.get("min_colors") and not s["primitives_without_color0"] and s["distinct_colors"] < spec["min_colors"]:
        issues.append(f"{name}: {s['distinct_colors']} distinct vertex colour(s), want at least {spec['min_colors']} (the palette)")
    if spec.get("textured"):
        if not s.get("images"):
            issues.append(f"{name}: textured contract but the GLB embeds no image (build with an atlas)")
        if s.get("primitives_without_texcoord0"):
            issues.append(f"{name}: {s['primitives_without_texcoord0']} primitive(s) without TEXCOORD_0; the script must "
                          "author a UV layer or the atlas renders as one texel")
        if s.get("textured_primitives", 0) < s["primitives"]:
            issues.append(f"{name}: {s['primitives'] - s.get('textured_primitives', 0)} primitive(s) whose material has no "
                          "base colour texture")
    return issues


def script_name(spec: dict) -> str:
    return str(spec.get("script") or "model.py")


def atlas_name(spec: dict) -> str:
    return str(spec.get("atlas") or "atlas.png")


TEXTURE_SOURCES = ("plan", "script")
TEXTURE_SIZES = (256, 4096)


def texture_from(spec: dict) -> str:
    """Where a textured contract's atlas comes from: "plan" (an atlas plan or a PNG handed to the build, the
    script authoring the UVs) or "script" (the script's `paint(name, position, normal)`, the runner unwrapping and
    painting the atlas: see mesh_jig.texel_bake). "" on a contract that is not textured."""
    if not spec.get("textured"):
        return ""
    return "script" if spec.get("texture_from") == "script" else "plan"


def texture_size(spec: dict) -> int:
    """The side of a script-painted atlas in pixels. Default 1024."""
    return int(spec.get("texture_size") or 1024)


def painted_name(glb_name: str) -> str:
    """The atlas the runner paints for one variant's GLB, beside it."""
    return Path(glb_name).stem + "_atlas.png"


def blend_name(glb_name: str) -> str:
    """The Blender file saved beside a GLB on a contract with `blend`."""
    return Path(glb_name).stem + ".blend"


def texture_mode(spec: dict) -> str:
    """How a model is coloured: "vertex" (COLOR_0 only), "atlas" (the texture only) or "multiply" (COLOR_0 x the
    texture: vertex colour everywhere, paint where the atlas is not white)."""
    if not spec.get("textured"):
        return "vertex"
    return "multiply" if spec.get("texture_mode") == "multiply" else "atlas"


def variant_names(spec: dict) -> list[str]:
    return list(spec.get("variants") or [""])


def expected_files(spec: dict, key: str = "glb") -> dict[str, str]:
    """variant -> file name for the pattern at spec[key]."""
    pattern = spec.get(key) or ""
    return {v: pattern.replace("{variant}", v) for v in variant_names(spec)} if pattern else {}


def judged_variant(spec: dict) -> str:
    return str(spec.get("judge_variant") or variant_names(spec)[0])


def judged_glb(spec: dict) -> str:
    """The file name of the variant that is rendered and measured."""
    return expected_files(spec).get(judged_variant(spec), "")


def check_dir(out_dir: str | Path, spec: dict) -> tuple[list[str], dict]:
    """Every variant's GLB under out_dir against the spec -> (issues, {file: stats})."""
    out_dir = Path(out_dir)
    issues, detail = [], {}
    if not (out_dir / script_name(spec)).is_file():
        issues.append(f"model script {script_name(spec)} missing (the script is the source; the GLB is built from it)")
    if spec.get("textured") and not (out_dir / atlas_name(spec)).is_file():
        issues.append(f"atlas {atlas_name(spec)} missing (the textured contract binds it at build time)")
    files = expected_files(spec)
    if not files:
        return issues + ["model block names no glb pattern"], detail
    for variant, fname in files.items():
        p = out_dir / fname
        if not p.is_file():
            issues.append(f"{fname} missing (variant {variant or 'default'})")
            continue
        try:
            s = stats(p)
        except (ValueError, KeyError, IndexError, struct.error) as e:
            issues.append(f"{fname}: unreadable GLB ({e})")
            continue
        detail[fname] = s
        issues += check(s, spec)
    return issues, detail


def write_report(out_dir: str | Path, spec: dict) -> dict:
    """check_dir, written to out_dir/glb_stats.json -> {"ok", "issues", "stats"}."""
    issues, detail = check_dir(out_dir, spec)
    rec = {"ok": not issues, "issues": issues, "stats": detail}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / REPORT).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec
