"""Check reference-led surface decisions against the material on each exported primitive."""
from io import BytesIO
import math

import numpy as np
from PIL import Image

from . import glb

MAPS = {"normal_map": "normalTexture", "roughness_map": "metallicRoughnessTexture",
        "occlusion_map": "occlusionTexture", "emission_map": "emissiveTexture"}


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("material_review needs a surface decision; use empty channels to retain a finish")
    claimed = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                            for k in ("region", "cue", "check")):
            raise ValueError("each material decision needs a region, reference cue and visual check")
        channels = row.get("channels")
        if not isinstance(channels, dict) or set(channels) - (MAPS.keys() | {"roughness", "emission"}):
            raise ValueError(f"{row['region']}: unknown or missing channels")
        nodes = row.get("nodes", [])
        if not isinstance(nodes, list) or any(not isinstance(n, str) or not n.strip() for n in nodes):
            raise ValueError("nodes must be an array of mesh-node names")
        if channels and not nodes:
            raise ValueError("a selected channel needs at least one named mesh node")
        maps = row.get("maps", {})
        if not isinstance(maps, dict) or any(key not in MAPS or key not in channels or not isinstance(path, str)
                                              or not path.strip() for key, path in maps.items()):
            raise ValueError("maps must name selected map channels and their image files")
        if "emission_rgb" in row:
            rgb = row["emission_rgb"]
            if (not ("emission" in channels or "emission_map" in channels) or not isinstance(rgb, list)
                    or len(rgb) != 3 or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in rgb)
                    or not max(rgb)):
                raise ValueError("emission_rgb needs three linear 0..1 values with a nonzero component")
        if any(n in claimed for n in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError("combine decisions for the same mesh node into one entry")
        claimed.update(nodes)
        for key, value in channels.items():
            if key == "roughness":
                if type(value) not in (float, int) or not 0 <= value <= 1:
                    raise ValueError("roughness must be a factor from 0 to 1")
            elif value is not True:
                raise ValueError(f"{key} must be true when selected")


def indexed(items, i, label):
    if type(i) is not int or not 0 <= i < len(items):
        raise ValueError(f"missing {label}")
    return items[i]


def check_map(doc, blob, prim, slot):
    texture = indexed(doc.get("textures", []), slot.get("index"), "texture")
    image = indexed(doc.get("images", []), texture.get("source"), "image")
    view = indexed(doc["bufferViews"], image.get("bufferView"), "embedded image data")
    start = view.get("byteOffset", 0)
    with Image.open(BytesIO(blob[start:start + view["byteLength"]])) as img:
        img.verify()
    transform = slot.get("extensions", {}).get("KHR_texture_transform", {})
    uv_set = transform.get("texCoord", slot.get("texCoord", 0))
    attrs = prim["attributes"]
    uv = glb.accessor(doc, blob, attrs[f"TEXCOORD_{uv_set}"])
    if uv.shape != (doc["accessors"][attrs["POSITION"]]["count"], 2) or not np.isfinite(uv).all():
        raise ValueError("invalid UV coordinates")
    if prim.get("mode", 4) != 4:
        raise ValueError("material review expects triangle primitives")
    idx = glb.accessor(doc, blob, prim["indices"]).ravel() if "indices" in prim else np.arange(len(uv))
    tri = uv[idx.astype(np.int64)].reshape(-1, 3, 2)
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    if not np.any(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-12):
        raise ValueError("UVs cover no triangle area")


def check(path, rows):
    validate(rows)
    if not path.is_file():
        raise ValueError(f"missing GLB: {path}")
    doc, blob = glb.read_glb(path)
    nodes = {}
    for i, _world in glb.mesh_nodes(doc):
        node = doc["nodes"][i]
        nodes.setdefault(node.get("name"), []).append(node)
    issues = []
    for row in rows:
        for name in row.get("nodes", []):
            found = nodes.get(name, [])
            if len(found) != 1:
                issues.append(f"{name}: expected one mesh node, found {len(found)}")
                continue
            primitives = doc["meshes"][found[0]["mesh"]]["primitives"]
            if not primitives:
                issues.append(f"{name}: missing primitive")
            for number, prim in enumerate(primitives):
                for key, expected in row["channels"].items():
                    try:
                        mat = (indexed(doc.get("materials", []), prim["material"], "material")
                               if "material" in prim else {})
                        pbr = mat.get("pbrMetallicRoughness", {})
                        if key == "roughness":
                            if not math.isclose(pbr.get("roughnessFactor", 1), expected, abs_tol=1e-5):
                                raise ValueError(f"expected roughness factor {expected}")
                        elif key == "emission" or key == "emission_map":
                            if max(mat.get("emissiveFactor", [0, 0, 0])) <= 0:
                                raise ValueError("emissive factor is black")
                            if "emission_rgb" in row and not np.allclose(mat["emissiveFactor"], row["emission_rgb"]):
                                raise ValueError("emissive colour differs from emission_rgb")
                        if key in MAPS:
                            slot = (pbr if key == "roughness_map" else mat)[MAPS[key]]
                            if key == "normal_map" and slot.get("scale", 1) == 0:
                                raise ValueError("normal scale is zero")
                            if key == "occlusion_map" and slot.get("strength", 1) == 0:
                                raise ValueError("occlusion strength is zero")
                            if key == "roughness_map" and pbr.get("roughnessFactor", 1) == 0:
                                raise ValueError("roughness factor is zero")
                            check_map(doc, blob, prim, slot)
                    except (KeyError, ValueError, IndexError, OSError) as exc:
                        issues.append(f"{name} primitive {number} {key}: {exc}")
    return issues

def apply(source, output, root, rows):
    """Apply named decisions to a copy of an embedded GLB, verifying it before replacing the output."""
    import copy
    import os
    import tempfile
    from pathlib import Path
    from . import rig

    validate(rows)
    if source.resolve() == output.resolve():
        raise ValueError("choose a separate output for the finished GLB")
    if output.suffix.lower() != ".glb" or not source.is_file():
        raise ValueError("materials needs an existing input GLB and a .glb output")
    doc, binary = glb.read_glb(source)
    if any("uri" in item for key in ("buffers", "images") for item in doc.get(key, [])):
        raise ValueError("embed the input's buffers and images before applying a finish")
    blob = bytearray(binary)
    textures = {}
    meshes = doc["meshes"]
    materials = doc.setdefault("materials", [])
    for row in rows:
        channels = row["channels"]
        if not channels:
            continue
        assigned = {}
        names = set(row["nodes"])
        for node in doc["nodes"]:
            if node.get("name") not in names or "mesh" not in node:
                continue
            mesh_id = node["mesh"]
            mesh = meshes[mesh_id]
            if any(other is not node and other.get("mesh") == mesh_id for other in doc["nodes"]):
                mesh = copy.deepcopy(mesh)
                node["mesh"] = len(meshes)
                meshes.append(mesh)
            for prim in mesh["primitives"]:
                old = prim.get("material")
                if old in assigned:
                    prim["material"] = assigned[old]
                    continue
                mat = copy.deepcopy(materials[old]) if old is not None else {}
                mat["name"] = row["region"]
                pbr = mat.setdefault("pbrMetallicRoughness", {})
                if "roughness" in channels:
                    pbr["roughnessFactor"] = channels["roughness"]
                if "emission" in channels or "emission_map" in channels:
                    if "emission_rgb" in row:
                        mat["emissiveFactor"] = row["emission_rgb"]
                    elif not max(mat.get("emissiveFactor", [0, 0, 0])):
                        mat["emissiveFactor"] = [1, 1, 1]
                for key, filename in row.get("maps", {}).items():
                    path = (root / filename).resolve()
                    if not path.is_relative_to(root.resolve()) or not path.is_file():
                        raise ValueError(f"{filename}: expected an image inside the project")
                    if path not in textures:
                        data = path.read_bytes()
                        with Image.open(BytesIO(data)) as im:
                            if im.format not in ("PNG", "JPEG"):
                                raise ValueError("use a PNG or JPEG material map")
                            mime = Image.MIME[im.format]
                            im.verify()
                        blob.extend(bytes(-len(blob) % 4))
                        views = doc["bufferViews"]
                        views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(data)))
                        blob.extend(data)
                        images = doc.setdefault("images", [])
                        images.append(dict(bufferView=len(views)-1, mimeType=mime))
                        slots = doc.setdefault("textures", [])
                        slots.append(dict(source=len(images)-1))
                        textures[path] = len(slots)-1
                    target = pbr if key == "roughness_map" else mat
                    target[MAPS[key]] = {"index": textures[path]}
                assigned[old] = prim["material"] = len(materials)
                materials.append(mat)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".glb", delete=False) as temp:
        pending = Path(temp.name)
    try:
        rig.write_glb(pending, doc, blob)
        issues = check(pending, rows)
        if issues:
            raise ValueError("; ".join(issues))
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
