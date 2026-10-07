"""GLBs written in-test, so no test needs Blender: boxes with a vertex colour per corner, and a cube with one colour
and one normal per face, wound outward."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from mesh_jig import glb

BOX_FACES = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
FACE_COLOURS = {"+x": (0.0, 0.0, 1.0), "-x": (1.0, 1.0, 0.0), "+y": (0.0, 1.0, 1.0),
                "-y": (1.0, 0.0, 1.0), "+z": (1.0, 0.0, 0.0), "-z": (0.0, 1.0, 0.0)}


def box(size, centre=(0, 0, 0)):
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = centre
    v = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
         (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
    return np.array([(x + cx, y + cy, z + cz) for x, y, z in v], dtype=np.float32), np.array(BOX_FACES, dtype=np.uint16)


def _pack(gltf: dict, blob: bytearray, path) -> Path:
    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"].append({"byteLength": len(blob)})
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    out = struct.pack("<III", glb.GLB_MAGIC, 2, 12 + 8 + len(js) + 8 + len(blob))
    out += struct.pack("<II", len(js), glb.CHUNK_JSON) + js + struct.pack("<II", len(blob), glb.CHUNK_BIN) + bytes(blob)
    Path(path).write_bytes(out)
    return Path(path)


def _adder(gltf: dict, blob: bytearray):
    def add(arr, gl_type, comp, target=None, stride=None):
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        if stride:
            view["byteStride"] = stride
        blob.extend(arr.tobytes())
        gltf["bufferViews"].append(view)
        gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1, "componentType": comp,
                                  "count": len(arr), "type": gl_type})
        return len(gltf["accessors"]) - 1
    return add


def _empty() -> dict:
    return {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
            "accessors": [], "bufferViews": [], "buffers": []}


def write_glb(path, parts, *, colours=True, interleave=False) -> Path:
    """parts: [(name, (verts, faces), translation)]; each part a mesh node. COLOR_0 cycles three colours."""
    gltf, blob = _empty(), bytearray()
    add = _adder(gltf, blob)
    palette = np.array([[0.48, 0.05, 0.03, 1], [0.8, 0.76, 0.66, 1], [0.03, 0.03, 0.04, 1]], dtype=np.float32)
    for name, (verts, faces), t in parts:
        attrs = {}
        if interleave:
            n = len(verts)
            rec = np.zeros(n, dtype=[("p", np.float32, 3), ("pad", np.float32, 1)])
            rec["p"] = verts
            attrs["POSITION"] = add(np.frombuffer(rec.tobytes(), dtype=np.uint8), "VEC3", 5126, 34962, stride=16)
            gltf["accessors"][-1]["count"] = n
        else:
            attrs["POSITION"] = add(verts, "VEC3", 5126, 34962)
        if colours:
            attrs["COLOR_0"] = add(palette[np.arange(len(verts)) % 3], "VEC4", 5126, 34962)
        idx = add(faces.ravel(), "SCALAR", 5123, 34963)
        gltf["meshes"].append({"name": name, "primitives": [{"attributes": attrs, "indices": idx}]})
        gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"]) - 1, "translation": list(t)})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)
    return _pack(gltf, blob, path)


def cube(size=0.5, centre=(0.0, 0.0, 0.0), colours=FACE_COLOURS):
    """Four vertices per face (so each face has its own colour and normal), counter-clockwise seen from outside.
    `size` is one number or (x, y, z)."""
    h = np.array([size] * 3 if np.isscalar(size) else size, dtype=np.float64) / 2
    c = np.array(centre, dtype=np.float64)
    pos, col, nrm, idx = [], [], [], []
    for key, rgb in colours.items():
        axis = "xyz".index(key[1])
        s = 1.0 if key[0] == "+" else -1.0
        n = np.zeros(3)
        n[axis] = s
        u = np.zeros(3)
        u[(axis + 1) % 3] = 1.0
        v = np.cross(n, u)
        base = len(pos)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pos.append(c + h * (n + a * u + b * v))
            col.append((*rgb, 1.0))
            nrm.append(n)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return (np.array(pos, np.float32), np.array(col, np.float32), np.array(nrm, np.float32), np.array(idx, np.uint16))


def write_cubes(path, meshes) -> Path:
    """meshes: [(positions, colours, normals, indices)] as `cube` returns them, one node each."""
    gltf, blob = _empty(), bytearray()
    add = _adder(gltf, blob)
    for pos, col, nrm, idx in meshes:
        prim = {"attributes": {"POSITION": add(pos, "VEC3", 5126), "COLOR_0": add(col, "VEC4", 5126),
                               "NORMAL": add(nrm, "VEC3", 5126)}, "indices": add(idx, "SCALAR", 5123)}
        gltf["meshes"].append({"primitives": [prim]})
        gltf["nodes"].append({"mesh": len(gltf["meshes"]) - 1})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)
    return _pack(gltf, blob, path)


def textured_glb(path, *, uvs=True, image=True, textured_material=True, png: bytes | None = None,
                 size=(0.066, 0.026, 0.080)) -> Path:
    """One box mesh with COLOR_0, optional TEXCOORD_0, an optional embedded PNG and a material that reads it."""
    verts, faces = box(size)
    gltf, blob = _empty(), bytearray()
    gltf.update({"materials": [], "images": [], "textures": [], "samplers": [{}]})
    add = _adder(gltf, blob)
    attrs = {"POSITION": add(verts, "VEC3", 5126, 34962),
             "COLOR_0": add(np.tile(np.array([[0.5, 0.1, 0.1, 1]], np.float32), (len(verts), 1)), "VEC4", 5126, 34962)}
    if uvs:
        attrs["TEXCOORD_0"] = add(np.zeros((len(verts), 2), np.float32), "VEC2", 5126, 34962)
    idx = add(faces.ravel(), "SCALAR", 5123, 34963)
    gltf["meshes"].append({"name": "K", "primitives": [{"attributes": attrs, "indices": idx, "material": 0}]})
    gltf["nodes"].append({"name": "K", "mesh": 0})
    gltf["scenes"][0]["nodes"].append(0)
    if image:
        data = png or (b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
        while len(blob) % 4:
            blob.append(0)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        gltf["images"].append({"bufferView": len(gltf["bufferViews"]) - 1, "mimeType": "image/png"})
        gltf["textures"].append({"sampler": 0, "source": 0})
    mat = {"name": "atlas", "pbrMetallicRoughness": {}}
    if textured_material and image:
        mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    gltf["materials"].append(mat)
    return _pack(gltf, blob, path)
