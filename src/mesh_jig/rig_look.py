"""Pictures of a rig for whoever writes it: cameras by name, a ruled rest pose, the skeleton, the weights, film strips.

A rig script is written in Blender metres and a GLB is drawn in glTF's, so a camera a builder has to place by hand
is placed wrongly first. Every camera here is fitted to the source model and asked for by name. The ruled views are
orthographic, so a joint's place can be read off them in the script's own axes.

Deterministic: same file, same bytes.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import glb, render, skin

ORTHO = ("front", "back", "left", "right", "top")
VIEWS = ORTHO + ("three-quarter",)
MARGIN = 1.45                 # the view's height over the model's largest extent: room for a limb to swing into
GRID_M = 0.1
INK, FAINT, AXIS = (40, 40, 40), (176, 174, 168), (120, 60, 60)
# Blender axis names of the picture's right and up in each orthographic view, and which way they run
READS = {"front": ("X", -1, "Z", 1), "back": ("X", 1, "Z", 1), "left": ("Y", -1, "Z", 1), "right": ("Y", 1, "Z", 1),
         "top": ("X", 1, "Y", 1)}
FACES = {"front": "the model faces you; its own left (X<0) is on the picture's right",
         "back": "seen from behind; its own left (X<0) is on the picture's left",
         "left": "its left side; it faces the picture's left (+Y is left)",
         "right": "its right side; it faces the picture's right (+Y is right)",
         "top": "from above; its front (+Y) is up the picture",
         "three-quarter": "from in front, its left and above"}


def to_gltf(p) -> np.ndarray:
    x, y, z = (float(v) for v in p)
    return np.array([x, z, -y])


def to_blender(points: np.ndarray) -> np.ndarray:
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def frame(lo, hi) -> tuple[np.ndarray, float]:
    """(centre in glTF axes, the height of an orthographic view that holds the model with room to move) from the
    model's bounds in Blender metres."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return to_gltf((lo + hi) / 2), float((hi - lo).max() * MARGIN)


def shot(view: str, lo, hi, aim=None, span: float | None = None) -> dict:
    """The camera called `view` for a model with these bounds (Blender metres). `aim` (a point in Blender metres)
    and `span` (metres across the picture) make it a close-up: a knee cannot be judged at the size of a whole
    figure."""
    if view not in VIEWS:
        raise ValueError(f"no view {view!r}; the views are {', '.join(VIEWS)}")
    centre, height = frame(lo, hi)
    if aim is not None:
        centre = to_gltf(aim)
    if span:
        height = float(span)
    far = 10.0 + height
    where = {"front": (0, 0, -far), "back": (0, 0, far), "left": (-far, 0, 0), "right": (far, 0, 0), "top": (0, far, 0)}
    if view in where:
        return {"name": view, "cam_pos": (centre + where[view]).tolist(), "cam_target": centre.tolist(), "ortho": height,
                "light": "camera", "aim": to_blender(centre).tolist()}
    d = height * 2.3
    return {"name": view, "cam_pos": (centre + np.array([-0.55, 0.3, -0.78]) * d).tolist(), "cam_target": centre.tolist(),
            "fov": 26, "light": "camera"}


SMALL = 4                     # pixels: a triangle whose box is no wider or taller is drawn with the rest of its kind


def rasterise(mesh: render.Mesh, view: dict, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`render.rasterise` for a square picture, with the same answer: (triangle id per pixel, barycentrics, depth).

    The measured renderer walks the triangles one at a time, which is two seconds a frame for a model of fifty
    thousand, and a build is looked at through forty frames. Nearly all of those triangles cover a pixel or two at
    this size, so the small ones are tested together, a pixel of their box at a time, and only the large ones are
    walked. A pixel goes to the nearest triangle and, at the same depth, to the lower index, as walking them gives.
    `render.rasterise` is left as it is: the measures are taken through it."""
    eye, basis, t = render.camera(view)
    cam = (mesh.tris - eye) @ basis.T
    z = cam[..., 2]
    ortho = float(view.get("ortho") or 0.0)
    if ortho:
        k = size / ortho
        sx, sy, w = size / 2 + cam[..., 0] * k, size / 2 - cam[..., 1] * k, np.ones_like(z)
    else:
        f = (size / 2) / t
        sx = size / 2 + cam[..., 0] / np.maximum(z, 1e-9) * f
        sy = size / 2 - cam[..., 1] / np.maximum(z, 1e-9) * f
        w = 1.0 / np.maximum(z, 1e-9)
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    x0 = np.maximum(np.floor(sx.min(axis=1)).astype(np.int64), 0)
    x1 = np.minimum(np.ceil(sx.max(axis=1)).astype(np.int64), size - 1)
    y0 = np.maximum(np.floor(sy.min(axis=1)).astype(np.int64), 0)
    y1 = np.minimum(np.ceil(sy.max(axis=1)).astype(np.int64), size - 1)
    keep = (z.min(axis=1) > render.NEAR) & (area < 0) & (x0 <= x1) & (y0 <= y1)
    small = keep & (x1 - x0 < SMALL) & (y1 - y0 < SMALL)
    ids = np.full(size * size, -1, dtype=np.int64)
    depth = np.full(size * size, np.inf)
    bary = np.zeros((size * size, 3))

    def cover(i, px, py):
        """For triangles i at pixel centres (px, py): which are inside, their corrected barycentrics, their depth."""
        xs, ys, a = sx[i], sy[i], area[i]
        w0 = ((xs[:, 1] - px) * (ys[:, 2] - py) - (xs[:, 2] - px) * (ys[:, 1] - py)) / a
        w1 = ((xs[:, 2] - px) * (ys[:, 0] - py) - (xs[:, 0] - px) * (ys[:, 2] - py)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        pc = np.stack([w0 * w[i, 0], w1 * w[i, 1], w2 * w[i, 2]], axis=-1)
        pc /= np.maximum(pc.sum(axis=-1, keepdims=True), 1e-12)
        return inside, pc, (pc * z[i]).sum(axis=-1)

    found = []
    for dy in range(SMALL):
        for dx in range(SMALL):
            i = np.flatnonzero(small & (x0 + dx <= x1) & (y0 + dy <= y1))
            if not len(i):
                continue
            inside, pc, d = cover(i, x0[i] + dx + 0.5, y0[i] + dy + 0.5)
            found.append((((y0[i] + dy) * size + x0[i] + dx)[inside], d[inside], i[inside], pc[inside]))
    for i in np.flatnonzero(keep & ~small):                  # the few large ones, a box of pixels each
        gx, gy = np.meshgrid(np.arange(x0[i], x1[i] + 1), np.arange(y0[i], y1[i] + 1))
        gx, gy = gx.ravel(), gy.ravel()
        inside, pc, d = cover(np.full(len(gx), i), gx + 0.5, gy + 0.5)
        found.append(((gy * size + gx)[inside], d[inside], np.full(int(inside.sum()), i), pc[inside]))
    if found:
        pixel_, d, tri, pc = (np.concatenate(part) for part in zip(*found))
        order = np.lexsort((tri, d, pixel_))                 # by pixel, then nearest, then the lower index
        first = order[np.concatenate(([True], np.diff(pixel_[order]) != 0))]
        ids[pixel_[first]], depth[pixel_[first]], bary[pixel_[first]] = tri[first], d[first], pc[first]
    return ids.reshape(size, size), bary.reshape(size, size, 3), depth.reshape(size, size)


def draw(mesh: render.Mesh, view: dict, size: int) -> Image.Image:
    ids, bary, _ = rasterise(mesh, view, size)
    _, basis, _ = render.camera(view)
    return Image.fromarray(render.shade(mesh, ids, bary, basis[2], render.lighting({"light": view.get("light", "stage")}, basis)))


def pixel(view: dict, size: int, point_gltf) -> tuple[float, float]:
    """Where a glTF point lands in an orthographic view's picture."""
    eye, basis, _ = render.camera(view)
    cam = (np.asarray(point_gltf, float) - eye) @ basis.T
    k = size / view["ortho"]
    return size / 2 + cam[0] * k, size / 2 - cam[1] * k


def rule(image: Image.Image, view: dict, lo, hi) -> Image.Image:
    """Grid lines every 0.1 m over an orthographic picture, numbered in the script's own axes (Blender metres)."""
    name = view["name"]
    if name not in READS:
        return image
    size = image.width
    across, a_sign, up, _ = READS[name]
    centre = np.asarray(view.get("aim", to_blender(frame(lo, hi)[0])), float)
    c_across, c_up = centre["XYZ".index(across)], centre["XYZ".index(up)]
    half, k = view["ortho"] / 2, size / view["ortho"]
    out = image.copy()
    d = ImageDraw.Draw(out)
    for step in range(int(np.floor((c_across - half) / GRID_M)), int(np.ceil((c_across + half) / GRID_M)) + 1):
        v = step * GRID_M
        x = size / 2 + (v - c_across) * a_sign * k
        if 0 <= x < size:
            d.line([(x, 0), (x, size)], fill=AXIS if step == 0 else FAINT, width=1)
            if step % 2 == 0:
                d.text((x + 2, size - 12), f"{v:+.1f}", fill=INK)
    for step in range(int(np.floor((c_up - half) / GRID_M)), int(np.ceil((c_up + half) / GRID_M)) + 1):
        v = step * GRID_M
        y = size / 2 - (v - c_up) * k
        if 0 <= y < size:
            d.line([(0, y), (size, y)], fill=AXIS if step == 0 else FAINT, width=1)
            if step % 2 == 0:
                d.text((2, y - 11), f"{v:+.1f}", fill=INK)
    d.text((size - 150, 2), f"{name}: across {across}, up {up} (m)", fill=INK)
    return out


def ruled_rest(source: str | Path, lo, hi, views=("front", "left", "back", "top"), size: int = 640) -> dict[str, Image.Image]:
    """The model at rest from each orthographic view, under the grid: where a joint goes is read off these."""
    mesh = skin.posed(source) if glb.read_glb(source)[0].get("skins") else render.load(source)
    return {name: rule(draw(mesh, shot(name, lo, hi), size), shot(name, lo, hi), lo, hi) for name in views}


def bone_colours(count: int) -> np.ndarray:
    """A colour a bone, far apart for neighbours in the list (the golden angle round the hue circle), sRGB 0..1."""
    return np.array([colorsys.hsv_to_rgb((0.11 + i * 0.618034) % 1.0, 0.85 if i % 2 else 0.6, 0.95 if i % 3 else 0.7)
                     for i in range(count)])


def corner_weights(doc: dict, blob: bytes) -> np.ndarray:
    """(triangles, 3, joints): each corner's weight on every joint of its skin, in `render.load`'s triangle order."""
    count = max((len(s["joints"]) for s in doc.get("skins", [])), default=0)
    rows = []
    for ni, _ in glb.mesh_nodes(doc):
        node = doc["nodes"][ni]
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            a = prim["attributes"]
            n = doc["accessors"][a["POSITION"]]["count"]
            idx = glb.accessor(doc, blob, prim["indices"]).astype(int).ravel() if "indices" in prim else np.arange(n)
            w = np.zeros((n, count))
            if "JOINTS_0" in a and "skin" in node:
                j = glb.accessor(doc, blob, a["JOINTS_0"]).astype(int)
                v = glb.accessor(doc, blob, a["WEIGHTS_0"])
                np.add.at(w, (np.repeat(np.arange(n), j.shape[1]), j.ravel()), v.ravel())
            rows.append(w[idx[: len(idx) // 3 * 3].reshape(-1, 3)])
    return np.concatenate(rows) if rows else np.zeros((0, 3, count))


def joint_names(doc: dict) -> list[str]:
    skins = doc.get("skins") or []
    return [doc["nodes"][j].get("name", f"joint{j}") for j in skins[0]["joints"]] if skins else []


def joint_heads(doc: dict, blob: bytes, clip: str | None = None, t: float = 0.0) -> dict[str, np.ndarray]:
    """Each joint's head in glTF axes, at rest or at a time of a clip."""
    world = skin.worlds(doc) if clip is None else skin.sample(doc, blob, clip, t)
    skins = doc.get("skins") or []
    return {doc["nodes"][j].get("name", f"joint{j}"): world[j][:3, 3] for j in (skins[0]["joints"] if skins else [])}


def bones(image: Image.Image, view: dict, doc: dict, blob: bytes, clip: str | None = None, t: float = 0.0,
          labels: bool = True) -> Image.Image:
    """The skeleton over an orthographic picture: a dot at each joint's head, a line to its parent's."""
    heads = joint_heads(doc, blob, clip, t)
    names = joint_names(doc)
    colours = (bone_colours(len(names)) * 255).astype(int)
    parent = {c: i for i, n in enumerate(doc["nodes"]) for c in n.get("children", [])}
    index = {doc["nodes"][j].get("name", f"joint{j}"): j for j in doc["skins"][0]["joints"]} if names else {}
    by_node = {j: n for n, j in index.items()}
    out = image.copy()
    d = ImageDraw.Draw(out)
    size = image.width
    for k, name in enumerate(names):
        x, y = pixel(view, size, heads[name])
        up = by_node.get(parent.get(index[name]))
        if up is not None:
            px, py = pixel(view, size, heads[up])
            d.line([(px, py), (x, y)], fill=(20, 20, 20), width=3)
            d.line([(px, py), (x, y)], fill=tuple(colours[k]), width=1)
    for k, name in enumerate(names):
        x, y = pixel(view, size, heads[name])
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=tuple(colours[k]), outline=(20, 20, 20))
        if labels:
            d.text((x + 6, y - 5), name, fill=(0, 0, 0))
    return out


def weights_mesh(rigged: str | Path) -> tuple[render.Mesh, list[str], np.ndarray]:
    """The rest pose painted by weight: each corner the blend of its bones' colours. -> (mesh, bones, colours)."""
    doc, blob = glb.read_glb(rigged)
    names = joint_names(doc)
    colours = bone_colours(len(names))
    mesh = skin.posed(rigged)
    w = corner_weights(doc, blob)
    painted = render.srgb_to_linear(w @ colours) if len(names) else mesh.colors
    return render.Mesh(mesh.tris, painted, mesh.normals), names, colours


def legend(names: list[str], colours: np.ndarray, width: int) -> Image.Image:
    per_row = max(1, width // 130)
    rows = (len(names) + per_row - 1) // per_row
    out = Image.new("RGB", (width, 16 * rows + 6), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for k, name in enumerate(names):
        x, y = 6 + (k % per_row) * 130, 3 + (k // per_row) * 16
        d.rectangle([x, y + 1, x + 10, y + 11], fill=tuple((colours[k] * 255).astype(int)), outline=(0, 0, 0))
        d.text((x + 14, y), name, fill=(0, 0, 0))
    return out


def row(images: list[Image.Image], title: str = "", foot: Image.Image | None = None) -> Image.Image:
    width = sum(i.width for i in images)
    height = max(i.height for i in images)
    top = 16 if title else 0
    out = Image.new("RGB", (width, top + height + (foot.height if foot else 0)), (255, 255, 255))
    if title:
        ImageDraw.Draw(out).text((6, 2), title, fill=(0, 0, 0))
    x = 0
    for i in images:
        out.paste(i, (x, top))
        x += i.width
    if foot:
        out.paste(foot, (0, top + height))
    return out


def weights_sheet(rigged: str | Path, lo, hi, views=("front", "left", "back"), size: int = 512) -> Image.Image:
    """The weights from each view with the skeleton over them and the bones' colours named underneath. A patch in
    the wrong colour is a wrong weight, seen before anything moves."""
    doc, blob = glb.read_glb(rigged)
    mesh, names, colours = weights_mesh(rigged)
    cells = []
    for name in views:
        view = shot(name, lo, hi)
        cells.append(bones(rule(draw(mesh, {**view, "light": "camera"}, size), view, lo, hi), view, doc, blob, labels=False))
    return row(cells, "WEIGHTS: each point in the blend of its bones' colours; dots are joints. Views: "
               + ", ".join(f"{v} ({FACES[v]})" for v in views), legend(names, colours, size * len(views)))


def cells(rigged: str | Path, clip: str, seconds: float, lo, hi, views=("left", "front"), phases: int = 6,
          size: int = 256, skeleton: bool = False) -> dict[str, list[Image.Image]]:
    """view -> a clip's frames at `phases` evenly spaced times. The ground the rest pose stands on is drawn as a
    line in the side and front views. A frame is a second of rendering, so whoever needs two strips of one build
    draws its frames once (`strip`'s `drawn`)."""
    doc, blob = glb.read_glb(rigged)
    ground = float(lo[2])
    out = {v: [] for v in views}
    for k in range(phases):
        t = seconds * k / phases
        mesh = skin.posed(rigged, clip, t)
        for name in views:
            view = shot(name, lo, hi)
            cell = draw(mesh, view, size)
            d = ImageDraw.Draw(cell)
            if name in ("front", "back", "left", "right"):
                y = pixel(view, size, to_gltf((0, 0, ground)))[1]
                d.line([(0, y), (size, y)], fill=AXIS, width=1)
                if skeleton:
                    cell = bones(cell, view, doc, blob, clip, t, labels=False)
                    d = ImageDraw.Draw(cell)
            d.text((4, 3), f"{name} t={t:.2f}s", fill=INK)
            out[name].append(cell)
    return out


def strip(rigged: str | Path, clip: str, seconds: float, lo, hi, views=("left", "front"), phases: int = 6,
          size: int = 256, skeleton: bool = False, drawn: dict | None = None) -> Image.Image:
    """A clip as a film strip: `phases` evenly spaced times left to right, a row a view. `drawn` is frames already
    made by `cells` for some of the views (same phases and size): only the views it lacks are rendered."""
    have = {v: c for v, c in (drawn or {}).items() if v in views and len(c) == phases and c[0].width == size}
    missing = tuple(v for v in views if v not in have)
    if missing:
        have.update(cells(rigged, clip, seconds, lo, hi, missing, phases, size, skeleton))
    rows = [row(have[v]) for v in views]
    out = Image.new("RGB", (rows[0].width, 16 + sum(r.height for r in rows)), (255, 255, 255))
    ImageDraw.Draw(out).text((6, 2), f"CLIP {clip}: {seconds:g} s, {phases} phases left to right; rows: "
                             + "; ".join(f"{v} ({FACES[v]})" for v in views) + ". The line is the rest pose's ground.",
                             fill=(0, 0, 0))
    y = 16
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out
