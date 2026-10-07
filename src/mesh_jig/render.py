"""The CPU renderer: a model GLB drawn from fixed cameras with a numpy z-buffer. No GPU, no game engine, no Blender
render, and the same bytes on every machine.

What it draws is a contract, not a look: named cameras (cam_pos, cam_target, vertical fov, 16:9), the model scaled so
its contract length is one unit, vertex colour as albedo (or the GLB's base-colour texture on a textured contract), a
flat ground colour, ambient and two directional lights, Lambert only: no shadows, no specular, no anti-aliasing.
Shape, proportion, pose and colour zones are faithful; shading is simplified. That is enough for every measure in
this package, and it is what lets an agent look at what it built seconds after building it.

The default cameras are named after the reference views they are compared with: `hero` (a three-quarter view from the
front left), `top` (nose at the bottom of the image), `side` (nose at the left) and `rear`. A project whose subject
does not fit them (a standing figure) lists its own under `model.shots`.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import glb, silhouette

RENDER_SIZE = (1280, 720)             # the measured renders
PREVIEW_SIZE = (640, 360)             # a quick look: a quarter of the pixels
NEAR = 0.05
BACKGROUND = (221, 219, 214)
# linear light: ambient colour x energy, and each directional light's travel direction with colour x energy. The
# Lambert 1/pi is folded into the light, so diffuse = albedo * NdotL * E.
AMBIENT = 0.55 * np.array([0.787, 0.787, 0.827])
LIGHTS = [
    (np.array([-0.4, -0.7, -0.57]), 1.1 * np.array([1.0, 1.0, 1.0])),        # key
    (np.array([0.64, -0.42, 0.64]), 0.45 * np.array([0.787, 0.846, 1.0])),   # fill
]
DIFFUSE = 0.75
# `light: "camera"` (jig.json `criteria.light`): white lights that turn with the camera, as a drawn turnaround is lit.
# A face that looks straight at the camera gets exactly 1.0, so it is drawn in the colour it stores, in every view;
# the stage lights above give a front-facing face of an upright subject 0.63 and one seen from above 1.15. Each
# direction is the light's travel in the camera's own (right, up, forward): the key from the upper left over the
# camera's shoulder, the fill from the lower right. The key's energy is whatever brings a camera-facing face to 1.
LIGHT_MODES = ("stage", "camera")
CAMERA_AMBIENT = 0.45
CAMERA_KEY, CAMERA_FILL, CAMERA_FILL_ENERGY = (0.33, -0.5, 0.8), (-0.77, 0.2, 0.6), 0.2
DEFAULT_SHOTS = [
    {"name": "hero", "cam_pos": [-1.25, 1.15, -1.05], "cam_target": [0.02, -0.02, 0.02], "fov": 27},
    {"name": "top", "cam_pos": [0.0, 2.3, -0.3], "cam_target": [0.0, 0.0, 0.0], "fov": 28},
    {"name": "side", "cam_pos": [-2.4, 0.12, 0.0], "cam_target": [0.0, 0.0, 0.0], "fov": 19},
    {"name": "rear", "cam_pos": [0.0, 0.3, 2.4], "cam_target": [0.0, 0.02, 0.0], "fov": 19},
]
COMPARE_SUFFIX = "_vs_reference"


class Mesh:
    """Triangles in model space (T, 3, 3), per-corner linear albedo (T, 3, 3) or UVs (T, 3, 2) + texture index (T,),
    per-corner normals (T, 3, 3)."""

    def __init__(self, tris: np.ndarray, colors: np.ndarray, normals: np.ndarray, uvs: np.ndarray | None = None,
                 tex_index: np.ndarray | None = None, textures: list[np.ndarray] | None = None):
        self.tris, self.colors, self.normals = tris, colors, normals
        self.uvs, self.tex_index, self.textures = uvs, tex_index, textures


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _image(gltf: dict, blob: bytes, tex: int) -> np.ndarray | None:
    try:
        src = gltf["textures"][tex]["source"]
        view = gltf["bufferViews"][gltf["images"][src]["bufferView"]]
    except (KeyError, IndexError):
        return None
    start = view.get("byteOffset", 0)
    im = Image.open(io.BytesIO(blob[start:start + view["byteLength"]])).convert("RGB")
    return srgb_to_linear(np.asarray(im, dtype=np.float64) / 255.0)


def load(path: str | Path, textured: bool = False, multiply: bool = False) -> Mesh:
    """Every triangle primitive of the default scene, world-transformed. Vertex-colour contract (`textured` False):
    albedo is COLOR_0, white where a primitive has none. Textured contract: the material's base-colour texture times
    its factor, the factor alone where there is no texture. `multiply` (glb.texture_mode "multiply"): COLOR_0 times
    the texture."""
    gltf, blob = glb.read_glb(path)
    mats = gltf.get("materials", [])
    tris, cols, norms, uvs, texi, textures, tex_slot = [], [], [], [], [], [], {}
    for ni, world in glb.mesh_nodes(gltf):
        rot = world[:3, :3]
        nrm_m = np.linalg.inv(rot).T if abs(np.linalg.det(rot)) > 1e-12 else rot
        for prim in gltf["meshes"][gltf["nodes"][ni]["mesh"]].get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            a = prim["attributes"]
            pos = glb.accessor(gltf, blob, a["POSITION"])
            pts = pos @ rot.T + world[:3, 3]
            idx = (glb.accessor(gltf, blob, prim["indices"]).astype(np.int64).ravel() if "indices" in prim
                   else np.arange(len(pos)))
            tri = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            if not len(tri):
                continue
            tris.append(pts[tri])
            if "NORMAL" in a:
                n = glb.accessor(gltf, blob, a["NORMAL"]) @ nrm_m.T
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                norms.append(n[tri])
            else:
                p = pts[tri]
                fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
                norms.append(np.repeat(fn[:, None, :], 3, axis=1))
            mat = mats[prim["material"]] if prim.get("material") is not None and prim["material"] < len(mats) else {}
            pbr = mat.get("pbrMetallicRoughness") or {}
            factor = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float64)
            slot = -1
            if textured and "baseColorTexture" in pbr and "TEXCOORD_0" in a:
                t = pbr["baseColorTexture"]["index"]
                if t not in tex_slot:
                    img = _image(gltf, blob, t)
                    if img is not None:
                        tex_slot[t] = len(textures)
                        textures.append(img * factor)
                slot = tex_slot.get(t, -1)
            if slot >= 0:
                uvs.append(glb.accessor(gltf, blob, a["TEXCOORD_0"])[:, :2][tri])
                cols.append(glb.accessor(gltf, blob, a["COLOR_0"])[:, :3][tri] if multiply and "COLOR_0" in a
                            else np.ones((len(tri), 3, 3)))
            else:
                uvs.append(np.zeros((len(tri), 3, 2)))
                if not textured and "COLOR_0" in a:
                    cols.append(glb.accessor(gltf, blob, a["COLOR_0"])[:, :3][tri])
                else:
                    cols.append(np.broadcast_to(factor if textured else np.ones(3), (len(tri), 3, 3)).copy())
            texi.append(np.full(len(tri), slot))
    if not tris:
        z = np.zeros((0, 3, 3))
        return Mesh(z, z, z)
    return Mesh(np.concatenate(tris), np.concatenate(cols), np.concatenate(norms),
                np.concatenate(uvs) if textures else None, np.concatenate(texi) if textures else None, textures or None)


def camera(shot: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(eye, rows of the camera basis [right, up, forward], tan of half the vertical fov). A look-at with +Y up, or
    with the shot's own `cam_up` (a camera looking straight down needs one to say which way the image's top points);
    `fov` is vertical, in degrees."""
    eye = np.array(shot["cam_pos"], dtype=np.float64)
    f = np.array(shot["cam_target"], dtype=np.float64) - eye
    f /= np.linalg.norm(f)
    up = np.array(shot.get("cam_up") or [0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(f @ up)) > 0.999 * float(np.linalg.norm(up)):   # looking along `up`: any stable right will do
        up = np.array([0.0, 0.0, -1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, np.stack([r, u, f]), math.tan(math.radians(float(shot.get("fov", 75))) / 2)


def rasterise(mesh: Mesh, shot: dict, *, scale: float = 1.0, width: int = PREVIEW_SIZE[0], height: int = PREVIEW_SIZE[1]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (triangle id per pixel, -1 = background; perspective-correct barycentrics (H, W, 3); depth). Back faces
    (glTF front = counter-clockwise) are culled; a triangle crossing the near plane is dropped. A shot with `ortho`
    (the height of the view in model units) is drawn orthographically: no foreshortening, as a drawn top, side or
    rear view has none."""
    eye, basis, t = camera(shot)
    cam = (mesh.tris * scale - eye) @ basis.T                 # (T, 3, 3): x right, y up, z forward
    z = cam[..., 2]
    ortho = float(shot.get("ortho") or 0.0)
    if ortho:
        px_per_unit = height / ortho
        sx = width / 2 + cam[..., 0] * px_per_unit
        sy = height / 2 - cam[..., 1] * px_per_unit
        w = np.ones_like(z)                                   # nothing to correct: screen space is already linear
    else:
        fpx = (height / 2) / t
        sx = width / 2 + cam[..., 0] / np.maximum(z, 1e-9) * fpx
        sy = height / 2 - cam[..., 1] / np.maximum(z, 1e-9) * fpx
        w = 1.0 / np.maximum(z, 1e-9)                         # 1/z is linear in screen space
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    keep = (z.min(axis=1) > NEAR) & (area < 0)                # y down: counter-clockwise in view has negative area
    ids = np.full((height, width), -1, dtype=np.int64)
    depth = np.full((height, width), np.inf)
    bary = np.zeros((height, width, 3))
    for i in np.flatnonzero(keep):
        xs, ys = sx[i], sy[i]
        x0, x1 = max(int(math.floor(xs.min())), 0), min(int(math.ceil(xs.max())), width - 1)
        y0, y1 = max(int(math.floor(ys.min())), 0), min(int(math.ceil(ys.max())), height - 1)
        if x0 > x1 or y0 > y1:
            continue
        px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a = area[i]
        w0 = ((xs[1] - px) * (ys[2] - py) - (xs[2] - px) * (ys[1] - py)) / a
        w1 = ((xs[2] - px) * (ys[0] - py) - (xs[0] - px) * (ys[2] - py)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        pc = np.stack([w0 * w[i, 0], w1 * w[i, 1], w2 * w[i, 2]], axis=-1)
        pc /= np.maximum(pc.sum(axis=-1, keepdims=True), 1e-12)
        d = pc @ z[i]
        win = inside & (d < depth[y0:y1 + 1, x0:x1 + 1])
        if not win.any():
            continue
        depth[y0:y1 + 1, x0:x1 + 1][win] = d[win]
        ids[y0:y1 + 1, x0:x1 + 1][win] = i
        bary[y0:y1 + 1, x0:x1 + 1][win] = pc[win]
    return ids, bary, depth


def lighting(spec: dict | None, basis: np.ndarray) -> tuple[np.ndarray, list]:
    """(ambient, [(travel direction, colour x energy)]) for a shot whose camera basis is `basis` (rows right, up,
    forward): the fixed stage lights, or with `light: "camera"` the pair that turns with the camera."""
    if (spec or {}).get("light", "stage") != "camera":
        return AMBIENT, LIGHTS
    key, fill = (np.array(d) / np.linalg.norm(d) for d in (CAMERA_KEY, CAMERA_FILL))
    key_energy = (1.0 - CAMERA_AMBIENT - fill[2] * CAMERA_FILL_ENERGY * DIFFUSE) / (key[2] * DIFFUSE)
    white = np.ones(3)
    return CAMERA_AMBIENT * white, [(key @ basis, key_energy * white), (fill @ basis, CAMERA_FILL_ENERGY * white)]


def shade(mesh: Mesh, ids: np.ndarray, bary: np.ndarray, view_dir: np.ndarray,
          light: tuple[np.ndarray, list] | None = None) -> np.ndarray:
    """uint8 RGB: background where ids < 0, else albedo * (ambient + sum of Lambert terms), sRGB-encoded. `light`
    is `lighting`'s pair; the stage lights when it is not given."""
    ambient, lights = light if light is not None else (AMBIENT, LIGHTS)
    h, w = ids.shape
    out = np.empty((h, w, 3))
    out[:] = np.array(BACKGROUND) / 255.0
    hit = ids >= 0
    if not hit.any():
        return (out * 255 + 0.5).astype(np.uint8)
    tid, b = ids[hit], bary[hit][..., None]                   # (N,), (N, 3, 1)
    alb = (mesh.colors[tid] * b).sum(axis=1)
    if mesh.textures:
        uv = (mesh.uvs[tid] * b).sum(axis=1)
        for k, tex in enumerate(mesh.textures):
            sel = mesh.tex_index[tid] == k
            if sel.any():
                th, tw = tex.shape[:2]
                u = np.mod(uv[sel, 0], 1.0)
                v = np.mod(uv[sel, 1], 1.0)                   # glTF v runs down the image
                alb[sel] *= tex[np.minimum((v * th).astype(int), th - 1), np.minimum((u * tw).astype(int), tw - 1)]
    n = (mesh.normals[tid] * b).sum(axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    flip = (n @ view_dir) > 0                                  # a normal facing away from the camera: shade its back
    n[flip] *= -1
    lit = np.broadcast_to(ambient, alb.shape).copy()
    for d, c in lights:
        d = d / np.linalg.norm(d)
        lit += np.clip(-(n @ d), 0, None)[:, None] * c * DIFFUSE
    out[hit] = linear_to_srgb(alb * lit)
    return (out * 255 + 0.5).astype(np.uint8)


def render(mesh: Mesh, shot: dict, *, scale: float = 1.0, width: int = PREVIEW_SIZE[0], height: int = PREVIEW_SIZE[1],
           spec: dict | None = None) -> Image.Image:
    ids, bary, _ = rasterise(mesh, shot, scale=scale, width=width, height=height)
    _, basis, _ = camera(shot)
    return Image.fromarray(shade(mesh, ids, bary, basis[2], lighting(spec, basis)))


# -- a project's view of it ----------------------------------------------------------------------------

def stage_scale(spec: dict) -> float:
    """The model is drawn one unit long: 1 / length_m when the contract names a length, else `stage_scale`."""
    return 1.0 / float(spec["length_m"]) if spec.get("length_m") else float(spec.get("stage_scale", 1.0))


def shots_for(spec: dict) -> list[dict]:
    """The shots to render, each with the `variant` it shows: the listed (or default) shots on the judged variant,
    and, with the default shots, a `hero_<variant>` of every other variant, so a fit that is built and gated is also
    looked at. Only a render with a reference view of the same name is measured."""
    variants = glb.variant_names(spec)
    judged = glb.judged_variant(spec)
    listed = spec.get("shots")
    base = list(listed or DEFAULT_SHOTS)
    if not listed:
        hero = next(s for s in DEFAULT_SHOTS if s["name"] == "hero")
        base += [{**hero, "name": f"hero_{v}", "variant": v} for v in variants if v and v != judged]
    return [{**s, "variant": s.get("variant", judged)} for s in base]


def touches_edge(img: Image.Image) -> bool:
    a = np.asarray(img.convert("RGB"), dtype=int)
    body = np.abs(a - np.array(BACKGROUND)).max(axis=2) > 0
    return bool(body[0].any() or body[-1].any() or body[:, 0].any() or body[:, -1].any())


def render_views(spec: dict, glb_dir: str | Path, out_dir: str | Path, *, views: list[str] | None = None,
                 size: tuple[int, int] = RENDER_SIZE) -> dict:
    """Render the project's shots from the GLBs under glb_dir into out_dir/<view>.png.
    -> {"renders": {view: path}, "cropped": [views whose subject touches the frame edge], "error": ""}. A GLB that
    is not built, a shot name that does not exist or a frame with nothing in it is the error; nothing is raised."""
    glb_dir, out_dir = Path(glb_dir), Path(out_dir)
    shots = shots_for(spec)
    names = [s["name"] for s in shots]
    unknown = [v for v in (views or []) if v not in names]
    if unknown:
        return {"renders": {}, "cropped": [], "error": f"no shot named {unknown}; the shots are {names}"}
    files = glb.expected_files(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, Mesh] = {}
    renders, cropped = {}, []
    width, height = size
    for shot in shots:
        if views and shot["name"] not in views:
            continue
        fname = files.get(shot["variant"], "")
        path = glb_dir / fname
        if not fname or not path.is_file():
            return {"renders": {}, "cropped": [], "error": f"{fname or 'the GLB'} is not built yet: build first"}
        if fname not in meshes:
            meshes[fname] = load(path, textured=bool(spec.get("textured")),
                                 multiply=glb.texture_mode(spec) == "multiply")
        ids, bary, _ = rasterise(meshes[fname], shot, scale=stage_scale(spec), width=width, height=height)
        if not (ids >= 0).any():
            return {"renders": {}, "cropped": [], "error":
                    f"{shot['name']}: NOTHING VISIBLE from this camera (outside the frame, behind the camera, or only "
                    "back faces: check the model's size, its centre and its face winding)"}
        basis = camera(shot)[1]
        img = Image.fromarray(shade(meshes[fname], ids, bary, basis[2], lighting(spec, basis)))
        if touches_edge(img):
            cropped.append(shot["name"])
        target = out_dir / f"{shot['name']}.png"
        img.save(target)
        renders[shot["name"]] = str(target)
    return {"renders": renders, "cropped": cropped, "error": ""}


def write_comparisons(renders: dict[str, str], references: dict[str, str | Path], out_dir: str | Path,
                      height: int = 360) -> dict[str, str]:
    """<view>_vs_reference.png for every render with a reference of the same name: the reference on the left, the
    render on the right, both cropped to the subject and scaled to one height."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for view, path in renders.items():
        if view not in references:
            continue
        target = out_dir / f"{view}{COMPARE_SUFFIX}.png"
        silhouette.side_by_side(references[view], path, f"REFERENCE {view}", f"YOUR BUILD {view}", height=height).save(target)
        out[view] = str(target)
    return out
