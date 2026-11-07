"""What a built GLB costs an engine, and what the stages after modelling (a rig, a texture) would find in it.

Read from the exported file alone, with numpy: no Blender, no model, the same bytes for the same GLB. These are
readings beside the fidelity score and never part of it. With a target (`target.resolve`) each reading is also held
to the target's limit and the record says how many hold; without one there are readings and no verdict.

    cost      triangles, vertices as the GPU gets them and as positions, mesh nodes, primitives, materials, bytes
    surface   connected pieces, how many hold nine tenths of the area, open and over-shared edges, slivers,
              triangles no outside view sees, vertices with a mirror twin across X = 0
    uv        (when any primitive has TEXCOORD_0) triangles with UVs, outside the unit square, collapsed to a
              point or line, how even the texel density is, how much of the square the triangles use
    limbs     the parts that leave the body side by side (arms, legs), read from cuts across the model as a rig
              script's author is shown them (`rig_brief.stretches`): how long each hangs clear, and how far above
              that sloping side parts stay joined to the body. A rig bends a part only where it hangs clear
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import glb, rig, rig_brief

WELD = 6                # decimals of a metre two corners must agree to, to be one vertex (`review.component_count`)
SLIVER = 20.0           # a triangle's longest edge over its height on that edge, above which it is a sliver
MOST = 0.9              # `pieces_90` counts the largest pieces until they hold this share of the surface
MIRROR_MM = 1.0         # a vertex and its mirror twin agree to this
GRID = 256              # cells across the model in the depth maps `buried` is read from
FACING = 0.6            # a sample is looked for only from directions within about 53 degrees of its normal
LIMB_M = 0.1            # separate parts shorter than this (an ear's tip) are not read as a limb
LOW_SHARE = 0.15        # a stretch ending this share of the height above the ground, not on it: legs whose feet join
MAX_SPLIT = 48          # a triangle is sampled on at most this many steps a side
DIRECTIONS = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)],
                      dtype=np.float64)


def read(path: str | Path) -> dict:
    """The GLB's triangles in world space with what each carries: `tris` (T, 3, 3), `ids` (T, 3) corner vertex
    numbers as exported (unique across primitives), `uvs` (T, 3, 2) with NaN where a primitive has none, `nodes`
    (a row a mesh node: name, triangles, primitives) and the file's counts.
    Raises ValueError for non-finite stored or world-space positions in any triangle primitive;
    no partial reading is returned. Missing UVs keep their NaN sentinel."""
    gltf, blob = glb.read_glb(path)
    tris, ids, uvs, nodes, offset, prims = [], [], [], [], 0, 0
    for ni, world in glb.mesh_nodes(gltf):
        node = gltf["nodes"][ni]
        mesh = gltf["meshes"][node["mesh"]]
        row = {"name": node.get("name") or mesh.get("name") or f"node{ni}", "triangles": 0, "primitives": 0}
        for pi, prim in enumerate(mesh.get("primitives", [])):
            if prim.get("mode", 4) != 4:
                continue
            pos = glb.accessor(gltf, blob, prim["attributes"]["POSITION"]).astype(np.float64)
            if not np.isfinite(pos).all():
                raise ValueError(f"{Path(path).name}: node {ni} ({row['name']}), primitive {pi}: "
                                 "non-finite stored vertex positions; readiness unavailable")
            pts = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3]
            if not np.isfinite(pts).all():
                raise ValueError(f"{Path(path).name}: node {ni} ({row['name']}), primitive {pi}: "
                                 "non-finite world-space vertex positions; readiness unavailable")
            idx = (glb.accessor(gltf, blob, prim["indices"]).astype(np.int64).ravel() if "indices" in prim
                   else np.arange(len(pos)))
            tri = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            tris.append(pts[tri])
            ids.append(tri + offset)
            if "TEXCOORD_0" in prim["attributes"]:
                uvs.append(glb.accessor(gltf, blob, prim["attributes"]["TEXCOORD_0"]).astype(np.float64)[tri])
            else:
                uvs.append(np.full((len(tri), 3, 2), np.nan))
            offset += len(pos)
            prims += 1
            row["triangles"] += len(tri)
            row["primitives"] += 1
        nodes.append(row)
    stack = lambda parts, shape: np.concatenate(parts) if parts else np.zeros(shape)
    return {"tris": stack(tris, (0, 3, 3)), "ids": stack(ids, (0, 3)).astype(np.int64), "uvs": stack(uvs, (0, 3, 2)),
            "nodes": nodes, "vertices": offset, "primitives": prims, "materials": len(gltf.get("materials") or []),
            "images": len(gltf.get("images") or []), "skins": len(gltf.get("skins") or []),
            "animations": len(gltf.get("animations") or []), "bytes": Path(path).stat().st_size}


def welded(tris: np.ndarray) -> np.ndarray:
    """(T, 3) vertex numbers with corners at one position made one vertex."""
    if not len(tris):
        return np.zeros((0, 3), dtype=np.int64)
    _, inverse = np.unique(np.round(tris.reshape(-1, 3), WELD), axis=0, return_inverse=True)
    return inverse.reshape(-1, 3).astype(np.int64)


def areas(tris: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2


def edges_of(weld: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unique undirected edges (E, 2), how many triangles use each), leaving out a triangle's collapsed sides."""
    pairs = np.sort(np.concatenate([weld[:, [0, 1]], weld[:, [1, 2]], weld[:, [2, 0]]]), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.unique(pairs, axis=0, return_counts=True)


def piece_labels(weld: np.ndarray) -> np.ndarray:
    """A label a triangle: triangles joined through shared vertices carry the same one."""
    if not len(weld):
        return np.zeros(0, dtype=np.int64)
    label = np.arange(int(weld.max()) + 1)
    a = np.concatenate([weld[:, 0], weld[:, 1]])
    b = np.concatenate([weld[:, 1], weld[:, 2]])
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]                      # jump to the label's own label, so a long strip settles in few rounds
        if np.array_equal(new, label):
            return label[weld[:, 0]]
        label = new


def slivers(tris: np.ndarray) -> float:
    """The share of triangles whose longest edge is more than SLIVER times their height on it (a flat one counts)."""
    return float(sliver_mask(tris).mean()) if len(tris) else 0.0


def sliver_mask(tris: np.ndarray) -> np.ndarray:
    """Which triangles are slivers."""
    if not len(tris):
        return np.zeros(0, dtype=bool)
    sides = np.stack([np.linalg.norm(tris[:, (i + 1) % 3] - tris[:, i], axis=1) for i in range(3)], axis=1)
    longest = sides.max(axis=1)
    height = 2 * areas(tris) / np.maximum(longest, 1e-12)
    return longest > SLIVER * height


def symmetry(tris: np.ndarray) -> float:
    """The share of vertices with a twin mirrored across X = 0, to a millimetre. 1 for a model built as one side
    and its mirror; a remeshed or hand-placed surface has few."""
    if not len(tris):
        return 0.0
    return float(_twinned(tris)[0].mean())


def unmirrored_mask(tris: np.ndarray) -> np.ndarray:
    """Which triangles have a corner with no twin mirrored across X = 0."""
    if not len(tris):
        return np.zeros(0, dtype=bool)
    twinned, inverse = _twinned(tris)
    return ~twinned[inverse].reshape(-1, 3).all(axis=1)


def _twinned(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(whether each distinct vertex, to the mirror's millimetre, has a twin across X = 0; each corner's vertex)."""
    grid, inverse = np.unique(np.round(tris.reshape(-1, 3) * (1000.0 / MIRROR_MM)).astype(np.int64), axis=0,
                              return_inverse=True)
    span = int(np.abs(grid).max()) + 1

    def key(p):
        return ((p[:, 0] + span) * (2 * span + 1) + (p[:, 1] + span)) * (2 * span + 1) + (p[:, 2] + span)

    return np.isin(key(grid * np.array([-1, 1, 1])), key(grid)), inverse.ravel()


def samples(tris: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over every triangle no further apart than `step` (the centroids of an even split of it), and
    the triangle each is on."""
    sides = np.stack([np.linalg.norm(tris[:, (i + 1) % 3] - tris[:, i], axis=1) for i in range(3)], axis=1)
    split = np.clip(np.ceil(sides.max(axis=1) / max(step, 1e-12)), 1, MAX_SPLIT).astype(np.int64)
    points, owner = [], []
    for n in np.unique(split):
        which = np.flatnonzero(split == n)
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        up = (i + j) <= n - 1
        down = (i + j) <= n - 2
        u = np.concatenate([(i[up] + 1 / 3), (i[down] + 2 / 3)]) / n
        v = np.concatenate([(j[up] + 1 / 3), (j[down] + 2 / 3)]) / n
        t = tris[which]
        p = (t[:, None, 0] * (1 - u - v)[None, :, None] + t[:, None, 1] * u[None, :, None]
             + t[:, None, 2] * v[None, :, None])
        points.append(p.reshape(-1, 3))
        owner.append(np.repeat(which, len(u)))
    return np.concatenate(points), np.concatenate(owner)


def buried(tris: np.ndarray, grid: int = GRID) -> float:
    """The share of triangles no outside view can see (`buried_mask`)."""
    return float(buried_mask(tris, grid).mean()) if len(tris) else 0.0


def buried_mask(tris: np.ndarray, grid: int = GRID) -> np.ndarray:
    """The share of triangles no outside view can see: drawn from 26 directions as depth maps of points spread over
    the surface, a triangle is seen when one of its points is the nearest thing in its cell from a direction that
    faces it. A part sunk inside another, the inside of a shell and the hidden half of two overlapping shapes are
    buried; a hollow that some direction looks into is not. Counted in triangles, since that is what is paid for.
    A surface closer than about two cells (a 128th of the model) under another reads as seen."""
    area = areas(tris)
    live = area > 0
    if not live.any():
        return np.ones(len(tris), dtype=bool)
    t = tris[live]
    normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    lo, hi = t.reshape(-1, 3).min(axis=0), t.reshape(-1, 3).max(axis=0)
    centre, across = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
    if across <= 0:
        return np.ones(len(tris), dtype=bool)
    cell = across / grid
    points, owner = samples(t, cell)
    points = points - centre
    seen = np.zeros(len(t), dtype=bool)
    for d in DIRECTIONS / np.linalg.norm(DIRECTIONS, axis=1, keepdims=True):
        helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        right = np.cross(helper, d)
        right /= np.linalg.norm(right)
        up = np.cross(d, right)
        col = np.clip(((points @ right) / cell + grid / 2).astype(np.int64), 0, grid - 1)
        row = np.clip(((points @ up) / cell + grid / 2).astype(np.int64), 0, grid - 1)
        flat = row * grid + col
        height = points @ d                               # larger is nearer the eye, which stands out along d
        nearest = np.full(grid * grid, -np.inf)
        np.maximum.at(nearest, flat, height)
        facing = normal[owner] @ d
        tilt = np.sqrt(np.maximum(1 - facing ** 2, 0)) / np.maximum(facing, 1e-9)
        ok = (facing >= FACING) & (height >= nearest[flat] - cell * (0.75 + tilt))
        seen[owner[ok]] = True
    out = np.ones(len(tris), dtype=bool)
    out[np.flatnonzero(live)[seen]] = False
    return out


def uv_reading(tris: np.ndarray, uvs: np.ndarray) -> dict | None:
    """None when no triangle has UVs. `with_uv`: the share of triangles that have them. Of those: `outside` the
    unit square, `collapsed` to a point or a line, `texel_spread` (how many times denser the texels are on the
    densest twentieth of the triangles than on the sparsest: 1 is even) and `used` (the triangles' area in the
    square: above 1 they overlap for certain, far below it the texture is mostly empty)."""
    has = ~np.isnan(uvs).any(axis=(1, 2))
    if not has.any():
        return None
    uv, world = uvs[has], areas(tris[has])
    a, b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    flat = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    both = (flat > 1e-12) & (world > 1e-12)
    density = np.sqrt(flat[both] / world[both])
    spread = float(np.percentile(density, 95) / max(np.percentile(density, 5), 1e-12)) if both.any() else 0.0
    out = ((uv < -1e-6) | (uv > 1 + 1e-6)).any(axis=(1, 2))
    return {"with_uv": round(float(has.mean()), 4), "outside": round(float(out.mean()), 4),
            "collapsed": round(float(np.mean(flat <= 1e-12)), 4), "texel_spread": round(spread, 2),
            "used": round(float(flat.sum()), 4)}


def limbs(tris: np.ndarray) -> list[dict]:
    """`tris` in glTF axes (+Y up). Each stretch of height where the model is two or more parts side by side, that
    leaves the body (not the top of the model) and is at least LIMB_M long: `parts`, `clear_m` (how long they hang
    clear), `joined_m` (how far above that sloping side parts stay joined to the body; 0 when they hang clear where
    they begin), `from_m` (the top of the clear stretch, above the ground) and `on_ground`. Low-poly cats here had
    arms joined to the flank for 0.25 m above 0.15 m hanging clear, and every rig made of one stretched at the armpit."""
    if not len(tris):
        return []
    points = rig.to_blender(tris.reshape(-1, 3).astype(np.float64)).reshape(-1, 3, 3)
    ground = float(points[..., 2].min())
    out = []
    for s in rig_brief.stretches(rig_brief.slices(points), ground):
        if not s["leaves"] or s["top"] - s["bottom"] < LIMB_M - 1e-9:
            continue
        out.append({"parts": s["count"], "clear_m": round(s["top"] - s["bottom"], 3),
                    "joined_m": round(s["joined"] - s["top"], 3) if s["joined"] is not None else 0.0,
                    "from_m": round(s["top"] - ground, 3), "on_ground": bool(s["on_ground"])})
    return out


def checks(reading: dict, target: dict, spec: dict | None = None) -> list[dict]:
    """Each reading a target limits, as {name, value, limit, passed, what}. `hard` marks the one the contract
    also holds a build to."""
    cost, surface = reading["cost"], reading["surface"]

    def row(name, value, limit, passed, what, hard=False):
        return {"name": name, "value": value, "limit": limit, "passed": bool(passed), "what": what, "hard": hard}

    out = [
        row("triangles", cost["triangles"], f"at most {target['max_triangles']}",
            cost["triangles"] <= target["max_triangles"], "the engine's budget for this model", hard=True),
        row("buried", surface["buried"], f"at most {target['max_buried']:.0%}",
            surface["buried"] <= target["max_buried"], "triangles no outside view can see"),
        row("few pieces", surface["pieces_90"], f"at most {target['max_pieces']}",
            surface["pieces_90"] <= target["max_pieces"], "connected pieces that hold nine tenths of the surface"),
        row("closed", surface["open_edges"], f"at most {target['max_open_edges']:.0%}",
            surface["open_edges"] <= target["max_open_edges"], "edges with one face or more than two"),
        row("slivers", surface["slivers"], f"at most {target['max_slivers']:.0%}",
            surface["slivers"] <= target["max_slivers"], "triangles thinner than 1 to 20"),
        row("symmetry", surface["symmetry"], f"at least {target['min_symmetry']:.0%}",
            surface["symmetry"] >= target["min_symmetry"], "vertices with a mirror twin across X = 0"),
        row("materials", cost["materials"], f"at most {target['max_materials']}",
            cost["materials"] <= target["max_materials"], "materials in the file"),
    ]
    if (spec or {}).get("textured"):
        uv = reading.get("uv") or {"with_uv": 0.0, "outside": 0.0, "collapsed": 0.0}
        bad = max(1 - uv["with_uv"], uv["outside"], uv["collapsed"])
        out.append(row("uv", round(1 - bad, 4), "every triangle with its own place in the square",
                       bad <= 0.01, "triangles with UVs inside the square and not collapsed"))
    return out


def measure(path: str | Path, spec: dict | None = None, target: dict | None = None, *,
            surface_symmetry: bool = False) -> dict:
    """The readings of one GLB, and against `target` (or the spec's own, `target.apply`) the checks and their tally."""
    target = target or (spec or {}).get("target")
    mesh = read(path)
    tris = mesh["tris"]
    weld = welded(tris)
    area = areas(tris)
    labels = piece_labels(weld)
    _, counts = edges_of(weld)
    by_piece = np.bincount(np.unique(labels, return_inverse=True)[1], weights=area) if len(labels) else np.zeros(0)
    by_piece = np.sort(by_piece)[::-1]
    total = float(area.sum())
    most = int(np.searchsorted(np.cumsum(by_piece), MOST * total) + 1) if total > 0 else 0
    hidden, thin = buried_mask(tris), sliver_mask(tris)
    reading = {
        "file": Path(path).name,
        "cost": {"triangles": int(len(tris)), "vertices": int(mesh["vertices"]),
                 "positions": int(len(np.unique(weld))), "mesh_nodes": len(mesh["nodes"]),
                 "primitives": mesh["primitives"], "materials": mesh["materials"], "images": mesh["images"],
                 "bytes": mesh["bytes"]},
        "nodes": sorted(mesh["nodes"], key=lambda n: (-n["triangles"], n["name"])),
        "surface": {"pieces": int(len(by_piece)), "pieces_90": min(most, int(len(by_piece))),
                    "largest_piece": round(float(by_piece[0] / total), 4) if total > 0 else 0.0,
                    "open_edges": round(float(np.mean(counts != 2)), 4) if len(counts) else 0.0,
                    "slivers": round(float(thin.mean()) if len(tris) else 0.0, 4),
                    "buried": round(float(hidden.mean()) if len(tris) else 0.0, 4),
                    "symmetry": round(symmetry(tris), 4)},
        "by_node": by_node(mesh["nodes"], {"buried": hidden, "slivers": thin, "unmirrored": unmirrored_mask(tris)}),
        "uv": uv_reading(tris, mesh["uvs"]),
        "rig": {"skins": mesh["skins"], "animations": mesh["animations"]},
        "limbs": limbs(tris),
        "height_m": round(float(np.ptp(tris[..., 1])), 3) if len(tris) else 0.0,
    }
    if surface_symmetry:
        from .surface_symmetry import mirrored_surface_agreement
        reading["surface_symmetry"] = mirrored_surface_agreement(tris)
    if (spec or {}).get("max_triangles"):
        reading["max_triangles"] = int(spec["max_triangles"])
    if target:
        reading["target"] = target["name"]
        reading["checks"] = checks(reading, target, spec)
        reading["passed"] = sum(c["passed"] for c in reading["checks"])
    return reading


def _share(v: float) -> str:
    return f"{v:.0%}" if v >= 0.095 or v == 0 else f"{v:.1%}"


def by_node(nodes: list[dict], masks: dict[str, np.ndarray]) -> list[dict]:
    """Each node's own share of each per-triangle reading, its triangles in the file's order: which node a model's
    buried triangles, slivers and unmirrored corners are in, which builders had to guess."""
    out, start = [], 0
    for n in nodes:
        end = start + n["triangles"]
        row = {"name": n["name"], "triangles": n["triangles"]}
        for key, mask in masks.items():
            row[key] = int(mask[start:end].sum())
        out.append(row)
        start = end
    return out


def worst_nodes(reading: dict, key: str, top: int = 3) -> str:
    """'Vest 300 of 750, Skin 120 of 3000': the nodes holding most of a reading's triangles."""
    rows = sorted((r for r in reading.get("by_node") or [] if r.get(key)), key=lambda r: -r[key])[:top]
    return ", ".join(f"{r['name']} {r[key]} of {r['triangles']}" for r in rows)


def lines(reading: dict) -> list[str]:
    """The reading as the feedback and the CLI print it."""
    cost, surface = reading["cost"], reading["surface"]
    out = []
    if reading.get("checks"):
        failed = [c for c in reading["checks"] if not c["passed"]]
        out.append(f"MESH READINESS for {reading['target']}: {reading['passed']} of {len(reading['checks'])} hold"
                   + ("" if not failed else "; NOT: " + "; ".join(
                       f"{c['name']} ({c['value'] if isinstance(c['value'], int) else _share(c['value'])}, "
                       f"wants {c['limit']}: {c['what']})" for c in failed)))
    else:
        out.append("MESH READINGS (no target is set, so nothing is held to a limit):")
    out.append(f"  cost: {cost['triangles']} triangles, {cost['vertices']} vertices ({cost['positions']} positions), "
               f"{cost['mesh_nodes']} mesh nodes, {cost['primitives']} primitives, {cost['materials']} materials, "
               f"{cost['bytes'] / 1024:.0f} KiB")
    out.append(f"  surface: {surface['pieces']} connected pieces, {surface['pieces_90']} of them nine tenths of the "
               f"area and the largest {_share(surface['largest_piece'])}; buried {_share(surface['buried'])}; open or over-shared edges {_share(surface['open_edges'])}; "
               f"slivers {_share(surface['slivers'])}; mirror symmetry {_share(surface['symmetry'])}")
    surface_symmetry = reading.get("surface_symmetry")
    if surface_symmetry is not None:
        if surface_symmetry.get("status") == "available":
            out.append("  mirrored surface across X=0: "
                       f"left to right {_share(surface_symmetry['left_to_right'])}, "
                       f"right to left {_share(surface_symmetry['right_to_left'])}, "
                       f"conservative {_share(surface_symmetry['agreement'])} "
                       f"({surface_symmetry['sample_count']} samples; "
                       f"{surface_symmetry['tolerance_mm']:g} mm tolerance; approximate)")
        else:
            out.append(f"  mirrored surface across X=0: unavailable ({surface_symmetry.get('reason', 'unknown')})")
        out.append("  This is approximate unsigned geometry weighted over all triangle area, including internal, "
                   "overlapping or duplicate surfaces; it is not visible-surface or union coverage, and large "
                   "matching areas can mask small asymmetric features. It cannot certify colour, winding, topology "
                   "or rigging. "
                   "The existing vertex-twin reading depends on correspondence and can fall after retessellation "
                   "while surfaces still match. Check the task's symmetry and topology requirements, inspect the "
                   "visible views, preserve the matching form, and allow intentional asymmetry; no construction "
                   "technique is required.")
    # where a failed check's triangles are, by node: builders guessed it from the share alone
    failed = {c["name"] for c in reading.get("checks") or [] if not c["passed"]}
    for check, key in (("buried", "buried"), ("slivers", "slivers"), ("symmetry", "unmirrored")):
        if (check in failed or not reading.get("checks")) and worst_nodes(reading, key):
            out.append(f"  {key} by node (triangles): {worst_nodes(reading, key)}")
    top = reading["nodes"][:4]
    if len(reading["nodes"]) > 1:
        out.append("  largest nodes: " + ", ".join(f"{n['name']} {n['triangles']}" for n in top))
    uv = reading.get("uv")
    if uv:
        out.append(f"  uv: {_share(uv['with_uv'])} of triangles have UVs; outside the square {_share(uv['outside'])}; "
                   f"collapsed {_share(uv['collapsed'])}; texel density spread {uv['texel_spread']:g}x; square used "
                   f"{uv['used']:.2f}")
    else:
        out.append("  uv: none (the texturing stage unwraps)")
    if reading.get("limbs"):
        height = reading.get("height_m") or 0.0
        said = []
        for limb in reading["limbs"]:
            share = f", {limb['clear_m'] / height:.0%} of the height" if height else ""
            span = (f"from {limb['from_m']:.2f} m down to {limb['from_m'] - limb['clear_m']:.2f} m above the ground "
                    f"({limb['clear_m']:.2f} m{share})")
            joined = f", joined to the body for {limb['joined_m']:.2f} m above that" if limb["joined_m"] else ""
            if limb["on_ground"]:
                said.append(f"legs: {limb['parts']} stand apart {span} and reach the ground{joined}")
            elif _low(limb, height):
                said.append(f"legs: {limb['parts']} parts stand apart {span} but are joined again below that, "
                            f"so they do not stand apart down to the ground (feet fused together?){joined}")
            else:
                said.append(f"arms (or a tail): {limb['parts']} parts side by side, the body among them, {span}"
                            f"{joined}")
        if arms_clear(reading) == 0:
            # what a rig made most of: arms modelled into the flank (or nearer it than a cut can part) all the way
            said.append(f"no arm hangs clear of the body: an arm, if there is one, is joined to it or nearer it than "
                        f"{rig_brief.GAP_M * 100:.0f} cm all the way down. Leave a gap of at least that between arm "
                        "and flank below the shoulder; a voxel remesh fuses gaps under about twice its voxel size")
        out.append("  limbs (a rig bends a limb only where it hangs clear of the body; where it is joined, the body "
                   "stretches with it): " + "; ".join(said))
    return out


def _low(limb: dict, height: float) -> bool:
    """A stretch that ends near the ground without reaching it: legs apart above feet that join again."""
    return bool(height) and limb["from_m"] - limb["clear_m"] < LOW_SHARE * height


def arms_clear(reading: dict) -> float | None:
    """How long the longest stretch above the legs hangs clear of the body, in metres: 0 when nothing does, None
    when the reading has no limbs (a record from before them, or a model with no stretch at all)."""
    if not reading.get("limbs"):
        return None
    height = reading.get("height_m") or 0.0
    return max([limb["clear_m"] for limb in reading["limbs"]
                if not limb["on_ground"] and not _low(limb, height)], default=0.0)


def surface_summary(reading: dict) -> str:
    """The optional sampled surface reading as a concise feedback line."""
    surface_symmetry = reading.get("surface_symmetry")
    if surface_symmetry is None:
        return ""
    if surface_symmetry.get("status") == "available":
        said = (f"sampled mirrored-surface agreement {_share(surface_symmetry['agreement'])} "
                f"(approximate; {surface_symmetry['sample_count']} samples at "
                f"{surface_symmetry['tolerance_mm']:g} mm; vertex correspondence is separate)")
    else:
        said = (f"sampled mirrored-surface agreement unavailable "
                f"({surface_symmetry.get('reason', 'unknown')}; vertex correspondence is separate)")
    return "mesh: " + said


def summary(reading: dict, *, include_surface: bool = True) -> str:
    """One line for the top of the feedback: the triangles against the ceiling, and the limbs when a rig would
    find them joined. An opt-in sampled surface reading is included when present. '' when there is nothing to say."""
    said = []
    tris, cap = reading["cost"]["triangles"], reading.get("max_triangles")
    if cap:
        said.append(f"{tris} of {cap} triangles ({tris / cap:.0%})")
    clear = arms_clear(reading)
    if clear == 0:
        said.append("no arm hangs clear of the body (see limbs below)")
    if reading.get("limbs") and not any(limb["on_ground"] for limb in reading["limbs"]):
        said.append("the legs do not stand apart down to the ground (see limbs below)")
    surface = surface_summary(reading) if include_surface else ""
    if surface:
        said.append(surface.removeprefix("mesh: "))
    return "mesh: " + "; ".join(said) if said else ""
