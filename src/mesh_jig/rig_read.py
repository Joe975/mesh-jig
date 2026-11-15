"""What a built rig does, read from the GLB with numpy: said to its author beside the contract's gates.

The contract check says whether a rig is in contract. A rig in contract can still drag a patch of hip up with an
arm, leave a required bone holding nothing, walk with both legs together or hang in the air, and the first builds
of a rig did each of those and passed. None of these readings is a gate or a score. Each is a line of the feedback
that names what a picture would show, with where and how much, in the script's own axes (Blender metres).

- `islands`: a bone's weight in more than one patch of one connected piece. The small patch is a stray weight.
- `holdings`: what each bone holds. A bone that holds nothing moves nothing.
- `motion`: for each clip, how far the part each bone holds travels (fore and aft, up and down, across), which
  left and right parts move together or in turn, and where the lowest point of the model is against the ground.
- `resting` and `touching`: what lies on what at rest, followed through each clip. On one skin, each thing worn
  against the skin; on a model in pieces, each pair of nodes that touch.

Deterministic. Nothing here knows what the subject is: left and right are the `.L` and `.R` of the bones' names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import glb, rig_look, skin

HELD = 0.05                   # a corner counts toward a bone's patch above this weight
ACROSS_SHOWN = 20             # bones in the table that sets the clips beside each other
STRAY_SHOWN = 8               # patches listed in the feedback, largest first
STRAY_MIN = 4                 # welded vertices: a smaller patch than this is not reported
WORN_M = 0.03                 # a node's corner this close to the largest node at rest lies on it
WORN_SAMPLE = 200             # corners of a worn node that are followed
WORN_SLIDES_MM = 15.0         # further than this from where its place under it has gone, and it is said
BULK = 0.5                    # a node with this share of the triangles is the skin; with none, the model is in pieces
TOUCH_M = 0.02                # corners of two nodes this close at rest touch
TOUCH_SAMPLE = 200            # corners of a node that are tried against the other nodes
PARTING_SHOWN = 6             # pairs listed in the feedback, the widest first
FOLDED = 0.5                  # an edge under this share of its rest length: the inside of a bend closing on itself
STRETCHED = 1.5               # an edge past this many times its rest length, as the contract's stretch gate counts
MOVES_MM = 5.0                # a part that travels less than this in a clip is still
NEAR_GROUND_MM = 8.0
DRIFT_MPS = 0.02              # feet moving slower than this while down, either way, stay put
FOOT_M = 0.03                 # a bone whose part rests this close to the ground may be a foot
SOLE_M = 0.02                 # a foot's sole: its corners this close to its lowest at rest
LIFTS_M = 0.005               # a foot that lifts less than this in a clip says nothing about direction
CONTACT_GAP_MM = 5.0          # ground contact as a game would read it: a vertex this near the rest ground touches it
STANCE_S = 0.05               # a touch shorter than this is a graze, not a stance
SKATE_SAID_MPS = 0.05         # a planted contact whose skate passes this is marked
BELOW_SAID_MM = 1.0           # a clip that goes this far below the rest ground between keys is marked


def to_blender(points: np.ndarray) -> np.ndarray:
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def weld(points: np.ndarray) -> np.ndarray:
    """An id per point, the same for points at one position (1e-6 m, as the rig's pieces are welded)."""
    return np.unique(np.round(points, 6), axis=0, return_inverse=True)[1].ravel()


def labels(count: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of `count` vertices under edges a-b: the least vertex id of each one's component."""
    label = np.arange(count)
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]                                        # pointer jumping: halves the passes a long limb needs
        if np.array_equal(new, label):
            return label
        label = new


def triangle_nodes(doc: dict) -> np.ndarray:
    """The name of the mesh node each triangle belongs to, in `render.load`'s triangle order."""
    names = []
    for ni, _ in glb.mesh_nodes(doc):
        node = doc["nodes"][ni]
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            a = doc["accessors"]
            count = (a[prim["indices"]]["count"] if "indices" in prim else a[prim["attributes"]["POSITION"]]["count"]) // 3
            names += [node.get("name", f"node{ni}")] * count
    return np.array(names)


def resting(rest: np.ndarray, nodes: np.ndarray) -> dict[str, tuple]:
    """What is worn: each node that lies on the largest node (most of a sample of its corners within WORN_M of
    it at rest), as (its sampled corners, the corner of the largest node nearest each, their distance at rest).
    A node joined to the body at one end only (a tail) is a limb, not something worn, and is left out."""
    names, counts = np.unique(nodes, return_counts=True)
    if len(names) < 2 or counts.max() < BULK * counts.sum():
        return {}
    under = names[counts.argmax()]
    base = np.argwhere(np.repeat((nodes == under)[:, None], 3, axis=1))
    base_points = rest[base[:, 0], base[:, 1]]
    out = {}
    for name in names:
        if name == under:
            continue
        corners = np.argwhere(np.repeat((nodes == name)[:, None], 3, axis=1))
        corners = corners[:: max(1, len(corners) // WORN_SAMPLE)]
        points = rest[corners[:, 0], corners[:, 1]]
        near = np.array([int(np.argmin(((base_points - p) ** 2).sum(axis=1))) for p in points])
        d0 = np.linalg.norm(points - base_points[near], axis=1)
        on = d0 <= WORN_M
        if on.mean() >= 0.5:
            out[str(name)] = (corners[on], base[near][on], d0[on])
    return out


def touching(rest: np.ndarray, nodes: np.ndarray) -> dict[tuple[str, str], tuple]:
    """A model in pieces (no node is BULK of it): each pair of nodes with corners within TOUCH_M of each other at
    rest, as (the corners of one, the corner of the other nearest each, their distance at rest). The stretch gate
    reads edges, and no edge joins two nodes: a sword in four nodes bent at the hip, and a wrap left behind by the
    arm it is wound on, each passed every gate. Corner to corner, so two large flat faces that touch far from
    their corners are not found."""
    names, counts = np.unique(nodes, return_counts=True)
    if len(names) < 2 or counts.max() >= BULK * counts.sum():
        return {}
    corners = np.argwhere(np.ones(rest.shape[:2], bool))                  # every (triangle, corner)
    owner = np.repeat(nodes, 3)
    points = rest.reshape(-1, 3)
    which = np.unique(owner, return_inverse=True)[1].ravel()
    _, first = np.unique(np.c_[np.round(points, 6), which], axis=0, return_index=True)    # one corner a position a node
    first = np.sort(first)
    out = {}
    for name in names:
        mine = first[owner[first] == name]
        mine = mine[:: max(1, len(mine) // TOUCH_SAMPLE)]
        other = first[owner[first] != name]
        d = np.linalg.norm(points[mine][:, None] - points[other][None], axis=2)
        near = d.argmin(axis=1)
        d0 = d[np.arange(len(mine)), near]
        for there in np.unique(owner[other[near]][d0 <= TOUCH_M]):
            on = (d0 <= TOUCH_M) & (owner[other[near]] == there)
            key = tuple(sorted((str(name), str(there))))
            a, b, at = out.get(key, (np.zeros((0, 2), int), np.zeros((0, 2), int), np.zeros(0)))
            out[key] = (np.concatenate([a, corners[mine[on]]]), np.concatenate([b, corners[other[near[on]]]]),
                        np.concatenate([at, d0[on]]))
    return out


def _mesh(rigged):
    doc, blob = glb.read_glb(rigged)
    tris = to_blender(skin.posed(rigged).tris)                # (T, 3, 3) at rest
    return doc, blob, tris, rig_look.corner_weights(doc, blob), rig_look.joint_names(doc)


def islands(rigged: str | Path) -> list[dict]:
    """Stray weights: for each bone, each patch of its weight that is cut off from a larger patch of it inside one
    connected piece of the mesh. -> [{"bone", "vertices", "centre", "lo", "hi", "main_vertices", "main_centre"}],
    largest first."""
    doc, blob, tris, weights, names = _mesh(rigged)
    if not len(names):
        return []
    ids = weld(tris.reshape(-1, 3)).reshape(-1, 3)
    count = int(ids.max()) + 1
    points = np.zeros((count, 3))
    points[ids.ravel()] = tris.reshape(-1, 3)
    edges_a, edges_b = np.concatenate([ids[:, 0], ids[:, 1]]), np.concatenate([ids[:, 1], ids[:, 2]])
    piece = labels(count, edges_a, edges_b)
    held = np.zeros((count, len(names)))
    np.maximum.at(held, ids.ravel(), weights.reshape(-1, len(names)))
    found = []
    for k, bone in enumerate(names):
        on = held[:, k] > HELD
        if not on.any():
            continue
        whole = on[ids].all(axis=1)                           # a triangle wholly in the bone's patch joins its corners
        t = ids[whole]
        patch = labels(count, np.concatenate([t[:, 0], t[:, 1]]), np.concatenate([t[:, 1], t[:, 2]]))
        for p in np.unique(piece[on]):
            inside = on & (piece == p)
            roots, sizes = np.unique(patch[inside], return_counts=True)
            if len(roots) < 2:
                continue
            main = roots[sizes.argmax()]
            main_pts = points[inside & (patch == main)]
            for r, size in zip(roots, sizes):
                if r == main or size < STRAY_MIN:
                    continue
                pts = points[inside & (patch == r)]
                found.append({"bone": bone, "vertices": int(size), "centre": pts.mean(axis=0).round(3).tolist(),
                              "lo": pts.min(axis=0).round(3).tolist(), "hi": pts.max(axis=0).round(3).tolist(),
                              "weight": round(float(held[inside & (patch == r), k].max()), 3),
                              "main_vertices": int(sizes.max()), "main_centre": main_pts.mean(axis=0).round(3).tolist(),
                              "main_lo": main_pts.min(axis=0).round(3).tolist(),
                              "main_hi": main_pts.max(axis=0).round(3).tolist()})
    return sorted(found, key=lambda f: -f["vertices"])


def separate(s: dict) -> str:
    """Which test on position leaves a stray patch out and keeps the main one: the axis the two are furthest apart
    on, and a cut between them. A builder told only to tighten a test took two builds to find which one."""
    if "main_lo" not in s:
        return ""
    best, axis, cut, side = 0.0, None, 0.0, ""
    for k, name in enumerate("xyz"):
        above = s["lo"][k] - s["main_hi"][k]           # the stray patch past the main one's high end
        below = s["main_lo"][k] - s["hi"][k]           # or short of its low end
        if above > best:
            best, axis, cut, side = above, name, (s["lo"][k] + s["main_hi"][k]) / 2, "<"
        if below > best:
            best, axis, cut, side = below, name, (s["hi"][k] + s["main_lo"][k]) / 2, ">"
    if axis is None:
        return ("It lies within the main patch's box: a distance to the bone's line (a smaller radius), not a cut on "
                "one axis, is what leaves it out.")
    return (f"A cut on {axis} leaves it out: {s['bone']}'s test needs {axis} {side} {cut:+.2f} (the two are "
            f"{best:.2f} m apart on {axis}).")


def root_bones(doc: dict) -> set[str]:
    """The joints that hang from no other joint. A root passes motion down and is expected to hold nothing."""
    joints = [j for skin_ in doc.get("skins", []) for j in skin_["joints"]]
    children = {c for j in joints for c in doc["nodes"][j].get("children", [])}
    return {doc["nodes"][j].get("name", f"joint{j}") for j in joints if j not in children}


def holdings(rigged: str | Path) -> dict[str, dict]:
    """bone -> what it holds: the corners it is the strongest bone of (count, bounds, centre), or count 0."""
    _, _, tris, weights, names = _mesh(rigged)
    strongest = weights.argmax(axis=2) if len(names) else np.zeros(tris.shape[:2], int)
    out = {}
    for k, bone in enumerate(names):
        pts = tris[strongest == k]
        out[bone] = ({"corners": int(len(pts)), "lo": pts.min(axis=0).round(3).tolist(), "hi": pts.max(axis=0).round(3).tolist(),
                      "centre": pts.mean(axis=0).round(3).tolist()} if len(pts) else {"corners": 0})
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-9 else None


def motion(rigged: str | Path, spec: dict) -> dict[str, dict]:
    """clip -> {"parts": {bone: travel of the part it holds, mm, as fore_aft / up_down / across and where it is at
    its extremes}, "pairs": {"thigh": correlation of the left and right parts' fore-aft travel}, "ground": {...}}."""
    doc, blob, rest, weights, names = _mesh(rigged)
    strongest = weights.argmax(axis=2) if len(names) else np.zeros(rest.shape[:2], int)
    ground = float(rest[..., 2].min())
    lying = resting(rest, triangle_nodes(doc))
    pairs = touching(rest, triangle_nodes(doc))
    out = {}
    for clip, c in spec["clips"].items():
        if not any(a.get("name") == clip for a in doc.get("animations", [])):
            continue
        keys = round(c["seconds"] * spec.get("fps", 30))
        times = np.linspace(0, c["seconds"], keys + 1)[:-1] if c.get("loop") else np.linspace(0, c["seconds"], keys + 1)
        frames = [to_blender(skin.posed(rigged, clip, float(t)).tris) for t in times]
        low = np.array([f[..., 2].min() for f in frames]) - ground
        deepest = frames[int(low.argmin())]
        corner = np.unravel_index(int(deepest[..., 2].argmin()), deepest.shape[:2])
        parts, tracks = {}, {}
        for k, bone in enumerate(names):
            mask = strongest == k
            if not mask.any():
                continue
            track = np.array([f[mask].mean(axis=0) for f in frames])
            tracks[bone] = track
            span = np.ptp(track, axis=0) * 1000
            parts[bone] = {"across_mm": round(float(span[0]), 1), "fore_aft_mm": round(float(span[1]), 1),
                           "up_down_mm": round(float(span[2]), 1),
                           "furthest_forward_s": round(float(times[track[:, 1].argmax()]), 3),
                           "highest_s": round(float(times[track[:, 2].argmax()]), 3)}
        sides = {}
        for bone in names:
            if bone.endswith(".L") and bone[:-2] + ".R" in tracks and bone in tracks:
                left, right = tracks[bone], tracks[bone[:-2] + ".R"]
                if max(np.ptp(left[:, 1]), np.ptp(right[:, 1])) * 1000 >= MOVES_MM:
                    value = corr(left[:, 1], right[:, 1])
                    if value is not None:
                        sides[bone[:-2]] = round(value, 2)
        slides = {}
        for node, (corners, under, d0) in lying.items():
            worst, at = 0.0, 0.0
            for t, f in zip(times, frames):
                d = np.abs(np.linalg.norm(f[corners[:, 0], corners[:, 1]] - f[under[:, 0], under[:, 1]], axis=1) - d0).max()
                if d > worst:
                    worst, at = float(d), float(t)
            slides[node] = {"mm": round(worst * 1000, 1), "at_s": round(at, 3)}
        parting = {}
        for (one, other), (a, b, d0) in pairs.items():
            worst, at, where = 0.0, 0.0, rest[a[0, 0], a[0, 1]]
            for t, f in zip(times, frames):
                d = np.abs(np.linalg.norm(f[a[:, 0], a[:, 1]] - f[b[:, 0], b[:, 1]], axis=1) - d0)
                if d.max() > worst:
                    worst, at, where = float(d.max()), float(t), rest[a[d.argmax(), 0], a[d.argmax(), 1]]
            parting[f"{one}/{other}"] = {"mm": round(worst * 1000, 1), "at_s": round(at, 3), "at": where.round(3).tolist()}
        out[clip] = {"seconds": c["seconds"], "loop": bool(c.get("loop")),
                     "stance": stance(frames, times, rest, strongest, names, ground, bool(c.get("loop"))),
                     "parts": parts, "pairs": sides, "worn": slides, "parting": parting,
                     "stretch": stretched(rest, frames, times, weights, names, doc, blob, clip),
                     "ground": {"sinks_mm": round(float(max(0.0, -low.min()) * 1000), 1),
                                "sinks_at_s": round(float(times[low.argmin()]), 3),
                                "lowest_is": names[strongest[corner]] if len(names) else "",
                                "lowest_rest": rest[corner].round(3).tolist(),
                                "lowest_point_highest_mm": round(float(low.max() * 1000), 1),
                                "highest_at_s": round(float(times[low.argmax()]), 3),
                                "share_of_keys_on_ground": round(float((low * 1000 <= NEAR_GROUND_MM).mean()), 2)}}
    return out


def stance(frames: list, times: np.ndarray, rest: np.ndarray, strongest: np.ndarray, names: list, ground: float,
           loop: bool) -> dict | None:
    """Which way each foot goes while it is down. A foot is a bone whose part rests within FOOT_M of the ground and
    lifts at least LIFTS_M in the clip; its sole is that part's corners within SOLE_M of its lowest at rest. It is
    down at the keys where its sole is within max(NEAR_GROUND_MM, a quarter of its lift) of the sole's lowest in the
    clip, so a foot that only grazes the ground is read too. Its backward speed (Blender -Y, m/s) is the mean over
    pairs of such keys in a row (the last and the first, for a loop). Feet under a body going forward on the spot
    are carried backward; a gait that plays backwards reads negative. None when no foot lifts."""
    if len(times) < 2:
        return None
    frames = np.asarray(frames)
    dt = float(times[1] - times[0])
    feet = {}
    for k, bone in enumerate(names):
        mask = strongest == k
        if not mask.any() or rest[mask][:, 2].min() - ground > FOOT_M:
            continue
        where = np.argwhere(mask)
        low = where[rest[mask][:, 2] <= rest[mask][:, 2].min() + SOLE_M]
        height = frames[:, low[:, 0], low[:, 1], 2].min(axis=1)
        lift = float(height.max() - height.min())
        if lift < LIFTS_M:
            continue
        y = frames[:, low[:, 0], low[:, 1], 1].mean(axis=1)
        down = height <= height.min() + max(NEAR_GROUND_MM / 1000, 0.25 * lift)
        back = (y - np.roll(y, -1)) / dt
        pairs = down & np.roll(down, -1)
        if not loop:
            pairs[-1] = False
        if pairs.any():
            feet[bone] = round(float(back[pairs].mean()), 3)
    if not feet:
        return None
    return {"backward_mps": round(float(np.median(list(feet.values()))), 3), "feet": feet}


def carries(doc: dict, blob: bytes, clip: str, t: float) -> np.ndarray:
    """(joints, 4, 4): where each joint of the skin carries a point of the rest pose at this time, in glTF axes."""
    joints = doc["skins"][0]["joints"]
    then, at_rest = skin.sample(doc, blob, clip, t), skin.worlds(doc)
    return np.stack([then[j] @ np.linalg.inv(at_rest[j]) for j in joints])


def stretched(rest, frames, times, weights, names, doc=None, blob=None, clip=None) -> dict:
    """Where a clip stretches the surface at its worst key: the triangles with an edge past 1.5 times its rest
    length, grouped by the bone that holds most of each and the bone that PULLS it: the one whose weight differs
    most from corner to corner, counted by how far that bone would carry the triangle from where its main bone
    takes it. Naming the two strongest bones instead named a belly's hips and spine, which were not moving, for a
    patch a forearm's faint weight was dragging, and a builder spent three builds on the wrong joint."""
    base = np.linalg.norm(rest - np.roll(rest, 1, axis=1), axis=2)
    areas = np.linalg.norm(np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=1) / 2
    worst, worst_mask, at, folded, folded_at = 0.0, None, 0.0, 0.0, 0.0
    for t, f in zip(times, frames):
        ratio = np.divide(np.linalg.norm(f - np.roll(f, 1, axis=1), axis=2), base, out=np.ones_like(base), where=base > 1e-8)
        mask = (ratio > STRETCHED).any(axis=1)
        share = float(areas[mask].sum() / max(areas.sum(), 1e-12))
        if share > worst:
            worst, worst_mask, at = share, mask, float(t)
        # the inside of a bend: the stretch gate reads edges that lengthen and has no word for ones that collapse
        squeezed = float(areas[(ratio < FOLDED).any(axis=1)].sum() / max(areas.sum(), 1e-12))
        if squeezed > folded:
            folded, folded_at = squeezed, float(t)
    if worst_mask is None or not len(names):
        return {"share": 0.0, "at_s": 0.0, "where": [], "folded_share": round(folded, 4), "folded_at_s": round(folded_at, 3)}
    w = weights[worst_mask]                                    # (stretched triangles, corners, joints)
    main = w.mean(axis=1).argmax(axis=1)
    spread = w.max(axis=1) - w.min(axis=1)                     # how much each bone's weight differs across the corners
    centres = rest[worst_mask].mean(axis=1)
    if doc is not None and clip is not None:
        c = np.c_[centres[:, 0], centres[:, 2], -centres[:, 1], np.ones(len(centres))]       # back to glTF's axes
        taken = np.einsum("jab,nb->nja", carries(doc, blob, clip, at), c)[..., :3]           # by every bone
        apart = np.linalg.norm(taken - taken[np.arange(len(c)), main][:, None], axis=2)      # from the main bone's
        spread = spread * apart
    spread[np.arange(len(main)), main] = 0.0
    puller = spread.argmax(axis=1)
    pulled = spread.max(axis=1) > 1e-9
    groups = {}
    for k in range(len(main)):
        key = (names[main[k]], names[puller[k]]) if pulled[k] else (names[main[k]],)
        groups.setdefault(key, []).append(k)
    total = max(areas.sum(), 1e-12)
    where = [{"bones": list(key), "share": round(float(areas[worst_mask][rows].sum() / total), 4),
              "centre": centres[rows].mean(axis=0).round(3).tolist()} for key, rows in groups.items()]
    return {"share": round(worst, 4), "at_s": round(at, 3), "where": sorted(where, key=lambda w: -w["share"])[:4],
            "folded_share": round(folded, 4), "folded_at_s": round(folded_at, 3)}


def _at(p) -> str:
    return "(" + ", ".join(f"{float(v):+.2f}" for v in p) + ")"


def text(rigged: str | Path, spec: dict, report: dict | None = None) -> str:
    """The readings as the lines a builder is given after a build that held its contract."""
    lines = []
    held = holdings(rigged)
    # a root holding nothing is as it should be, and saying so beside the faults had a builder give its root the
    # ear tips "so it is a real bone"
    roots = root_bones(glb.read_glb(rigged)[0])
    empty = [b for b, h in held.items() if not h["corners"] and b not in roots]
    lines.append("WHAT EACH BONE HOLDS (the vertices it is the strongest bone of; bounds in Blender metres):")
    for bone, h in held.items():
        lines.append(f"  {bone}: nothing" + (" (the root: as it should be)" if bone in roots else "") if not h["corners"] else
                     f"  {bone}: {h['corners'] // 3} triangles' worth, lo {_at(h['lo'])} hi {_at(h['hi'])}")
    if empty:
        lines.append("! " + ", ".join(empty) + (" holds" if len(empty) == 1 else " hold") + " nothing: no vertex "
                     "has it as its strongest bone, so posing it moves only its children. Its weight test catches "
                     "nothing, or another bone's test wins everywhere it reaches.")
    stray = islands(rigged)
    if stray:
        lines.append("STRAY WEIGHTS (a patch of a bone's weight cut off from that bone's main patch on the same "
                     "connected surface; when the bone turns, this patch is dragged with it. Tighten that bone's "
                     "test so it leaves the place out. A left bone with a patch on the right side, or the reverse, "
                     "is a test that forgot the sign of x):")
        for s in stray[:STRAY_SHOWN]:
            lines.append(f"! {s['bone']}: {s['vertices']} vertices round {_at(s['centre'])}, lo {_at(s['lo'])} hi "
                         f"{_at(s['hi'])}, weight up to {s['weight']:.2f}; its main patch is {s['main_vertices']} "
                         f"round {_at(s['main_centre'])}. {separate(s)}")
        if len(stray) > STRAY_SHOWN:
            lines.append(f"  ...and {len(stray) - STRAY_SHOWN} smaller patches")
    else:
        lines.append("STRAY WEIGHTS: none found.")
    moves = motion(rigged, spec)
    if len(moves) > 1:
        # one clip beside another: a builder asked for a run "clearly bigger" than its walk delivered one whose
        # feet travelled less than the walk's, with each clip's own numbers in front of it and no line that put
        # the two together
        size = {b: max(max(m["parts"].get(b, {}).get(k, 0.0) for k in ("fore_aft_mm", "up_down_mm", "across_mm"))
                       for m in moves.values()) for b in held}
        most = sorted((b for b in held if size[b] >= MOVES_MM), key=lambda b: -size[b])[:ACROSS_SHOWN]
        if most:
            lines.append("THE CLIPS BESIDE EACH OTHER (fore-aft / up-down travel of the part each bone holds, mm; the "
                         f"{len(most)} that move most). Where the brief wants one clip bigger, livelier or quieter "
                         "than another, this is where it shows:")
            for bone in most:
                lines.append(f"  {bone}: " + ", ".join(
                    f"{clip} {m['parts'].get(bone, {}).get('fore_aft_mm', 0):.0f}/{m['parts'].get(bone, {}).get('up_down_mm', 0):.0f}"
                    for clip, m in moves.items()))
    for clip, m in moves.items():
        r = (report or {}).get("clips", {}).get(clip, {})
        head = f"CLIP {clip} ({m['seconds']:g} s)"
        if r:
            head += (f": loop seam {r['loop_mm']:.2f} mm, furthest any point goes from rest {r['max_displacement_mm']:.0f} mm, "
                     f"{r['stretch_area_fraction'] * 100:.2f}% of the surface stretched past 1.5x (worst edge "
                     f"{r['max_edge_ratio']:.1f}x)")
        lines.append(head)
        moving = {b: p for b, p in m["parts"].items()
                  if max(p["across_mm"], p["fore_aft_mm"], p["up_down_mm"]) >= MOVES_MM}
        still = [b for b in m["parts"] if b not in moving]
        if moving:
            lines.append("  how far the part each bone holds travels, mm (and when it is furthest forward / highest, s):")
        for bone, p in moving.items():
            lines.append(f"    {bone}: fore-aft {p['fore_aft_mm']:.0f} ({p['furthest_forward_s']:.2f}), up-down "
                         f"{p['up_down_mm']:.0f} ({p['highest_s']:.2f}), across {p['across_mm']:.0f}")
        if still:
            lines.append(f"  still (under {MOVES_MM:g} mm): " + ", ".join(still))
        # said flat, with no mark: arms that breathe together and ears that bounce together are meant, and a
        # reading in capitals was taken for a fault by the first builder shown one
        turns = [pair for pair, value in m["pairs"].items() if value <= -0.5]
        both = [pair for pair, value in m["pairs"].items() if value >= 0.5]
        other = [pair for pair in m["pairs"] if pair not in turns and pair not in both]
        if turns:
            lines.append("  left and right take turns (one forward as the other is back): " + ", ".join(turns))
        if both:
            lines.append("  left and right move together (both forward at once): " + ", ".join(both))
        if other:
            lines.append("  left and right neither together nor in turn: " + ", ".join(other))
        for w in m["stretch"]["where"]:
            if w["share"] >= 0.002:
                lines.append(f"  stretched at {m['stretch']['at_s']:.2f} s: {w['share'] * 100:.2f}% of the surface "
                             + (f"in the part {w['bones'][0]} holds, pulled by {w['bones'][1]}'s weight" if len(w["bones"]) == 2
                                else "inside " + w["bones"][0])
                             + f", round {_at(w['centre'])} at rest")
        off = {n: w for n, w in m["worn"].items() if w["mm"] >= WORN_SLIDES_MM}
        if off:
            lines.append("! comes away from the surface it lies on (it and the surface nearest it follow different "
                         "bones there: either it is weighted apart from what it covers, or a limb's region reaches "
                         "under it and takes the surface but not it; one function for both cures the first, a "
                         "narrower limb region the second): "
                         + ", ".join(f"{n} by {w['mm']:.0f} mm at {w['at_s']:.2f} s" for n, w in off.items()))
        apart = sorted(((w["mm"], pair, w) for pair, w in m["parting"].items() if w["mm"] >= WORN_SLIDES_MM), reverse=True)
        if apart:
            # said flat: two flaps of a skirt are meant to open, and marked as a fault they would be pinned shut
            lines.append("  nodes that touch at rest and part (the two follow different bones where they meet. Meant, "
                         "for two flaps that open. A fault for ONE thing in several nodes, a sword or a strap: "
                         "give every node of it the same one bone by name. A fault for a layer round a limb: give "
                         "it the limb's own tests on position): "
                         + "; ".join(f"{pair.replace('/', ' and ')} by {w['mm']:.0f} mm at {w['at_s']:.2f} s round {_at(w['at'])}"
                                     for _, pair, w in apart[:PARTING_SHOWN])
                         + (f"; and {len(apart) - PARTING_SHOWN} more" if len(apart) > PARTING_SHOWN else ""))
        if m["stretch"]["folded_share"] >= 0.002:
            lines.append(f"  folded at {m['stretch']['folded_at_s']:.2f} s: {m['stretch']['folded_share'] * 100:.2f}% of the "
                         "surface has an edge squeezed under half its length (the inside of a bend closing on itself; "
                         "no gate reads this)")
        g = m["ground"]
        if g["sinks_mm"] > 1:
            lines.append(f"! goes {g['sinks_mm']:.0f} mm below the ground it stands on at rest (worst at "
                         f"{g['sinks_at_s']:.2f} s): the lowest point then is on {g['lowest_is']}'s part, the vertex "
                         f"that rests at {_at(g['lowest_rest'])}")
        if g["share_of_keys_on_ground"] == 0:
            lines.append(f"! never touches the ground: the lowest point of the model is at least "
                         f"{g['lowest_point_highest_mm']:.0f} mm above it at some key and never within {NEAR_GROUND_MM:g} mm")
        else:
            lines.append(f"  ground: something is within {NEAR_GROUND_MM:g} mm of it in {g['share_of_keys_on_ground'] * 100:.0f}% of "
                         f"the keys; the lowest point is highest, {max(0.0, g['lowest_point_highest_mm']):.0f} mm up, at "
                         f"{g['highest_at_s']:.2f} s")
        st = m.get("stance")
        if m.get("loop") and st:
            speed, each = st["backward_mps"], ", ".join(f"{b} {v:+.2f}" for b, v in st["feet"].items())
            if speed < -DRIFT_MPS:
                lines.append(f"! feet slide FORWARD while they are down ({-speed:.2f} m/s; backward speed by foot: "
                             f"{each}). A gait on the spot that does this plays backwards, and no gate reads it: feet "
                             "under a body going forward are carried toward the back while they are down")
            elif speed > DRIFT_MPS:
                lines.append(f"  feet are carried backward while they are down ({speed:.2f} m/s; by foot: {each}), "
                             "as under a body going forward on the spot")
            else:
                lines.append(f"  feet stay put while they are down (by foot: {each}): a clip that steps in place or "
                             "travels over the ground")
    lines += contact_text(ground_contacts(rigged, spec))
    return "\n".join(lines)


def ground_contacts(rigged: str | Path, spec: dict) -> dict[str, dict]:
    """clip -> how the clip meets the ground as the game-ready stage reads it, with no contract: how far the model
    goes below its rest ground (sampled at 120 Hz, so a sag between keys shows), and for each cluster of vertices that
    touches the ground (game_measure's cluster finder) its planted backward speed (median), skate (RMS difference of
    the planted velocity from the clip's travel speed), duty, touchdown phases and grazes (touches under STANCE_S).
    Readings only: no gate reads them."""
    from . import game_measure as gm
    body = gm.Body(rigged)
    gap = CONTACT_GAP_MM/1000
    out = {}
    for clip, c in spec['clips'].items():
        if clip not in body.clips:
            continue
        loop = bool(c.get('loop', False))
        first = gm.clip_sample(body, clip, loop, {})
        hit = np.flatnonzero(((first.z-first.ground) <= gap).any(axis=0))
        groups = gm.clusters(body, hit)
        contacts = {f'{g["joint"]}#{k}': g['ids'] for k, g in enumerate(groups)}
        sampled = gm.clip_sample(body, clip, loop, contacts) if contacts else first
        read = gm.legs_read(sampled.contacts, sampled.times, sampled.duration, loop, sampled.ground, gap, STANCE_S)
        planted = {k: x for k, x in read.items() if x['stance']}
        travel, per = gm.speeds(planted, np.array([0.0, 1.0])) if planted else (None, {})
        rows = []
        for k, g in enumerate(groups):
            key = f'{g["joint"]}#{k}'
            x = read[key]
            if not x['stance'] and not x['grazes']:
                continue
            rows.append({**gm.describe(body, g), 'duty': round(x['duty'], 3),
                         'touchdown_phases': [round(t/sampled.duration, 3) for t in x['touchdowns']],
                         'speed_mps': _r3(per.get(key, {}).get('speed_mps')),
                         'skate_mps': _r3(per.get(key, {}).get('skate_mps')),
                         'grazes': [{'time_s': round(gz['time_s'], 3), 'clearance_mm': round(gz['clearance_mm'], 1)+0.0}
                                    for gz in x['grazes']]})
        out[clip] = {'below_rest_ground_mm': round(max(0.0, first.ground-first.lowest)*1000, 1),
                     'travel_speed_mps': None if travel is None else round(travel, 3)+0.0, 'contacts': rows}
    return out


def _r3(v):
    return None if v is None else round(float(v), 3)+0.0


def contact_text(found: dict[str, dict]) -> list[str]:
    """The ground-contact readings as lines: what a game that plants these feet would make of each clip."""
    if not found:
        return []
    lines = [f"GROUND CONTACT AS A GAME WOULD USE IT (vertices within {CONTACT_GAP_MM:g} mm of the rest ground, grouped "
             "by the bone that carries them; a planted contact should move back at one speed through its whole stance, "
             f"toe-off included; readings, no gate). Speeds in m/s, phases as shares of the clip:"]
    for clip, f in found.items():
        head = f"  {clip}: travel {f['travel_speed_mps']:.3f}" if f['travel_speed_mps'] is not None else f"  {clip}: nothing planted"
        lines.append(head)
        if f['below_rest_ground_mm'] >= BELOW_SAID_MM:
            lines.append(f"! {clip} goes {f['below_rest_ground_mm']:.1f} mm below the rest ground, read 120 times a second (a "
                         "planted limb that sags between keys does this; more keys or easing between them cure it)")
        for c in f['contacts']:
            bits = [f"{c['joint']} on {', '.join(c['nodes'][:2])}", f"duty {c['duty']:.2f}"]
            if c['speed_mps'] is not None:
                bits.append(f"planted {c['speed_mps']:.3f} back, skate {c['skate_mps']:.3f}")
            if c['touchdown_phases']:
                bits.append('touches down at ' + ', '.join(f'{p:.2f}' for p in c['touchdown_phases']))
            if c['grazes']:
                bits.append('grazes ' + ', '.join(f"at {g['time_s']:.2f} s ({g['clearance_mm']:.1f} mm)" for g in c['grazes']))
            marked = (c['skate_mps'] or 0) >= SKATE_SAID_MPS or c['grazes']
            lines.append(('! ' if marked else '    ')+'; '.join(bits))
    return lines
