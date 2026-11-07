"""Articulations: the parts a project says a rig will bend (`review.articulations`), and how much of each stands free
of the rest of the model.

A rig bends a part only where it is free: where the part is joined to the body, or nearer it than a cut can part,
the body stretches with it. Low-poly cats with arms modelled into the flank rigged worst by every rigger and every
reviewer, and no builder acted on a reading that said so; a gate is what
builders act on. Nothing here knows what an arm is: the project gives each part's path from root to tip, read off
its reference (`mesh-jig refs --articulation`), and the measure walks it.

The walk cuts the model across the path every STEP_M. At each cut the plane meets the surface in closed outlines;
outlines nearer each other than the gap are one island. The cut is free when the island the path passes through
stays within REACH times the part's radius of the path and no other articulation's path passes through it: joined
to the body, the island runs on into it. The free length is the run of free cuts from the tip inward, which is
where a rig's joints go.

Coordinates are Blender metres (+Z up, +Y front, X < 0 the model's left), as the review's landmarks, with Z
measured above the build's lowest point: a path read off a reference is then right for every build however it is
placed on Z. numpy only.
"""
from __future__ import annotations

import math

import numpy as np

STEP_M = 0.01           # one cut a centimetre along the path
REACH = 3.0             # an island reaching further from the path than this many radii runs on into the body: a paw
                        # or a tail tip is wider than the part, and a build puts it a few cm off the reference's path
GAP_M = 0.03            # the default gap: the rig stage's GAP_M, the nearest two surfaces a cut between them parts
KINDS = ("bend",)


def validate(items) -> None:
    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
        raise ValueError("review.articulations must be a list of objects")
    names = set()
    for item in items:
        name = item.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("review articulation needs a unique name")
        names.add(name)
        if set(item) - {"name", "kind", "path_m", "radius_m", "min_free", "min_gap_m"}:
            raise ValueError(f"review articulation {name}: unknown setting")
        if item.get("kind", "bend") not in KINDS:
            raise ValueError(f"review articulation {name}: kind must be one of {', '.join(KINDS)} "
                             "(a rigid hinge is not measured yet)")
        path = item.get("path_m")
        if (not isinstance(path, list) or len(path) < 2 or any(
                not isinstance(p, list) or len(p) != 3 or not all(_finite(v) for v in p) for p in path)):
            raise ValueError(f"review articulation {name}: path_m needs two or more [x, y, z] points, root first")
        if _length(np.asarray(path, float)) <= STEP_M:
            raise ValueError(f"review articulation {name}: path_m is shorter than {STEP_M} m")
        if not _finite(item.get("radius_m")) or item["radius_m"] <= 0:
            raise ValueError(f"review articulation {name}: radius_m must be a positive number of metres")
        if not _finite(item.get("min_free", 0.5)) or not 0 <= item.get("min_free", 0.5) <= 1:
            raise ValueError(f"review articulation {name}: min_free must be between 0 and 1")
        if not _finite(item.get("min_gap_m", GAP_M)) or item.get("min_gap_m", GAP_M) <= 0:
            raise ValueError(f"review articulation {name}: min_gap_m must be a positive number of metres")


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def off_path(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Each point's distance to the nearest point of the polyline."""
    a, b = path[:-1], path[1:]
    ab = b - a
    t = np.clip(((points[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(points[:, None] - (a + t[..., None] * ab), axis=-1).min(axis=1)


def _length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def stations(path: np.ndarray, step: float = STEP_M) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """(distance from the root, point, unit direction) every `step` along a polyline, at the middle of each step."""
    seg = np.diff(path, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    ends = np.concatenate([[0.0], np.cumsum(lengths)])
    out = []
    for t in np.arange(step / 2, ends[-1], step):
        k = min(int(np.searchsorted(ends, t, side="right")) - 1, len(seg) - 1)
        while lengths[k] == 0:
            k -= 1
        d = seg[k] / lengths[k]
        out.append((float(t), path[k] + d * (t - ends[k]), d))
    return out


def section(tris: np.ndarray, point: np.ndarray, normal: np.ndarray, near: float) -> np.ndarray:
    """The segments (k, 2, 3) where the plane through `point` with `normal` cuts the triangles, those within `near`
    of the point. A vertex on the plane counts as in front of it, so an edge is cut once."""
    lo = tris.min(axis=1) - point
    hi = tris.max(axis=1) - point
    close = (np.maximum(np.maximum(-hi, lo), 0) ** 2).sum(axis=1) <= near * near
    t = tris[close]
    if not len(t):
        return np.zeros((0, 2, 3))
    d = (t - point) @ normal
    side = d >= 0
    cut = side.any(axis=1) & ~side.all(axis=1)
    t, d, side = t[cut], d[cut], side[cut]
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        crosses = side[:, i] != side[:, j]
        f = np.where(crosses, d[:, i] / np.where(crosses, d[:, i] - d[:, j], 1.0), 0.0)
        out.append((t[:, i] + (t[:, j] - t[:, i]) * f[:, None], crosses))
    points = np.stack([p for p, _ in out], axis=1)            # (k, 3 edges, 3)
    which = np.stack([c for _, c in out], axis=1)             # exactly two edges of a cut triangle cross
    return points[which].reshape(-1, 2, 3)


def islands(segments: np.ndarray, gap: float, normal: np.ndarray | None = None) -> list[tuple]:
    """The cut's outlines grouped into islands: outlines that share an end are one, outlines nearer each other than
    `gap` are one, and (given the plane's `normal`) an outline inside another is one with it: a part sunk into the
    body cuts as a small outline within the body's, apart by more than the gap and still joined. Each island as
    (points sampled along its segments no further apart than a third of the gap, its segments)."""
    if not len(segments):
        return []
    _, ends = np.unique(np.round(segments.reshape(-1, 3), 6), axis=0, return_inverse=True)
    ends = ends.reshape(-1, 2)
    parent = list(range(int(ends.max()) + 1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in ends:
        parent[root(a)] = root(b)
    label = np.array([root(a) for a, _ in ends])
    spacing = gap / 3
    samples = {}
    for key in np.unique(label):
        segs = segments[label == key]
        counts = np.maximum(np.ceil(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) / spacing).astype(int), 1)
        pts = [s[0] + (s[1] - s[0]) * (np.arange(n + 1)[:, None] / n) for s, n in zip(segs, counts)]
        samples[int(key)] = np.vstack(pts)
    keys = list(samples)
    group = {k: k for k in keys}

    def top(k):
        while group[k] != k:
            k = group[k]
        return k

    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            a, b = samples[keys[x]], samples[keys[y]]
            if top(keys[x]) == top(keys[y]):
                continue
            near = min(float(np.sqrt(((a[i:i + 512, None] - b[None]) ** 2).sum(-1).min()))
                       for i in range(0, len(a), 512))
            if near < gap:
                group[top(keys[y])] = top(keys[x])
    out = {}
    for k in keys:
        out.setdefault(top(k), []).append(k)
    return [(np.vstack([samples[k] for k in v]), segments[np.isin(label, v)]) for v in out.values()]


def inside(point: np.ndarray, normal: np.ndarray, segments: np.ndarray) -> bool:
    """Whether the point lies within the closed outlines the segments draw on the cut plane: the outline of a thick
    part is further from a path down its middle than its radius, and the path is still in it. Even-odd, along one
    in-plane direction."""
    u = np.cross(normal, (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    a = np.stack([(segments[:, 0] - point) @ u, (segments[:, 0] - point) @ v], axis=1)
    b = np.stack([(segments[:, 1] - point) @ u, (segments[:, 1] - point) @ v], axis=1)
    crosses = (a[:, 1] > 0) != (b[:, 1] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = a[:, 0] + (b[:, 0] - a[:, 0]) * (-a[:, 1]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(crosses & (at > 0)) % 2)


def crossings(paths: list[np.ndarray], point: np.ndarray, normal: np.ndarray, near: float) -> list[np.ndarray]:
    """Where other parts' paths cross the cut plane, within `near` of the point."""
    out = []
    for other in paths:
        d = (other - point) @ normal
        for k in range(len(other) - 1):
            if (d[k] >= 0) != (d[k + 1] >= 0):
                c = other[k] + (other[k + 1] - other[k]) * (d[k] / (d[k] - d[k + 1]))
                if np.linalg.norm(c - point) <= near:
                    out.append(c)
    return out


def closed_outlines(segments: np.ndarray) -> list[np.ndarray]:
    """The cut's outlines that close on themselves (every end shared by two segments), as segment arrays."""
    if not len(segments):
        return []
    _, ends = np.unique(np.round(segments.reshape(-1, 3), 6), axis=0, return_inverse=True)
    ends = ends.reshape(-1, 2)
    parent = list(range(int(ends.max()) + 1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in ends:
        parent[root(a)] = root(b)
    label = np.array([root(a) for a, _ in ends])
    counts = np.bincount(ends.ravel(), minlength=len(parent))
    out = []
    for key in np.unique(label):
        mine = label == key
        if (counts[np.unique(ends[mine])] == 2).all():
            out.append(segments[mine])
    return out


def sunk(tris: np.ndarray, point: np.ndarray, normal: np.ndarray, outline: np.ndarray, path: np.ndarray,
         reach: float) -> bool:
    """Whether the part's outline in this cut lies inside a closed outline of another part of the model: a limb
    sunk into the body cuts as a small ring within the body's, further from it than any gap, and is joined. Read on
    the whole plane, where the body's outline is whole and closed. Only an outline that is itself body-like counts,
    running further from the path than `reach` and wider than twice it: a paw built as its own shape round the end of
    a leg holds the leg's ring too, and is the limb's own."""
    if outline is None or not len(outline):
        return False
    probe = outline[0, 0]
    own = {tuple(np.round(p, 6)) for p in outline.reshape(-1, 3)}
    for ring in closed_outlines(section(tris, point, normal, float("inf"))):
        if own & {tuple(np.round(p, 6)) for p in ring.reshape(-1, 3)}:
            continue                                # the part's own outline
        if not inside(probe, normal, ring):
            continue
        pts = ring.reshape(-1, 3)
        centre = pts.mean(axis=0)
        if off_path(pts, path).max() > reach and 2 * np.linalg.norm(pts - centre, axis=1).max() > 2 * reach:
            return True
    return False


def measure(tris: np.ndarray, item: dict, others: list[dict] = ()) -> dict:
    """One articulation on a model's triangles (Blender metres): the check record. `free_m` is the run of free cuts
    from the tip inward; `joined_m` how far of the path from the root is not free (where the part is joined or too
    near); `missing_m` how much of the path has no surface within the radius (a part shorter than its path, or
    elsewhere). `others` are the project's other articulations: a cut whose island another one's path passes
    through is joined."""
    ground = float(tris[..., 2].min()) if len(tris) else 0.0
    lift = np.array([0.0, 0.0, ground])
    path = np.asarray(item["path_m"], float) + lift
    others = [np.asarray(o["path_m"], float) + lift for o in others if o is not item and o["name"] != item["name"]]
    radius, gap = float(item["radius_m"]), float(item.get("min_gap_m", GAP_M))
    reach = REACH * radius
    length = _length(path)
    state, where = [], []                       # per cut, root first: "free", "joined" or "missing"; its point
    off = 0.0                                   # how far off its path the part runs where it is free
    for along, point, direction in stations(path):
        where.append((along, point))
        segs = section(tris, point, direction, reach + 2 * gap)
        # the part is the island the path passes nearest: another island within the radius is a neighbour (the
        # flank beside an arm), and it is joined to the part only if the gap between them is under `gap`
        found, nearest, mine = None, radius, None
        for island, outline in islands(segs, gap):
            within = inside(point, direction, outline)
            dist = 0.0 if within else float(np.linalg.norm(island - point, axis=1).min())
            if dist <= nearest and (found is None or dist < nearest or within):
                # how far the island strays from the part's path as a whole: at a bend the cut runs along the next
                # segment, which is the part, not the body
                found, nearest, mine = float(off_path(island, path).max()), dist, outline
                centre = island.mean(axis=0)
                span = 2 * float(np.linalg.norm(island - centre, axis=1).max())
                aside = float(np.linalg.norm(centre - point))
        # an island that strays past the reach but is no wider than a free part could be is the part built off
        # its path, not the part run into the body: a builder's curled tail failed as "joined" with nothing near it
        strays = found is not None and found > reach and span > 2 * reach
        if found is not None and found > reach and not strays:
            off = max(off, aside)
        if found is None:
            state.append("missing")
        elif (strays or any(inside(c, direction, mine) for c in crossings(others, point, direction, reach))
              or sunk(tris, point, direction, mine, path, reach)):
            # it runs on into the body, or another declared part passes through the same island (legs pressed
            # together, a paw fused to a thigh)
            state.append("joined")
        else:
            state.append("free")
    step = length / max(len(state), 1)
    end = len(state)
    while end and state[end - 1] == "missing":  # a path that runs on past the part's tip
        end -= 1
    start = end
    while start and state[start - 1] == "free":
        start -= 1
    free_m = round((end - start) * step, 3)
    fraction = round(free_m / length, 3) if length else 0.0
    minimum = float(item.get("min_free", 0.5))
    runs = []                                   # the cuts root to tip as runs: [state, metres]
    for s in state:
        if runs and runs[-1][0] == s:
            runs[-1][1] += step
        else:
            runs.append([s, step])
    stop = {}
    if start:                                   # where the run from the tip stops, in the build's own coordinates
        along, point = where[start - 1]
        stop = {"stops_at_m": [round(float(v), 3) + 0.0 for v in point], "stops_along_m": round(along, 3),
                "stops_because": state[start - 1]}
    return {"name": f"articulation:{item['name']}", "passed": fraction >= minimum,
            "length_m": round(length, 3), "free_m": free_m, "free_fraction": fraction,
            "joined_m": round(state[:start].count("joined") * step, 3),
            "missing_m": round(state.count("missing") * step, 3),
            "radius_m": radius, "min_gap_m": gap, "min_free": minimum,
            "cuts": ", ".join(f"{s} {m:.2f}" for s, m in runs), **stop,
            **({"off_path_m": round(off, 3)} if off else {})}


def describe(check: dict) -> str:
    """The check in words, as the feedback prints a gate."""
    free, length = check.get("free_m", 0.0), check.get("length_m", 0.0)
    said = f"free of the rest for {free:.2f} m of its {length:.2f} m ({check.get('free_fraction', 0):.0%}"
    if check.get("passed"):
        return said + ")"
    said += f", needs {check.get('min_free', 0):.0%})"
    if check.get("stops_at_m") and check.get("stops_because") == "joined":
        x, y, z = check["stops_at_m"]
        said += (f": the free run from the tip stops {check['stops_along_m']:.2f} m along it from its root, at "
                 f"({x:+.3f}, {y:+.3f}, {z:+.3f}) m in your model's own coordinates, where it is joined to the body "
                 f"or another part, or nearer one than {check.get('min_gap_m', GAP_M) * 100:.0f} cm (in any "
                 "direction: look at the side view too). A rig bending it there stretches the body; leave that gap "
                 "below the joint")
    elif check.get("joined_m"):
        said += (f": for {check['joined_m']:.2f} m from its root it is joined to the body or nearer it than "
                 f"{check.get('min_gap_m', GAP_M) * 100:.0f} cm, so a rig bending it stretches the body. Leave that "
                 "gap below the joint")
    if check.get("cuts"):
        said += f" (root to tip: {check['cuts']} m)"
    if check.get("off_path_m"):
        said += (f"; it runs up to {check['off_path_m']:.2f} m off its path, which the gate allows (it reads the part "
                 "within the path's radius, and joined only where the cut runs on into something wider)")
    if check.get("missing_m"):
        said += (f"; for {check['missing_m']:.2f} m of its path there is no surface within "
                 f"{check.get('radius_m', 0):.2f} m of it: the part is shorter than the reference's, or elsewhere")
    return said


def brief_text(review: dict | None) -> str:
    """What the brief says of a project's articulations; '' when it states none."""
    items = (review or {}).get("articulations") or []
    if not items:
        return ""
    said = "; ".join(f"{i['name']} (free for {i.get('min_free', 0.5):.0%} of its "
                     f"{_length(np.asarray(i['path_m'], float)):.2f} m)" for i in items)
    return ("ARTICULATIONS (gates): parts a rig will bend, each along `path_m` from its root to its tip in Blender "
            "metres, X and Y as the model is placed and Z above its lowest point (a build whose feet are at Z = -0.6 "
            "has the path's Z values 0.6 lower in its own coordinates; a failure says where in yours). A part passes "
            "when, from its tip inward, it stands free of the rest of the model for at least `min_free` of its "
            "length. It is cut across its path every centimetre; a cut is free when the part there is found within "
            "`radius_m` of the path (it may sit that far off it), has no other surface nearer it than `min_gap_m` "
            f"(3 cm unless stated) in any direction, the side as well as the front, is not sunk inside the body, "
            f"and does not run on into something wider than {2 * REACH:g} radii (the body, or another declared "
            "part). Join it to the body at the root only, and keep a bend (an elbow, a curled tail) clear of the "
            "body too. " + said)


def failed_lines(checks: list[dict]) -> list[str]:
    """The failed articulation gates as lines, for a builder that is shown no other gate."""
    return [f"gate {c['name']}: FAILED - {describe(c)}" for c in checks
            if str(c.get("name", "")).startswith("articulation:") and c.get("passed") is False]
