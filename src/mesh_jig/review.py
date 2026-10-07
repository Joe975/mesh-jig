"""Opt-in deterministic regional, named-part, visibility and sampled garment-clearance checks.

No semantic inference: landmarks and probes are an explicit frozen review contract.
Geometry inputs/landmarks/probes use Blender metres (+Z up, +Y front, X<0 anatomical left).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import glb, render, resemblance, silhouette
from .palette import zone_rgb


def _vector(v, n=3):
    return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float))
            and not isinstance(x, bool) and math.isfinite(x) for x in v)


def _limit(v, lo=0, hi=1):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi


def validate(config: dict, spec: dict, refs: dict, zones: tuple = ()) -> None:
    if not isinstance(config, dict) or set(config) - {"parts", "regions", "landmarks", "visibility", "clearance"}:
        raise ValueError("review needs parts/regions/landmarks/visibility/clearance")
    parts = config.get("parts", {})
    if not isinstance(parts, dict):
        raise ValueError("review.parts must map part names to exact exported node names")
    for group in ("regions", "landmarks", "visibility", "clearance"):
        entries = config.get(group, [])
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            raise ValueError(f"review.{group} must be a list of objects")
    used = set()
    for name, part in parts.items():
        nodes = part.get("nodes") if isinstance(part, dict) else None
        if not isinstance(nodes, list) or not nodes or any(not isinstance(n, str) or not n for n in nodes):
            raise ValueError(f"review part {name}: nodes needs nonempty exact GLB node names")
        if set(part) - {"nodes", "smooth_min", "max_components"}:
            raise ValueError(f"review part {name}: unknown diagnostic setting")
        if len(set(nodes)) != len(nodes) or used.intersection(nodes):
            raise ValueError("review.parts node assignments must be unique")
        used.update(nodes)
        if "smooth_min" in part and not _limit(part["smooth_min"]):
            raise ValueError("review.parts smooth_min must be between 0 and 1")
        if "max_components" in part and (type(part["max_components"]) is not int or part["max_components"] < 1):
            raise ValueError("review.parts max_components must be a positive integer")
    if parts and spec.get("join"):
        raise ValueError("review.parts requires join=false so exported node identity survives")
    shots = {s["name"] for s in render.shots_for(spec) if s["variant"] == glb.judged_variant(spec)}
    for r in config.get("regions", []):
        if r.get("view") not in refs or r["view"] not in shots or not r.get("name"):
            raise ValueError("review region needs a name and reference/judged camera")
        b = r.get("box")
        if not _vector(b, 4) or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise ValueError("review region box needs normalized [x0,y0,x1,y1] on the whole-subject canvas")
        for key in ("min_iou", "min_zone"):
            if key in r and not _limit(r[key]):
                raise ValueError(f"review region {key} must be between 0 and 1")
        if "min_zone" in r and not zones:
            raise ValueError("review region min_zone requires a palette")
    for group in ("landmarks", "visibility"):
        for item in config.get(group, []):
            if item.get("part") not in parts:
                raise ValueError(f"review.{group} names an undeclared part")
            if group == "landmarks":
                b = item.get("bounds_m")
                if not isinstance(b, list) or len(b) != 2 or not all(_vector(v) for v in b) or any(
                        a >= z for a, z in zip(*b)):
                    raise ValueError("review landmark bounds_m needs [min_xyz,max_xyz] in Blender metres")
            else:
                if item.get("view") not in shots:
                    raise ValueError("review visibility needs a judged camera")
                for key in ("min_fraction", "max_fraction"):
                    if key in item and not _limit(item[key]):
                        raise ValueError("review visibility fractions must be between 0 and 1")
                if item.get("min_fraction", 0) > item.get("max_fraction", 1):
                    raise ValueError("review visibility minimum exceeds maximum")
    for item in config.get("clearance", []):
        if item.get("inner") not in parts or item.get("outer") not in parts or item["inner"] == item["outer"]:
            raise ValueError("review clearance needs distinct declared inner/outer parts")
        if not _limit(item.get("min_m"), 0, float("inf")) or not _limit(item.get("max_m"), item["min_m"], float("inf")):
            raise ValueError("review clearance needs finite 0 <= min_m <= max_m")
        if item.get("inner_hit", "first") not in ("first", "last") or item.get("outer_hit", "first") not in ("first", "last"):
            raise ValueError("review clearance inner_hit/outer_hit must be first or last")
        rays = item.get("rays")
        if not isinstance(rays, list) or not rays or any(not _vector(r.get("origin_m")) or
                not _vector(r.get("direction")) or np.linalg.norm(r["direction"]) == 0 for r in rays):
            raise ValueError("review clearance rays need origin_m and nonzero direction in Blender axes")


FINE = 2 * resemblance.GRID     # the canvas a region is read on for its colour and edges: a region is a third of a
                                # view or less, and on the whole view's canvas a face is a dozen cells across


def _fine(a: np.ndarray, box) -> np.ndarray:
    """A region's box cut from a FINE canvas, out to whole cells and padded square: what the cell measures take."""
    cell = resemblance.CELL
    x0, y0, x1, y1 = (int(v * FINE) for v in box)
    cut = a[y0 // cell * cell:-(-y1 // cell) * cell, x0 // cell * cell:-(-x1 // cell) * cell]
    side = max(cut.shape[0], cut.shape[1])
    return np.pad(cut, [(0, side - cut.shape[0]), (0, side - cut.shape[1])] + [(0, 0)] * (cut.ndim - 2))


def region_reading(a, ma, b, mb, box) -> dict:
    """{"colour", "edges", "score"} of one region on the FINE canvases of the build (a) and the reference (b): how
    close its colours are (`resemblance.colour_closeness`), whether its edges are in place
    (`resemblance.structure_similarity`), and their mean. A region the reference has and the build leaves empty has
    nothing in common with it, 0; a region with no edge on either side is read on its colour alone; `score` is None
    where the reference itself has nothing in the box."""
    a, ma, b, mb = _fine(a, box), _fine(ma, box), _fine(b, box), _fine(mb, box)
    if not mb.any():
        return {"colour": None, "edges": None, "score": None}
    colour = resemblance.colour_closeness(resemblance.cell_colours(a, ma), resemblance.cell_colours(b, mb))
    edges = resemblance.structure_similarity(resemblance.edge_histograms(a, ma), resemblance.edge_histograms(b, mb))
    colour = 0.0 if colour is None else colour
    parts = [colour] + ([edges] if edges is not None else [])
    return {"colour": colour, "edges": edges, "score": round(sum(parts) / len(parts), 4)}


def regions_score(checks: list[dict]) -> float | None:
    """The mean of the regions' own scores (`region_reading`): the `regions` term of a score that asks for it."""
    scores = [c["score"] for c in checks if c.get("score") is not None]
    return round(sum(scores) / len(scores), 4) if scores else None


def is_observation(check: dict) -> bool:
    """Regions without acceptance limits are measurements, including legacy records."""
    return bool(check.get("observation")) or ("view" in check and "iou" in check
            and "min_iou" not in check and "min_zone" not in check)


def regions(renders: dict, refs: dict, config: list, zones: tuple) -> list[dict]:
    out = []
    cache, fine = {}, {}
    for r in config:
        view = r["view"]
        observation = "min_iou" not in r and "min_zone" not in r
        if view not in renders:
            out.append({"name": r["name"], "view": view, "observation": observation,
                        "passed": None if observation else False, "error": f"missing render {view}",
                        **_limits(r, "min_iou", "min_zone")})
            continue
        if view not in cache:
            cache[view] = (resemblance.normalised(renders[view]), resemblance.normalised(refs[view]))
            fine[view] = (*resemblance.normalised(renders[view], FINE), *resemblance.normalised(refs[view], FINE))
        (a, ma), (b, mb) = cache[view]
        x0, y0, x1, y1 = [int(v * resemblance.GRID) for v in r["box"]]
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        ma, mb = ma[y0:y1, x0:x1], mb[y0:y1, x0:x1]
        iou = silhouette.iou(ma, mb)
        zone = None
        union = ma | mb
        if zones and union.any():
            p = zone_rgb(zones)
            ca = ((a[..., None, :] - p) ** 2).sum(-1).argmin(-1)
            cb = ((b[..., None, :] - p) ** 2).sum(-1).argmin(-1)
            zone = round(float(((ca == cb) & ma & mb).sum() / union.sum()), 4)
        passed = bool(mb.any()) and iou >= r.get("min_iou", 0) and ("min_zone" not in r or
                                                          zone is not None and zone >= r["min_zone"])
        out.append({"name": r["name"], "view": view, "iou": iou, "zone": zone,
                    "reference_pixels": int(mb.sum()), "observation": observation,
                    "passed": None if observation else passed, **_limits(r, "min_iou", "min_zone"),
                    **region_reading(*fine[view], r["box"])})
    return out


CLOSEUP_PREFIX = "region_"
CLOSEUP_HEIGHT = 420        # a region is a third of a view or less: at the comparison pictures' 360 a face is 145 wide
CLOSEUP_MARGIN = 0.15       # of the box's longer side, shown around it: what the region sits in


def pixel_box(img: str | Path, box, margin: float = 0.0) -> tuple[int, int, int, int]:
    """A region's `box` (0..1 on the square canvas the subject is fitted to, as `regions` cuts it) in the picture's own
    pixels, widened by `margin` of its longer side and held to the picture: `references.region_box` the other way."""
    with Image.open(img) as im:
        size = im.size
        bx0, by0, bx1, by1 = silhouette.subject_box(im.convert("RGB"), 0.0, img)
    side = max(bx1 - bx0, by1 - by0)
    left, top = bx0 - (side - (bx1 - bx0)) / 2, by0 - (side - (by1 - by0)) / 2
    x0, y0, x1, y1 = (float(v) for v in box)
    pad = margin * max(x1 - x0, y1 - y0)
    return (max(0, round(left + (x0 - pad) * side)), max(0, round(top + (y0 - pad) * side)),
            min(size[0], round(left + (x1 + pad) * side)), min(size[1], round(top + (y1 + pad) * side)))


def closeups(renders: dict, refs: dict, config: list, out_dir: str | Path, height: int = CLOSEUP_HEIGHT) -> dict[str, str]:
    """region_<name>.png for every named region whose view was drawn: the reference's region on the left and the
    build's on the right, each cut from its own picture at full size and enlarged to one height. A region's gate
    reads a box a builder otherwise sees a hundred-odd pixels wide in the whole view's comparison."""
    out_dir = Path(out_dir)
    out = {}
    for r in config or []:
        view = r["view"]
        if view not in renders or view not in refs:
            continue
        panels = []
        for path in (refs[view], renders[view]):
            x0, y0, x1, y1 = pixel_box(path, r["box"], CLOSEUP_MARGIN)
            if x1 <= x0 or y1 <= y0:
                break
            with Image.open(path) as im:
                cut = im.convert("RGB").crop((x0, y0, x1, y1))
            panels.append(cut.resize((max(1, round(cut.width * height / cut.height)), height), Image.LANCZOS))
        if len(panels) < 2:
            continue
        label, gap = 28, 24
        sheet = Image.new("RGB", (panels[0].width + gap + panels[1].width, height + label), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for x, panel, who in ((0, panels[0], "REFERENCE"), (panels[0].width + gap, panels[1], "YOUR BUILD")):
            sheet.paste(panel, (x, label))
            d.text((x + 6, 6), f"{who} {r['name']} ({view} view)", fill=(0, 0, 0))
        d.line([(panels[0].width + gap // 2, 0), (panels[0].width + gap // 2, sheet.height)], fill=(0, 0, 0), width=2)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / (CLOSEUP_PREFIX + re.sub(r"[^A-Za-z0-9_.-]", "_", str(r["name"])) + ".png")
        sheet.save(target)
        out[r["name"]] = str(target)
    return out


def _limits(item: dict, *keys: str) -> dict:
    """The limits a check was held to, kept in its record: a failed check then says what it needed (`describe`)."""
    return {k: item[k] for k in keys if k in item}


def part_indices(path: str | Path, parts: dict) -> dict[str, np.ndarray]:
    """Triangle order matches render.load, including scene-node transforms and triangle primitives."""
    doc, _ = glb.read_glb(path)
    by_node, offset = {}, 0
    for ni, _ in glb.mesh_nodes(doc):
        node = doc["nodes"][ni]
        mesh = doc["meshes"][node["mesh"]]
        count = sum((doc["accessors"][p["indices"]]["count"] if "indices" in p else
                     doc["accessors"][p["attributes"]["POSITION"]]["count"]) // 3
                    for p in mesh.get("primitives", []) if p.get("mode", 4) == 4)
        name = node.get("name") or mesh.get("name") or f"node{ni}"
        by_node.setdefault(name, []).extend(range(offset, offset + count))
        offset += count
    return {name: np.array([i for node in part["nodes"] for i in by_node[node]]
                          if all(node in by_node for node in part["nodes"]) else [], dtype=int)
            for name, part in parts.items()}


def component_count(tris: np.ndarray) -> int:
    """Weld only coincident positions (1e-6 m); intersections are not fusion."""
    if not tris.size:
        return 0
    _, ids = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0, return_inverse=True)
    parent = list(range(int(ids.max()) + 1))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, c in ids.reshape(-1, 3):
        parent[root(b)] = parent[root(a)]
        parent[root(c)] = parent[root(a)]
    return len({root(i) for i in range(len(parent))})


def ray_distance(tris: np.ndarray, origin, direction, hit: str = "first") -> float | None:
    """First positive surface intersection, double sided, metres; no closed-volume assumption."""
    if not tris.size:
        return None
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    valid = np.abs(det) > 1e-12
    inv = np.divide(1, det, out=np.zeros_like(det), where=valid)
    s = np.asarray(origin) - tris[:, 0]
    u = inv * (s * h).sum(1)
    q = np.cross(s, e1)
    v = inv * (q * d).sum(1)
    t = inv * (q * e2).sum(1)
    live = valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-8)
    return float(t[live].max() if hit == "last" else t[live].min()) if live.any() else None


def measure(path: str | Path, spec: dict, config: dict, renders: dict, refs: dict, zones: tuple = (),
            region_checks: list[dict] | None = None) -> dict:
    """Every check of the review as {"checks", "diagnostics", "warnings"}. `region_checks` are the regions' own
    when the caller has read them already (`regions`), as an evaluation has: its score may count them."""
    validate(config, spec, refs, zones)
    checks = (list(region_checks) if region_checks is not None
              else regions(renders, refs, config.get("regions", []), zones))
    warnings, diagnostics = [], {}
    parts = config.get("parts", {})
    if not parts:
        return {"checks": checks, "diagnostics": diagnostics, "warnings": warnings}
    mesh = render.load(path)
    selected = part_indices(path, parts)
    # glTF -> Blender, without bbox normalization
    tris = np.stack([mesh.tris[..., 0], -mesh.tris[..., 2], mesh.tris[..., 1]], axis=-1)
    for name, ids in selected.items():
        if not len(ids):
            checks.append({"name": f"part:{name}", "passed": False, "error": "declared part missing from GLB"})
            continue
        ns = mesh.normals[ids]
        smooth = float((np.max(np.linalg.norm(ns - ns[:, :1], axis=2), axis=1) > 1e-5).mean())
        components = component_count(mesh.tris[ids])
        diagnostics[name] = {"triangles": len(ids), "varying_corner_normal_fraction": round(smooth, 4),
                             "position_welded_components": components}
        if smooth < parts[name].get("smooth_min", 0):
            warnings.append(f"{name}: varying corner normals {smooth:.1%}; inspect skin shading (flat may be intentional)")
        if "max_components" in parts[name] and components > parts[name]["max_components"]:
            warnings.append(f"{name}: {components} disconnected position-welded components; joining is not fusion")
    for item in config.get("landmarks", []):
        ids = selected[item["part"]]
        pts = tris[ids].reshape(-1, 3)
        lo, hi = np.asarray(item["bounds_m"])
        # Whole part bounds, not only a centroid which can hide a wrong attachment/side.
        passed = bool(len(pts)) and bool((pts >= lo - 1e-7).all() and (pts <= hi + 1e-7).all())
        checks.append({"name": f"landmark:{item['part']}", "passed": passed,
                       "bounds_m": [pts.min(0).tolist(), pts.max(0).tolist()] if len(pts) else None,
                       "allowed_m": [lo.tolist(), hi.tolist()]})
    shots = {s["name"]: s for s in render.shots_for(spec)}
    frames = {}
    for item in config.get("visibility", []):
        name, view = item["part"], item["view"]
        ids = selected[name]
        if view not in frames:
            frames[view] = render.rasterise(mesh, shots[view], scale=render.stage_scale(spec), width=320, height=180)[0]
        subset = render.Mesh(mesh.tris[ids], mesh.colors[ids], mesh.normals[ids])
        isolated = render.rasterise(subset, shots[view], scale=render.stage_scale(spec), width=320, height=180)[0]
        projected = int((isolated >= 0).sum())
        visible = int(np.isin(frames[view], ids).sum()) if len(ids) else 0
        fraction = visible / projected if projected else 0
        passed = bool(len(ids)) and fraction >= item.get("min_fraction", 0) and fraction <= item.get("max_fraction", 1)
        # A minimum visibility needs nonzero projected area; absent parts never pass a maximum-only check.
        if item.get("min_fraction", 0) > 0 and not projected:
            passed = False
        checks.append({"name": f"visibility:{name}:{view}", "passed": passed, "projected_pixels": projected,
                       "visible_pixels": visible, "visible_fraction": round(fraction, 4),
                       **_limits(item, "min_fraction", "max_fraction")})
    for item in config.get("clearance", []):
        samples = []
        for ray in item["rays"]:
            a = ray_distance(tris[selected[item["inner"]]], ray["origin_m"], ray["direction"], item.get("inner_hit", "first"))
            b = ray_distance(tris[selected[item["outer"]]], ray["origin_m"], ray["direction"], item.get("outer_hit", "first"))
            gap = b - a if a is not None and b is not None else None
            samples.append({**ray, "inner_distance_m": a, "outer_distance_m": b,
                            "gap_m": round(gap, 6) if gap is not None else None,
                            "passed": gap is not None and item["min_m"] <= gap <= item["max_m"]})
        checks.append({"name": f"clearance:{item['inner']}:{item['outer']}",
                       "passed": all(s["passed"] for s in samples), "samples": samples,
                       **_limits(item, "min_m", "max_m")})
    return {"checks": checks, "diagnostics": diagnostics, "warnings": warnings}


def _point(v) -> str:
    return "(" + ", ".join(f"{float(x):g}" for x in v) + ")"


def _direction(v) -> str:
    """A ray's direction in words when it runs along one axis (`-Y`), else as it was given."""
    d = [float(x) for x in v]
    along = [i for i, x in enumerate(d) if abs(x) > 1e-9]
    return ("+" if d[along[0]] > 0 else "-") + "XYZ"[along[0]] if len(along) == 1 else _point(d)


def describe(check: dict) -> str:
    """One check of a review, in words: what it read and, when it failed, what it needed. A check that passed reads
    as its numbers alone (`outline 0.98, zones 0.61`), so the line of passed gates stays one line and still says how
    near each is to its limit. A record made before the limits were kept in it says what it read and no limit."""
    if check.get("error"):
        return str(check["error"])
    kind, *rest = str(check.get("name", "")).split(":")
    passed = bool(check.get("passed"))
    if "view" in check and "iou" in check:                              # a marked region of one view
        iou, zone = check["iou"], check.get("zone")
        stated = "min_iou" in check or "min_zone" in check              # an older record holds no limits
        if passed:
            said = [f"outline {iou:.2f}"] if "min_iou" in check or not stated else []
            if zone is not None and ("min_zone" in check or not stated):
                said.append(f"zones {zone:.2f}")
            return ", ".join(said)
        where = f"in its box on the {check['view']} view"
        if not check.get("reference_pixels"):
            return f"the reference has nothing {where}"
        low_iou = "min_iou" in check and iou < check["min_iou"]
        low_zone = "min_zone" in check and (zone is None or zone < check["min_zone"])
        if not (low_iou or low_zone):
            return (f"{where}: outline overlap {iou:.3f}" + (f", colour zones {zone:.3f}" if zone is not None else "")
                    + " (its limits are in the brief's review block)")
        short = [f"outline overlap {iou:.3f}, needs {check['min_iou']:g}"] if low_iou else []
        if low_zone:
            short.append(("colour zones not read" if zone is None else f"colour zones {zone:.3f}")
                         + f", needs {check['min_zone']:g}")
        other = [] if low_iou else [f"outline overlap {iou:.2f}"]
        if zone is not None and not low_zone:
            other.append(f"colour zones {zone:.2f}")
        return f"{where}: " + "; ".join(short) + (f" ({', '.join(other)})" if other else "")
    if kind == "landmark":
        if passed:
            return ""
        got, allowed = check.get("bounds_m"), check.get("allowed_m")
        if not got:
            return "the part has no geometry"
        if not allowed:
            return f"the part spans {_point(got[0])} to {_point(got[1])} m, outside the box the contract gives it"
        out = []
        for i, axis in enumerate("XYZ"):
            for end, sign in ((0, -1.0), (1, 1.0)):
                over = (got[end][i] - allowed[end][i]) * sign
                if over > 1e-7:
                    out.append(f"it reaches {axis} {got[end][i]:+.3f} m, {over * 1000:.0f} mm past the "
                               f"{allowed[end][i]:+.3f} its box allows")
        return "; ".join(out) or "outside the box the contract gives it"
    if kind == "visibility":
        shown, view = check.get("visible_fraction") or 0.0, rest[1] if len(rest) > 1 else "?"
        if passed:
            return f"{shown:.0%} shows"
        if not check.get("projected_pixels"):
            return f"none of it falls in the {view} view"
        if "min_fraction" in check and shown < check["min_fraction"]:
            return f"{shown:.0%} of it shows in the {view} view, needs at least {check['min_fraction']:.0%}"
        if "max_fraction" in check and shown > check["max_fraction"]:
            return f"{shown:.0%} of it shows in the {view} view, at most {check['max_fraction']:.0%} may"
        return f"{shown:.0%} of it shows in the {view} view (its limits are in the brief's review block)"
    if kind == "clearance":
        inner, outer = (rest + ["the inner part", "the outer part"])[:2]
        samples = check.get("samples") or []
        gaps = [s["gap_m"] * 1000 for s in samples if s.get("gap_m") is not None]
        if passed:
            return f"gap {min(gaps):.1f} to {max(gaps):.1f} mm" if gaps else ""
        allowed = (f", allowed {check['min_m'] * 1000:g} to {check['max_m'] * 1000:g} mm"
                   if "min_m" in check and "max_m" in check else "")
        out = []
        for s in samples:
            if s.get("passed"):
                continue
            ray = f"the ray from {_point(s.get('origin_m') or ())} m along {_direction(s.get('direction') or (0, 0, 0))}"
            if s.get("inner_distance_m") is None:
                out.append(f"{ray} meets no {inner}")
            elif s.get("outer_distance_m") is None:
                out.append(f"{ray} meets no {outer}")
            else:
                out.append(f"gap {s['gap_m'] * 1000:+.1f} mm from {inner} out to {outer} on {ray}{allowed}")
        return "; ".join(out) or "a probe failed"
    return ""
