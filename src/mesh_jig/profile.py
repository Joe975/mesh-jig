"""Measured reference outlines: the ortho views' silhouettes as numbers the agent can build to.

The reference views are the only statement of the shape that is not prose. `silhouette` scores a finished render
against them; this turns the same masks into a table BEFORE anything is built: at stations along the model, the top
view's half-width, the side view's top and bottom lines and the rear view's top and bottom lines, all in millimetres
in model axes (+Y nose, +Z up, X across, origin at the centre of the bounding box). A brief that says "a faceted
armoured wedge" gets a body half as wide as the reference; the numbers say how wide and how tall, station by station.

Views are read by name: `top` (nose at the bottom of the image), `side` (nose at the left), `rear`. The length in
millimetres comes from the contract; every other scale follows from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import glb, outline, silhouette
from .palette import Zone

STATIONS = 11
ZONE_COLS = 24          # columns in a zone map; rows follow the view's aspect
ZONE_MIN = 0.35         # a cell is painted only when this share of it is inside the silhouette
SECTION = 0.005         # half-thickness of a station's section, as a share of the length (or span): about one pixel
                        # row of a reference view
VIEWS = ("top", "side", "rear")


def _bbox(path: str | Path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    m = silhouette.mask(path)
    b = silhouette.bbox(m)
    if b is None:
        raise ValueError(f"{path}: the mask is empty")
    return m, b


def _clamp(v: float, lo: int, hi: int) -> int:
    """A station on the outline's own last pixel, never one past it (the tail station lands on bbox end)."""
    return int(min(max(round(v), lo), hi))


def _spans(m: np.ndarray, cols: list[int]) -> list[tuple[int, int] | None]:
    """First and last set row in each column, None where the column is empty."""
    out = []
    for c in cols:
        rows = np.nonzero(m[:, c])[0] if 0 <= c < m.shape[1] else np.array([])
        out.append((int(rows.min()), int(rows.max())) if len(rows) else None)
    return out


def measurable(references: dict[str, str | Path]) -> bool:
    return all(v in references and Path(references[v]).is_file() for v in VIEWS)


def measure(references: dict[str, str | Path], length_mm: float, stations: int = STATIONS) -> dict:
    """{"length_mm", "span_mm", "height_mm", "top": [{y, half_width}], "side": [{y, z_top, z_bottom}],
    "rear": [{x, z_top, z_bottom}]} from the top, side and rear views. Every number is millimetres in model axes."""
    out: dict = {"length_mm": round(float(length_mm), 1), "stations": stations}
    tm, tb = _bbox(references["top"])
    k = length_mm / (tb[3] - tb[1])                     # mm per pixel on the top view
    span = (tb[2] - tb[0]) * k
    cx = (tb[0] + tb[2]) / 2
    ys = [length_mm / 2 - i * length_mm / (stations - 1) for i in range(stations)]
    top = []
    for y in ys:
        r = _clamp(tb[3] - 1 - (length_mm / 2 - y) / k, tb[1], tb[3] - 1)
        cols = np.nonzero(tm[min(max(r, 0), tm.shape[0] - 1)])[0]
        half = (max(abs(cols.min() - cx), abs(cols.max() - cx)) * k) if len(cols) else 0.0
        top.append({"y": round(y, 1), "half_width": round(float(half), 1)})
    sm, sb = _bbox(references["side"])
    ks = length_mm / (sb[2] - sb[0])                    # nose at the left
    side_cols = [_clamp(sb[0] + (length_mm / 2 - y) / ks, sb[0], sb[2] - 1) for y in ys]
    side_spans = _spans(sm, side_cols)
    height = (sb[3] - sb[1]) * ks
    zc = (sb[1] + sb[3]) / 2
    side = [{"y": round(y, 1),
             "z_top": round(float((zc - s[0]) * ks), 1) if s else None,
             "z_bottom": round(float((zc - s[1]) * ks), 1) if s else None}
            for y, s in zip(ys, side_spans)]
    rm, rb = _bbox(references["rear"])
    kr = span / (rb[2] - rb[0])
    rcx = (rb[0] + rb[2]) / 2
    rzc = rb[3] - height / kr / 2                       # align the rear view's height with the side view's
    xs = [i * (span / 2) / (stations - 1) for i in range(stations)]
    rear_cols = [_clamp(rcx + x / kr, rb[0], rb[2] - 1) for x in xs]
    rear = [{"x": round(x, 1),
             "z_top": round(float((rzc - s[0]) * kr), 1) if s else None,
             "z_bottom": round(float((rzc - s[1]) * kr), 1) if s else None}
            for x, s in zip(xs, _spans(rm, rear_cols))]
    out |= {"span_mm": round(float(span), 1), "height_mm": round(float(height), 1), "top": top, "side": side, "rear": rear}
    return out


def _clip(poly: np.ndarray, axis: int, lo: float, hi: float) -> np.ndarray:
    """A convex polygon (N, 3) cut down to lo <= coordinate[axis] <= hi (Sutherland-Hodgman, two planes)."""
    for bound, sign in ((lo, 1.0), (hi, -1.0)):
        if not len(poly):
            break
        d = sign * (poly[:, axis] - bound)
        out = []
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            if d[i] >= 0:
                out.append(poly[i])
            if (d[i] >= 0) != (d[j] >= 0):
                out.append(poly[i] + (d[i] / (d[i] - d[j])) * (poly[j] - poly[i]))
        poly = np.array(out) if out else np.zeros((0, 3))
    return poly


def section(tris: np.ndarray, axis: int, centre: float, half: float) -> np.ndarray:
    """The points bounding the mesh's SURFACE inside the slab centre +/- half along `axis`: every triangle that
    reaches the slab, clipped to it. (N, 3), empty when the slab holds no surface."""
    lo, hi = centre - half, centre + half
    c = tris[..., axis]
    cut = [_clip(t, axis, lo, hi) for t in tris[(c.max(axis=1) >= lo) & (c.min(axis=1) <= hi)]]
    cut = [q for q in cut if len(q)]
    return np.concatenate(cut) if cut else np.zeros((0, 3))


def profile_from_mesh(tris: np.ndarray, length_mm: float, stations: int = STATIONS,
                      span_mm: float | None = None) -> dict:
    """Triangles in MODEL axes and millimetres ((T, 3, 3): x across, +y nose, +z up, centred) -> the same tables
    `measure` returns. Each station is a thin SECTION of the surface (SECTION of the length either side of it), the
    counterpart of the single pixel row or column `measure` reads at that station.

    A section, not the vertices near the station: a long flat face has no vertex between its ends, so a measure
    read off vertices reports nothing there (or some other part's vertices), and adding edge loops to an unchanged
    shape moves the number. A station whose section holds no surface reads None. The origin is the centre of the
    mesh's bounding box, as it is for the reference."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    if tris.size == 0:
        raise ValueError("no triangles")
    pts = tris.reshape(-1, 3)
    # the reference tables are measured about the centre of each view's bounding box, so the mesh is too: a model
    # that sits 5 mm high on its origin is not 5 mm wrong at every station
    tris = tris - (pts.min(axis=0) + pts.max(axis=0)) / 2
    pts = tris.reshape(-1, 3)
    span = float(2 * np.abs(pts[:, 0]).max()) if span_mm is None else float(span_mm)
    ys = [length_mm / 2 - i * length_mm / (stations - 1) for i in range(stations)]
    top, side = [], []
    for y in ys:
        sel = section(tris, 1, y, SECTION * length_mm)
        top.append({"y": round(y, 1), "half_width": round(float(np.abs(sel[:, 0]).max()), 1) if len(sel) else 0.0})
        side.append({"y": round(y, 1),
                     "z_top": round(float(sel[:, 2].max()), 1) if len(sel) else None,
                     "z_bottom": round(float(sel[:, 2].min()), 1) if len(sel) else None})
    xs = [i * (span / 2) / (stations - 1) for i in range(stations)]
    rear = []
    for x in xs:
        half = SECTION * span
        sel = np.concatenate([section(tris, 0, x, half), section(tris, 0, -x, half)])
        rear.append({"x": round(x, 1),
                     "z_top": round(float(sel[:, 2].max()), 1) if len(sel) else None,
                     "z_bottom": round(float(sel[:, 2].min()), 1) if len(sel) else None})
    return {"length_mm": round(float(pts[:, 1].max() - pts[:, 1].min()), 1), "span_mm": round(span, 1),
            "height_mm": round(float(pts[:, 2].max() - pts[:, 2].min()), 1), "stations": stations,
            "top": top, "side": side, "rear": rear}


def glb_triangles(path: str | Path) -> np.ndarray:
    """Every triangle of a GLB in MODEL axes and millimetres, (T, 3, 3): the file is glTF axes (x right, y up, nose
    along -z)."""
    gltf, blob = glb.read_glb(path)
    out = []
    for ni, world in glb.mesh_nodes(gltf):
        mesh = gltf["meshes"][gltf["nodes"][ni]["mesh"]]
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            pos = glb.accessor(gltf, blob, prim["attributes"]["POSITION"])
            pts = (np.concatenate([pos, np.ones((len(pos), 1))], axis=1) @ world.T)[:, :3] * 1000.0
            pts = np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)   # glTF (x, y up, -z nose) -> model (x, +y nose, +z up)
            idx = (glb.accessor(gltf, blob, prim["indices"]).astype(np.int64).ravel() if "indices" in prim
                   else np.arange(len(pos)))
            out.append(pts[idx[: len(idx) // 3 * 3].reshape(-1, 3)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def compare(reference: dict, mesh: dict) -> dict:
    """Per-station differences (mesh minus reference, millimetres) plus the mean absolute error of each table."""
    out: dict = {"overall": {k: round(mesh[k] - reference[k], 1) for k in ("length_mm", "span_mm", "height_mm")}}
    errs: list[float] = []
    for table, keys in (("top", ("half_width",)), ("side", ("z_top", "z_bottom")), ("rear", ("z_top", "z_bottom"))):
        rows = []
        for r, m in zip(reference[table], mesh[table]):
            row = {"x" if table == "rear" else "y": r.get("x", r.get("y"))}
            for k in keys:
                a, b = r.get(k), m.get(k)
                row[k] = None if a is None or b is None else round(b - a, 1)
                if row[k] is not None:
                    errs.append(abs(row[k]))
            rows.append(row)
        out[table] = rows
    out["mae_mm"] = round(float(np.mean(errs)), 2) if errs else None
    return out


def glb_diff(reference: dict, path: str | Path) -> dict:
    """A built GLB's own outline against the reference tables: deterministic, no render. {} when it cannot be
    measured (no reference profile, or a GLB the reader refuses)."""
    if not reference:
        return {}
    if reference.get("mode") == "standing":
        from . import standing
        return standing.compare(reference, standing.from_mesh(glb_triangles(path), reference))
    try:
        if reference.get("mode") == outline.MODE:
            return outline.compare(reference, outline.from_mesh(glb_triangles(path), reference))
        mesh = profile_from_mesh(glb_triangles(path), reference["length_mm"], reference["stations"])
        return compare(reference, mesh)
    except (ValueError, KeyError, IndexError, OSError):
        return {}


def compare_text(diff: dict) -> str:
    """The differences as the agent reads them: + means the model is bigger than the reference there."""
    if diff.get("mode") == "standing":
        from . import standing
        return standing.compare_text(diff)
    if diff.get("mode") == outline.MODE:
        return outline.compare_text(diff)

    def cell(row, keys):
        return "/".join("-" if row[k] is None else f"{row[k]:+.0f}" for k in keys)
    lines = [
        f"YOUR MODEL AGAINST THE REFERENCE OUTLINES (millimetres, + means your model is bigger there; measured from "
        f"your GLB, not judged). Mean absolute error {diff['mae_mm']} mm; overall length {diff['overall']['length_mm']:+.1f}, "
        f"span {diff['overall']['span_mm']:+.1f}, height {diff['overall']['height_mm']:+.1f}:",
        "- top view half-width error per station:",
        "    " + "  ".join(f"y{r['y']:+.0f}:{cell(r, ('half_width',))}" for r in diff["top"]),
        "- side view z_top/z_bottom error per station:",
        "    " + "  ".join(f"y{r['y']:+.0f}:{cell(r, ('z_top', 'z_bottom'))}" for r in diff["side"]),
        "- rear view z_top/z_bottom error across the half-span:",
        "    " + "  ".join(f"x{r['x']:.0f}:{cell(r, ('z_top', 'z_bottom'))}" for r in diff["rear"]),
        "Close the biggest of these and the outline matches the reference.",
    ]
    return "\n".join(lines)


def _zone_letter(rgb: np.ndarray, zones: tuple[Zone, ...]) -> str:
    """The palette letter nearest this painted colour."""
    return min(zones, key=lambda z: float(np.sum((rgb - np.array(z[2], dtype=float)) ** 2)))[0]


def zone_map(path: str | Path, zones: tuple[Zone, ...], cols: int = ZONE_COLS) -> list[str]:
    """The view's colour zones as rows of palette letters ('.' outside the silhouette), the image read on a grid."""
    if not zones:
        return []
    im = silhouette._load(path)
    a = np.asarray(im).astype(float)
    m = silhouette.mask(im)
    b = silhouette.bbox(m)
    if b is None:
        return []
    x0, y0, x1, y1 = b
    cw = (x1 - x0) / cols
    rows = max(1, int(round((y1 - y0) / cw)))
    out = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            xs = slice(int(x0 + c * cw), max(int(x0 + c * cw) + 1, int(x0 + (c + 1) * cw)))
            ys = slice(int(y0 + r * cw), max(int(y0 + r * cw) + 1, int(y0 + (r + 1) * cw)))
            cell = m[ys, xs]
            if cell.size == 0 or cell.mean() < ZONE_MIN:
                line += "."
                continue
            pix = a[ys, xs][cell]
            # the lit side of a cell, not its mean: painted shadow drags a pale colour into a dark one
            line += _zone_letter(np.percentile(pix, 75, axis=0), zones)
        out.append(line)
    return out


VIEW_HINT = {"top": "nose at the bottom", "side": "nose at the left", "rear": "nose away from you"}


def zone_text(maps: dict[str, list[str]], zones: tuple[Zone, ...], craft: bool = True) -> str:
    """The zone maps as the agent reads them. Only a craft has a nose to say where it is in a view: any other
    subject is oriented by its cameras, which the brief gives."""
    lines = ["COLOUR ZONES READ OFF THE REFERENCE VIEWS (each view on a grid, one letter per cell: "
             + ", ".join(f"{k} {name}" for k, name, _ in zones) + ", . background). "
             "This is where each colour sits in each view; match the zones, not the pixel grid:"]
    for view, rows in maps.items():
        hint = VIEW_HINT.get(view) if craft else None
        lines.append(f"  {view}" + (f" ({hint}):" if hint else ":"))
        lines += [f"    {r}" for r in rows]
    return "\n".join(lines)


def _zz(s: dict) -> str:
    """`z_top/z_bottom` for one station, `-` where the view has nothing at it."""
    f = lambda v: "-" if v is None else f"{v:+.0f}"   # noqa: E731
    return f"{f(s['z_top'])}/{f(s['z_bottom'])}"


def text(p: dict) -> str:
    """The table as the agent reads it."""
    if p.get("mode") == "standing":
        from . import standing
        return standing.text(p)
    if p.get("mode") == outline.MODE:
        return outline.text(p)
    lines = [
        "MEASURED REFERENCE OUTLINES (millimetres, model axes: +Y nose, +Z up, X across, origin at the centre of the "
        "bounding box). Measured from the reference views' silhouettes, not by eye:",
        f"- overall: length {p['length_mm']} along Y, span {p['span_mm']} across X, height {p['height_mm']} along Z.",
        "- top view, half-width at each station (y -> how far the outline reaches from the centre line):",
        "    " + "  ".join(f"y{s['y']:+.0f}:{s['half_width']:.0f}" for s in p["top"]),
        "- side view, the top and bottom of the outline at the same stations (y -> z_top/z_bottom):",
        "    " + "  ".join(f"y{s['y']:+.0f}:{_zz(s)}" for s in p["side"]),
        "- rear view, the top and bottom of the outline across the half-span (x -> z_top/z_bottom):",
        "    " + "  ".join(f"x{s['x']:.0f}:{_zz(s)}" for s in p["rear"]),
        "Build to these numbers: they are the reference's own proportions, and every measure compares the same outlines.",
    ]
    return "\n".join(lines)
