"""Reference outlines for any subject, read through the project's own cameras.

`profile.measure` needs a craft (a `length_m`, views named top, side and rear, the nose where those names put it) and
`standing` needs a hand calibration (pixel origins, named heights). A project with neither had no outline table at
all: nothing measured to build to, and no outline error. This is the table for that project. It asks only what the
cameras already say:

- a view counts when its shot is orthographic and looks along one model axis, so picture-right and picture-up are
  each a model axis (read from `render.camera`, never from the view's name);
- each such view is sampled at `STATIONS` stations along the longer side of its silhouette, and at each the outline's
  two ends across are recorded, signed, in model (Blender) axes, from the centre of the bounding box;
- the views' bounding boxes give the subject's proportions (least squares over every view, since drawn views never
  agree exactly), and one unit ties them together: millimetres when the contract declares an extent along an axis a
  view shows (`height_m`, `length_m`, `span_m`, `axis_extents_m`), thousandths of the longest extent when it does not.

A built GLB is read the same way (`from_mesh`): a thin section of its surface at each station, placed at the same
fraction of its own extent. With no declared extent the reference is scaled to the model's extent along the unit
axis, so the error is in millimetres on the model's own scale, as the outline overlap is scale-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import glb, render, silhouette

MODE = "views"
STATIONS = 11
AXES = "XYZ"                     # model (Blender) axes: X across, Y front to back, Z up
GLB_TO_MODEL = {"X": "X", "Y": "Z", "Z": "Y"}       # the axis a GLB extent in `axis_extents_m` is in the model


def view_axes(shot: dict) -> tuple[tuple[int, float], tuple[int, float]] | None:
    """((axis, sign) of picture-right, (axis, sign) of picture-up) in model axes for an orthographic shot that looks
    along one axis; None for a perspective shot or one that looks across two."""
    if not shot.get("ortho"):
        return None
    _eye, basis, _fov = render.camera(shot)
    out = []
    for v in basis[:2]:
        b = np.array([v[0], -v[2], v[1]])                    # glTF (x, y up, z) -> model (x, -z, y)
        k = int(np.argmax(np.abs(b)))
        if abs(abs(b[k]) - 1.0) > 1e-6:
            return None
        out.append((k, 1.0 if b[k] > 0 else -1.0))
    return out[0], out[1]


def measurable_views(references: dict[str, str | Path], spec: dict) -> dict[str, tuple]:
    """view -> its axes, for every shot of the judged variant that has a reference and looks along one axis."""
    judged = glb.judged_variant(spec)
    out = {}
    for shot in render.shots_for(spec):
        name = shot["name"]
        if name in references and shot.get("variant") == judged and Path(references[name]).is_file():
            axes = view_axes(shot)
            if axes is not None:
                out[name] = axes
    return out


def declared_mm(spec: dict) -> dict[int, float]:
    """model axis -> the extent the contract declares along it, in millimetres."""
    out: dict[int, float] = {}
    for key, axis in (("height_m", "Z"), ("length_m", "Y"), ("span_m", "X")):
        if spec.get(key):
            out[AXES.index(axis)] = float(spec[key]) * 1000.0
    for g, metres in (spec.get("axis_extents_m") or {}).items():
        out.setdefault(AXES.index(GLB_TO_MODEL[g]), float(metres) * 1000.0)
    return out


def _extents(boxes: dict[str, tuple[int, int, float, float]]) -> dict[int, float]:
    """The subject's relative extent along each axis a view shows: least squares, in log space, over every view's
    bounding-box aspect. The first axis seen is 1; the caller rescales."""
    seen = sorted({a for ka, kb, _w, _h in boxes.values() for a in (ka, kb)})
    rows, rhs = [], []
    for ka, kb, w, h in boxes.values():
        r = np.zeros(len(seen))
        r[seen.index(ka)], r[seen.index(kb)] = 1.0, -1.0
        rows.append(r)
        rhs.append(np.log(w / h))
    pin = np.zeros(len(seen))
    pin[0] = 1.0
    sol, *_ = np.linalg.lstsq(np.array(rows + [pin]), np.array(rhs + [0.0]), rcond=None)
    return {a: float(np.exp(v)) for a, v in zip(seen, sol)}


def measure(references: dict[str, str | Path], spec: dict, stations: int = STATIONS) -> dict:
    """{"mode": "views", "unit_axis", "unit_mm" (None: thousandths), "extents": {axis: of the unit axis},
    "views": {view: {"along", "across", "half", "rows": [{"f", "min", "max"}]}}}, {} when no view can be read.
    `f` is the station's signed place along the view's `along` axis as a fraction of the extent (-0.5 .. 0.5 from the
    bounding box centre); `min` and `max` are the outline's ends along `across`, as fractions of the unit axis's
    extent, from the bounding box centre. A station the silhouette does not reach reads None."""
    axes = measurable_views(references, spec)
    masks, boxes = {}, {}
    for view, ((ka, _sa), (kb, _sb)) in axes.items():
        m = silhouette.mask(references[view])
        b = silhouette.bbox(m)
        if b is None:
            continue
        masks[view] = (m, b)
        boxes[view] = (ka, kb, float(b[2] - b[0]), float(b[3] - b[1]))
    if not boxes:
        return {}
    rel = _extents(boxes)
    declared = {a: mm for a, mm in declared_mm(spec).items() if a in rel}
    unit = max(declared or rel, key=lambda a: (rel[a], -a))     # the longest side that has a size, else the longest
    rel = {a: v / rel[unit] for a, v in rel.items()}
    out: dict = {"mode": MODE, "unit_axis": AXES[unit], "unit_mm": declared.get(unit), "stations": stations,
                 "extents": {AXES[a]: round(v, 4) for a, v in sorted(rel.items())}, "views": {}}
    for view, (m, (x0, y0, x1, y1)) in masks.items():
        (ka, sa), (kb, sb) = axes[view]
        w, h = x1 - x0, y1 - y0
        wide = w >= h                                        # stations run along the longer side
        along, across, s_along, s_across = (ka, kb, sa, sb) if wide else (kb, ka, sb, sa)
        k = rel[along] / (w if wide else h)                  # unit-axis fractions per pixel
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rows = []
        for i in range(stations):
            f = (i + 0.5) / stations                         # along the picture: left to right, or bottom to top
            if wide:
                line = np.flatnonzero(m[y0:y1, min(x0 + int(f * w), x1 - 1)]) + y0
                ends = [(cy - line.min()) * k, (cy - line.max() - 1) * k] if len(line) else []
            else:
                line = np.flatnonzero(m[min(y0 + int((1 - f) * h), y1 - 1), x0:x1]) + x0
                ends = [(line.min() - cx) * k, (line.max() + 1 - cx) * k] if len(line) else []
            ends = [float(e * s_across) for e in ends]       # pixel edges bound the body; signed in the model axis
            rows.append({"f": round(s_along * (f - 0.5), 4),
                         "min": round(min(ends), 5) if ends else None, "max": round(max(ends), 5) if ends else None})
        rows.sort(key=lambda r: r["f"])
        out["views"][view] = {"along": AXES[along], "across": AXES[across], "half": round(0.5 / (w if wide else h), 5),
                              "rows": rows}
    return out


def from_mesh(tris: np.ndarray, reference: dict) -> dict:
    """Triangles in model axes and millimetres ((T, 3, 3), as `profile.glb_triangles` gives them) read as `measure`
    reads a view: at each station a section of the surface one reference pixel thick, its ends across in millimetres
    from the centre of the mesh's bounding box. `scale_mm` is what one reference unit is on this model."""
    from . import profile
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    if tris.size == 0:
        raise ValueError("no triangles")
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre, extent = (lo + hi) / 2, hi - lo
    scale = reference.get("unit_mm") or float(extent[AXES.index(reference["unit_axis"])])
    out = {"mode": MODE, "scale_mm": scale, "extents_mm": {a: round(float(extent[AXES.index(a)]), 3) for a in AXES},
           "views": {}}
    for view, table in reference["views"].items():
        along, across = AXES.index(table["along"]), AXES.index(table["across"])
        rows = []
        for r in table["rows"]:
            at = centre[along] + r["f"] * extent[along]
            sel = profile.section(tris, along, at, table["half"] * extent[along])
            rows.append({"at_mm": round(float(r["f"] * extent[along]), 1),
                         "min_mm": round(float(sel[:, across].min() - centre[across]), 3) if len(sel) else None,
                         "max_mm": round(float(sel[:, across].max() - centre[across]), 3) if len(sel) else None})
        out["views"][view] = {"along": table["along"], "across": table["across"], "rows": rows}
    return out


def compare(reference: dict, mesh: dict) -> dict:
    """Per station, the model's two ends minus the reference's (millimetres on the model's scale) and the width
    difference; `overall` is each extent's difference; `mae_mm` the mean absolute end error. A station only one side
    has costs that side's whole width, so leaving a part out never lowers the error."""
    scale = mesh["scale_mm"]
    errors, missing, extra, views = [], [], [], {}
    for view, table in reference["views"].items():
        rows = []
        for r, m in zip(table["rows"], mesh["views"][view]["rows"]):
            name = f"{table['along'].lower()}{m['at_mm']:+.0f}"
            has_r, has_m = r["min"] is not None, m["min_mm"] is not None
            d = {"name": name, "min_mm": None, "max_mm": None, "width_mm": None}
            if has_r and has_m:
                d["min_mm"] = round(m["min_mm"] - r["min"] * scale, 1)
                d["max_mm"] = round(m["max_mm"] - r["max"] * scale, 1)
                d["width_mm"] = round(d["max_mm"] - d["min_mm"], 1)
                errors += [abs(d["min_mm"]), abs(d["max_mm"])]
            elif has_r:
                missing.append(f"{view}:{name}")
                errors.append((r["max"] - r["min"]) * scale)
            elif has_m:
                extra.append(f"{view}:{name}")
                errors.append(m["max_mm"] - m["min_mm"])
            rows.append(d)
        views[view] = {"along": table["along"], "across": table["across"], "rows": rows}
    overall = {a: round(mesh["extents_mm"][a] - v * scale, 1) for a, v in reference["extents"].items()}
    return {"mode": MODE, "overall": overall, "views": views, "missing_stations": missing, "extra_stations": extra,
            "scale_mm": round(scale, 1), "declared": reference.get("unit_mm") is not None,
            "mae_mm": round(float(np.mean(errors)), 2) if errors else None}


def _unit_words(reference: dict) -> str:
    if reference.get("unit_mm"):
        return "millimetres"
    return (f"THOUSANDTHS of the subject's extent along {reference['unit_axis']} (the contract declares no size: "
            f"multiply by the extent along {reference['unit_axis']} you build, in mm, and divide by 1000)")


def text(reference: dict) -> str:
    """The table as the agent reads it."""
    k = reference.get("unit_mm") or 1000.0
    num = lambda v: "-" if v is None else f"{v * k:+.0f}"   # noqa: E731
    lines = [
        f"MEASURED REFERENCE OUTLINES, in {_unit_words(reference)}. Blender axes (+Z up), every position from the centre "
        "of the subject's bounding box. Read off the reference views' silhouettes through the project's cameras, not "
        "by eye:",
        "- overall extents: " + ", ".join(f"{a} {v * k:.0f}" for a, v in reference["extents"].items()) + ".",
    ]
    for view, table in reference["views"].items():
        a, c = table["along"], table["across"]
        ext = reference["extents"][a]
        lines.append(f"- {view} view, at stations along {a}: where the outline ends along {c} ({a.lower()} -> "
                     f"the -{c} end/the +{c} end):")
        lines.append("    " + "  ".join(f"{a.lower()}{r['f'] * ext * k:+.0f}:{num(r['min'])}/{num(r['max'])}"
                                         for r in table["rows"]))
    lines.append("Build to these numbers: they are the reference's own proportions, and every measure compares the same "
                 "outlines. The ends are signed and are NOT mirrored: a subject that is not symmetric reads as it is drawn.")
    return "\n".join(lines)


def compare_text(diff: dict) -> str:
    """The differences as the agent reads them: + means the model reaches further out at that end."""
    def cell(row) -> str:
        if row["min_mm"] is None:
            return "none"
        return f"{-row['min_mm']:+.0f}/{row['max_mm']:+.0f}"
    lines = [
        "YOUR MODEL AGAINST THE REFERENCE OUTLINES (millimetres"
        + ("" if diff.get("declared") else f" on your model's own scale: the reference is sized to your extent along "
           f"the unit axis, {diff['scale_mm']:.0f} mm") + "; + means your model reaches FURTHER OUT at that end, - "
        f"that it falls short; measured from your GLB, not judged). Mean absolute error {diff['mae_mm']} mm; overall "
        "extents " + ", ".join(f"{a} {v:+.0f}" for a, v in diff["overall"].items()) + ":"]
    for view, table in diff["views"].items():
        a, c = table["along"], table["across"]
        lines.append(f"- {view} view, stations along {a}, error at the -{c} end/the +{c} end:")
        lines.append("    " + "  ".join(f"{r['name']}:{cell(r)}" for r in table["rows"]))
    for label, names in (("the reference has body and yours has none", diff["missing_stations"]),
                         ("yours has body and the reference has none", diff["extra_stations"])):
        if names:
            lines.append(f"- stations where {label}: {', '.join(names)}")
    lines.append("Close the biggest of these and the outline matches the reference.")
    return "\n".join(lines)
