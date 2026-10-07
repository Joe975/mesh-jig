"""Calibrated upright outlines; Blender metres, +Z up/+Y front, no recentering or mirroring.

All reference images share a declared pixel scale, derived from ONE height calibration.
Origins are original-image pixels at world x/y=0 and z=0. Masks may be internally resized.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import glb, profile, render, silhouette

# world horizontal axis and sign of image-right in Blender axes
VIEW_AXES = {"front": (0, -1), "rear": (0, 1), "side": (1, -1), "right": (1, 1)}


def _positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate(config: dict, spec: dict, refs: dict) -> None:
    if not isinstance(config, dict) or config.get("mode") != "standing":
        raise ValueError("profile.mode must be standing")
    if not _positive(spec.get("height_m")):
        raise ValueError("standing profile needs model.height_m")
    origin = config.get("origin_m", [0, 0, 0])
    if not isinstance(origin, list) or len(origin) != 3 or not all(isinstance(v, (int, float))
            and not isinstance(v, bool) and math.isfinite(v) for v in origin):
        raise ValueError("profile.origin_m must be a finite Blender [x,y,z] ground-origin point")
    c = config.get("calibration") or {}
    if c.get("view") not in VIEW_AXES or not _positive(c.get("height_px")):
        raise ValueError("profile.calibration needs a standing view and positive height_px in original pixels")
    origins = c.get("origins_px")
    if not isinstance(origins, dict) or not origins or c["view"] not in origins:
        raise ValueError("profile.calibration.origins_px needs the calibration view and every sampled view")
    shots = {s["name"]: s for s in render.shots_for(spec)}
    for view, origin in origins.items():
        if view not in VIEW_AXES or view not in refs:
            raise ValueError(f"standing profile: missing/unsupported reference {view}")
        if not isinstance(origin, (list, tuple)) or len(origin) != 2 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in origin):
            raise ValueError(f"{view}: origin_px must be [horizontal zero, ground baseline]")
        shot = shots.get(view, {})
        if shot.get("variant") != glb.judged_variant(spec):
            raise ValueError(f"{view}: standing profile camera must draw the judged variant")
        if not _positive(shot.get("ortho")):
            raise ValueError(f"{view}: standing profiles require calibrated orthographic shots")
        _, basis, _ = render.camera(shot)
        axis, sign = VIEW_AXES[view]
        # model X -> GLB X, model Y -> GLB -Z
        horizontal = np.array([sign, 0, 0] if axis == 0 else [0, 0, -sign])
        if not np.allclose(basis[0], horizontal, atol=1e-6) or not np.allclose(basis[1], [0, 1, 0], atol=1e-6):
            raise ValueError(f"{view}: camera must be cardinal, upright, and follow the standing view orientation")
    stations = config.get("stations")
    if not isinstance(stations, list) or not stations:
        raise ValueError("profile.stations needs named {name, z_m} stations above the fixed ground origin")
    names = set()
    for s in stations:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not s["name"] or s["name"] in names:
            raise ValueError("standing stations need unique nonempty names")
        z = s.get("z_m")
        if not isinstance(z, (int, float)) or not math.isfinite(z) or not 0 <= z <= spec["height_m"]:
            raise ValueError("station z_m must be within the declared standing height")
        names.add(s["name"])


def measure(refs: dict, spec: dict, config: dict) -> dict:
    validate(config, spec, refs)
    c = config["calibration"]
    mm_per_px = spec["height_m"] * 1000 / c["height_px"]
    out = {"mode": "standing", "height_mm": spec["height_m"] * 1000,
           "origin_m": config.get("origin_m", [0, 0, 0]),
           "mm_per_pixel": mm_per_px, "stations": config["stations"], "views": {}}
    for view, (ox, baseline) in c["origins_px"].items():
        m = silhouette.mask(refs[view])
        with Image.open(refs[view]) as im:
            kx, ky = im.width / m.shape[1], im.height / m.shape[0]
        axis, sign = VIEW_AXES[view]
        rows = []
        for s in config["stations"]:
            r = int(math.floor((baseline - s["z_m"] * 1000 / mm_per_px) / ky))
            xs = np.flatnonzero(m[r]) if 0 <= r < m.shape[0] else np.array([])
            # Pixel edges bound occupied area; keep signed asymmetric endpoints.
            values = [(xs.min() * kx - ox) * mm_per_px * sign,
                      ((xs.max() + 1) * kx - ox) * mm_per_px * sign] if len(xs) else []
            rows.append({"name": s["name"], "z_mm": s["z_m"] * 1000,
                         "min_mm": round(min(values), 3) if values else None,
                         "max_mm": round(max(values), 3) if values else None})
        out["views"][view] = {"axis": "xy"[axis], "rows": rows}
    return out


def from_mesh(tris: np.ndarray, reference: dict) -> dict:
    """Model-axis millimetres, actual origin preserved; section size is one calibrated pixel."""
    tris = np.asarray(tris, dtype=float) - np.asarray(reference.get("origin_m", [0, 0, 0])) * 1000
    out = {"mode": "standing", "height_mm": float(np.ptp(tris[..., 2])) if tris.size else 0, "views": {}}
    for view, table in reference["views"].items():
        axis = "xy".index(table["axis"])
        rows = []
        for row in table["rows"]:
            points = profile.section(tris, 2, row["z_mm"], reference["mm_per_pixel"] / 2)
            rows.append({"name": row["name"], "z_mm": row["z_mm"],
                         "min_mm": round(float(points[:, axis].min()), 3) if len(points) else None,
                         "max_mm": round(float(points[:, axis].max()), 3) if len(points) else None})
        out["views"][view] = {"axis": table["axis"], "rows": rows}
    return out


def compare(reference: dict, mesh: dict) -> dict:
    errors, missing, extra, views = [], [], [], {}
    for view, table in reference["views"].items():
        rows = []
        for r, m in zip(table["rows"], mesh["views"][view]["rows"]):
            present_r, present_m = r["min_mm"] is not None, m["min_mm"] is not None
            if present_r and not present_m:
                missing.append(f"{view}:{r['name']}")
            if present_m and not present_r:
                extra.append(f"{view}:{r['name']}")
            d = {"name": r["name"], "z_mm": r["z_mm"], "missing": present_r and not present_m}
            for key in ("min_mm", "max_mm"):
                d[key] = round(m[key] - r[key], 3) if present_r and present_m else None
                if d[key] is not None:
                    errors.append(abs(d[key]))
            d["width_mm"] = round(d["max_mm"] - d["min_mm"], 3) if present_r and present_m else None
            rows.append(d)
        views[view] = {"axis": table["axis"], "rows": rows}
    # An absent station must not improve MAE by dropping the largest discrepancy.
    return {"mode": "standing", "height_error_mm": round(mesh["height_mm"] - reference["height_mm"], 3),
            "views": views, "missing_stations": missing, "extra_stations": extra,
            "mae_mm": round(float(np.mean(errors)), 3) if errors and not missing and not extra else None}


def text(reference: dict) -> str:
    lines = [f"STANDING OUTLINES (Blender mm relative to fixed ground origin {reference.get('origin_m', [0, 0, 0])} m; "
             f"+Y front; {reference['mm_per_pixel']:.4f} mm/pixel).",
             "Signed endpoints retain handedness; front/rear measure X, side/right measure Y."]
    for view, table in reference["views"].items():
        lines.append(view + ": " + "  ".join(f"{r['name']} z{r['z_mm']:.0f}: {r['min_mm']}/{r['max_mm']}"
                                             for r in table["rows"]))
    return "\n".join(lines)


def compare_text(diff: dict) -> str:
    lines = [f"STANDING OUTLINE ERRORS (model minus reference, mm): height {diff['height_error_mm']:+.3f}; "
             f"endpoint MAE {diff['mae_mm']}; missing {diff['missing_stations']}; extra {diff['extra_stations']}."]
    for view, table in diff["views"].items():
        lines.append(view + " min/max/width: " + "  ".join(
            f"{r['name']} z{r['z_mm']:.0f}:{r['min_mm']}/{r['max_mm']}/{r['width_mm']}" for r in table["rows"]))
    return "\n".join(lines)
