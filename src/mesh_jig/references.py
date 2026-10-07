"""What the measures see in each reference view, in the picture's own pixels: for whoever checks a reference before
freezing it, or writes a calibration (`criteria.outline`) or a region (`criteria.review.regions`) from one.

Everything is read with the scoring mask (`silhouette.mask`) and the canvas the regions are cut from
(`resemblance.normalised`), so a number taken from here is the number the measures use. Read-only: nothing is
written but the pictures `write_seen` is asked for.

- `facts(path)`: the subject's box, its ground line and centre, how many separate pieces the mask keeps and which
  edges of the picture it reaches.
- `warnings(facts)`: what in those facts spoils a measure, in words.
- `region_box(path, pixel_box)`: a rectangle drawn on the reference, as the `box` a review region takes.
- `text(project)`: the report `mesh-jig refs` prints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import brief, glb, outline, render, resemblance, silhouette, standing

SCALE_TOLERANCE = 0.02      # standing views whose heights differ by more than this do not share a pixel scale
SEEN_SUFFIX = "_seen"


def facts(path: str | Path) -> dict:
    """{"size": [w, h], "box": [x0, y0, x1, y1] | None, "width_px", "height_px", "centre_x", "ground_y", "pieces",
    "touches": [edge, ...]} in the picture's own pixels (x right, y down; the box's right and bottom are the first
    pixel past the subject). The mask is made at `silhouette.MAX_DIM` pixels at most, so on a larger picture a
    number is good to `within_px`: two pixels of the mask."""
    m = silhouette.mask(path)
    with Image.open(path) as im:
        w, h = im.size
    out: dict = {"size": [w, h], "box": None, "pieces": 0, "touches": [],
                 "within_px": max(1, math.ceil(2 * max(w, h) / silhouette.MAX_DIM)) if max(w, h) > silhouette.MAX_DIM else 1}
    b = silhouette.bbox(m)
    if b is None:
        return out
    kx, ky = w / m.shape[1], h / m.shape[0]
    x0, y0, x1, y1 = b
    box = [round(x0 * kx), round(y0 * ky), round(x1 * kx), round(y1 * ky)]
    out.update(box=box, width_px=box[2] - box[0], height_px=box[3] - box[1], centre_x=round((box[0] + box[2]) / 2),
               ground_y=box[3], pieces=len(silhouette._components(m)),
               touches=[edge for edge, hit in (("left", x0 == 0), ("top", y0 == 0), ("right", x1 == m.shape[1]),
                                               ("bottom", y1 == m.shape[0])) if hit])
    return out


def warnings(f: dict) -> list[str]:
    """What in a view's facts spoils a measure."""
    if f["box"] is None:
        return ["no subject found: nothing in the picture stands off the colour of its border"]
    out = []
    if f["touches"]:
        out.append(f"the subject reaches the {', '.join(f['touches'])} edge of the picture. The background colour is "
                   "read off the border and a cut-off subject is measured as drawn: widen the crop")
    if f["pieces"] > 1:
        out.append(f"{f['pieces']} separate pieces are taken as the subject. If one is a label, a floor shadow or part "
                   "of the next view, tighten the crop or paint it out (`sheet.crops.<view>.mask`); a piece under "
                   f"{silhouette.MIN_COMPONENT:.0%} of the largest is dropped already")
    return out


def region_box(path: str | Path, pixel_box) -> list[float]:
    """A rectangle [x0, y0, x1, y1] in the reference picture's pixels, as the `box` of a `criteria.review` region:
    0..1 on the square canvas the subject is fitted to, longer side filling it, shorter side centred
    (`resemblance.normalised`)."""
    f = facts(path)
    if f["box"] is None:
        raise ValueError("no subject in the picture: a region is placed on the subject's own canvas")
    x0, y0, x1, y1 = (float(v) for v in pixel_box)
    if not (x0 < x1 and y0 < y1):
        raise ValueError("a region is x0 y0 x1 y1 in pixels, with x0 < x1 and y0 < y1")
    bx0, by0, bx1, by1 = f["box"]
    side = max(bx1 - bx0, by1 - by0)
    pad_x, pad_y = (side - (bx1 - bx0)) / 2, (side - (by1 - by0)) / 2
    out = [(x0 - bx0 + pad_x) / side, (y0 - by0 + pad_y) / side, (x1 - bx0 + pad_x) / side, (y1 - by0 + pad_y) / side]
    out = [round(min(1.0, max(0.0, v)), 3) for v in out]
    if not (out[0] < out[2] and out[1] < out[3]):
        raise ValueError(f"the rectangle misses the subject, whose box is {f['box']}")
    return out


def camera_words(shot: dict | None) -> str:
    """Where a view's camera stands and how its picture is turned, in Blender axes, and what it is read for."""
    if shot is None:
        return "no camera of this name: never measured (name the picture after a camera, or add a shot)"
    _eye, (right, up, forward), _fov = render.camera(shot)
    kind = ("orthographic, along one axis: read for the outline table" if outline.view_axes(shot) is not None
            else "orthographic, across two axes: compared, not read for the outline table" if shot.get("ortho")
            else "perspective: compared, not read for the outline table")
    return (f"the camera stands on the {brief.blender_axis(-forward)} side, picture right is "
            f"{brief.blender_axis(right)}, picture up is {brief.blender_axis(up)} ({kind})")


def calibration(views: dict[str, dict]) -> dict | None:
    """A `criteria.outline.calibration` to start from, from the standing views' own boxes: the height of the first
    of front, side, rear, right the project has, and for each of them [the middle of its box, its ground line]. The
    middle of the box is the subject's centreline only when nothing sticks out to one side."""
    found = [v for v in standing.VIEW_AXES if views.get(v, {}).get("box")]
    if not found:
        return None
    return {"view": found[0], "height_px": views[found[0]]["height_px"],
            "origins_px": {v: [views[v]["centre_x"], views[v]["ground_y"]] for v in found}}


def text(project, regions: list[tuple[str, list[float]]] = ()) -> str:
    """The report: each reference view as the measures see it, what would spoil a measure, the standing views'
    heights against each other with a calibration to start from, and each asked region as a review `box`."""
    refs = project.references
    shots = {s["name"]: s for s in render.shots_for(project.spec)
             if s.get("variant") in (None, glb.judged_variant(project.spec))}
    views = {v: facts(p) for v, p in refs.items()}
    lines = ["REFERENCE VIEWS as the measures see them (each picture's own pixels: x right, y down):"]
    for v, f in views.items():
        w, h = f["size"]
        if f["box"] is None:
            lines.append(f"- {v}: {w}x{h}")
        else:
            lines.append(f"- {v}: {w}x{h}, subject box {f['box']} = {f['width_px']} wide, {f['height_px']} tall; "
                         f"centre x {f['centre_x']}, ground y {f['ground_y']}"
                         + (f" (to within {f['within_px']} px: the mask is {silhouette.MAX_DIM} px at most)"
                            if f["within_px"] > 1 else ""))
        lines.append(f"    {camera_words(shots.get(v))}")
        lines += [f"    ! {w}" for w in warnings(f)]
    start = calibration(views)
    if start:
        base = views[start["view"]]["height_px"]
        heights = {v: views[v]["height_px"] for v in start["origins_px"]}
        lines.append("\nSTANDING VIEWS, for a calibrated outline table (`criteria.outline`; docs/ORGANIC.md):")
        lines.append("- subject heights: " + ", ".join(
            f"{v} {px} px" + ("" if v == start["view"] else f" ({px / base - 1:+.1%})") for v, px in heights.items()))
        off = [v for v, px in heights.items() if abs(px / base - 1) > SCALE_TOLERANCE]
        if off:
            lines.append(f"    ! {', '.join(off)} differ from {start['view']} by more than {SCALE_TOLERANCE:.0%}: the "
                         "standing table needs every view drawn at one pixel scale. Redraw or rescale the sheet; do "
                         "not calibrate each view to its own height")
        lines.append("- a calibration to start from. `height_px` is the subject's height in the calibration view and "
                     "each origin is [x of the subject's centreline, y of the ground]. The x here is the middle of the "
                     "subject's box: move it wherever a tail, an arm or a weapon pushes the box off the centreline:")
        lines.append('  "calibration": ' + json.dumps(start))
        lines.append("- a station's height from a pixel row y in the calibration view: "
                     "z_m = (ground y - y) / height_px * model.height_m")
    for view, pixel_box in regions:
        if view not in refs:
            raise ValueError(f"--region {view}: no reference view of that name (there are {', '.join(refs)})")
        lines.append(f"\nREGION on {view}, pixels {[round(v) for v in pixel_box]}, as a `criteria.review.regions` entry:")
        lines.append("  " + json.dumps({"name": "<name>", "view": view, "box": region_box(refs[view], pixel_box)}))
    return "\n".join(lines)


def write_seen(project, out_dir: str | Path) -> list[Path]:
    """<view>_seen.png for every reference view: the picture with everything the mask does not take as the subject
    washed out, the subject's box, its ground line and its centre line drawn on it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for view, path in project.references.items():
        f = facts(path)
        with Image.open(path) as im:
            im = im.convert("RGB")
        m = Image.fromarray(np.where(silhouette.mask(path), 255, 0).astype(np.uint8), "L").resize(im.size, Image.NEAREST)
        seen = Image.composite(im, Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.8), m)
        if f["box"]:
            d = ImageDraw.Draw(seen)
            x0, y0, x1, y1 = f["box"]
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(200, 40, 40))
            d.line([(0, y1), (im.width, y1)], fill=(40, 90, 200))
            d.line([(f["centre_x"], 0), (f["centre_x"], im.height)], fill=(40, 160, 70))
        p = out_dir / f"{view}{SEEN_SUFFIX}.png"
        seen.save(p)
        written.append(p)
    return written
