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


def across(project, view: str, centre_px: float | None = None, who: str = "--articulation") -> dict:
    """How a reference view whose camera looks across (not up or down) maps to Blender metres: `right`, the Blender
    axis the picture's right runs along; `per_px`, the subject's scale in it (`model.height_m` over its height in
    pixels); `centre`, the model's centreline in the picture (`centre_px`, else the project's calibration origin for
    the view, else the middle of the subject's box, which a tail or a raised arm pushes off it) and `said`, which of
    those it is; and the view's `facts`. A pixel (x, y) is `right * (x - centre) * per_px` across and
    `(ground_y - y) * per_px` above the ground."""
    refs = project.references
    if view not in refs:
        raise ValueError(f"{who} {view}: no reference view of that name (there are {', '.join(refs)})")
    height = (project.spec or {}).get("height_m")
    if not height:
        raise ValueError(f"{who} needs model.height_m: a standing subject's scale comes from its height")
    shot = next((s for s in render.shots_for(project.spec) if s["name"] == view), None)
    if shot is None or "cam_pos" not in shot:
        raise ValueError(f"{who} {view}: the project has no camera of that name")
    forward = np.asarray(shot.get("cam_target", (0, 0, 0)), float) - np.asarray(shot["cam_pos"], float)
    if abs(forward[1]) > 1e-6 * np.linalg.norm(forward):
        raise ValueError(f"{who} {view}: its camera looks up or down; pick a view that looks across")
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    right = np.array([right[0], -right[2], right[1]])          # glTF -> Blender
    f = facts(refs[view])
    if f["box"] is None:
        raise ValueError(f"{who} {view}: no subject found in the picture")
    calibration = ((project.data.get("criteria") or {}).get("outline") or {}).get("calibration") or {}
    origins = calibration.get("origins_px") or {}
    if centre_px is not None:
        centre, said = float(centre_px), f"the centreline given, x {centre_px:g}"
    elif view in origins:
        centre, said = float(origins[view][0]), f"the calibration's centreline, x {origins[view][0]:g}"
    else:
        centre, said = float(f["centre_x"]), (f"the middle of the subject's box, x {f['centre_x']}: if a tail or an "
                                               "arm pushes the box off the body's centreline, give --centre")
    return {"right": right, "per_px": float(height) / f["height_px"], "centre": centre, "said": said, "facts": f}


def articulation_path(project, view: str, points_px: list[tuple[float, float]],
                      centre_px: float | None = None) -> tuple[list[list[float]], str]:
    """A `review.articulations` path from points picked on one reference view, root first: each pixel turned into
    Blender metres through the view's camera (`across`), with Z above the ground line. The axis the view looks along
    is 0: a part off that plane (an arm held forward, read on the front view) needs it set by hand, or read off a
    second view. The second value returned says which centreline was used."""
    frame = across(project, view, centre_px)
    right, per_px, ground_y = frame["right"], frame["per_px"], frame["facts"]["ground_y"]
    out = []
    for x, y in points_px:
        p = right * (float(x) - frame["centre"]) * per_px + np.array([0.0, 0.0, (ground_y - float(y)) * per_px])
        out.append([round(float(v), 3) + 0.0 for v in p])
    return out, frame["said"]


RULED_SUFFIX = "_ruled"
RULE_MARGIN = (52, 22)      # pixels added left of and below a ruled picture for its numbers


def rule_step(height_m: float) -> float:
    """A round spacing that puts about ten lines up a subject: 0.1 m on a 1 m one, 0.02 m on a 0.2 m one."""
    raw = height_m / 10
    power = 10 ** math.floor(math.log10(raw))
    return next(power * k for k in (1, 2, 5, 10) if power * k >= raw * 0.75)


def ground_z(project) -> float:
    """Blender Z of the ground the reference views stand on: the outline's `origin_m` Z, else 0."""
    stated = ((project.data.get("criteria") or {}).get("outline") or (project.spec or {}).get("profile") or {})
    return float((stated.get("origin_m") or (0, 0, 0))[2])


def ruled(project, view: str, picture: Image.Image | None = None) -> Image.Image | None:
    """The reference view with lines at round Blender coordinates: across, the axis its picture's right runs along,
    from the model's centreline (`across`); up, Blender Z, from the ground at the outline's `origin_m` Z (0 when the
    project states none). A builder reads where a part of the reference sits in the coordinates its script writes,
    which the side-by-sides, each cropped and scaled to its own subject, cannot show. Lines at 0 are red. None for a
    view `across` cannot read (no height, a camera looking up or down, no subject). `picture`, drawn in the
    reference's own pixels (`in_frame`), is ruled in place of the reference."""
    try:
        frame = across(project, view, who="ruled")
    except ValueError:
        return None
    ground = ground_z(project)
    axis = int(np.argmax(np.abs(frame["right"])))
    sign, name = float(np.sign(frame["right"][axis])), "XYZ"[axis]
    per_px, centre, ground_y = frame["per_px"], frame["centre"], frame["facts"]["ground_y"]
    step = rule_step(float(project.spec["height_m"]))
    digits = max(0, -math.floor(math.log10(step)))
    if picture is not None:
        im = picture.convert("RGB")
    else:
        with Image.open(project.references[view]) as im:
            im = im.convert("RGB")
    left, below = RULE_MARGIN
    out = Image.new("RGB", (im.width + left, im.height + below), (255, 255, 255))
    out.paste(im, (left, 0))
    d = ImageDraw.Draw(out)

    def values(ends):
        lo, hi = min(ends), max(ends)
        return [k * step for k in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)]

    def colour(at_zero):
        return (220, 60, 60) if at_zero else (150, 160, 235)

    for value in values([sign * (x - centre) * per_px for x in (0, im.width - 1)]):
        x = left + centre + value / (sign * per_px)
        d.line([(x, 0), (x, im.height - 1)], fill=colour(abs(value) < step / 100))
        d.text((x - 14, im.height + 5), f"{name}{value:+.{digits}f}", fill=(30, 30, 140))
    for z in values([ground + (ground_y - y) * per_px for y in (0, im.height - 1)]):
        y = ground_y - (z - ground) / per_px
        d.line([(left, y), (out.width - 1, y)], fill=colour(abs(z) < step / 100))
        d.text((2, y - 6), f"Z{z:+.{digits}f}", fill=(30, 30, 140))
    return out


def in_frame(project, view: str, render_path: str | Path) -> Image.Image | None:
    """A build's render of `view` resampled into the reference view's own pixels: each pixel shows the point of the
    build at the Blender coordinates the reference's pixel reads as (`across`, the ground at `ground_z`), so the two
    share a scale, a centreline and a ground line. Ruled alike (`ruled`), a tail 0.13 m short is 0.13 m short
    against the same line, which the side-by-sides, each scaled to its own subject, hide. None for a view `ruled`
    cannot read or a camera that is not orthographic."""
    try:
        frame = across(project, view, who="ruled")
    except ValueError:
        return None
    shot = next((s for s in render.shots_for(project.spec) if s["name"] == view), None)
    if not shot or not shot.get("ortho"):
        return None
    with Image.open(render_path) as r:
        pixels = np.asarray(r.convert("RGB"))
    height, width = pixels.shape[:2]
    eye, (right, up, _forward), _ = render.camera(shot)
    scale, per_px = render.stage_scale(project.spec), frame["per_px"]
    px_per_unit = height / float(shot["ortho"])
    f = frame["facts"]
    w, h = Image.open(project.references[view]).size
    xs = (np.arange(w) + 0.5 - frame["centre"]) * per_px                    # metres along the picture's right
    zs = ground_z(project) + (f["ground_y"] - np.arange(h) - 0.5) * per_px    # Blender Z
    col = np.floor(width / 2 + (xs * scale - right @ eye) * px_per_unit).astype(int)
    row = np.floor(height / 2 - (zs * scale - up @ eye) * px_per_unit).astype(int)
    inside = (row[:, None] >= 0) & (row[:, None] < height) & (col[None] >= 0) & (col[None] < width)
    out = pixels[np.clip(row, 0, height - 1)][:, np.clip(col, 0, width - 1)]
    out[~inside] = pixels[0, 0]
    return Image.fromarray(out)


def ruled_pair(project, view: str, render_path: str | Path) -> Image.Image | None:
    """The ruled reference and the build's render ruled in the same frame (`in_frame`), side by side and labelled."""
    picture = in_frame(project, view, render_path)
    if picture is None:
        return None
    left, right = ruled(project, view), ruled(project, view, picture)
    label, gap = 18, 12
    out = Image.new("RGB", (left.width + gap + right.width, left.height + label), (255, 255, 255))
    out.paste(left, (0, label))
    out.paste(right, (left.width + gap, label))
    d = ImageDraw.Draw(out)
    d.text((4, 3), f"REFERENCE {view}, Blender metres", fill=(0, 0, 0))
    d.text((left.width + gap + 4, 3), f"YOUR BUILD {view}, the same lines", fill=(0, 0, 0))
    return out


def write_ruled_pairs(project, renders: dict[str, str], out_dir: str | Path) -> dict[str, str]:
    """ruled_<view>.png for every rendered view `ruled_pair` can draw, by view name."""
    out_dir, written = Path(out_dir), {}
    for view, path in renders.items():
        if view in project.references:
            picture = ruled_pair(project, view, path)
            if picture is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"ruled_{view}.png"
                picture.save(target)
                written[view] = str(target)
    return written


def write_ruled(project, out_dir: str | Path) -> dict[str, str]:
    """<view>_ruled.png for every reference view `ruled` can read, by view name."""
    out_dir = Path(out_dir)
    written = {}
    for view in project.references:
        picture = ruled(project, view)
        if picture is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{view}{RULED_SUFFIX}.png"
            picture.save(path)
            written[view] = str(path)
    return written


def probe_thousandths(project, view: str, x: float, y: float) -> tuple[list[float], str]:
    """A pixel of a reference view in the brief's outline units for a project that states no size: thousandths of
    the subject's extent along the outline's unit axis, from the centre of its bounding box, Blender axes. A builder
    of such a project was told `--probe` needs model.height_m and placed its first attempt by eye, 15% off."""
    refs = project.references
    if view not in refs:
        raise ValueError(f"--probe {view}: no reference view of that name (there are {', '.join(refs)})")
    reading = outline.measure(refs, project.spec)
    shot = next((s for s in render.shots_for(project.spec) if s["name"] == view), None)
    if not reading or shot is None:
        raise ValueError(f"--probe {view}: the project states no size and no outline can be read through its cameras")
    _, (r, u, _), _ = render.camera(shot)
    up_axis = int(np.argmax(np.abs(u)))                 # glTF; Blender's Z is glTF's Y
    f = facts(refs[view])
    if f["box"] is None:
        raise ValueError(f"--probe {view}: no subject found in the picture")
    blender = {0: "X", 1: "Z", 2: "Y"}
    extent = reading["extents"][blender[up_axis]]
    per_px = 1000.0 * extent / f["height_px"]
    across = (float(x) - (f["box"][0] + f["box"][2]) / 2) * per_px
    upward = ((f["box"][1] + f["box"][3]) / 2 - float(y)) * per_px
    right = np.array([r[0], -r[2], r[1]])
    up = np.array([u[0], -u[2], u[1]])
    p = right * across + up * upward
    return [round(float(v), 0) + 0.0 for v in p], reading["unit_axis"]
