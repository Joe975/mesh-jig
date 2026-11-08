"""Start a project, and cut its reference views out of one design sheet.

A design sheet (a turnaround with labelled views) already holds every view; the crop boxes are written by hand, or by
an agent looking at the sheet, into the `sheet` block of jig.json:

    "sheet": {"source": "sheet.png", "mask_sample_x": 8,
              "crops": {"hero": {"box": [x0, y0, x1, y1], "mask": [[x0, y0, x1, y1], ...]}, "top": {...}, ...}}

Each view is cropped to refs/<view>.png. `mask` rectangles are painted first with the sheet's own background (the
row's colour at x = `mask_sample_x`), so title text and view labels never reach a measure or a judge. Views are named
after the cameras (hero, top, side, rear) so renders pair with them by name.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from .project import JIG_FILE, Project

BRIEF_STUB = """\
# {name}

What the model is, in a sentence, then its parts as a list: one bullet per part a viewer would name, each saying
where it sits and what colour it is. The judge reads this beside the pictures, so write what must be there, not how
to build it.
"""


def cut(image: Image.Image, box: list[int], masks: list[list[int]] = (), sample_x: int = 8) -> Image.Image:
    """Crop `box` after painting each mask rectangle row by row with the pixel at x=sample_x on that row."""
    im = image.convert("RGB").copy()
    px = im.load()
    w, h = im.size
    for x0, y0, x1, y1 in masks:
        for y in range(max(0, y0), min(h, y1)):
            c = px[min(sample_x, w - 1), y]
            for x in range(max(0, x0), min(w, x1)):
                px[x, y] = c
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"crop box {box} is outside the {w}x{h} sheet")
    return im.crop((x0, y0, x1, y1))


def write_refs(project: Project) -> list[Path]:
    """Cut every view named in the project's `sheet` block into its refs directory. The files written."""
    sheet = project.data.get("sheet")
    if not isinstance(sheet, dict) or not sheet.get("source") or not sheet.get("crops"):
        raise ValueError(f"{JIG_FILE} has no `sheet` block with a `source` and `crops`")
    source = project.root / sheet["source"]
    if not source.is_file():
        raise ValueError(f"sheet image {source} does not exist")
    image = Image.open(source)
    project.refs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for view, spec in sheet["crops"].items():
        if not isinstance(spec, dict) or "box" not in spec:
            raise ValueError(f"sheet.crops.{view} needs a `box` [x0, y0, x1, y1]")
        try:
            crop = cut(image, spec["box"], spec.get("mask", []), int(sheet.get("mask_sample_x", 8)))
        except ValueError as e:
            raise ValueError(f"sheet.crops.{view}: {e}") from None
        p = project.refs_dir / f"{view}.png"
        crop.save(p)
        written.append(p)
    return written


STANDING_VIEWS = ("front", "side", "rear", "right", "hero", "top")
# the pirate cat's cameras, in stage units (stage_scale makes the subject one unit tall): the front looks at the
# model's +Y face, the side at its anatomical left (-X), the top down with the front toward the picture's bottom
STANDING_SHOTS = [
    {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.12},
    {"name": "side", "cam_pos": [-3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.12},
    {"name": "rear", "cam_pos": [0, 0, 3], "cam_target": [0, 0, 0], "ortho": 1.12},
    {"name": "right", "cam_pos": [3, 0, 0], "cam_target": [0, 0, 0], "ortho": 1.12},
    {"name": "hero", "cam_pos": [-2.4, 1.35, -3.7], "cam_target": [0, 0, 0], "ortho": 1.12},
    {"name": "top", "cam_pos": [0, 3, 0], "cam_up": [0, 0, 1], "cam_target": [0, 0, 0], "ortho": 1.12},
]


# The colour contract a new project starts with: a texture the script paints, 1024 px (the Quest atlas). It took
# per-face vertex colour's triangle-edge sawtooth off every boundary, and unlike an atlas plan it asks the script
# for no UVs, which weaker builders got wrong (docs/run-notes/overnight-horror-tex-20261108.md).
PAINTED = {"textured": True, "texture_from": "script", "texture_size": 1024}


def template(name: str, length_m: float, sheet_file: str = "", sheet_size: tuple[int, int] | None = None,
             height_m: float | None = None) -> dict:
    """A jig.json to start from: a contract whose script paints its texture (PAINTED), for a craft `length_m`
    long, nose along -Z, scored on its top, side and rear views, under the lights that turn with each camera. The
    views and the light are stated in `criteria`, so whoever changes the subject sees them to change.

    With `height_m`, a standing subject's instead (a character, a mech, a figure): held to that height, centred at
    mid-height, its objects kept apart, the six cameras of STANDING_SHOTS scaled to it, scored on the five
    orthographic views, and the outline's ground at its feet; `mesh-jig refs` then prints the calibration to add."""
    if height_m is not None:
        if not height_m > 0:
            raise ValueError("--height must be a positive number of metres")
        data = {
            "name": name, "brief": "brief.md", "refs": "refs", "palette": {},
            "model": {"script": "model.py", "glb": f"{name}.glb", "join": False, "height_m": height_m,
                      "height_tolerance": 0.05, "stage_scale": round(1 / height_m, 6), "max_triangles": 10000,
                      "min_triangles": 100, "single_mesh": False, **PAINTED,
                      "shots": [dict(s) for s in STANDING_SHOTS]},
            "criteria": {"views": {v: 1 for v in STANDING_VIEWS if v != "hero"}, "light": "camera"},
        }
        if sheet_file:
            w, h = sheet_size or (0, 0)
            data["sheet"] = {"source": sheet_file, "mask_sample_x": 8,
                             "crops": {v: {"box": [0, 0, w, h]} for v in STANDING_VIEWS}}
        return data
    data: dict = {
        "name": name,
        "brief": "brief.md",
        "refs": "refs",
        "palette": {},
        "model": {"script": "model.py", "glb": f"{name}.glb", "join": True, "length_m": length_m,
                  "length_tolerance": 0.05, "forward": "-Z", "max_triangles": 1500, "min_triangles": 100,
                  "single_mesh": True, **PAINTED},
        "criteria": {"views": {"top": 1, "side": 1, "rear": 1}, "light": "camera"},
    }
    if sheet_file:
        w, h = sheet_size or (0, 0)
        data["sheet"] = {"source": sheet_file, "mask_sample_x": 8,
                         "crops": {v: {"box": [0, 0, w, h]} for v in ("hero", "top", "side", "rear")}}
    return data


def init(root: str | Path, *, name: str = "", length_m: float = 1.0, sheet: str | Path | None = None,
         height_m: float | None = None) -> list[Path]:
    """Write jig.json, brief.md and refs/ into `root` (created). Refuses to overwrite a jig.json. With `sheet`, the
    image is copied in and a `sheet` block with one full-frame box per view is written for you to tighten."""
    root = Path(root)
    jig = root / JIG_FILE
    if jig.exists():
        raise ValueError(f"{jig} already exists")
    root.mkdir(parents=True, exist_ok=True)
    name = name or root.resolve().name
    sheet_file, size = "", None
    written = []
    if sheet is not None:
        src = Path(sheet)
        if not src.is_file():
            raise ValueError(f"sheet image {src} does not exist")
        sheet_file = "sheet" + src.suffix.lower()
        shutil.copyfile(src, root / sheet_file)
        size = Image.open(root / sheet_file).size
        written.append(root / sheet_file)
    jig.write_text(json.dumps(template(name, length_m, sheet_file, size, height_m), indent=2) + "\n", encoding="utf-8")
    written.append(jig)
    brief = root / "brief.md"
    if not brief.exists():
        brief.write_text(BRIEF_STUB.format(name=name), encoding="utf-8")
        written.append(brief)
    (root / "refs").mkdir(exist_ok=True)
    return written
