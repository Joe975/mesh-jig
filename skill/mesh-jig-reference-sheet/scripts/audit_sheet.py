"""Read-only coarse projection-scale audit; independent of mesh-jig model scoring."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def audit(image: Image.Image, config: dict) -> dict:
    mode = config.get("mode", "chroma")
    if mode not in ("chroma", "alpha"):
        raise ValueError("mode must be chroma or alpha")
    tolerance = float(config.get("tolerance_pct", 3))
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 100:
        raise ValueError("tolerance_pct must be finite and between 0 and 100")
    thresholds = {"min_chroma": 35, "max_min_channel": 210, "min_alpha": 127}
    for key, default in thresholds.items():
        value = config.get(key, default)
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError(f"{key} must be an integer from 0 to 255")
        thresholds[key] = value
    panels = config.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ValueError("panels must be a nonempty mapping of names to rectangles")
    rgba = np.asarray(image.convert("RGBA"), dtype=np.int16)
    if mode == "alpha" and np.all(rgba[:, :, 3] == 255):
        raise ValueError("alpha mode needs a transparent background, not an opaque sheet")
    views = {}
    for name, box in panels.items():
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError(f"{name}: panel must be four integer coordinates")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError(f"{name}: panel is outside the sheet or has no area")
        crop = rgba[y0:y1, x0:x1]
        mask = crop[:, :, 3] > thresholds["min_alpha"]
        if mode == "chroma":
            rgb = crop[:, :, :3]
            mask &= (rgb.max(2) - rgb.min(2) > thresholds["min_chroma"])
            mask &= rgb.min(2) < thresholds["max_min_channel"]
        y, x = np.where(mask)
        if not len(x):
            raise ValueError(f"{name}: no foreground under the configured mask")
        xmin, xmax, ymin, ymax = int(x.min()), int(x.max()), int(y.min()), int(y.max())
        views[name] = {
            "panel": box,
            "subject_bbox": [x0 + xmin, y0 + ymin, x0 + xmax + 1, y0 + ymax + 1],
            "width": xmax - xmin + 1,
            "height": ymax - ymin + 1,
            "touches_panel_edge": xmin == 0 or ymin == 0 or xmax == x1-x0-1 or ymax == y1-y0-1,
        }
    comparisons = []
    for check in config.get("comparisons", []):
        dimensions = check.get("dimensions")
        if dimensions is None:
            dimensions = [[name, check.get("axis")] for name in check.get("views", [])]
        if not isinstance(dimensions, list) or len(dimensions) < 2:
            raise ValueError("each comparison needs at least two view/dimension pairs")
        values = []
        for pair in dimensions:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("comparison dimensions must be [view, width-or-height] pairs")
            name, axis = pair
            if name not in views or axis not in ("width", "height"):
                raise ValueError(f"unknown view or dimension in comparison: {pair}")
            values.append(views[name][axis])
        spread = 100 * (max(values) - min(values)) / max(values)
        comparisons.append({"name": check.get("name", ""), "dimensions": dimensions,
                            "values": values, "spread_pct": round(spread, 6),
                            "within_tolerance": spread <= tolerance})
    return {"size": [image.width, image.height], "mode": mode, "thresholds": thresholds,
            "tolerance_pct": tolerance, "views": views, "comparisons": comparisons,
            "within_tolerance": all(c["within_tolerance"] for c in comparisons) if comparisons else None,
            "scope": "Coarse mask bounds only; does not verify anatomy, camera calibration or shared 3D geometry."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.resolve() in (args.sheet.resolve(), args.config.resolve()):
            raise ValueError("output must not overwrite the input sheet or config")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        with Image.open(args.sheet) as image:
            result = audit(image, config)
        result["sheet"] = str(args.sheet.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.error(str(error))
    print(f"Coarse dimension audit: {result['within_tolerance']}; output: {args.out.resolve()}")
    return 1 if result["within_tolerance"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
