"""Paint a texture atlas from a JSON plan: the deterministic way a model gets an image into a build.

The agent writes the PLAN (text) and mesh-jig paints it, so a model script that may not open files still gets a
painted surface. Pillow only; no randomness outside the seeded `speckle` op, so the same plan is the same PNG on
every machine.

The plan:

    {"size": 1024,                       # square atlas, pixels (default 1024)
     "background": "white",              # a palette name or "#rrggbb"
     "palette": {"crimson": "#b53c2e"},  # optional additions to the project's palette
     "ops": [                            # painted in order; coordinates are 0..1 of the atlas (u right, v DOWN)
       {"op": "rect",    "box": [u0, v0, u1, v1], "color": "crimson"},
       {"op": "line",    "points": [[u, v], ...], "color": "#2c3339", "width": 0.004},
       {"op": "polygon", "points": [[u, v], ...], "color": "steel"},
       {"op": "ellipse", "box": [u0, v0, u1, v1], "color": "cyan"},
       {"op": "speckle", "box": [u0, v0, u1, v1], "color": "gunmetal", "density": 0.02, "size": 0.003, "seed": 7},
       {"op": "stripes", "box": [u0, v0, u1, v1], "color": "gunmetal", "count": 6, "width": 0.003, "axis": "u"}
     ]}

Widths and sizes are fractions of the atlas side too, so a plan reads the same at 512 and 2048. Every op may carry
"alpha" 0..1 to blend over what is under it (default 1.0). Unknown ops, missing fields and colours the palette does
not know are errors: a plan that the painter has to guess at is a plan the author got wrong.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

BASE_PALETTE = {"black": "#000000", "white": "#ffffff"}
OPS = ("rect", "line", "polygon", "ellipse", "speckle", "stripes")
MAX_SIZE = 4096


def _rgb(spec: str, palette: dict[str, str]) -> tuple[int, int, int]:
    s = palette.get(str(spec), str(spec))
    if not (s.startswith("#") and len(s) == 7):
        raise ValueError(f"unknown colour {spec!r} (palette {sorted(palette)} or #rrggbb)")
    return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))


def _box(op: dict, n: int) -> tuple[int, int, int, int]:
    b = op.get("box")
    if not (isinstance(b, list) and len(b) == 4):
        raise ValueError(f"{op.get('op')}: box must be [u0, v0, u1, v1]")
    u0, v0, u1, v1 = (float(x) for x in b)
    if not (0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1):
        raise ValueError(f"{op.get('op')}: box {b} must be inside 0..1 with u0<u1 and v0<v1")
    return round(u0 * n), round(v0 * n), max(round(u1 * n) - 1, round(u0 * n)), max(round(v1 * n) - 1, round(v0 * n))


def _points(op: dict, n: int, least: int) -> list[tuple[float, float]]:
    pts = op.get("points")
    if not (isinstance(pts, list) and len(pts) >= least and all(isinstance(p, list) and len(p) == 2 for p in pts)):
        raise ValueError(f"{op.get('op')}: points must be at least {least} [u, v] pairs")
    return [(float(u) * n, float(v) * n) for u, v in pts]


def _px(op: dict, key: str, n: int, default: float) -> int:
    return max(1, round(float(op.get(key, default)) * n))


def _layer(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def paint_op(base: Image.Image, op: dict, palette: dict[str, str]) -> None:
    n = base.width
    kind = op.get("op")
    if kind not in OPS:
        raise ValueError(f"unknown op {kind!r} (one of {OPS})")
    colour = _rgb(op.get("color", ""), palette)
    alpha = float(op.get("alpha", 1.0))
    if not 0 < alpha <= 1:
        raise ValueError(f"{kind}: alpha must be in (0, 1]")
    layer, d = _layer(n)
    fill = colour + (255,)
    if kind == "rect":
        d.rectangle(_box(op, n), fill=fill)
    elif kind == "ellipse":
        d.ellipse(_box(op, n), fill=fill)
    elif kind == "line":
        d.line(_points(op, n, 2), fill=fill, width=_px(op, "width", n, 0.004), joint="curve")
    elif kind == "polygon":
        d.polygon(_points(op, n, 3), fill=fill)
    elif kind == "speckle":
        x0, y0, x1, y1 = _box(op, n)
        rng = random.Random(int(op.get("seed", 0)))
        density = float(op.get("density", 0.02))
        r = _px(op, "size", n, 0.003)
        count = int(density * (x1 - x0 + 1) * (y1 - y0 + 1) / max(1, r * r))
        for _ in range(count):
            x, y = rng.randint(x0, x1), rng.randint(y0, y1)
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    elif kind == "stripes":
        x0, y0, x1, y1 = _box(op, n)
        count = int(op.get("count", 4))
        w = _px(op, "width", n, 0.003)
        if count < 1:
            raise ValueError("stripes: count must be >= 1")
        along_u = op.get("axis", "u") == "u"
        span = (x1 - x0) if along_u else (y1 - y0)
        for i in range(count):
            t = (i + 0.5) / count * span
            if along_u:
                d.rectangle([x0 + t - w / 2, y0, x0 + t + w / 2, y1], fill=fill)
            else:
                d.rectangle([x0, y0 + t - w / 2, x1, y0 + t + w / 2], fill=fill)
    if alpha < 1:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer.putalpha(a)
    base.alpha_composite(layer)


def paint(plan: dict, palette: dict[str, str] | None = None) -> Image.Image:
    """The plan as an RGB image. `palette` is the project's (jig.json); the plan's own entries are added over it."""
    size = int(plan.get("size", 1024))
    if not 16 <= size <= MAX_SIZE:
        raise ValueError(f"size must be 16..{MAX_SIZE}")
    colours = dict(BASE_PALETTE)
    for source in (palette or {}, plan.get("palette") or {}):
        for k, v in source.items():
            if not (isinstance(v, str) and v.startswith("#") and len(v) == 7):
                raise ValueError(f"palette {k!r} must be #rrggbb")
            colours[str(k)] = v
    im = Image.new("RGBA", (size, size), _rgb(plan.get("background", "white"), colours) + (255,))
    ops = plan.get("ops")
    if not isinstance(ops, list):
        raise ValueError("plan needs an ops list")
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ValueError(f"ops[{i}] is not an object")
        try:
            paint_op(im, op, colours)
        except ValueError as e:
            raise ValueError(f"ops[{i}]: {e}") from None
    return im.convert("RGB")


def paint_file(plan_path: str | Path, out_png: str | Path, palette: dict[str, str] | None = None) -> str:
    """Paint the plan at plan_path to out_png. "" when painted, else the error text (nothing is written)."""
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        im = paint(plan, palette)
    except (ValueError, OSError) as e:
        return f"PAINT_ERROR: {e}"
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return ""
