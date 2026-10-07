"""The project's palette as colour zones, and a palette read off the reference views when the project has none.

A zone is (letter, name, (r, g, b)). The letter is what a zone map prints for a cell (`profile.zone_map`), and the
RGB is what `resemblance` snaps every cell to before it asks whether the candidate and the reference agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import silhouette

Zone = tuple[str, str, tuple[int, int, int]]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def hex_rgb(value: str) -> tuple[int, int, int]:
    s = str(value)
    if not (s.startswith("#") and len(s) == 7):
        raise ValueError(f"colour {value!r} must be #rrggbb")
    try:
        return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    except ValueError:
        raise ValueError(f"colour {value!r} must be #rrggbb") from None


def zones(palette: dict[str, str] | None) -> tuple[Zone, ...]:
    """The palette as zones, each with a letter of its own: the first free letter of the name, else the first free
    letter of the alphabet."""
    out: list[Zone] = []
    used: set[str] = set()
    for name, value in (palette or {}).items():
        candidates = [c for c in str(name).upper() if c in LETTERS] + list(LETTERS)
        letter = next((c for c in candidates if c not in used), None)
        if letter is None:
            raise ValueError("a palette holds at most 26 colours")
        used.add(letter)
        out.append((letter, str(name), hex_rgb(value)))
    return tuple(out)


def zone_rgb(zs: tuple[Zone, ...]) -> np.ndarray:
    return np.array([z[2] for z in zs], dtype=float)


REFINE_MIN_CELLS = 8        # a colour read off fewer cells than this keeps the value it had


def refine(references: dict[str, str | Path], palette: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
    """(the palette with each colour moved to what the reference shows where it reads as that colour, the cells each
    was read off). `suggest` cuts every body pixel, shadow included, so its colours are duller than the reference's
    paint, and a build that stores them exactly is duller too. Here each view is read as the zone measure reads it
    (`resemblance.cell_colours`: the lit side of each cell), each cell goes to its nearest colour, and a colour
    becomes the median of its cells. A colour with fewer than REFINE_MIN_CELLS cells keeps its value: an eye or a
    buckle is too few cells to read a paint from. One pass, deterministic. It changes what the zones are measured
    against, so it is done before attempts are compared, not between them."""
    from . import resemblance                   # resemblance imports this module
    zs = zones(palette)
    if not zs:
        return {}, {}
    rgb = zone_rgb(zs)
    found: list[list[np.ndarray]] = [[] for _ in zs]
    for path in references.values():
        cells = resemblance.cell_colours(*resemblance.normalised(path))
        lit = cells[~np.isnan(cells[..., 0])]
        if not len(lit):
            continue
        nearest = ((lit[:, None, :] - rgb[None]) ** 2).sum(-1).argmin(1)
        for k in range(len(zs)):
            found[k].append(lit[nearest == k])
    out, counts = {}, {}
    for (_letter, name, _rgb), value, cells in zip(zs, palette.values(), found):
        cells = np.concatenate(cells) if cells else np.zeros((0, 3))
        counts[name] = len(cells)
        if len(cells) >= REFINE_MIN_CELLS:
            r, g, b = (int(round(float(v))) for v in np.median(cells, axis=0))
            value = f"#{r:02x}{g:02x}{b:02x}"
        out[name] = value
    return out, counts


def suggest(references: dict[str, str | Path], colors: int = 5) -> dict[str, str]:
    """A palette read off the reference views: the body pixels of every view (inside the silhouette), median-cut to
    `colors` colours, largest share first, named c1..cN. Deterministic. A starting point to rename and prune, not
    a measurement: shading on a painted sheet splits one paint into two entries."""
    body = []
    for path in references.values():
        im = silhouette._load(path)
        m = silhouette.mask(im)
        if m.any():
            body.append(np.asarray(im)[m])
    if not body:
        return {}
    pixels = np.concatenate(body)
    strip = Image.fromarray(pixels.reshape(1, -1, 3).astype(np.uint8), "RGB")
    q = strip.quantize(colors=max(1, colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    table = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)
    out = {}
    for i, (_n, index) in enumerate(counts):
        r, g, b = table[index * 3: index * 3 + 3]
        out[f"c{i + 1}"] = f"#{r:02x}{g:02x}{b:02x}"
    return out
