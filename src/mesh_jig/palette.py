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


SUGGEST_CUT = 64            # colours the body pixels are median-cut to before they are merged down


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0-255 (..., 3) to CIELAB (D65), where a distance reads about as far apart as the eye sees it."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def suggest(references: dict[str, str | Path], colors: int = 5) -> dict[str, str]:
    """A palette read off the reference views: the body pixels of every view (inside the silhouette), median-cut to
    SUGGEST_CUT colours, then the two whose merging adds least to the spread (Ward's cost, in CIELAB) merged until
    `colors` are left, largest share first, named c1..cN. Deterministic. A starting point to rename and prune, not a
    measurement. A median cut straight to `colors` splits the largest paint into its shades and loses a second hue
    with a tenth of the pixels (an owl's slate-blue wing tips and glasses: none of seven colours within 28 CIELAB
    units of it, where this finds one within 3). An eye or a beak of under a percent of the pixels is still merged
    into its neighbour: sample those off the sheet."""
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
    q = strip.quantize(colors=max(SUGGEST_CUT, colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    table = q.getpalette()
    found = sorted(q.getcolors(), key=lambda c: (-c[0], c[1]))
    rgb = np.array([table[i * 3: i * 3 + 3] for _n, i in found], dtype=float)
    groups = [[float(n), rgb[k] * n, lab(rgb[k])] for k, (n, _i) in enumerate(found)]   # count, rgb sum, lab mean
    while len(groups) > max(1, colors):
        n = np.array([g[0] for g in groups])
        centres = np.array([g[2] for g in groups])
        cost = (n[:, None] * n[None] / (n[:, None] + n[None])) * ((centres[:, None] - centres[None]) ** 2).sum(-1)
        cost[np.tril_indices(len(groups))] = np.inf
        a, b = np.unravel_index(int(np.argmin(cost)), cost.shape)
        (na, sa, la), (nb, sb, lb) = groups[a], groups[b]
        groups[a] = [na + nb, sa + sb, (la * na + lb * nb) / (na + nb)]
        del groups[b]
    groups.sort(key=lambda g: -g[0])
    out = {}
    for i, (count, total, _lab) in enumerate(groups):
        r, g, b = (int(round(float(v))) for v in total / count)
        out[f"c{i + 1}"] = f"#{r:02x}{g:02x}{b:02x}"
    return out
