"""Resemblance: an offline, drift-free scalar for how much a render looks like its reference view.

A vision judge's absolute score drifts between batches by more than one edit is worth, so a loop that asks the judge
about every step is selecting on batch luck. `silhouette` and `profile` measure the OUTLINE without a model call;
this measures what the outline misses and what a judge actually pays for: colour zones and markings that land where
the reference's do. The inner loop climbs on these numbers and calls the pairwise judge only to promote.

Two terms per view, both computed on the subject cropped to its bounding box and scaled onto one canvas (the same
normalisation `silhouette` uses, so a slender body against a chunky reference costs here too):

- `zone`: the share of cells, inside both silhouettes, whose nearest palette colour agrees. This is placement of
  colour. It needs the project's palette; without one the term is left out.
- `structure`: per cell, a histogram of edge orientations weighted by edge strength (a HOG cell), compared by
  cosine similarity; cells with no edges on either side are skipped, a cell with edges on one side only scores 0.
  This is placement of markings: detail slid across the surface moves every edge, and the score drops with it.

`score` is their mean. It is a ranking number for an inner loop; it is not a judge and never promotes anything on
its own.

A third term is reported beside them, and counts in a project's score only where the project asks for it
(`criteria.score`):

- `colour`: how close each cell's colour is to the reference's in the same place, 1 for the same colour and 0 from
  `COLOUR_REACH` apart in Oklab, over the cells inside both silhouettes. `zone` asks only which palette colour is
  nearest, so a build painted the palette exactly is told nothing about a palette that is off, and a paint that
  reads as its neighbour under the light is a cell wrong or not at all. This needs no palette.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from . import silhouette
from .palette import Zone, zone_rgb

GRID = 320          # canvas side: twice silhouette.GRID. At 160 fine detail collapses into flat; at 320 the structure
                    # term separated a flat repaint (0.78) from the reference's exact paint (0.93) on the calibration set
CELL = 8            # cell size on the canvas: 40 x 40 cells
BINS = 8            # edge orientation bins over 0..pi
EDGE_EPS = 4.0      # summed gradient magnitude under which a cell counts as flat
COLOUR_REACH = 0.25  # the Oklab distance from which two colours have nothing in common: the cat's orange is 0.22
                     # from its brown and 0.19 from its cream, and about 0.02 is just visible
ORTHO_VIEWS = silhouette.ORTHO_VIEWS
REPORT = "resemblance.json"


def normalised(img: str | Path | Image.Image, grid: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """(rgb float array grid x grid x 3, mask grid x grid): the subject cropped to its bounding box, scaled so the
    longer side is `grid`, centred; pixels outside the mask are 0."""
    im = silhouette._load(img)
    m = silhouette.mask(img)            # by path when it is one: the mask of this file may already be known
    b = silhouette.bbox(m)
    rgb = np.zeros((grid, grid, 3), dtype=float)
    out_m = np.zeros((grid, grid), dtype=bool)
    if b is None:
        return rgb, out_m
    x0, y0, x1, y1 = b
    crop = im.crop((x0, y0, x1, y1))
    h, w = y1 - y0, x1 - x0
    k = grid / max(h, w)
    nw, nh = min(grid, max(1, round(w * k))), min(grid, max(1, round(h * k)))
    small = np.asarray(crop.resize((nw, nh), Image.BILINEAR), dtype=float)
    mask_small = np.asarray(Image.fromarray(np.where(m[y0:y1, x0:x1], 255, 0).astype(np.uint8), "L")
                            .resize((nw, nh), Image.NEAREST)) > 127
    ox, oy = (grid - nw) // 2, (grid - nh) // 2
    rgb[oy:oy + nh, ox:ox + nw] = small * mask_small[..., None]
    out_m[oy:oy + nh, ox:ox + nw] = mask_small
    return rgb, out_m


def _cells(a: np.ndarray, cell: int) -> np.ndarray:
    """(rows, cols, cell, cell, ...) view of a canvas whose side is a multiple of `cell`."""
    n = a.shape[0] // cell
    return a[:n * cell, :n * cell].reshape(n, cell, n, cell, *a.shape[2:]).swapaxes(1, 2)


def zone_letters(rgb: np.ndarray, mask: np.ndarray, zones: tuple[Zone, ...], cell: int = CELL,
                 min_fill: float = 0.35) -> np.ndarray:
    """Each cell's nearest palette index from the lit side of its pixels (the 75th percentile, as profile.zone_map
    does: painted shadow drags a pale colour into a dark one), -1 where the cell is mostly outside the silhouette."""
    return letters(cell_colours(rgb, mask, cell, min_fill), zones)


def letters(lit: np.ndarray, zones: tuple[Zone, ...]) -> np.ndarray:
    """`cell_colours` snapped to the palette: each cell's nearest palette index, -1 where it has no colour."""
    palette = zone_rgb(zones)
    n = lit.shape[0]
    out = np.full((n, n), -1, dtype=int)
    for r in range(n):
        for c in range(n):
            if not np.isnan(lit[r, c, 0]):
                out[r, c] = int(np.argmin(((palette - lit[r, c]) ** 2).sum(axis=1)))
    return out


def cell_colours(rgb: np.ndarray, mask: np.ndarray, cell: int = CELL, min_fill: float = 0.35) -> np.ndarray:
    """(rows, cols, 3): the colour each cell is read as, the lit side of its pixels (their 75th percentile); nan where
    the cell is mostly outside the silhouette. `zone_letters` snaps it to the palette, `palette.refine` reads the
    reference's own paints from it."""
    pc, mc = _cells(rgb, cell), _cells(mask, cell)
    n = pc.shape[0]
    out = np.full((n, n, 3), np.nan)
    for r in range(n):
        for c in range(n):
            m = mc[r, c]
            if m.mean() >= min_fill:
                out[r, c] = np.percentile(pc[r, c][m], 75, axis=0)
    return out


def oklab(rgb255: np.ndarray) -> np.ndarray:
    """sRGB 0..255 as Oklab (Ottosson 2020): equal distances there look about equally different."""
    c = np.asarray(rgb255, dtype=float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l = 0.4122214708 * c[..., 0] + 0.5363325363 * c[..., 1] + 0.0514459929 * c[..., 2]
    m = 0.2119034982 * c[..., 0] + 0.6806995451 * c[..., 1] + 0.1073969566 * c[..., 2]
    s = 0.0883024619 * c[..., 0] + 0.2817188376 * c[..., 1] + 0.6299787005 * c[..., 2]
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], axis=-1)


def colour_closeness(a: np.ndarray, b: np.ndarray, reach: float = COLOUR_REACH) -> float | None:
    """How close two fields of `cell_colours` are over the cells both have: each cell 1 for the same colour, falling
    to 0 at `reach` apart in Oklab, and their mean. None when they share no cell."""
    both = ~np.isnan(a[..., 0]) & ~np.isnan(b[..., 0])
    if not both.any():
        return None
    d = np.linalg.norm(oklab(a[both]) - oklab(b[both]), axis=-1)
    return round(float(np.clip(1.0 - d / reach, 0.0, 1.0).mean()), 4)


def zone_agreement(a: np.ndarray, b: np.ndarray) -> float | None:
    """Share of cells inside both silhouettes whose palette letter agrees; None when they share no cell."""
    both = (a >= 0) & (b >= 0)
    if not both.any():
        return None
    return round(float((a[both] == b[both]).mean()), 4)


def edge_histograms(rgb: np.ndarray, mask: np.ndarray, cell: int = CELL, bins: int = BINS) -> np.ndarray:
    """Per cell, an orientation histogram (0..pi, `bins` bins) of the greyscale gradient inside the mask, weighted by
    gradient magnitude. The silhouette's own edge is excluded (mask eroded by one pixel) so the outline, which
    `silhouette` already scores, does not dominate."""
    grey = rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(grey)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    mag = np.where(inner, mag, 0.0)
    b = np.minimum((ang / np.pi * bins).astype(int), bins - 1)
    mc, bc = _cells(mag, cell), _cells(b, cell)
    n = mc.shape[0]
    hist = np.zeros((n, n, bins), dtype=float)
    for k in range(bins):
        hist[..., k] = np.where(bc == k, mc, 0.0).sum(axis=(2, 3))
    return hist


def structure_similarity(ha: np.ndarray, hb: np.ndarray, eps: float = EDGE_EPS) -> float | None:
    """Mean cosine similarity of the two histogram fields over cells with edges on either side; a cell with edges on
    one side only scores 0. None when neither has any edge."""
    ea, eb = ha.sum(-1), hb.sum(-1)
    live = (ea > eps) | (eb > eps)
    if not live.any():
        return None
    dot = (ha * hb).sum(-1)
    norm = np.sqrt((ha * ha).sum(-1) * (hb * hb).sum(-1))
    cos = np.where(norm > 0, dot / np.maximum(norm, 1e-9), 0.0)
    return round(float(cos[live].mean()), 4)


def compare(render: str | Path | Image.Image, reference: str | Path | Image.Image, zones: tuple[Zone, ...] = (),
            cell: int = CELL, grid: int = GRID) -> dict:
    ra, ma = normalised(render, grid)
    rb, mb = normalised(reference, grid)
    la, lb = cell_colours(ra, ma, cell), cell_colours(rb, mb, cell)
    zone = zone_agreement(letters(la, zones), letters(lb, zones)) if zones else None
    structure = structure_similarity(edge_histograms(ra, ma, cell), edge_histograms(rb, mb, cell))
    parts = [v for v in (zone, structure) if v is not None]
    return {"zone": zone, "structure": structure, "colour": colour_closeness(la, lb),
            "score": round(statistics.fmean(parts), 4) if parts else None}


def measure_views(renders: dict[str, str | Path], references: dict[str, str | Path], zones: tuple[Zone, ...] = (),
                  out_dir: Path | None = None, cell: int = CELL, grid: int = GRID,
                  *, weights: dict[str, float] | None = None) -> dict:
    """view -> compare() for every render with a reference of the same name; `ortho` is the mean score over the
    orthographic views present and `score` the mean over every view. Writes resemblance.json into out_dir."""
    views = {name: compare(renders[name], references[name], zones, cell, grid)
             for name in sorted(renders) if name in references}
    scored = {v: d["score"] for v, d in views.items() if d["score"] is not None}
    ortho = [scored[v] for v in ORTHO_VIEWS if v in scored]
    rec = {"views": views, "score": round(statistics.fmean(scored.values()), 4) if scored else None,
           "ortho": round(statistics.fmean(ortho), 4) if ortho else None}
    if weights is not None:
        from .scoring import aggregate, weak_views
        rec.update(score=aggregate(views, "score", weights), ortho=aggregate(views, "score", weights),
                   zone=aggregate(views, "zone", weights),
                   weights=weights, weak_views=weak_views(views, "score", weights))
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / REPORT).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def feedback_lines(rec: dict) -> list[str]:
    lines = []
    for v, d in (rec.get("views") or {}).items():
        parts = []
        if d.get("zone") is not None:
            parts.append(f"colour zones agree {d['zone']:.0%}")
        if d.get("colour") is not None:
            parts.append(f"colours {d['colour']:.2f} close")
        if d.get("structure") is not None:
            parts.append(f"markings/edges in place {d['structure']:.2f}")
        if parts:
            lines.append(f"{v}: " + ", ".join(parts))
    return lines
