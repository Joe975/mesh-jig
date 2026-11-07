"""The shape carved from every axis-aligned orthographic reference outline at once, and the overlap it reaches in
each view: the most any one shape can agree with all the outlines together.

Builders said the views disagree (the top view with the side views, on where the arms go) and stopped working on
the view they blamed. A carved shape says whether that is true of the outlines: where
it reaches 0.95 in a view, the outlines agree and what is left is in the build.

Each view's outline, cut to its box as the overlap measure cuts it, is laid over the model's box along the two axes
its camera sees; the model's height sets the scale (the views that look across it), and a view that looks along it
takes its across scale from them. A voxel is kept when it falls inside every outline. Projected back along each
camera and cut and scaled as the measure does (`silhouette.normalise`), it is compared with the reference's outline.
A perspective view, or one not along an axis, is neither carved by nor measured. numpy only.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import render, silhouette

RESOLUTION = 128        # voxels along the model's longest side


def _axis(v: np.ndarray) -> tuple[int, float] | None:
    """(axis index, sign) of a unit vector along one glTF axis, else None."""
    k = int(np.argmax(np.abs(v)))
    return (k, float(np.sign(v[k]))) if abs(v[k]) > 0.999 else None


def views(spec: dict, references: dict) -> dict:
    """The views a hull can be carved from: {view: (right axis, sign, up axis, sign, look axis)} in glTF axes."""
    out = {}
    for shot in render.shots_for(spec):
        name = shot["name"]
        if name not in references or not shot.get("ortho") or name in out:
            continue
        _, (r, u, f), _ = render.camera(shot)
        ra, ua, fa = _axis(r), _axis(u), _axis(f)
        if ra and ua and fa:
            out[name] = (ra[0], ra[1], ua[0], ua[1], fa[0])
    return out


def _cut(m: np.ndarray) -> np.ndarray | None:
    b = silhouette.bbox(m)
    return None if b is None else m[b[1]:b[3], b[0]:b[2]]


def extents(cuts: dict, axes: dict) -> np.ndarray | None:
    """The model's box in glTF axes, up (Y) 1: each axis's extent from a view that sees it, at that view's scale, its
    scale from the height of a view that looks across the model."""
    ext = [None, 1.0, None]
    scale = {}
    for v, (ra, _, ua, _, _) in axes.items():
        if ua == 1:
            scale[v] = 1.0 / cuts[v].shape[0]
            ext[ra] = ext[ra] or cuts[v].shape[1] * scale[v]
    for v, (ra, _, ua, _, _) in axes.items():
        if ua != 1:
            known = ext[ra] or ext[ua]
            if known is None:
                continue
            per = known / cuts[v].shape[1 if ext[ra] else 0]
            ext[ra] = ext[ra] or cuts[v].shape[1] * per
            ext[ua] = ext[ua] or cuts[v].shape[0] * per
    return None if any(e is None for e in ext) else np.array(ext, float)


def carve(references: dict, spec: dict, resolution: int = RESOLUTION):
    """(voxels: bool [x, y, z] in glTF axes, the views it was carved from), or (None, {}) when the views cannot
    make one: fewer than two, or an axis no view sees."""
    axes = views(spec, references)
    cuts = {v: _cut(silhouette.mask(references[v])) for v in axes}
    axes = {v: a for v, a in axes.items() if cuts[v] is not None}
    if len(axes) < 2:
        return None, {}
    ext = extents(cuts, axes)
    if ext is None:
        return None, {}
    n = np.maximum(np.round(ext / ext.max() * resolution).astype(int), 2)
    hull = np.ones(tuple(n), dtype=bool)
    for v, (ra, rs, ua, us, fa) in axes.items():
        m = cuts[v]
        h, w = m.shape
        # the voxel centres along the camera's right and up, as the outline's columns and rows
        cols = ((np.arange(n[ra]) + 0.5) / n[ra] * w).astype(int)
        rows = ((np.arange(n[ua]) + 0.5) / n[ua] * h).astype(int)
        if rs < 0:
            cols = cols[::-1]
        if us > 0:                                  # the picture's top is the up axis's high end
            rows = rows[::-1]
        table = m[np.ix_(rows, cols)].T             # [right index, up index]
        lay = table if ra < ua else table.T          # in the hull's axis order
        hull &= np.expand_dims(lay, fa)
    return hull, axes


def project(hull: np.ndarray, view: tuple) -> np.ndarray:
    """The hull seen along a view's camera, as a picture (rows top down, columns left to right)."""
    ra, rs, ua, us, fa = view
    seen = hull.any(axis=fa)
    rest = [a for a in range(3) if a != fa]
    img = seen if rest.index(ra) == 0 else seen.T  # [right, up]
    if rs < 0:
        img = img[::-1]
    if us > 0:
        img = img[:, ::-1]
    return img.T


def reach(references: dict, spec: dict) -> dict[str, float]:
    """{view: the overlap the carved shape reaches with that view's reference outline}, the overlap measure's own
    cut and scale; {} when no hull can be carved."""
    hull, axes = carve(references, spec)
    if hull is None:
        return {}
    out = {}
    for v, axis in axes.items():
        seen = project(hull, axis)
        ref = silhouette.mask(references[v])
        # drawn larger at its own proportions (the measure keeps each view's aspect), then cut and scaled alike
        h, w = seen.shape
        big = np.asarray(Image.fromarray(np.where(seen, 255, 0).astype(np.uint8), "L")
                         .resize((w * 4, h * 4), Image.NEAREST)) > 127
        out[v] = silhouette.iou(silhouette.normalise(big), silhouette.normalise(ref))
    return out

