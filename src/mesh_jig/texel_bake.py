"""A texture painted by the model script: which point of the model each texel of the atlas shows, the script's
colour for it, and the gutter between islands.

The script cannot read or write an image, and unwrapping a mesh by hand is where a textured build goes wrong. So
on a `texture_from: "script"` contract the script says what colour a point of the model is,

    def texture(name, position, normal):    # the mesh object's name; (N, 3) arrays, one row per texel
        return colours                      # (N, 3), sRGB 0..1: a palette hex's channels over 255

and the runner unwraps every mesh into one atlas, asks `texture` about each texel's point and writes the image. A
colour is then a function of where a point is on the model (inside this ellipsoid, above that height, facing
forward), which is how a builder already thinks about a part.

This module is the arithmetic of that: no bpy, numpy only (Blender's own Python has it), so it is tested without
Blender. An atlas array's row 0 is v = 0, the bottom of the image, as Blender stores pixels.
"""
import numpy as np


def rasterise(uv, position, normal, size):
    """Which triangle each texel centre lies in, and the point and normal of the model there.

    uv (T, 3, 2) in 0..1; position and normal (T, 3, 3), a row per corner. -> (covered (S, S) bool, point
    (S, S, 3), normal (S, S, 3) unit length, triangle (S, S) int, -1 where nothing covers). A triangle too small to
    hold a texel centre is given the texel its centroid falls in when nothing else has it, so every face of the
    model has a texel of its own colour to be sampled from."""
    uv = np.asarray(uv, dtype=np.float64) * size
    position, normal = np.asarray(position, dtype=np.float64), np.asarray(normal, dtype=np.float64)
    covered = np.zeros((size, size), dtype=bool)
    point = np.zeros((size, size, 3))
    facing = np.zeros((size, size, 3))
    owner = np.full((size, size), -1, dtype=np.int64)
    area = ((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
            - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1]))
    missed = []
    for i in range(len(uv)):
        xs, ys, a = uv[i, :, 0], uv[i, :, 1], area[i]
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), size - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), size - 1)
        hit = None
        if abs(a) > 1e-12 and x0 <= x1 and y0 <= y1:
            px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((xs[1] - px) * (ys[2] - py) - (xs[2] - px) * (ys[1] - py)) / a
            w1 = ((xs[2] - px) * (ys[0] - py) - (xs[0] - px) * (ys[2] - py)) / a
            w2 = 1.0 - w0 - w1
            hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if hit is None or not hit.any():
            missed.append(i)
            continue
        w = np.stack([w0[hit], w1[hit], w2[hit]], axis=-1)
        rows, cols = np.nonzero(hit)
        rows, cols = rows + y0, cols + x0
        covered[rows, cols] = True
        point[rows, cols] = w @ position[i]
        facing[rows, cols] = w @ normal[i]
        owner[rows, cols] = i
    for i in missed:
        col, row = (int(np.clip(v, 0, size - 1)) for v in uv[i].mean(axis=0))
        if not covered[row, col]:
            covered[row, col] = True
            point[row, col] = position[i].mean(axis=0)
            facing[row, col] = normal[i].mean(axis=0)
            owner[row, col] = i
    length = np.linalg.norm(facing, axis=-1, keepdims=True)
    facing = np.where(length > 1e-12, facing / np.maximum(length, 1e-12), 0.0)
    return covered, point, facing, owner


def paint(paint_fn, names, part, covered, point, normal):
    """The atlas's colours, (S, S, 3) sRGB 0..1: the script's `texture(name, positions, normals)` asked once for each mesh, about
    every texel of it. `part` (S, S) is the index into `names` of the mesh a texel belongs to. What the script
    returns is held to one finite (r, g, b) row per position; a ValueError names the mesh and what came back."""
    colour = np.zeros(covered.shape + (3,))
    for k, name in enumerate(names):
        sel = covered & (part == k)
        n = int(sel.sum())
        if not n:
            continue
        got = paint_fn(name, point[sel].copy(), normal[sel].copy())
        try:
            got = np.asarray(got, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"texture({name!r}, ...) must return an array of numbers, one (r, g, b) row per position") from None
        if got.shape == (3,):                               # one colour for the whole mesh
            got = np.broadcast_to(got, (n, 3))
        if got.shape != (n, 3):
            raise ValueError(f"texture({name!r}, ...) was given {n} positions and returned shape {got.shape}: it must "
                             f"return one (r, g, b) row per position, shape ({n}, 3), or one (r, g, b) for them all")
        if not np.isfinite(got).all():
            raise ValueError(f"texture({name!r}, ...) returned values that are not finite")
        if got.max() > 1.5:
            raise ValueError(f"texture({name!r}, ...) returned values up to {got.max():.0f}: colours are sRGB 0..1 (a "
                             "hex channel over 255), not 0..255")
        colour[sel] = np.clip(got, 0.0, 1.0)
    return colour


def dilate(colour, covered, steps):
    """The colours spread `steps` texels out from every island into the gutter around it, so a texel sampled at an
    island's edge is the island's colour and not the background's. -> (colour, covered) after spreading."""
    colour, covered = colour.copy(), covered.copy()
    steps = max(0, int(steps))
    if not steps or covered.all():
        return colour, covered
    total = np.empty_like(colour)
    count = np.empty(covered.shape)
    for _ in range(steps):
        if covered.all():
            break
        total.fill(0)
        count.fill(0)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ys = slice(max(dy, 0), covered.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), covered.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), covered.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), covered.shape[1] + min(-dx, 0))
            mask = covered[ys, xs]
            np.add(total[yd, xd], colour[ys, xs], out=total[yd, xd], where=mask[..., None])
            count[yd, xd] += mask
        grow = ~covered & (count > 0)
        np.divide(total, count[..., None], out=colour, where=grow[..., None])
        covered |= grow
    return colour, covered


def pixels(colour, covered):
    """The atlas as Blender's `Image.pixels` wants it: RGBA float32, flat, bottom row first. A texel nothing reached
    takes the mean colour of the painted ones, so a stray sample reads as a mid tone and not as black."""
    out = np.ones(colour.shape[:2] + (4,), dtype=np.float32)
    fill = colour[covered].mean(axis=0) if covered.any() else np.full(3, 0.5)
    out[..., :3] = np.where(covered[..., None], colour, fill)
    return out.ravel()
