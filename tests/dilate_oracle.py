"""Independent straightforward gutter oracle for exact parity tests."""
import numpy as np


def dilate(colour, covered, steps):
    """The colours spread `steps` texels out from every island into the gutter around it, so a texel sampled at an
    island's edge is the island's colour and not the background's. -> (colour, covered) after spreading."""
    colour, covered = colour.copy(), covered.copy()
    for _ in range(max(0, int(steps))):
        if covered.all():
            break
        total = np.zeros_like(colour)
        count = np.zeros(covered.shape)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src_c = np.zeros_like(colour)
            src_m = np.zeros_like(covered)
            ys = slice(max(dy, 0), covered.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), covered.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), covered.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), covered.shape[1] + min(-dx, 0))
            src_c[yd, xd] = colour[ys, xs]
            src_m[yd, xd] = covered[ys, xs]
            total += np.where(src_m[..., None], src_c, 0.0)
            count += src_m
        grow = ~covered & (count > 0)
        colour[grow] = total[grow] / count[grow][:, None]
        covered |= grow
    return colour, covered
