"""Experimental mirrored-surface agreement, separate from readiness's vertex correspondence reading.

Input triangles are in Blender metres. Sampling is deterministic and area weighted, and each sampled point is
compared with the actual opposite triangle surface through a bounded-memory BVH. The area weights include internal,
overlapping, and duplicate triangles; this is not visible-surface coverage or a union-of-surfaces measure. This is a
diagnostic, not a gate.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

TOLERANCE_MM = 1.0  # Matches the existing vertex-correspondence tolerance; this is a separate geometry reading.


def mirrored_surface_agreement(triangles: np.ndarray, *, max_samples: int = 4096,
                               tolerance_mm: float = TOLERANCE_MM, leaf_size: int = 8) -> dict:
    """Compare surface on either side of X=0 with the reflected surface on the other side.

    Returns ``status='unavailable'`` for malformed, non-finite, all-degenerate, or one-sided evidence. The
    ``agreement`` is the smaller directional fraction within tolerance, so a small or missing side cannot vanish
    in a one-sided average. The samples are a deterministic, area-weighted low-discrepancy approximation; small
    features between samples and differences below tolerance can be missed. All triangle area contributes to the
    sample weights, including internal, overlapping, or duplicate triangles, so this is not visible-surface or
    union-of-surfaces coverage.
    """
    result = {"status": "unavailable", "reason": "invalid_input", "agreement": None,
              "left_to_right": None, "right_to_left": None,
              "sample_count": 0, "samples_per_direction": {"left_to_right": 0, "right_to_left": 0},
              "tolerance_mm": None, "triangle_distance_tests": 0,
              "approximation": "bounded deterministic samples weighted by all triangle area; unsigned distance; "
                              "not visible-surface or union coverage"}
    try:
        tri = np.asarray(triangles, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return result
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or len(tri) == 0 or not np.isfinite(tri).all():
        return result
    if isinstance(tolerance_mm, bool) or not isinstance(tolerance_mm, Real):
        result["reason"] = "invalid_parameters"
        return result
    try:
        tolerance_mm = float(tolerance_mm)
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "invalid_parameters"
        return result
    if (isinstance(max_samples, bool) or not isinstance(max_samples, Integral) or max_samples < 2
            or isinstance(leaf_size, bool) or not isinstance(leaf_size, Integral) or leaf_size < 1
            or not math.isfinite(tolerance_mm) or tolerance_mm < 0):
        result["reason"] = "invalid_parameters"
        return result
    max_samples, leaf_size = int(max_samples), int(leaf_size)
    result["tolerance_mm"] = tolerance_mm

    # Normalize each face before clipping, so cyclic and reversed corner orders produce identical polygons.
    tri = _canonical_vertices(tri)
    left = _clip_side(tri, positive=False)
    right = _clip_side(tri, positive=True)
    if not len(left):
        result["reason"] = "missing_left_area"
        return result
    if not len(right):
        result["reason"] = "missing_right_area"
        return result
    left_area, right_area = _areas(left), _areas(right)
    if not np.isfinite(left_area).all() or not np.isfinite(right_area).all():
        result["reason"] = "invalid_area"
        return result
    left, left_area = left[left_area > 0], left_area[left_area > 0]
    right, right_area = right[right_area > 0], right_area[right_area > 0]
    if not len(left):
        result["reason"] = "no_left_area"
        return result
    if not len(right):
        result["reason"] = "no_right_area"
        return result

    # Canonical geometry order makes the low-discrepancy sequence independent of face and node order.
    left, left_area = _canonical(left, left_area)
    right, right_area = _canonical(right, right_area)
    count = max_samples // 2
    lp = _surface_samples(left, left_area, count)
    rp = _surface_samples(right, right_area, count)
    # Reflect X and build the target indexes once per direction. Each BVH uses O(T) storage, never O(S*T).
    ltree = _BVH(left, leaf_size)
    rtree = _BVH(right, leaf_size)
    tol = tolerance_mm / 1000.0
    l2r = _within(lp * np.array([-1.0, 1.0, 1.0]), rtree, tol)
    r2l = _within(rp * np.array([-1.0, 1.0, 1.0]), ltree, tol)
    result.update(status="available", reason=None, left_to_right=float(l2r.mean()), right_to_left=float(r2l.mean()),
                  agreement=float(min(l2r.mean(), r2l.mean())), sample_count=len(lp) + len(rp),
                  samples_per_direction={"left_to_right": len(lp), "right_to_left": len(rp)},
                  left_area_m2=float(left_area.sum()), right_area_m2=float(right_area.sum()),
                  triangle_distance_tests=ltree.distance_tests + rtree.distance_tests)
    return result


def _areas(tri: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5


def _clip_side(tri: np.ndarray, *, positive: bool) -> np.ndarray:
    """Clip each triangle to one open half-space; triangles on X=0 contribute no directional evidence."""
    out = []
    for face in tri:
        if np.all(np.abs(face[:, 0]) <= 1e-15):
            continue
        poly = [p for p in face]
        clipped = []
        for a, b in zip(poly, poly[1:] + poly[:1]):
            ina = a[0] >= 0 if positive else a[0] <= 0
            inb = b[0] >= 0 if positive else b[0] <= 0
            if ina:
                clipped.append(a)
            if ina != inb:
                t = -a[0] / (b[0] - a[0])
                clipped.append(a + t * (b - a))
        if len(clipped) >= 3:
            for i in range(1, len(clipped) - 1):
                out.append((clipped[0], clipped[i], clipped[i + 1]))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3, 3)


def _canonical_vertices(tri: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(tri)
    for i, face in enumerate(tri):
        order = np.lexsort((face[:, 2], face[:, 1], face[:, 0]))
        normalized[i] = face[order]
    return normalized


def _canonical(tri: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Exact sorted vertex coordinates are the key; equal keys mean equal geometry, so tie order is immaterial.
    normalized = _canonical_vertices(tri)
    keys = normalized.reshape(len(normalized), 9)
    order = np.lexsort(tuple(keys[:, i] for i in range(8, -1, -1)))
    return normalized[order], area[order]


def _radical_inverse(n: int, base: int) -> float:
    out, f = 0.0, 1.0 / base
    while n:
        n, digit = divmod(n, base)
        out += digit * f
        f /= base
    return out


def _surface_samples(tri: np.ndarray, area: np.ndarray, count: int) -> np.ndarray:
    cumulative = np.cumsum(area)
    total = cumulative[-1]
    # Stratified area quantiles with two deterministic radical-inverse coordinates.
    q = (np.arange(count, dtype=np.float64) + 0.5) * (total / count)
    owners = np.searchsorted(cumulative, q, side="right")
    points = np.empty((count, 3), dtype=np.float64)
    for i, owner in enumerate(owners):
        # Combine the area quantile with low-discrepancy barycentrics without a random generator or global state.
        u = _radical_inverse(i + 1, 2)
        v = _radical_inverse(i + 1, 3)
        root = math.sqrt(min(1.0, max(0.0, u)))
        bary = (1.0 - root, root * (1.0 - v), root * v)
        points[i] = bary @ tri[owner]
    return points


class _BVH:
    """Median-split triangle BVH; node and leaf storage is linear in target triangle count."""
    def __init__(self, triangles: np.ndarray, leaf_size: int):
        self.triangles = triangles
        self.leaf_size = leaf_size
        self.nodes: list[tuple[np.ndarray, np.ndarray, int, int, int, int]] = []
        self.indices: list[int] = []
        self.distance_tests = 0
        self.root = self._build(np.arange(len(triangles), dtype=np.int32))

    def _build(self, ids: np.ndarray) -> int:
        part = self.triangles[ids]
        lo, hi = part.min(axis=(0, 1)), part.max(axis=(0, 1))
        index = len(self.nodes)
        self.nodes.append((lo, hi, -1, -1, 0, 0))
        if len(ids) <= self.leaf_size:
            start = len(self.indices)
            self.indices.extend(map(int, ids))
            self.nodes[index] = (lo, hi, -1, -1, start, len(ids))
            return index
        centers = part.mean(axis=1)
        axis = int(np.argmax(hi - lo))
        order = np.argsort(centers[:, axis], kind="mergesort")
        mid = len(ids) // 2
        a = self._build(ids[order[:mid]])
        b = self._build(ids[order[mid:]])
        self.nodes[index] = (lo, hi, a, b, 0, 0)
        return index

    @staticmethod
    def _box_distance2(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        delta = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        return float(delta @ delta)

    def nearest2(self, point: np.ndarray) -> float:
        best = math.inf
        stack = [self.root]
        while stack:
            node = stack.pop()
            lo, hi, a, b, start, size = self.nodes[node]
            if self._box_distance2(point, lo, hi) >= best:
                continue
            if a < 0:
                ids = self.indices[start:start + size]
                for tri_id in ids:
                    self.distance_tests += 1
                    best = min(best, _point_triangle_distance2(point, self.triangles[tri_id]))
            else:
                da = self._box_distance2(point, self.nodes[a][0], self.nodes[a][1])
                db = self._box_distance2(point, self.nodes[b][0], self.nodes[b][1])
                if da < db:
                    stack.extend((b, a))
                else:
                    stack.extend((a, b))
        return best


def _point_triangle_distance2(p: np.ndarray, tri: np.ndarray) -> float:
    """Squared Euclidean distance to a triangle (Ericson region tests), including edges and vertices."""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return float(ap @ ap)
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return float(bp @ bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        d = p - q
        return float(d @ d)
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return float(cp @ cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        d = p - q
        return float(d @ d)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + w * (c - b)
        d = p - q
        return float(d @ d)
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    d = p - (a + ab * v + ac * w)
    return float(d @ d)


def _within(points: np.ndarray, tree: _BVH, tolerance: float) -> np.ndarray:
    tol2 = tolerance * tolerance
    return np.fromiter((tree.nearest2(p) <= tol2 for p in points), dtype=bool, count=len(points))
