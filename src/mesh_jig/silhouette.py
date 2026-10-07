"""Silhouette match per view: a deterministic, noise-free number for outline and proportion.

A vision judge cannot rank near-identical model candidates (its per-call spread is larger than the differences), and
outline and proportion are what the next edit needs. So this measures them against the reference views with no model
call:

- `mask(img)`: foreground = pixels far from the border's background colour, holes filled (flood fill from the border),
  only components at least MIN_COMPONENT of the largest kept (drops a sheet's view label and speckle).
- `normalise(mask)`: crop to the bounding box, scale so the longer side is GRID, centre on a GRID x GRID canvas. The
  aspect ratio survives, so a slender body against a chunky reference costs overlap.
- `compare(render, reference)`: IoU of the two normalised masks, both aspect ratios (box height / width), and the
  share of the reference the render misses and of the render the reference lacks.
- `overlay(...)`: the diff an agent can read: grey where both have body, red where only the reference does (add),
  blue where only the candidate does (remove).

Hero views are three-quarter perspectives from a painted sheet and a render camera that do not match exactly; their
IoU is indicative, the orthographic top / side / rear are the ones to rank on (`ORTHO_VIEWS`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

GRID = 160
BG_DISTANCE = 28.0          # RGB distance from the background colour that counts as body on a painted sheet
FLAT_DISTANCE = 2.5         # on a render: its ground is exactly one colour, and a lit pale face can sit 1-12 away from it
MIN_COMPONENT = 0.02        # components smaller than this share of the largest are dropped
ORTHO_VIEWS = ("top", "side", "rear")
REPORT = "silhouette.json"
OVERLAY_PREFIX = "sil_"
MAX_DIM = 512               # masks are computed at this size
MEDIAN = 5                  # median filter on the raw mask: a painted sheet's paper grain and render aliasing
MAX_COMPONENTS = 32         # labelled at most; after the median filter a sheet has a handful (body, label, barrel)


def _load(img: str | Path | Image.Image) -> Image.Image:
    im = img if isinstance(img, Image.Image) else Image.open(img)
    im = im.convert("RGB")
    if max(im.size) > MAX_DIM:
        k = MAX_DIM / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR)
    return im


def background_colour(a: np.ndarray) -> np.ndarray:
    """Median of the one-pixel border: a render and a sheet are both flat light grounds."""
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).reshape(-1, 3)
    return np.median(border, axis=0)


def _spread(open_: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """One pass along the rows: every run of consecutive open pixels holding a reached pixel becomes reached."""
    h, w = open_.shape
    start = open_.copy()
    start[:, 1:] &= ~open_[:, :-1]                      # the first pixel of each run
    ids = np.where(open_, np.cumsum(start.ravel()).reshape(h, w), 0)
    hit = np.bincount(ids.ravel(), weights=reach.ravel(), minlength=int(ids.max()) + 1) > 0
    hit[0] = False
    return hit[ids]


def flood(open_: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """The pixels of `open_` 4-connected to the pixel at seed (row, col): a flood fill in whole-array passes, rows
    then columns until nothing changes, instead of a per-pixel walk in Python."""
    reach = np.zeros_like(open_)
    if not open_[seed]:
        return reach
    reach[seed] = True
    count = 1
    while True:
        reach = _spread(open_, reach)
        reach = _spread(open_.T, reach.T).T
        now = int(reach.sum())
        if now == count:
            return reach
        count = now


def _fill_holes(fg: np.ndarray) -> np.ndarray:
    """Everything not reachable from the border through background is body (pale panels inside a dark outline)."""
    ground = np.pad(~fg, 1, constant_values=True)      # a one-pixel moat, so the whole border is one region
    return ~flood(ground, (0, 0))[1:-1, 1:-1]


def _components(fg: np.ndarray) -> list[np.ndarray]:
    """Connected components (4-connected) as boolean masks, largest first: at most MAX_COMPONENTS are labelled;
    anything left is speckle and dropped."""
    left = fg.copy()
    comps = []
    while len(comps) < MAX_COMPONENTS:
        ys, xs = np.nonzero(left)
        if len(ys) == 0:
            break
        comp = flood(left, (int(ys[0]), int(xs[0])))
        comps.append(comp)
        left &= ~comp
    return sorted(comps, key=lambda m: -int(m.sum()))


def body_distance(d: np.ndarray) -> float:
    """The threshold for this image: a border that is exactly one colour is a render (FLAT_DISTANCE), any grain
    means a painted sheet (BG_DISTANCE). One threshold for both loses pale wings on a pale render ground."""
    border = np.concatenate([d[0], d[-1], d[:, 0], d[:, -1]])
    return FLAT_DISTANCE if float(np.percentile(border, 75)) < 0.5 else BG_DISTANCE


_MASKS: dict[tuple, np.ndarray] = {}
MASK_CACHE = 64


def mask(img: str | Path | Image.Image, distance: float | None = None) -> np.ndarray:
    """The subject of an image as a boolean array (at most MAX_DIM on a side). A mask asked for by file path is kept
    (keyed on the file's size and modification time): one evaluation asks for each view's a dozen times."""
    if isinstance(img, Image.Image):
        return _mask(img, distance)
    st = Path(img).stat()
    key = (str(Path(img).resolve()), st.st_mtime_ns, st.st_size, distance)
    if key not in _MASKS:
        if len(_MASKS) >= MASK_CACHE:
            _MASKS.pop(next(iter(_MASKS)))
        m = _mask(img, distance)
        m.flags.writeable = False
        _MASKS[key] = m
    return _MASKS[key]


def _mask(img: str | Path | Image.Image, distance: float | None = None) -> np.ndarray:
    a = np.asarray(_load(img)).astype(np.float32)
    d = np.linalg.norm(a - background_colour(a), axis=2)
    distance = body_distance(d) if distance is None else distance
    raw = Image.fromarray(np.where(d > distance, 255, 0).astype(np.uint8), "L").filter(ImageFilter.MedianFilter(MEDIAN))
    fg = _fill_holes(np.asarray(raw) > 127)
    comps = _components(fg)
    if not comps:
        return fg
    biggest = int(comps[0].sum())
    keep = np.zeros_like(fg)
    for c in comps:
        if c.sum() >= MIN_COMPONENT * biggest:
            keep |= c
    return keep


def bbox(m: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def aspect(m: np.ndarray) -> float | None:
    b = bbox(m)
    return None if b is None else round((b[3] - b[1]) / max(1, b[2] - b[0]), 3)


def normalise(m: np.ndarray, grid: int = GRID) -> np.ndarray:
    b = bbox(m)
    out = np.zeros((grid, grid), dtype=bool)
    if b is None:
        return out
    crop = m[b[1]:b[3], b[0]:b[2]]
    h, w = crop.shape
    k = grid / max(h, w)
    nw, nh = min(grid, max(1, round(w * k))), min(grid, max(1, round(h * k)))
    im = Image.fromarray(np.where(crop, 255, 0).astype(np.uint8), "L").resize((nw, nh), Image.NEAREST)
    x0, y0 = (grid - nw) // 2, (grid - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = np.asarray(im) > 127
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return round(int((a & b).sum()) / union, 4) if union else 0.0


def void_fraction(img: str | Path | Image.Image) -> float:
    """Backdrop seen THROUGH the body, as a fraction of the filled outline. `mask` fills holes on purpose (pale panels
    inside a dark outline are body), so a solid lump and a pierced one have the same silhouette; this is the number
    that tells them apart. Recorded, not gated: on a sheet whose body has backdrop-coloured panels it measures paint."""
    a = np.asarray(_load(img)).astype(np.float32)
    d = np.linalg.norm(a - background_colour(a), axis=2)
    raw = np.asarray(Image.fromarray(np.where(d > body_distance(d), 255, 0).astype(np.uint8), "L")
                     .filter(ImageFilter.MedianFilter(MEDIAN))) > 127
    filled = mask(img)
    return round(float((filled & ~raw).sum()) / max(1, int(filled.sum())), 4)


def compare(render: str | Path | Image.Image, reference: str | Path | Image.Image) -> dict:
    rm, fm = mask(render), mask(reference)
    rn, fn = normalise(rm), normalise(fm)
    return {"iou": iou(rn, fn), "aspect": aspect(rm), "reference_aspect": aspect(fm),
            "missing": round(float((fn & ~rn).sum()) / max(1, int(fn.sum())), 4),   # reference body the render lacks
            "extra": round(float((rn & ~fn).sum()) / max(1, int(rn.sum())), 4),      # render body the reference lacks
            "void": void_fraction(render), "reference_void": void_fraction(reference)}


def overlay(render: str | Path | Image.Image, reference: str | Path | Image.Image, grid: int = GRID * 2) -> Image.Image:
    rn, fn = normalise(mask(render), grid), normalise(mask(reference), grid)
    out = np.full((grid, grid, 3), 250, dtype=np.uint8)
    out[rn & fn] = (150, 150, 150)
    out[fn & ~rn] = (220, 40, 40)      # add: the reference has body here
    out[rn & ~fn] = (40, 90, 220)      # remove: only the candidate has body here
    return Image.fromarray(out, "RGB")


SUBJECT_MARGIN = 0.06   # subject crop margin, share of the larger side


def subject_box(img: Image.Image, margin: float = SUBJECT_MARGIN, source: str | Path | None = None
                ) -> tuple[int, int, int, int]:
    """The subject's bounding box in `img` pixels (silhouette mask, painted sheet or render), with a margin. `source`
    is the file `img` was opened from, when there is one: its mask may already be known."""
    m = mask(source if source is not None else img)
    b = bbox(m)
    if b is None:
        return 0, 0, img.width, img.height
    k = img.width / m.shape[1]
    x0, y0, x1, y1 = (round(v * k) for v in b)
    pad = round(margin * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)


def subject_panel(img: str | Path | Image.Image, height: int) -> Image.Image:
    """`img` cropped to its subject and scaled to `height`: framing and image size stop being differences."""
    im = (img if isinstance(img, Image.Image) else Image.open(img)).convert("RGB")   # full size: _load would shrink it
    im = im.crop(subject_box(im, source=None if isinstance(img, Image.Image) else img))
    w = max(1, round(im.width * height / max(1, im.height)))
    return im.resize((w, height), Image.LANCZOS)


def side_by_side(left: str | Path | Image.Image, right: str | Path | Image.Image, left_label: str, right_label: str,
                 height: int = 420, gap: int = 24) -> Image.Image:
    """LEFT | RIGHT, both cropped to the subject and scaled to `height`, labelled above."""
    a, b = subject_panel(left, height), subject_panel(right, height)
    label = 28
    out = Image.new("RGB", (a.width + gap + b.width, height + label), (255, 255, 255))
    out.paste(a, (0, label))
    out.paste(b, (a.width + gap, label))
    d = ImageDraw.Draw(out)
    d.text((6, 6), left_label, fill=(0, 0, 0))
    d.text((a.width + gap + 6, 6), right_label, fill=(0, 0, 0))
    d.line([(a.width + gap // 2, 0), (a.width + gap // 2, out.height)], fill=(0, 0, 0), width=2)
    return out


def measure_views(renders: dict[str, str | Path], references: dict[str, str | Path], out_dir: Path | None = None,
                  *, weights: dict[str, float] | None = None) -> dict:
    """view -> compare() for every render with a reference of the same name; `ortho_iou` is the mean over the
    orthographic views present (the ranking number). Writes silhouette.json and sil_<view>.png into out_dir."""
    views = {}
    for name in sorted(renders):
        if name not in references:
            continue
        views[name] = compare(renders[name], references[name])
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            overlay(renders[name], references[name]).save(Path(out_dir) / f"{OVERLAY_PREFIX}{name}.png")
    ortho = [views[v]["iou"] for v in ORTHO_VIEWS if v in views]
    rec = {"views": views, "ortho_iou": round(sum(ortho) / len(ortho), 4) if ortho else None,
           "legend": "grey both, red only the reference (add body), blue only the candidate (remove body)"}
    if weights is not None:
        from .scoring import aggregate, weak_views
        rec.update(ortho_iou=aggregate(views, "iou", weights), weights=weights,
                   weak_views=weak_views(views, "iou", weights))
    if out_dir is not None:
        (Path(out_dir) / REPORT).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def feedback_lines(rec: dict) -> list[str]:
    """One line per view: overlap, and the proportion gap in words."""
    lines = []
    for v, d in (rec.get("views") or {}).items():
        a, ra = d.get("aspect"), d.get("reference_aspect")
        prop = ""
        if a and ra:
            if a < ra * 0.9:
                prop = f"; too flat or too wide: box height/width {a} vs reference {ra}"
            elif a > ra * 1.1:
                prop = f"; too tall or too narrow: box height/width {a} vs reference {ra}"
        lines.append(f"{v}: silhouette overlap {d['iou']:.2f}, missing {d['missing']:.0%} of the reference, "
                     f"{d['extra']:.0%} of yours outside it{prop}")
    return lines
