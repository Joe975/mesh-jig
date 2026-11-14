"""From a turntable video of a real object to a six-view photo reference sheet.

Subcommands, in the order they are used (skill/mesh-jig-video-reference/SKILL.md):

  frames  VIDEO OUT [--fps 4]                       every frame at --fps as OUT/f_NNN.jpg (ffmpeg)
  contact FRAMES OUT --box x0,y0,x1,y1 [--first N --last M --cols 14 --width 170]
                                                    numbered thumbnails of one region, to pick views from
  cut     OUT VIEW=FRAME:x0,y0,x1,y1[:rot] ... [--model isnet-general-use]
                                                    crop each view (OUT/<view>_src.png) and segment it with
                                                    rembg (OUT/<view>.png, RGBA). rot: degrees anticlockwise
  clean   CUT OUT VIEW ... [--accent g --accent-margin 10]
                                                    one subject mask per view, the view on flat white
                                                    (OUT/<view>.png) and the mask (OUT/<view>_mask.png)
  overlay CUT CLEAN OUT VIEW ...                    each source crop with its mask outline, to check by eye
  sheet   CLEAN OUT [--top-scale 1]                 the labelled 3x2 sheet

Only `cut` needs anything beyond numpy and Pillow: rembg (pip install "rembg[cpu]") and its model download.
`frames` needs ffmpeg on PATH. The rest is deterministic: same inputs, same bytes out.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = ("front", "side", "rear", "right", "hero", "top")
LAYOUT = (("front", "FRONT"), ("side", "LEFT SIDE"), ("rear", "REAR"),
          ("right", "RIGHT SIDE"), ("hero", "HERO"), ("top", "TOP"))
CHANNEL = {"r": 0, "g": 1, "b": 2}


# ----------------------------------------------------------------------------------------------- masks, numpy only

def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components of a boolean image: (labels, count), labels 1..count, 0 for background.

    Run-length union-find: a run joins every run in the row above that overlaps it or touches it diagonally."""
    h, w = mask.shape
    parent: list[int] = [0]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    runs: list[tuple[int, int, int, int]] = []     # (row, start, end exclusive, provisional id)
    prev: list[tuple[int, int, int]] = []
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    for y in range(h):
        starts = np.flatnonzero(edges[y] == 1)
        ends = np.flatnonzero(edges[y] == -1)
        row = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            for ps, pe, pid in prev:
                if ps <= e and pe >= s:            # overlap, or touch at a corner (8-connected)
                    a, b = find(rid), find(pid)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            row.append((s, e, rid))
            runs.append((y, s, e, rid))
        prev = row
    out = np.zeros((h, w), dtype=np.int32)
    final: dict[int, int] = {}
    for y, s, e, rid in runs:
        root = find(rid)
        out[y, s:e] = final.setdefault(root, len(final) + 1)
    return out, len(final)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r pixels (a (2r+1)-wide square), separable."""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for d in range(1, r + 1):
            grown |= np.roll(out, d, axis) & _valid(out.shape, axis, d)
            grown |= np.roll(out, -d, axis) & _valid(out.shape, axis, -d)
        out = grown
    return out


def _valid(shape, axis, d):
    """Where np.roll by d along axis brought in a real pixel rather than one wrapped round."""
    v = np.ones(shape, dtype=bool)
    idx = [slice(None)] * 2
    idx[axis] = slice(0, d) if d > 0 else slice(shape[axis] + d, None)
    v[tuple(idx)] = False
    return v


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return ~dilate(~mask, r)


def subject_mask(alpha: np.ndarray, rgb: np.ndarray, *, alpha_min: int = 110, accent: str | None = "g",
                 accent_margin: int = 10, accent_floor: int = 40, touch_px: int = 10, min_piece: float = 0.02,
                 max_hole: int = 40) -> np.ndarray:
    """The subject in one view, from a segmenter's alpha and the source crop's colours.

    - core: alpha >= alpha_min. A segmenter is sure of these.
    - accent: pixels where one channel (r, g or b) beats both others by more than accent_margin and is above
      accent_floor, counted only where they connect to a band touch_px wide around the core. A saturated part a
      segmenter leaves see-through (a glowing blade, a coloured lens) comes back; the same colour elsewhere in the
      picture does not. accent=None turns it off.
    - pieces under min_piece of the largest are dropped: mesh-jig takes a separate piece of 2% or more as subject
      (docs/JIG.md, Reference views), so a smaller scrap is noise.
    - enclosed holes up to max_hole pixels are filled; larger gaps (between legs) stay open."""
    core = alpha >= alpha_min
    if accent:
        c = rgb[..., CHANNEL[accent]].astype(np.int16)
        others = np.delete(rgb, CHANNEL[accent], axis=-1).astype(np.int16).max(axis=-1)
        hot = (c - others > accent_margin) & (c > accent_floor)
        lab, _ = label(hot | dilate(core, touch_px))
        keep = np.unique(lab[core])
        mask = np.isin(lab, keep[keep > 0]) & (hot | core)
    else:
        mask = core.copy()
    mask = erode(dilate(mask, 1), 1)                       # close one-pixel cracks
    lab, n = label(mask)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        mask = np.isin(lab, np.flatnonzero(sizes >= min_piece * sizes.max()) + 1)
    holes, hn = label(~mask)
    if hn:
        border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
        sizes = np.bincount(holes.ravel(), minlength=hn + 1)
        small = [i for i in range(1, hn + 1) if i not in set(border.tolist()) and sizes[i] <= max_hole]
        mask |= np.isin(holes, small)
    return mask


def on_white(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray, margin: float = 0.06) -> Image.Image:
    """The masked subject on pure white, trimmed to the mask with a margin of `margin` of its longer side.

    The mask is opaque; on its one-pixel rim the segmenter's alpha (at least half) softens the edge."""
    rim = mask & ~erode(mask, 1)
    a = np.where(rim, np.maximum(alpha, 128), np.where(mask, 255, 0)).astype(np.float32) / 255
    comp = (rgb.astype(np.float32) * a[..., None] + 255 * (1 - a[..., None])).round().astype(np.uint8)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = round(margin * max(y1 - y0, x1 - x0))
    canvas = Image.new("RGB", (int(x1 - x0) + 2 * m, int(y1 - y0) + 2 * m), "white")
    canvas.paste(Image.fromarray(comp[y0:y1, x0:x1]), (m, m))
    return canvas


def tight(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------------------------------------------- the sheet

def compose(views: dict[str, tuple[Image.Image, np.ndarray]], *, top_scale: float = 1.0, pad: int = 40,
            label_h: int = 70, font: ImageFont.ImageFont | None = None) -> Image.Image:
    """A 3x2 sheet: FRONT, LEFT SIDE, REAR / RIGHT SIDE, HERO, TOP. Each entry is (view on white, its mask).

    Every view but top keeps its own pixel scale: from one camera at one distance that is one scale. Standing views
    are bottom-aligned on one ground line per row; top is centred and scaled by top_scale."""
    tiles = {}
    for name, (im, mask) in views.items():
        x0, y0, x1, y1 = tight(mask)
        w, h = x1 - x0, y1 - y0
        mx, my = (im.width - w) // 2, (im.height - h) // 2
        tile = im.crop((mx, my, mx + w, my + h))
        if name == "top" and top_scale != 1:
            tile = tile.resize((round(w * top_scale), round(h * top_scale)), Image.LANCZOS)
        tiles[name] = tile
    cell_w = max(t.width for t in tiles.values()) + 2 * pad
    rows = [LAYOUT[:3], LAYOUT[3:]]
    row_h = [max((tiles[v].height for v, _ in row if v in tiles), default=0) + 2 * pad for row in rows]
    sheet = Image.new("RGB", (3 * cell_w, sum(row_h) + 2 * label_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = font or _font(max(8, round(label_h * 4 / 7)))
    y = 0
    for row, rh in zip(rows, row_h):
        for c, (v, text) in enumerate(row):
            if v not in tiles:
                continue
            t = tiles[v]
            x = c * cell_w + (cell_w - t.width) // 2
            ty = y + (rh - t.height) // 2 if v == "top" else y + rh - pad - t.height
            sheet.paste(t, (x, ty))
            tw = draw.textlength(text, font=font)
            draw.text((c * cell_w + (cell_w - tw) / 2, y + rh + 10), text, fill="black", font=font)
        y += rh + label_h
    return sheet


def _font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


# ------------------------------------------------------------------------------------------------------ the CLI

def parse_view(spec: str) -> tuple[str, str, tuple[int, int, int, int], int]:
    """VIEW=FRAME:x0,y0,x1,y1[:rot]. FRAME may carry a Windows drive letter (C:/...)."""
    m = re.fullmatch(r"(\w+)=(.+):(\d+),(\d+),(\d+),(\d+)(?::(-?\d+))?", spec)
    if not m:
        raise SystemExit(f"view spec {spec!r}: expected VIEW=FRAME:x0,y0,x1,y1[:rot]")
    name, frame = m.group(1), m.group(2)
    if name not in VIEWS:
        raise SystemExit(f"view {name!r}: one of {', '.join(VIEWS)}")
    box = tuple(int(m.group(i)) for i in range(3, 7))
    return name, frame, box, int(m.group(7) or 0)


def cmd_frames(a):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-vf", f"fps={a.fps}", "-q:v", "2",
                    str(out / "f_%03d.jpg")], check=True)
    n = len(list(out.glob("f_*.jpg")))
    print(f"{n} frames at {a.fps} fps in {out}: frame N is at about (N-1)/{a.fps} s")


def cmd_contact(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))
    frames = [f for f in frames if a.first <= int(f.stem[2:]) <= (a.last or 10**9)]
    x0, y0, x1, y1 = (int(v) for v in a.box.split(","))
    th = round(a.width * (y1 - y0) / (x1 - x0))
    rows = -(-len(frames) // a.cols)
    sheet = Image.new("RGB", (a.cols * a.width, rows * th), "black")
    font = _font(max(12, a.width // 8))
    for i, f in enumerate(frames):
        im = Image.open(f).convert("RGB").crop((x0, y0, x1, y1)).resize((a.width, th))
        d = ImageDraw.Draw(im)
        tag = f.stem[2:]
        d.rectangle((0, 0, d.textlength(tag, font=font) + 6, font.size + 6), fill="black")
        d.text((3, 2), tag, fill="yellow", font=font)
        sheet.paste(im, ((i % a.cols) * a.width, (i // a.cols) * th))
    sheet.save(a.out, quality=90)
    print(f"{len(frames)} frames -> {a.out}")


def cmd_cut(a):
    try:
        from rembg import new_session, remove
    except ImportError:
        raise SystemExit('cut needs rembg: pip install "rembg[cpu]" in a separate venv')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    session = new_session(a.model)
    for spec in a.views:
        name, frame, box, rot = parse_view(spec)
        im = Image.open(frame).convert("RGB").crop(box)
        if rot:
            im = im.rotate(rot, expand=True)
        im.save(out / f"{name}_src.png")
        remove(im, session=session, post_process_mask=False).save(out / f"{name}.png")
        print(name, im.size)


def cmd_clean(a):
    cut, out = Path(a.cut), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.views:
        alpha = np.asarray(Image.open(cut / f"{v}.png").convert("RGBA"))[..., 3]
        rgb = np.asarray(Image.open(cut / f"{v}_src.png").convert("RGB"))
        mask = subject_mask(alpha, rgb, alpha_min=a.alpha_min, accent=None if a.accent == "none" else a.accent,
                            accent_margin=a.accent_margin, touch_px=a.touch_px, min_piece=a.min_piece)
        on_white(rgb, alpha, mask).save(out / f"{v}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{v}_mask.png")
        x0, y0, x1, y1 = tight(mask)
        h, w = mask.shape
        edge = " TOUCHES THE CROP EDGE: widen the crop" if (x0 == 0 or y0 == 0 or x1 == w or y1 == h) else ""
        print(f"{v}: subject {x1 - x0}x{y1 - y0} px at ({x0},{y0}){edge}")


def cmd_overlay(a):
    cut, clean = Path(a.cut), Path(a.clean)
    tiles = []
    for v in a.views:
        rgb = np.asarray(Image.open(cut / f"{v}_src.png").convert("RGB")).copy()
        m = np.asarray(Image.open(clean / f"{v}_mask.png")) > 0
        rgb[m & ~erode(m, 2)] = (255, 0, 255)
        im = Image.fromarray(rgb)
        tiles.append(im.resize((round(im.width * a.height / im.height), a.height)))
    sheet = Image.new("RGB", (sum(t.width + 6 for t in tiles), a.height), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    sheet.save(a.out, quality=90)


def cmd_sheet(a):
    clean = Path(a.clean)
    views = {v: (Image.open(clean / f"{v}.png").convert("RGB"), np.asarray(Image.open(clean / f"{v}_mask.png")) > 0)
             for v in VIEWS if (clean / f"{v}.png").exists()}
    sheet = compose(views, top_scale=a.top_scale)
    sheet.save(a.out)
    print(f"sheet {sheet.size[0]}x{sheet.size[1]} -> {a.out}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("frames"); s.add_argument("video"); s.add_argument("out")
    s.add_argument("--fps", type=float, default=4); s.set_defaults(fn=cmd_frames)
    s = sub.add_parser("contact"); s.add_argument("frames"); s.add_argument("out")
    s.add_argument("--box", required=True); s.add_argument("--first", type=int, default=1)
    s.add_argument("--last", type=int, default=0); s.add_argument("--cols", type=int, default=14)
    s.add_argument("--width", type=int, default=170); s.set_defaults(fn=cmd_contact)
    s = sub.add_parser("cut"); s.add_argument("out"); s.add_argument("views", nargs="+")
    s.add_argument("--model", default="isnet-general-use"); s.set_defaults(fn=cmd_cut)
    s = sub.add_parser("clean"); s.add_argument("cut"); s.add_argument("out"); s.add_argument("views", nargs="+")
    s.add_argument("--alpha-min", type=int, default=110)
    s.add_argument("--accent", choices=("r", "g", "b", "none"), default="g")
    s.add_argument("--accent-margin", type=int, default=10); s.add_argument("--touch-px", type=int, default=10)
    s.add_argument("--min-piece", type=float, default=0.02); s.set_defaults(fn=cmd_clean)
    s = sub.add_parser("overlay"); s.add_argument("cut"); s.add_argument("clean"); s.add_argument("out")
    s.add_argument("views", nargs="+"); s.add_argument("--height", type=int, default=700)
    s.set_defaults(fn=cmd_overlay)
    s = sub.add_parser("sheet"); s.add_argument("clean"); s.add_argument("out")
    s.add_argument("--top-scale", type=float, default=1.0); s.set_defaults(fn=cmd_sheet)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
