"""When a rigged model's clips make a sound: the cues a sound script is timed to, read from the GLB with numpy.

A sound that goes with a clip has to land where the clip lands: a paw on the ground, the moment the feet leave
it, the moment they meet it again. Those times are in the animation, not in anybody's head, so they are read from
the posed vertices and handed to the sound script as numbers (`sound.cues_for`).

- `footfalls`: each time one side's lowest point comes down onto the ground, with the side (the model's own left
  is X<0, as everywhere in mesh-jig). A side whose lowest point never lifts makes none.
- `takeoff` and `landing`: in a clip whose lowest point rises clear of the ground (a leap), when it leaves the ground
  and when it meets it again.
- `crouch`: when the top of the model is lowest before the takeoff (the gather before a leap).
- `apex`: when the lowest point of the model is highest in the air.

Times are seconds into the clip. Each is where a threshold is crossed, found between two sampled frames by linear
interpolation, so a cue is good to about a millisecond on a clip keyed at 30 fps. Deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import glb, skin

FPS = 120                     # frames sampled a second of clip
CENTRE = 0.15                 # vertices within this share of the half-width of the middle belong to neither side
LIFT_SHARE = 0.25             # a side is down while its lowest point is within this share of its own rise,
FOOT_NEAR_M = 0.005           # ... and within this of its lowest: a foot is on the ground when it is on it
LIFT_MIN_M = 0.005            # a side that rises less than this does not step
STANCE_SHARE = 0.4            # a side down for less than this share of its longest stance scuffed, it did not step
AIR_SHARE = 0.02              # the model is in the air when its lowest point is this share of its height up
AIR_MIN_M = 0.01


def _clips(doc, blob) -> dict[str, float]:
    out = {}
    for a in doc.get("animations", []):
        end = max(float(glb.accessor(doc, blob, s["input"]).max()) for s in a["samplers"])
        out[a.get("name", f"clip{len(out)}")] = end
    return out


def _vertices(doc, blob, world) -> np.ndarray:
    """Every skinned or placed vertex at the pose `world`, in Blender axes (Z up, the model facing -Y or +Y)."""
    parts = []
    for ni, _ in glb.mesh_nodes(doc):
        for prim in doc["meshes"][doc["nodes"][ni]["mesh"]]["primitives"]:
            if prim.get("mode", 4) == 4:
                parts.append(skin.primitive_positions(doc, blob, ni, prim, world)[0])
    p = np.concatenate(parts)
    return np.stack((p[:, 0], -p[:, 2], p[:, 1]), axis=1)


def _crossings(down: np.ndarray, times: np.ndarray, values: np.ndarray, level: float, loop: bool,
               stance: float = 0.0) -> list[float]:
    """Times where `down` (a frame is below `level`) turns on, interpolated on `values` between the two frames.
    With `stance`, a time is kept only when `down` then stays on for at least that share of its longest run."""
    out, runs = [], []
    n = len(down)
    for i in range(n if loop else n - 1):
        j = (i + 1) % n
        if not down[i] and down[j]:
            a, b = values[i], values[j]
            u = float(np.clip((a - level) / (a - b), 0, 1)) if a != b else 0.0
            t0 = times[i]
            t1 = times[j] if j > i else times[-1] + (times[1] - times[0])
            k = 0
            while k < n and down[(j + k) % n] and (loop or j + k < n):
                k += 1
            out.append(round(float(t0 + u * (t1 - t0)), 4))
            runs.append(k)
    if stance and runs:
        out = [t for t, k in zip(out, runs) if k >= stance * max(runs)]
    return out


def read(path: str | Path, clips: dict[str, bool | None] | None = None, fps: int = FPS) -> dict[str, dict]:
    """clip -> {"seconds", "loop", "footfalls": [[t, "L"|"R"], ...], "takeoff", "landing", "crouch", "apex"}.
    `clips` names the clips to read and whether each loops (default: every clip; None, or a clip not given, loops
    when its last pose is its first). A cue the clip does not have is left out."""
    doc, blob = glb.read_glb(path)
    lengths = _clips(doc, blob)
    rest = _vertices(doc, blob, skin.worlds(doc))
    ground = float(rest[:, 2].min())
    height = float(np.ptp(rest[:, 2]))
    half = float(np.abs(rest[:, 0]).max())
    left, right = rest[:, 0] < -CENTRE * half, rest[:, 0] > CENTRE * half
    out = {}
    for clip, seconds in lengths.items():
        if clips is not None and clip not in clips:
            continue
        count = max(int(round(seconds * fps)), 2)
        first = _vertices(doc, blob, skin.sample(doc, blob, clip, 0.0))
        last = _vertices(doc, blob, skin.sample(doc, blob, clip, seconds))
        loop = clips[clip] if clips and clips[clip] is not None else bool(np.abs(first - last).max() < 1e-4)
        times = np.linspace(0, seconds, count + 1)
        times = times[:-1] if loop else times
        frames = [_vertices(doc, blob, skin.sample(doc, blob, clip, float(t))) for t in times]
        lowest = np.array([f[:, 2].min() for f in frames]) - ground
        top = np.array([f[:, 2].max() for f in frames])
        cues: dict = {"seconds": round(seconds, 4), "loop": loop}
        falls = []
        air = max(AIR_SHARE * height, AIR_MIN_M)
        for side, mask in (("L", left), ("R", right)):
            if not mask.any():
                continue
            low = np.array([f[mask, 2].min() for f in frames]) - ground
            rise = float(np.ptp(low))
            if rise < LIFT_MIN_M:
                continue
            level = float(low.min()) + min(LIFT_SHARE * rise, FOOT_NEAR_M)
            falls += [[t, side] for t in _crossings(low <= level, times, low, level, loop, STANCE_SHARE)]
        cues["footfalls"] = sorted(falls)
        up = lowest > air
        if up.any() and not up.all():
            # a leap is a clip whose lowest point rises `air` off the ground; it leaves the ground and meets it again
            # where the lowest point crosses FOOT_NEAR_M, as a foot's footfall does
            touch = max(float(lowest.min()), 0.0) + FOOT_NEAR_M
            lifts = _crossings(lowest > touch, times, -lowest, -touch, loop)
            lands = _crossings(lowest <= touch, times, lowest, touch, loop)
            if lifts and lands:
                takeoff = lifts[0]
                landing = next((t for t in lands if t > takeoff), lands[0])
                cues["takeoff"], cues["landing"] = takeoff, landing
                before = times <= takeoff
                cues["crouch"] = round(float(times[before][top[before].argmin()]), 4) if before.any() else 0.0
                cues["apex"] = round(float(times[lowest.argmax()]), 4)
        out[clip] = cues
    return out


def text(cues: dict[str, dict]) -> str:
    """The cues as lines for a person or a model."""
    lines = []
    for clip, c in cues.items():
        bits = [f"{clip}: {c['seconds']:g} s, {'loops' if c['loop'] else 'plays once'}"]
        if c.get("footfalls"):
            # a leap's two feet meeting the ground at its landing are footfalls too: said so, they read as a fault
            landing = c.get("landing")
            bits.append("footfalls " + ", ".join(
                f"{t:.3f} s ({s}" + (", the landing" if landing is not None and abs(t - landing) < 0.02 else "") + ")"
                for t, s in c["footfalls"]))
        for key in ("crouch", "takeoff", "apex", "landing"):
            if key in c:
                bits.append(f"{key} {c[key]:.3f} s")
        lines.append("; ".join(bits))
    if lines:
        lines.append("(loops or plays once as the GLB says; a sound.json `loop` overrides it, and `sound-info` shows "
                     "the cues as a sound script is given them)")
    return "\n".join(lines)
