"""What a rig script's author has to be told, written from the source model and the contract.

`facts` is the model as a rig meets it (nodes, connected pieces, bounds, where it stands, and a table of slices
through it by height), `rules` the script's interface, `task` the contract's clips and gates in words. `mesh-jig
rig-info` prints them and `rig_agent` opens a run with them. The numbers are the source's own: nothing here knows
what kind of subject it is.
"""
from __future__ import annotations

import math

import json
import re
from pathlib import Path

import numpy as np

from . import glb, rig

SLICE_M = 0.05                # a slice's thickness
GAP_M = 0.03                  # two runs of a slice further apart than this are two parts
MAX_NODES = 40                # lines of the node list: nodes named alike (`wrap`, `wrap.001`, ...) share one
SIDE_M = 0.05                 # a piece wholly this far to one side of the middle may be a limb of its own
MAX_PIECES = 8                # a node in this many pieces or fewer has each listed with its bounds
SHORT_LEG = 0.25              # legs shorter than this share of the height: the recipe's sizes, in leg lengths, shrink
BULK = 0.5                    # a node with this share of the vertices is the model's skin; with none, it is in pieces


def load(source: str | Path) -> tuple[dict, list]:
    doc, blob = glb.read_glb(source)
    return rig.model_data(doc, blob)


def _v(p, places=2) -> str:
    return "(" + ", ".join(f"{float(v):+.{places}f}" for v in p) + ")"


def triangles(doc: dict, blob: bytes, name: str | None = None) -> np.ndarray:
    """(T, 3, 3): the node's triangles in Blender metres, or every node's when no name is given."""
    out = []
    for ni, world in glb.mesh_nodes(doc):
        node = doc["nodes"][ni]
        if name is not None and node.get("name", f"node{ni}") != name:
            continue
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            pos = glb.accessor(doc, blob, prim["attributes"]["POSITION"])
            pts = rig.to_blender(pos @ world[:3, :3].T + world[:3, 3])
            idx = glb.accessor(doc, blob, prim["indices"]).astype(int).ravel() if "indices" in prim else np.arange(len(pos))
            out.append(pts[idx[: len(idx) // 3 * 3].reshape(-1, 3)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def slices(tris: np.ndarray) -> list[tuple[float, list[tuple[float, float, float, float]]]]:
    """Top to bottom, a cut every SLICE_M: its height, and each separate run across X as (x from, x to, y from,
    y to). Two runs in one cut are two parts side by side at that height: two legs, an arm beside a flank. A cut is
    where the plane crosses the triangles, so a mesh of a few large faces reads as a dense one does."""
    out = []
    top, bottom = float(tris[..., 2].max()), float(tris[..., 2].min())
    a, b = tris, np.roll(tris, -1, axis=1)                     # each triangle's three edges, a -> b
    for k in range(int(np.ceil((top - bottom) / SLICE_M))):
        z = top - (k + 0.5) * SLICE_M
        da, db = a[..., 2] - z, b[..., 2] - z
        crossing = (da * db <= 0) & (da != db)
        if not crossing.any():
            continue
        u = np.where(crossing, da / np.where(da != db, da - db, 1.0), 0.0)[..., None]
        hit = a + (b - a) * u                                   # (T, 3, 3): where each crossing edge meets the plane
        cut = crossing.any(axis=1)
        x = np.where(crossing, hit[..., 0], np.nan)[cut]
        y = np.where(crossing, hit[..., 1], np.nan)[cut]
        spans = np.stack([np.nanmin(x, axis=1), np.nanmax(x, axis=1), np.nanmin(y, axis=1), np.nanmax(y, axis=1)], axis=1)
        spans = spans[np.argsort(spans[:, 0], kind="stable")]
        runs = [list(spans[0])]
        for x0, x1, y0, y1 in spans[1:]:
            if x0 - runs[-1][1] > GAP_M:
                runs.append([x0, x1, y0, y1])
            else:
                runs[-1] = [runs[-1][0], max(runs[-1][1], x1), min(runs[-1][2], y0), max(runs[-1][3], y1)]
        out.append((z, [tuple(float(v) for v in r) for r in runs]))
    return out


def stretches(rows: list, ground: float) -> list[dict]:
    """Each stretch of height over which the surface is two or more separate runs, top to bottom: `count` runs,
    from `top` down to `bottom`, the runs at its `middle` cut, whether it is `on_ground`, whether it `leaves` the body
    (the cuts above it are one run) and, for side parts that slope outward, the height they are `joined` to the body
    from (None when they hang clear where they begin). `begins` is where the parts' room for joints starts: `joined`
    when there is one, else `top`. `landmarks` says these in words and `readiness` reads them for a mesh."""
    def lone(k):
        # one cut between fewer runs above and more below: the split has come a cut early on one side (a sword at
        # one hip, one arm a little lower than the other). It is the top of the stretch below, not one of its own
        return 0 < k < len(rows) - 1 and len(rows[k - 1][1]) < len(rows[k][1]) < len(rows[k + 1][1])

    out, k = [], 0
    while k < len(rows):
        count, start = len(rows[k][1]), k
        while k + 1 < len(rows) and len(rows[k + 1][1]) == count:
            k += 1
        top, bottom = rows[start][0] + SLICE_M / 2, rows[k][0] - SLICE_M / 2
        if count > 1 and not (k == start and lone(k)):
            middle = rows[(start + k) // 2][1]
            joined, core = None, 0.0
            if start > 0:
                # side parts that slope outward are joined to the body for a stretch before they hang clear: the
                # single run above is then wider than the middle part alone, and the joint is at the top of that
                inner = middle[1:-1]
                core = max((b - a for a, b, _, _ in inner), default=0.0)
                above = start - 2 if lone(start - 1) else start - 1
                j = above
                while core and len(rows[start - 1][1]) < count and j >= 0 and len(rows[j][1]) == 1 and rows[j][1][0][1] - rows[j][1][0][0] > 1.1 * core:
                    j -= 1
                if core and j < above:
                    joined = rows[j + 1][0] + SLICE_M / 2
            out.append({"count": count, "top": top, "bottom": bottom, "middle": middle,
                        "on_ground": bottom <= ground + SLICE_M, "leaves": start > 0, "joined": joined, "core": core,
                        "begins": top if joined is None else joined})
        k += 1
    return out


def thickness(s: dict) -> str:
    """For side parts that slope out from a joint (an arm hanging away from the body), how thick each is ACROSS
    itself, which is what a limb's radius is: a horizontal slice through a sloped arm is wider than the arm, and a
    builder who took half the slice's width for ARM_RADIUS had its forearm take the hips' weight."""
    if s.get("joined") is None or not s.get("middle") or len(s["middle"]) < 3:
        return ""
    a, b, c, d = s["middle"][-1]                      # the side part at x > 0, at the middle of the stretch
    mid_x, mid_z = (a + b) / 2, (s["top"] + s["bottom"]) / 2
    dx, dz = abs(mid_x) - s["core"] / 2, s["joined"] - mid_z
    if dz <= 0 or dx <= 0:
        return ""
    slope = math.atan2(dx, dz)
    across = max((b - a) * math.cos(slope), d - c)
    body = min(s["middle"][1:-1], key=lambda m: abs((m[0] + m[1]) / 2))
    room = mid_x - body[1]                            # from the side part's middle line in to the body's side
    said = (f"Each slopes about {math.degrees(slope):.0f} degrees out from upright, so across itself it is about "
            f"{across:.2f} m thick where a slice is {b - a:.2f} m wide, and its middle line is {room:.2f} m from the "
            f"body's side here. A radius a little over half its thickness, about {across / 2 * 1.15:.3f} m, reaches "
            "all of it")
    if across / 2 * 1.15 >= room - 0.02:
        return said + (f"; but that is within 2 cm of the body, so it would take the body's weight: keep the radius "
                       f"under {room - 0.02:.3f} m and leave the rest of the part to a test on its own side of the body.")
    return said + (f", and stays {room - across / 2 * 1.15:.2f} m clear of the body's side here; nearer the hand, "
                   "where the body may be wider, read the slice table before trusting it.")


def short_legs(rows: list, ground: float, height: float) -> list[str]:
    """When the legs are short for the model: the recipe sizes its gait, bounce and crouch in leg lengths, so on a
    short-legged model they come to a few millimetres, and its BAND is a large share of the leg. All three builders
    of one cat (legs 0.19 m of 1.19 m) found their walk too small to read and worked out why by hand."""
    legs = [s for s in stretches(rows, ground) if s["on_ground"] and s["count"] >= 2]
    if not legs or not height:
        return []
    leg = max(s["top"] - s["bottom"] for s in legs)
    if leg >= SHORT_LEG * height:
        return []
    return [f"- THE LEGS ARE SHORT: they stand apart for {leg:.2f} m of a {height:.2f} m model ({leg / height:.0%}). "
            "The recipe's lift, bounce and hop are in leg lengths (LEG), so here they come to a few millimetres and "
            f"a walk barely reads: scale them up by about {min(2.0, SHORT_LEG * height / leg):.1f} (the recipe's "
            "GAIT_SCALE), and keep BAND, the blend across a joint, under about "
            f"{leg / 5:.2f} m (a fifth of the leg), not the recipe's 0.06. Do not scale the stride: it is in leg "
            "lengths too, and half a stride past the leg's reach makes `legs` drop the hips to reach it (a builder's "
            "thighs went level, and the build reported only stretch)."]


def landmarks(rows: list, ground: float) -> list[str]:
    """What the slice table says about where parts leave the body, in sentences: each stretch of height over which
    the surface is two or more separate runs, where it becomes so, and each pinch of a single run. A builder given
    the table alone put a short-legged model's hips at its belt, as on a person, 0.26 m above where its legs begin;
    the heights are said outright so they are read and not assumed."""
    out = []
    for s in stretches(rows, ground):
        count, top, bottom, middle = s["count"], s["top"], s["bottom"], s["middle"]
        # with the middle of each in x AND y: a limb's line was put at y = 0 by a builder who read only the x
        # extents, 0.085 m off the middle of an arm, and its region then missed the back half of the arm
        parts = "; ".join(f"x {a:+.2f}..{b:+.2f}, y {c:+.2f}..{d:+.2f}, its middle at x {(a + b) / 2:+.2f} y {(c + d) / 2:+.2f}"
                          for a, b, c, d in middle)
        how = ("They reach the ground: the model stands on them" if s["on_ground"] else "They end above the ground")
        if not s["leaves"]:
            where = f"they join the body at z = {bottom:+.2f}"
        elif s["joined"] is None:
            # also where the cut above has MORE runs: legs that begin at the height the hands hang to follow
            # three runs (arm, body, arm) and still leave the body here
            where = (f"they leave the body at z = {top:+.2f}, so the joint each one turns about is at about that "
                     "height, a little above it at most")
        else:
            joined, core = s["joined"], s["core"]
            where = (f"they hang clear of the body below z = {top:+.2f}, and are joined to it from z = "
                     f"{joined:+.2f} down (the single run there is wider than the middle part's {core:.2f} m): "
                     f"they slope outward, and the joint each one turns about is near the top of that "
                     f"stretch, about z = {joined:+.2f}, {core / 2:.2f} m out from the middle")
        if not s["leaves"]:
            # parts that rise from the body to the top (ears, horns, a crest): read by a builder as limbs with
            # "room for joints", they are not what the arms and legs are
            room = ("They rise from the body to the top of the model: ears, horns or a crest, which ride the head "
                    "unless the contract asks for a bone of their own.")
        else:
            room = f"A part {s['begins'] - bottom:.2f} m long has room for joints only inside that length."
        across = thickness(s)
        out.append(f"- from z = {top:+.2f} down to z = {bottom:+.2f} the surface is {count} separate parts side by "
                   f"side ({parts}). {how}; {where}. {room}" + (f" {across}" if across else ""))
    single = [(z, runs[0]) for z, runs in rows if len(runs) == 1]
    for j in range(1, len(single) - 1):
        (z, (a, b, _, _)), before, after = single[j], single[j - 1][1], single[j + 1][1]
        width = b - a
        above = max(r[1][1] - r[1][0] for r in single[:j])
        below = max(r[1][1] - r[1][0] for r in single[j + 1:])
        if width <= before[1] - before[0] and width <= after[1] - after[0] and width < 0.85 * min(above, below):
            out.append(f"- at z = {z:+.2f} the surface pinches to {width:.2f} m across, with wider parts above "
                       f"({above:.2f} m) and below ({below:.2f} m): a neck or a waist, where what is above turns on "
                       "what is below.")
    return out


def named_alike(nodes: list[dict]) -> list[list[dict]]:
    """The nodes in groups that share a name but for a numbered suffix (`wrap`, `wrap.001`), largest group first.
    A model built of many pieces names them so, and a builder shown the forty largest of eighty-one never saw the
    sword it was asked to keep whole."""
    groups: dict[str, list[dict]] = {}
    for n in nodes:
        groups.setdefault(re.sub(r"\.\d+$", "", n["name"]), []).append(n)
    size = lambda g: sum(p["vertices"] for n in g for p in n["pieces"])
    return sorted(groups.values(), key=lambda g: -size(g))


def facts(source: str | Path) -> str:
    """The model in Blender metres, as the script's `skeleton(model)` and `weights(name, position, piece)` see it."""
    doc, blob = glb.read_glb(source)
    model, _ = rig.model_data(doc, blob)
    lo, hi = np.array(model["lo"]), np.array(model["hi"])
    stands = (f"- IT STANDS ON z = {lo[2]:+.3f}, NOT on z = 0: the ground is the lowest point of the rest pose, and the "
              f"origin is {abs(lo[2]):.2f} m {'above' if lo[2] < 0 else 'below'} it." if abs(lo[2]) >= 0.0005 else
              "- IT STANDS ON z = +0.000: the ground is the lowest point of the rest pose, and here that is the "
              "origin's own height.")
    lines = ["THE MODEL (Blender metres: +Z up, +Y is the model's front, X<0 is the model's own left). It is frozen: "
             "a rig adds bones, weights and clips and changes no vertex.",
             f"- bounds: lo {_v(lo)}, hi {_v(hi)}; size {_v(hi - lo)}",
             f"{stands} Its top is z = {hi[2]:+.3f}.",
             f"- {len(model['nodes'])} mesh nodes. `weights` is called once for each (per material of it), with that "
             "node's name. A piece is a set of triangles joined through shared positions:"]
    count = lambda n: sum(p["vertices"] for p in n["pieces"])
    groups = named_alike(model["nodes"])
    for g in groups[:MAX_NODES]:
        if len(g) == 1:
            n = g[0]
            lines.append(f"  - {n['name']}: {count(n)} vertices, {len(n['pieces'])} piece{'s' if len(n['pieces']) != 1 else ''}, "
                         f"lo {_v(n['lo'])} hi {_v(n['hi'])}, centre {_v(n['centroid'])}")
            if 1 < len(n["pieces"]) <= MAX_PIECES:
                # which piece is which: a builder lost two builds to an arm it took for one capsule, which was a
                # forearm-and-hand piece and an upper arm fused into the torso's
                lines.append("      its pieces, largest first (`weights` is given each vertex's piece id):")
                for q in sorted(n["pieces"], key=lambda q: -q["vertices"]):
                    lines.append(f"      piece {q['id']}: {q['vertices']} vertices, lo {_v(q['lo'])} hi {_v(q['hi'])}")
                aside = [q for q in n["pieces"] if q["hi"][0] < -SIDE_M or q["lo"][0] > SIDE_M]
                if aside:
                    # a builder weighted arms built as their own pieces by distance to a line for four builds, and
                    # the shoulder stretched every time
                    lines.append("      pieces wholly to one side of the middle (" + ", ".join(
                        f"{q['id']} at x {'<' if q['hi'][0] < 0 else '>'} 0" for q in aside) + ") may be limbs built "
                        "as their own: weight such a limb by its piece id (the recipe's ARM_PIECES), not by distance")
        else:
            names = sorted(n["name"] for n in g)
            said = ", ".join(names) if len(names) <= 3 else f"{names[0]}, {names[1]} ... {names[-1]}"
            lines.append(f"  - {len(g)} nodes named alike ({said}): {sum(count(n) for n in g)} vertices in all, "
                         f"lo {_v(np.min([n['lo'] for n in g], axis=0))} hi {_v(np.max([n['hi'] for n in g], axis=0))}")
    if len(groups) > MAX_NODES:
        lines.append(f"  - ...and {sum(len(g) for g in groups[MAX_NODES:])} smaller nodes")
    nodes = sorted(model["nodes"], key=lambda n: -count(n))
    # one skin with a few things worn on it is read through the skin; a model built of many pieces, none of them
    # most of it, has no such node, and the largest (a hood, of one model here) says nothing of its legs
    whole = count(nodes[0]) < BULK * sum(count(n) for n in nodes)
    name = None if whole else nodes[0]["name"]
    through = ("THE WHOLE MODEL (it is in pieces and no node is most of it, so this is every node together)" if whole
               else f"`{name}` (the node with the most vertices)")
    lines += ["", f"SLICES THROUGH {through}, top to bottom, one every "
                  f"{SLICE_M * 100:.0f} cm. Each run is one part crossing that height: its X extent, then its Y "
                  "extent. Where a slice has two or three runs the parts are separate there (two legs; an arm "
                  "hanging clear of the flank); where it has one, they are joined, and a joint between them sits "
                  "where the runs merge. Put a limb's bone down the middle of its run, and its joint where the "
                  "run meets the body:"]
    rows = slices(triangles(doc, blob, name))
    for z, runs in rows:
        lines.append(f"  z {z:+.2f}: " + "  |  ".join(f"x {a:+.2f}..{b:+.2f} (y {c:+.2f}..{d:+.2f})" for a, b, c, d in runs))
    marks = landmarks(rows, float(lo[2])) + short_legs(rows, float(lo[2]), float(hi[2] - lo[2]))
    if marks:
        lines += ["", "WHAT THE SLICES SAY (go by these heights, not by where a person's joints would be: a "
                      "character's proportions are its own, and a hip placed at the belt of a short-legged one is in "
                      "the middle of its belly):"] + marks
    return "\n".join(lines)


RULES = """THE RIG SCRIPT (rig.py). It returns data; the harness builds the skin and the clips. Imports: only `math` and
`numpy`. Define all three top-level functions:

    import math
    import numpy as np

    def skeleton(model):
        # model: {"lo", "hi", "nodes": [{"name", "lo", "hi", "centroid", "pieces": [{"id", "lo", "hi", "centroid", "vertices"}]}]}
        # -> a list of bones, PARENT FIRST, exactly one with parent None. head is the joint the bone turns about.
        return [dict(name="root", head=(0, 0, 0), tail=(0, 0, .1), parent=None),
                dict(name="body", head=(0, 0, .4), tail=(0, 0, .6), parent="root")]

    def weights(name, position, piece):
        # name: the node. position: (N, 3) array of that node's vertices, Blender metres, read-only. piece: (N,) ids.
        # -> one bone's name for the whole node, or {bone: array of N weights >= 0}. Every vertex needs a weight > 0
        #    on some bone; the harness keeps the largest four and normalises them. No fallback is applied.
        return "body"

    WORN = {"Coat": "Body"}     # optional: {node worn: node it lies on}. `weights` is not called for a worn node;
                                # each of its vertices takes the weights of the nearest vertex of what is under it

    def pose(clip, t):
        # t in seconds, from 0 to the clip's length. -> {bone: {"rotate": [(axis, degrees), ...], "move": (x, y, z)}}
        # Bones it leaves out stay at rest. A bone may also be given just the list of rotations.
        return {"body": {"rotate": [("X", 5 * math.sin(2 * math.pi * t))]},
                "root": {"move": (0, 0, .02 * abs(math.sin(2 * math.pi * t)))}}

ROTATIONS, worked through once. A rotation ("X", 30) turns the bone, and every bone below it, 30 degrees about the
model's X axis through the bone's HEAD. The axes are the model's own at rest (X across, Y front, Z up), carried
along by the parent's turn; there is no bone roll and no bone-local axis to think about. The sense is right-handed:
- ("X", +a): what hangs BELOW the joint swings FORWARD (toward +Y), what stands ABOVE it tips BACK (toward -Y). So a
  leg whose thigh is given ("X", +30) swings its foot FORWARD, and a spine given ("X", -10) leans the chest forward.
- ("Y", +a): what hangs below the joint swings toward -X (the model's own left); what stands above tips toward +X.
- ("Z", +a): turns the front (+Y) toward -X (the model's own left), seen from above counter-clockwise.
A knee that bends the natural way (the foot going back and up behind the thigh) is ("X", -a) on the shin; an elbow
that bends the forearm forward and up is ("X", +a) on the forearm.
Several rotations in one list apply in the order given, each about the axes as the PARENT holds them, not about
axes the earlier rotations in the list have turned: [("X", 90), ("Z", 90)] swings a hanging part forward and then
turns that about the upright Z, so it ends pointing toward the model's own left. `move` is an offset in metres in
the parent's rest-aligned frame; to move the whole model (a bounce, a sway) move the root.

A BEND STRETCHES ITS OUTSIDE. Where a limb of radius r turns by an angle a (in radians) and the weights blend from
one bone to the next over a band of length h along the limb, the outside of the bend is stretched about 1 + r*a/h
times and the inside is folded. A thick, short limb bent far needs a long band: under 1.5x takes h >= 2*r*a (a limb
0.08 m in radius bent 45 degrees: 0.13 m). When the band cannot be that long, put the joint's head toward the side
that opens (the kneecap for a knee, the back of the arm for an elbow), or bend it less.

LOOPS: a looping clip of length L must give the same pose at t = 0 and t = L. Write every angle as a function of
the phase p = 2 * math.pi * t / L (sin and cos of p, 2p, ...) and it loops by construction.

WEIGHTS are tests on where a vertex is, not on which node it belongs to: two layers at the same place (a vest over a
chest, a belt over hips) then get the same weights and move together. Give every node the same position tests as the
skin under it unless it is a rigid thing that rides one bone (an eye, a buckle, a sword), which returns that
bone's name: for every node of it, when it is made of several, or it bends where a region's edge crosses it.
Blend across a joint over a few centimetres (np.clip of a distance, 0 to 1) so the surface bends instead of tearing.
A limb's test has to exclude the body beside it: a test on one coordinate alone ("x > 0.25 is the arm") also catches
whatever else is out there (a hip, a foot) and drags it along."""


# What three builders with a shell arrived at for one skin with limbs (docs/KNOWN_GAPS.md), and what the first
# API runs did not find in sixteen builds. The code is held to a figure it has to keep on the ground
# (`tests/test_rig_tools.py`), so it is not an illustration.
RECIPE_CODE = """
# YOUR NUMBERS go above this code, read off the slices and the ruled pictures. Metres; an x is the model's right
# side (X>0) and is mirrored for the left.
#   HIP_Z, KNEE_Z, ANKLE_Z      the heights of the hip, knee and ankle joints
#   LEG_X, LEG_Y, LEG_RADIUS    the leg's own upright line, and a little over half the leg's thickness
#   FOOT_FRONT, FOOT_BACK       how far the toes reach in front of that line, and the heel behind it
#   SHOULDER, HAND, ARM_RADIUS  the two ends of the arm's middle line as (x, y, z), and a little over half its thickness
#   ELBOW_Z, NECK_Z, WAIST_Z    the heights of the elbow, of the pinch under the head, and of the waist

BAND = 0.06            # half the length of a blend ACROSS A JOINT, along the limb; lengthen it where a joint stretches
EDGE = 0.02            # how soft the SIDE of a limb's region is; keep it short, or the region reaches the body beside it
WORN = {}              # {node worn: node it lies on}, e.g. {"Coat": "Body"}: each vertex of a worn node takes the
                       # weights of the nearest vertex of what is under it, and `weights` is not called for it
try:
    TAIL               # optional, with YOUR NUMBERS: ((x, y, z) where it leaves the body, (x, y, z) further along
except NameError:      # ... its tip, a little over half its thickness): a tail of a bone a stretch, `tail` first,
    TAIL = None        # then tail.001 ..., each weighted round its own stretch and swung later than the one before
try:
    ARM_PIECES         # optional: {".L": (node, piece id), ".R": (node, piece id)} for arms built as pieces of
except NameError:      # their own (rig-info lists each node's pieces): weighted by piece, not by distance to a line
    ARM_PIECES = None
try:
    GAIT_SCALE         # optional: how many times larger the clips' lifts, bounce and hop are than the numbers below
except NameError:      # say, in leg lengths: on short legs (rig-info says when) they come to a few mm; not the stride
    GAIT_SCALE = 1.0
TAIL_BONES = ["tail"] + [f"tail.{k:03d}" for k in range(1, len(TAIL) - 2)] if TAIL else []
HIPS = (0.0, LEG_Y, HIP_Z)                                             # the hips turn about the point between the hip joints
THIGH_LEN, SHIN_LEN = HIP_Z - KNEE_Z, KNEE_Z - ANKLE_Z
LEG = THIGH_LEN + SHIN_LEN

def skeleton(model):
    ground, top = model["lo"][2], model["hi"][2]
    bones = [dict(name="root", head=(0, 0, ground), tail=(0, 0, ground + 0.1), parent=None),
             dict(name="hips", head=HIPS, tail=(0, LEG_Y, WAIST_Z), parent="root"),
             dict(name="spine", head=(0, LEG_Y, WAIST_Z), tail=(0, LEG_Y, NECK_Z), parent="hips"),
             dict(name="head", head=(0, LEG_Y, NECK_Z), tail=(0, LEG_Y, top), parent="spine")]
    for side, s in ((".L", -1.0), (".R", 1.0)):                        # .L is the model's own left, X<0
        shoulder, hand = np.array([s * SHOULDER[0], SHOULDER[1], SHOULDER[2]]), np.array([s * HAND[0], HAND[1], HAND[2]])
        elbow = shoulder + (hand - shoulder) * (SHOULDER[2] - ELBOW_Z) / (SHOULDER[2] - HAND[2])
        x = s * LEG_X
        bones += [dict(name="upper_arm" + side, head=tuple(shoulder), tail=tuple(elbow), parent="spine"),
                  dict(name="forearm" + side, head=tuple(elbow), tail=tuple(hand), parent="upper_arm" + side),
                  dict(name="thigh" + side, head=(x, LEG_Y, HIP_Z), tail=(x, LEG_Y, KNEE_Z), parent="hips"),
                  dict(name="shin" + side, head=(x, LEG_Y, KNEE_Z), tail=(x, LEG_Y, ANKLE_Z), parent="thigh" + side),
                  dict(name="foot" + side, head=(x, LEG_Y, ANKLE_Z), tail=(x, LEG_Y + FOOT_FRONT, ground), parent="shin" + side)]
    for k, bone in enumerate(TAIL_BONES):
        bones.append(dict(name=bone, head=tuple(TAIL[k]), tail=tuple(TAIL[k + 1]),
                          parent="hips" if k == 0 else TAIL_BONES[k - 1]))
    return bones                                                       # add what else the contract asks for (ears)

def ramp(v, a, b):     # 0 where v is at a or before it, 1 where it is at b or beyond, smooth between
    u = np.clip((v - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)

def chain(position, points, bones):
    # A chain of bones along a bending line (a tail): each vertex goes to the bone of the stretch its nearest point
    # on the line lies on, blended over BAND either side of each joint, as the limb's own split (tip first). Capsules
    # round each stretch overlap at a bend and pulled its outer side two ways; a run of a curled tail stretched 5.8%.
    pts = np.asarray(points, float)
    seg = pts[1:] - pts[:-1]
    length = np.linalg.norm(seg, axis=1)
    start = np.concatenate([[0.0], np.cumsum(length)])
    best, along = np.full(len(position), np.inf), np.zeros(len(position))
    for k in range(len(seg)):
        t = np.clip((position - pts[k]) @ seg[k] / max(length[k] ** 2, 1e-12), 0.0, 1.0)
        d = np.linalg.norm(position - (pts[k] + t[:, None] * seg[k]), axis=1)
        closer = d < best
        best, along = np.where(closer, d, best), np.where(closer, start[k] + t * length[k], along)
    parts = []
    for k in range(len(bones) - 1, 0, -1):                             # tip first: each takes what lies beyond its joint
        parts.append((bones[k], ramp(along, start[k] - BAND, start[k] + BAND)))
    return parts + [(bones[0], 1.0)]

def near(position, a, b, radius, edge=EDGE):
    # A limb's region: 1 within `radius` of the line from a (where it joins what it hangs from) to b (its far
    # end), 0 beyond radius + edge to the side. Along the line it fades in over BAND either side of a: that is
    # the joint's blend, and it is long. To the side it stops short (EDGE), so a limb hanging beside the body
    # takes none of the body.
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(b - a)
    along = (position - a) @ (b - a) / length                          # metres from a toward b
    on_line = a + np.clip(along, -BAND, length)[:, None] * (b - a) / length
    d = np.linalg.norm(position - on_line, axis=1)
    return ramp(d, radius + edge, radius) * ramp(along, -BAND, BAND)

def split(shares):
    # in order -> {bone: weight}: (bone, share of what is left), or (a limb's share of what is left, [(bone, share
    # of what is left OF THE LIMB)], its far end first). A limb takes its whole weight first and its bones divide
    # that between them, so the edge of a forearm is forearm and trunk, not upper arm too; the last bone named
    # takes all that remains
    out, left = {}, 1.0
    for bone, share in shares:
        if isinstance(share, list):
            take = left * bone
            for part, cut in share:
                out[part] = out.get(part, 0.0) + take * cut
                take = take * (1.0 - cut)
            left = left - left * bone
        else:
            out[bone] = out.get(bone, 0.0) + left * share
            left = left * (1.0 - share)
    return out

def weights(name, position, piece):
    x, z = position[:, 0], position[:, 2]
    shares = []
    for side, s in ((".L", -1.0), (".R", 1.0)):
        mine = ramp(s * x, -0.01, 0.01)                                # this side of the middle only
        arm = mine * near(position, (s * SHOULDER[0], SHOULDER[1], SHOULDER[2]), (s * HAND[0], HAND[1], HAND[2]), ARM_RADIUS)
        if ARM_PIECES and side in ARM_PIECES:                         # an arm built as its own piece is that piece
            node, which = ARM_PIECES[side]
            arm = (piece == which).astype(float) if name == node else np.zeros(len(x))
        # the leg's own column, from above the hip down to the ankle
        top, ankle = (s * LEG_X, LEG_Y, HIP_Z + 4 * BAND), (s * LEG_X, LEG_Y, ANKLE_Z)
        # a foot sticks out from under the shin: it is all that lies below the ankle, AND below the knee whatever
        # is outside the shin's column (the instep, the toes, the heel). Cut by height alone, the top of the instep
        # goes to the shin, and a bent knee then drives the toes through the ground
        foot = np.maximum(ramp(z, ANKLE_Z + BAND / 2, ANKLE_Z - BAND / 2),
                          (1 - near(position, top, ankle, LEG_RADIUS)) * ramp(z, KNEE_Z, KNEE_Z - BAND))
        # a leg is what lies below the hip joint AND round the leg's own line, or is foot: a belly or a rump that
        # hangs over the legs is at their height and is not leg, and cut by height alone it swings with the thigh
        leg = mine * ramp(z, HIP_Z + BAND, HIP_Z - BAND) * np.maximum(near(position, top, ankle, LEG_RADIUS, BAND), foot)
        shares += [(leg, [("foot" + side, foot),                                      # the far end first
                          ("shin" + side, ramp(z, KNEE_Z + BAND, KNEE_Z - BAND)),
                          ("thigh" + side, 1.0)]),
                   (arm, [("forearm" + side, ramp(z, ELBOW_Z + BAND, ELBOW_Z - BAND)),
                          ("upper_arm" + side, 1.0)])]
    if TAIL:                                                           # before the trunk: a tail is not the rump
        whole = np.max([near(position, TAIL[k], TAIL[k + 1], TAIL[-1]) for k in range(len(TAIL_BONES))], axis=0)
        shares += [(whole, chain(position, TAIL[:-1], TAIL_BONES))]
    shares += [("head", ramp(z, NECK_Z - BAND, NECK_Z + BAND)),
               ("spine", ramp(z, WAIST_Z - 2 * BAND, WAIST_Z + 2 * BAND)),
               ("hips", np.ones(len(x)))]                              # whatever is left
    return split(shares)

def rx(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def ry(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rz(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def turns(m):          # a rotation matrix as the list a bone takes
    return [("X", math.degrees(math.atan2(m[2, 1], m[2, 2]))),
            ("Y", math.degrees(-math.asin(max(-1.0, min(1.0, m[2, 0]))))),
            ("Z", math.degrees(math.atan2(m[1, 0], m[0, 0])))]

def legs(out, move, turn, feet):
    # Put each ankle where `feet` says and solve its leg, whatever the hips do. move: the hips from rest, (x, y, z)
    # metres. turn: the hips' (pitch, roll, yaw) in degrees. feet: {".L": (forward, up, tip), ".R": ...}, each
    # ankle's offset from rest in metres and the foot's pitch in degrees (+ lifts the toes).
    pitch, roll, yaw = turn
    hips = rz(yaw) @ ry(roll) @ rx(pitch)
    reach = (THIGH_LEN + SHIN_LEN) * 0.998
    hip, ankle = {}, {}
    for side, s in ((".L", -1.0), (".R", 1.0)):
        hip[side] = np.array(HIPS) + move + hips @ (np.array([s * LEG_X, LEG_Y, HIP_Z]) - HIPS)
        ankle[side] = np.array([s * LEG_X, LEG_Y + feet[side][0], ANKLE_Z + feet[side][1]])
    # the hips come down wherever a leg would be too short to reach its foot
    drop = max(0.0, max(hip[k][2] - ankle[k][2] - math.sqrt(max(reach ** 2 - np.sum((ankle[k] - hip[k])[:2] ** 2), 0.0))
                        for k in hip))
    out["hips"] = {"move": (move[0], move[1], move[2] - drop), "rotate": [("X", pitch), ("Y", roll), ("Z", yaw)]}
    for side in hip:
        d = hips.T @ (ankle[side] - hip[side] + (0, 0, drop))             # hip to ankle, as the hips hold it
        far = min(float(np.linalg.norm(d)), reach)
        bend = math.acos(max(-1.0, min(1.0, (far ** 2 - THIGH_LEN ** 2 - SHIN_LEN ** 2) / (2 * THIGH_LEN * SHIN_LEN))))
        # where the ankle is from the hip with the knee bent and the thigh at rest, and the swing that aims it
        swing = (math.atan2(-math.hypot(d[0], d[2]), d[1])
                 - math.atan2(-THIGH_LEN - SHIN_LEN * math.cos(bend), -SHIN_LEN * math.sin(bend)))
        splay = math.atan2(-d[0], -d[2])
        thigh = ry(math.degrees(splay)) @ rx(math.degrees(swing))
        out["thigh" + side] = [("X", math.degrees(swing)), ("Y", math.degrees(splay))]
        out["shin" + side] = [("X", -math.degrees(bend))]                 # the knee bends the natural way
        out["foot" + side] = turns((hips @ thigh @ rx(-math.degrees(bend))).T @ rx(feet[side][2]))

def step(u, g):
    # One foot through a cycle, u from 0 (it lands, ahead) to 1: (forward, up, tip).
    duty, stride = g["duty"], g["stride"]
    if u < duty:                                   # on the ground, carried back under the body
        return stride * (0.5 - u / duty), 0.0, 0.0
    v = (u - duty) / (1 - duty)                    # in the air, 0 to 1
    m = -0.5 * stride * (1 - duty) / duty          # it leaves and lands with some of the ground's speed
    forward = stride * (3 * v ** 2 - 2 * v ** 3 - 0.5) + m * (2 * v ** 3 - 3 * v ** 2 + v)
    up = g["lift"] * math.sin(math.pi * v) ** 2
    # toes down as it leaves, up as it lands, never by more than its height above the ground allows
    tip = -g["toe"] * math.sin(2 * math.pi * v)
    most = math.degrees(math.asin(min(1.0, up / (FOOT_FRONT if tip < 0 else FOOT_BACK))))
    return forward, up, max(-most, min(most, tip))

def gait(p, g):
    # One cycle of a walk or a run. p: the phase, 2 * math.pi * t / L. g: the clip's numbers (see WALK and RUN).
    out = {}
    u = (p / (2 * math.pi)) % 1.0
    feet = {".L": step(u, g), ".R": step((u + 0.5) % 1.0, g)}
    mid = g["duty"] / 2                                               # the left foot is under the body
    if g["duty"] < 0.5:                                               # a run: both feet are off the ground between steps
        gap = 0.5 - g["duty"]
        away = abs((u - g["duty"] - gap / 2 + 0.25) % 0.5 - 0.25)
        hop = g["air"] * math.cos(math.pi / 2 * min(1.0, away / (gap / 2))) ** 2
        feet = {k: (f, up + hop, tip) for k, (f, up, tip) in feet.items()}
    over_left = math.cos(2 * math.pi * (u - mid))                     # +1 with the weight over the left foot, -1 the right
    lead = math.cos(2 * math.pi * u)                                  # +1 with the left foot furthest forward
    height = -g["crouch"] + g["bounce"] * math.cos(4 * math.pi * (u - mid))
    roll = g["roll"] * over_left                                      # the hip of the leg in the air drops
    legs(out, (-g["sway"] * over_left, 0.0, height), (0.0, roll, -g["yaw"] * lead), feet)
    out["spine"] = [("X", -g["lean"]), ("Y", -roll), ("Z", 1.5 * g["yaw"] * lead)]      # the shoulders turn against the hips
    out["head"] = [("X", 0.6 * g["lean"]), ("Z", -0.5 * g["yaw"] * lead)]               # and the head looks ahead
    for side, s in ((".L", -1.0), (".R", 1.0)):
        ahead = s * lead                                              # an arm goes forward with the other side's leg
        out["upper_arm" + side] = [("Y", s * g["close"]), ("X", g["arm"] * ahead)]
        out["forearm" + side] = [("X", g["elbow"] + g["elbow_swing"] * ahead)]
    return out

def idle(p, g):
    # A living stand: the weight goes from one foot to the other and back while the feet stay where they are.
    out = {}
    over_right = math.sin(p)
    breath = math.sin(p - 1.0)
    planted = (0.0, 0.0, 0.0)
    legs(out, (g["shift"] * over_right, 0.0, -g["crouch"] - 0.002 * breath), (0.0, -g["roll"] * over_right, g["yaw"] * math.sin(p + 2.0)),
         {".L": planted, ".R": planted})
    out["spine"] = [("X", g["breath"] * breath), ("Y", 1.5 * g["roll"] * over_right), ("Z", -g["yaw"] * math.sin(p + 2.0))]
    out["head"] = [("X", -0.8 * g["breath"] * breath + g["nod"] * math.sin(2 * p)), ("Y", -0.5 * g["roll"] * over_right),
                   ("Z", g["look"] * math.sin(p + 0.8))]
    for side, s in ((".L", -1.0), (".R", 1.0)):
        out["upper_arm" + side] = [("Y", s * g["arms"] * (1 - breath)), ("X", g["arms"] * math.sin(p + 1.5 + 0.5 * s))]
        out["forearm" + side] = [("X", 2 * g["arms"] + g["arms"] * math.sin(p + 2.5 + 0.5 * s))]
    return out

# A clip's numbers: lengths in leg lengths, sideways ones in LEG_X (the whole of it is the hips right over one
# foot), angles in degrees. These are a start that walks and runs; tune them.
WALK = dict(duty=0.60,                      # the share of a cycle a foot is on the ground: over 0.5 walks, under it runs
            stride=0.8 * LEG,               # how far a foot is carried back on the ground
            lift=0.15 * LEG, toe=10,        # how high the ankle goes in the air, and how far the foot tips
            crouch=0.03 * LEG, bounce=0.0,  # the hips' height: lower by crouch; bounce + is highest over the standing
                                            # foot (a walk), - is highest between steps (a run)
            sway=0.15 * LEG_X, roll=2.5, yaw=8,   # the hips go toward the standing foot, drop on the other side, turn with the leading leg
            lean=2, arm=25, close=0, elbow=15, elbow_swing=8, air=0.0)
RUN = dict(duty=0.40, stride=1.1 * LEG, lift=0.22 * LEG, toe=15, crouch=0.08 * LEG, bounce=-0.05 * LEG,
           sway=0.08 * LEG_X, roll=1.5, yaw=11, lean=12, arm=42, close=0, elbow=70, elbow_swing=10,
           air=0.10 * LEG)                  # how high both feet are off the ground between steps
IDLE = dict(shift=0.2 * LEG_X, crouch=0.02 * LEG, roll=2.0, yaw=1.5, breath=1.5, nod=1.0, look=5, arms=3)
for g in (WALK, RUN):
    for k in ("lift", "bounce", "air"):          # up and down only: a stride is in leg lengths already, and one past
                                                # the leg's reach drops the hips (a builder's thighs went level)
        g[k] *= GAIT_SCALE
SECONDS = {"walk": 1.0, "run": 0.6, "idle": 2.0}                       # each clip's length, from the contract

def pose(clip, t):
    p = 2 * math.pi * t / SECONDS[clip]
    out = idle(p, IDLE) if clip == "idle" else gait(p, WALK if clip == "walk" else RUN)
    if TAIL:           # side to side, wider when it runs, and a lift at twice the rate, late on the hips' motion,
        sway = (6 if clip == "idle" else 12 if clip == "walk" else 16) / len(TAIL_BONES) ** 0.5
        for k, bone in enumerate(TAIL_BONES):                          # each stretch later than the one before
            out[bone] = [("Z", sway * math.sin(p - 0.8 - 0.6 * k)), ("X", -0.4 * sway * math.sin(2 * p - 0.8 - 0.6 * k))]
    # add what else the model has beyond arms and legs (ears, a cloak): out["ear.L"] = [("Y", 8 * math.sin(p))]
    return out
"""

RECIPE = """A RECIPE FOR A BODY WITH TWO ARMS AND TWO LEGS, one skin or many pieces. It is a whole rig script but for
your numbers, and it is what the rigs that worked here did. Start from it as it stands; the numbers are yours to
read off the slices and the pictures. `rig-info --source ... --contract ... --write-template rig.py` writes it as a
script with each number a placeholder that names itself when a build reaches it.

WEIGHTS BY REGION, NEVER BY DISTANCE TO THE BONES. Cut the surface into regions by tests on position and hand the
weight out by splitting what is left, limb ends first, the trunk last. Do NOT weight each vertex by its distance to
every bone (1 / distance ** p, or a gaussian of distance, normalised over all the bones): every vertex then holds a
little of every bone, the flank goes up with the arm, the belly follows the legs, and the whole surface stretches.
No tuning of the power or the radius fixes it. A run of this that did so failed the stretch gate sixteen builds
in a row; the weights sheet of such a rig is a muddy gradient, where a good one is flat colours with a narrow blend
at each joint.

PLACE THE FEET AND SOLVE THE LEGS. Do not write a gait as sines on the thigh and the shin: the feet then go where
the angles take them, and the body can only bob over them. `legs` is told where each ankle is to be and works out
the thigh, shin and foot turns that put it there, whatever the hips are doing. `step` is one foot's path through a
cycle: on the ground and carried straight back for `duty` of it, lifted and brought forward for the rest. `gait`
is a walk or a run made of two of them half a cycle apart, and `idle` a stand with both feet where they are. A
foot on the ground then stays on the ground and does not slide while the hips go over it, drop on the other side
and turn with the leading leg, and the shoulders turn against them: that is what makes a body look as if it
carries its weight. The rigs made here from sines on the angles moved fore and aft only, and were read as a
stiff shuffle. Where a leg is too short to reach its foot, `legs` lowers the hips by what is missing, so a stride
too long for the legs shows as a crouch and never as a foot off the ground. This holds only while the whole foot,
toes and instep included, is on the foot bone: if a clip still sinks, the readings name the bone whose part is
lowest, and a `shin` there means the foot's region is too small. A sole that is not flat under the ankle (FOOT_FRONT
or FOOT_BACK too short for it) dips when the foot tips: lower `toe`.

BAND DECIDES THE STRETCH MORE THAN THE MOTION DOES. On the model this recipe was first tried on, with one and the
same run (thighs 38 degrees, knees 70), the surface stretched past 1.5x was 4.5% with BAND = 0.02, 2.7% with 0.04,
2.2% with 0.06 and 1.9% with 0.09; a run a fifth larger still passed a 3% gate from 0.06 up. Lengthen the band
before you shrink a clip.
""" + "\n".join("    " + line for line in RECIPE_CODE.strip().splitlines()) + """

YOUR NUMBERS are constants you define from the model, above the code. HIP_Z is the height at which the legs
become separate parts (WHAT THE SLICES SAY), not the waist: everything the leg bones hold must be leg. KNEE_Z and
ANKLE_Z lie between it and the ground, the knee about half way. NECK_Z is the pinch under the head. SHOULDER is
where an arm's run leaves the body's, HAND its far end: both are points on the arm's own middle line, in x AND y
(WHAT THE SLICES SAY gives each part's middle; an arm's y is rarely 0), and ARM_RADIUS is a little over half the
arm's thickness, so the capsule holds the whole arm and none of the flank. The same goes for the leg's line
(LEG_X, LEG_Y) with LEG_RADIUS: a little over half the leg's own thickness, not the foot's. The line is upright:
where a leg slopes, take the x half way down it. FOOT_FRONT and FOOT_BACK are read off the side view.

WHAT IS WORN: name everything that lies on the body (a vest, a belt, a strap round a limb) in WORN, each against
the node it lies on. It then takes the weights of the surface under it, vertex by vertex, and the two move as
one: the body cannot come through its clothes and the readings' `comes away` line stays empty. Tests on position
do not promise that, because two layers a centimetre apart fall on different sides of a region's edge. Only a
rigid thing on a part that does not bend under it (an eye or a nose on a head) is left out of WORN and has
`weights` return that bone's name; a buckle on a waist that bends is better worn. BUT where an arm is joined to
the flank (WHAT THE SLICES SAY gives how far down) the surface nearest a vest's armhole or a belt's side can be
the arm's, and a worn layer there takes arm weight and tears when the arm swings: three rigs of low-poly cats did.
Then leave the vest, belt and buckle out of WORN and give them `weights` by position with trunk bones only (no
arm bone), as the skin under them has. Further bones a contract asks
for (a tail, ears) get their own region before the trunk's. A model in many pieces has no skin to wear things on:
leave WORN empty there, since every node takes the same tests on position, and give one bone by name to each
rigid thing (every node of a sword and its scabbard to `hips`).

HOW BIG, in leg lengths (LEG, hip to ankle), so the same numbers suit a short-legged body and a long-legged one.
What goes sideways is in LEG_X, half the distance between the feet: 1.0 of it puts the hips right over one foot.
A walk: a stride of 0.7 to 0.9, the ankle lifted 0.12 to 0.2, the hips swaying 0.1 to 0.2 of LEG_X toward the
standing foot and turning 6 to 10 degrees, arms 20 to 30 degrees, almost no lean. A run is another gait and not a quick
walk: `duty` under 0.5 so that both feet leave the ground, a stride of 1.0 to 1.3, the ankle lifted 0.2 to 0.3,
`air` 0.08 to 0.12, a crouch of 0.06 to 0.12, a lean of 10 to 15 degrees, arms 35 to 50 with the elbows bent 60
to 90 throughout. The stride is what makes a run the bigger clip, and it costs the knee nothing. What bends
the knee is how near the ankle comes to the hip: `lift`, `air` and `crouch` together. Where a run stretches at
the knee, lengthen BAND, then take some of those three off, and keep the stride. An idle is a shift of weight, not a nod: the hips 0.15 to 0.3 of LEG_X to each side with the
feet still, a degree or two of breath in the spine, the head turning a few degrees, the arms a little, a tail
more. THE CLIPS BESIDE EACH OTHER shows how far each part goes in each clip."""


def gates(spec: dict) -> str:
    lines = ["THE CONTRACT (rig.json; you cannot change it):",
             f"- required bones (exact names): {', '.join(spec.get('bones', {}).get('required', [])) or 'none'}; at most "
             f"{spec.get('bones', {}).get('max', 64)} bones. You may add others.",
             f"- {spec.get('fps', 30)} keys a second, at most {spec.get('max_influences', 4)} bones a vertex"
             + ("; every connected piece must follow ONE bone at full weight (a rigid model)" if spec.get("rigid") else "")]
    for name, c in spec["clips"].items():
        said = [f"{c['seconds']:g} s", "loops" if c.get("loop") else "plays once"]
        if c.get("loop"):
            said.append(f"last key within {spec.get('loop_mm_max', 2):g} mm of the first")
        if "min_displacement_mm" in c:
            said.append(f"some point moves at least {c['min_displacement_mm']:g} mm from rest")
        if "ground_drop_mm_max" in c:
            said.append(f"nothing goes more than {c['ground_drop_mm_max']:g} mm below the rest pose's ground")
        if "stretch_area_max" in c:
            said.append(f"at most {c['stretch_area_max'] * 100:g}% of the surface in triangles with an edge stretched past 1.5x")
        if "max_edge_ratio" in c:
            said.append(f"no edge stretched past {c['max_edge_ratio']:g}x")
        lines.append(f"- clip `{name}`: " + "; ".join(said))
    return "\n".join(lines)


def text(source: str | Path, spec: dict) -> str:
    # a model whose every piece rides one bone has no blend to get wrong and no use for the recipe
    return "\n\n".join([facts(source), RULES] + ([] if spec.get("rigid") else [RECIPE]) + [gates(spec)])


def contract(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


RECIPE_BONES = {"root", "hips", "spine", "head"} | {f"{b}.{s}" for b in ("upper_arm", "forearm", "thigh", "shin", "foot")
                                                     for s in "LR"}
NUMBERS = {                     # the recipe's own numbers, as its comment names them
    "HIP_Z": "the height of the hip joint", "KNEE_Z": "the height of the knee joint",
    "ANKLE_Z": "the height of the ankle joint", "LEG_X": "the leg's upright line, its x (the model's right, X>0)",
    "LEG_Y": "the leg's upright line, its y", "LEG_RADIUS": "a little over half the leg's own thickness",
    "FOOT_FRONT": "how far the toes reach in front of the leg's line", "FOOT_BACK": "how far the heel reaches behind it",
    "SHOULDER": "the arm's middle line at the shoulder, (x, y, z), the right arm", "HAND": "the same line at the hand",
    "ARM_RADIUS": "a little over half the arm's thickness ACROSS the arm, not the slice's width: a sloped arm's slices "
                  "are wider than the arm", "ELBOW_Z": "the height of the elbow",
    "NECK_Z": "the height of the pinch under the head", "WAIST_Z": "the height of the waist",
}


def template(spec: dict | None, suggested: dict[str, str] | None = None) -> str:
    """The recipe as a rig script to start from (`rig-info --write-template`): every number it needs left as a
    placeholder that says what it is and stops the build until it is set, and the bones the contract requires that
    the recipe does not make named at the top. Builders sliced the recipe out of the brief by hand, and one edit that
    half applied cost a build."""
    required = [b for b in ((spec or {}).get("bones") or {}).get("required", []) if b not in RECIPE_BONES]
    head = ["import math", "", "import numpy as np", "",
            "# A rig script from mesh-jig's recipe (rig-info --write-template). Set every SET_ME below from the slices",
            "# and the ruled pictures (rig-info --out): a build stops on the first one left, naming it.",
            "",
            "def SET_ME(name, what):",
            "    raise ValueError(f'set {name} in rig.py: {what}')",
            ""]
    suggested = suggested or {}
    head += [f"{name} = {suggested[name]}    # read off the slices ({what}): check it against the rest pictures"
             if name in suggested else f'{name} = SET_ME("{name}", "{what}")' for name, what in NUMBERS.items()]
    head += [(f"TAIL = {suggested['TAIL']}    # read off the tail node: from where it meets the body to its tip, "
              "then a radius covering its thickest stretch; n points make n - 1 bones" if "TAIL" in suggested else
              "TAIL = None    # a tail: ((x, y, z) where it leaves the body, (x, y, z) at each bend, ..., (x, y, z) its "
              "tip, then a little over half its thickness, enough to cover its thickest stretch): n points make n - 1 "
              "bones, tail, tail.001, ..."),
             "# A close-up of a knee or an elbow: rig-preview --rig ... --contract ... --views front left --aim X Y Z "
             "--span 0.3 --out DIR",
             (f"GAIT_SCALE = {suggested['GAIT_SCALE']}    # the legs are short for the model (rig-info): the "
              "recipe's lifts, bounce and hop, in leg lengths, scaled up so a walk reads" if "GAIT_SCALE" in suggested
              else "GAIT_SCALE = 1.0    # the clips' lifts, bounce and hop, times this"),
             (f"ARM_PIECES = {suggested['ARM_PIECES']}    # the arms are pieces of their own, weighted by piece: "
              "check them in rig-info's piece list" if "ARM_PIECES" in suggested else
              "ARM_PIECES = None    # arms built as their own pieces: {'.L': (node, piece id), '.R': (node, piece id)}")]
    required = [b for b in required if b != "tail"] + (["tail (set TAIL above)"] if "tail" in required else [])
    if required:
        head += ["", f"# The contract also requires {', '.join(required)}, which the recipe does not make: add each to",
                 "# the skeleton, give it weights by region and move it in the clips (the recipe's comments say where a",
                 "# tail goes)."]
    tail = ""
    if "BAND" in suggested:                         # after the recipe: its functions read BAND when they run
        tail = (f"\n\nBAND = {suggested['BAND']}    # the legs are short (rig-info): a blend across a joint under a fifth "
                "of the leg, not the recipe's 0.06\n")
    return "\n".join(head) + "\n" + RECIPE_CODE + tail


def tail_line(tris: np.ndarray, body: np.ndarray, bones: int = 3) -> str:
    """A tail node's middle line as the recipe's TAIL: points from where it meets the body (its point nearest the
    body's surface, `body` the other nodes' points) out to its tip, each the middle of a stretch of it by distance
    from that root, and a radius covering how far its surface lies from that line. Two builders measured a curled
    tail's line by hand, and one missed it by 0.1 m."""
    pts = tris.reshape(-1, 3)
    body = body.reshape(-1, 3)
    if len(body) > 4000:
        body = body[:: len(body) // 4000 + 1]
    near_body = np.array([np.linalg.norm(body - q, axis=1).min() for q in pts[:: max(1, len(pts) // 800)]])
    root = pts[:: max(1, len(pts) // 800)][near_body.argmin()]
    far = np.linalg.norm(pts - root, axis=1)
    edges = np.linspace(0, far.max() + 1e-9, bones + 2)
    line, radius = [root], 0.0
    for k in range(1, bones + 2):
        ring = pts[(far >= edges[k - 1]) & (far < edges[k])]
        if not len(ring):
            continue
        centre = ring.mean(axis=0)
        radius = max(radius, float(np.median(np.linalg.norm(ring - centre, axis=1))))
        if k > 1:
            line.append(centre)
    line.append(pts[far.argmax()])
    # how far the tail's surface lies from that line, all along it (a ring by distance from the root cuts across a
    # bend and reads it as thick)
    path = np.asarray(line)
    a, b = path[:-1], path[1:]
    ab = b - a
    t = np.clip(((pts[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    off = np.linalg.norm(pts[:, None] - (a + t[..., None] * ab), axis=-1).min(axis=1)
    radius = float(np.percentile(off, 95))
    said = ", ".join("(" + ", ".join(f"{v:.3f}" for v in p) + ")" for p in line)
    return f"({said}, {max(radius * 1.15, 0.02):.3f})"


def pinches(rows: list) -> list[float]:
    """The heights where a single run narrows between wider ones above and below (a neck, a waist), as the slice
    text says them."""
    single = [(z, runs[0]) for z, runs in rows if len(runs) == 1]
    out = []
    for j in range(1, len(single) - 1):
        (z, (a, b, _, _)), before, after = single[j], single[j - 1][1], single[j + 1][1]
        width = b - a
        above = max(r[1][1] - r[1][0] for r in single[:j])
        below = max(r[1][1] - r[1][0] for r in single[j + 1:])
        if width <= before[1] - before[0] and width <= after[1] - after[0] and width < 0.85 * min(above, below):
            out.append(z)
    return out


def suggest(source: str | Path) -> dict[str, str]:
    """The recipe's numbers as the slices give them, each as the Python it is written as, for `--write-template` to
    start from: a builder set fourteen numbers by hand from prose. What the slices cannot give is left out (and stays
    a placeholder): a model with no legs apart has no hip to read."""
    doc, blob = glb.read_glb(source)
    model, _ = rig.model_data(doc, blob)
    count = lambda n: sum(p["vertices"] for p in n["pieces"])          # noqa: E731
    nodes = sorted(model["nodes"], key=lambda n: -count(n))
    name = None if count(nodes[0]) < BULK * sum(count(n) for n in nodes) else nodes[0]["name"]
    rows = slices(triangles(doc, blob, name))
    ground, top = float(model["lo"][2]), float(model["hi"][2])
    found = stretches(rows, ground)
    r = lambda v: round(float(v), 3)                                    # noqa: E731
    out: dict[str, str] = {}
    legs = sorted((s for s in found if s["on_ground"] and s["count"] >= 2), key=lambda s: s["bottom"] - s["top"])
    if legs:
        s = legs[0]
        hip = s["begins"]
        # the leg at x > 0 in each cut of its stretch, bottom up: its own thickness is its narrowest (a foot or a paw
        # widens it at the bottom, and the middle cut of a short leg is in the foot), and the ankle is where the
        # foot's widening stops. Read at the middle cut, three builders were given a radius of the paw's
        runs = [(z, run) for z, cut in reversed(rows) if s["bottom"] - 1e-9 <= z <= s["top"] + 1e-9
                for run in cut if run[0] > 0]
        if runs:
            z_thin, (a, b, c, d) = min(runs, key=lambda zr: zr[1][1] - zr[1][0])
            thick = b - a
            # the top of the foot's widening (a ball of a foot narrows again at its sole, so read from above)
            wide = [z for z, run in runs if run[1] - run[0] > 1.2 * thick and z < z_thin]
            ankle = max(wide) if wide else ground + 0.12 * (hip - ground)
            ankle = min(ankle, hip - 0.04)
            out.update(HIP_Z=f"{r(hip)}", KNEE_Z=f"{r((hip + ankle) / 2)}", ANKLE_Z=f"{r(ankle)}",
                       LEG_X=f"{r((a + b) / 2)}", LEG_Y=f"{r((c + d) / 2)}", LEG_RADIUS=f"{r(0.55 * thick)}")
        sole = [run for run in rows[-1][1] if run[0] > 0] if rows else []
        if sole:
            out.update(FOOT_FRONT=f"{r(max(sole[0][3] - (c + d) / 2, 0.02))}",
                       FOOT_BACK=f"{r(max((c + d) / 2 - sole[0][2], 0.02))}")
    arms = sorted((s for s in found if not s["on_ground"] and s["leaves"] and s["count"] >= 3),
                  key=lambda s: s["bottom"] - s["begins"])
    if arms:
        s = arms[0]
        a, b, c, d = s["middle"][-1]                                    # the arm at x > 0
        joint = s["begins"]
        shoulder_x = s["core"] / 2 if s.get("joined") is not None else (a + b) / 2
        out.update(SHOULDER=f"({r(shoulder_x)}, {r((c + d) / 2)}, {r(joint)})",
                   HAND=f"({r((a + b) / 2)}, {r((c + d) / 2)}, {r(s['bottom'])})",
                   ELBOW_Z=f"{r((joint + s['bottom']) / 2)}", ARM_RADIUS=f"{r(0.55 * max(b - a, d - c))}")
        across = thickness(s)
        m = re.search(r"under (\d\.\d+) m", across) or re.search(r"about (\d\.\d+) m, reaches", across)
        if m:
            out["ARM_RADIUS"] = m.group(1)
        # and clear of the body all the way down, not just at the middle: a radius read there took the hips' weight
        # at the hand on a cat whose belly widens below
        rooms = []
        for z, runs in rows:
            if s["bottom"] <= z <= s["top"]:
                body = [run for run in runs if run[0] <= 0 <= run[1]]
                along = (joint - z) / max(joint - s["bottom"], 1e-9)
                line_x = shoulder_x + along * ((a + b) / 2 - shoulder_x)
                if body:
                    rooms.append(line_x - body[0][1])
        if rooms:
            out["ARM_RADIUS"] = f"{r(max(0.02, min(float(out['ARM_RADIUS']), min(rooms) - 0.04)))}"
        # arms built as pieces of their own, one each side at the arms' height: weighted by piece
        node = next((n for n in model["nodes"] if n["name"] == name), None)
        if node is not None:
            def at_arms(q):
                return q["lo"][2] < joint and q["hi"][2] > s["bottom"]
            left = [q for q in node["pieces"] if q["hi"][0] < -SIDE_M and at_arms(q)]
            right = [q for q in node["pieces"] if q["lo"][0] > SIDE_M and at_arms(q)]
            if len(left) == 1 and len(right) == 1:
                out["ARM_PIECES"] = (f'{{".L": ("{name}", {left[0]["id"]}), ".R": ("{name}", {right[0]["id"]})}}')
                # the arm is that piece: its shoulder at the piece's top, its hand at its bottom. Taken from where
                # the slices merge, the shoulder floated above an arm whose top lobe stays in the body
                q = right[0]
                cx, cy = (q["lo"][0] + q["hi"][0]) / 2, (q["lo"][1] + q["hi"][1]) / 2
                top_z, low_z = q["hi"][2] - 0.01, q["lo"][2] + 0.01
                lobe = joint - top_z > 0.03
                out.update(SHOULDER=f"({r(cx)}, {r(cy)}, {r(top_z)})" + (
                               f"    # the arm piece's top: the body holds a shoulder lobe above it, up to "
                               f"z {r(joint)}, which stays with the spine" if lobe else ""),
                           HAND=f"({r(cx)}, {r(cy)}, {r(low_z)})", ELBOW_Z=f"{r((top_z + low_z) / 2)}")
    above = (float(out["SHOULDER"].split("#")[0].split(", ")[2].strip().rstrip(")")) if "SHOULDER" in out
             else float(out["HIP_Z"]) if "HIP_Z" in out else ground)
    necks = [z for z in pinches(rows) if z > above]
    if necks:                                       # the lowest pinch above the shoulders: under the head
        out["NECK_Z"] = f"{r(min(necks))}"
    short = short_legs(rows, ground, top - ground)
    if short:
        m = re.search(r"scale them up by about (\d+\.\d)", short[0])
        if m:
            out["GAIT_SCALE"] = m.group(1)
        m = re.search(r"under about (\d+\.\d+) m \(a fifth", short[0])
        if m:
            out["BAND"] = m.group(1)
    tail = next((n for n in model["nodes"] if "tail" in n["name"].lower()), None)
    if tail is not None:
        others = [triangles(doc, blob, n["name"]) for n in model["nodes"] if n["name"] != tail["name"]]
        if others:
            out["TAIL"] = tail_line(triangles(doc, blob, tail["name"]), np.concatenate([o.reshape(-1, 3) for o in others]))
    if "HIP_Z" in out and "SHOULDER" in out:
        out["WAIST_Z"] = f"{r((float(out['HIP_Z']) + above) / 2)}"
    return out
