# Frozen mesh rigging

The initial rig stage takes a finished **static GLB**, a trusted declarative Python script and a JSON contract.
It appends skin and animation data without importing and re-exporting the geometry through Blender. Original
position, normal, colour, UV and index accessors remain intact. The modelling renderer and scores are unchanged.

This implements manual blended weights and rigid pieces, and a loop that has an API model write them. It does
not implement a video-reference stage: no video extraction, automatic weights,
motion score, IK, collision certification or animation-aware results viewer. The static viewer still shows the
rest pose.

## Commands

```powershell
mesh-jig rig-build --source source.glb --script rig.py --contract rig.json --out animated.glb
mesh-jig rig-check --source source.glb --rig animated.glb --contract rig.json
mesh-jig rig-preview --rig animated.glb --contract rig.json --clip hop --times 0 .25 .5 .75 1 --out poses
```

Three more are for whoever writes the script, a person or an agent:

```powershell
mesh-jig rig-info --source source.glb --contract rig.json --out rest
mesh-jig rig-preview --rig animated.glb --contract rig.json --views left front three-quarter --out poses
mesh-jig rig-check --source source.glb --rig animated.glb --contract rig.json --readings
```

- `rig-info` prints the model as a script meets it (nodes, connected pieces, bounds, the height it stands on, a
  table of slices by height that shows where legs and arms separate), the script's rules with the sense of each
  rotation, and the contract's gates in words. One skin is sliced through the skin; a model in pieces, where no
  node is half of it, is sliced whole, and its nodes named alike (`wrap`, `wrap.001`) share a line. With `--out` it draws the rest pose from four sides under a grid
  ruled every 0.1 m in Blender metres, so a joint's place is read off a picture.
- `rig-preview --views` draws every clip as a film strip from cameras by name (`front`, `back`, `left`, `right`,
  `top`, `three-quarter`), fitted to the model, and a weights sheet: the rest pose with each point in the blend of
  its bones' colours and the skeleton over it. The contract's `shot` is one camera in glTF axes; these need none.
- `rig-check --readings` adds what no gate reads (`rig_read`): what each bone holds and which hold nothing, stray
  weights (a patch of a bone's weight cut off from its main patch on one connected surface), how far the part each
  bone holds travels in each clip, whether left and right parts take turns or move together, where a clip
  stretches the surface and between which bones, and where the lowest point of the model is against the ground.
  What lies on what is followed too: on one skin each thing worn, and on a model in pieces each pair of nodes
  that touch at rest and part in a clip (no edge joins two nodes, so the stretch gate reads nothing there).
  They are diagnostics: no gate, no score.

A `rig-build` that fails its contract prints the whole report with the numbers and exits 1, and each issue says by
how much (`walk: pose sinks below its rest ground (76.8 mm, 15 allowed)`). With `--failed kept.glb` the build is
left there, not delivered, for `rig-check --readings` and `rig-preview --views` to show what it did.

Use absolute paths when operating from another directory. Preview writes individual CPU poses and a film strip.
Outputs are ordinary glTF 2.0 skins and LINEAR joint animation, usable by Godot. All original embedded materials
and textures stay in the file. Compressed/extension-required, sparse, already skinned or animated sources are
refused. The GLB must use one embedded buffer and triangle primitives.

## An API model writes the rig

```powershell
mesh-jig agent <rig dir> --workflow rig --model claude-haiku-5-5 --url https://api.anthropic.com/v1 --attempts 8 --max-cost 6
```

`<rig dir>` holds `source.glb`, `rig.json` (the contract) and `brief.md` (what the clips are to be, in words: a
contract cannot say that a walk should read as a walk). The model has no shell. It writes
`attempts/<name>/rig.py` with file tools and calls:

- `build`: the script held to the contract. The answer is the verdict with every gate's number, the readings of
  `rig-check --readings`, the weights sheet and a film strip of every clip from the side and the front. A build
  out of contract answers the same for what it built and is not counted. After three in a row it is told to
  change its approach and not its numbers.
- `look`: more frames of one clip at named times and views, with the skeleton or as a close-up (`aim`, `span`).
- `done`: names the attempt to deliver, which is copied to `<rig dir>/animated.glb` beside its `rig.py`. It is
  refused while builds are unused.

It opens with the brief, `rig-info`'s text and the ruled rest pose. `--attempts` is the number of builds in
contract. With `--reviewer`, after each build in contract a fresh call of the same model, shown only the brief
and three-view strips, says what each clip reads as and names up to five faults (kept as the attempt's
`review.md`). It is off by default: on the one model it was measured on, the runs with it came out no better and
took longer. The run's record is `<rig dir>/agent/<stamp>_<model>_rig/report.json`; each attempt keeps its
`report.json` and pictures.
There is no score: nothing ranks one rig attempt against another, and the model chooses what it delivers.

`rig-info` also carries a recipe for a body with two arms and two legs, left out for a `rigid` contract: a
whole script but for the model's numbers. Its weights are by region. Its motion places each foot and solves the
leg for it (`legs`), so a foot on the ground stays there while the hips sway, roll and turn over it, and its
sizes are in leg lengths. A test runs it as given.

## Script contract

Only `math` and `numpy` imports are permitted; the existing denylist remains in force. This is a containment
check, **not a sandbox**; only run scripts you trust. Scripts execute on the host and must finish promptly.
They author data and never call Blender. Define all three top-level functions:

```python
import math
import numpy as np

def skeleton(model):
    return [dict(name="root", head=(0, 0, 0), tail=(0, 0, .1), parent=None),
            dict(name="body", head=(0, 0, .4), tail=(0, 0, .6), parent="root")]

def weights(name, position, piece):
    return "body"  # or {"root": array_of_weights, "body": another_array}

def pose(clip, t):
    return {"body": {"rotate": [("X", 5 * math.sin(2 * math.pi * t))]},
            "root": {"move": (0, 0, .1 * abs(math.sin(2 * math.pi * t)))}}
```

Positions use Blender metres: **+Z up, +Y forward**, preserving the delivered source origin. The model's own
left is X<0 (Blender's habit for a character that faces -Y is the other way), and it stands on its lowest point,
which is not z = 0 for a model centred on its origin. `pose(clip, t)` takes `t` in seconds. A positive rotation
is right-handed: `("X", +a)` swings what hangs below a joint forward and tips what stands above it back.
Rotations in one list apply in order, each about the parent's axes. `rig-info` prints all of this with the model. The `model`
dictionary supplies `lo`, `hi`, and `nodes`: each node has `name`, `lo`, `hi`, `centroid`, and its `pieces`.
Each piece has `id`, `lo`, `hi`, `centroid`, `vertices`. Coincident positions weld at six decimal places, matching
the mesh stage's component diagnostic, including across material primitives inside a node. Intersections do not
fuse. Different nodes are independent pieces even if they touch.

A script may also define a top-level `WORN = {"Vest": "Skin", "Belt": "Skin"}`: each node named takes, vertex by
vertex, the weights of the nearest vertex of the node it lies on, and `weights` is not called for it. Two layers
then move as one. Tests on position do not promise that: a vest a centimetre off a chest fell on the other side
of an arm region's edge and slid 35 mm over it.

Bones must be parent-first with unique names, nonzero lengths and one root. The head specifies the pivot;
the tail records the authored direction but does not introduce bone roll. Joint frames are parallel to the
model axes at rest, carried by their parent. Rotations apply in listed order about those axes, in degrees.
`move` is an offset in the parent's rest-aligned frame, carried by that parent's rotation. To author travel
in world axes, move the root. Missing bone poses are at rest. Weights receive read-only world-space positions
and piece IDs, separately for each material primitive; IDs refer to the complete node. Missing/negative/nonfinite
weights fail. The runner keeps the largest influences deterministically and normalizes them; no nearest-bone
fallback is applied.

```json
{
  "fps": 30,
  "max_influences": 4,
  "bones": {"required": ["root", "body"], "max": 40},
  "rigid": true,
  "loop_mm_max": 2,
  "clips": {"hop": {"seconds": 1, "loop": false, "min_displacement_mm": 10}},
  "shot": {"cam_pos": [-2, 1.5, -3], "cam_target": [0, .6, 0], "fov": 36}
}
```

`fps` is an integer 1–120; durations must be whole sample-frame counts and at most 60 seconds. Influence count
is 1–4. `rigid: true` requires every welded component to follow one bone at full weight, then checks edge
length preservation at every exported key. Leave it out for blended skinning. Loop endpoints are measured in
millimetres; interpolation between keys uses shortest-arc quaternion slerp. Non-loop clips retain their final
pose. Replay or return to idle is the engine's responsibility.

Optional per-clip gates: `min_displacement_mm`, `ground_drop_mm_max` (relative to source ground),
`max_edge_ratio`, `stretch_area_max` (fraction of rest surface area in triangles with an edge ratio above 1.5).
The report includes loop seams, displacement, worst stretch, stretched area and minimum Y. These are diagnostics
and declared gates, not an anatomical-quality score. Ground sampling includes all vertices and does not know
which foot should be planted. `events` is retained as animation extras; it does not execute gameplay callbacks.

A successful build writes `<output>.rig.json` with source/script/output hashes, the frozen contract and checks.
A failed contract leaves any prior delivery intact. Original meshes may remain unreferenced in the JSON;
their original binary data remains in the buffer, and only the augmented mesh instances are active.
All vertices remain unchanged at rest, with inverse binds checked within 10 micrometres.

## Known limits

The CPU posed-normal path handles rigid rotation/translation and blended rotation. It does not claim parity
for nonuniformly scaled skinned source nodes. Avoid those until an engine fixture establishes that path.
Neither self-intersection, planted-foot sliding nor volume loss is gated yet. A rigid fused character moves
as one piece; the rig stage cannot invent articulated limbs by changing topology. Author explicit joint
regions and inspect deformation when choosing blends. A large edge ratio can occur on a tiny seam: review
stretched surface area and the pose pictures together before accepting or rejecting a blended rig.

Tests in `tests/test_rig.py` exercise frozen accessors, nonidentity binds, rotations, interpolation,
blended weights, component welding, loops, failed-delivery preservation, the CLI and script restrictions.
