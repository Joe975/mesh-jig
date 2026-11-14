# Optional facial rigging

For animation-ready geometry, declare the [early facial capability contract](FACE_CAPABILITIES.md)
before modelling. It adds attempt bindings and geometric smoke checks; this standalone
stage's `bound` status and `mouth_topology` declaration alone do not prove a cavity or
convincing expression.

The face stage binds a finished static GLB to explicitly authored facial regions,
then exports ordinary glTF skins and seven named animation clips. It is opt-in:
modelling, body rigging, their scores and existing exports are unchanged.
It reuses the frozen-geometry [rig stage](RIGGING.md), preserving the original
position, normal, UV, colour, index, material and texture data at rest.

```powershell
mesh-jig face-info --source source.glb --contract face.json
mesh-jig face-build --source source.glb --contract face.json --out face.glb
mesh-jig face-check --source source.glb --contract face.json --rig face.glb
mesh-jig face-preview --rig face.glb --contract face.json --out face-preview
```

The build writes `face.glb`, `face.face.json` (capabilities, source/output hashes,
frozen contract and deformation checks), and `face.face-rig.py` (reproducible rig
script). A failed build preserves an existing delivery. Preview writes an
expression contact sheet and talk/look GIFs. The look clip holds the configured
point; a fixed target does not move. The usual `rig-preview` tools can inspect
additional views using the rig contract returned by `face.rig_contract`.

## Defaults

| Clip / runtime state | Motion |
|---|---|
| `look_at` | Neck shares the turn, with each eye independently aimed at the point after the neck transform. Neck yaw/pitch limits 45/25 degrees; residual eye limits 30/25 degrees. |
| `talk` | Small jaw openings and subtle head nod/turn, in a two-second loop. No phonemes or audio synchronization. |
| `smile` | Both mouth corners rise with a slight brow lift. |
| `anger` | Brows lower and slope inward; corners tighten downward. |
| `sadness` | Corners lower, inner brows rise, head tips down. |
| `confusion` | Asymmetric raised brow and sideways head tilt. |
| `yell` | Full configured jaw opening (28 degrees by default), angry brows and slight head lift. |

Expressions hold their pose; all exported clips last two seconds. The caller
chooses looping and transitions. All amplitudes are conservative starting points,
not a claim that every topology makes a recognizable expression.

## Binding contract, version 1

Author `face.json` after examining the source with `rig-info`. Coordinates are
**Blender metres, +Z up, +Y forward, left at X<0**, in the source model's space.
No face detector guesses landmarks. Exact node names are checked; ambiguous
names, nonexistent nodes and controls with no exported influence are refused.

```json
{
  "version": 1,
  "fps": 24,
  "face_scale_m": 0.3,
  "look_target": [0.4, 2.0, 1.2],
  "mouth_topology": "closed_surface",
  "controls": {
    "neck": {
      "pivot": [0, 0, 0.85],
      "region": {"nodes": ["Head", "EyeLeft", "EyeRight"]}
    },
    "eye_l": {
      "pivot": [-0.06, 0.1, 1.05],
      "region": {"nodes": ["EyeLeft"]}
    },
    "eye_r": {
      "pivot": [0.06, 0.1, 1.05],
      "region": {"nodes": ["EyeRight"]}
    },
    "jaw": {
      "pivot": [0, 0, 0.98],
      "region": {"nodes": ["Head"], "center": [0, 0.09, 0.93],
                 "radius": [0.10, 0.15, 0.07], "inner": 0.35}
    }
  },
  "gates": {"stretch_area_max": 0.08},
  "preview": {"aim": [0, 0, 1.0], "span": 0.5}
}
```

This illustrative partial map needs measured values and actual node names. Add
`mouth_l`, `mouth_r`, `brow_l`, `brow_r` for the other expressions. Each control
has a `pivot` and a `region`. A region names whole nodes, or restricts those nodes
with `box: [[xmin,ymin,zmin],[xmax,ymax,zmax]]`, or with `center`, positive
`radius` and optional `inner` (0..less than 1, default 0.5). Ellipsoids have full
weight inside their inner fraction and smooth falloff to zero at their boundary.
Box boundaries are hard: prefer smooth selections on a continuous skin.

`neck` is mandatory. All other controls are optional children of the neck;
the body remains attached to `face_root`. Include eyes, lips, hats and other
head attachments in the neck region. Its optional `ramp: [axis,start,end]`
(axis 0/1/2 = X/Y/Z) blends a neck attachment from zero to full weight. Facial
regions are intersected with that neck weight. Overlapping facial weights are
normalized; the existing exporter deterministically keeps four influences.
Whole eye objects, including iris/pupil components, should follow one pivot.
Use the neck's optional `rigid_nodes: ["Hat", "Head", "EyeLeft", "EyeRight"]`
to give those complete nodes full neck attachment, bypassing its box/ellipsoid
and ramp. The names must also appear in its region. This lets a fused body's
neck blend while a separate skull and its accessories follow the turn fully;
the facial child controls still deform their selected vertices.

`face_scale_m` scales corner and brow travel. `jaw_open_degrees` is in (0,45],
default 28. `fps` uses the rig stage's integer 1–120 range; use at least 24 for
talking (very low rates can undersample the motion). Optional `gates` are
`max_edge_ratio` and `stretch_area_max`, with the same sampled interpretation
as ordinary rig checks. No face score is added to modelling evaluations.

## Capabilities and failure

`face-info` reports **bound**, **partial**, or **unavailable** per state, and
the actual post-pruning vertex count for each control. Bound means the required
controls have influences; it is not proof of visible or anatomically correct
motion. `required: ["look_at", "yell"]` refuses delivery unless each listed
state is bound. Leaving it out allows a partial result with explicit limitations.

`mouth_topology` is an author declaration: `unknown` (default), `closed_surface`
or `open_mouth`. Talking and yelling are partial unless `open_mouth` is declared.
The tool cannot certify that declaration. A painted mouth, closed surface or
separate lips over an uncut head cannot become a cavity through skinning.
Yell still commands the jaw, but the resulting deformation may look like a
stretched chin. To achieve a wide open mouth, author separated lips/jaw, an
interior and suitable topology in the modelling stage, then bind those vertices.
An eye patch is not a hidden eye. Set `visible_eyes: ["right"]` (or `["left"]`)
for a genuinely monocular character; `look_at` then requires only that eye and
the neck. The default remains both eyes. Early capability binding templates
carry this field from the project's requested visible eyes.

The checker verifies frozen rest geometry/colour, inverse binds, exact facial
weights and pivots, hierarchy, clip presence, controller agreement and the usual
sampled deformation gates. These do not certify self-intersection, attachment
gaps, volume preservation, gaze visibility or emotional readability.

## Runtime point tracking

```python
from mesh_jig import face, face_motion
config = face.contract(face_json_dictionary)
deltas = face_motion.pose_at(config, "look_at", target=(x, y, z), intensity=1.0)
talk = face_motion.pose_at(config, "talk", t=elapsed_seconds, intensity=0.7)
```

Call `pose_at` every frame as the target moves. It is stateless and deterministic.
Targets at a pivot retain a neutral aim; targets outside the range clamp, and a
target behind the character cannot be reached. Each eye computes convergence
from its transformed pivot. `intensity` is 0..1. For exported coordinates,
convert Blender vectors `(x,y,z)` to glTF `(x,z,-y)` and rotations with
`rig.quaternion`. Deltas are applied to the named `face_*` joints' rest transforms:
`move` adds to the rest translation; `rotate` replaces the rest-aligned rotation.
Ordered rotations follow the [rig convention](RIGGING.md#script-contract).

World-space targets must first be transformed by the inverse character transform.
The stage does not ship a Godot/Unity plugin, world-space scene adapter, automatic
state blending, or a joint override mixer. Do not run baked clips and runtime
updates on the same bones simultaneously. For return to neutral, set the face
joint deltas to identity or call a state with intensity zero.

## Integration limits and validation

Version 1 accepts static modelling exports only. Existing body skins/animations,
morph targets, sparse accessors and required compression/extensions are refused
rather than silently stripped. To combine face controls with an existing
locomotion rig, use the separate [character composition stage](CHARACTER.md).
It accepts the original static source, compatible body rig and this face binding.
No existing published model is replaced by either stage.

`tests/test_face.py` covers known geometry, gaze convergence, expression signs,
real exported clips, rest preservation, partial capabilities, refusal paths,
delivery preservation, overlapping weights, runtime inputs and CLI previews.
Practical trial outputs belong under ignored `local-work/`, not the public repo.
