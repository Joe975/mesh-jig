# Body and face on one character

`character-compose` adds facial controls to an existing compatible body skin.
It exports one GLB with the original body clips, independent facial clips, and
optional simultaneous demonstration clips. It does not change the frozen source
or manufacture missing eyes, lips or a mouth cavity. Declare the
[early face requirements](FACE_CAPABILITIES.md) before modelling for those.

```sh
mesh-jig character-compose --source source.glb --body body.glb --face face.json --contract character.json --out character.glb
mesh-jig character-preview --rig character.glb --body walk --expression talk --gaze 0.3 4 0.7 --out local-work/character-preview
```

Use the **static GLB from which the body was built** as `source`, the body-stage
export as `body`, and the [face binding's `rig` object](FACE.md) as `face.json`.
The output must differ from both inputs. The contract is:

```json
{
  "version": 1,
  "anchor": "head",
  "neck": "additive",
  "fps": 24,
  "max_influences": 4,
  "demos": [
    {"name": "walk_talk", "body": "walk", "expression": "talk"},
    {"name": "walk_yell_gaze", "body": "walk", "expression": "yell",
     "gaze_target": [0.5, 4, 0.7]}
  ]
}
```

`anchor` names an existing uniquely named body joint. Choose `neck` explicitly:

- `body`: body owns all head/neck movement. Facial nods/tilts are suppressed;
  gaze turns the eyes relative to the currently animated head.
- `additive`: a new child `face_neck` adds the facial nod/turn under the body's
  head transform. The original body head channel is never replaced or evaluated
  twice. When gaze is active it owns this extra turn, replacing the expression's
  nod; the expression still controls jaw, corners and brows.

`fps` (1–120, default 24) samples facial and baked tracks. Use at least 24 for
talk. `max_influences` is 1–4, default 4. `demos` is optional; names must be unique
safe identifiers. Expressions are `talk`, `smile`, `anger`, `sadness`,
`confusion`, `yell`. Omitting expression and gaze gives neutral face controls.

## Binding and preservation

Supported body rigs follow mesh-jig's rest convention: every joint's world rest
orientation is identity and scale is one. They can have arbitrary rest
translations, translation/rotation animations, and transformed source meshes.
Only LINEAR and STEP joint animation is supported. Arbitrary imported skeleton
retargeting, rotated/scaled joint rest frames, scale animation, morph targets,
sparse/compressed geometry and extra weight sets are refused.

`face_neck` is attached beneath the anchor; eyes, jaw, corners and brows are its
children. Authored pivots are source Blender metres (+Y forward, +Z up). The
exporter converts axes, computes parent-local offsets and per-mesh inverse binds.
Original joint hierarchy, animation channels, embedded images, materials and
source geometry remain intact. The `face_` node namespace and `face.`/`combined.`
clip prefixes are reserved. Existing collisions or already composed inputs fail.

For each vertex with anchor weight `a`, only that weight is redistributed among
the original anchor and new facial controls according to the authored facial
regions. All other body influences stay unchanged. Neutral facial transforms
therefore reproduce the original body deformation, including mixed head/torso
weights. A selected vertex with no anchor weight is an error. Narrow its facial
selection or repair the body binding; the composer does not invent a nearest-bone
attachment. Weights exceeding the configured limit fail instead of being pruned.

Include head accessories in the neck selection when they should share additive
facial head motion. Do not select the torso merely because a standalone face rig
used it for a neck transition. Choose body-owned neck motion if the body rig
already supplies the desired turn. Mixed body influences can attenuate facial
motion; numerical attachment is not proof that a seam or deformation looks good.

The delivery is re-read and checked against original body playback in rest and at
three times per clip before replacing the output. The adjacent
`character.character.json` records input/output hashes, binding, ownership,
influence counts, rest/replay errors and clip categories. This checks preservation,
not anatomy or expression quality; `visual_approval` remains `pending`.

## Playback: choose one route

**Live controller (Python):** samples one original body clip and one expression,
then solves an optional gaze target after accounting for the current body/head
transform. Every call starts from rest, so repeated changes cannot accumulate
rotations or leave a previous expression behind.

```python
from mesh_jig.character import Player

player = Player("character.glb")
world_matrices = player.worlds("walk", elapsed, expression="talk",
                               gaze_target=[0.3, 4, 0.7])
posed_mesh = player.mesh("walk", elapsed, expression="yell")
neutral = player.mesh()  # rest body and rest face
```

Targets use Blender **character/model space**, not external scene world space.
Invert the character's external scene transform first. Animated body root motion
is already accounted for. Targets outside the eye/neck angular limits clamp.
Expressions sample the exported clips (two-second loops); `face_time` may set
their independent clock. Body times clamp at clip ends; the caller loops body
time if appropriate. `worlds` returns glTF world matrices; `skin.posed(...,
world=...)` consumes them. This is a reference CPU player, not a realtime engine
adapter, audio lip-sync system or crossfade mixer.

**Engine layers:** play one original body clip plus one `face.<state>` clip on
disjoint channel masks at full influence. Body clips target original joints;
facial clips target appended facial joints only. Sample both from rest each frame,
merge their local TRS channels, then update the hierarchy and skin once. Disable
the previous face action before switching; reset unused channels to rest when
stopping. Do not let automatic full-skeleton rest tracks overwrite another layer.
`skin.sample_layers(doc, blob, [("walk", t), ("face.talk", t % 2)])` implements
this contract and refuses overlapping node/property ownership. Blender's glTF
importer plus disjoint NLA REPLACE tracks is tested against it. Other engines need
an equivalent masked player.

`face.look_at` holds a point relative to the **rest head frame**. It cannot track a
fixed scene point during body motion by itself. Use the live controller for
dynamic gaze, or a baked demo for a predetermined point. Layering two independent
face clips (for example talk plus look_at) is refused because both own the same
facial channels. The live controller explicitly composes expression and gaze.

**Baked demonstrations:** play `combined.<name>` alone. Each contains the original
body channels unchanged plus sampled facial channels, including gaze compensation
for the body's moving head. Never layer it over its original body/face clips.
These are ordinary glTF animations usable without a custom mixer. They are marked
nonlooping; they agree with live poses at sampled keys, while dynamic gaze between
keys is an interpolation approximation. Raise sampling rate if that matters.

## Offline example and verification

```sh
python examples/face-capability/compose_example.py local-work/combined-puppet
```

The generator refuses an existing directory. It builds a five-joint moving body
under the one-eyed [open-mouth fixture](../examples/face-capability/make_example.py),
then writes source/body/character GLBs, exact contracts and two playback GIFs.
The face close-up follows the moving neck; the whole-character view stays fixed.
The puppet's closed resting lips reveal an actual recessed interior when opened.
It demonstrates attachment, simultaneous motion and control ownership, not
production character quality. A closed mouth surface cannot provide a real opening just because
it has these animations. Both examples generate their geometry; no external character assets are needed.

`tests/test_character_composition.py` checks actual GLBs: rest/body preservation,
moving/rotating head attachment, fixed-point gaze, mouth-opening rays during
body motion, repeated switches, influence/channel conflicts, transformed source
meshes, monocular gaze and CLI export/preview. A real Blender import and NLA replay
compares deformed vertices with the CPU sampler, including sub-key times and
returns to neutral. Review rendered playback separately for intersections,
seams, expression readability and source appearance. This work adds no modelling
score and does not establish that builders produce better characters.
