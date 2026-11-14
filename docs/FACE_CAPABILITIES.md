# Facial capabilities declared before modelling

`face_capabilities` is an optional top-level **asset construction requirement** in
`jig.json`. Declare it when setting up the references, before the first model.
It is separate from visual criteria and never changes the resemblance score.
Projects that omit it keep their existing modelling/evaluation behavior.

```json
"face_capabilities": {
  "version": 1,
  "required": ["look_at", "talk", "smile"],
  "optional": ["anger", "sadness", "confusion", "yell"],
  "eyes": ["right"],
  "min_mouth_gap_m": 0.02,
  "min_interior_depth_m": 0.01
}
```

The seven capability names are the [face-stage states](FACE.md#defaults).
Required and optional lists must be disjoint and contain unique names; at least
one capability is requested. `eyes` explicitly lists the visible `left` and/or
`right` eyes when `look_at` is requested and is omitted otherwise. A one-eyed
character legitimately requests one eye. Do not invent geometry under its patch.
Directions are the character's own left/right, with left at X<0.
Thresholds are positive metres in the unscaled source model; their defaults are
20 mm opening and 10 mm depth. Set appropriate values for the character's size.
They are fixed requirements, not parameters the builder should relax to pass.

Project setup also accepts:

```sh
mesh-jig init local-work/my-character --height 1.2 --face-required look_at talk smile --face-optional yell --face-eyes right
```

The requirements appear in `mesh-jig brief`, the API visual builder's opening
brief, and both coding-agent and API visual-review briefs. Builders should keep
the neutral reference appearance, including a mouth that looks closed at rest.
Independent eye surfaces and iris/pupil regions, a usable jaw/lip opening with
interior surfaces, and deformable corners/brows must exist in the exported model
where the requested states need them. Separate parts, shared meshes with region
selectors, and other suitable construction techniques are allowed. This stage
does not manufacture a mouth cavity from a closed skin.

## Bind each attempt's actual surfaces

Write `attempts/<name>/face_bindings.json` beside its model script. `rig` is the
existing [face rig contract](FACE.md#binding-contract-version-1), in Blender
metres (+Y forward, +Z up). `semantics` identifies the surfaces the functional
checks inspect. Exact exported node names are required. Region selectors have
the same `nodes`, optional `box`, or `center`/`radius`/`inner` grammar as the rig.
Nodes need not be separate objects for each semantic region.

This compact talk-only binding illustrates the shape; replace every region and
pivot with measurements of the actual character:

```json
{
  "version": 1,
  "rig": {
    "version": 1,
    "face_scale_m": 0.6,
    "look_target": [0, 3, 0.65],
    "jaw_open_degrees": 35,
    "preview": {"aim": [0, 0, 0.46], "span": 1.2},
    "controls": {
      "neck": {"pivot": [0, 0, 0], "region": {"nodes": ["Head", "UpperLip", "LowerLip", "Interior", "LeftCorner", "RightCorner"]}},
      "jaw": {"pivot": [0, -0.13, 0.36], "region": {"nodes": ["LowerLip"]}},
      "mouth_l": {"pivot": [-0.29, 0.17, 0.36], "region": {"nodes": ["LeftCorner"]}},
      "mouth_r": {"pivot": [0.29, 0.17, 0.36], "region": {"nodes": ["RightCorner"]}}
    }
  },
  "semantics": {
    "mouth": {
      "upper_lip": {"nodes": ["UpperLip"]},
      "lower_lip": {"nodes": ["LowerLip"]},
      "interior": {"nodes": ["Interior"]}
    }
  }
}
```

For gaze, add `semantics.eyes.right` and/or `.left`, each containing `surface`,
`iris`, and `pupil` regions, plus the corresponding `eye_r`/`eye_l` rig controls.
Set `rig.visible_eyes` to the same eye list as the project; binding templates do
this automatically. A monocular `look_at` binding needs only its visible eye.
Identical regions for these three roles are rejected. Eye regions must move
independently of the neck and expose sampled visible vertices. Speech/yell need
distinct upper lip, lower lip and interior selections. Expressions bind their
mouth corners/brows; the brief lists the controls for each requested state.
Always author `rig.preview.aim` and `span` to frame the face.

After final script and binding edits:

```sh
mesh-jig face-bind local-work/my-character a01
mesh-jig eval local-work/my-character --script local-work/my-character/attempts/a01/model.py
mesh-jig preview local-work/my-character a01
```

The API visual builder can read/write its `face_bindings.json` and call the
no-argument `bind_face` tool before `submit`. File access remains confined to the
current attempt. `mesh-jig new --from` and visual-loop continuation carry the
binding forward with the script; review it when changing geometry and reseal.

The seal stores SHA-256 of the script and normalized project requirements.
Changed scripts or requirements invalidate it. Export inspection resolves every
region against that attempt's actual GLB. Each test report additionally records
the source GLB, script and binding hashes. For an already frozen export, an
optional `source_sha256` in the binding requires that exact GLB; a mismatch fails.
Sealing is provenance, not certification that the author's semantics are correct.
The workflow does not prove that an arbitrary externally supplied GLB was produced
by the sealed script; ordinary evaluation builds it, and preview inspects the
current `out/` export. Do not replace files behind a completed evaluation.

## Evidence and feedback

Evaluation writes `face_capabilities` and portable `face_pictures` fields into
`eval.json`. `face-test/test.glb`, its reproducible rig script, and 320-pixel face
PNGs provide neutral, full mouth opening, four gaze extremes and requested
expression/talking poses. These are diagnostic held poses, not the final looping
talk animation. Ordinary preview writes its own copies under `preview/face-test`.
The final facial deliverable uses `face-build` with the binding's `rig`, or
[character-compose](CHARACTER.md) to attach it to a compatible body rig.
The original static GLB is preserved.

Automatic checks establish only:

- The current script/requirements match the seal; selectors name existing,
  nonempty exported geometry and configured controls have actual influence.
- Eye/iris/pupil samples move relative to the neck in all four tested directions,
  with some sampled front visibility at neutral. Neck-only motion cannot pass.
- The jaw moves the lower lip and increases its separation from the upper lip.
  Nine double-sided, all-geometry rays through the central opening must hit the
  selected interior first and at sufficient depth. A front skin hiding a tagged
  interior fails. The actual yell pose also needs this evidence and brow motion.
- Requested expression regions show local movement; a head tilt alone is
  insufficient. Missing optional capabilities remain feedback, not required gates.

A successful state is named `sampled_evidence`, never “approved.” Required
checks add the separate `face_capabilities_sampled` gate; optional-only contracts
add no gate. Static numeric scores remain unchanged. Functional failures and
their images reach the next visual modelling attempt even with `checks=none`.

**Visual review remains pending.** Central rays are not a full cavity/topology
proof. They can miss off-axis occlusion and do not establish convincing anatomy,
an enclosed oral volume, teeth/tongue quality, expression readability, eye aiming
quality, or the absence of intersections. Detached parts can move and pass a
motion test. Review neutral fidelity separately from open-mouth, gaze and
expression close-ups; repair actual geometry rather than changing declarations.
Current probes assume a face oriented +Y with a roughly horizontal mouth. Extreme
asymmetry, sideways mouths and nonstandard anatomy may require manual assessment
and a future probe schema. Existing skinned/morphed source models are unsupported.
Body/face composition is a separate [optional stage](CHARACTER.md) after body
rigging. These modelling probes stay on the static source. Audio-driven lip sync
is not implemented.

## Reproducible offline example

```sh
python examples/face-capability/make_example.py local-work/open-puppet
python examples/face-capability/make_example.py local-work/blocked-puppet --blocked
mesh-jig eval local-work/open-puppet --script local-work/open-puppet/attempts/a01/model.py --no-ledger
```

The generator refuses to overwrite a directory. Both projects include a normal
Blender modelling script, full bindings and a source GLB. The schematic puppet's
lips meet at rest; the hinged lower lip reveals a recessed interior. It has one
visible eye, its iris/pupil, a patch, and brows/corners. The blocked version adds
a solid front surface while retaining the `open_mouth` claim and jaw weights:
speech and yelling must fail. Inspect `attempts/a01/face-report.json` and
`face-test/*.png`. This is an engineering fixture, not a visually approved
character. All geometry is generated by the example; no existing character asset or API is needed.

The focused tests also exercise both brief routes, API feedback with scores
withheld, malformed/stale bindings, optional versus required states, legitimate
monocular gaze, no-op legacy behavior, export preservation and real Blender
evaluation. The examples generate their own inputs and preserve their outputs for inspection.
