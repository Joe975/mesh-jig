---
name: mesh-jig
description: Build a low-poly 3D model as a Blender Python script and measure it against reference views (concept sheet, turnaround, top/side/rear) with the mesh-jig CLI. Use when asked to model an asset from reference images, match a model to concept art, or improve a model's silhouette, proportions or colour zones. Needs Blender and mesh-jig installed from a checkout of its repo (`pip install -e .`).
---

# mesh-jig

You write the model script. mesh-jig builds it in Blender, checks it against a contract, renders it on the CPU and
measures the renders against the reference views. The numbers are deterministic and free, so run them on every
attempt. Use visual critique to decide what to change; use numbers to diagnose defects, not choose a visual winner.

For a new run, choose an ignored project directory such as `local-work/<run>/` and verify it is not already
tracked. Keep the complete run private. Use [mesh-jig-publish-results](../mesh-jig-publish-results/SKILL.md)
to share selected metrics, images or GLBs later; do not commit the raw project to a public repository.

## The loop

Make and change files with your file tools (write, edit), not shell heredocs, `sed -i` or `python -c`, and run
each `mesh-jig` command on its own rather than chained with others: a sandboxed run may allow nothing more, and
a denied command is not a sign that the next one will be.

1. `mesh-jig doctor <project>` once. It must find Blender and report no project problems (its warnings about the
   reference pictures are not problems: build on).
2. `mesh-jig brief <project>` and read all of it: the brief, the script rules, the contract, the reference's outline
   table when the project has one, the colour zone maps, and how this project is measured. Open every reference
   view it lists. The brief is long: run it on its own, not beside `doctor` or a `cat`, and if your shell tool cuts
   long output, save it to a file in the project and read that.
3. Write `<project>/attempts/<name>/model.py`. One idea per attempt, named for what it changes. Never edit an old
   attempt, so the ledger stays true: start each new one with `mesh-jig new <project> <name> --from <attempt>`,
   which copies the script (and any atlas plan) into `attempts/<name>/`, then read and change the copy. The first
   has nothing to copy: `mesh-jig new <project> a01-baseline` makes its directory and you write `model.py` there
   (an upright character can start from `mesh-jig docs recipe`).
4. Build a complete candidate. Optional `mesh-jig build` and `mesh-jig preview` can help repair it, but do not
   run preview after preview on a candidate just to push its numeric score past the previous attempt's. A preview
   has no outline table: for a free
   draft that has one, `mesh-jig eval ... --no-ledger --short` measures it all and adds nothing to the ledger. `build` is quick and prints the triangle
   count against the ceiling and the nodes holding the most: run it before `eval` when near the ceiling, and before
   the first `eval` of a script you have changed a lot, since an `eval` that fails the build or the contract still
   leaves a FAILED line in the ledger. It also names any vertex colour far from every palette colour (faces a paint
   step missed).
5. `mesh-jig eval <project> --script <project>/attempts/<name>/model.py --note "<what you changed>"` on every
   completed candidate, including numeric regressions. Repair failed hard build/contract checks. Preserve every
   evaluated attempt and its ledger entry.
6. Read the output, all of it: do not cut it with `head`, `tail` or `grep`. Its first lines are the verdict (the
   numbers, each gate that FAILED and what it needed, where the attempt stands against your best one and which
   views moved); the views, the colours and every row of the outline table follow. Open the `*_vs_reference.png`
   files it names (reference left, your build right) and the `sil_*.png` outline diffs (red: add body here, blue:
   remove body here). Where it names `ruled_*.png` pairs, open those too: the reference and your build at one
   scale on lines at round Blender metres, which is where to read how far a part of yours ends from where the
   reference's does. When the project marks regions it also names `region_*.png` close-ups: each marked region
   cut from the reference and from your build and enlarged, side by side. A face or a small part is read there,
   not in the whole view, and a preview draws them too. To read an evaluation again run
   `mesh-jig show <project> <attempt>`, never `eval` a second time: that adds a second line to the ledger. The
   whole output is also written to `attempts/<name>/feedback.txt`: when your shell cuts long output, read that file
   in parts rather than piping `eval` through `head`. `eval --short` prints only the verdict block (the numbers,
   each gate, where it stands and which score terms moved) and leaves the rest to that file; read the rest when
   the verdict does not say what to change.
7. Obtain a fresh visual critique from the brief, references, comparison views and close-ups, without builder
   reasoning or numeric scores. `mesh-jig review-brief <project> <attempt>` prints what to hand a reviewer: the
   instructions `mesh-jig agent` gives its own reviewer, the design brief and every picture to open. Start a
   separate reviewer (a subagent that has seen nothing of your work) with that text and those pictures and
   nothing else; when you cannot start one, review from the same text yourself and label it self-review at the
   top. Save the critique in the attempt's `review.md`: up to five concrete fixes and features to preserve.
   Fix the largest visible mismatch first and go to 3. Do not restart the model without a structural reason.

Start with a baseline that gets the overall proportions right. Use five evaluated attempts unless the user
specifies another budget or stopping rule. Report the visual choice with reasons, highest numeric score and
latest candidate separately. Do not discard a visually useful result because its numeric score fell.

For a user-authorized API run, `mesh-jig agent <project>` implements this loop with five iterations by default.
Choose the provider/model explicitly; `--max-cost` supports known direct Anthropic 5.5 models. The command calls
paid APIs: do not start one merely because you are using this skill. `--checks builder|both` are experimental
numeric-feedback variants; `--workflow numeric` retains the older measurement-led tool loop.

## Keep the selections

After inspecting saved renders, optionally record latest, numeric best and your visual choice with
`mesh-jig checkpoint <project> round-05 --visual-attempt <attempt-name> --visual-source self-review --reason
"<visible reason>"`. It copies selected artifacts and makes a sheet without new evaluations or model calls.
Leave visual options out if no choice was made. Use `--before round-05` on a later checkpoint for a comparison.
Read [selection checkpoints](../../docs/SELECTIONS.md) for the format, evidence and verification options.
Keep these raw snapshots private. A checkpoint does not change the task's attempt budget or stopping rule.

## Reading the numbers

Which views count, which outline table there is and how a row of it reads differ from subject to subject (a craft,
a standing character, anything else). The brief's HOW THIS PROJECT IS MEASURED section says them for this project:
go by it.

- **outline error (mm)**: your GLB's own outline against the reference's, station by station. `+` is your model
  bigger or reaching further out. It is reported beside the score, not in it, and a project with no outline table
  has none.
- **overlap**: silhouette IoU per view, averaged over the views the brief says count. "too flat or too wide" lines
  name a proportion problem.
- **zones**: share of cells whose colour matches the reference's. Low zones with high overlap means the shape is
  right and the paint is in the wrong place.
- **colour**: how close each cell's colour is to the reference's in the same place, 1 for the same colour. It
  falls for the right palette colour in the wrong shade, which zones do not see.
- **resemblance**: zones plus edges in place.
- **regions**: each region the project marks (a face, a paw), read on its own and finer than the whole view. The
  `REGIONS` line lists them weakest first.
- **numeric score**: the number to rank measured agreement on, not visual quality. It is the mean of overlap and resemblance unless the
  project states its own mix, and then the brief's HOW THIS PROJECT IS MEASURED section and the first line of
  every evaluation say what it is made of. Investigate disagreement with the images instead of optimizing it blindly.
- **gates**: limits a project may set on a measure, a marked region or a named part. They are counted on the first
  line and are not in the score. A FAILED line says what the gate read and what it needed. An attempt that scores
  higher and fails a gate your best one passes is a trade, not a gain: the `standing` line says when that happened.

`attempts/ledger.md` has one line per attempt, with its numbers and its gates. It is your memory between sessions:
read it before starting.

## When you report

Whatever the task tells you to report, end it by asking the user whether they want to see the results in the
dashboard:

```
mesh-jig view <project> --open
```

That is a page on this machine (http://localhost:8796) with every attempt beside the reference: the numbers, the
comparison pictures, each build turning in 3D, and a button that opens the folder a build's GLB is in, for taking
it into another program. Ask, and do not open it unasked: the command serves the page until it is stopped and
never returns, so it is not a step of the loop and not something to wait on. On a yes, start it in the background
and give the user the address it prints (add `--port <n>` when it says the port is taken). When you cannot leave a
command running, or there is nobody to answer, put the command in the report for them to run.

## Script rules

- `def build():` at top level. Import only `bpy`, `bmesh`, `mathutils`, `math`, `random`, and `numpy` and
  `openvdb`, which ship in Blender's own Python.
- No `open`, `exec`, `eval`, `getattr`, `type`, dunder attributes, `bpy.ops.wm/render/image`, no importer or
  exporter, `bpy.app`, `bpy.utils`, and nothing in Blender, `numpy` or `openvdb` that reads or writes a file (a
  `bpy.data` collection's `load`, an image's `save`, a `filepath`). No picture is read into the model: its shape
  and colour come from the script. The runner refuses the script.
- Blender metres, +Z up. Follow the brief's placement instructions: for a calibrated standing model,
  preserve its fixed ground origin and do not recenter its bounding box; without standing calibration, keep it
  centred on the origin. A craft's nose points along **+Y** and it mirrors across X = 0;
  for anything else the brief says which way the model faces and where each camera stands.
- Colour comes from the contract the brief states. The default is a texture the script paints (below): colour is
  a function of where a point is, and the triangles do not limit its edges. On a vertex-colour contract it is a
  `CORNER` `FLOAT_COLOR` attribute, one palette colour per face, in **linear** RGBA (the palette in `jig.json` is
  sRGB hex: convert it), and every colour boundary is a triangle edge.
- Faces outward (`bmesh.ops.recalc_face_normals`), no zero-area faces.
- `Vector((x, y, z))` takes one sequence. bmesh ops that delete or rebuild geometry invalidate faces you held.

## Textured contracts

A new project's default. When the brief says YOUR SCRIPT PAINTS THE TEXTURE there is no plan and no `--paint`:
define `texture(name, position, normal)` beside `build()`, as the brief shows. It is handed numpy arrays of points
on each object's surface and returns a colour for each, from the `PALETTE` global. Draw markings in the
coordinates `build()` made each part from (along, around, chord, span), with normals as soft weights only, as the
brief says. Author no UVs and no materials. A preview shows the painted model, so look at the paint as often as
at the shape.

When the brief says TEXTURED CONTRACT without that: write `atlas_plan.json` beside the script, add
`--paint <that file>` to `eval`, and author a UV layer in the script that maps each face onto the region painted
for it, as (u, 1 - v) of the plan's point: Blender's v runs up. Paint the reference's own markings where the
reference has them. Detail in the wrong place scores below no detail.

## The judge

`mesh-jig eval ... --judge --against <another attempt directory>` also asks a vision model which of the two is
closer to the reference. It costs money and takes a minute.

- Use only when the task authorizes paid pairwise judging, against your last judged attempt or the baseline.
- Only a DECISIVE verdict is a result. "Not decisive" means the judge can't tell them apart: don't re-ask until it
  says yes.
- Read its reasons. They say what a viewer sees that the numbers don't.
- Run it in the foreground and wait for it. A judge left running when your turn ends is killed and paid for anyway.

## Starting a project from a sheet

For a new or inconsistent generated sheet, first use [mesh-jig-reference-sheet](../mesh-jig-reference-sheet/SKILL.md)
to generate, review and repair its cross-view anatomy, components, pose and projections. Once accepted, freeze
the selected reference before comparing model attempts.

```
mesh-jig init <dir> --length <metres> --sheet <image>
```

Then look at the sheet, set one crop box per view in `jig.json` (`sheet.crops`: hero, top, side, rear; `mask` out
any title text), run `mesh-jig sheet <dir>`, then `mesh-jig refs <dir> --out <another dir>` and fix the crops until
it prints no `!` line (a view touching the picture's edge, a label or shadow taken as subject) and each
`<view>_seen.png` shows the subject and nothing else. Its camera lines say which way each view must face. With
`model.height_m` set it also writes `<view>_ruled.png`: the view with lines at round Blender coordinates, to read
where a part of the reference sits in the numbers your script writes. Run
`mesh-jig palette <dir> --write` and rename the
colours, and write `brief.md`: one bullet per part a viewer would name, with where it sits and its colour.
`docs/JIG.md` in the mesh-jig repo documents every field, and its Reference views section lists what a view has
to be for the measures to read it.

`init` writes a craft's contract: a nose, a length and the four default cameras. For anything upright (a
character, a mech, a figure) give `init --height <metres>` instead of `--length`: it writes a standing contract with
six views (front, side, rear, right, hero, top), and `mesh-jig refs` then prints the calibration to copy into
`criteria.outline` (step 4 of `docs/GUIDE.md` in the mesh-jig repo). The outline table is then read through those cameras.

What a reference is judged on is written beside it, in `jig.json`'s `criteria` block, never assumed from the kind
of subject:

```json
"criteria": {"views": {"front": 1, "side": 1, "rear": 1, "top": 1},
             "gates": {"resemblance_min": 0.6}}
```

`views` are the views the score is taken over, with weights (list the orthographic views that show whether the
model is right; leave a painted hero out). `gates` are limits on the measures. `outline` and `review` are the
standing table and the named-region checks below. `init` states a craft's three views: change them when you change
the subject. `mesh-jig criteria <dir>` reads back what is in force, where it is stated and what is not stated;
`--write` puts it all in the block without moving a number. Set the criteria before the first attempt and do not
change them while attempts are being compared. `docs/JIG.md`, Criteria, has each key.

## Standing/organic models (opt-in)

The three files linked below live in the mesh-jig checkout, which a copy of this skill may not sit beside:
`mesh-jig docs organic`, `mesh-jig docs contract` and `mesh-jig docs recipe` print them wherever it runs.

Read [the organic workflow](../../docs/ORGANIC.md) before building an upright character: it has the steps for
measuring a calibration and a region box from the reference's pixels with `mesh-jig refs` (`--region VIEW X0 Y0 X1
Y1` prints a region entry). Use the
[configuration template](../../examples/character-contract/jig.example.json) as a schema example; replace its
illustrative calibration/regions/parts/probes with actual frozen reference measurements. Keep height_m and
render stage_scale distinct. Use one master physical pixel scale, per-view ground/horizontal origins and named
height stations; never independently fit each view's bounding height or mirror asymmetric outline extents.
List all five orthographic views in `criteria.views` and inspect weak views; keep illustrative hero out of it
until its camera is calibrated. Omitted settings retain craft behavior and historical scores.

The [safe worked recipe](../../examples/organic/recipe.py) demonstrates remesh fusion (joining alone is not
fusion), bounded smoothing/decimation, a painted `texture()`, shaped ears/muzzle/digits,
transported curve frames and garments fitted to finished skin. Keep the existing script checker imports and
denylist. Do not force intentional flat styles to smooth or fuse clothing/accessory islands into skin.

Use opt-in named regions, exact exported node identities, signed landmark bounds, per-view z-buffer visibility
and explicit clearance probes for important features. Test mirrored handedness, missing features, garment
intrusion and hidden belt/buckle. Probe concave fits using the appropriate first/last surface policy, including
between loft stations. A garment can pass ring endpoints and still cross skin between rings; reserve neighboring
loft envelopes and inspect the imported CPU render. These checks cover declared probes/regions, not every pose
or every triangle. Keep evidence, configuration and fit decisions in versioned project files.

When the brief names a TARGET (the project's `target`, or `--target quest` / `--target pcvr` on your commands), its
triangle ceiling is the contract's and every evaluation ends with a MESH READINESS block: checks that hold or do
not (buried triangles, pieces, open edges, slivers, symmetry, materials). They do not fail a build or move the
score. Read them as you read a failed gate: a model that matches the reference and holds them is worth more to
the stages after this one than one that only matches. Pass the same `--target` on every command of a run.
