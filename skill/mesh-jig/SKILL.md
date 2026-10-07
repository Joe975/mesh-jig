---
name: mesh-jig
description: Build a low-poly 3D model as a Blender Python script and measure it against reference views (concept sheet, turnaround, top/side/rear) with the mesh-jig CLI. Use when asked to model an asset from reference images, match a model to concept art, or improve a model's silhouette, proportions or colour zones. Needs Blender and mesh-jig installed from a checkout of its repo (`pip install -e .`).
---

# mesh-jig

You write the model script. mesh-jig builds it in Blender, checks it against a contract, renders it on the CPU and
measures the renders against the reference views. The numbers are deterministic and free, so run them on every
attempt and let them, not your impression of a picture, decide what to change next.

For a new run, choose an ignored project directory such as `local-work/<run>/` and verify it is not already
tracked. Keep the complete run private. Use [mesh-jig-publish-results](../mesh-jig-publish-results/SKILL.md)
to share selected metrics, images or GLBs later; do not commit the raw project to a public repository.

## The loop

1. `mesh-jig doctor <project>` once. It must find Blender and report no project problems.
2. `mesh-jig brief <project>` and read all of it: the brief, the script rules, the contract, the reference's outline
   table when the project has one, the colour zone maps, and how this project is measured. Open every reference
   view it lists.
3. Write `<project>/attempts/<name>/model.py`. One idea per attempt, named for what it changes. Never edit an old
   attempt: copy it to a new directory, so the ledger stays true.
4. Look before you measure. `mesh-jig build <project> --script <project>/attempts/<name>/model.py`, then
   `mesh-jig preview <project> <project>/attempts/<name>`. A preview is free: it draws and measures the build as an
   evaluation would and prints the same overlap, colour zones, resemblance and numeric score, how your colours
   read, and where the build stands against your best measured attempt. It adds nothing to the ledger and is not
   an attempt, so an attempt that is not evaluated yet may be edited, built and previewed as often as it helps.
   When a task gives you a fixed number of attempts, this is how each one is made to count.
5. `mesh-jig eval <project> --script <project>/attempts/<name>/model.py --note "<what you changed>"` once the
   preview is ahead of your best attempt. This is the attempt: it writes the ledger line and gives what a preview
   does not, the outline table and the gates.
6. Read the output, all of it: do not cut it with `head`, `tail` or `grep`. Its first lines are the verdict (the
   numbers, each gate that FAILED and what it needed, where the attempt stands against your best one and which
   views moved); the views, the colours and every row of the outline table follow. Open the `*_vs_reference.png`
   files it names (reference left, your build right) and the `sil_*.png` outline diffs (red: add body here, blue:
   remove body here). When the project marks regions it also names `region_*.png` close-ups: each marked region
   cut from the reference and from your build and enlarged, side by side. A face or a small part is read there,
   not in the whole view, and a preview draws them too. To read an evaluation again run
   `mesh-jig show <project> <attempt>`, never `eval` a second time: that adds a second line to the ledger.
7. Fix the largest error first and go to 3.

Start with a baseline attempt that only gets the overall proportions right: the brief's HOW THIS PROJECT IS
MEASURED section names the numbers that give them, and says what to work on after that.

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
- **numeric score**: the one number to rank attempts on. It is the mean of overlap and resemblance unless the
  project states its own mix, and then the brief's HOW THIS PROJECT IS MEASURED section and the first line of
  every evaluation say what it is made of. Work on what the score is made of, in proportion.
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
- Blender metres, +Z up, centred on the origin. A craft's nose points along **+Y** and it mirrors across X = 0;
  for anything else the brief says which way the model faces and where each camera stands.
- Colour is a `CORNER` `FLOAT_COLOR` attribute, one palette colour per face, in **linear** RGBA (the palette in
  `jig.json` is sRGB hex: convert it).
- Faces outward (`bmesh.ops.recalc_face_normals`), no zero-area faces.
- `Vector((x, y, z))` takes one sequence. bmesh ops that delete or rebuild geometry invalidate faces you held.

## Textured contracts

When the brief says TEXTURED CONTRACT: write `atlas_plan.json` beside the script, add `--paint <that file>` to
`eval`, and author a UV layer in the script that maps each face onto the region painted for it. Paint the
reference's own markings where the reference has them. Detail in the wrong place scores below no detail.

When the brief says YOUR SCRIPT PAINTS THE TEXTURE there is no plan and no `--paint`: define
`texture(name, position, normal)` beside `build()`, as the brief shows. It is handed numpy arrays of points on each
object's surface and returns a colour for each, from the `PALETTE` global, so a marking is a test on where a
point is ("inside this ellipsoid", "facing forward"). Author no UVs and no materials. A preview shows the painted
model, so look at the paint as often as at the shape.

## The judge

`mesh-jig eval ... --judge --against <another attempt directory>` also asks a vision model which of the two is
closer to the reference. It costs money and takes a minute.

- Ask only after the numbers have moved, and always against your last judged attempt or the baseline.
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
`<view>_seen.png` shows the subject and nothing else. Its camera lines say which way each view must face. Run
`mesh-jig palette <dir> --write` and rename the
colours, and write `brief.md`: one bullet per part a viewer would name, with where it sits and its colour.
`docs/JIG.md` in the mesh-jig repo documents every field, and its Reference views section lists what a view has
to be for the measures to read it.

`init` writes a craft's contract: a nose, a length and the four default cameras. For any other subject (a
character, a mech, a prop) take out `length_m` and `forward` and give it its own orthographic `shots`, as step 4 of
`docs/GUIDE.md` in the mesh-jig repo says. The outline table is then read through those cameras.

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
fusion), bounded smoothing/decimation, final-topology CORNER color restoration, shaped ears/muzzle/digits,
transported curve frames and garments fitted to finished skin. Keep the existing script checker imports and
denylist. Do not force intentional flat styles to smooth or fuse clothing/accessory islands into skin.

Use opt-in named regions, exact exported node identities, signed landmark bounds, per-view z-buffer visibility
and explicit clearance probes for important features. Test mirrored handedness, missing features, garment
intrusion and hidden belt/buckle. Probe concave fits using the appropriate first/last surface policy, including
between loft stations. A garment can pass ring endpoints and still cross skin between rings; reserve neighboring
loft envelopes and inspect the imported CPU render. These checks cover declared probes/regions, not every pose
or every triangle. Keep evidence, configuration and fit decisions in versioned project files.
