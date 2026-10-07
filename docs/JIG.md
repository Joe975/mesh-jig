# jig.json and the project directory

```
<project>/
  jig.json            the contract
  brief.md            what the model must show, in prose (the judge reads it)
  refs/<view>.png     reference views, named after the cameras: hero, top, side, rear
  attempts/<name>/    model.py, atlas_plan.json; out/, renders/, compare/, eval.json are written by eval
  attempts/ledger.md  one line per attempt
```

## Top level

| Field | Meaning |
|---|---|
| `name` | Project name. Default: the directory name. |
| `brief` | Brief file. Default `brief.md`. |
| `refs` | Reference view directory. Default `refs`. |
| `palette` | `{"name": "#rrggbb"}`: the colours the reference is painted in. Turns on the zone measure and the zone maps in `mesh-jig brief`, and names colours for atlas plans. |
| `model` | The build contract ([model](#model)). Required. |
| `sheet` | Optional: cut `refs/` from one design sheet with `mesh-jig sheet` ([sheet](#sheet)). |
| `criteria` | What this reference is judged on: the scored views, a calibrated outline table, gates and review ([Criteria](#criteria)). |

## Reference views

`mesh-jig refs <project>` checks the first group below and prints each view's subject box, its ground line and
where its camera stands. `--out <dir>` also writes `<view>_seen.png`: the picture with everything the mask doesn't
take as the subject washed out.

**Every view**

- One picture per view in `refs/`, named after its camera (`refs/front.png` for the shot named `front`). A picture
  with no camera of its name is never measured.
- One flat background colour all the way round the subject. The background is read off the picture's border and
  the subject is whatever stands off it, so a pale part at the subject's edge that's close to the background is
  lost. A pale panel inside a darker outline is kept.
- The whole subject inside the picture, not touching an edge.
- Nothing else in the picture: no label, no floor shadow, no sliver of the next view. A separate piece at least
  2% the size of the largest one is taken as subject, and a shadow touching the feet becomes part of the outline.
  Tighten the crop, or paint it out with `sheet.crops.<view>.mask`.
- The same design and pose in every view. One model can't match views that disagree, and the numbers won't say
  which view is wrong.

For the score, views needn't share a scale or be centred: overlap, zones and resemblance fit each view to its own
box. A view's proportions (height against width) do count.

**Each view faces the way its camera looks**

Otherwise the model is measured against a mirror image. `mesh-jig refs` prints each camera's side and which way
picture-right points. For the cameras of a project's `jig.json`, plus the
`right` the cat adds (Blender axes: +Y is the subject's front, +Z up, the subject's left is -X):

| View | Shows | In the picture |
|---|---|---|
| `front` | the front | the subject's left is on the right |
| `side` | the subject's left side | the front faces left |
| `right` | the subject's right side | the front faces right |
| `rear` | the back | the subject's left is on the left |
| `top` | from straight above | the front points down |
| `hero` | a three-quarter view | compared for you to look at; not a metric projection |

The four default (craft) cameras are in [Cameras](#cameras). Draw `front`, `side`, `right`, `rear` and `top`
without perspective: the outline table is read through orthographic cameras that look along one axis.

**Only for some criteria**

- Zones (`palette`): flat, readable colour areas. A cell is the palette colour nearest its pixels, so heavy
  shading or a gradient reads as another colour.
- A calibrated standing table (`criteria.outline`): `front`, `side`, `rear` and `right` drawn at one pixel scale,
  upright, standing on a ground line. `mesh-jig refs` prints their heights against each other and flags a view
  more than 2% off. [ORGANIC.md](ORGANIC.md) has the steps.

## Criteria

`model` says what a build must be. `criteria` says what this reference is judged on. It lives in the project's own
`jig.json`: mesh-jig's code holds nothing about a particular subject.

```json
"criteria": {
  "views":   {"front": 1, "side": 1, "rear": 1, "top": 1},
  "outline": {"mode": "standing", "calibration": {"view": "front", "height_px": 462, "origins_px": {"front": [250, 490]}},
              "stations": [{"name": "head", "z_m": 1.02}, {"name": "hip", "z_m": 0.42}]},
  "gates":   {"outline_mm_max": 12, "zones_min": 0.5, "resemblance_min": 0.6},
  "review":  {"parts": {"tail": {"nodes": ["Tail"]}}, "regions": [{"name": "face", "view": "front", "box": [0.25, 0.05, 0.75, 0.38], "min_iou": 0.65}]},
  "score":   {"overlap": 1, "colour": 1, "edges": 1, "regions": 1}
}
```

Every key is optional.

| Key | What it states | Left out |
|---|---|---|
| `views` | The views overlap, zones, resemblance and the numeric score are taken over, each with a positive weight. Each needs a camera (`model.shots`, or a default shot) and a reference view of its name. A view not listed is still rendered and compared for you to look at. | **The fallback**: top, side and rear when the project has them, else every compared view. The brief, `doctor` and an evaluation's feedback say it's the fallback. |
| `outline` | A calibrated outline table at heights you name, for a standing subject. Needs `model.height_m`. Fields in [ORGANIC.md](ORGANIC.md). | The table is read automatically: the craft table with `model.length_m` and top, side and rear views, else through the orthographic cameras ([Cameras](#cameras)). |
| `gates` | Limits on the measures: `outline_mm_max` (outline error, mm), `zones_min` (zone agreement, 0..1), `resemblance_min` (0..1). Each is reported as passed or FAILED in the feedback and `eval.json`. | No limits. The numbers are still recorded. |
| `review` | Named regions, mesh parts, landmarks, visibility and clearance probes. A region becomes an acceptance gate only with `min_iou` or `min_zone`; otherwise it is an observation. Fields in [ORGANIC.md](ORGANIC.md). Every region is also drawn enlarged beside the reference's (`compare/region_<name>.png`). | No regional checks. |
| `score` | What the numeric score is a weighted mean of. Terms: `overlap`, `zones`, `colour` (how close each cell's colour is to the reference's in the same place, in Oklab: 1 the same, 0 from 0.25 apart; it needs no palette), `edges`, and `regions` (the mean over `review.regions` of each region's own colour and edges, read on a canvas twice as fine). A term at 0 is left out. `regions` needs marked regions and `zones` a palette. The first line of an evaluation names the mix. | Overlap and resemblance (zones and edges) in equal parts, which is overlap 2, zones 1, edges 1. Colour and regions are still measured and reported. |
| `light` | How a build is lit when it is drawn. `"camera"`: white lights that turn with each camera, from its upper left, so a face that looks straight at a camera is drawn in exactly the colour it stores, in every view. Use it for anything with a front, a side and a rear. | `"stage"`: two fixed lights set for a craft seen from above. A face toward the `front` or `side` camera of an upright subject gets 0.63 of the light and one seen from above 1.15, so one paint reads as two colours. |

Two things outside the block also decide what's measured: `palette` turns on the zone measure, and the sizes in
`model` (`length_m`, `span_m`, `height_m`) are the build contract and give the outline table its millimetres.

`mesh-jig palette <project> --write` cuts every body pixel, shadow included, so its colours are duller than the
reference's paint. `mesh-jig palette <project> --refine` prints what each colour would move to, read off the lit
side of the scored views where they read as it. Name the ones to move (`--refine orange red --write`): a colour
that covers little of the subject, an eye or a buckle, is read off its neighbours' shading and is better left or
set by hand. Do it before the first attempt, since the zones are measured against the palette.

A gate that fails doesn't fail the attempt: `ok` means built, in contract, rendered and measured.

An unthresholded region is still measured and may contribute to `score.regions`, but cannot pass or fail:
its review record has `observation: true` and `passed: null`. Feedback, the ledger, streak and viewer count only
acceptance gates. For older records, region checks with `view` and `iou` but no stored `min_iou` or `min_zone`
are displayed as observations too; no evaluation file or numeric score is rewritten. `doctor` and `criteria`
warn about these regions and projects with no explicit quality acceptance limits. Asset contract checks still apply.

State `score` before the first attempt. Scores under different criteria are not comparable.

### Giving a reference its criteria

1. Decide which reference views say whether the model is right and write them in `views`. A three-quarter `hero`
   painting is usually left out: it isn't a metric projection.
2. `mesh-jig criteria <project>` reads back what's in force, where `jig.json` states each part, and what isn't
   stated. `mesh-jig doctor <project>` prints the same and reports anything wrong with the block.
3. Add `gates` once a few attempts show what numbers are reachable. Add `outline` and `review` for a subject that
   needs them ([ORGANIC.md](ORGANIC.md)).
4. Freeze the block before comparing attempts or runs: a number is only comparable under the same criteria.

`mesh-jig criteria <project> --write` writes what's in force into the block, the fallback views included, and takes
the same settings out of their older positions (below). No number moves.

A prop or a machine with front, side, rear and top views needs only:

```json
"criteria": {"views": {"front": 1, "side": 1, "rear": 1, "top": 1}}
```

### Older positions

Each criterion is also read from a position outside the block, and measures the same from there.
State each setting once: the same setting in both places is a project problem.

| In `criteria` | Older position |
|---|---|
| `views` | top-level `scoring`: `{"views": {...}}`, or `{"preset": "character"}` for front, side, rear, right, top |
| `outline` | `model.profile` |
| `gates.outline_mm_max`, `gates.zones_min`, `gates.resemblance_min` | `model.profile_mae_max`, `model.placement_min`, `model.resemblance_min` |
| `review` | top-level `review` |

## model

| Field | Meaning |
|---|---|
| `script` | Script name inside an attempt. Default `model.py`. |
| `variants` | Fit names, e.g. `["gun", "missile"]`. The script reads the global `VARIANT`. Absent: one GLB. |
| `glb` | Output pattern, `{variant}` substituted. Default `<name>.glb` or `<name>_{variant}.glb`. |
| `join` | Join every mesh object into one. |
| `length_m`, `length_tolerance` | Extent along the forward axis, and its relative tolerance (default 0.05). Also sets the render scale and the outline tables' millimetres. |
| `span_m`, `span_tolerance` | Extent across X (default tolerance 0.10). When set it replaces the "no wider than long" check. |
| `height_m`, `height_tolerance` | GLB Y extent (Blender Z), and its relative tolerance (default 0.05). Turns off the craft checks on the longest axis and on tallness, with `forward` set or not; `length_m` and `span_m` are still checked. Doesn't change the render scale. |
| `axis_extents_m`, `axis_tolerances` | `{"X": 0.9, "Z": 0.85}` in GLB axes, with a relative tolerance per axis (default 0.05). Extents are positive, tolerances finite and not negative. A Y extent that contradicts `height_m` is rejected. |
| `forward` | `"-Z"`: a craft with a nose. Absent: no nose, no mirror plane, no orientation check. |
| `max_triangles`, `min_triangles` | Triangle budget per GLB. |
| `single_mesh` | Exactly one mesh node. |
| `vertex_colors` | Every primitive carries `COLOR_0`. |
| `min_colors` | At least this many distinct vertex colours. |
| `textured` | The textured contract: every build needs an atlas (`--paint plan.json` or `--atlas file.png`), UVs on every mesh, and the image embedded in the GLB. |
| `atlas` | Atlas file name. Default `atlas.png`. |
| `texture_mode` | `"multiply"`: albedo is `COLOR_0` times the atlas. Default: the atlas alone. |
| `texture_from` | `"script"`: the script paints the atlas. It defines `texture(name, position, normal)` beside `build()` and authors no UVs: the runner unwraps every mesh into one atlas, asks `texture` for the colour of the point each texel shows and binds the image ([A texture the script paints](#a-texture-the-script-paints)). Default `"plan"`: an atlas plan or a PNG is handed to the build and the script authors the UVs. |
| `texture_size` | The side of a script-painted atlas in pixels, 256 to 4096. Default 1024. |
| `blend` | Also save the scene as `<glb name>.blend` beside each GLB, images packed: the file handed on for texturing, rigging and animation. On a vertex-colour build each mesh without a material is given one that reads its colour attribute, in the file and in the GLB. |
| `judge_variant` | The variant that is rendered and measured. Default: the first. |
| `shots` | Cameras, replacing the four defaults ([Cameras](#cameras)). |
| `stage_scale` | Render scale when there is no `length_m`. Default 1.0. |

The outline table's calibration and the gates on the measures belong to the reference, not the build: they're in
[`criteria`](#criteria).

`out/glb_stats.json` reports `zero_area_triangles` (exactly collapsed) and
`near_degenerate_triangles` (positive area below the relative cutoff). Their sum
is the existing `degenerate_triangles` statistic. The cutoff is computed per
primitive in world space: area below `(max primitive extent * 1e-5)^2`, with the
extent floored at `1e-9` metres. The existing contract fails when the combined
count exceeds 1% of the GLB's triangles. Passing it therefore does not promise
zero collapsed faces. These diagnostic fields do not affect the numeric score.

## Axes

- The script: Blender metres, +Z up, the nose along **+Y**, mirror across X = 0, centred on the origin.
- The GLB: glTF axes, Y up, the nose along **-Z**. The contract's `length_m` is the GLB's Z extent, `span_m` its X.
- The outline tables: model axes in millimetres (+Y nose, +Z up, X across), origin at the bounding box centre.

## Cameras

The model is drawn so its contract length is one unit. A render is measured against the reference view of the same
name. A shot with no reference view is drawn for you to look at.

| Default shot | Camera | Reference view should show |
|---|---|---|
| `hero` | three-quarter, from front-left and above, fov 27 | nose lower-left |
| `top` | from above, fov 28 | nose at the bottom |
| `side` | from the left, fov 19 | nose at the left |
| `rear` | from astern, fov 19 | the tail |

With the default shots, each other variant is also drawn from the hero camera as `hero_<variant>`.

A custom shot:

```json
{"name": "top", "cam_pos": [0, 2.3, 0], "cam_target": [0, 0, 0], "cam_up": [0, 0, 1], "ortho": 1.15}
```

| Key | Meaning |
|---|---|
| `name` | Pairs the render with `refs/<name>.png`. |
| `cam_pos`, `cam_target` | In model units (one unit = `length_m`), glTF axes. |
| `fov` | Vertical field of view in degrees (perspective). |
| `ortho` | Height of the view in model units. Draws the shot orthographically; `fov` is ignored. |
| `cam_up` | Which way the image's top points. Needed for a camera looking straight down. Default `[0, 1, 0]`. |
| `variant` | Which variant this shot draws. Default: the judged one. |

Which outline table a project gets, first that applies:

1. `criteria.outline` (`standing`): the calibrated table, read at named heights through `front`, `side`, `rear`
   and `right`. `hero` and `top` aren't sampled, though `criteria.views` may still score them.
   [ORGANIC.md](ORGANIC.md) has what it needs.
2. `length_m` with `top`, `side` and `rear` views (`profile`): the craft table, read by view name.
3. Otherwise (`outline`): every shot with `ortho` that looks along one model axis and has a reference view of its
   name, read at 11 stations along the longer side of its silhouette: the outline's two ends across, signed, in
   Blender axes from the bounding box centre. The view's name isn't used. Millimetres when the contract declares
   an extent along an axis a view shows (`height_m`, `length_m`, `span_m`, `axis_extents_m`); else thousandths of
   the subject's longest extent, and a GLB is compared at its own size. No such shot: no table.

## sheet

```json
"sheet": {"source": "sheet.png", "mask_sample_x": 8,
          "crops": {"hero": {"box": [15, 40, 1070, 650], "mask": [[15, 20, 530, 170]]},
                    "top": {"box": [1060, 20, 1510, 350]}}}
```

`box` is `[x0, y0, x1, y1]` in sheet pixels. Each `mask` rectangle is painted with the sheet's own background (the
row's colour at x = `mask_sample_x`) before cropping, so a title or a view label doesn't end up in a measure.

## The script

- Defines `def build():` at top level. Imports only `bpy`, `bmesh`, `mathutils`, `math`, `random`, `numpy` and
  `openvdb`. The last two ship in Blender's own Python (numpy 2.3 and openvdb 13 in Blender 5.2); a build of
  Blender without `openvdb` fails such a script at its import.
- No `open`, `exec`, `eval`, `getattr`, `type`, dunder attributes, `bpy.ops.wm` / `render` / `image`, no importer
  or exporter (`bpy.ops.export_scene`, `import_mesh` and the rest), `bpy.app`, `bpy.utils`, and nothing in `numpy`
  or `openvdb` that reads or writes a file (`numpy.load`, `save`, `fromfile`, `memmap`, `numpy.lib`, an array's
  `tofile`, `openvdb.read`, `openvdb.write`). A module's forbidden parts are found under whatever name it was
  imported. The runner refuses the script before running it.
- No file reached through Blender either, so no reference picture is read into the model and no picture is written
  anywhere: no `load` (`bpy.data.images.load`, and the same on fonts, volumes and the other collections), no `save`,
  `save_render` or `unpack`, no `filepath` or `filepath_raw` of anything, and no call given `filepath=` or
  `directory=`. These are refused by name whatever they are called on. An image the script makes itself
  (`bpy.data.images.new`) is allowed.
- The runner wipes the scene, calls `build()`, then exports every object as one GLB.
- Vertex colours are linear RGBA on a `CORNER` `FLOAT_COLOR` attribute. The palette in `jig.json` is sRGB hex.

## A texture the script paints

With `"textured": true, "texture_from": "script"` the script defines one more function at top level:

```python
def texture(name, position, normal):
    colour = np.tile(np.array(PALETTE["orange"]), (len(position), 1))
    belly = (np.linalg.norm((position - BELLY_CENTRE) / BELLY_RADII, axis=1) < 1) & (normal[:, 1] > 0.2)
    colour[belly] = PALETTE["cream"]
    return colour
```

- It is called once for each mesh object after `build()`. `name` is the object's name; `position` and `normal`
  are `(N, 3)` numpy arrays, a row per texel: the point of the object's surface that texel shows, in the Blender
  metres the script built in, and the unit normal there.
- It returns `(N, 3)` sRGB colours 0..1, or one `(r, g, b)` for the whole object. Anything else fails the build
  and says what came back.
- `PALETTE` is a global the runner defines: each palette name as `(r, g, b)` in those units. A texture is sRGB,
  so there is no conversion to linear.
- The runner applies every modifier, unwraps all the meshes into one atlas (islands sized by their area on the
  model, so texels are the same size everywhere), paints it, spreads each island a few texels into the gutter and
  binds the image as the one material's base colour. The atlas is left as `out/<atlas>` and embedded in the GLB.

## An atlas plan

```json
{"size": 1024, "background": "white", "palette": {"moss": "#335522"},
 "ops": [{"op": "rect", "box": [0.0, 0.0, 0.5, 0.5], "color": "crimson"},
         {"op": "line", "points": [[0.0, 0.9], [1.0, 0.9]], "color": "#2c3339", "width": 0.004},
         {"op": "polygon", "points": [[0.6, 0.1], [0.9, 0.1], [0.75, 0.4]], "color": "steel"},
         {"op": "ellipse", "box": [0.6, 0.5, 0.9, 0.8], "color": "cyan"},
         {"op": "stripes", "box": [0.55, 0.85, 1.0, 1.0], "color": "gunmetal", "count": 6, "width": 0.003, "axis": "u"},
         {"op": "speckle", "box": [0.0, 0.5, 0.5, 0.85], "color": "gunmetal", "density": 0.02, "size": 0.003, "seed": 7}]}
```

Coordinates, widths and sizes are 0..1 of the atlas, u right, v down. Colours are the project's palette names, the
plan's own, `black`, `white`, or `#rrggbb`. Any op takes `"alpha"`. The same plan paints the same PNG on every
machine.

## Environment

| Variable | Meaning |
|---|---|
| `MESH_JIG_BLENDER` | The Blender executable. Default: `blender` on PATH, then the usual install places. |
| `MESH_JIG_JUDGE_URL` | OpenAI-compatible endpoint. Default `https://openrouter.ai/api/v1`. |
| `MESH_JIG_JUDGE_MODEL` | Default `deepseek/deepseek-v4.1-flash`. |
| `MESH_JIG_JUDGE_KEY` | Else the provider's own key (see `MESH_JIG_AGENT_KEY`). A localhost URL needs none. |
| `MESH_JIG_AGENT_URL` | Endpoint for `mesh-jig agent` (`--url` wins). Default `https://openrouter.ai/api/v1`. |
| `MESH_JIG_AGENT_MODEL` | Model for `mesh-jig agent` (`--model` wins). No default. |
| `MESH_JIG_AGENT_KEY` | Else the provider's own key, chosen by the URL's host: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` (api.anthropic.com), `GEMINI_API_KEY` (generativelanguage.googleapis.com), else `OPENAI_API_KEY`. The judge falls back the same way. |
| `MESH_JIG_ENV_FILE` | A `KEY=VALUE` file that fills unset variables. Default `.env` in the directory mesh-jig is run from, when it exists (`.env.example` is the template). |
