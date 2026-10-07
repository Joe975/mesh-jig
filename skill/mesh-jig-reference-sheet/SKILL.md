---
name: mesh-jig-reference-sheet
description: Generate and iteratively review multi-view image reference sheets for mesh-jig modeling, keeping anatomy, components, markings, pose and camera projections consistent. Use before modeling when a new turnaround or repaired reference sheet is requested.
---

# Reference sheets that describe one model

Produce a usable modeling reference, not several attractive variations. This skill covers image generation,
review and refinement; the [mesh-jig skill](../mesh-jig/SKILL.md) covers building the eventual mesh.
Do not launch builders or alter an existing modeling contract unless requested.

## Establish the projection contract

Before generating, write down the subject, intended style, large parts, color zones, fixed pose, view layout,
camera directions and any asymmetric features. Preserve the user's design; choose missing routine details
that make the sheet easier to model. Ask only when a missing choice materially changes the requested result.

- Choose a primary front view for width/height and one true side for depth. Other views must describe that
  same design and pose. Keep orthographic views at the same scale and standing baseline within each row.
- Include the views that resolve the asset: usually front, side, rear, top and three-quarter; add the opposite
  side when handedness matters. A roomy three-column/two-row sheet works well for five or six views.
- Specify **anatomical** left/right, not just page left/right. In mesh-jig, Blender +Z is up and +Y is front;
  anatomical left is X<0, matching the default left-side camera. A left-eye patch appears viewer-right in front.
- Freeze one pose. Parallel feet at equal fore-aft depth and equally held arms are easier to compare than a
  walking pose. Paired limbs/components overlap in a strict side view when their depths match.
- Specify attachments in 3D terms: front-only buckle, rear-center knot, paired lateral stacks, tail in the
  sagittal plane. Say how each projects or becomes occluded, rather than repeating it visibly everywhere.
- A top view is vertical overhead, with front pointing DOWN the page for mesh-jig. It shows footprints and
  correctly occluded surfaces, not an elevated front view. Tall features don't expose long vertical walls.
- Use a plain background, generous gutters, fully contained silhouettes, restrained shading, small labels
  outside silhouettes, and no floor shadows. Keep color zones and large forms readable; avoid unnecessary
  tiny markings or accessories that drift across angles. Do not simplify away requested features.
- Once the sheet is cut into a project's `refs/`, `mesh-jig refs <project> --out <dir>` shows each view as the
  measures see it and flags a view that touches its edge or carries a label or shadow. The repo's `docs/JIG.md`,
  Reference views, is the list the measures hold a view to.

Save the initial prompt with the reference. A useful scaffold is:

```text
Use case: stylized-concept
Asset type: production reference sheet for modeling one [subject] in mesh-jig.
Subject/parts/colors: [specific design, materials and asymmetric features].
Pose: [one frozen pose; paired limbs and attachments positioned in 3D].
Views/layout: [labels, anatomical sides, facing directions, orthographic vs hero].
Invariants: same proportions, pose, component count, markings and clothing in every panel.
Occlusion: [what disappears or overlaps in side, rear and vertical overhead].
Presentation: plain background, no shadows, labels outside fully contained silhouettes.
```

## Generate, inspect, repair

Use the available built-in image-generation tool for new raster sheets and all image edits. Follow the
imagegen skill when available. Inspect local input images before editing them; supply the inspected local
paths to the tool, or use its recent-image mechanism when no local path exists. Do not substitute Python
painting, resizing or compositing for image edits. The audit below only reads pixels.

Save candidates in a versioned project folder, such as `examples/<subject>/reference/`, with numbered
filenames and corresponding exact prompts. Copy the selected output into the repo; don't leave a project
reference only in the tool's generated-image storage. Preserve transparency if present. Keep prior useful
candidates so a later regression can be discarded without losing the best base.

Inspect **every panel after every edit**, including panels the prompt said to preserve. The generator can
change supposedly untouched regions, move an attachment in the opposite direction, or overcorrect a small
requested change. "Preserve exactly" is a prompt instruction, not a verified guarantee.

Review these relationships:

| Check | What must agree |
| --- | --- |
| Proportions and pose | Total extents, head/body ratio, shoulder/hip/knee/ankle levels; side limbs overlap appropriately. |
| Organic anatomy | Same ears, muzzle, paws, limb count and tail; no different body shape invented in rear. |
| Mechanical construction | Same components and attachment locations; distinguish fronts, backs, hinges and openings. |
| Handedness | Patches, tools, scars or fittings stay on the same anatomical side through rotations. |
| Clothing/material zones | Front opening, continuous back, hem/belt order, seams and attachment sides remain compatible. |
| Tail or projecting parts | Same base, size, height, curve and tip marking; distinguish depth foreshortening from a new bend. |
| Overhead camera | Correct footprint, lateral spread, rear extension and occlusion; no visible feet/face by habit. |
| Hero view | Agrees with the orthographic design; its perspective is not used for orthographic size checks. |

Fix the largest contradiction first. Request one localized correction or a small linked group; identify
the edit target, the master view, what changes, and what stays fixed. Keep accepted anatomy/style/pose/labels
as invariants. Do not regenerate the whole character to repair one tail or strap.

An edit request should describe the **desired endpoint** as well as direction. For example: "rear tail tip
must reach the same armpit-relative height as the left-side master; keep its pelvis attachment fixed."
Pixel coordinates can supplement this on a known-size sheet, but are approximate and may be ignored.
Do not compare raw y coordinates between different sheet rows: subtract each view's baseline or row offset.

If edits overshoot and undershoot, use the best candidate as the edit target and a second inspected candidate
as a **positional guide for the single feature**. Ask for an intermediate position, explicitly excluding
style, anatomy and other panels from that guide. Recheck the result; averaging two references is not proof.
If the tool stops making useful progress, retain the best candidate and state the unresolved contradictions
instead of claiming acceptance. No fixed number of edits guarantees consistency.

## Coarse dimension audit

Once the major design contradictions are repaired, measure approximate foreground bounds on label-free
panel rectangles. Compare only dimensions that correspond to the same world axis:

- Standing front/side/rear/opposite-side **heights**.
- Front/rear **width**, and overhead **width**.
- Side/opposite-side **width** (world depth), and overhead **height** (world depth).

Do not require a perspective hero to match these numbers. Check major landmarks separately: identical outer
bounds can hide a wrong face, an arm pose change, misplaced straps or a different tail curve.

Use [scripts/audit_sheet.py](scripts/audit_sheet.py) with a sheet and a JSON config. It writes measurements,
comparison spreads and a coarse pass/fail result, without editing the image. Config panel boxes are
`[x0,y0,x1,y1]` with exclusive upper bounds. Keep labels out and leave space around the subject.

```json
{
  "mode": "chroma",
  "panels": {
    "front": [8,22,504,505], "side": [520,22,1016,505],
    "rear": [1032,22,1528,505], "right": [8,530,504,1010],
    "hero": [520,530,1016,1010], "top": [1032,540,1528,1000]
  },
  "tolerance_pct": 3,
  "comparisons": [
    {"name": "standing height", "axis": "height", "views": ["front","side","rear","right"]},
    {"name": "world width", "axis": "width", "views": ["front","rear","top"]},
    {"name": "world depth", "dimensions": [["side","width"],["right","width"],["top","height"]]}
  ]
}
```

Those boxes are for the pirate-cat sheet only: author new boxes for each layout. Save the config beside the
sheet so the audit can be repeated. From the mesh-jig checkout, with its virtual environment's Python:

```
python skill/mesh-jig-reference-sheet/scripts/audit_sheet.py --sheet local-work/my-model/reference/sheet.png --config local-work/my-model/reference/audit-config.json --out local-work/my-model/reference/audit.json
```

Exit 0 means the configured coarse comparisons pass; exit 1 means a spread exceeds tolerance; exit 2 means
invalid input. Spread is `100 * (largest-smallest) / largest`. Three percent was a useful goal on the pirate
cat, not a universal guarantee or mandatory threshold for all artwork. Choose a tolerance justified by the
intended use, not one loosened to conceal an actual contradiction.

Default `chroma` mode counts visible pixels with channel spread >35 and minimum channel <210. It omits white
background, neutral labels and most fine whiskers, **but also omits neutral gray/black geometry and pale
details**. Verify the reported boxes against the image; don't use it to accept a monochrome mech. For a truly
transparent background, `alpha` mode measures alpha >127, including neutral parts. Opaque white images cannot
use alpha mode as a subject mask. Thresholds are configurable; this is separate from mesh-jig's calibrated
model scoring. Touching a panel boundary is flagged for inspection, not silently interpreted as full extent.
Expand a too-tight measurement rectangle and rerun before accepting its extent; retain the original model
contract crops when builders are already using them, and document any small measurement difference.

## Select and hand off

Accept only after visual relationships and appropriate dimension checks agree at the intended precision.
Write a project review naming the selected file, iteration decisions, master views, handedness, camera
directions, measurements/method/tolerance, and remaining ambiguities. Save exact prompts, candidates and
audit configuration/results in the project's private version control. Present the final sheet to the user.

Describe acceptance as **visually coherent raster reference**, not exact geometric identity. Exact calibrated
projection consistency requires rendering every view from one shared 3D model. Fur, whiskers, seams, highlights
and rivets can remain illustrative if that is compatible with the task; unresolved major anatomy/components
must be called out. Numerical bounds alone do not clear the visual review.

For a requested modeling handoff, use `mesh-jig sheet` with matching crops and camera names, inspect the
actual crops, write a part-by-part brief and freeze the reference contract before comparing builders. This
skill does not authorize launching model runs by itself.

