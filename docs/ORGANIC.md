# Standing characters and organic construction

Use [JIG.md](JIG.md) for the field schema and the
[configuration template](../examples/character-contract/jig.example.json) as a starting point.
Replace its placeholder cameras, dimensions, palette and regions for your reference.

Set `model.height_m` and explicit front, side, right, rear and top cameras. Choose the scored views in
`criteria.views`. Use orthographic references for metric comparisons. The exported GLB uses Y up and
front along -Z; model scripts build in Blender coordinates as described by `mesh-jig brief`.

Run `mesh-jig refs <project> --out <directory>` to inspect silhouette boxes. For a standing outline set
`criteria.outline.mode` to `standing`, give `calibration.view`, its `height_px` and `origins_px` in the
original reference pixels, then specify named stations with `z_m`. Calibration origins must represent
the same ground or explicitly declared origin in every view. Freeze these settings before measuring attempts.

Define `criteria.review.regions` for features that matter. Supply `min_iou` or `min_zone` to make a region
an acceptance gate; otherwise it is an observation. Named parts, landmark bounds, visibility and clearance
probes need metadata that matches actual exported geometry. Run `doctor` and read every warning.

The [construction recipe](../examples/organic/recipe.py) demonstrates shape fusion and colour storage.
It is a reusable technique, not a completed asset. Joining objects alone does not fuse their surfaces.
Inspect silhouettes, depth, facial features, attachments and shading from several views. A score cannot
establish clean topology, riggability or convincing anatomy.
