# Default octopus motion brief

The rig uses eight arm chains and the frozen reference's continuous skin and underside cups. Keep the model's rest geometry and texture unchanged. `rig.json` fixes 24 fps, the skeleton limit, clip durations and deformation gates.

- **crawl**: a low, alternating four-arm support gait in place.
- **walk**: a looping in-place gait with successive arm contacts, visibly higher recovery steps and a small body bob. The ends meet without a pop.
- **leap**: one short jump **forward along Blender +Y** in 1.25 seconds. The body rises, travels forward roughly 0.28 m, tucks the arms in flight, and lands at the new position. It is not a stationary hop.
- **turn**: asymmetric arm motion and a root yaw excursion, returning to the starting pose.

Inspect the exported clips from front, side, top and three-quarter views. The numeric rig gates do not establish the action's readability or certify every mesh collision.
