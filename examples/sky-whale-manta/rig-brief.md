# Sky whale motion contract

Rig the chosen frozen source GLB without changing its geometry, colors or source accessors. Use the same requested Codex model at high effort as the modeling stage. Make three successful in-contract rig builds, with a total ceiling of nine rig-build calls (failures preserved); report a shortfall if the ceiling is reached. One idea per immutable rig attempt.

- `hover_glide`: four-second seamless loop. Calm suspended glide with small breathing/bobbing, gentle fin flex and restrained tail follow-through.
- `fin_flap`: two-second seamless loop. Bilaterally coordinated manta fin stroke with a softer tip lag, visible lift and recovery, and complementary whale tail motion. Keep the face stable and body rounded.
- `gentle_turn`: four-second seamless loop. A gentle bank/yaw excursion and return to the starting heading, with asymmetric fin support and tail follow-through. It is a looping turning gesture in place, not travel around a world path.

Use blended weights on continuous surfaces, smoothly varying fin root/tip and tail influences, and keep eyes/throat details attached to the head. The required bones are root, body, fin_L, fin_R, tail and fluke; extra bones up to 24 are permitted. No ground-drop gate is appropriate for a hovering creature. Loop seams <=2 mm, per-clip movement and deformation limits are fixed in rig.json.

For every attempt run rig-check --readings and inspect front, top and three-quarter filmstrips and the weights sheet. Inspect at least nine equally spaced samples including both endpoints for every clip, then selected inter-key times on the delivered rig. Verify joints, weights, all three exported clips, source preservation, loop continuity, stretch and visible deformations. Preview a playable local animation if supported; otherwise explicitly label the evidence sampled filmstrips, not continuous playback. No claim of animation quality from static numeric measures alone. Preserve remaining root creases, detached markings, fin collapse or unnatural cadence in the report. There is no rig numeric ranking; choose visually and report all gate results.
