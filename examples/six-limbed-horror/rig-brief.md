# Six-limbed horror sprint and leap

Rig this builder's frozen textured GLB. Preserve its mesh, rest pose, materials,
and painted texture. Articulate the head, torso, and all six legs. Preserve
attachment around the limb roots, knees, wrists or ankles, and toes. Use the
contract's deformation gates on both clips and inspect the film strips for
pinching, sliding, detached parts, and intersections.

- `sprint` is a one-second in-place loop with zero net root travel. Make it a
  fast six-legged gait, with opposed limb groups, low recovery, distinct stance
  and swing, and readable reach and push-off. Keep the heavy torso low, with
  restrained weight shifts and head stabilization. In-place travel may show
  backward stance-foot motion; avoid a foot floating through its support phase.
- `leap` is a 1.5-second one-shot forward along Blender +Y. Start supported,
  crouch, drive off the legs, and show a clear airborne phase (aim for all
  vertices at least 150 mm above rest ground at the apex). Reach forward into
  landing, absorb the impact through the legs and torso, and settle supported
  about 0.8–1.0 m forward of the start. Leave the travel in the root joint.
  Avoid a rigid-body jump or a slide before takeoff and after landing.

Target three in-contract candidates. Use at most twelve build submissions,
including failures. Pick the one with the clearest gait, airborne phase and
landing while preserving the creature's anatomy. Explain visible tradeoffs.
