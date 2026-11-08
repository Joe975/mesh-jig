# Frozen caterpillar motion brief

Rig the selected generation GLB without changing its geometry or colours.
Freeze source.glb and its SHA256 before the first rig attempt. Use the same
requested native model at high effort as built it. Do not compare another
builder's rig or change this contract. Exactly three successful in-contract
rig attempts, preserving failures separately. At most twelve submitted rig
builds total: if that ceiling is reached, report incomplete rather than relax gates.

- idle, 3 seconds looping: subtle breathing and asynchronous frond sway. The face
  remains readable, the body rests on its prolegs, and roots stay attached.
- crawl, 2 seconds looping in place: a readable rear-to-front contraction wave,
  staggered short proleg lift/plant, small head response and secondary frond sway.
  Do not substitute whole-creature bobbing for locomotion. Root net translation
  is zero so the same clip can be placed by an engine later.
- gentle_turn, 3 seconds looping: head leads a modest leftward bend, trunk and
  fronds follow, then settle back to the initial heading. This is an anticipation
  and return turn, not accumulated root yaw or a spin.

Use rig-info on the frozen source, rig-build with --failed for every candidate,
rig-check --readings, and rig-preview for front, right, top and three-quarter
views. Inspect every clip's filmstrips and the weights sheet after each attempt.
Also inspect at least one ordered playback or dense contact sheet spanning a
full cycle of every selected clip. Report that as sampled visual motion evidence,
not a continuous-video or anatomical-quality score. Check all exported joints,
normalized nonnegative weights, clips/durations, unchanged source accessors,
loop seams, ground drop, stretched area, separated contacts and frond attachment.
Explicitly note remaining sliding, interpenetration or volume-loss defects;
existing static/rig numeric checks do not certify these.
