# Game-ready characters in another engine

Hand this page to the coding agent that will make a packaged character work in an engine mesh-jig ships no code
for: Unity, Unreal, three.js, Bevy, your own. Keep `src/mesh_jig/game_runtime.py` open beside it. That file is the
specification; this page tells you how to read it, in the order the work is done. [GAME_READY.md](GAME_READY.md)
has the field tables.

You are done when `mesh-jig game-check` passes on traces your engine wrote, for the scenarios below. The checker
does not care how your runtime works inside. It skins the real mesh from your trace and measures it.

## 1. Know what you have

Import `package/character.glb` with your engine's glTF importer. It gives you the mesh, materials, skeleton, skin
weights and every clip, sampled and LINEAR. Check the clips it gives you against the GLB before anything else: an
importer may resample them. Godot's resamples every track at its bake rate (30 Hz by default), and its editor import
dropped keys on the armour at every rate tried, a 2.8 degree error at the walk's swing extreme. If yours does, read
the tracks from the GLB yourself and sample them as `game_measure.Clip.local` does (the Godot addon ships them as
`godot/character.clips.json`). The playback gate catches it. Read `package/character.runtime.json` yourself. glTF carries no animation
graph and no IK, so you rebuild these from the JSON:

| Rebuild | From | Reference |
|---|---|---|
| The blend graph: layers, masks, `blend1d`, `states`, `oneshots` | `graph`, `parameters`, `clips`, `triggers`, `oneshots` | `Runtime.animate`, `Runtime.node`, `Runtime.blend1d`, `Runtime.states`, `Runtime.oneshots` |
| Ground adaptation: planting, the lock, foot height, body height, tilt | `limbs` (role `leg`), `ground`, `body`, `clips[*].legs`, `measurement` | `Runtime.adapt`, `Runtime.lock`, `Runtime.foot_height`, `Runtime.planted`, `Runtime.reach` |
| Leg and limb solvers | `limbs[*].solver`, `joints`, `segment_lengths_m`, `pole`, bend limits | `Runtime.two_bone`, `Runtime.fabrik`, `Runtime.chain` |
| Aims | `aims` | `Runtime.aim` with `game_scenario.aim_angles` and `game_scenario.aim_direction` |
| Limb targets | `limbs` | `Runtime.step` |
| Sockets | `sockets` (offsets in the joint's frame) | attach to the joint, nothing to solve |

Movement, input, physics and navigation stay yours. The runtime never moves the character.

## 2. Get axes, handedness and units right first

Do this before anything else. Every later gate assumes it.

- Everything in the runtime file, scenarios and traces is **glTF axes**: right-handed, +Y up, metres, seconds.
- The character faces **-Z** in model space.
- Yaw is a rotation about +Y, right-handed: positive yaw turns -Z toward -X. Scenario yaw is in degrees.
- Matrices are 4x4 and act on column vectors: `world = root @ model`.
- Quaternions in the GLB are glTF's `(x, y, z, w)`.

**If your engine is right-handed and Y up** (three.js, Bevy, Godot), use every value as it is.

**If your importer flips an axis** (any left-handed engine, Unity among them), find out which axis it negates.
Import a model with a marked left side and look, or read the importer. Call the flip `S`: a diagonal matrix with one
-1, on the flipped axis. Then carry every runtime-file and scenario value through the same `S`:

| Value | Convert |
|---|---|
| A point or direction (positions, contact and clearance points, poles, aim axes, socket offsets, targets) | `p' = S p` |
| A 4x4 or 3x3 matrix | `M' = S M S` |
| A quaternion, flipped axis k | negate the two vector components that are not k (flip X: `(x, -y, -z, w)`) |
| A yaw about +Y, when the flip is X or Z | negate it |
| A heightfield over x and z | negate the flipped coordinate; heights are unchanged by an X or Z flip |
| `ground(x, z)` your runtime calls | convert the query point into your space and the answer back |

Points in a joint's frame (contact, clearance, socket offsets) convert with the same `p' = S p`, because the joint
frames convert by `S M S`. `S` is its own inverse. Convert back with it when you write the trace: **the trace is
always glTF axes**.

A worked example. Say your importer negates X. The pole `[0, 0, -1]` stays `[0, 0, -1]`. A target at
`[0.3, 1.0, -0.5]` becomes `[-0.3, 1.0, -0.5]`. A root yaw of 30 degrees becomes -30. The joint quaternion
`(0.1, 0.2, 0.0, 0.97)` becomes `(0.1, -0.2, -0.0, 0.97)`. Your engine's world matrix `W'` for a joint goes into the
trace as `S W' S`.

How the playback gate catches a wrong flip: a mirrored character has its left and right swapped. In the walk, the
left foot touches down at one phase and the right half a cycle later, so a mirror puts every foot in the wrong place
and `playback_mm` reads centimetres. Check playback on a walk, not an idle: an idle can be nearly symmetric and hide
it. Converting a value twice, or a vector but not its matrix, shows the same way.

## 3. Implement the interface

Write these, with these names if your language allows:

| Call | Effect |
|---|---|
| `set_parameter(name, value)` | Takes effect this frame. No smoothing: a game that wants a smooth change ramps the value. |
| `trigger(name)` | Fires matching `states` transitions on the next `step`, then is forgotten. |
| `play(name)` | Starts that one-shot on the next `step`; a replay restarts it. |
| `set_aim(name, point or none)` | World point; none returns the aim to the animation at its turn rate. |
| `set_limb_target(name, point or none, weight)` | World point; weight 0 to 1, applied as given each frame. |
| `set_ground_enabled(on)` | Ground adaptation fades in or out over `ground.blend_s`; its state before the first `step` applies at once. |
| `step(dt, root_world_matrix, ground) -> joint world matrices` | One frame. |

The runtime asks the game one thing: `ground(x, z) -> (height, normal, walkable)`, glTF world axes. In a game that
is a raycast down. In the scenario player it is the heightfield. The reference only uses the height.

## 4. Do the work in this order each frame

`Runtime.step` is the order. Keep it.

1. **Graph** (`Runtime.animate`): advance clocks by `dt`, evaluate each layer, lay it over the layers below by its
   weight on the joints in `mask_joints` (every animated node when it is null), and collect each leg's stance vote.
   Clear pending triggers and plays.
2. **Forward kinematics** (`Runtime.place`): joint world = root world matrix x model-space matrix through the glTF
   node hierarchy, including nodes that are not joints.
3. **Ground adaptation** (`Runtime.adapt`), when the package has legs and a `ground` block.
4. **Aims** (`Runtime.aim`), in the runtime file's order.
5. **Limb targets** (the loop at the end of `Runtime.step`), in limb order.

Section 9 has every rule inside each step that you cannot read off a field table.

## 5. Know what may differ and what may not

**May differ.** The gates do not compare your pose to the reference's, except playback.
- How you blend rotations (nlerp, slerp, your engine's mixer).
- Solver internals: your engine's two-bone or FABRIK node, its iteration count, its pole handling.
- Smoothing curves, if the gates still hold.
- Performance tricks: caching, threading, fewer ground queries. The reference asks the ground often: for every
  contact and clearance point of every leg, and ten times a point while a leg swings. The armour asks about 320
  times a frame and the six-limbed horror about 1530; in GDScript that is 1.0 and 3.4 ms a frame.
- Floating-point precision, within reason. The reference is double precision. In a single-precision engine take dot
  and cross products and rotation angles in double: a single-precision dot of two unit vectors under 3.4e-4 rad apart
  is exactly 1, so a shortest-arc turn read from it vanishes (0.15 mm at a shin in the Godot port).

**May not differ.** The checker depends on these.
- The scenario clock: frame times exactly as `game_scenario.frame_times` makes them.
- Event order: every event at a frame's time is applied, in file order, before that frame's `step`.
- The trace format, layout, axes, joint order and hashes (section 6).
- Single-clip playback with the ground off on flat ground: the pose must be the GLB's own clip at
  `(t + offset_s) mod duration`.
- That the runtime never moves the root.

## 6. Write a scenario player and a trace

Port `game_runtime.run`. It is about thirty lines.

1. Read the scenario file's bytes. `scenario_sha256` is SHA-256 of exactly those bytes, as hex.
2. Frame times (`game_scenario.frame_times`): start at 0. Take the `dt` list in turn, repeating. Cut a step short
   when it would pass an event time or the duration (within 1e-9 s), so it lands exactly there. Stop at the
   duration.
3. For each frame: apply every event whose time is at or before this frame's time (plus 1e-9), in file order. A
   `terrain_patch` raises the grid points inside its rectangle (edges included) by `offset`, from then on.
4. Root (`game_scenario.root_at`, `root_matrix`): interpolate `[t, x, y, z, yaw_degrees]` linearly, yaw included.
   The matrix is translation times `[[c, 0, s], [0, 1, 0], [-s, 0, c]]`, with c and s the cosine and sine of yaw.
5. Call `step(dt, root, ground)`. `dt` is 0 on the first frame and the gap to the previous frame after.
6. Write the frame.

The heightfield (`game_scenario.Ground`) is a grid over x and z. Split each cell along its (x0, z0)-(x1, z1)
diagonal. With `u`, `v` the cell fractions in x and z and corner heights `a` (x0, z0), `b` (x1, z0), `c` (x1, z1),
`d` (x0, z1): the height is `a(1-u) + b(u-v) + c v` when `u >= v`, else `a(1-v) + c u + d(v-u)`. Outside the grid,
clamp to its edge.

The trace is one JSON object:

```json
{"format": "mesh-jig game trace", "version": 1,
 "runtime": {"name": "your runtime", "version": 1},
 "hashes": {"glb_sha256": "...", "runtime_sha256": "...", "scenario_sha256": "..."},
 "joints": ["root", "hips", "..."],
 "matrix_layout": "4x4 world matrices as 16 numbers in column-major order ...",
 "frames": [{"time": 0.0, "root": [16 numbers], "joints": [[16 numbers], "..."]}]}
```

- `glb_sha256` and `runtime_sha256`: SHA-256 of the package's two files, as hex.
- `joints`: the runtime file's `skeleton.joints` names, in that order. One matrix per joint per frame, same order.
- Matrices are column-major (glTF's order): numbers 0 to 3 are the first column, 12, 13, 14 the translation.
- Every value is in glTF axes and metres. `time` is the frame time in seconds; it must match the scenario's frame
  within 1e-6.
- The reference rounds matrices to 7 decimals and times to 9. You need not, but deterministic bytes make a port
  easy to diff.

Run `mesh-jig game-check --package package --scenario s.json --trace yours.json`.

## 7. Get the gates passing in this order

Start every scenario from the [examples](../examples/game-ready/) or `tests/gamekit.py` (`walk`, `still`), which
write them for the in-test fixtures. Fix each step before the next.

1. **Playback, ground off, flat ground, one clip.** Set the speed parameter to the walk's measured value and
   `expect` the walk. Failures: axes or handedness (section 2), clip time (the clock starts at 0 and the synced
   phase adds `phase_offset`), sampling (LINEAR between keys, shortest-arc for rotations), the wrong FK order.
2. **Ground on, flat ground.** Contact gap, slip and support should hold. Slip failing means the lock is wrong.
   Contact gap failing means the foot height or the stance vote is wrong.
3. **Slope.** Penetration or contact gap failing usually means the foot height ignores the ground under each point,
   or the body is not lowered for reach.
4. **Step.** Penetration from a toe means clearance points or the look-ahead are missing. A joint-speed spike means
   the foot height jumps: the look-ahead weight must fade in and out.
5. **Mid-run raise.** `recovery_mm` failing means the runtime cached the ground. Ask every frame.
6. **Aims.** `limits_deg` failing means yaw and pitch are not clamped, or not measured in the parent's frame from
   the rest axis. `aim_error_deg` failing means the turn rate is too slow for `settle_s`, or the turn is not shared
   along the chain.
7. **Limb targets.** `limb_target_error_mm` failing on a limb under an aim chain means aims ran after limbs.
   `stretch_mm` failing means the solver moves the tip instead of turning joints.
8. **Variable dt** (a `dt` list of mixed steps). Anything that fails only here is a per-frame constant that should
   be per-second: smoothing must be `1 - exp(-dt / blend_s)`, rates must multiply by `dt`.

`joint_speed_mps` failing anywhere means a pop: a fade that snaps, a weight that jumps, or a solver that flips.
`deepest_penetration` in the report names the node and joint that went into the ground.

## 8. Use your engine's own graph and IK, and still prove it

You need not port the reference. Build the graph in your engine's animation tree (blend spaces, state machines,
layers with bone masks) and use its IK nodes, if the trace holds.

- Map `blend1d` with `sync` to a synced blend space, but check the sync rule: phases align at each clip's
  `phase_offset`, the cycle frequency lerps between the two points on the cycle, and a point with `cycle: false`
  runs on its own clock. Many engines lerp durations instead; that slips more and may fail `slip_mps`.
- Map `states` to a state machine with linear cross-fades of `fade_s`, `oneshots` to a layer you fade in and out.
- Engine two-bone IK nodes usually need the pole as a world point: use the hip plus the runtime-file pole.
- Engine foot-IK nodes rarely know about the lock, the contact's animated clearance or clearance points. Feed them
  targets computed as in `Runtime.adapt`.
- Write the trace from your engine's final bone world matrices, converted to glTF axes. Then run `game-check`.

## 9. The exact rules

These are the places where the reference's behaviour is not obvious from the field tables. Each names its code.

**Clocks and the graph**
- A node's clock is 0 on the frame it starts and grows by `dt` each frame after (`Runtime.clock`). Clips loop by
  `t mod duration` or hold their last pose (`Runtime.clip_time`).
- `blend1d` picks the two points either side of the parameter: at or below the first value it is the first pair
  with weight 0; at or above the last, the last pair with weight 1; else the pair whose lower value is the largest at
  or below the parameter. Unsynced points share the blend's own clock. A nested node's own weight is not used inside
  a blend.
- A synced blend keeps a phase that starts at 0 and grows by `dt` times this frame's frequency: the lerp of the two
  points' `1/duration` when both are on the cycle, the one's when one is, 0 when neither. A point on the cycle plays
  at `((phase + phase_offset) mod 1) * duration`; a point with `cycle: false` at `blend clock mod duration`.
- A `states` node checks its transitions, in list order, before evaluating; the first that fires wins. `on` fires
  if that trigger was called since the last frame. `at_end` fires when the source clip's clock, as of the last frame,
  is at or past its duration. A transition resets the destination's clocks; the source keeps playing through the
  fade. The fade is `elapsed / fade_s`; elapsed is 0 on the transition's frame and grows by `dt` after.
- Fading from a state to a null state lays the old pose over the layers below at weight `1 - u`; from null to a
  state, the new pose at weight `u`; between two states, their blend at weight 1.
- A one-shot's clock is 0 on the frame `play` takes effect; its weight is
  `max(0, min(1, tau / fade_in_s, (duration - tau) / fade_out_s))`; it stops once `tau > duration`.
- Triggers and plays are forgotten at the end of each frame's graph pass, used or not.
- The stance vote (`Runtime.planted`): every clip that reached a leg's tip joint, through layers whose mask covers it
  (or no mask), counts by its share times the layer weights over it. A leg is planted when the shares of clips whose
  `stance_phase` windows hold the clip's phase reach half the total. A window whose end is past 1 wraps.

**Ground adaptation** (`Runtime.adapt`)
- The ground weight starts at 1 or 0 from the enabled state before the first frame, then moves toward on or off by
  `dt / blend_s`. At 0 every leg state, the body offset and the tilt reset.
- The animation's ground in the world is the height of the point `(0, rest ground, 0)` of model space, placed by the
  root matrix (the root's height plus the rest ground, for a root that only yaws). The reference reads the rest ground
  from the GLB as the rest pose's lowest vertex; `measurement.rest_ground_y` is that number rounded to six decimals
  (packages built before it was added lack it, and a port can read the GLB). A contact point's animated clearance is
  its height above that ground.
- A contact point is touching when its animated clearance is at most `measurement.contact_gap_m`.
- The lock (`Runtime.lock`) holds while the leg is planted or any contact point touches. On a frame it holds that
  follows a frame it held, it subtracts the mean horizontal movement of the contact points that touched in both frames
  (if any did) and subtracts the root's change of yaw (`atan2(R[0,2], R[2,2])`). Then, in this order: when not held,
  or when the leg is out of reach, shift and turn are multiplied by `exp(-dt / blend_s)`; then the shift's length is
  capped at `relock_m` (the turn is not capped). It then stores this frame's points, which touched, whether it held
  and the root's yaw for the next frame.
- Out of reach: the horizontal distance from the leg's first joint to the animated contact centre plus last frame's
  shift is more than the leg's reach (`Runtime.reach`: `0.999` of the length, or for `two_bone` the length its upper bend
  limit allows if that is less).
- The foot's target rotation is `yaw(turn)` times the tip's animated world rotation. Its target XZ is the animated
  contact centre plus the shift, less the centre of the contact points turned by that target rotation (the tip, not
  the contact centre, is what moves).
- The foot's height (`Runtime.foot_height`). `lift` is the lowest animated clearance of the contact points, at least
  0. The points are the contact points then `clearance.points`, all in the tip's frame, turned by the target rotation
  and placed at the target XZ. Each point needs `ground(x, z) + lift - (its offset's y)` of the tip, and the tip's
  height is the largest need: `lift` applies to the clearance points too. `clearance.heights` is information only;
  the runtime does not read it.
- The look-ahead. The foot's velocity is (target XZ - last frame's target XZ) / `dt`, updated only when there was a
  last frame and `dt > 0`, and 0 until then. Its weight `ahead` steps by `dt / blend_s` toward 0 while the vote says
  planted and toward 1 while it does not, clamped to 0..1 (the vote, not the lock). The step `d` is velocity times
  `lookahead_s`, and with `n = 8` samples (`LOOKAHEAD_SAMPLES`): when `ahead > 0` and `|d|^2 > 1e-12`, read
  `back = ground(p - d/n)` and for k = 1..n, `s = k/n`, `rise = ground(p + s d) - (here + k (here - back))`, and the
  point needs at least `here + ahead (1 - s) rise` in place of `here`. Ground continued from behind is subtracted, so
  a uniform slope adds nothing. All of this state (velocity, last XZ, weight) clears with the leg state when the
  ground weight reaches 0.
- The body: `want` is the smallest difference between a planted leg's target height and its animated tip height
  (all legs if none is planted), clipped to the drop and rise limits and smoothed by `1 - exp(-dt / blend_s)` (set at
  once on the first frame). `bound` is the largest offset at which every one of those same legs (planted, or all) still
  reaches its target from its first joint as animated:
  `min(target_y - hip_y + sqrt(max(reach^2 - horizontal distance^2, 0)))`. The body joint moves by the ground weight times `clip(min(smoothed want, bound))`.
- Tilt: with `tilt` on and three or more planted legs, the goal is the rotation from +Y to the upward normal of the
  plane fitted through those legs' tip targets by orthogonal least squares (the direction of least spread of the
  centred points: the last right singular vector, or the covariance's eigenvector of least eigenvalue); otherwise none. It follows the goal by slerp with
  `1 - exp(-dt / blend_s)`. The body joint's subtree turns about the body joint by that tilt laid on by the ground
  weight (slerp from no rotation to the tilt by the weight).
- Each leg is solved toward its current tip plus the ground weight times (target minus current tip). The tip then
  turns toward its target rotation by the ground weight (slerp from no rotation to the needed turn).

**Solvers**
- `two_bone`: the pole is the runtime-file pole turned by the root's rotation only. The reach is clamped between the
  lengths the lower and upper bend limits allow, and to 0.999 of the full length. If the pole lies along the leg, the
  current knee plane is kept. Each bone turns by the shortest rotation onto its new direction.
- `chain`: FABRIK, root pinned, 30 passes; a target at or beyond the full length straightens the chain at it. Joints
  turn root first, each by the shortest rotation onto its solved segment; a bend past `bend_degrees_max` from the
  segment before is held at the limit.
- Aims: the frame is the world rotation of the first aim joint's parent (the root's if none). The rest axis in that
  frame is the parent's rest rotation transposed times `axis_model`. Yaw and pitch come from
  `game_scenario.aim_angles`. The aim keeps an offset from the animated angles. Each frame the offset moves toward
  (the clamped target angles minus the animated ones), or toward zero with no target, by at most `degrees_per_s * dt`
  measured as the length of the (yaw, pitch) step. With a target, the final angles are clamped to the limits. The
  turn from the animated axis to the final one is shared equally: each joint, root first, turns by
  `angle / joint count` about the same world axis, through its own position.
- Limb targets: after aims, a limb with a target is solved toward its current tip plus the weight times (target
  minus tip). The tip keeps its world rotation.

**The scenario player**
- Read the scenario's numbers exactly. Godot's JSON parser reads `0.016666666666666666` (1/60 as Python writes it)
  one unit in the last place low, so its frame clock drifts from the reference's by such units. Since the lock holds
  while a contact touches, that moves a gate value by under 1% (a clock nudged 1e-9 s either way, armour and biped
  scenarios; `tests/test_game_runtime.py` holds it), but traces then differ joint for joint by up to 2 mm for a frame
  where a phase lands exactly on a stance window's edge. To compare a port joint for joint, give it the numbers bit for
  bit (the Godot harness takes a sidecar of IEEE-754 bits).

## 10. What is out of scope

- Physics: balance, falls, ragdoll, pushes. Contacts are geometric.
- Root motion, navigation, a player controller.
- Humanoid retargeting and mocap. The rigs are generic; joint frames are parallel to the model's axes at rest.
- Sole pitch and roll to the ground normal; footstep choice (moving a foot off an edge); keeping shins and bodies
  out of risers.
- Additive layers, 2D blends, turning in place.
