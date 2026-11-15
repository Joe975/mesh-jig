# Game-ready characters

This stage takes a rigged, animated GLB and a `character.json` contract. It measures the clips from posed
vertices, holds them to gates and writes a package a game can use: the GLB, a neutral runtime description and a
report. A reference runtime plays the package through a scenario that stands in for a game, and an independent
checker holds the resulting trace to gates. These are slices 1 to 4 of the [plan](GAME_READY_PLAN.md): the package,
the reference runtime and checker, the [Godot addon](#godot), and the porting page. To make a package work in another
engine, hand your coding agent [GAME_READY_ENGINES.md](GAME_READY_ENGINES.md).

The manual contract describes different bodies through optional blocks: a blend graph, limbs, ground adaptation,
aims and sockets. A biped can declare two legs, a caterpillar sixteen, a turret none. Each rig and engine still
needs its own measured checks.

## Commands

For bounded API preparation of a frozen rig, start with [the game-agent workflow](GAME_AGENT.md).
The manual commands below support the wider contract. Agent/browser playback currently supports two-bone ground
legs and clip/state/blend1d graphs; other features require an engine port and its own checks.

```powershell
mesh-jig game-info --rig animated.glb
mesh-jig game-info --rig animated.glb --contract character.json
mesh-jig game-build --rig animated.glb --contract character.json --out package
mesh-jig game-run --package package --scenario walk-slope.json --out trace.json
mesh-jig game-check --package package --scenario walk-slope.json --trace trace.json --report check.json
mesh-jig game-preview --package package --scenario walk-slope.json --trace trace.json --out look
```

- `game-info` prints what an author needs: joints with parents and rest positions, mesh nodes with their rest
  boxes, clips with duration and loop, for each clip the vertex clusters that touch the ground and when, and the
  range of every joint's bend in each clip (so `bend_degrees` is read, not guessed). A
  cluster planted part of each cycle is a foot; its node and box are a contact. With `--contract` it also checks
  the contract against the GLB.
- `game-build` writes `package/`. The output directory must be new or empty. Exit 0: every gate held. Exit 1: a
  gate failed; only `character.report.json` is written, so nothing looks delivered. Exit 2: invalid input (an
  unknown key, a name the GLB lacks, a wrong hash); nothing is written.
- `game-run` plays a scenario with the reference runtime and writes the trace. The trace file must be new.
- `game-check` skins the package's mesh from the trace and holds it to the scenario's gates. It does not use the
  runtime. Exit 0, 1 or 2 as above. Any runtime that writes a trace in this format can be checked: that is how a
  port to another engine proves itself.
- `game-preview` draws the trace on the scenario's heightfield: `filmstrip.png` (a three-quarter row and a side row)
  and `run.gif`, from one camera for the whole run.

Keep trials under ignored `local-work/`. Start from the
[example contract](../examples/game-ready/character.example.json), written for the pixel armour. Its names,
boxes and limits are that model's, not defaults.

## The package

| File | What it is |
|---|---|
| `character.glb` | The input, byte for byte. |
| `character.runtime.json` | The neutral description, in **glTF axes** (Y up, metres, right-handed). |
| `character.report.json` | Every gate, reading and measurement, with hashes. |

The contract is in Blender axes; the runtime file is not. A Blender `(x, y, z)` becomes glTF `(x, z, -y)`. An
engine that is left-handed (Unity) flips one axis on import and must flip every runtime vector the same way.

`character.runtime.json` holds:

- `skeleton`: every joint by name with its glTF node index, parent, rest translation, rotation and scale, and rest
  position. Enough for FK without reading the GLB's node tree.
- `clips`: per clip the duration, loop, whether it is locomotion, travel speed, root travel, loop seam, events,
  and per leg the duty, stance windows and touchdowns as phases (0 to 1 of the cycle; a stance that ends past 1 wraps), speed and spread.
- `graph`: the contract's layers with `"measured"` replaced by the number, `measured: true` beside it, each synced
  point's `phase_offset` and `cycle` (`false` for a clip in which no leg touches down, such as an idle: it runs on
  its own clock), `sync_clock: "frequency_lerp"` on synced blends, and each mask expanded to `mask_joints`.
- `limbs`: joints, segment lengths, solver settings, a two-bone pole in glTF model axes, and for a leg its contact
  points and its `clearance` points in the tip joint's rest frame. Clearance points are the rest of the foot: the
  vertices the tip joint carries most, less the contact, rounded to a micrometre and made unique, then at most 32
  by farthest-point sampling (from the first point in lexicographic order, each next point the one farthest from
  those chosen). `clearance.heights` gives each one's height above the contact plane (the contact's lowest rest
  point).
- `body`, `ground`, `aims` (axis in model and in the last joint's frame), `sockets` (offset in model and joint
  frame), `parameters`, `triggers`, `oneshots`.
- `blending`, `measurement` (`sample_hz`, `contact_gap_m`, and `rest_ground_y`: the rest pose's lowest vertex, glTF
  y, which the runtime's animation ground is measured from), `capabilities` (in words) and `hashes` (GLB, contract,
  measurement code).

Blocks the contract leaves out appear as empty lists or `null`, so the file always has the same keys.

## Contract v1

Blender metres: +Z up, +Y forward, the model's own left is X<0. Unknown keys are refused at every level. Only
`version`, `rig_sha256` and `graph` are required.

| Field | Meaning |
|---|---|
| `version` | `1`. |
| `rig_sha256` | SHA-256 of the GLB. `game-info` prints it. |
| `forward` | Horizontal three-vector, default `[0, 1, 0]`. Travel speed is measured along it. |
| `parameters` | `{name: {min, max, default}}`. A `blend1d` must name a declared parameter. |
| `graph` | `{layers: [{name, node, mask?}]}`. Layers apply in order; a later layer replaces the joints in its `mask` (each named joint and its descendants; no mask is every joint). |
| `body` | `{joint}`: what ground adaptation raises, lowers and tilts. Required with `ground`. |
| `limbs` | `[{name, role, joints, solver, ...}]`. Each joint is the direct child of the one before. Limbs share no joints. |
| `ground` | `{drop_m_max, rise_m_max, tilt, relock_m, blend_s, lookahead_s}`: run-time settings, carried to the runtime file. `lookahead_s` is how far ahead in time a swinging foot reads the ground. Needs legs and `body`; `tilt` needs three or more legs. |
| `aims` | `[{name, joints, axis, yaw_degrees, pitch_degrees, degrees_per_s}]`. `axis` is the last joint's pointing direction at rest, in model axes. Yaw within -180..180, pitch within -90..90. |
| `sockets` | `[{name, joint, offset}]`: offset at rest, in model axes. |
| `criteria` | Exactly the limits this contract's gates use (table below). A missing one and an unused one are both refused. |

Graph nodes:

| Type | Keys |
|---|---|
| `clip` | `clip`. |
| `blend1d` | `parameter`, `points: [{clip or node, value}]` (two or more), optional `sync`. `value` is a number or `"measured"` (the clip's travel speed; needs legs). A synced blend's points are clips that loop. |
| `states` | `start`, `states: {name: node or null}` (`null` adds nothing), `transitions: [{from, to, on or at_end, fade_s}]`. `on` names a trigger. `at_end: true` leaves a state that plays one clip. |
| `oneshots` | `clips: {name: {fade_in_s, fade_out_s}}`, each fired by name. |

Fades must be longer than zero. Additive layers and 2D blends are later.

Limbs:

| Key | Meaning |
|---|---|
| `role` | Any name. `leg` bears weight, takes part in ground adaptation and needs `contact`; only a leg has one. A leg may use any solver: a three-bone leg is a `chain` of four joints. |
| `solver` | `two_bone` (exactly three joints; needs `pole` and `bend_degrees: [low, high]` within 0..180), `chain` (two or more joints; needs `bend_degrees_max`), or `none`. |
| `contact` | `{node, box: [lo, hi]}`: the mesh node's vertices inside the box at rest. It must select vertices. Two contacts must not share a vertex. |

A clip loops when the GLB's animation `extras.loop` says so (the rig stage writes it). A clip with no such flag
loops when it is a `blend1d` point. Locomotion clips are the clips of `blend1d` points.

## What is measured

Clips are sampled 120 times a second. A loop is sampled over one cycle and wraps.

- **Ground and planted.** The ground is the rest pose's lowest vertex, as `rig-check` uses. A leg is planted in a
  sample when its lowest contact vertex is within `contact_gap_mm_max` of that ground, in a run of planted samples
  at least `stance_s_min` long. A shorter touch is a graze: no stance, no touchdown, and a reading. Duty and stance
  windows follow from this, so they depend on the gap. A dip below the ground still counts as planted, and shows
  as `below_rest_ground_mm`.
- **Planted velocity.** The mean horizontal velocity of the contact vertices that are within the gap at both ends
  of a sample step. A foot rolling on its toe counts only the vertices still on the ground.
- **Per leg per clip:** stance windows, touchdowns (interpolated where the contact crosses the gap), duty, the
  median backward speed of the planted contact and its spread (standard deviation), grazes, and skate (RMS over
  all planted samples of the difference from the travel speed).
- **Per clip:** duration, loop seam, root travel, how far the model goes below the rest ground, and travel speed:
  the median over every leg's planted samples together. The median follows flat-foot stance; skate still reads a
  heel-toe roll that slides.
- **Per synced blend:** the reference leg (the first leg in contract order that touches down in every clip that
  steps), each clip's phase offset (that leg's first touchdown over the duration; a clip with no touchdowns has
  offset 0), each leg's touchdown phases after the reference, and the half-and-half blend of each neighbouring
  pair. The blend is done on joint local transforms (translations and scales lerp, rotations shortest-arc nlerp)
  and then skinned. The clips that step share one phase, and their cycle frequencies lerp. A clip in which no leg
  touches down (an idle) is off the cycle: it runs on its own clock and adds nothing to the frequency or phase, so
  between an idle and the first gait the gait plays at its own frequency. The planted slip is read at the mean of
  the two travel speeds.
- **Per fade:** the largest distance between the layer's mask joints at the fade's two ends, with joints outside
  the mask held at rest. A null state, and a one-shot's base, is the layers below at every parameter's default.
  A trigger can fire at any time, so the worst time is used.

## Gates

| Gate | Criterion | Reads |
|---|---|---|
| `loop_seam_mm` | `loop_seam_mm_max` | Each looping clip: the largest vertex distance between its first and last pose. |
| `root_travel_mm` | `root_travel_mm_max` | Each locomotion clip: horizontal travel of the skeleton root and body joint, start to end. Locomotion plays in place. |
| `legs_planted` | must be 0 | Each locomotion clip: legs that never come within the gap of the ground. |
| `skate_mps` | `skate_mps_max` | Each leg in each locomotion clip: RMS of the planted velocity's difference from the travel speed backward. This is what skates once the game moves the root. |
| `points_in_order` | must be 0 | Each blend1d: values missing, outside the parameter's range, or not strictly increasing. |
| `gait_phase_difference` | `gait_phase_difference_max` | Each synced blend: the largest difference between two neighbours' touchdown phases, leg by leg, as a share of the cycle. A leg that touches down a different number of times reads 0.5. |
| `blend_slip_mps` | `blend_slip_mps_max` | Each synced blend: planted slip of the half-and-half blend of two neighbours. |
| `fade_joint_speed_mps` | `fade_joint_speed_mps_max` | Each transition and one-shot fade: the pose distance over the fade time. |

`contact_gap_mm_max` and `stance_s_min` are no gates. They say how near the ground a contact vertex is planted
and how long a touch must last to be a stance. Both are required whenever there is a leg. Every gate in the report has its value, limit, worst item and every item.

## Readings

Readings never fail a build.

- `unused_joints`: joints no block names (body, limbs, aims, sockets, masks).
- `missed_contacts`: vertex clusters that touch the ground in a locomotion clip and belong to no contact. Clusters
  are grouped by the joint that carries each vertex most, then joined through shared triangles, welded positions
  and neighbouring cells. This names a leg the author left out. It also shows a contact box that misses part of a
  sole.
- `rigid_limbs`: limbs of three or more joints whose every inner angle moves less than 1 degree over the
  measured clips.
- `grazes`: per leg per clip, each touch shorter than `stance_s_min`, with its time and its lowest clearance.
- `bend_outside`: a clip whose own bend leaves a limb's declared range (a two-bone's inner angle outside
  `bend_degrees`, a chain's bend past `bend_degrees_max`): the IK would hold the clip somewhere it does not go.
- `below_rest_ground_mm`: per clip, how far the lowest vertex goes below the rest ground, sampled at 120 Hz so a
  sag between keys shows.

## Limits

- Duty and stance windows depend on `contact_gap_mm_max`. A swing that lifts slowly stays within the gap longer.
  A foot that rolls heel to toe is planted through the roll; if the rolling edge moves, skate reads it.
- The ground is the rest pose's. A clip that crouches or crawls lower than its rest pose reads deep below it,
  and a body whose rest pose floats has its ground at its lowest point, not at its feet.
- A half-and-half blend on joint rotations does not put the foot half way between the two clips' feet. A leg
  shortens, and the foot can go below the ground; the blend slip reads what that does.
- Only `JOINTS_0` and `WEIGHTS_0` are read (four influences). Joints and animated nodes need translation,
  rotation and scale, not a matrix. Morph targets are ignored.
- `game-build` runs nothing. The runtime and checker below are what hold ground adaptation, aims and limb targets.

## Scenarios

A scenario is JSON in **glTF axes** (Y up, metres, right-handed; the character faces -Z). It stands in for the game:
the player moves the root, makes the calls and hands the runtime the ground. Unknown keys are refused.
[Example scenarios](../examples/game-ready/) are written for the armour package built from
`character.trial.json`: the example contract with two clip limits raised as trial limits (`skate_mps_max` 0.3,
`blend_slip_mps_max` 0.9) so the armour, whose clip gates fail at 0.05 and 0.1, can be packaged and played.

| Field | Meaning |
|---|---|
| `version` | `1`. |
| `terrain` | `{x, z, heights}`: increasing grid coordinates (2 to 257 each) and a height row for each z. Each cell is split along its (x0, z0)-(x1, z1) diagonal; the runtime's ground and the checker use the same surface. |
| `dt` | Frame steps in seconds (0.001 to 0.1), used in turn and repeated. |
| `duration` | Seconds. |
| `root` | `[time, x, y, z, yaw_degrees]` keys from 0 to at least the duration, linear between keys. The root is the model's origin in the world; yaw turns about +Y, right-handed. The runtime never moves it. A character stands on its rest pose's lowest vertex, so `y` is the ground height minus that vertex's model height. |
| `events` | In time order: `set_parameter {name, value}`, `trigger {name}`, `play {name}`, `set_aim {name, point}`, `set_limb_target {name, point, weight}`, `set_ground_enabled {enabled}`, `terrain_patch {rect: [[x0, z0], [x1, z1]], offset}`. A point is world `[x, y, z]` or `null`. A patch raises the grid points inside the rectangle from its time on. The ground is on at the start. |
| `expect` | Optional `{clip, offset_s}`: the scenario is steady playback of one clip, at clip time `(t + offset_s)` modulo its duration. A synced clip's offset is its `phase_offset` times its duration. It says when each leg should be planted (from the clip's stance windows), and with the ground off on flat ground, what the pose should be. |
| `criteria` | Exactly the limits the scenario's gates use (below). |

Frames start at 0 and step by `dt`. A step that would pass an event time or the end is cut short to land on it, so
every call happens on a frame. Calls at a frame's time apply before that frame.

## Traces

| Field | Meaning |
|---|---|
| `format`, `version` | `"mesh-jig game trace"`, `1`. |
| `runtime` | `{name, version}` of whatever made it. |
| `hashes` | `glb_sha256`, `runtime_sha256` (the package's two files) and `scenario_sha256`: SHA-256 of the scenario file's bytes, as read. The checker hashes the file it is given and refuses a trace whose hashes differ. |
| `joints` | Joint names, in the runtime file's `skeleton.joints` order. |
| `matrix_layout` | The words for the layout below. |
| `frames` | One per scenario frame: `{time, root, joints}`. `root` is the root's world matrix; `joints` holds one world matrix per joint. |

Every matrix is 4x4, written as 16 numbers in glTF's **column-major** order: entries 0 to 3 are the first column,
and 12, 13, 14 are the translation. World space, glTF axes, metres. Time is seconds from the scenario's start. A
joint's world matrix is the root's world matrix times its model-space matrix (the glTF node hierarchy). Mesh nodes
that are no joint follow their nearest joint ancestor.

## The reference runtime

`game_runtime.py` is plain numpy, written to be read and ported. It reads `character.glb` for the skeleton, skin and
clips, and `character.runtime.json` for everything else. Its interface is `set_parameter(name, value)`,
`trigger(name)`, `play(name)`, `set_aim(name, point or None)`, `set_limb_target(name, point or None, weight)`,
`set_ground_enabled(on)` and `step(dt, root_matrix, ground) -> joint world matrices`, where
`ground(x, z) -> (height, normal, walkable)`. Each frame:

1. **Clocks and graph.** Every active node's clock advances by `dt` (a node starts at 0 on its first frame). Layers
   apply in order; a layer replaces the joints in its mask (all joints without one) by its weight.
   - `clip`: the clip at its clock, wrapped if it loops, held at its end if not.
   - `blend1d`: the two points either side of the parameter, blended by where it lies between them. Synced:
     points on the cycle share one phase that advances by `dt` times the lerp of their frequencies, each at
     `(phase + phase_offset)` of its duration; a point with `cycle: false` runs on the blend's own clock.
   - `states`: a trigger (or a played-out clip, for `at_end`) starts a linear cross-fade over `fade_s`. A null
     state fades the layer in or out over the layers below.
   - `oneshots`: `play` starts (or restarts) a clip; its weight rises over `fade_in_s` and falls over the last
     `fade_out_s`.
   Rotations blend by shortest-arc nlerp; translations and scales lerp.
2. **Forward kinematics.** World = root matrix x model-space matrix.
3. **Ground adaptation**, when on and the package has legs. The weight of the whole adaptation moves toward on or
   off over `blend_s`.
   - A leg is planted when the clips that make its pose, weighted by their shares, say stance at their phase.
   - The lock: while planted, and after while any contact point is still on the animation's ground, a horizontal
     shift cancels the world drift of the contact points that stay on that ground from one frame to the next, and
     a turn about +Y cancels the root's change of yaw. The shift is never longer than `relock_m`: past it the
     foot slides, only as fast as the animation pulls it further. Out of reach, or once the foot is off the ground,
     both fade to zero with time constant `blend_s`.
   - The foot's height: every contact point and clearance point asks for the ground under it, and the foot goes as
     low as keeps every point at least the contact's animated clearance above what it asks for (never below the
     ground). While the leg is not planted, a point also looks `lookahead_s` ahead along the foot's horizontal
     velocity, read at 8 points: ground a share s of the way that stands above the ground continued from just
     behind the point counts for (1 - s) of that rise, so a riser lifts the foot gradually and a slope does not
     lift it. The look-ahead's weight rises and falls over `blend_s` around lift-off and touchdown.
   - The body joint moves up or down by the lowest planted target's difference from the animation (smoothed with
     `blend_s`), and at once by as much more as any planted leg needs to reach; within `drop_m_max` and
     `rise_m_max`.
   - With `tilt` and three or more planted legs, the body turns to the plane fitted through their targets, laid on
     by the adaptation's weight.
   - Each leg is solved, and its tip turned toward its animated orientation plus the lock's turn, by the
     adaptation's weight. Turning the ground off or on fades every part of the adaptation together.
4. **Aims.** Yaw and pitch are measured in the frame of the first aim joint's parent, from the axis at rest. The aim
   keeps an offset from the animation that moves at most `degrees_per_s` toward the target (clamped to the limits),
   or toward zero with no target. The turn is shared equally along the chain.
5. **Limb targets.** A limb with a target is solved toward the target, by the game's weight. The weight is applied
   as given: a game that wants a smooth move ramps it. Limbs come after aims because an aim's chain (a spine) may
   carry a limb (an arm).

Solvers: `two_bone` puts the knee in the plane of hip, target and pole and clamps the reach so the knee's inner
angle stays within `bend_degrees`. `chain` runs FABRIK (root pinned, 30 passes), then turns each joint, root first,
by the shortest rotation onto its solved segment, holding any bend past `bend_degrees_max`. Every smoothing step is
in seconds, so a different frame rate gives the same motion.

## Run-time gates

`game-check` reads only the package, the scenario and the trace. "Should be planted" comes from `expect` alone.
Contact gap, penetration and recovery leave out the `grace_s` after each terrain patch.

| Gate | Criterion | Applies when | Reads |
|---|---|---|---|
| `playback_mm` | `playback_mm_max` | `expect`, ground off throughout, flat terrain | Largest vertex distance from the GLB's own clip at the expected time, placed by the root. Proves import, axes and timing. |
| `contact_gap_mm` | `contact_gap_mm_max` | `expect` and legs | Each leg's lowest contact vertex above or below the ground while the clip says it is planted. |
| `slip_mps` | `slip_mps_max` | legs | Each leg's RMS horizontal speed of its contact vertices that are within the contact gap of the ground in two frames running; the worst leg. |
| `support_fraction` | `support_fraction_min` | legs | Share of time with at least as many legs touching as the clip's own minimum (with `expect`) or the locomotion clips' minimum. |
| `penetration_mm` | `penetration_mm_max` | always | Deepest vertex or triangle centre below the ground. |
| `stretch_mm` | `stretch_mm_max` | always | Every limb segment's length against rest. |
| `limits_deg` | `limit_violation_deg_max` | limbs with a solver, or aims | Two-bone inner angle outside `bend_degrees`, chain bend past `bend_degrees_max`, aim yaw or pitch outside its limits, and, while an aim has a target, its turn past `degrees_per_s`. |
| `joint_speed_mps` | `joint_speed_mps_max` | always | Fastest joint, in the root's frame (the game's own movement is not counted). Catches pops at fades and snaps. |
| `aim_error_deg` | `aim_error_deg_max`, `settle_s` | an aim target | After `settle_s`, the angle between the aim axis and a target that is within the limits. |
| `limb_target_error_mm` | `limb_target_error_mm_max`, `settle_s` | a limb target | After `settle_s`, tip distance from a weight-1 target within reach. |
| `recovery_mm` | `recovery_mm_max`, `grace_s` | a terrain patch | From `grace_s` after a patch until the next: the worst penetration and planted contact gap. |

Readings: each leg's slip, the legs the support count needs, each leg's touching share against the clip's, the
deepest transient penetration within grace, and samples outside the terrain.

## Run-time limits

- A two-bone leg's knee is solved in a plane; feet keep their animated pitch and roll. On a slope the uphill edge
  of a sole touches and the downhill edge floats. Sole pitch and roll to the ground are later.
- FABRIK knows nothing of `bend_degrees_max`; the bend is held after it, so a bent target can be missed. The
  checker counts any target within the chain's length as reachable, so such a miss shows as tracking error.
- A limb target's weight is applied as given; nothing smooths a new target.
- A contact keeps the animation's clearance, so at a stance window's edge it is up to the build's contact gap above
  the ground. The scenario's contact gap limit has to allow that.
- The checker trusts the scenario's `expect` for when legs should be planted. Without it, contact gap is not read.
- A foot is kept out of the ground at its contact and clearance points only. An edge that passes between those
  points (a sole straddling a step edge) can still go a few millimetres in.
- A foot that lands across a riser rests on whatever part of it is over the higher ground; its contact then hangs
  above the lower ground and contact gap reads it. Moving a footstep off an edge is not done.
- Only feet are kept out of risers. Shins, knees and bodies are not, and no gate tells a foot from a shin: read
  `deepest_penetration` for which node went in.

 in-test with known answers: a biped with walk and run at known
speed, duty and phase; a four-legged body; a body with one five-joint chain and no legs; a turret with no limbs.
Each defect fails its own gate or reading: a skating walk, a walk that travels, a pace blended with a walk, a
missing leg, a snap fade, a stance that dips below the rest ground, a mid-swing graze, a slow end of stance, and the
contract refusals.

`tests/test_game_runtime.py` plays the biped over a slope, a step and a mid-run raise (each passing with the ground
on and failing contact gap and penetration with it off), flat playback, variable dt, four legs with tilt, a chain
limb's targets and an aim's limits and turn rate. `tests/test_game_check.py` breaks the reference runtime on purpose:
no stance lock (slip), bones that stretch (stretch), terrain ignored (contact gap and penetration), a snapped fade
(joint speed), a wrong clip speed and a flipped axis (playback), aim limits ignored (limits), a ground that never
updates after a patch (recovery). Each fails its own gate on a scenario the reference holds.

## Godot

Godot 4.7. `game-build` writes `package/godot/` with every package: it is small, has no options, and a package without
it would have to be rebuilt before it could go into a Godot project.

| File | What it is |
|---|---|
| `godot/character.tscn` | `character.glb` instanced under a `MeshJigCharacter` node, with the runtime and clips files set on it. All three by paths relative to the scene, so the package folder can sit anywhere in a project. The model is raised by its rest ground, so the node's origin is where the character stands. |
| `godot/character.clips.json` | Every joint track of the GLB's clips, key for key. The addon plays these, not Godot's imported animations (below). |
| `godot/addons/mesh_jig_character/` | The addon: the same files for every character. |

### Install and use

1. Copy the package folder into the project, say to `res://characters/armour/`.
2. Move its `godot/addons/mesh_jig_character/` to `res://addons/` (once per project; a newer package's copy replaces it).
3. Instance `characters/armour/godot/character.tscn` under the node the game moves (a `CharacterBody3D`), at the
   bottom of its collision shape.
4. Drive it from the game's script:

```gdscript
func _physics_process(delta: float) -> void:
	move_and_slide()                                 # the game's own movement
	$Character.set_parameter("speed", Vector2(velocity.x, velocity.z).length())
```

A character that is already moving on its first frame needs the same call in `_ready` too: the skeleton's first
update comes before the first physics frame.

The other calls have the same names as everywhere: `trigger(name)`, `play(name)`, `set_aim(name, point or null)`,
`set_limb_target(name, point or null, weight)`, `set_ground_enabled(enabled)`. Points are world positions.
`socket(name)` returns a node that follows a socket of the runtime file: a `BoneAttachment3D` on its joint with a
child at its offset. Parent a weapon to it. It follows the runtime's pose exactly (0.000 mm over 3000 frames on the
armour's `grip.R`). `debug_state()` returns what the last frame did: each layer's weight and clips at their times,
states, planted legs, body offset, aim offsets, and `step_usec`, the frame's cost. Print it or put it on a label.

Several characters can share a project and a level: each package brings its own scene and clips, and the addon is one
copy in `res://addons/`. The armour and the six-limbed horror walking side by side on a slope both held every gate.

### The ground

The runtime asks for the ground under every contact and clearance point of each foot, and while a foot swings,
along its look-ahead too: a Callable `(x, z) -> {height, normal, walkable}`.

- By default a child `Ground` node casts a physics ray straight down at each point, from `ground_ray_up` above the
  character's origin, `ground_ray_length` long, against `ground_collision_mask` (exported on the character). It
  ignores every collision body above the character in the tree, so the game's own capsule is no ground. A miss
  reports the character's rest ground, so the leg plays as animated there.
- `set_ground_query(callable)` replaces it. `ground_heightfield.gd` is the scenario heightfield (what the harness
  uses); a game can pass anything with that signature. The character keeps a reference to the Callable's object: a
  Callable to a method holds none, so a ground object nothing else held used to be freed.

What it costs, measured headless over 3000 frames after load (Ryzen 7700X, Godot 4.7.2, the addon's own work: the
runtime step and the bone writes):

| Character | Ground queries a frame | Mean | Median | 99th percentile |
|---|---|---|---|---|
| Armour, walking (18 joints, 2 legs) | 318 | 1.05 ms | 0.97 ms | 2.2 ms |
| Six-limbed horror, sprinting (29 joints, 6 legs) | 1529 | 3.4 ms | 3.0 ms | 6.9 ms |

The heightfield query costs the same as the raycast, so the time is the GDScript around the queries, about 2 µs
each. The count is the reference's: every contact point and up to 32 clearance points a foot, ten readings each while
the foot swings. Fewer clearance points or a shorter look-ahead in the runtime file, or a port of the foot-height step
to C# or C++, are the levers; none is built.

### Headless

```powershell
$env:MESH_JIG_GODOT = 'C:\path\to\Godot_v4.7.2-stable_win64_console.exe'
mesh-jig game-run --runtime godot --package package --scenario walk-slope.json --out trace.json
mesh-jig game-check --package package --scenario walk-slope.json --trace trace.json
```

`game-run --runtime godot` finds Godot in `MESH_JIG_GODOT`, then `godot` or `godot4` on the PATH. It runs
`engines/godot/harness/play_scenario.gd` with `--headless --xr-mode off` and no project: the GLB is loaded with
`GLTFDocument` at run time, the addon by path. The harness places the character at the scenario's root, makes the
calls at their times, gives it the heightfield as its ground and drives the clock itself (the skeleton in manual mode,
advanced by each scenario step). It reads each frame's joints from the skeleton after the modifier ran. The trace
names its runtime `godot-mesh-jig-character`. `tests/test_game_godot.py` plays every scenario the reference is tested
on and holds each Godot trace to the gates its reference trace gets. It is skipped when Godot is not found.

### How it works

- **The graph is evaluated in script** (`mesh_jig_graph.gd`), as the reference does it. It is not an `AnimationTree`.
  A synced `blend1d` shares one phase whose frequency lerps between the two points. Godot's `BlendSpace1D` with `sync`
  plays each point on its own clock. Measured on the in-test biped after 0.3 s at blend 0.5: its pose is 0.04 degrees
  from an own-clock blend and 0.90 degrees from the shared-phase one. Ground adaptation also needs each clip's time and
  share (the stance vote), which an `AnimationTree` does not report.
- **One `SkeletonModifier3D`** (`mesh_jig_modifier.gd`) runs the runtime each skeleton update, after any animation,
  and writes every joint's pose. The runtime (`mesh_jig_runtime.gd`) is the reference's steps 2 to 5, function for
  function with the same names: `lock`, `reach`, `foot_height`, `adapt`, `two_bone`, `fabrik`, `chain`, `aim`. A
  change to the reference can be ported by diff. It was last ported from the reference at commit 871e2ba4 (foot
  clearance with look-ahead, the lock held while a contact touches, aims before limb targets). It is a `RefCounted` because `Node3D` already has `rotate` and
  `translate`. The game's `AnimationPlayer` is stopped; it still holds the clips for previewing in the editor.
- **Clips come from `character.clips.json`**, sampled linearly with slerp, as the reference samples the GLB. Godot's
  scene import resamples every track at its bake rate (30 Hz by default). On the in-test biped, keyed at 60 Hz, that
  moved the feet 2.3 mm. On the armour it dropped keys at every rate tried, with its optimizer off: the walk's swing
  extreme went, a 2.8 degree error at 30 Hz and 1.1 degrees at 120 Hz.
- **Dot and cross products are taken in double precision.** `Vector3` is single precision. Two unit vectors less than
  3.4e-4 rad apart have a dot of exactly 1, so a small turn vanished: 0.15 mm at the end of the biped's shin. With
  the fix every joint stays within 0.001 mm of the reference.
- **Scenario numbers reach the harness as their exact bits.** Godot's JSON parser reads `0.016666666666666666` (1/60
  as Python writes it) one unit in the last place low. The clock then drifts by such units, and a stance decision on
  a window's edge flips. `game-run` hands the harness the scenario's numbers in a sidecar as IEEE-754 bits. The runtime
  file's numbers (six decimals) parse exactly.
- Unknown keys in the runtime file are reported with `push_warning` and ignored. `foot_height` is one function, so a
  change to the foot-height rule ports alone.

### What the addon does not do

- Move the character, root motion, physics, navigation: the game's job.
- Show the runtime in the editor: it runs in the game, not as a `@tool`. The editor shows the rest pose.
- Combine with an `AnimationTree` or `AnimationPlayer` the game drives: the modifier writes every joint each frame.
- Match the reference joint for joint in a game: a game's frame clock differs from the scenario's in the last bits,
  so a phase landing exactly on a stance window's edge can plant a leg one frame later. On the armour's walk that is a
  1.7 mm difference for one frame and no change to any gate (before the lock held while a contact touches, it moved
  slip from 0.055 to 0.078 m/s; [KNOWN_GAPS.md](KNOWN_GAPS.md)).
