# A game-ready character stage (plan)

Status: 13 November 2026. Slices 1 to 4 are built: the package (1), the reference runtime and checker (2), the
Godot addon (3, ported from the reference at 871e2ba4; every in-test and real-model scenario gives the reference's
gate values) and the porting page (4). Slice 5 is part built: the rig stage's ground readings and brief; its
locomotion-set brief section and the choice of rig gates remain. See [GAME_READY.md](GAME_READY.md) for what exists
and [GAME_READY_ENGINES.md](GAME_READY_ENGINES.md) for porting it.
The trials on real models are in [KNOWN_GAPS.md](KNOWN_GAPS.md). This page keeps the decisions and the slices still
to do; built detail lives in those pages, not here.

Goal: a stage that takes a rigged, animated GLB and writes a package a game can use with little work: the model, its
clips, a blend graph and IK that adapts to the ground at run time.

## Decided

1. **Godot is the built target.** Other engines get the neutral sidecar, the reference runtime, the trace checker
   and the porting page. Their users have an AI write the engine side and prove it with `game-check`.
2. **Any model, not only bipeds.** The contract is optional blocks (blend graph, limbs, ground adaptation, aims,
   sockets). Writing `character.json` for an unusual body is itself a benchmark of the model that writes it.
3. **Blending and IK only.** The game owns movement, input, physics and navigation. The runtime never moves the root.
4. **Frozen input, contract, build, independent check, preview.** No combined score; separate gates; exit codes 0
   held, 1 measured failure, 2 invalid input. Every limit a gate uses is a criteria key; unknown keys are refused.
5. **glTF carries what it can; the sidecar carries the rest.** Mesh, skin and clips go through the engine's glTF
   import. The graph, IK, contacts and clip meanings go in `character.runtime.json`, in glTF axes. No authoring
   tool transfers IK or animation trees to an engine; every engine rebuilds them, so they are described once,
   neutrally, and tested by trace.
6. **The reference runtime is the specification.** Runtimes are not compared pose for pose (engines blend
   differently); each must hold the gates on its own trace. Only single-clip playback compares poses.

Decisions made while building, kept here because later slices depend on them:

- The ground a clip is measured against is the rest pose's lowest vertex, as `rig-check` uses. A stance needs at
  least `stance_s_min`; shorter touches are grazes. Travel speed is the median of planted samples; skate is the
  RMS difference from it.
- In a synced blend, phases align at each clip's measured touchdown, cycle frequencies lerp, and a clip in which no
  leg touches down (an idle) runs on its own clock.
- The planted-foot lock is a drift correction, not a pinned point: it cancels the world drift of the contact points
  still on the animation's ground, holds while any contact point touches, is capped at `relock_m`, and fades after
  lift-off.
- A foot keeps its contact's animated clearance above the real ground, and its clearance points (the rest of the
  foot) out of it; a swinging foot looks ahead `lookahead_s`, counting rise above the continued slope.
- A scenario's hash is the SHA-256 of its file's bytes, so any runtime can reproduce it.

## Remaining slices

5. **Clips worth packaging, the rest.** The rig stage now reads ground contact as a game would and its brief says
   what makes a clip set packageable. Still to do: a locomotion-set brief section with worked keys (idle, in-place
   gaits, jump) and, once readings have been watched on a few rigs, a decision on which of them become rig gates.
6. **Transfer and the benchmark.** The bounded [game agent](GAME_AGENT.md) is implemented for two-bone ground
   characters, with immutable owner criteria and independent scenarios. A four-legged horse trial found frozen
   run-slip and minimum-reach defects; it correctly withheld game-ready delivery. Broader positive transfer to
   other bodies and solver capabilities remains to be measured.
7. **Later.** Sole pitch and roll to the ground normal, footstep choice off edges, keeping shins and bodies out of
   risers, additive layers, 2D blends, turn in place, face merged into the body skeleton, a merged-skin copy, a web
   runtime capabilities beyond the current clip/state/blend1d and two-bone ground profile. The reference site's
   shared browser runtime now reads sealed per-character sidecars; it does not run an agent for visitors.

Each slice ends with something that runs and tests that fail without it.

## Not planned

Physics (balance, ragdoll, pushes; contacts stay geometric), root motion, Humanoid retargeting, mocap, navigation,
a player controller.
