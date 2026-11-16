# Prepare an animated character for a game

Run this after rigging and authoring clips. The game agent proposes a JSON contract and playback policy for a
frozen animated GLB. It cannot edit the mesh, skeleton, weights or clips. If those inputs fail, repair them in the
rigging stage and create a new game job. Visitors to the reference site never run an agent or call an API.

## Create and run a job

Use absolute paths for your checkout, animated GLB and new job directory. In PowerShell:

```powershell
$repo = 'C:/path/to/mesh-jig'
$rig = 'C:/path/to/animated.glb'
$job = 'C:/path/to/mesh-jig/local-work/my-game-job'
& "$repo/.venv/Scripts/python.exe" -m mesh_jig game-info --rig $rig
& "$repo/.venv/Scripts/python.exe" -m mesh_jig game-init --rig $rig --out $job --legs 4 --gaits walk run
& "$repo/.venv/Scripts/python.exe" -m mesh_jig agent $job --workflow game --model YOUR_CONFIGURED_MODEL --effort high --attempts 3 --max-turns 12 --max-cost 8
```

Use the actual leg count and gait names from `game-info`. The API model uses the existing
[provider configuration](API.md). A reserved cost cap requires the direct Anthropic transport; other transports
are refused when `--max-cost` is supplied. Capped game runs count the exact native input with the provider before
each generation request and reserve a 10% plus 1,024-token margin, cold-cache input cost and the full reply cap.
A failed count prevents generation. Calls, builds and reply tokens have separate limits. Invalid build
proposals consume attempts. This stage needs no Blender or running game engine.

`game-init` copies the GLB to `animated.glb` and writes `game.json` and `brief.md`. Review the required gaits and
acceptance limits before running. These three files are hashed and checked throughout the run. The model cannot
weaken the job's limits, omit required legs/gaits or introduce filesystem paths or executable scripts.

## Read the result

Each run is under `agent/<run>/`, with a private transcript, usage accounting and `report.json`. Each counted
build has a proposal, playback policy, package report and independent scenario checks. The scenario checker skins
the actual mesh from runtime joint traces; a valid contract alone is insufficient.

The API agent receives `game-info` measurements, structured build/check results and temporal scene renders.
Its tools are `build`, `review`, `done` and `report_failure`. After a build, pictures enter the next model turn.
The agent must call `review` with concrete clip/time/limb observations and a `needs_revision` decision before
another build or delivery. A build and its review cannot happen in the same turn. The configured backend must
accept images. Existing call, turn, token and cost caps still apply, including image input accounting.

Exit 0 means `report.json.final` names a numerically passing, reviewed attempt and its package. Exit 1 means there is no accepted
delivery. Exit 2 means invalid setup/configuration. A candidate package may exist when the package measurements
pass but the scenario checks fail. Read `final`; do not copy the latest candidate and call it game-ready.

## Review the solved motion

Each renderable build gets three sheets for its fastest required gait: an IK-off source control, a cross-slope
and a short obstacle scene. Each sheet has side and quarter views, up to seven frames, actual graph stance
labels and matching trace metadata. Context frames accompany the worst failed contact or motion check and
its immediate neighbours. The renderer draws the actual skinned mesh and the same triangles queried by the
reference runtime. No Blender, browser, GPU or separate reviewer API call is needed.

The obstacle scene has stairs, an uneven ridge and a low wall scaled to leg reach. Its route follows rig size
at the measured gait speed, with an eight-second cap and flat margins for approach and exit. It is diagnostic and does
not change the job's acceptance limits. It is not the reference site's full course. Review its saved scenario
and checks alongside the pictures; a finite strip cannot certify every footfall or every clip. Inspect the
final consuming browser/game for its materials, full route, action clips, landing and loop boundaries.

Look for planted-foot sliding or sudden jumps, sole penetration, implausible folds and abrupt body movement.
A lifted swing foot is expected. Compare the source control and adjacent frames before calling it a defect.
Record unresolved issues with `needs_revision: true`; that attempt cannot be delivered. Review feedback is
self-review by the builder, not an independent visual score or certificate.

Failed package measurements remain failed. When a diagnostic runtime can be derived, it lives only under
`attempts/<name>/package/diagnostic/`, marked `accepted: false`, so the agent can still inspect and revise it.
It never fills the normal delivery paths. Invalid contracts or missing measured travel may prevent rendering;
the build reports that failure. Picture hashes bind review to the exact images, and missing, changed or
unreviewed evidence prevents delivery. Scene images, traces, reviews and transcripts stay in the private job.

The harness checks each required gait with ground adaptation off, on flat ground and on a cross-slope, using
variable frame times. It also checks measured contacts, clip coverage, loop/blend measurements and playback
fidelity. These finite checks do not certify arbitrary obstacles or game physics. Add the consuming game's
scenarios and inspect its render before shipping. The existing [manual package workflow](GAME_READY.md) supports
more features than this deliberately limited agent/browser profile.

## Current agent and browser profile

One skeleton root, one body joint, 1–32 ground legs with three joints each, two-bone solvers, no aims, and
`clip`, `states` or `blend1d` graphs. The browser course expects glTF -Z forward and LINEAR or STEP TRS tracks.
Every clip appears exactly once in the default cycle. Playback declares measured gait travel, authored root
travel, stationary actions or in-place airborne travel referenced to a measured gait. Landing exits require a
measured flight and landing. Individually selected clips loop.

The reference viewer opens Run by default (Sprint when that is the running clip's name, otherwise the first
available clip). All animations remains an optional selection using the packaged cycle. Shared clip links win
over the default.

Ground IK places feet and adapts the body within reach limits. It does not plan footfalls, avoid shin/body
collisions, vault a wall or invent a stair animation. Tall steps and unequal leg segments can make targets
unreachable. The horse demo is an experimental transfer example: walking passed the agent checks, running
exceeded the slip limit, and tall stairs expose a leg-reach limit. It has no accepted game-ready delivery.

## Add a character to the reference viewer

The site uses a shared renderer/course plus a small per-character runtime sidecar. The sidecar carries anonymous
joint indices, measured contact probes, tracks, clip order, movement policy and camera framing. It is bound by
SHA-256 to the exact sanitized public GLB. It adds no new GLB download. Ordinary clip viewing still loads the
terrain modules only after terrain is requested; only the selected character's sidecar is fetched.

Site maintainers prepare a sidecar against an already curated public GLB, review its motion/contact results,
seal its exact bytes, register its model binding and enable that key in the site selection. Preserve the
[publication rules](PUBLICATION.md): raw jobs, contracts, source paths and agent transcripts stay private.
Any experimental result needs a model-specific visible limitation. A registry entry records reviewed playback,
not a certificate that every terrain check passed. Benchmark publication should retain renderable failed IK
candidates as diagnostic scenes; follow [Publish benchmark failures](../skill/mesh-jig-publish-results/SKILL.md#publish-benchmark-failures).
Before publication, check shared links, reduced motion,
comparison, GIF restoration, disposal and a phone viewport. Measure performance on actual mobile hardware;
desktop software rendering cannot establish a mobile frame budget.
