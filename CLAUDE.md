# Developing mesh-jig

Read README.md, docs/JIG.md and docs/KNOWN_GAPS.md. Production code lives under src/mesh_jig;
skills hold the model-building and harness workflows. Tests use synthetic inputs and temporary directories.
For modeling rather than development, use the included `.claude/skills/mesh-jig` entry or read
`skill/mesh-jig/SKILL.md`. See `docs/CLIENTS.md` for local client/MCP setup.

Use the repository virtual environment. Run `python -m pytest -q` after logic changes. Keep measures
deterministic. The Blender runner uses only Blender-shipped modules and the standard library. Preserve
the script denylist, model write confinement and legacy scoring behavior unless deliberately changing them
with regression coverage. No model is called by an evaluation without an explicit backend.

## Changing a measure

When changing scoring, camera projection or rendering, add a synthetic regression case that exposes
the changed behavior and run the relevant measure/render tests plus the full suite. Remeasure incumbent
and candidate models with the same saved configuration and implementation; do not compare old scores
directly with a new yardstick. The harness-loop skill describes separate projects for that remeasurement.

Keep research projects in ignored local-work/ or experiments/ and private version control. Do not commit
personal reference images, prompts, run records or raw logs. Publish chosen metrics/images/GLBs only with
skill/mesh-jig-publish-results/SKILL.md, as a reviewed sealed showcase. Before publishing run the
privacy gate described in docs/PUBLICATION.md. Never merge private research history into this public branch.
