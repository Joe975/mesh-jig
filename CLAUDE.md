# Developing mesh-jig

Read README.md, docs/JIG.md and docs/KNOWN_GAPS.md. Production code lives under src/mesh_jig;
skills hold the model-building and harness workflows. Tests use synthetic inputs and temporary directories.

Use the repository virtual environment. Run `python -m pytest -q` after logic changes. Keep measures
deterministic. The Blender runner uses only Blender-shipped modules and the standard library. Preserve
the script denylist, model write confinement and legacy scoring behavior unless deliberately changing them
with regression coverage. No model is called by an evaluation without an explicit backend.

Keep research projects in ignored local-work/ or experiments/ and private version control. Do not commit
personal reference images, prompts, run records, raw logs or generated models. Before publishing run the
privacy gate described in docs/PUBLICATION.md. Never merge private research history into this public branch.
