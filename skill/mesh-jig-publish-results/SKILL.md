---
name: mesh-jig-publish-results
description: Keep mesh-jig runs private by default and prepare selected public showcases with runtime, token usage, cost, chosen images and GLB models. Use when sharing results, publishing a benchmark or deciding which run artifacts may be committed.
---

# Share selected results

Keep the complete run private. Publish a small, newly generated package under `showcase/<slug>/`.
The helper copies only explicitly selected inputs; it never copies a run directory, transcript or raw report.
It does not commit or push.

## Keep ordinary runs out of Git

Before a new run, use `local-work/<run>/` or another ignored directory. Check the planned path with
`git check-ignore`; check `git ls-files` too, because an ignore rule does not protect already tracked files.
The public repository ignores `local-work/`, `experiments/`, `private/`, attempt and agent directories.
Never force-add raw runs. Keep full private records in a separate private repository or normal backup.

If private material is already committed, preserve it locally and inspect the history before publishing.
Deleting it in a later commit or adding .gitignore does not remove its ancestors. Do not merge private
research history into the public branch. A fresh clone of the public repository is a suitable publication
checkout; no history rewrite is needed for an ordinary new showcase.

On a fresh public clone, create/install its virtual environment and enable the local push guard once:

```text
<venv-python> <checkout>/skill/mesh-jig-publish-results/scripts/curate.py protect --repo <public-checkout>
```

Git does not clone hooks. `protect` uses the active virtual environment and preserves an existing custom
hook; integrate the guard into such a hook deliberately. Reinstall after moving the checkout or environment.

## Prepare the selected package

Ask only for missing choices that affect what is shared: run/scope, cost provenance and selected visual assets.
A request to create this skill is not permission to publish a real run. A request to publish selected results
authorizes preparing and reviewing that selection; resolve unknown sources or scope before making claims.

Use the checkout's virtual-environment Python and absolute paths. Replace the symbolic paths below with
absolute paths for that machine:

```text
<venv-python> <checkout>/skill/mesh-jig-publish-results/scripts/curate.py prepare --report <private-report.json> --out <private-draft-directory> --title "Haiku 5.5: first five attempts" --driver "mesh-jig API agent via OpenRouter" --scope "Five successful evaluations, one run" --cost-kind reported --effort low --evaluation <selected-eval.json> --image <selected-render.png> --glb <selected-model.glb>
```

`--image`, `--glb` and `--note` may repeat. Without visual arguments it produces a metrics-only package.
The output directory must be new. Output filenames are generic, not the source filenames.

The helper extracts a fixed set of model, usage, runtime, completion and evaluation fields. It excludes
project paths/names, endpoints, timestamps, prompts, freeform agent summaries/errors and per-call logs.
Missing metrics remain unknown. Supply public context with `--scope` and short `--note` arguments;
these are intentional public prose, so review them too.

- Label cost `reported`, `estimated`, or `unavailable`. A subscription/list-price estimate is not an invoice.
- Runtime is the selected report's wall time. Say whether the report covers a fresh run or a continuation.
  Do not pair continuation tokens/cost with total-run runtime or the wrong model's result.
- Input/cache counters are provider-defined. Completion tokens include reasoning where the report says so;
  reasoning is a subset, not extra output to add again. Do not infer model calls from driver turns.
- Choose the evaluation and model from the stated scope. The tool cannot prove that arbitrary selected files
  belong together. Verify model/driver/effort, successful attempt count and evaluation/model pairing locally.
- Preserve relevant limitations: one run is not a general ranking; different drivers/prompts are different
  workflows; a changing checkout is not a frozen benchmark. Numeric score is not finished visual quality.

Images are decoded to a fresh RGB/RGBA PNG, removing EXIF, text and other metadata. GLBs retain core glTF 2
geometry/materials/animation, remove names/extras/generator metadata, repack referenced buffer views and
normalize embedded images. External URIs, unknown fields and glTF extensions are refused. Do not bypass a
refusal: export a supported self-contained GLB or omit it. `.blend` and raw scripts/logs are never allowed.

## Review and seal

Read the generated README and summary, inspect every output image for visible personal data, and render
or view each sanitized model. Compare it with the selected original for geometry, materials and animation
as applicable. Metadata removal does not remove sensitive text baked into pixels, textures or geometry.
Keep references private unless the user explicitly selected them for sharing.

After that review:

```text
<venv-python> <checkout>/skill/mesh-jig-publish-results/scripts/curate.py seal --package <private-draft-directory>
```

This writes `release.json`, pinning the exact public bytes by SHA-256. It is a record of the reviewed bytes,
not proof that their visible content is non-sensitive. If anything changes, review and seal a new package.

## Install and publish

Install only that sealed package into a clean public checkout:

```text
<venv-python> <checkout>/skill/mesh-jig-publish-results/scripts/curate.py install --package <private-draft-directory> --repo <public-checkout> --slug haiku-55-first-five
<venv-python> <public-checkout>/tools/check_publication.py HEAD
```

`install` refuses a dirty public checkout or an existing showcase name, validates all files, copies the
package and adds only its exact paths to `PUBLIC_FILES.txt`. Review the diff, then stage the explicit
`showcase/<slug>/` path and inventory file. Commit with the public checkout's normal attribution settings,
rerun the guard against the new commit, and push only the intended public branch if publishing is authorized.
Do not use `git add .`, `--all`, `--mirror`, disable the guard or import private commits.

The guard checks each historical showcase against its own manifest. It permits selected PNG/GLB files only
inside a valid sealed showcase, rejects altered bytes and catches raw artifacts elsewhere. Existing software
files still use the public inventory. See [publication policy](../../docs/PUBLICATION.md).

When reporting, distinguish prepared/committed/pushed state, list what was included, and confirm the private
run stayed untouched. Give the public metrics with their scope and cost provenance. Link to the showcase
only after it exists; never expose the source run paths in public prose.
