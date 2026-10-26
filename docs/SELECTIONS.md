# Save selection checkpoints

Freeze latest, numeric best and a visual choice separately. Each checkpoint copies the selected
scripts, evaluations, GLBs, saved renders and reference views. It writes `checkpoint.json`, a
comparison sheet and a hash seal under the project's `checkpoints/<name>/` directory.
It makes no build, render, evaluation or model call.

## Record a choice

After evaluating and inspecting your attempts:

```sh
mesh-jig checkpoint local-work/my-model round-05 --visual-attempt a04 --visual-source self-review --reason "Cleaner face and wing transitions."
```

Use your actual attempt name. Sources are `human`, `self-review`, `reviewer` and `pairwise`.
Use `self-review` for your own assessment; name another source only when that review happened.
`--review-file attempts/a04/review.md` also copies the evidence. Source labels record a claim,
not independent verification. The reason is required with a visual choice.

Omit the visual options to leave that choice unknown. Numeric best is never used as a substitute.
Existing evaluation, scoring and numerical-best behaviour stay unchanged.

## Compare checkpoints

```sh
mesh-jig checkpoint local-work/my-model round-10 --before round-05 --visual-attempt a08 --visual-source self-review --reason "Better expression despite the lower score."
mesh-jig checkpoint local-work/my-model round-10 --verify
```

The before/after sheet shows latest, numeric and visual selections beside the reference. It uses
only saved images, preserving their aspect ratios. Missing images get labelled placeholders and
warnings; missing scripts or GLBs and invalid images stop creation. `--views hero side` selects
existing views. No reference is required, but its panel is labelled unavailable when absent.

Checkpoints are write-once. Pick a new name to change a selection. `--verify` checks their copied
files, inventory and recorded hashes. Original attempts may later move without changing a snapshot.
The seal detects accidental changes; it is not a signature or OS write protection. Keep checkpoints
private: copied scripts, evaluations, references and review text can contain private information.
Use the [publication workflow](../skill/mesh-jig-publish-results/SKILL.md) for shared results.

## Read the format

`checkpoint.json` uses schema version 1:

- `scope`: successful measured attempts, scores, evaluation/script hashes and ordering keys.
- `selections`: attempt names for `latest`, `numeric_best` and `visual`; unknown choices are `null`.
- `numeric_best_ties`: all tied winners. The earliest in scope wins, matching existing numerical ranking.
- `visual_choice`: source, reason and optional hashed evidence.
- `attempts`: selected artifact paths, byte counts, SHA-256 hashes and recorded evaluation versions.
- `inputs`, `references`, `inputs_sha256`: copied task inputs and their identities.
- `before`, `views`, `comparison`, `warnings`: comparison provenance and unavailable-image notices.

Order uses the recorded evaluation start time, then directory name. Legacy records without a start
time use the evaluation file's modification time. `--through a05` limits scope to that successful
attempt and its predecessors. A failed or unfinished attempt cannot become a selection. An unscored
success can be latest or visual; numeric best is unknown when no candidate has a numeric score.
Before/after input hashes must match. These records preserve saved scores; they do not recalibrate
metrics, prove that historical artifacts were untouched, or turn preferences into quality ratings.
Missing or different recorded evaluation versions produce comparability warnings.

## Use CLI, MCP or Python

Codex and Claude Code can run the commands above. The MCP server and API agent expose `checkpoint`
with the same arguments, using `visual_attempt`, `visual_source`, `review_file` and `through`.
Checkpoint names and attempt names are directory names, not paths. The API agent confines evidence
files to its attempt directories. Existing runs create no checkpoints unless the tool is called.

```python
from mesh_jig import project, selection

record = selection.create(project.load("local-work/my-model"), "round-05",
                          visual_attempt="a04", visual_source="self-review",
                          reason="Cleaner face and wing transitions.")
selection.verify("local-work/my-model/checkpoints/round-05")
```

Research visual keepers can pass their explicitly reviewed incumbent and review evidence through
this function. Do not infer a visual choice from their numerical report fields.
