# Publication policy

The public repository contains software, reusable instructions, configuration templates, synthetic tests
and explicitly selected showcases. Full personal projects and raw experiment records stay private.
Keep them under ignored `local-work/` or `experiments/`, including all references, scripts, evaluations,
models, prompts, reviews, raw logs and screenshots.

`PUBLIC_FILES.txt` is the reviewed file inventory. `tools/check_publication.py` checks every file and commit
reachable from the ref being published against that inventory, and checks recognizable credential and
personal-path patterns. It also checks the working tree before publication. Unexpected files fail closed.
It prints paths and categories, never matching secret values. Review new inventory entries deliberately.

Run the checker with the repository virtual environment before publishing:

```
python tools/check_publication.py HEAD
```

The prepared checkout also installs this check as its local pre-push hook. Hooks are not installed by a
Git clone; use the results skill's `protect --repo <absolute-public-checkout>` command from the checkout's
virtual environment to install one. It preserves existing custom hooks. Run the checker in CI or before
manual publication as well.
No automated scan proves that arbitrary prose or image content contains no personal data. The explicit
inventory and exclusion of research content reduce that risk.

## Selected showcases

Use [mesh-jig-publish-results](../skill/mesh-jig-publish-results/SKILL.md) to extract selected runtime,
token, cost and evaluation fields into a fresh package, with only explicitly chosen images and GLBs.
Label reported versus estimated cost and state the run/continuation scope. Source reports, paths,
prompts and transcripts are excluded. Images lose metadata; supported GLBs lose identifying metadata,
external resources and unused buffer views. Inspect the sanitized visual outputs before sealing them.

`showcase/<slug>/release.json` pins every selected file by SHA-256. The package contains generated
README/summary files and optional normalized PNG/GLB files. Installing a sealed package adds its exact
paths to `PUBLIC_FILES.txt`. Raw directories remain forbidden, and arbitrary binary files are still
blocked. The guard validates each historical package with its own manifest, so a later deletion or
replacement cannot hide an invalid earlier publication. Sealing records reviewed bytes; it does not
establish that text visible in a picture or model is appropriate to share.

Adding .gitignore rules or deleting a file does not remove earlier commits. A clean publication root avoids
bringing private ancestors along. Replacing an existing remote history requires a deliberate remote update.
Previously public commits may remain in clones, forks, caches or hosting-provider retention after replacement.
