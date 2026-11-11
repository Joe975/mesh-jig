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


## Repeated history checks

Every audit still visits all commits reachable from every requested ref, checks each commit's metadata
and each path against the current inventory, and reads working files afresh. Within one invocation,
identical historical showcase snapshots are validated once. A snapshot key includes its full package
path and every filename, mode and Git blob ID, so an added file, changed byte or mode cannot hide behind
an unchanged release manifest. Invalid snapshots keep a failure for every affected commit.

A single binary `git cat-file --batch` reader serves object requests. Its invocation-local blob cache
is limited to 32 MiB; larger objects are read without being retained. Missing objects, malformed responses
and reader failures block publication. There is no disk cache and no trust in a previous successful audit.
`--no-cache` and `--no-batch` disable these performance features for comparison, without disabling checks.

## Install several selected packages

Prepare, visually review and seal each package separately. Then, from the virtual environment, use one
explicit batch instead of repeating single-package installation:

```text
python skill/mesh-jig-publish-results/scripts/curate.py install-many --repo <absolute-public-checkout> --package first-result <absolute-sealed-package-one> --package second-result <absolute-sealed-package-two>
```

Every `--package` takes a slug followed by an absolute package directory. The existing `install` command
remains supported. The batch validates all inputs before writing, audits the clean public baseline once,
rechecks the checkout, installs only the selected paths and replaces the inventory once. It never commits
or pushes. Review and stage the explicit package paths and inventory, commit them, then run the complete
committed-tip guard and the normal pre-push hook. Batching does not authorize publishing additional data.

The installer rejects existing/duplicate slugs and links or reparse points. A checkout-local exclusive
lock coordinates these installers. HEAD, index, tracked file bytes, inventory and destinations are checked
again around the writes. On an ordinary pre-commit failure, cleanup removes only unchanged files this
invocation created. Once inventory replacement succeeds, the installation is committed to the filesystem;
a later cleanup error leaves the complete packages and inventory intact and reports an error.

This is not a multi-path filesystem transaction or a lock on unrelated editors. Do not edit the publication
checkout concurrently. A process crash or external writer can leave a partial installation or stale lock.
Inspect the reported state before retrying; never remove a lock belonging to a running installer or bypass
the guard. Raw source runs are never changed, copied wholesale, committed or published by installation.


## Fixed-history performance check

Measured on Windows x64 with Python 3.13.3 and Pillow 12.2.0, using public commit
`3497c4dc2f67ab2e02f2f0397534d8de0fe7245f`: 123 reachable commits and 38 packages in
the checked-out tip (332 showcase files, 184,736,123 bytes).

- Original guard: 649.24 seconds, 20,303 Git processes (20,055 object-reader processes),
  2,150 package validations.
- Optimized guard: 17.57 and 16.13 seconds, 249 Git processes (one object-reader process),
  99 package validations on each run.
- All three full-history audits returned the same empty failure set, including fresh working files.
  The observed improvement was approximately 37–40 times for this snapshot.

Each measurement used a fresh Python process and the same clean fixed-commit worktree, inventory,
validator bytes and installed dependencies. The two optimized runs bracketed the original run.
The original validator was used for both implementations so installer changes could not affect the
comparison. Counts include every Git child and every call to package validation within the audit;
setup/status probes are outside the timer. No publisher guard ran concurrently. Operating-system
caches were not flushed and unrelated desktop activity was not controlled, so these are local
measurements, not a cross-machine performance guarantee.

Regression tests compare exact failure sets against the original traversal with cache and batch modes
independently enabled/disabled. Cases include unsafe deleted ancestry, later sealing, changed or extra
package files, private metadata, divergent refs, forbidden names/modes, current inventory and working-file
changes, malformed or missing objects and reader cleanup failures. Installer tests cover all-input
preflight, duplicate/existing slugs, links and parent traversal, concurrent edits, partial writes, rollback
ownership, the inventory commit point, locks and clear recovery outcomes.
