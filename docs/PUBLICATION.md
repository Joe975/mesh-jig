# Publication policy

The public repository contains software, reusable instructions, configuration templates and synthetic tests.
Personal projects and experiment results belong in separate private storage or private version control.
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
Git clone; install it for each new checkout. Run the command in CI or before every manual push as well.
No automated scan proves that arbitrary prose or image content contains no personal data. The explicit
inventory and exclusion of research content reduce that risk.

Adding .gitignore rules or deleting a file does not remove earlier commits. A clean publication root avoids
bringing private ancestors along. Replacing an existing remote history requires a deliberate remote update.
Previously public commits may remain in clones, forks, caches or hosting-provider retention after replacement.
