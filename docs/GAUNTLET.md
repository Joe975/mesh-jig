# Visual gauntlet

## Default workflow

`mesh-jig agent <project>` uses visual review by default, with five iterations. The legacy numeric loop is
`--workflow numeric`; hybrid diagnostic inputs are opt-in `--checks builder` or `--checks both`.
See [the guide](GUIDE.md) for provider selection, limits and accounting.

Visual review is a provisional workflow choice, not a validated visual ranking algorithm. Deterministic evaluation
and hard contract validation remain mandatory. Highest numeric score, latest candidate and a human's visual
selection are distinct. Every candidate is retained; reviews and budgets are checkpointed.

## The engine

The gauntlet uses two independent API roles: an implementor writes a complete Blender model, then an adversarial
reviewer compares all its views and marked close-ups with the reference. The reviewer gives up to five prioritized
fixes and names features to preserve. Its critique and the previous script are passed to the implementor for the
next iteration. The reviewer is fresh each cycle and sees no score, script, reasoning or review history.

Gauntlet agents receive the design brief, build rules, cameras, palette and images. The implementor also receives
the fixed regions/parts/visibility/clearance contract, including exact mesh-node identities. They cannot read
evaluation or ledger files. Deterministic mesh-jig measurements still run on every submission for later comparison,
but agents see only build failures and the reviewer's visual critique. Failed builds are repairable within the
iteration's turn ceiling and do not count as iterations.

A note left at `<project>/steer.md` reaches the builder before its next turn, once, and never the reviewer.

A round that cannot build is not lost at once:

- With three turns of a conversation left and no build that passed, the last tool result of each turn says how many
  are left and to submit.
- When the last turn ends and the script on disk was never submitted, or was changed after its last submission, the
  loop submits it and takes the build if it passes.
- A round that still has no passing build starts once more as a new conversation: the brief, the references, the
  previous round's review when there is one, the script on disk and the last build error. The run ends when that
  fails too. Both conversations' turns count against `--max-turns`.

`report.json` lists each of these in `recoveries` (`loop-submitted` or `restarted`, the round, whether a build
passed).

## Compare with the numeric loop

Run from the checkout using its virtual environment (use `.venv/bin/python` on Unix), with absolute paths and a
fresh output directory:

```powershell
& "$PWD/.venv/Scripts/python.exe" -m mesh_jig.gauntlet compare `
  "$PWD/local-work/my-seed" `
  --out "$PWD/local-work/my-comparison" --iterations 10
```

The source must have its `refs/` images prepared already (`mesh-jig sheet` if needed). The command makes two fresh
projects, `gauntlet/` and `mesh-jig/`, and runs them concurrently. It copies only the frozen contract, brief,
references and original sheet, never previous attempts. `comparison.json` records identical SHA-256 seed hashes,
model, endpoint, budget and both outcomes. Existing output directories are refused.

The control uses `mesh_jig.agent.run` with the same number of evaluated attempts, normal metric feedback, previews
and no paid pairwise judge. Both arms have the same model, output token cap and implementor turn ceiling (24 per
iteration by default). The gauntlet makes extra reviewer calls, so this compares workflows at equal revision counts,
not equal API cost or tokens. There is no preview loop in the gauntlet arm, so a difference cannot be attributed
solely to adding a reviewer.

For a gauntlet alone, use `run <fresh-project>` without `--out`. Both modes accept `--iterations`, `--max-turns`,
`--max-tokens` and `--model`. Both roles default to `claude-haiku-5-5` at Anthropic, using `ANTHROPIC_API_KEY`;
no other key or model is substituted.

## Inspect the evidence

Each gauntlet attempt contains the usual GLB, renders, comparison images and `eval.json`, plus `review.md`.
`gauntlet/report.json` checkpoints every API call and completed cycle, recording usage by role. Transcripts omit
image bytes. A failure, or a round with no passing build in both of its conversations, produces an incomplete result
and nonzero exit status. Runs are not resumable; use a fresh output directory to retry. API usage without provider
cost fields is not a dollar-cost estimate.

The viewer accepts both project directories together:

```powershell
& "$PWD/.venv/Scripts/python.exe" -m mesh_jig view `
  "$PWD/local-work/my-comparison/gauntlet" `
  "$PWD/local-work/my-comparison/mesh-jig" --open
```

Compare final and best-scoring attempts, their failed gates and their visual quality. Reviewer text guides
construction; it is not an independent verdict between workflows. One pair is a pilot, not evidence of a reliable
gain. Keep raw runs under ignored `local-work/`; publish selected results with the publish-results skill.

## Four-arm sweep

`python -m mesh_jig.gauntlet_sweep <source> --out <fresh-private-output>` runs plain gauntlet, unchanged mesh-jig,
gauntlet with previous checks supplied to the builder, and gauntlet with checks supplied to both roles. `--iterations`
accepts 1–10 (default 10) and `--workers` 1–4 (default 4). Missing reference crops are generated in a private seed
copy. Existing outputs are refused.

For Sonnet 5.5 at high effort, add `--model claude-sonnet-5-5 --effort high`. High is Sonnet's documented API
default; the compatibility endpoint ignores `reasoning_effort`, so the runner uses the default and records that.
Other explicit effort settings are refused. Long reasoning can leave the endpoint silent: use
`--idle-seconds 900 --call-timeout-seconds 1800` to keep the 180-second watchdog from cancelling productive calls.

For Opus 5.5 high use `--model claude-opus-5-5 --effort high --budget-usd <limit>`. This path uses the native
Messages API with explicit high effort and preserves signed thinking on tool continuations. Its budget guard
reserves a full worst-case request before dispatch, settles to usage, and keeps the reservation after an ambiguous
failure. There are no automatic retries. The guard can stop a run early; an incomplete result is retained.

The sweep checkpoints each call's token usage, duration and estimated list-price cost in `accounting.json`, and
aggregates outcomes in `sweep.json`. Estimates use the selected model's rates, label unpriced calls, and are not
provider billing. Source and skill hashes identify the working tree, including uncommitted code.
