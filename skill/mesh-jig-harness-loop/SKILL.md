---
name: mesh-jig-harness-loop
description: Improve mesh-jig itself (what a builder is told and shown, the agent loop, the measures, the renderer) by generations, each one change tested with fresh builder runs and a blind visual review, until N generations in a row show no improvement. Use for goals like "iterate on the harness design to improve the created models until three generations in a row show no improvement", or when asked whether a change to mesh-jig makes the models better. Needs a checkout of the mesh-jig repo installed with `pip install -e .`, Blender, and a builder (an API model or a coding agent). Not for building one model: that is the mesh-jig skill.
---

# Improving the harness by generations

The [mesh-jig skill](../mesh-jig/SKILL.md) improves one model. This one improves what every model is built with.
You change the harness, fresh builders build with it, a reviewer who does not know which models are which scores
them beside the incumbent's, and the change stays only if the models got better. You stop when changes stop
helping.

## The goal, as a person gives it

```text
Use the mesh-jig-harness-loop skill. Iterate on the harness design to improve the models it produces, until three
generations in a row show no improvement.

Seed: local-work/my-seed
Builder: mesh-jig agent, anthropic/claude-sonnet-5.5, low effort, five attempts
Runs a side: 3
Stop early at: 12 generations or $40
```

Only the first sentence is required. Defaults for what it leaves out: three runs a side, stop after three, margins
as `generation.py` has them, at most ten generations. The seed, the builder and a money cap have no default: ask
for them once, with what a generation will cost (runs a side x the builder's cost per run, twice that for a
generation that is kept), before anything is spent.

## Words

- **Harness**: everything between the goal and a GLB that is not the builder or the task. `skill/mesh-jig/SKILL.md`,
  what `brief` prints, what `eval` says and draws after an attempt, the `agent` loop and its prompt, the cameras, the
  measures, the renderer.
- **Seed**: a project with no attempts (`jig.json`, `brief.md`, reference views or the sheet they are cut from).
  The task. More than one seed is better than one: a harness tuned on a mech is not thereby better at a cat.
- **Builder**: the model, effort, attempt count and prompt that build from a seed. Held fixed, so that the harness
  is the only thing that differs between the sides.
- **Run**: one builder on one fresh copy of a seed. What it delivers is its best measured attempt.
- **Generation**: one change to the harness, the runs built with it, and the verdict on them.
- **Incumbent**, **candidate**: the harness as last kept and the runs it made; the harness with this generation's
  change and its runs.

## What stays fixed for the whole loop

Write these into `GOAL.md` before the first run. Changing one starts a new loop, since nothing before it compares.

- The seeds' reference views and `brief.md`. `generation.py sheets` refuses sides whose reference views differ.
- The builder, and how it is launched. A run never sees another run, the reviews or this skill.
- Runs a side, the margins, the stop rule and the caps.
- `review.md` and `scripts/generation.py`: the yardstick. Improving those is a separate piece of work.

Anything else in the repo may change, the measures and the renderer included, with three exceptions that are
safety and not design: the script denylist (`script_check`), the confinement of what `agent` may write, and the
tests. A test changes with the behaviour it tests; it is not deleted to get a green run.

## Before the first generation

1. Read `CLAUDE.md`, `docs/KNOWN_GAPS.md`, `docs/LESSONS.md` and your private experiment records. Read `GENERATIONS.md` of any earlier loop: an idea rejected there is not new.
2. Work on a branch. The tests pass (`python -m pytest -q`) and `mesh-jig doctor` finds Blender.
3. Make the loop's directory, `experiments/<loop name>/`: `GOAL.md` (the goal as given and everything fixed),
   `GENERATIONS.md` (one row per generation: the change, its commit, the verdict line, kept or reverted, cost), and
   `.gitignore` copied from [loop.gitignore](loop.gitignore).
4. Build the baseline: the runs of the harness as found, in `g00/runs/<name>/`. These are the first incumbent.

## A run

A fresh directory per run, never reused (the ledger is shown to the builder). Copy the seed in, cut the reference
views if the seed holds a sheet, record what it ran on, start the builder, and wait for it in the foreground.

```text
mesh-jig sheet <run>
git rev-parse --short HEAD > <run>/harness-commit.txt
mesh-jig agent <run> --model <model> --attempts 5 --effort low --max-cost 1.50
```

`mesh-jig agent` reads the working tree, so the change only has to be saved. A coding agent as the builder gets the
committed `skill/mesh-jig/SKILL.md` as its only skill, in a directory outside the repo, with nothing of the
person's own setup. Save its exact prompt in the private experiment directory. Commit the change before those runs. Runs of one generation may go two to
four at a time. A run that died (no measured attempt) is run again under a new name, and the record says so; a run
that built badly is a result and stays.

## One generation

1. **Pick one change** (next section) and write its row in `GENERATIONS.md` first: what changes, what you expect to
   see differently on the sheets. A change you cannot say that about is not worth a generation.
2. **Make it**, with the test that fails without it. All tests pass. Commit: `g03: <the change>`.
3. **Build** the candidate's runs in `g03/runs/`.
4. **Make the blind sheets.** The incumbent's runs are the ones from the generation last kept (`g00` at first),
   reviewed again every time: a score is never compared with a score from another review (`docs/LESSONS.md` 1).

   ```text
   python skill/mesh-jig-harness-loop/scripts/generation.py sheets experiments/<loop>/g03/review --incumbent experiments/<loop>/g00/runs/* --candidate experiments/<loop>/g03/runs/*
   ```

5. **Have them reviewed** by someone who has seen neither the change nor the key: a subagent or a new chat with a
   fresh context, or a person. Give it `g03/review/sheets/` and tell it to follow `REVIEW.md` there, nothing more.
   Not you: you know what you changed and you want it to have worked. Do not open `key.json`.
6. **Read the verdict.**

   ```text
   python skill/mesh-jig-harness-loop/scripts/generation.py verdict experiments/<loop>/g03/review
   ```

7. **NO IMPROVEMENT**: `git revert` the change's commit, so the log shows what was tried and that it came out.
   The incumbent is unchanged.
8. **IMPROVED**: measure it again before believing it. Build the same number of fresh candidate runs in
   `g03/confirm/runs/`, make sheets in `g03/confirm` against the same incumbent runs, a fresh reviewer, `verdict`.
   Improved again: the change is kept and the confirming runs are the incumbent from here on. Not improved: revert,
   and the generation counts as no improvement. Replication is required before accepting a gain.
9. **Record and count.** Fill in the row, record the generation in private version control, then:

   ```text
   python skill/mesh-jig-harness-loop/scripts/generation.py status experiments/<loop>
   ```

   It prints CONTINUE or STOP. Go to 1 or to "When it stops".

A change that broke the build or the tests is not a generation: fix it or drop it before any run is paid for.

## What counts as improvement

`verdict` decides, not you. Both must hold:

- **The models look better.** The candidate's mean visual score (0 to 100) is above the incumbent's by at least
  the margin (5 points, about what one reviewer can tell apart on a sheet) and by at least the noise, the standard
  error of that difference read off the spread of each side's own runs.
- **The numbers did not fall.** The candidate's mean numeric score is not more than 0.02 below the incumbent's,
  which is what two runs of one setup differ by.

Visual review evaluates model quality; numeric measures guard reference matching. Neither alone establishes an improvement.

## When the change touches a measure, the renderer or the cameras

Then an unchanged GLB reads or looks different, and the incumbent's stored numbers and renders are from another
yardstick. Do what `CLAUDE.md` asks under "Changing a measure" first. Then give `sheets` the incumbent's delivered
attempts measured again by the candidate's code, one new project per run:

```text
python skill/mesh-jig-harness-loop/scripts/generation.py remeasure experiments/<loop>/g00/runs/<name> experiments/<loop>/g03/incumbent/<name>
```

(`--jig <the seed's jig.json>` when the project file changed too) and pass `--incumbent experiments/<loop>/g03/incumbent/*`.
Both rows of sheets are then drawn by one renderer and both sides scored by one set of measures. A measure cannot
vouch for its own change: here the numeric guard only says the new number does not rank the new models below the
old ones, and the reviewer carries the decision. Say so in the row.

## Picking the next change

From evidence, strongest first:

1. The reviewer's `notes.md`: a failure that recurs on both sides' sheets is the harness's, not a run's bad luck.
2. What the builders said they could not tell or could not do: their final reports, `report.json`, refused calls.
3. The ledgers: a number that did not move when the model did, attempts that got worse and were kept anyway.
4. Sheets where the numeric and the visual score disagree, after the verdict is in. They name what the measures do
   not see, or see wrongly.
5. The gaps in `docs/KNOWN_GAPS.md` and findings in `experiments/` that nobody has acted on.

Prefer the cheaper kind of change when two would address the same failure: what the builder is told, then what it
is shown after an attempt, then the loop it runs in, then a measure or the renderer. One idea a generation. Three
variations of a rejected idea are not three generations with no improvement: after a rejection, move to a different
failure or a different kind of change.

## When it stops

On STOP from `status`, or at a cap (say which).

1. If anything was kept, measure the whole gain once, fresh: new runs on the final harness in `final/runs/`, sheets
   in `final/review` against the `g00` runs (remeasured if any kept change touched a measure), a fresh reviewer,
   `verdict`. That line, not the sum of the generations' gains, is what the loop achieved.
2. Report: baseline against final; each kept change with its two gains; each rejected idea in a line; runs, cost and
   wall time; what you would try next and why you did not. The report goes at the top of `GENERATIONS.md`.
3. `docs/KNOWN_GAPS.md` gets an entry under "Tried, and what it showed" pointing at the loop. Leave the branch for the person to merge.

## What stays private

Keep goals, generation reports, prompts, scripts, evaluations, ledgers, model files, reviews and logs in private version control. The public software repository does not include experiment records.
