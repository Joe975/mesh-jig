---
name: mesh-jig-keep-going
description: Build a mesh-jig model with no fixed number of attempts, stopping when it stops improving - three evaluated attempts in a row that gain no more than a small margin on the numeric score. Use when asked to "keep going until it stops improving", "run until three attempts in a row show no improvement", "don't stop at five", or to run a builder with no attempt cap. It sets the length of the loop in the mesh-jig skill, which it needs.
---

# Keep going until it stops improving

The [mesh-jig skill](../mesh-jig/SKILL.md) is the loop: write an attempt, evaluate it, fix the largest error. A
task usually says how many attempts to make. This skill is for a task that does not: you keep making attempts and
`mesh-jig streak` says when to stop.

## The task, as a person gives it

```text
Use the mesh-jig and mesh-jig-keep-going skills. This directory is a mesh-jig project (jig.json, brief.md, refs/):
build the model it describes. Keep going until three attempts in a row show no improvement.

Stop early at: 30 attempts
```

Only the first paragraph is required. What it may also give, and the default when it does not:

| | default | the option |
|---|---|---|
| attempts in a row with no improvement that end the run | 3 | `--patience N` |
| how much an attempt has to beat the last improvement by to count as one | 0.002 | `--margin N` (`--margin 0`: any higher score counts) |
| a ceiling on evaluated attempts | none | `--max-attempts N` |

Use the same options on every call in a run. Do not change them part way: a run that moved its own stopping rule
cannot be read against another.

## The rule

Keep making evaluated attempts until three in a row have gained no more than 0.002 on the numeric score.

- Only a measured attempt counts. A failed build, or one the contract refuses, is not an attempt: fix it and
  evaluate again.
- An attempt is an improvement when it beats the last improvement by more than the margin. A smaller step is
  within what an unrelated edit moves the score by, and a tie is never one.
- The first attempt always counts as one.
- The best attempt is the highest score, whether or not it counted as an improvement.

Do not count it yourself. After every evaluation run:

```
mesh-jig streak <project>
```

It reads every attempt's `eval.json` in the order they were measured and prints what it counted. The last line
starts `KEEP GOING` or `STOP`. On `KEEP GOING`, make the next attempt. On `STOP`, make no more attempts and report.

The output names the best evaluated attempt, its gates, the current streak and the stop reason.

With a ceiling, or told to count any higher score, the call is the same one each time:

```
mesh-jig streak <project> --max-attempts 30
mesh-jig streak <project> --margin 0 --max-attempts 30
```

## While it says keep going

Everything in the mesh-jig skill holds: one idea an attempt, a new directory each time, the largest error first.
A long run adds three practical checks.

- **After an attempt that did not improve, start the next one from the best attempt, not from the one that
  missed.** `mesh-jig streak` names the best. Copy its `model.py`.
- **Read the `gates:` line every time.** The score does not include the gates. An attempt that
  scores higher and fails a gate the best one passed is a trade, not a gain: undo what broke the gate before going
  on.
- **Paint parts, not cells.** The colour measure rises when faces are coloured to follow the reference's zone map
  cell by cell, whatever part they belong to. Keep each part the palette colour the brief gives it. When the score went up and the
  comparison picture looks worse, the picture is right.

The margin prevents arbitrarily small score changes from indefinitely extending a run.

## When it says stop

Report:

- why it stopped (the streak, or the ceiling) and how many attempts were evaluated
- the best attempt and its numbers
- the highest attempt that passes every gate, when `mesh-jig streak` names one, and which gates the best fails
- the attempt you would hand over, when looking at the pictures it is not the best scoring one, and why
- what is still wrong

End the report as the mesh-jig skill ends any report. Ask the user whether they want to see the results in the
dashboard (`mesh-jig view <project> --open`). That command never returns, so start it in the background, and only
on a yes.

## For whoever starts the run

- **Give it a ceiling and a budget.** Improvement can continue indefinitely. Set explicit attempt and cost limits.
- **Inspect the final pictures.** A higher numeric score is not sufficient to accept a model.
- **`mesh-jig agent` has the same rule as `--patience N`.** `mesh-jig agent <project> --model <id> --patience 3
  --attempts 30` runs an API model until three evaluated attempts in a row show no improvement, with `--attempts`
  as the ceiling and `--margin` as above. Every evaluation's result ends with what `mesh-jig streak` prints, and
  an evaluation after `STOP` is refused. `--max-cost`, `--max-wall` and `--max-turns` still end a run early.
