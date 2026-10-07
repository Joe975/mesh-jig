# Blind visual review

You are scoring 3D models against the reference they were built from. Everything you need is in this directory.
Do not read anything outside it: the directory above holds the answer to which sheet came from where, and a score
given with that knowledge is worth nothing.

- `R01.jpg`, `R02.jpg`, ...: one model each. Top row: the reference views. Bottom row: the model's renders of the
  same views, in the same columns. Each picture is cropped to its subject and scaled to one height, so absolute
  size and position are not shown and are not scored.
- `brief-<name>.md`: what the subject is, part by part. A sheet's title names the brief it goes with.

The sheets are in a shuffled order. Some were made one way and some another. You are not told which, and the
order carries no information.

## Scoring

Open every sheet before scoring any, so you know the range. Then for each sheet give five marks from 0 to 10:

- `silhouette`: outline and proportions in each view against the reference above it
- `parts`: the brief's parts are there, and where the reference has them
- `colour`: the reference's colours on the right parts
- `shape`: each part is formed like the reference's, not a box standing in for it; nothing floats, intersects
  wrongly or flickers
- `likeness`: at a glance, the same object as the reference

and an overall `score` from 0 to 100, your own judgement of how close a translation of the reference the model is,
not a sum of the marks. As a guide: under 45 is a blockout (the big masses, few readable parts), 45 to 65 has the
detail but is off-model, above 65 is a close translation. Add a `note`: one sentence naming what is most right and
most wrong, in terms of parts.

Score what is visible. Do not reward detail that the reference does not have, or polish in the wrong place.

When all are scored, go back over the highest, the lowest and any within five points of each other, side by side,
and correct the order where it is wrong. Sheets you cannot tell apart get the same score.

## What to write

`scores.json` in this directory, every sheet, nothing else in the file:

```json
{
 "R01": {"silhouette": 6, "parts": 5, "colour": 6, "shape": 4, "likeness": 5, "score": 48,
         "note": "Banded stacks and a framed window, but the arms are plain boxes hanging straight down."}
}
```

Then `notes.md` beside it: what the better models get right that the rest do not, failures that recur across
sheets, and how small a difference in score you would defend.
