# Sounds from code

A stage after rigging: a model writes a Python script that synthesises a character's sounds, and the ones that go
with an animation clip are made to fit it. Nobody listens during the loop. The author is given what the sounds are for,
the clip's own times, and after each build the numbers and spectrograms of what it made.

## A sound directory

| File | What |
| --- | --- |
| `sound.json` | the contract: which sounds, how long, which clip each goes with, which of its cues it must start a sound on |
| `brief.md` | what the sounds are for, in words |
| `animated.glb` | the rigged model the clip sounds go with (the rig stage's output); only needed when a sound names a `clip` |
| `attempts/<name>/sounds.py` | an author's script |

```json
{
  "animated": "animated.glb",
  "sounds": {
    "meow": {"seconds": [0.4, 0.9], "for": "a cat calling"},
    "run":  {"clip": "run", "on": "footfalls"},
    "jump": {"clip": "jump", "loop": false, "on": ["takeoff", "landing"]}
  }
}
```

A sound with `seconds` is played once and may be any length between the two. A sound with `clip` is exactly the clip's
length to the sample, loops when the clip does (`loop` overrides what the GLB says: a leap whose last pose is its first
still plays once), and must start a sound within 35 ms of each cue named in `on`. `required` defaults to true. Other
keys (`for`, `hear`) are carried for people and outside judges; the check does not read them.

`expect` says what a sound should measure as (pitch range and shape, voicing, periodicity, centroid, flatness, time to
half level, where it is loudest, flutter), and `between` the same for a stretch of a clip's sound between two cues
(`{"takeoff-landing": {...}}`; `start` and `end` are its ends). Each build reads every sound against them; they are
readings, not gates. A pulsed growl's pitch reads as its mouth's ringing (a heuristic of the tracker), so ask a growl
for its flutter, not its pitch.

## Cues

`mesh-jig sound-cues animated.glb` reads them from the posed vertices at 120 frames a second (`sound_cues`):

- `footfalls`: each side's lowest point (the model's left is X<0) coming down to within 5 mm of its lowest. A side
  down for less than 40% of its longest stance only scuffed the ground and is not counted, so a swing foot that dips
  near the ground as the other lands is not a footfall.
- `takeoff` and `landing`, in a clip whose lowest point rises 2% of the model's height clear of the ground: where the
  lowest point leaves the ground and meets it again (the same 5 mm).
- `crouch`: the top of the model lowest before the takeoff. `apex`: the lowest point highest.

## The script

```python
def sounds(cues):
    # cues["run"] == {"clip": "run", "seconds": 0.6, "loop": True, "footfalls": [[0.294, "R"], [0.594, "L"]]}
    # cues["meow"] == {"seconds": None, "shortest": 0.4, "longest": 0.9, "loop": False}
    return {"meow": samples, "run": samples, ...}       # 1-D float arrays, 44,100 Hz, -1..1
```

It imports numpy, math and random only (`script_check`'s denylist, a stricter import list), runs in a child process
(`sound_run`, `python -I -B`, 60 s) twice, and both runs must write the same bytes. The harness writes the WAVs.

`mesh-jig sound-info <dir>` prints what an author starts from: the brief, the contract, the cues as the script is
given them, what each sound should measure as, and the recipe (`sound_recipe`). `--write-template
<dir>/attempts/01/sounds.py` writes the recipe with a `sounds(cues)` that already holds the contract, to replace one
sound at a time.

## The check

`mesh-jig sound-check <dir> --script sounds.py [--out DIR]` (`sound.check`). A sound played once: length in bounds,
peak 0.50 to 0.98, RMS above -30 dBFS, DC within 0.01, under 0.02 s of silence before and 0.10 s after, first and last
samples within 0.02 of zero and the first and last millisecond under a quarter of the peak. A loop: the exact length,
RMS above -45 dBFS (footfalls are mostly gaps), its last sample running into its first as neighbours do (within 0.05,
or twice the largest step between neighbours in the 2 ms either side, since a footfall wrapping round is loud noise
across the seam); no fades. Every cue named in `on`: an onset at least a quarter as strong as the sound's strongest
within 35 ms. Onsets are spectral flux over 40 log bands, 512-sample windows every 128 samples (2.9 ms), read wrapped
round for a loop; a strong onset on no cue is recorded in report.json and not said to the author.

`--out` keeps the WAVs, prints what each measures as (`sound_measure`: pitch at five points, periodicity, spectral
centroid and flatness, the loudness shape by tenths, amplitude flutter) and draws a spectrogram sheet with the cues as
yellow lines.

## The agent

`mesh-jig agent <dir> --workflow sound --model M --url U [--attempts 6] [--effort low] [--max-tokens 20000]`
(`sound_agent`): file tools, `build` and `done`, no shell, as the rig workflow. The opening carries the brief, the
contract, the cues as the script gets them, what each sound should measure as (`expect`), a recipe (`sound_recipe`:
a voice through a moving mouth, shaped noise, a thud, a growl, a break-up, placing on a loop) and the recipe's example
call drawn as a spectrogram. A build answers with the verdict, each cue's offset, the measurements, each sound read
against `expect`, and the sheet as a PNG. Builds out of contract are not counted. From the third build in contract,
one with every sound as it should measure may be delivered before the builds are used. Whatever stops a run, its last
build in contract is delivered: the script and WAVs are copied to `<dir>/delivered/`.

## One script, many models

A script reads its cues at run time, so one script makes each model's sounds fitted to that model's own clips: copy
the sound directory, put another rig in as `animated.glb`, and `mesh-jig sound-check <dir> --script sounds.py
--out <dir>/wav`. A clip without a cue the contract names (a jump that never leaves the ground) is refused before the
script runs, with the reason.

## Not here

Whether a sound sounds like what it is for. The check is the contract and the measurer describes; neither hears.
