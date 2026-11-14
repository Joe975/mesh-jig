---
name: mesh-jig-video-reference
description: Turn a video of a real object (a miniature, a toy, a part turned by hand or on a turntable) into a six-view photo reference sheet, then into a clean mesh-jig modeling reference with an image generator. Use when the user supplies a video or photos of a physical object to model, rather than asking for a sheet from a text description.
---

# From a video to a reference sheet

A short video of a real object turning holds every view mesh-jig needs. This skill picks the frames, cuts the
object out, lays the views on a sheet and hands that sheet to an image generator to remove what a photo can't
give: perspective, a blurry top, parts lost against the background. The photo sheet is the design master, so
the result describes the real object instead of a generated guess at it.

Two stages, two skills:

1. **Photo sheet** (this skill): frames, cutouts, sheet. Deterministic apart from the segmenter.
2. **Tidy** ([mesh-jig-reference-sheet](../mesh-jig-reference-sheet/SKILL.md)): the image generator repairs the
   photo sheet into a modeling reference, reviewed panel by panel, with the dimension audit.

Do not launch builders. That is [mesh-jig](../mesh-jig/SKILL.md), once the sheet is accepted.

## What makes a good clip

If you can ask for a new clip, ask for this. Each point fixes a defect the scripts can't.

- One slow full turn, about 20 s, on a turntable or by hand. A steady speed makes the quarter turns easy to find.
- Camera level with the middle of the object, as far back as the frame allows, then zoomed in. Close and above
  gives perspective and a downward tilt, and every view inherits them.
- A plain background that differs from the object in both colour and brightness. Dark parts against a dark
  background are what the segmenter loses.
- Hands out of the frame, or at least not behind the object.
- Then a sharp shot from straight above that fills the frame. A distant, oblique overhead shot gives an unusable
  top view.

The base the object stands on is fine: the segmenter usually leaves it out, and `clean` drops what is left.

## Setup

- `ffmpeg` on PATH.
- The segmenter in its own virtual environment, kept away from mesh-jig's pinned numpy and Pillow:
  `python -m venv <dir>` then `<dir>/Scripts/python -m pip install "rembg[cpu]"`. The first `cut` downloads
  the selected model. `isnet-general-use` is the default; check available memory before choosing a larger model.
- Everything else runs with mesh-jig's own Python. Work under `local-work/<subject>-<date>/`.

`scripts/video_sheet.py --help` lists the subcommands. In the commands below, `$V` is that script.

**Starting from photos:** skip frame extraction and the numbered contact sheet. Supply one photo per view
to `cut` in step 3 using the same `view=image:crop` syntax. Include a separate overhead photo. Keep the camera
distance consistent for the standing views; the sheet preserves their pixel scale rather than calibrating it.
The filenames and crop boxes below are illustrative. No source photos or videos are bundled with this skill;
the automated tests draw their own shapes and masks.

## 1. Frames

```
python $V frames clip.mp4 local-work/reference/frames --fps 4
python $V contact local-work/reference/frames local-work/reference/contact.png --box 450,0,1100,1080
```

`frames` writes `f_NNN.jpg`; frame N is at about (N-1)/fps seconds. `contact` tiles numbered thumbnails of
one region of every frame, with `--first`/`--last` to zoom in on a stretch. Look at the whole clip first,
choose a `--box` that holds the object in every frame, then look closer.

## 2. Pick the views

Find one full turn: the frame where the front faces the camera, and the frame where it does again. At 4 fps a
20 s turn is about 80 frames. Then:

| View | Pick the frame where | In the picture |
|---|---|---|
| `front` | the face is centred and paired parts look symmetric about it | subject's left on page right |
| `rear` | the back's centre line is centred, about half a turn from front | subject's left on page left |
| `side` | the front points page **left** (the subject's left side faces the camera) | front faces left |
| `right` | the front points page **right** | front faces right |
| `hero` | front-left three-quarter, between `front` and `side` | not measured |
| `top` | from the overhead shot | front points down the page |

Judge each pick by the object, not by the frame count: a hand-turned object doesn't rotate evenly. Where
neighbouring frames are equally good, take the sharper one.

**Check handedness before cutting.** Pick one asymmetric feature (a weapon, a patch, a raised arm) and follow it
through the four standing views. It must sit on the same anatomical side in every view. Front and rear are
mirror images of each other in the picture, so it swaps page side between them. Write the feature down: the
tidy prompt needs it.

## 3. Cut out

```
<rembg venv python> $V cut local-work/reference/cut front=local-work/reference/frames/f_073.jpg:480,150,1260,1000 side=local-work/reference/frames/f_055.jpg:480,150,1260,1000 rear=... right=... hero=... top=local-work/reference/frames/f_103.jpg:650,350,1300,850:-90
```

Each spec is `view=frame:x0,y0,x1,y1[:rot]`. Use **one crop box for every turntable view**: one camera at one
distance gives one scale, which the sheet keeps. Leave room around the object. `rot` turns the top view so the
front points down, in degrees anticlockwise: a top shot with the front pointing page-right needs `-90`.
`cut` writes each crop as `<view>_src.png` and the segmenter's cutout as `<view>.png`.

## 4. Clean and check

```
python $V clean local-work/reference/cut local-work/reference/clean front side rear right hero top --accent g
python $V overlay local-work/reference/cut local-work/reference/clean local-work/reference/overlay.png front side rear right hero top
```

`clean` makes one mask per view. The segmenter's sure pixels (alpha >= 110) are the core. Saturated pixels of
the `--accent` channel that connect to a band round the core are added back: segmenters leave glowing or
saturated parts (a lit blade, a lens) half transparent. Use `--accent r`, `g` or `b` for the object's saturated
colour, or `none`. Pieces under 2% of the largest are dropped, mesh-jig's own threshold for a separate piece.
Small holes are filled. Each view goes onto pure white as `<view>.png`, and its mask as `<view>_mask.png`.
A view whose subject touches its crop edge is flagged: widen that crop and cut again.

**Look at the overlay.** It draws each mask's outline over the source crop. Things to look for:

- a dark part lost against a dark background, or a thin part (an antenna, a blade tip) cut short
- a scrap of base, hand or shadow kept beside a foot
- a part hidden in one frame that the neighbouring frame shows: try that frame instead

Fix by changing the frame or the crop, or by changing `--accent`. Don't paint masks by hand to make a view
agree with the others: that is the tidy's job, and it records what it changed.

## 5. The photo sheet

```
python $V sheet local-work/reference/clean local-work/reference/photo-sheet.png --top-scale 1.9
```

A 3x2 sheet: FRONT, LEFT SIDE, REAR / RIGHT SIDE, HERO, TOP. The turntable views keep their own pixel scale and
stand on one ground line a row. The top view comes from another distance: `--top-scale` makes its width about
the front's.

What the photo sheet is not: orthographic, or calibrated. Standing heights differ by several percent with pose,
tilt and small changes in distance. Don't use it for `criteria.outline`, and don't cut it straight into `refs/`
for scoring unless the person accepts that.

## 6. Tidy with the image generator

Hand the photo sheet to an image generator as the **edit target and design master**, with the source frames as
detail references. Follow [mesh-jig-reference-sheet](../mesh-jig-reference-sheet/SKILL.md) for everything
from here: inspect every panel after every edit, one localized repair at a time, the dimension audit, a
`REVIEW.md`. Write the task down in a file beside the photo sheet so the run can be repeated. It should say:

- Read `skill/mesh-jig-reference-sheet/SKILL.md` and use the built-in image generation tool.
- The inputs and their roles: the photo sheet is the master, the frames are for detail only and include a
  hand, a base and a background that don't belong.
- Keep the design and the photographed pose. Do not redesign, do not make the pose symmetric.
- The defects to fix: perspective (front, sides, rear and top must read as orthographic at one scale), the top
  view if it came from a poor overhead shot, any part the cutout lost (name it and the view), base colour on the
  feet, cutout fringes.
- The conventions from the table in step 2, and the handedness you wrote down.
- Where to write: numbered candidates with their exact prompts, the audit, crops into `refs/`, and the review.
- A cap on generations, and to keep the best candidate and name what is unresolved rather than claim acceptance.

With Codex as the generator (its built-in `image_gen` tool, no API key needed), from the checkout:

```
codex exec -m <model> -s workspace-write -o <dir>/last-message.md -i <dir>/photo-sheet.png -i <frame> ... - < <dir>/task.md
```

Pass `-m` with a model the account offers. A model named in the Codex config can be one the CLI's login can't
use, and the run then fails at the first request.

**Review the result.** Cleanup and a plausible overhead repair do not establish orthographic views or a
common physical scale. A generated repair can also change a pose, reshape parts or repaint a panel.

- Ask for cleanup and the top view first, one edit each. Ask for perspective repair last, and only if the pose
  survives it.
- Compare each panel of each candidate with its source frame, not only with the previous candidate. Prefer a
  candidate that keeps the photographed pose over one that measures better but changes the object.
- The real fix for perspective is the clip: shoot from farther back, level with the object.
- The audit's default `chroma` mode can miss a dark, metallic subject. Run it
  with `"min_chroma": 0, "max_min_channel": 210`, and check its boxes against a plain darkness mask before
  trusting them.

## Hand off

The accepted sheet is a visually coherent raster reference of the photographed object, not exact geometry.
Cut it into a project's `refs/` (`mesh-jig sheet`, or the crops the tidy wrote), run `mesh-jig refs <project>`
and read what it flags before any build.
