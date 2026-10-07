# Using mesh-jig

The [README](../README.md) introduces the tool. [JIG.md](JIG.md) documents every project field.

## Start a project

Create a project under the ignored `local-work/` directory. Supply a sheet with consistent views:

```
mesh-jig doctor
mesh-jig init local-work/my-model --sheet concept.png
```

Edit `sheet.crops` in `jig.json`: each view has a pixel box `[x0,y0,x1,y1]`. Keep labels and floor
shadows outside the crop. Leave space around the entire silhouette. Then inspect the crops:

```
mesh-jig sheet local-work/my-model
mesh-jig refs local-work/my-model --out local-work/reference-check
mesh-jig palette local-work/my-model --colors 5 --write
mesh-jig criteria local-work/my-model
mesh-jig criteria local-work/my-model --write
mesh-jig doctor local-work/my-model
mesh-jig brief local-work/my-model
```

Write a part-by-part `brief.md`. Freeze the cameras, dimensions, palette, scored views and acceptance
limits before comparing attempts. Upright subjects need their own cameras and height contract; use
[the organic workflow](ORGANIC.md). Reference artwork may contain projection inconsistencies: inspect
it before trying to optimize a model against it.

## Iterate

Ask your coding agent to use [mesh-jig](../skill/mesh-jig/SKILL.md). It writes a Blender script with a
`build()` function inside each candidate's directory. The build runner owns export and rendering:

```
mesh-jig build local-work/my-model --script local-work/my-model/attempts/a00/model.py
mesh-jig preview local-work/my-model local-work/my-model/attempts/a00
mesh-jig check local-work/my-model local-work/my-model/attempts/a00
mesh-jig eval local-work/my-model --script local-work/my-model/attempts/a00/model.py --note "initial volumes"
mesh-jig show local-work/my-model a00
mesh-jig streak local-work/my-model
mesh-jig view local-work/my-model --open
```

Inspect reference/build comparisons, silhouette overlays and regional closeups. Fix the largest visual
error, copy the best script into a new attempt and repeat. Preview before evaluating. Failed builds do
not count as successfully evaluated attempts. Keep old attempts intact so their records remain meaningful.

The GLB is under `attempts/<name>/out/`. Exports use Y up and front along -Z. Follow the script rules
printed by `brief`; the checker refuses forbidden file access but is not a security sandbox.

## API agent and judge

The optional API loop uses an OpenAI-compatible endpoint configured with `MESH_JIG_AGENT_URL`,
`MESH_JIG_AGENT_MODEL` and `MESH_JIG_AGENT_KEY`. Copy `.env.example` to ignored `.env` and fill it
locally. Environment variables take precedence. The judge has separate `MESH_JIG_JUDGE_*` settings.

```
mesh-jig agent local-work/my-model --attempts 5
mesh-jig agent local-work/my-model --attempts 30 --patience 3
mesh-jig judge local-work/my-model local-work/my-model/attempts/a00 --self-check --samples 8
```

API calls can cost money. Set explicit turn, attempt and cost limits appropriate to the provider.
The evaluator does not call a model unless a backend is requested. Keep raw transcripts private.

## The words

Overlap is silhouette intersection over union. Resemblance compares colour zones and edge structure.
Colour compares local colours in Oklab. Regions measure named crops more closely. The numeric score is
the mix stated by `criteria.score`, or the legacy equal mix of overlap and resemblance. Asset contract
checks cover structural requirements; acceptance gates cover explicit quality limits. An unthresholded
region is an observation, not a passed gate.

Scores are comparable only under the same criteria and measures. A higher score can accompany a worse
looking model. Review pictures and gates alongside the numbers. [Known gaps](KNOWN_GAPS.md) documents
limitations; [publication policy](PUBLICATION.md) explains how to keep research and personal data private.
