# Using mesh-jig

The [README](../README.md) covers installation and route selection. Run the [synthetic smoke test](QUICKSTART.md)
first. [JIG.md](JIG.md) documents every project field. [Client setup](CLIENTS.md) covers Codex/Claude Code;
[API setup](API.md) covers the independent hosted/local runner and judge.

Run relative paths from the checkout with its environment. In PowerShell, replace `python` with
`.\.venv\Scripts\python.exe` and `mesh-jig` with `.\.venv\Scripts\python.exe -m mesh_jig`.
In POSIX shells, activate with `. .venv/bin/activate` first.

## Configure your own reference

Create a project under the ignored `local-work/` directory. Supply a sheet with consistent views:

```
mesh-jig doctor
mesh-jig init local-work/my-model --sheet concept.png
```

`concept.png` is your own image relative to the launch directory; it is not supplied by the repository.
`init` copies it into the project. It creates placeholder full-sheet crops and a craft contract with
top/side/rear scored views. Do not measure those unchanged placeholders as independent reference views.

Before cutting references, edit `local-work/my-model/jig.json` and its generated `brief.md`:

1. Set `sheet.crops`: each view has a pixel box `[x0,y0,x1,y1]`. Keep labels and floor shadows outside
   the subject; leave space around the entire silhouette. [Reference views](JIG.md#reference-views)
   explains projection, backgrounds and masks. Inspect the actual image, not just box coordinates.
2. Set the physical dimensions, triangle/color requirements and cameras for your subject. For an upright
   character remove the craft's `length_m` and `forward`, supply `height_m` and explicit orthographic
   `model.shots`, then set `criteria.views` to those reference names. Follow [the organic workflow](ORGANIC.md)
   and replace every illustrative calibration value in its template with real reference measurements.
3. State scored views, weights and gates; describe each visible part and color in `brief.md`.

Then cut the configured crops and inspect what the mask actually sees:

```
mesh-jig sheet local-work/my-model
mesh-jig refs local-work/my-model --out local-work/reference-check
mesh-jig palette local-work/my-model --colors 5 --write
mesh-jig criteria local-work/my-model
mesh-jig criteria local-work/my-model --write
mesh-jig doctor local-work/my-model
mesh-jig brief local-work/my-model
```

Inspect every `<view>_seen.png` in `local-work/reference-check/`; fix warnings, labels, clipping or
shadows before proceeding. Name the extracted palette colors and confirm the brief/criteria output.
Freeze cameras, dimensions, crops, palette, scored views, limits and brief before comparing attempts.
Reference artwork can have inconsistent projections; inspect it before optimizing against it.

## Iterate

Follow [client setup](CLIENTS.md) or explicitly ask your coding agent to read
[mesh-jig](../skill/mesh-jig/SKILL.md). It writes a Blender script with a
`build()` function inside each candidate's directory. The build runner owns export and rendering:

```
mesh-jig build local-work/my-model --script local-work/my-model/attempts/a00/model.py
mesh-jig preview local-work/my-model local-work/my-model/attempts/a00
mesh-jig check local-work/my-model local-work/my-model/attempts/a00
mesh-jig eval local-work/my-model --script local-work/my-model/attempts/a00/model.py --note "initial volumes"
mesh-jig show local-work/my-model a00
mesh-jig streak local-work/my-model
```

Inspect reference/build comparisons, silhouette overlays and regional closeups. Fix the largest visual
error, copy the best script into a new attempt and repeat. Preview before evaluating. Failed builds do
not count as successfully evaluated attempts. Keep old attempts intact so their records remain meaningful.

The GLB is under `attempts/<name>/out/`. Exports use Y up and front along -Z. Follow the script rules
printed by `brief`; the checker refuses forbidden file access but is not a security sandbox.

## API agent and judge

Read [API setup](API.md) before these commands: they deliberately make model calls. Configure a
compatible vision/tool model and provider credentials, or an existing localhost model server.
The runner does not reuse a coding-client subscription. The judge has separate backend settings.

```
mesh-jig agent local-work/my-model --attempts 5 --max-turns 40 --max-wall 600
mesh-jig agent local-work/my-model --attempts 30 --patience 3 --max-turns 120 --max-wall 1800
mesh-jig judge local-work/my-model local-work/my-model/attempts/a00 --self-check --samples 8
```

API calls can cost money. Set explicit turn, attempt and cost limits appropriate to the provider.
The evaluator does not call a model unless a backend is requested. Keep raw transcripts private.
The API guide explains why reported-cost caps can overshoot and cannot work without provider cost data.

## Versions and troubleshooting

Python must be 3.12+; pinned dependencies are in [pyproject.toml](../pyproject.toml). Blender runs as a
separate background process; its bundled Python is separate from the checkout's environment. No hosted
model or GPU is required for deterministic CPU rendering. The software searches Windows, macOS and Linux
installation locations, but that is not a certification of every platform/Blender release.

This documentation update checks the synthetic example locally on Windows with Python 3.13.3 and Blender
5.2.1 LTS. The available environment has NumPy 2.4.0/Pillow 12.1.1, which differ from the project's pinned
2.5.0/12.2.0; no dependency reinstall was performed, so fresh pinned installation remains unverified here.
POSIX commands are shell-syntax checked when a shell is available; macOS/Linux builds and authenticated
Codex/Claude Code/API sessions are not exercised by this update. The [script rules](JIG.md#the-script)
describe optional Blender `openvdb` requirements; avoid assuming it exists in every distribution.

| Symptom | Check |
|---|---|
| `mesh-jig` not found | Activate the environment in this terminal, or use its Python with `-m mesh_jig`; install from the checkout, not PyPI. |
| `No module named mesh_jig` | Use the environment where `pip install -e ".[mcp,dev]"` succeeded; keep the source checkout in place. |
| Blender missing | Put `blender` on PATH or set `MESH_JIG_BLENDER` to an executable in the process environment, then run `doctor`. |
| PowerShell activation blocked | Use the README's explicit `.venv` Python commands; no execution-policy change is needed. |
| No views / bad masks / all views identical | Verify `concept.png`, crop boxes and view names; rerun `sheet`, inspect `refs --out`, and correct shadows/labels/backgrounds. |
| Character has wrong dimensions/cameras | Replace the craft defaults using the organic workflow before attempting a build. |
| Script refused or contract failed | Read the full checker/gate output and `brief`; repair the current unevaluated candidate, preserve evaluated attempts. The denylist is not a sandbox. |
| API model/key/cost/context error | Use [API troubleshooting](API.md#judge-outputs-and-recovery); ordinary local builds need no model key. |
| MCP server missing or disconnected | Use [client troubleshooting](CLIENTS.md#connection-troubleshooting), absolute server/project paths and the `[mcp]` extra. |
| Dashboard never returns / port busy | It is a foreground server. Ctrl+C stops it; use `--port 8797` if needed. Start it only when you intend to browse. |

PowerShell Blender override (replace with your installed executable):

```powershell
$env:MESH_JIG_BLENDER = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
.\.venv\Scripts\python.exe -m mesh_jig doctor
```

POSIX equivalent: `export MESH_JIG_BLENDER=/absolute/path/to/blender`. The API `.env` loader is not used
by every local build/MCP entry point; set Blender in the shell or server environment. For command-specific
options, use `mesh-jig --help`, `mesh-jig agent --help` or `mesh-jig view --help`.

## The words

Overlap is silhouette intersection over union. Resemblance compares colour zones and edge structure.
Colour compares local colours in Oklab. Regions measure named crops more closely. The numeric score is
the mix stated by `criteria.score`, or the legacy equal mix of overlap and resemblance. Asset contract
checks cover structural requirements; acceptance gates cover explicit quality limits. An unthresholded
region is an observation, not a passed gate.

Scores are comparable only under the same criteria and measures. A higher score can accompany a worse
looking model. Review pictures and gates alongside the numbers. [Known gaps](KNOWN_GAPS.md) documents
limitations; [publication policy](PUBLICATION.md) explains how to keep research and personal data private.

To share a selected result, use [mesh-jig-publish-results](../skill/mesh-jig-publish-results/SKILL.md).
It prepares a scoped metrics summary and chosen images/models, leaving the complete run private.
