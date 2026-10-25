# Build from your references

[Install](../README.md#install) and run the [smoke test](QUICKSTART.md) first. Stay in the checkout;
use the README's shell conventions. Choose [Codex/Claude Code](CLIENTS.md) or the [API runner](API.md).

## Configure your own reference

Supply `concept.png`:

```
mesh-jig doctor
mesh-jig init local-work/my-model --sheet concept.png
```

`init` copies the image and creates full-sheet placeholder crops with a craft contract.
Before cutting references, edit `jig.json` and `brief.md`:

1. Set `sheet.crops` boxes `[x0,y0,x1,y1]`. Leave room around each silhouette; exclude labels and
   shadows. See [reference views and masks](JIG.md#reference-views).
2. Set dimensions, triangle/colour requirements and cameras. Characters need `height_m` and explicit
   orthographic `model.shots` instead of craft `length_m`/`forward`. Follow [the organic workflow](ORGANIC.md)
   and replace template calibration values with measured reference values.
3. Set scored views, weights and gates. Describe the parts and colours in `brief.md`.

Cut and inspect the configured views:

```
mesh-jig sheet local-work/my-model
mesh-jig refs local-work/my-model --out local-work/reference-check
mesh-jig palette local-work/my-model --colors 5 --write
mesh-jig criteria local-work/my-model
mesh-jig criteria local-work/my-model --write
mesh-jig doctor local-work/my-model
mesh-jig brief local-work/my-model
```

Inspect every `<view>_seen.png`, fix clipping/labels/shadows and name the palette colours.
Freeze the reference, cameras, dimensions, palette, criteria and brief before comparing attempts.

## Iterate

Your agent follows the [modelling skill](../skill/mesh-jig/SKILL.md) and writes a `build()` script:

```
mesh-jig build local-work/my-model --script local-work/my-model/attempts/a00/model.py
mesh-jig preview local-work/my-model local-work/my-model/attempts/a00
mesh-jig check local-work/my-model local-work/my-model/attempts/a00
mesh-jig eval local-work/my-model --script local-work/my-model/attempts/a00/model.py --note "initial volumes"
mesh-jig show local-work/my-model a00
mesh-jig streak local-work/my-model
```

Inspect comparisons, silhouette overlays, regional closeups and gates. Copy the best script to a new
attempt and fix the largest error. Keep evaluated attempts intact; failed builds aren't successful
evaluations. GLBs are in `attempts/<name>/out/`, Y up, front along -Z.

## API agent and judge

Configure [API credentials, compatibility and budgets](API.md) before running these model calls:

```
mesh-jig agent local-work/my-model --attempts 5 --max-turns 40 --max-wall 600
mesh-jig agent local-work/my-model --attempts 30 --patience 3 --max-turns 120 --max-wall 1800
mesh-jig judge local-work/my-model local-work/my-model/attempts/a00 --self-check --samples 8
```

## Versions and troubleshooting

Requires Python 3.12+ and the [pinned dependencies](../pyproject.toml). Blender runs separately with
its own Python; mesh-jig renders on the CPU. Optional `openvdb` availability varies by Blender build;
see [script rules](JIG.md#the-script).

Validation snapshot, 2026-10-25: the smoke test passed on Windows, Python 3.13.3 and Blender 5.2.1 LTS.
The available NumPy 2.4.0/Pillow 12.1.1 differ from the pinned 2.5.0/12.2.0. Fresh pinned installation,
macOS/Linux execution and authenticated client/provider sessions remain unverified. POSIX blocks passed
shell syntax checks; client setup matched installed help and official docs. Check current provider
model availability and protocol support yourself.

| Problem | Fix |
|---|---|
| Command/module missing | Use the checkout's installed environment or its Python with `-m mesh_jig`. |
| PowerShell activation blocked | Use explicit `.venv` Python commands from the README. |
| Blender missing | Put it on PATH or set `MESH_JIG_BLENDER` in the shell/server environment. |
| Bad/identical reference masks | Fix crops, labels and shadows; rerun `sheet` and inspect `refs --out`. |
| Wrong character dimensions | Replace craft defaults using the organic workflow. |
| Script/gate failure | Read `brief` and the full error; repair an unevaluated candidate. The checker isn't a sandbox. |
| API or MCP failure | Check [API controls](API.md#judge-outputs-and-recovery) or [MCP setup](CLIENTS.md#connection-troubleshooting). |
| Dashboard blocks / port busy | Ctrl+C stops the server; use `--port 8797`. |

Blender override in PowerShell (use your installed path):

```powershell
$env:MESH_JIG_BLENDER = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
.\.venv\Scripts\python.exe -m mesh_jig doctor
```

POSIX: `export MESH_JIG_BLENDER=/absolute/path/to/blender`. The API `.env` file isn't loaded by every
local/MCP entry point. For options, run `mesh-jig --help`, `mesh-jig agent --help` or `mesh-jig view --help`.

## Read the scores

Overlap measures silhouette IoU; resemblance measures colour zones and edges. Colour compares Oklab
values; regions measure named crops. `criteria.score` defines the mix, defaulting to equal overlap and
resemblance. Gates are separate: an unthresholded region is an observation, not acceptance.

Compare scores only under the same criteria and measures, then inspect the images.
See [known gaps](KNOWN_GAPS.md). Keep raw runs private and share selected outputs through the
[publication workflow](../skill/mesh-jig-publish-results/SKILL.md).
