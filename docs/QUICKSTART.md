# A key-free first success

Finish [installation](../README.md#install) first and stay in the checkout. This supplied colored-box
script generates its own three reference renders with Blender, then you build and measure the same script.
It needs no agent, API key, external artwork or downloads. It is a deterministic installation/pipeline
self-check, not evidence that an AI has learned to model or that a real reference is correctly calibrated.

## Run it

PowerShell, with no environment activation needed:

```powershell
.\.venv\Scripts\python.exe examples/smoke/prepare.py local-work/smoke
.\.venv\Scripts\python.exe -m mesh_jig doctor local-work/smoke
.\.venv\Scripts\python.exe -m mesh_jig brief local-work/smoke
.\.venv\Scripts\python.exe -m mesh_jig build local-work/smoke --script local-work/smoke/attempts/a00/model.py
.\.venv\Scripts\python.exe -m mesh_jig preview local-work/smoke local-work/smoke/attempts/a00
.\.venv\Scripts\python.exe -m mesh_jig eval local-work/smoke --script local-work/smoke/attempts/a00/model.py
.\.venv\Scripts\python.exe -m mesh_jig show local-work/smoke a00
```

POSIX, after `. .venv/bin/activate`:

```sh
python examples/smoke/prepare.py local-work/smoke
mesh-jig doctor local-work/smoke
mesh-jig brief local-work/smoke
mesh-jig build local-work/smoke --script local-work/smoke/attempts/a00/model.py
mesh-jig preview local-work/smoke local-work/smoke/attempts/a00
mesh-jig eval local-work/smoke --script local-work/smoke/attempts/a00/model.py
mesh-jig show local-work/smoke a00
```

Preparation refuses an existing destination, including a partial project. Choose a new directory and
substitute it in every command when repeating the check. No previous project is deleted or overwritten.
It invokes local Blender to build a reference fixture, without adding an evaluation to the ledger.

## What success looks like

The supplied box has blue sides and a gold top, is 1 m tall, 0.6 m wide and 0.4 m deep, and has 12 triangles.
`doctor` should find Blender and no project problems. Optional backend warnings are expected without keys.
`eval` should report built, in contract, rendered and measured, with overlap/resemblance/numeric score
near 1. Generated outputs stay under ignored `local-work/smoke/`:

| Path | Contents |
|---|---|
| `jig.json`, `brief.md`, `refs/` | Fixed contract, description and three synthetic reference views |
| `reference-build/` | Local fixture used to produce those references |
| `attempts/a00/model.py` | Supplied candidate script |
| `attempts/a00/out/box.glb` | Exported Y-up GLB |
| `attempts/a00/eval.json`, `numeric.json` | Full evaluation and numeric results |
| `attempts/a00/compare/` | Reference/build comparisons and silhouette overlays |
| `attempts/ledger.md` | Exactly one evaluation row after the commands above |

`preview` measures without a ledger row. `show` reads the recorded evaluation without duplicating it.
Open the comparisons and inspect the GLB as well as its scores. For a dashboard, run this yourself;
it starts a local server and stays running until Ctrl+C:

```sh
mesh-jig view local-work/smoke --open
```

Move on to [your own references](GUIDE.md#configure-your-own-reference), [Codex/Claude Code](CLIENTS.md),
or a deliberately configured [API run](API.md). Do not treat the synthetic self-match as a useful new
modeling task for an API agent: its answer is already supplied.
