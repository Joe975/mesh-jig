# Test a local build

[Install](../README.md#install), then run these commands from the checkout. The supplied box generates
its own reference renders: a pipeline self-check with no agent, key or external artwork.

## Run it

PowerShell:

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

Preparation refuses existing destinations, including partial projects. To repeat, choose a new directory
and substitute it throughout. Preparation builds the reference fixture; it doesn't record an evaluation.

## What success looks like

Expect a passing contract and score near 1. Unconfigured backend warnings are fine without keys.
This self-match checks the pipeline, not AI modelling quality. Outputs stay in `local-work/smoke/`:

| Path | Contents |
|---|---|
| `jig.json`, `brief.md`, `refs/`, `reference-build/` | Contract, description and reference fixture |
| `attempts/a00/model.py`, `attempts/a00/out/box.glb` | Supplied script and exported GLB |
| `attempts/a00/eval.json`, `attempts/a00/numeric.json`, `attempts/a00/compare/` | Results and comparison images |
| `attempts/ledger.md` | One evaluation row |

Inspect the GLB and comparisons. `preview` adds no ledger row; `show` reads results without another
evaluation. To browse the dashboard (Ctrl+C to stop):

```sh
mesh-jig view local-work/smoke --open
```

Next, [prepare your own reference](GUIDE.md#configure-your-own-reference) and choose
[a coding client](CLIENTS.md) or [API builder](API.md). The smoke example's answer is already supplied.
