# mesh-jig

Tools a coding agent calls to build a Blender model script and measure it against reference views.
Every attempt is built, checked against its asset contract, rendered from fixed cameras and scored
deterministically. The result is a GLB for a game engine.

![The pirate cat reference over three runs: Opus 5.5 in Claude Code, GPT-6.1 Sol in Codex, and Qwen3.8-27B on a local card](showcase/pirate-cat-opus/image-01.png)

The top row is the reference sheet. Each row under it is one agent's best attempt at it.

## Install

Requires Git, Python 3.12+ and [Blender](https://www.blender.org/download/). Install from a checkout;
the package is not distributed on PyPI. Run these commands in a terminal, choosing your shell.

Windows PowerShell (uses the environment directly; no activation-policy change needed):

```powershell
git clone https://github.com/Joe975/mesh-jig.git
cd mesh-jig
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[mcp,dev]"
.\.venv\Scripts\python.exe -m mesh_jig doctor
```

Linux/macOS POSIX shell:

```sh
git clone https://github.com/Joe975/mesh-jig.git
cd mesh-jig
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[mcp,dev]"
mesh-jig doctor
```

Stay in the checkout for all relative paths below. On PowerShell, replace `python` with
`.\.venv\Scripts\python.exe` and `mesh-jig` with `.\.venv\Scripts\python.exe -m mesh_jig`.
The doctor may say the optional judge/agent is unconfigured; local tools still work.
See [versions and troubleshooting](docs/GUIDE.md#versions-and-troubleshooting) if Blender is missing.

## First successful local build

No API key, agent or artwork is needed for this synthetic pipeline check:

```sh
python examples/smoke/prepare.py local-work/smoke
mesh-jig doctor local-work/smoke
mesh-jig brief local-work/smoke
mesh-jig build local-work/smoke --script local-work/smoke/attempts/a00/model.py
mesh-jig preview local-work/smoke local-work/smoke/attempts/a00
mesh-jig eval local-work/smoke --script local-work/smoke/attempts/a00/model.py
mesh-jig show local-work/smoke a00
```

The supplied script should build, pass its contract and measure near 1 against its own reference renders.
This verifies the pipeline, not modeling ability. [The walkthrough](docs/QUICKSTART.md) explains outputs
and contains an exact PowerShell version. Use a fresh directory if rerunning preparation.

## Choose a builder

| Route | What writes the script | Credentials and billing |
|---|---|---|
| [Codex](docs/CLIENTS.md#codex) | Your local coding-agent session via CLI or MCP | Its existing account/authentication; no mesh-jig API key for local tools |
| [Claude Code](docs/CLIENTS.md#claude-code) | Your local coding-agent session via CLI or MCP | Its existing account/authentication; no mesh-jig API key for local tools |
| [Direct hosted API](docs/API.md#hosted-endpoint) | `mesh-jig agent` calls a compatible vision/tool model | Provider credentials; calls can be billed separately from a client subscription |
| [Direct local API](docs/API.md#local-endpoint) | `mesh-jig agent` calls your existing local model server | No key required for localhost; compute and model-server setup are yours |

The optional vision judge is a separate model call. Deterministic build/render/evaluation does not call
a model unless a judge backend is explicitly requested. Client plans, API access and quotas are separate.

The checkout includes skill entry files for both clients. See [client setup](docs/CLIENTS.md) for MCP
registration, discovery checks and a first prompt. CLI access also works without MCP.

## Start your own model

Supply your own reference sheet and follow [the guide](docs/GUIDE.md#configure-your-own-reference):
**edit crops, cameras, dimensions and scored views before cutting/measuring references**. Freeze them
and the brief before building. Default initialization is for a craft, not a standing character.
The [reference-sheet skill](skill/mesh-jig-reference-sheet/SKILL.md) helps create consistent views.

Every evaluation leaves reference/build comparisons and silhouette overlays beside its numbers.
To browse results, run this yourself (it serves until stopped with Ctrl+C):

```sh
mesh-jig view local-work/my-model --open
```

![One view of an evaluation: the reference beside the build, and the outline diff](showcase/steampunk-mech-opus/image-02.png)

Read [the guide](docs/GUIDE.md) for the workflow and [JIG.md](docs/JIG.md) for the contract fields.

## What comes out

![The steampunk mech reference over three runs: Opus 5.5 in Claude Code, GPT-6.1 Sol in Codex, and Qwen3.8-27B on a local card](showcase/steampunk-mech-opus/image-01.png)

Three of these models are here as GLBs, each with its run's time and tokens:

- [Opus's pirate cat](showcase/pirate-cat-opus/model-01.glb), attempt 16 of 19 ([the run](showcase/pirate-cat-opus/README.md))
- [Qwen's pirate cat](showcase/pirate-cat-qwen-local/model-01.glb), from a local card, attempt 5 of 5
  ([the run](showcase/pirate-cat-qwen-local/README.md))
- [Opus's steampunk mech](showcase/steampunk-mech-opus/model-01.glb), attempt 10 of 13
  ([the run](showcase/steampunk-mech-opus/README.md))

Each row is one run with its own driver and prompt, so the pictures are examples and not a ranking.

[Bruce owl: Astra A/B, Sol, Luna low/medium/high and Qwen](showcase/bruce-owl-comparison/README.md)
shows seven selected models together, with different workflows/budgets and numeric versus visual choices
spelled out. It is not a controlled model ranking.

## Scores and limits

Scores help rank attempts under one fixed project contract. Inspect the images and asset gates before
accepting a model. Numeric matching alone does not establish finished visual quality. The script checker
is a denylist, not a sandbox; run model scripts you trust. See [known gaps](docs/KNOWN_GAPS.md).

## Development and privacy

Use the virtual environment to run `python -m pytest -q`. Tests use synthetic geometry and records;
Blender integration tests skip when Blender is unavailable.

This repository publishes software, reusable instructions, synthetic tests and explicitly selected showcases.
Keep full experiments, personal references, prompts and transcripts in private local storage. Use the
[results-sharing skill](skill/mesh-jig-publish-results/SKILL.md) to publish selected metrics, images and GLBs.
See [publication policy](docs/PUBLICATION.md). Everything published here is MIT; see [LICENSE](LICENSE).
