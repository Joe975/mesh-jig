# Use Codex or Claude Code

Finish [installation](../README.md#install) and the [smoke test](QUICKSTART.md), then start your client
in the checkout. Both can edit scripts and call the CLI. Optional MCP returns images inline through
`brief`, `build`, `preview`, `evaluate`, `paint` and `judge`.

## Codex

The included `.agents/skills/mesh-jig/SKILL.md` points to the [modelling workflow](../skill/mesh-jig/SKILL.md).
Check the skill selector in a new session, or invoke `$mesh-jig`. No user-wide copy is needed.

For MCP, run one block from the checkout. It registers a server in your client configuration;
choose another name if `mesh-jig` is already registered.

PowerShell:

```powershell
$jigPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
codex mcp add mesh-jig -- $jigPython -m mesh_jig.mcp_server
codex mcp get mesh-jig
```

POSIX:

```sh
jig_python="$(pwd)/.venv/bin/python"
codex mcp add mesh-jig -- "$jig_python" -m mesh_jig.mcp_server
codex mcp get mesh-jig
```

## Claude Code

The included `.claude/skills/mesh-jig/SKILL.md` reads the same workflow. Check `/mesh-jig` in a new
session. `CLAUDE.md` holds development instructions.

Optional MCP setup (choose one shell):

```powershell
$jigPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
claude mcp add --transport stdio --scope local mesh-jig -- $jigPython -m mesh_jig.mcp_server
claude mcp get mesh-jig
```

```sh
jig_python="$(pwd)/.venv/bin/python"
claude mcp add --transport stdio --scope local mesh-jig -- "$jig_python" -m mesh_jig.mcp_server
claude mcp get mesh-jig
```

`local` scope registers this checkout in your local client configuration. Don't commit machine-specific
MCP settings. Cloud/Cowork skill discovery differs from local Claude Code.

## Check the connection

Restart the client, inspect `/mcp` in its terminal UI and complete any normal approval.
Call `brief` with the **absolute** path of `local-work/smoke`; expect text and three images.
For Codex app/managed environments, use their MCP UI if CLI settings aren't shared. A remote session
needs its own checkout, environment and Blender; local desktop paths won't work there.

## A first prompt

```text
Use the mesh-jig skill. Read the local-work/smoke brief and existing a00 result.
Open all reference and comparison views. Explain the contract and where the exported GLB is.
This is a synthetic self-match; do not call a judge or API runner, or evaluate a00 again.
```

For real modelling, [prepare a reference project](GUIDE.md#configure-your-own-reference), replace the
project path and specify an attempt budget. Billing/key differences are in the [README](../README.md#choose-a-builder).

## Connection troubleshooting

- Use absolute server/project paths and the `[mcp]` extra. Re-register after moving the environment.
- For Blender discovery, set `MESH_JIG_BLENDER` in the server environment: Codex accepts `--env`;
  Claude accepts `--env` before `--transport stdio`. Quote paths containing spaces.
- The stdio server waits for MCP input, not HTTP requests. Check client status and call `brief`;
  don't disable permission controls. [Local build troubleshooting](GUIDE.md#versions-and-troubleshooting)
  covers Blender and dashboard errors.

Syntax checked against installed `mcp add/get --help` and official [Codex skills](https://developers.openai.com/codex/skills),
[Codex MCP](https://developers.openai.com/codex/mcp), [Claude skills](https://code.claude.com/docs/en/skills)
and [Claude MCP](https://code.claude.com/docs/en/mcp) docs. See [validation limits](GUIDE.md#versions-and-troubleshooting).
