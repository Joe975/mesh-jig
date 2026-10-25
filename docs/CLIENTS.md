# Codex and Claude Code

First finish [installation](../README.md#install) and the [local smoke test](QUICKSTART.md).
These instructions concern local sessions on the machine that has Blender and the checkout. A remote
session needs its own files, Python environment and Blender; desktop paths do not exist there automatically.

Both clients can write scripts and call the CLI directly. MCP is optional: it returns comparison images
inline instead of requiring the client to open files. Its tools are `brief`, `build`, `preview`, `evaluate`,
`paint` and `judge`; file editing still belongs to the coding client. Local tools need no mesh-jig API key.
Your client's own authentication, plan and quotas still apply. Leave the paid judge unused unless intended.

## Codex

Start Codex in this checkout. The included `.agents/skills/mesh-jig/SKILL.md` entry points to the canonical
[modeling skill](../skill/mesh-jig/SKILL.md); `AGENTS.md` also routes tasks to it. No user-wide skill copy
is required. In a new session, check the skill selector for `mesh-jig`; you can invoke `$mesh-jig` explicitly.

To add the optional local MCP server, run **one** shell block from the checkout. This writes a server entry
to your Codex configuration; it does not sign in or configure a model provider. Choose another server name
if `mesh-jig` is already registered, and do not overwrite a working configuration inadvertently.

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

Restart the local Codex session and use `/mcp` in the terminal UI to check the server. Ask it to call
`brief` with the **absolute** path of `local-work/smoke`; expect text and three reference images. App and
managed-environment configuration surfaces can differ; use their MCP UI if the CLI configuration is not
shared. A listed configuration alone does not prove that the server connected.

Verified syntax sources: [official skills documentation](https://developers.openai.com/codex/skills) and
[official MCP documentation](https://developers.openai.com/codex/mcp), plus installed `codex mcp add --help`
and `codex mcp get --help`. This documentation update did not perform an authenticated client session.

## Claude Code

Start `claude` in this checkout. The included `.claude/skills/mesh-jig/SKILL.md` entry reads the same
canonical modeling skill. Check `/mesh-jig` in the skill/command menu in a new local session. `CLAUDE.md`
separately contains development instructions; it is not a replacement for the modeling skill.

Optional MCP setup, from the checkout (choose one shell):

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

`local` scope keeps registration specific to this checkout in your local client configuration. Restart
Claude Code, inspect `/mcp`, complete any normal client approval, and request the same `brief` call.
Do not commit a machine-specific MCP configuration. Skill discovery in cloud/Cowork sessions differs
from local Claude Code; this setup does not configure those environments.

Verified syntax sources: [official skills documentation](https://code.claude.com/docs/en/skills) and
[official MCP documentation](https://code.claude.com/docs/en/mcp), plus installed `claude mcp add --help`
and `claude mcp get --help`. Connection and model interaction were not exercised for this update.

## A first prompt

After the smoke evaluation, ask either client:

```text
Use the mesh-jig skill. Read the local-work/smoke brief and existing a00 result.
Open all reference and comparison views. Explain the contract and where the exported GLB is.
This is a synthetic self-match; do not call a judge or API runner, or evaluate a00 again.
```

For real modeling, first [prepare your own frozen reference project](GUIDE.md#configure-your-own-reference),
then replace the project in the prompt and specify an attempt budget. The client writes each candidate
script and drives the build/preview/evaluation loop. It can use CLI commands if MCP is unavailable.

## Connection troubleshooting

- Use the absolute virtual-environment Python executable, with arguments `-m mesh_jig.mcp_server`.
  Installing the `[mcp]` extra is required. Re-register after moving the checkout or environment.
- Every MCP project/script path is local to the server. Use absolute paths; its launch directory need
  not be the checkout. A client and server on different machines cannot share local path strings.
- If Blender is not discoverable, set `MESH_JIG_BLENDER` in the server's inherited environment or
  registration environment (`--env` for Codex, `--env` before `--transport stdio` for Claude Code).
  Use an absolute executable path and quote values containing spaces. The API `.env` loader is not a
  general MCP/Blender environment initializer.
- A manual stdio server waits for MCP input; it is not an HTTP dashboard. Use client status and the
  `brief` smoke check instead of waiting for a web address. Use [the guide](GUIDE.md#versions-and-troubleshooting)
  for local build and dashboard errors. Do not disable client permission controls to make setup pass.
