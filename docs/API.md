# Direct API builder and optional judge

This route is separate from [Codex and Claude Code](CLIENTS.md). It does not reuse a coding client's
subscription or login. Start with [installation](../README.md#install), verify the local
[smoke project](QUICKSTART.md), then prepare a [real reference project](GUIDE.md#configure-your-own-reference).
Run from the checkout. Shell examples assume the installed environment; on PowerShell use its explicit
Python executable as in the README.

## Protocol and model

`mesh-jig agent` requires an **OpenAI-compatible Chat Completions** base URL. The client appends
`/chat/completions`, sends tools and image messages, and reads streamed Chat Completions responses.
Do not supply a full completion URL or assume a native Messages/Responses endpoint works. Use a model
and endpoint that accept vision input, function tools and streaming. Provider model identifiers are
not necessarily Codex/Claude Code model aliases. Check your provider's current model list and compatibility;
no live provider/model compatibility was certified by this documentation update.

## Hosted endpoint

Copy `.env.example` to ignored `.env` in the **directory from which you launch mesh-jig**, then edit it
locally. Uncomment the settings you use; commented examples do nothing. For example:

```dotenv
MESH_JIG_AGENT_URL=https://openrouter.ai/api/v1
MESH_JIG_AGENT_MODEL=REPLACE_WITH_YOUR_PROVIDER_VISION_TOOL_MODEL_ID
MESH_JIG_AGENT_KEY=REPLACE_WITH_YOUR_PROVIDER_KEY
```

Replace both placeholders before running. Never commit keys or pass them as command-line arguments.
An explicit environment variable wins over `.env`; `--url` and `--model` override their settings.
`MESH_JIG_ENV_FILE` can select an absolute env-file path when you launch elsewhere. Without `*_KEY`,
the URL selects `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or otherwise
`OPENAI_API_KEY`. This key lookup is not a native-protocol adapter. The agent has **no default model**.

Once configured, check setup without making a model call, then deliberately start the paid run:

```sh
mesh-jig doctor local-work/my-model
mesh-jig agent local-work/my-model --attempts 5 --max-turns 40 --max-wall 600 --max-cost 1.50
```

These are illustrative limits, not a cost estimate. `--max-cost` uses the dollar cost returned by the
endpoint. If it returns no positive cost, a capped run stops **after its first call**, which may already
be billed. Calls are charged before their usage is known, so a cap can be exceeded by a call; this is
not a prepaid or provider-enforced ceiling. Configure provider-side limits where available. Turn and
wall-time caps do not guarantee any particular dollar spend. The optional `--judge` enables additional
calls through a separately configured backend.

## Local endpoint

Start your own compatible vision/tool model server separately; mesh-jig does not install or launch it.
Replace the model ID and port with what that existing server actually serves:

```sh
mesh-jig agent local-work/my-model --url http://localhost:1234/v1 --model YOUR_LOADED_MODEL_ID --attempts 5 --max-turns 40 --max-wall 600 --cache off
```

Localhost/127.0.0.1 needs no key according to mesh-jig; your server may require one. Clear unwanted
`MESH_JIG_AGENT_KEY`/provider-key overrides in your own launch environment, or use a clean environment
if the server rejects them. Do not use a dollar cap with a server that reports no monetary cost; use
turn/time/attempt limits. An endpoint on another host requires a configured key in mesh-jig. Your local
compute, server model/context limits and its API compatibility remain your responsibility.

## Judge, outputs and recovery

The optional pairwise judge needs vision input and compatible responses. Its settings are independent:
`MESH_JIG_JUDGE_URL`, `MESH_JIG_JUDGE_MODEL`, `MESH_JIG_JUDGE_KEY`. Its defaults and fallback keys are in
[the environment table](JIG.md#environment). Keep it disabled for key-free local builds. `doctor` can
report it unconfigured without failing an otherwise valid local project.

Attempts, comparisons and GLBs are stored [as in the walkthrough](QUICKSTART.md#what-success-looks-like).
The API runner additionally writes `agent/<timestamp>_<model>/report.json` and `transcript.jsonl` under
the project, and prints that run directory. Preserve partial projects/reports after interruption. Inspect the ledger before continuing;
do not re-evaluate a completed attempt to read it. Use `mesh-jig show` instead. A fresh project copied
from the frozen seed gives the clearest independent run. Raw API records stay private.

For errors, check model/key/base URL, image/tool/stream support and the server context window. Set
`--context-tokens` to the endpoint capacity and reserve reply tokens if long runs exceed context.
`--idle` can be raised for servers that reason silently; it does not fix an incompatible API. Inspect
provider errors locally without publishing raw transcripts. [CLI help](GUIDE.md#versions-and-troubleshooting)
lists all budgets and recovery controls. No API calls are made by installation or the smoke example.
