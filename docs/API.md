# Run an API builder or judge

[Install](../README.md#install), check the [smoke example](QUICKSTART.md), then
[prepare your reference project](GUIDE.md#configure-your-own-reference). Run from the installed checkout;
use the README's PowerShell command substitutions where needed.

## Protocol and model

Supply an **OpenAI-compatible Chat Completions base URL** and a model supporting vision, tools and
streaming. mesh-jig appends `/chat/completions`; native Messages/Responses endpoints aren't interchangeable.
The agent has no default model. Use the provider's model ID, which can differ from coding-client aliases.

## Hosted endpoint

Copy `.env.example` to ignored `.env` in the launch directory. Uncomment the settings and replace
both placeholders:

```dotenv
MESH_JIG_AGENT_URL=https://openrouter.ai/api/v1
MESH_JIG_AGENT_MODEL=REPLACE_WITH_YOUR_PROVIDER_VISION_TOOL_MODEL_ID
MESH_JIG_AGENT_KEY=REPLACE_WITH_YOUR_PROVIDER_KEY
```

Environment variables override `.env`; `--url` and `--model` override their settings.
`MESH_JIG_ENV_FILE` selects another env file by absolute path. [Fallback keys and defaults](JIG.md#environment)
are provider-specific. Keep keys out of Git and command arguments. API access/billing is separate from
coding-client subscriptions.

Check configuration, then start the paid run:

```sh
mesh-jig doctor local-work/my-model
mesh-jig agent local-work/my-model --attempts 5 --max-turns 40 --max-wall 600 --max-cost 1.50
```

These are example limits, not a cost estimate. `--max-cost` uses provider-reported dollars **after**
each call, so it can overshoot. Without positive reported cost it stops after the first, potentially
billed call. Use provider-side spending limits where available; turn/time caps don't cap dollars.

## Local endpoint

Start your own compatible server, then substitute its loaded model ID and port:

```sh
mesh-jig agent local-work/my-model --url http://localhost:1234/v1 --model YOUR_LOADED_MODEL_ID --attempts 5 --max-turns 40 --max-wall 600 --cache off
```

mesh-jig requires no key for localhost/127.0.0.1; your server may require one. Other hosts need a key.
Check inherited key settings if the server rejects authentication. Without reported monetary cost,
use attempt/turn/time limits rather than `--max-cost`.

## Judge, outputs and recovery

The optional paid judge uses separate `MESH_JIG_JUDGE_*` settings; `--judge` enables extra calls.
Check [current provider compatibility and validation limits](GUIDE.md#versions-and-troubleshooting)
before choosing a backend.

The runner prints its `agent/<timestamp>_<model>/` directory, containing `report.json` and
`transcript.jsonl`. [Attempt outputs](QUICKSTART.md#what-success-looks-like) stay beside the models.
Keep raw records private. Preserve interrupted runs; inspect the ledger and use `mesh-jig show`
to read results without duplicating evaluations. Use a fresh frozen seed for an independent run.

For failures, check the base URL, model, key and image/tool/stream support. Set `--context-tokens`
to the server's capacity; reserve reply space. Raise `--idle` for silent reasoning, not protocol errors.
Run `mesh-jig agent --help` for all controls.
