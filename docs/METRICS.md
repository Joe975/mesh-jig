# Per-stage metrics (forward-only, opt-in)

Keep telemetry under the run's ignored `local-work/` directory. It is independent of model scores,
contract gates, old reports and the public site. No historical results are rewritten.

## New API runs

Global options precede the subcommand. Use the same directory and run name across stages:

```text
mesh-jig --metrics-dir local-work/trial/metrics --metrics-run trial --metrics-stage model agent local-work/trial/model --model MODEL
mesh-jig --metrics-dir local-work/trial/metrics --metrics-run trial --metrics-stage rig-animation agent local-work/trial/rig --workflow rig --model MODEL
python -m mesh_jig.metrics local-work/trial/metrics --markdown
```

The commands still need their normal approved model settings and budget caps. Telemetry neither
changes budgets nor calls a model itself. Never add metrics as a reason to launch a paid test.

For Python callers, `with mesh_jig.metrics.stage(directory, name, run):` provides the same context.
For native coding agents, set `MESH_JIG_METRICS_DIR`, `MESH_JIG_METRICS_RUN` and
`MESH_JIG_METRICS_STAGE` in that child environment. Jig commands inherit them. Use a fresh stage
identifier when the task changes. This captures tools, **not the native model's usage by itself**.
Native usage capture is described by `python -m mesh_jig.metrics_native --help`.

## What the ledger means

Each event is written atomically in `events/`. Independent processes do not append to the same
file. Start events survive interruptions; spans without end events are unfinished, not successful.
A crash can leave a `.tmp` file, which is ignored. Reporting fails on malformed final events rather
than silently excluding them. Keep this raw ledger private. The derived JSON/Markdown has no
prompts, responses, raw errors, session IDs, credentials, local paths or endpoint addresses.

- A **stage** is the explicit task scope, e.g. model, rig-animation, facial, terrain or export.
  Do not invent separate rig/animation token totals when one conversation did both.
- `evaluation`, `rig-build`, `sound-build` count actual tool invocations, with pass/fail/unfinished
  counts. These are not unique scripts or model API requests. Re-evaluating a script counts again.
- A **request** is one HTTP attempt. Retries are separate. A budget refusal before dispatch is not
  a request. Cached tokens and reasoning tokens are details of inclusive input/output totals,
  not extra tokens to add. Anthropic's separate input/cache fields are normalized explicitly.
- Tokens are **known subtotals** with reporting coverage. Missing/failed calls never become zero.
  Explicit provider zero is valid. Provider totals are not estimates from character length.
  Visual and rig accounting also expose cache read/write subtotals and reporting-call coverage.
  Native Anthropic compatibility totals include uncached input plus cache reads plus cache writes exactly once;
  per-TTL writes are priced separately. An incomplete cache receipt cannot release a reserved spend cap.
- Only provider-reported numeric USD cost is included by this ledger. Missing cost is unknown,
  not free; subscription prices, list-price estimates and budget reservations are not invoices.
  The Markdown table shows each run, stage and known USD subtotal with reporting coverage.
  Native receipt imports currently capture time and tokens; their cost remains unknown.
- Timing uses monotonic durations. `observed_active_s` is the union of observed intervals (not
  their sum); `summed_s` is worker-time and can exceed elapsed time with parallel workers.
  Tool/request spans overlap the containing stage: do not add them to stage duration.
  Native imported durations cannot locate execution on the wall clock, so their active union is
  unavailable. Queued time, manual pauses, GPU utilization and uninstrumented work remain unknown.

## Current integration and limits

The CLI measures commands; API transports measure blocking and streaming HTTP attempts, including
local endpoints and native Anthropic. Evaluation, rig and sound tool boundaries are instrumented.
The opt-in environment/context can be used for optional stages on feature branches without
changing their scores. Their detailed per-tool boundaries still need integration when merged.

No promise is made that every native provider exposes usage. Codex/Claude/Antigravity counters need
explicit supported exports. Resumed sessions require proven cumulative-vs-incremental semantics;
never sum snapshots blindly. Export only a sanitized derived summary to a future results page,
not the event directory. The Markdown stage table is available now; existing site entries remain
unchanged. Tokens across different tokenizers are useful costs/work indicators, not identical units
of model capability.

## Native receipt import example

Capture a fresh, stage-scoped CLI invocation with its structured output enabled. Then import its
saved JSON/JSONL receipt explicitly (this command does not run a model):

```powershell
python -m mesh_jig.metrics_native local-work/metrics/run-001 local-work/codex-model.jsonl --provider codex --run run-001 --stage model --invocation model-01 --scope invocation
```

For a provider-confirmed cumulative resumed session, use `--scope cumulative --before <snapshot>`;
the same-session counters must not decrease. Antigravity input semantics remain unknown unless
verified and specified with `--agy-input includes-cache` or `excludes-cache`. Do not import the
same conversation usage through multiple sources or combine native session totals with direct API
receipts covering those same calls. Only separate, non-overlapping usage belongs in one ledger.
Native imports are explicit, not automatic desktop session scraping. They count native sessions,
not inferred HTTP attempts. Split future work into stage-scoped invocations to obtain stage stats.
