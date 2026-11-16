# Development principles

Use deterministic measures to guide iteration and independent visual review to judge model quality.
Freeze the reference contract before comparing candidates. Confirm an apparent gain with fresh runs.
Keep an incumbent and record rejected changes privately so future work can learn from them.

Treat instructions as executable interfaces: test documented commands, links, script rules and CLI defaults.
Every logic change needs a fast deterministic regression test. Prefer synthetic fixtures with known geometry
and expected behavior over machine-specific archived runs. Preserve exact output parity when optimizing memory.

Keep software and private research separate. Ignoring a tracked file does not remove its earlier versions
from Git history. Publish only an explicitly reviewed file inventory and scan the entire history being sent.

## Long reasoning, context and caching

The DeepSeek high-effort horse/horror trials reported US$4.001279 in total charges. They used controlled
continuations, process steering and harness fixes, so they do not establish an effort-quality ranking.

- **Preserve reasoning continuity.** The visual and rig loops now return completed OpenRouter
  `reasoning_details` unchanged with tool results. Truncated replies are discarded and retried within the turn
  budget; their tools never run. These fixes have deterministic regressions.
- **Budget retained context separately from output.** A 100k reply allowance does not raise the rig loop's
  90k soft context limit or its 8k reserve. The horse compacted eight times in 84 turns. A provider-aware context
  limit, a reserve matching the actual output allowance, and larger-context comparisons are the next experiment;
  increased context has not yet demonstrated a quality gain here.
- **Improve cache reuse rather than assuming caching is off.** Final rig-stage input cache reads were 80.6%
  for horse and 75.6% for horror. Keep system/brief/reference prefixes stable, separate implicit-cache retention
  from provider-specific explicit markers, and test opaque per-stage `session_id` routing across changing
  iteration prompts. Cache hits reduce repeated input work, not new reasoning generation.
- **Persist complete feedback.** The builder rebuilt unchanged failed scripts to recover stretch diagnostics.
  Save structured readings and readable feedback beside each attempt so compaction can point to exact evidence.
- **Feed terrain failures upstream.** Horse's run-slope slip remained 0.696–0.700 m/s across three configuration
  proposals. Check joint reach, stance and landing during rigging, before freezing the asset. Treat the suspected
  reach limitation as a hypothesis until independently isolated; widened joint bounds are not a verified repair.
- **Keep limited passes limited.** Horror passed primary flat/slope checks but failed obstacle stress checks,
  despite a reassuring self-review. Show numeric stress results alongside worst-event frames and preserve their
  failed status. Both published terrain demos remain experimental.

See OpenRouter's [reasoning replay contract](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and [prompt caching guidance](https://openrouter.ai/docs/guides/best-practices/prompt-caching).
