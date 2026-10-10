# Pirate cat: Opus 5.5 in Claude Code

| Measure | Value |
|---|---|
| Model | claude-opus-5-5 |
| Driver | Claude Code 2.1.289 with the mesh-jig skill |
| Effort | high |
| Scope | One fresh run with no attempt cap, stopped when three evaluated attempts in a row had not beaten the best. The model is attempt 16 of 19, the best by numeric score. |
| Completed | true |
| Evaluations | 19 |
| Runtime seconds | 3415 |
| Driver turns | 249 |
| Prompt tokens | 19129203 |
| Cached prompt tokens | unknown |
| Cache write tokens | unknown |
| Completion tokens | 271665 |
| Reasoning tokens | unknown |
| Model calls | unknown |
| Cost USD (estimated) | 12.316233 |
| Numeric score | 0.744 |
| Gates passed | 17 |
| Gates total | 19 |

Token counts use the report's provider conventions. Reasoning is included in completion tokens; cache counters are not extra prompt tokens. Estimated cost is not an invoiced charge.

The picture sets this attempt under the reference views and above two other runs on the same reference: GPT-6.1 Sol in Codex and Qwen3.8-27B through mesh-jig agent on a local RTX 4090. Each run had its own driver and prompt, so the rows are examples and not a ranking.

Cost is the list-price figure Claude Code reported for a run made on a subscription. This is one run, and the numeric score does not measure finished visual quality.

![Selected output](image-01.png)

[Download selected model](model-01.glb)
