# Development principles

Use deterministic measures to guide iteration and independent visual review to judge model quality.
Freeze the reference contract before comparing candidates. Confirm an apparent gain with fresh runs.
Keep an incumbent and record rejected changes privately so future work can learn from them.

Treat instructions as executable interfaces: test documented commands, links, script rules and CLI defaults.
Every logic change needs a fast deterministic regression test. Prefer synthetic fixtures with known geometry
and expected behavior over machine-specific archived runs. Preserve exact output parity when optimizing memory.

Keep software and private research separate. Ignoring a tracked file does not remove its earlier versions
from Git history. Publish only an explicitly reviewed file inventory and scan the entire history being sent.
