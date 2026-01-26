---
name: lessons-run-config
description: "Run configuration notes for GOLD external-teacher training."
---

# Run Configuration Lessons

- Teacher settings used in this repo for `full_logprobs`:
  - `format=top_p`, `top_p=0.9999`, `max_top_k=2048` (adjust only if teacher supports it).
- Keep alignment logging enabled for debugging:
  - `--log-rollouts --log-rollouts-steps 1 --rollouts-per-log N`
  - `--log-alignment-groups --log-alignment-groups-steps 1`
