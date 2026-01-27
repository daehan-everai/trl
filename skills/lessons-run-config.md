---
name: lessons-run-config
description: "Run configuration notes for GOLD external-teacher training."
---

# Run Configuration Lessons

- Teacher settings used in this repo for `full_logprobs`:
  - `format=top_p`, `top_p=0.9999`, `max_top_k=2048` (adjust only if teacher supports it).
- Default to **on-policy only** for GOLD runs: set `--lmbda 1.0` (off-policy disabled).
- For matched-only ULD loss (prevents reward hacking by pushing mass into unmatched vocab):
  - `--disable-unmatched-loss`
  - `--uld-matched-divergence skew_kl`
  - `--uld-matched-forward-weight 0.0 --uld-matched-reverse-weight 1.0`
  - `--uld-renorm-matched-probs`
- Keep alignment logging enabled for debugging:
  - `--log-rollouts --log-rollouts-steps 1 --rollouts-per-log N`
  - `--log-alignment-groups --log-alignment-groups-steps 1`
- Checkpoints live under `runs/gold-external-teacher/<run_name>/checkpoint-<step>`.
