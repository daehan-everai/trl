---
name: trl-experiment-ops
description: "Run or restart TRL/GOLD experiments in /workspace/trl with correct cache/artifact locations, .env credentials, W&B tracking, and rich completion tables."
---

# TRL Experiment Ops

Use /workspace for all caches and artifacts (never / or /root). For this repo, prefer `/workspace/trl/...`.

## Always set cache/artifact paths

Set before any install or run:

```bash
export HF_HOME=/workspace/trl/.cache/hf
export HUGGINGFACE_HUB_CACHE=/workspace/trl/.cache/hf/hub
export TRANSFORMERS_CACHE=/workspace/trl/.cache/hf/transformers
export HF_DATASETS_CACHE=/workspace/trl/.cache/hf/datasets
export XDG_CACHE_HOME=/workspace/trl/.cache
export WANDB_DIR=/workspace/trl/wandb
export WANDB_CACHE_DIR=/workspace/trl/.cache/wandb
export PIP_CACHE_DIR=/workspace/trl/.cache/pip
export TMPDIR=/workspace/trl/.tmp
export TORCH_HOME=/workspace/trl/.cache/torch
```

If a run started without these, stop it and restart with the envs above.

## Credentials

Ensure `/workspace/trl/.env` includes `WANDB_API_KEY` and `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`). If a script does not load `.env`, load it manually before running.

## W&B + rich

Install in the workspace cache if missing (prefer venv with system site packages to avoid /root disk pressure):

```bash
python -m venv --system-site-packages /workspace/trl/.venv
PIP_CACHE_DIR=/workspace/trl/.cache/pip TMPDIR=/workspace/trl/.tmp \
  /workspace/trl/.venv/bin/pip install wandb rich
```

For `scripts/run_gold_external_teacher.py`, always enable per-step completion logs:

```bash
--log-rollouts --log-rollouts-steps 1 --rollouts-per-log <N>
```

Rich tables render only when `rich` is installed.

## Logs and outputs

Write log files under `runs/` and keep `output_dir` under `/workspace/trl/...`.

If you see “no space left on device”, purge stray root caches (e.g., `/root/.cache/pip`) and restart with the workspace cache envs above.
