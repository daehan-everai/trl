---
name: lessons-env-deps
description: "Environment, cache, and dependency lessons for TRL/GOLD runs."
---

# Environment & Dependencies Lessons

- Keep HF/Transformers/Datasets/Torch caches under `/workspace/models` (per repo convention) to avoid root disk issues.
- Use `/workspace/trl/.venv/bin/python` to ensure local TRL and correct dependencies.
- Ensure `.env` contains `HF_TOKEN` (base model access + Hub upload) and load it before running scripts.
- Missing `datasets` causes immediate crashes; ensure `datasets`, `transformers`, `peft`, `wandb`, `rich` are installed in the venv.
- For large Hub uploads, set `HF_XET_HIGH_PERFORMANCE=1` to improve transfer reliability.
