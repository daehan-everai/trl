---
name: lessons-env-deps
description: "Environment, cache, and dependency lessons for TRL/GOLD runs."
---

# Environment & Dependencies Lessons

- Keep HF/Transformers/Datasets/Torch caches under `/workspace/models` to avoid root disk issues.
- Use `/workspace/trl/.venv/bin/python` to ensure local TRL and correct dependencies.
- Missing `datasets` causes immediate crashes; ensure `datasets`, `transformers`, `peft`, `wandb`, `rich` are installed in the venv.
