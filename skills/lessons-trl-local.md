---
name: lessons-trl-local
description: "Local TRL codepath gotchas and fixes."
---

# Local TRL Lessons

- The runs use the **local TRL** package from this repo (`/workspace/trl/trl`), not site-packages.
- Changes to local files take effect immediately when using `/workspace/trl/.venv/bin/python`.
- Apex-related bug: guard `use_apex` with `getattr(self, "use_apex", False)` in `trl/experimental/gold/gold_trainer.py`.
- GOLD training logic lives in `trl/experimental/gold/gold_trainer.py`; use that for fixes and instrumentation.
- If unsure, sanity-check with `python -c "import trl; print(trl.__file__)"` to confirm local import.
