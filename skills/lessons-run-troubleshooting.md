---
name: lessons-run-troubleshooting
description: "Diagnosing GOLD training run exits and stalled runs."
---

# Run Troubleshooting Lessons

- If a run stops without an explicit error in the log, check `trainer_state.json`:
  - Compare `global_step` vs `max_steps`. If `global_step < max_steps`, the process likely died externally (OOM/preemption/manual kill).
- Search logs quickly for hard failures:
  - `rg -n "Traceback|ERROR|Exception|OOM|out of memory|Killed" <log>`
- Confirm whether a run is still alive:
  - `ps -ef | rg -n "gold|run_gold_external_teacher|gold_trainer"`
- If logs end mid-rollout and no error is present, treat it as a non-graceful termination and restart from the latest checkpoint.
