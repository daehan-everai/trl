---
name: gold-external-teacher-run
description: "Run GOLD external-teacher training in /workspace/trl with correct caches, .env credentials, vLLM teacher endpoint settings, and W&B + rich logging."
---

# Run Guide (GOLD external teacher)

## Purpose
Run the GOLD external-teacher training against the vLLM teacher endpoint and log rollouts to W&B.

## Prereqs
- `.env` contains:
  - `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`)
  - `WANDB_API_KEY`
  - Optional defaults: `STUDENT_MODEL_ID`, `TEACHER_VLLM_MODEL_NAME`
- Python deps are installed:
  - External-teacher run (LoRA + W&B): `pip install -e '.[peft]' wandb rich`
  - Local dummy-teacher run (optional): `pip install fastapi uvicorn`
- Teacher endpoint is reachable and supports `full_logprobs` with `format="top_p"`.
- Local caches should live under `/workspace` to avoid root disk space issues.
- `HUGGINGFACE_HUB_TOKEN` is required even for `--report-to-none` or `--save-processed-dataset-only`.
- `WANDB_API_KEY` is required unless `--report-to-none` is set.

## Dataset fix & push (if needed)
If the dataset needs to end on a user turn and drop `prompt`:
```bash
python scripts/fix_french_conversations_dataset.py --repo-id EverAI-AI/french-conversations-prompt --split train
```

## Environment vars (keep caches on /workspace/trl)
```bash
export HF_HOME=/workspace/trl/.cache/hf
export HUGGINGFACE_HUB_CACHE=/workspace/trl/.cache/hf/hub
export TRANSFORMERS_CACHE=/workspace/trl/.cache/hf/transformers
export HF_DATASETS_CACHE=/workspace/trl/.cache/hf/datasets
export WANDB_DIR=/workspace/trl/wandb
export PIP_CACHE_DIR=/workspace/trl/.cache/pip
export TMPDIR=/workspace/trl/.tmp
```

## Run command (current config)
Note: the `--teacher-url` ngrok host is ephemeral; update it before running on a fresh VM. If the saved dataset path
does not exist, use `--dataset-id EverAI-AI/french-conversations-prompt` instead.

```bash
nohup env HF_HOME=/workspace/trl/.cache/hf \
  HUGGINGFACE_HUB_CACHE=/workspace/trl/.cache/hf/hub \
  TRANSFORMERS_CACHE=/workspace/trl/.cache/hf/transformers \
  HF_DATASETS_CACHE=/workspace/trl/.cache/hf/datasets \
  WANDB_DIR=/workspace/trl/wandb \
  TMPDIR=/workspace/trl/.tmp \
  /workspace/trl/.venv/bin/python scripts/run_gold_external_teacher.py \
  --model-id EverAI-AI/MagistSmall-Raven-ALT-2 \
  --dataset-id runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt \
  --dataset-split train \
  --teacher-tokenizer deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --teacher-url https://spasmodically-untimeous-marlene.ngrok-free.dev/v1/completions \
  --teacher-model-name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --teacher-full-logprobs-format top_p \
  --teacher-full-logprobs-top-p 0.99 \
  --teacher-full-logprobs-max-top-k 4096 \
  --uld-matched-divergence skew_kl \
  --uld-matched-forward-kl-weight 0.0 \
  --uld-matched-reverse-kl-weight 1.0 \
  --uld-renorm-matched-probs \
  --teacher-timeout 10 \
  --teacher-max-retries 1 \
  --max-length 1280 \
  --max-completion-length 128 \
  --max-messages-per-example 30 \
  --teacher-max-input-tokens 2950 \
  --use-lora \
  --lora-r 32 \
  --learning-rate 1e-5 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 12 \
  --gradient-accumulation-steps 4 \
  --log-rollouts \
  --log-rollouts-steps 1 \
  --rollouts-per-log 3 \
  --log-alignment-groups \
  --log-alignment-groups-steps 1 \
  --log-alignment-groups-max-samples 1 \
  --disable-unmatched-loss \
  --wandb-run-name full_epoch_patch_v43 \
  > runs/gold-external-teacher/full_epoch_patch_v43.log 2>&1 & echo $!
```

## Quick sanity (1-step, local dummy teacher)
Use this to validate the training loop without relying on an external vLLM endpoint (dummy teacher only supports `base64_dense`):
```bash
env HF_HOME=/workspace/trl/.cache/hf \
  HUGGINGFACE_HUB_CACHE=/workspace/trl/.cache/hf/hub \
  HF_DATASETS_CACHE=/workspace/trl/.cache/hf/datasets \
  WANDB_DIR=/workspace/trl/wandb \
  WANDB_MODE=disabled \
  python scripts/run_gold_external_teacher.py \
  --model-id hf-internal-testing/tiny-random-gpt2 \
  --dataset-id EverAI-AI/french-conversations-prompt \
  --dataset-split 'train[1:50]' \
  --teacher-tokenizer hf-internal-testing/tiny-random-gpt2 \
  --teacher-full-logprobs-format base64_dense \
  --teacher-preflight-requests 0 \
  --max-length 512 \
  --max-completion-length 8 \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --report-to-none
```

## Monitoring
- Log file: `runs/gold-external-teacher/teacher_run.log`
- W&B: use the run URL printed in the log.
- Check for teacher request errors:
```bash
rg -n "Teacher endpoint error|prompt_len|positions_min|positions_max" runs/gold-external-teacher/teacher_run.log
```

## Common failures & fixes
- **Root disk full**: ensure caches point to `/workspace`.
- **Teacher 400 for length**: cap to 4095 (`--max-length` and `--teacher-max-input-tokens`).
- **Teacher endpoint rejects positions**: verify vLLM supports full logprobs for requested positions and that it includes all requested positions.

## Notes
- Rollouts are logged to W&B via `wandb.Table` when `--log-rollouts` is enabled.
- Secrets are loaded from `.env`; do not hardcode tokens in scripts.
