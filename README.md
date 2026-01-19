# GOLD External Teacher Run

## Purpose
Run GOLD external-teacher training against a vLLM teacher endpoint and log rollouts to W&B.

## Prereqs
- `.env` contains:
  - `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`)
  - `WANDB_API_KEY`
  - Optional defaults: `STUDENT_MODEL_ID`, `TEACHER_VLLM_MODEL_NAME`
- Python deps are installed:
  - External-teacher run (LoRA + W&B): `pip install -e '.[peft]' wandb rich`
  - Local dummy-teacher run (optional): `pip install fastapi uvicorn`
- Teacher endpoint is reachable and supports `full_logprobs` with `format="top_p"` (avoid `base64_dense` to reduce ngrok payload cost).
- Local caches should live under `/workspace` to avoid root disk space issues.

## Dataset fix and push (if needed)
Only use this if you want on-policy-only runs (`--lmbda 1.0`). Any off-policy loss (`--lmbda < 1.0`) requires the
dataset to end on an assistant turn so the last assistant message is used as the completion:
```bash
python scripts/fix_french_conversations_dataset.py --repo-id EverAI-AI/french-conversations-prompt --split train
```

## Environment vars (keep caches on /workspace)
```bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export WANDB_DIR=/workspace/wandb
```

## Run command (current config)
```bash
nohup env HF_HOME=/workspace/.cache/huggingface \
  HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
  TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  WANDB_DIR=/workspace/wandb \
  python scripts/run_gold_external_teacher.py \
  --model-id EverAI-AI/MagistSmall-Raven-ALT-2 \
  --dataset-id EverAI-AI/french-conversations-prompt \
  --dataset-split train \
  --teacher-tokenizer deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --teacher-url https://spasmodically-untimeous-marlene.ngrok-free.dev/v1/completions \
  --teacher-model-name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --teacher-full-logprobs-format top_p \
  --teacher-full-logprobs-top-p 0.9999 \
  --teacher-full-logprobs-max-top-k 1024 \
  --uld-matched-divergence skew_kl \
  --uld-matched-forward-kl-weight 0.7 \
  --uld-matched-reverse-kl-weight 0.3 \
  --save-steps 50 \
  --teacher-timeout 10 \
  --teacher-max-retries 1 \
  --max-length 1280 \
  --max-completion-length 128 \
  --max-messages-per-example 30 \
  --save-processed-dataset runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt \
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
  --lmbda 1.0 \
  --disable-unmatched-loss \
  > runs/gold-external-teacher/teacher_run.log 2>&1 & echo $!
```

## Save a sliced dataset (no training)
```bash
python scripts/run_gold_external_teacher.py \
  --model-id EverAI-AI/MagistSmall-Raven-ALT-2 \
  --dataset-id EverAI-AI/french-conversations-prompt \
  --dataset-split train \
  --max-length 1280 \
  --max-completion-length 128 \
  --max-messages-per-example 30 \
  --save-processed-dataset runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt \
  --save-processed-dataset-only \
  --report-to-none
```
To reuse a saved dataset, pass the path as `--dataset-id` (split is ignored unless the saved dataset contains splits).

## Best practices
- Start with a short sanity run (`--max-steps 1` or small split) before a full epoch.
- Keep rollouts logging at `--log-rollouts-steps 1` until outputs look sane, then increase.
- For on-policy-only runs (`--lmbda 1.0`), the dataset should end on a user turn.
- For any off-policy loss (`--lmbda < 1.0`), the dataset must end on an assistant turn and include at least one user turn.
- When `0 < --lmbda < 1`, on-policy and off-policy losses are computed every batch and combined as a weighted sum.
- Off-policy loss is CE on dataset completions; on-policy CE anchors are disabled (`--uld-crossentropy-weight` is ignored).
- Keep `--max-length` and `--teacher-max-input-tokens` within the teacher’s context window.
- Prompt budget is `--max-length - --max-completion-length`.
- Use `--max-messages-per-example` to slice long conversations instead of dropping them when lowering prompt budgets.
- Save processed datasets with `--save-processed-dataset` so future runs reuse the same slicing.
- Checkpoints are saved every `--save-steps` into a per-run subdirectory under `--output-dir`.
- For hybrid ULD matched tokens, you can choose `--uld-matched-divergence` and optionally weight forward/reverse KL.
- Always verify W&B completions end at the stop marker and do **not** include a new user turn.

## Quick sanity (1-step, local dummy teacher)
Use this to validate the training loop without relying on an external vLLM endpoint (dummy teacher only supports `base64_dense`):
```bash
env HF_HOME=/workspace/.cache/huggingface \
  HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  WANDB_DIR=/workspace/wandb \
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
- Confirm off-policy is active by checking `off_policy_loss` when `--lmbda < 1`.
- Confirm completions are valid by checking `num_valid_completion_tokens` stays > 0.
- For cross-tokenizer alignment, monitor `alignment_groups_student` / `alignment_groups_teacher` for sudden drops.
- Alignment group samples (when enabled) are written to `runs/gold-external-teacher/<run>/alignment_groups.jsonl`.
- W&B tables: `completions` (on-policy) and `off_policy_completions` (dataset completions).
- For `--uld-matched-divergence skew_kl`, `forward` means KL(teacher||student) and `reverse` means KL(student||teacher).
- Check for teacher request errors:
```bash
rg -n "Teacher endpoint error|prompt_len|positions_min|positions_max" runs/gold-external-teacher/teacher_run.log
```

## Changing models
### Student model
- Rollout stopping is keyed off the chat template stop marker. For ChatML models it is `<|im_end|>`.
- If a new student uses a different template (e.g., `<|eot_id|>` or `</s>`), update the template or adjust
  the stop marker in `trl/experimental/gold/gold_trainer.py` to match that string.
- No hardcoded token IDs are required; the stop string is detected from the template.

### Teacher model
- Update: `--teacher-tokenizer`, `--teacher-model-name`, and `--teacher-url`.
- Set `--teacher-max-input-tokens` to the teacher’s real context limit (or slightly under it).
- Ensure the teacher endpoint supports `full_logprobs` with `format="top_p"`.
- If using sparse top-p logprobs, increase `--teacher-full-logprobs-max-top-k` to reduce tail-mass approximation.
- `--teacher-model-name` must match the model actually served by the endpoint (404s mean a mismatch).

## Notes
- Run notes (2026-01-09)
  - Observed: rollouts degenerate after ~40 steps (missing or corrupted `<|im_start|>`/`<|im_end|>` and extra user turns appear in completions).
  - Observed: completion logs included `<|im_start|>assistant` and sometimes subsequent user/assistant turns; prompt/completion boundary looked misaligned.
  - Observed: prompt logs sometimes start with an assistant turn (dataset may begin with assistant content, or prompt extraction may include an initial assistant message).
  - Observed: log file initially lacked step-by-step prompt/completion tables due to missing `rich` and stdout redirection.
  - Observed: stop behavior was inconsistent when using left-padded prompts; completions could include extra turns even when `--log-rollouts-steps 1` was enabled.
  - Approach: installed `rich`, routed `rich` output to stdout, and added a plain-text fallback so rollout tables land in `runs/gold-external-teacher/teacher_run.log`.
  - Approach: fixed conversational prompt/completion extraction to split only on the final assistant turn (keep full history), and preserved prompt/completion text for ULD alignment.
  - Approach: expanded stop candidates to include `<|im_end|>`, `<|eot_id|>`, and newline variants derived from chat template/special tokens; added post-generation trimming after the first stop marker.
  - Approach: corrected stop-length handling and label/completion slicing to use per-example prompt lengths (left padding) instead of global prompt length.
  - Status: current run still shows extra user turns in some completions when inspecting samples; degeneration persists and needs further investigation.
- Run notes (2026-01-19)
  - Change: added `--uld-renorm-matched-probs` to renormalize matched-token probabilities before matched divergence.
  - Change: added alignment-group logging (`--log-alignment-groups`, `--log-alignment-groups-steps`, `--log-alignment-groups-max-samples`).
  - Observed: alignment group counts stayed stable through step 200+; no degeneration in completion structure.
  - Observed: final-step loss was ~1.33 while matched_loss stayed ~5.33; this is expected when the last gradient-accumulation chunk is shorter and loss is scaled by the current accumulation steps.
  - Output: merged checkpoint-200 pushed to `EverAI-AI/MagistSmall-Raven-ALT-2_full_epoch_patch_v43`.
- Rollouts are logged to W&B via `wandb.Table` when `--log-rollouts` is enabled.
- Secrets are loaded from `.env`; do not hardcode tokens in scripts.
