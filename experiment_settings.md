# GOLD external teacher runs

## full_epoch_patch_v42
- Status: running (reverse KL active with updated naming)
- Log: `runs/gold-external-teacher/full_epoch_patch_v42.log`
- W&B run: `rts1s9q6` (run-20260114_140202-rts1s9q6)
- Args:
```
--model-id EverAI-AI/MagistSmall-Raven-ALT-2
--dataset-id runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt
--dataset-split train
--teacher-tokenizer deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-url https://spasmodically-untimeous-marlene.ngrok-free.dev/v1/completions
--teacher-model-name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-full-logprobs-format top_p
--teacher-full-logprobs-top-p 0.9999
--teacher-full-logprobs-max-top-k 1024
--uld-matched-divergence skew_kl
--uld-matched-forward-kl-weight 0.0
--uld-matched-reverse-kl-weight 1.0
--teacher-timeout 10
--teacher-max-retries 1
--max-length 1280
--max-completion-length 128
--max-messages-per-example 30
--teacher-max-input-tokens 2950
--use-lora
--lora-r 32
--learning-rate 1e-5
--num-train-epochs 1
--per-device-train-batch-size 12
--gradient-accumulation-steps 4
--log-rollouts
--log-rollouts-steps 1
--rollouts-per-log 3
--lmbda 1.0
--disable-unmatched-loss
--wandb-run-name full_epoch_patch_v42
```

## full_epoch_patch_v43
- Status: completed (renorm matched probs + alignment logging)
- Log: `runs/gold-external-teacher/full_epoch_patch_v43.log`
- W&B run: `oys3ugl3` (run-20260119_074127-oys3ugl3)
- Args:
```
--model-id EverAI-AI/MagistSmall-Raven-ALT-2
--dataset-id runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt
--dataset-split train
--teacher-tokenizer deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-url https://spasmodically-untimeous-marlene.ngrok-free.dev/v1/completions
--teacher-model-name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-full-logprobs-format top_p
--teacher-full-logprobs-top-p 0.99
--teacher-full-logprobs-max-top-k 4096
--uld-matched-divergence skew_kl
--uld-matched-forward-kl-weight 0.0
--uld-matched-reverse-kl-weight 1.0
--uld-renorm-matched-probs
--teacher-timeout 10
--teacher-max-retries 1
--max-length 1280
--max-completion-length 128
--max-messages-per-example 30
--teacher-max-input-tokens 2950
--use-lora
--lora-r 32
--learning-rate 1e-5
--num-train-epochs 1
--per-device-train-batch-size 12
--gradient-accumulation-steps 4
--log-rollouts
--log-rollouts-steps 1
--rollouts-per-log 3
--log-alignment-groups
--log-alignment-groups-steps 1
--log-alignment-groups-max-samples 1
--lmbda 1.0
--disable-unmatched-loss
--wandb-run-name full_epoch_patch_v43
```

## full_epoch_patch_v41
- Status: stopped manually (run terminated to change KL terminology)
- Log: `runs/gold-external-teacher/full_epoch_patch_v41.log`
- W&B run: `xjonyzzv` (run-20260114_100227-xjonyzzv)
- Args:
```
--model-id EverAI-AI/MagistSmall-Raven-ALT-2
--dataset-id runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt
--dataset-split train
--teacher-tokenizer deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-url https://spasmodically-untimeous-marlene.ngrok-free.dev/v1/completions
--teacher-model-name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-full-logprobs-format top_p
--teacher-full-logprobs-top-p 0.9999
--teacher-full-logprobs-max-top-k 1024
--uld-matched-divergence skew_kl
--uld-matched-forward-kl-weight 0.0
--uld-matched-reverse-kl-weight 1.0
--teacher-timeout 10
--teacher-max-retries 1
--max-length 1280
--max-completion-length 128
--max-messages-per-example 30
--teacher-max-input-tokens 2950
--use-lora
--lora-r 32
--learning-rate 1e-5
--num-train-epochs 1
--per-device-train-batch-size 12
--gradient-accumulation-steps 4
--log-rollouts
--log-rollouts-steps 1
--rollouts-per-log 3
--lmbda 1.0
--disable-unmatched-loss
--wandb-run-name full_epoch_patch_v41
```

## full_epoch_patch_v40
- Status: failed during model download (no space left on device)
- Log: `runs/gold-external-teacher/full_epoch_patch_v40.log`
- W&B: not initialized (no metadata available)
- Args: not captured (run failed before W&B init)

## full_epoch_patch_v39
- Status: failed during model download (no space left on device)
- Log: `runs/gold-external-teacher/full_epoch_patch_v39.log`
- W&B: not initialized (no metadata available)
- Args: not captured (run failed before W&B init)

## full_epoch_patch_v38
- Status: failed during model download (no space left on device)
- Log: `runs/gold-external-teacher/full_epoch_patch_v38.log`
- W&B: not initialized (no metadata available)
- Args: not captured (run failed before W&B init)

## full_epoch_patch_v37
- Status: completed/ran to ~step 35 (see log)
- Log: `runs/gold-external-teacher/full_epoch_patch_v37.log`
- W&B run: `0qevn2nl` (run-20260114_070300-0qevn2nl)
- Args:
```
--model-id EverAI-AI/MagistSmall-Raven-ALT-2
--dataset-id runs/gold-external-teacher/datasets/french-conversations-30msg-1152prompt
--dataset-split train
--teacher-tokenizer deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-url https://spasmodically-untimeous-marlene.ngrok-free.dev/v1/completions
--teacher-model-name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
--teacher-full-logprobs-format top_p
--teacher-full-logprobs-top-p 0.9999
--teacher-full-logprobs-max-top-k 1024
--uld-matched-divergence skew_kl
--uld-matched-forward-kl-weight 0.7
--uld-matched-reverse-kl-weight 0.3
--teacher-timeout 10
--teacher-max-retries 1
--max-length 1280
--max-completion-length 128
--max-messages-per-example 30
--teacher-max-input-tokens 2950
--use-lora
--lora-r 32
--learning-rate 1e-5
--num-train-epochs 1
--per-device-train-batch-size 12
--gradient-accumulation-steps 4
--log-rollouts
--log-rollouts-steps 1
--rollouts-per-log 3
--lmbda 1.0
--disable-unmatched-loss
--wandb-run-name full_epoch_patch_v37
```
