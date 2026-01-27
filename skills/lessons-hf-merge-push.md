---
name: lessons-hf-merge-push
description: "Merging LoRA checkpoints and publishing to Hugging Face Hub."
---

# HF Merge & Push Lessons

- The base model `EverAI-AI/MagistSmall-Raven-ALT-2` uses **Mistral3**; `AutoModelForCausalLM` fails to load it.
  - Use `transformers.models.mistral3.Mistral3ForConditionalGeneration` instead.
- Access to the base repo and uploads require `HF_TOKEN` from `.env`.
- Merge flow (CPU OK for LoRA): load base model, load adapter with `PeftModel`, `merge_and_unload()`, then `save_pretrained(..., safe_serialization=True)`.
- Tokenizer: prefer the checkpoint tokenizer (may include updated chat template). If `pad_token` is missing, set it to `eos_token` before saving.
- Keep non-training artifacts identical to base unless explicitly changed:
  - Copy `config.json` and `tokenizer_config.json` from the base repo.
  - Include `preprocessor_config.json`, `processor_config.json`, `special_tokens_map.json` from base.
- Large uploads: bump `huggingface_hub` HTTP timeouts using `_http.set_client_factory` and enable `HF_XET_HIGH_PERFORMANCE=1` for reliability.
- Verify upload with `HfApi().list_repo_files(repo_id)`.
