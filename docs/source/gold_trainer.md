# General Online Logit Distillation (GOLD) Trainer

[![All_models-GOLD-blue](https://img.shields.io/badge/All_models-GOLD-blue)](https://huggingface.co/models?other=sft,gold)

## Overview

General Online Logit Distillation (GOLD) is an extension of Universal Logit Distillation (ULD) that supports
student/teacher pairs with different tokenizers. It aligns the textual spans produced by both tokenizers and merges the
associated logits so no completion tokens are dropped. This enables cross-tokenizer knowledge distillation, including
mixed model families (for example, LLaMA students with Qwen teachers).

Key capabilities:

1. **Cross-tokenizer alignment** – GOLD incrementally decodes the student and teacher tokens, groups passages with the same visible text, and merges probabilities inside each group. This guarantees loss terms are computed over the full completion even when token boundaries differ.
2. **Hybrid ULD loss** – when `uld_use_hybrid_loss` is enabled, GOLD compares exact vocabulary matches directly and falls back to the original sorted-probability ULD loss for unmatched tokens. This improves stability for students whose vocabularies only partially overlap with the teacher.
3. **Seamless integration with GKD** – GOLD inherits the on-policy vs. off-policy scheduling from the [`experimental.gkd.GKDTrainer`], so you can combine sequence-level KD, generalized JSD, and cross-tokenizer distillation in a single training run.

> [!NOTE]
> GOLD is currently part of the `trl.experimental` namespace. APIs may change without notice while the feature is iterated on.

## Usage tips

The [`GOLDTrainer`] subclasses [`SFTTrainer`] and accepts the same datasets as other TRL trainers (lists of ChatML style
messages). Important configuration flags on [`GOLDConfig`] include:

* `use_external_teacher_vllm` – enables the external vLLM teacher endpoint (required for GOLD).
* `teacher_vllm_base_url`, `teacher_vllm_model_name` – how to reach the external teacher and select its model name.
* `teacher_tokenizer_name_or_path` – required when `use_uld_loss=True`; GOLD uses the teacher tokenizer to align tokens.
* `use_uld_loss`, `uld_use_hybrid_loss`, `uld_hybrid_matched_weight`, `uld_hybrid_unmatched_weight` – enables and weights ULD loss terms.
* `beta`, `lmbda` – control the matched-token divergence and the on-/off-policy mix.

A minimal end-to-end example:

```python
from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer
from transformers import AutoTokenizer

student_name = "meta-llama/Llama-3.2-1B-Instruct"
teacher_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset(
    "HuggingFaceTB/Countdown-Task-GOLD",
    "verified_Qwen2.5-0.5B-Instruct",
    split="train",
)

training_args = GOLDConfig(
    output_dir="gold-model",
    per_device_train_batch_size=1,
    teacher_tokenizer_name_or_path=teacher_name,
    use_external_teacher_vllm=True,
    teacher_vllm_base_url="http://localhost:8000/v1/completions",
    teacher_vllm_model_name=teacher_name,
    use_uld_loss=True,
    uld_use_hybrid_loss=True,
)

trainer = GOLDTrainer(
    model=student_name,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```

If you need finer-grained control over the student model instantiation, pass an already-initialized `AutoModelForCausalLM` as `model`.

### Expected dataset type

GOLD requires a [conversational](dataset_formats#conversational) [language modeling](dataset_formats#language_modeling) dataset, e.g.:

```python
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}
```

`GOLDTrainer` keeps the raw messages so the ChatML collator can construct prompts and completions with the correct
boundaries.

## GOLDTrainer

[[autodoc]] experimental.gold.GOLDTrainer
    - train
    - generate_on_policy_outputs
    - save_model
    - push_to_hub

## GOLDConfig

[[autodoc]] experimental.gold.GOLDConfig
