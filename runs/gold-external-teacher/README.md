---
base_model: EverAI-AI/MagistSmall-Raven-ALT-2
library_name: transformers
model_name: gold-external-teacher
tags:
- generated_from_trainer
- gold
- unsloth
- trl
licence: license
---

# Model Card for gold-external-teacher

This model is a fine-tuned version of [EverAI-AI/MagistSmall-Raven-ALT-2](https://huggingface.co/EverAI-AI/MagistSmall-Raven-ALT-2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/lucas01/trl-gold-external-teacher/runs/fywxlj04) 

### Status (2026-01-11)

- Active run log: `runs/gold-external-teacher/full_epoch_patch_v7.log` (W&B run `valiant-thunder-34`, id `fywxlj04`).
- Teacher sparse logprobs: `top_p=0.9999`, `max_top_k=2048`.
- Student rollout sampling: `temperature=0.7`, `top_p=0.9`, `top_k=50`.
- Unmatched loss: disabled (`--disable-unmatched-loss`) for matched-only distillation.

### Key changes (so far)

- Fixed prompt_end handling for left-padded prompts and improved stop trimming to avoid ChatML tag leakage.
- Always append ChatML generation suffix if truncation dropped it.
- Stop sequences include space-prefixed variants and both stop criteria run during generation.
- Hybrid ULD unmatched loss now ignores sparse teacher tail (zero-prob) instead of forcing student to zero.
- Added CLI args for student rollout sampling: `--student-temperature`, `--student-top-p`, `--student-top-k`.

### Trial and error notes

- `runs/gold-external-teacher/full_epoch_patch_v3.log`: degeneration (repetitive "Avec plaisir ..." babble) around steps 37-41; markup leak at step 18 (raw:: html / IM_START markers).
- `runs/gold-external-teacher/full_epoch_patch_v4.log` and `runs/gold-external-teacher/full_epoch_patch_v5.log`: aborted due to missing CLI args for student sampling.
- `runs/gold-external-teacher/full_epoch_patch_v6.log`: unmatched loss masked sparse tail, but late-step degeneration persisted (around steps 36-40) with repetitive "Avec ..." output.
- `runs/gold-external-teacher/full_epoch_patch_v7.log`: unmatched loss disabled; mid-epoch looked healthy (steps ~19-23), but late steps (87-91) collapsed to repeated "avec" token.
- `runs/gold-external-teacher/debug_patch_run_len2048_mc256_steps2_v3.log`: no ChatML tag leakage in completions after stop/prompt fixes.
- `runs/gold-external-teacher/metrics_plot.png`: partial metrics plot from earlier run.


This model was trained with GOLD.

### Framework versions

- TRL: 0.27.0.dev0
- Transformers: 4.57.3
- Pytorch: 2.4.1+cu124
- Datasets: 4.4.2
- Tokenizers: 0.22.2

## Citations

Cite GOLD as:

```bibtex
@misc{patino2025unlocking,
    title        = {{Unlocking On-Policy Distillation for Any Model Family}},
    author       = {Carlos Miguel Patiño and Kashif Rasul and Quentin Gallouédec and Ben Burtenshaw and Sergio Paniego and Vaibhav Srivastav and Thibaud Frere and Ed Beeching and Lewis Tunstall and Leandro von Werra and Thomas Wolf},
    year         = 2025,
    url          = {https://huggingface.co/spaces/HuggingFaceH4/general-on-policy-logit-distillation},
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```
