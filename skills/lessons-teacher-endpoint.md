---
name: lessons-teacher-endpoint
description: "Teacher endpoint behaviors and constraints for GOLD external-teacher runs."
---

# Teacher Endpoint Lessons

- Use `/v1/completions` for teacher-mode `full_logprobs` with **token-ID prompts**.
- Avoid `/v1/chat/completions` for teacher mode: it expects messages, applies chat templates, and breaks token-ID position alignment.
- When sending token IDs, keep prompts as a list of ints and request `full_logprobs` (the trainer expects per-position logprobs).
- If you see `Missing logprobs for required position ...`, check vLLM **prefix caching**:
  - When prefix caching is enabled, vLLM can reuse cached KV and **skip computing logits** for cached positions.
  - That causes missing `full_logprobs` rows and a 400 error.
  - Fix: disable prefix caching on the teacher server (`enable_prefix_caching=False`).
- Keep teacher sampling aligned with the trainer (used here): `format=top_p`, `top_p=0.9999`, `max_top_k=2048`.
- Ensure the correct teacher model name is configured (e.g., `deepseek-ai/DeepSeek-V3.1`) and the URL is current.
