---
name: lessons-teacher-endpoint
description: "Teacher endpoint behaviors and constraints for GOLD external-teacher runs."
---

# Teacher Endpoint Lessons

- Use `/v1/completions` for teacher-mode `full_logprobs` with **token-ID prompts**.
- Avoid `/v1/chat/completions` for teacher mode: it expects messages, applies chat templates, and breaks token-ID position alignment.
- If you see `Missing logprobs for required position ...`, check vLLM **prefix caching**:
  - When prefix caching is enabled, vLLM can reuse cached KV and **skip computing logits** for cached positions.
  - That causes missing `full_logprobs` rows and a 400 error.
  - Fix: disable prefix caching on the teacher server (`enable_prefix_caching=False`).
- Ensure the correct teacher model name is configured (e.g., `deepseek-ai/DeepSeek-V3.1`).
