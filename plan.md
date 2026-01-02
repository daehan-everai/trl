# Plan: External vLLM Teacher Distillation for `GOLDTrainer` (Dummy Teacher First)

This follows `techspec.md` but starts with a dummy HTTP “teacher” that returns random full-vocab logprobs, so we can
isolate trainer/loss integration work from real vLLM deployment details.

## Dummy teacher endpoint contract (stable for early integration)

Request (minimal):
```json
{
  "model": "qwen3-30b-3a-teacher",
  "prompt": [1, 2, 3],
  "max_tokens": 0,
  "stream": false,
  "n": 1,
  "full_logprobs": {
    "enabled": true,
    "positions": [0, 1, 2],
    "dtype": "fp16",
    "format": "base64_dense"
  }
}
```

Response (minimal):
```json
{
  "choices": [
    {
      "index": 0,
      "text": "",
      "full_logprobs": {
        "dtype": "fp16",
        "format": "base64_dense",
        "positions": [0, 1, 2],
        "logprobs": "base64..."
      }
    }
  ]
}
```

All values are log-softmax log-probabilities.

## Commit plan (each item = one commit)

1. [x] Add external-teacher config flags + validation in `trl/experimental/gold/gold_config.py` (enable external teacher mode, endpoint params, force on-policy-only via `lmbda=1.0`, force distribution-only via `uld_crossentropy_weight=0.0`, require `use_uld_loss=True` and a teacher tokenizer path). (commit: cc990ef8)

2. [x] Add an HTTP teacher client helper (e.g. `trl/experimental/gold/teacher_client.py`) that builds the request payload, retries with backoff, and decodes `base64_dense` fp16 into a `torch.Tensor` with shape `[len(positions), vocab_size]`. (commit: be2ae8b1)

3. [x] Add a dummy teacher server script `scripts/dummy_teacher_server.py` (FastAPI) implementing `GET /health/` and `POST /v1/completions`, returning random fp16 `log_softmax` matrices encoded as `base64_dense` (configurable vocab size/seed/host/port). (commit: 8cac9ad3)

4. [x] Fix ULD alignment to match next-token prediction (use `pos-1` logits vs answer tokens) so both local-teacher and external-teacher modes agree on which timestep distributions correspond to which completion tokens. (commit: d740be8b)

5. [x] Implement spec Hybrid ULD in `ULDLoss` for hybrid mode: matched tokens use reverse KL `D_KL(p_student || p_teacher)` on string-matched vocab IDs; unmatched tokens use rank-sorted L1; weights remain adaptive by default (or respect existing fixed weights if set). (commit: 4c19f24f)

6. [x] Gate teacher initialization in `trl/experimental/gold/gold_trainer.py`: when external teacher mode is enabled, do not load/prepare `self.teacher_model`; still initialize `teacher_tokenizer` and the new teacher client. (commit: c0967de8)

7. [x] Implement the external teacher path in `GOLDTrainer.compute_loss`: build teacher token IDs from `original_prompt_text`/`original_completion_text`, compute completion positions with correct shift, fetch teacher logprobs for only those positions, and compute the Hybrid ULD loss without materializing a full `[seq_len, vocab]` teacher tensor. (commit: e7e818d0)

8. [x] Add focused tests: unit tests for reverse KL matched term + unmatched sorted L1 behavior on tiny tensors, and a lightweight integration test that hits the dummy teacher server and runs the external-teacher `compute_loss` path end-to-end (finite scalar loss, backprop-safe, no local teacher model load). (commit: fd95d08f)

## Example Training (Dummy Teacher + MagistSmall)

9. [x] Allow running from source checkout by tolerating missing `trl` package metadata when generating a model card. (commit: 8f3f41a8)

10. [x] Add a runnable example that starts the dummy teacher server, loads `EverAI-AI/MagistSmall-Raven-DPO6-c`, and runs a short GOLD on-policy external-teacher training with WandB. (commit: 023f4e47)

11. [x] Update the external-teacher client + dummy server/tests to parse OpenAI-style vLLM `choices[0].full_logprobs.logprobs` responses. (commit: a60b145e)

12. [x] Update `techspec.md` to document the OpenAI-style `choices[0].full_logprobs.logprobs` response format. (commit: 3c28e98c)
