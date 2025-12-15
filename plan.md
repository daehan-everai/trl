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
  "full_logprobs": {
    "dtype": "fp16",
    "shape": [3, 151936],
    "data": "base64..."
  }
}
```

All values are log-softmax log-probabilities.

## Commit plan (each item = one commit)

1. Add external-teacher config flags + validation in `trl/experimental/gold/gold_config.py` (enable external teacher mode, endpoint params, force on-policy-only via `lmbda=1.0`, force distribution-only via `uld_crossentropy_weight=0.0`, require `use_uld_loss=True` and a teacher tokenizer path).

2. Add an HTTP teacher client helper (e.g. `trl/experimental/gold/teacher_client.py`) that builds the request payload, retries with backoff, and decodes `base64_dense` fp16 into a `torch.Tensor` with shape `[len(positions), vocab_size]`.

3. Add a dummy teacher server script `scripts/dummy_teacher_server.py` (FastAPI) implementing `GET /health/` and `POST /v1/completions`, returning random fp16 `log_softmax` matrices encoded as `base64_dense` (configurable vocab size/seed/host/port).

4. Fix ULD alignment to match next-token prediction (use `pos-1` logits vs answer tokens) so both local-teacher and external-teacher modes agree on which timestep distributions correspond to which completion tokens.

5. Implement spec Hybrid ULD in `ULDLoss` for hybrid mode: matched tokens use reverse KL `D_KL(p_student || p_teacher)` on string-matched vocab IDs; unmatched tokens use rank-sorted L1; weights remain adaptive by default (or respect existing fixed weights if set).

6. Gate teacher initialization in `trl/experimental/gold/gold_trainer.py`: when external teacher mode is enabled, do not load/prepare `self.teacher_model`; still initialize `teacher_tokenizer` and the new teacher client.

7. Implement the external teacher path in `GOLDTrainer.compute_loss`: build teacher token IDs from `original_prompt_text`/`original_completion_text`, compute completion positions with correct shift, fetch teacher logprobs for only those positions, and compute the Hybrid ULD loss without materializing a full `[seq_len, vocab]` teacher tensor.

8. Add focused tests: unit tests for reverse KL matched term + unmatched sorted L1 behavior on tiny tensors, and a lightweight integration test that hits the dummy teacher server and runs the external-teacher `compute_loss` path end-to-end (finite scalar loss, backprop-safe, no local teacher model load).
