
# External vLLM Teacher Distillation for GOLDTrainer
## (Mistral Small Student, Qwen3-30B-3A Teacher, Full-Logprobs Mode)

## 1. Overview

### 1.1 Goal

We want to modify the existing GOLDTrainer so that:

- The **teacher model is no longer loaded as a local HF model**, but instead
- GOLDTrainer **queries an already-implemented vLLM full-logprobs “teacher mode” endpoint** to obtain teacher log-probabilities.
- **Only on-policy distillation** is used (no off-policy data).
- The loss is **purely distribution-based**, **without any CE term**, because there will be **no ground-truth labels**.
- We use **Hybrid ULD**:
  - For **vocab-matched tokens** (same token string in student and teacher), we apply **full reverse KL** \(D_{KL}(p_{student} \Vert p_{teacher})\).
  - For **vocab-unmatched tokens**, we apply a **rank-sorted L1 distance** between the probability vectors (ULD-style).

The teacher runs in vLLM “full logprobs teacher mode” (already implemented outside this scope) and is **colocated** with the training process:

- Student: Mistral Small–based model (trained by GOLDTrainer).
- Teacher: Qwen3-30B-3A served via vLLM in a separate process.
- Hardware: Assume **2x H100** initially, but the design must be compatible with scaling to **8x H100** later (by changing vLLM config, not GOLDTrainer).

### 1.2 Scope

In scope:

- Changes inside the `gold` trainer codepath so that:
  - Teacher predictions are fetched from the **existing vLLM teacher API** instead of a local teacher model.
  - The loss function is changed to:
    - Reverse KL on vocab-matched tokens.
    - Rank-sorted L1 on vocab-unmatched tokens.
    - No CE component.
  - Only on-policy trajectories are used for training.

Out of scope:

- Implementing or modifying the vLLM teacher mode itself (already done).
- Deployment and scaling details of the vLLM server (TP/PP, KV cache, etc.).
- Non–full-logprobs teacher modes.

---

## 2. High-Level Architecture

### 2.1 Models

- **Student**
  - Backbone: Mistral Small family (e.g., a Mistral Small instruct variant).
  - Location: Training node GPU(s), managed by GOLDTrainer (FSDP/DP compatible).
  - Responsibilities:
    - Generate on-policy completions for prompts.
    - Produce distributions for distillation loss.

- **Teacher**
  - Backbone: Qwen3-30B-3A.
  - Location: Separate vLLM process (colocated in the same pod/node, but not part of the training process).
  - Mode: Full-logprobs teacher mode (forward-only), returning **normalized log-probabilities** (log_softmax) for each requested token position.

### 2.2 Information Flow per Training Step

1. GOLDTrainer receives a batch of prompts from the dataset.
2. The **student** generates on-policy completions (no off-policy).
3. The concatenated `prompt + completion` token IDs are sent to the **vLLM teacher endpoint** to obtain full log-probabilities over the teacher vocab at selected positions.
4. GOLDTrainer computes:
   - Student distributions over the student vocab for the same positions.
   - Hybrid ULD loss:
     - Reverse KL for matched tokens.
     - Rank-sorted L1 for unmatched tokens.
5. The loss is backpropagated only through the student; teacher is external and fixed.

---

## 3. Loss Design

### 3.1 Notation

- Teacher distribution at time step \(t\): \( p_t(t) \in \mathbb{R}^{V_t} \)
- Student distribution at time step \(t\): \( p_s(t) \in \mathbb{R}^{V_s} \)
- Teacher vocab: \( \mathcal{V}_t \), Student vocab: \( \mathcal{V}_s \)
- Teacher tokenizer: `tok_t`, Student tokenizer: `tok_s`
- Shared vocabulary entries (string-level match):

  \[
  \mathcal{M} = \{ (i_t, i_s) \mid tok_t(i_t) = tok_s(i_s) \}
  \]

- Answer token positions (completion tokens after the prompt): \( \mathcal{T} \)

We assume:

- Teacher provides **log_softmax log-probs** for each requested position (full vocab).
- Student logits are converted to log-probs via `log_softmax`, possibly with temperature.

### 3.2 Vocabulary Mapping (Initialization)

Once, at trainer initialization:

1. Fetch `get_vocab(): Dict[str, int]` from both tokenizers.
2. Build a mapping from teacher IDs to student IDs based on exact string match:

   ```python
   mapping_teacher_to_student: Dict[int, int]
   matched_teacher_ids: Set[int]
   matched_student_ids: Set[int]
   ```

3. Define:
   - `unmatched_teacher_ids = V_t \ matched_teacher_ids`
   - `unmatched_student_ids = V_s \ matched_student_ids`

This mapping is reused throughout training.

### 3.3 Matched Part: Reverse KL

For each time step \(t \in \mathcal{T}\):

1. Extract teacher probabilities for matched teacher IDs:

   \[
   q_i(t) = p_t(t, i_t) \quad \text{for } i_t \in \mathcal{V}_t^{matched}
   \]

   where \( \mathcal{V}_t^{matched} = \{ i_t \mid i_t \in matched\_teacher\_ids \} \).

2. Map to student IDs:

   \[
   i_s = mapping\_teacher\_to\_student[i_t]
   \]

   and get the student probabilities:

   \[
   p_i(t) = p_s(t, i_s)
   \]

3. Compute reverse KL on the matched support:

   \[
   L_{\text{matched}}(t) = \sum_{i \in \mathcal{V}_t^{matched}} p_i(t)\, \log \frac{p_i(t)}{q_i(t)}.
   \]

4. Average over time steps:

   \[
   L_{\text{matched}} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} L_{\text{matched}}(t).
   \]

### 3.4 Unmatched Part: Rank-Sorted L1

For each time step \(t \in \mathcal{T}\):

1. Teacher unmatched probabilities:

   \[
   u^{(T)}_t = \{ p_t(t, i_t) \mid i_t \in \mathcal{V}_t \setminus matched\_teacher\_ids \}.
   \]

2. Student unmatched probabilities:

   \[
   u^{(S)}_t = \{ p_s(t, i_s) \mid i_s \in \mathcal{V}_s \setminus matched\_student\_ids \}.
   \]

3. Sort descending by value:

   ```python
   uT_sorted = sort(u_t_T, descending=True)
   uS_sorted = sort(u_t_S, descending=True)
   ```

4. Pad the shorter vector with zeros so that:

   \[
   \tilde{u}^{(T)}_t, \tilde{u}^{(S)}_t \in \mathbb{R}^{K}
   \]

   for a common length \(K\).

5. Compute L1 distance:

   \[
   L_{\text{unmatched}}(t) = \left\| \tilde{u}^{(S)}_t - \tilde{u}^{(T)}_t \right\|_1.
   \]

6. Average over time steps:

   \[
   L_{\text{unmatched}} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} L_{\text{unmatched}}(t).
   \]

### 3.5 No CE Term

Because there are **no labels** (no ground-truth target tokens), we **remove the CE term entirely**:

\[
L_{\text{CE}} \equiv 0.
\]

The total loss is:

\[
L_{\text{total}} = \alpha \cdot L_{\text{matched}} + (1 - \alpha) \cdot L_{\text{unmatched}},
\]

where:

- Default \(\alpha\) is derived from the matched vocab ratio:

  \[
  \alpha = \frac{|\mathcal{V}_t^{matched}|}{|\mathcal{V}_t|},
  \]

- But \(\alpha\) can be exposed as a config hyperparameter to override this heuristic.

---

## 4. Training Loop Changes

### 4.1 On-Policy Only

- Set `lmbda = 1.0` in `GOLDConfig`, so that **every training step uses on-policy data**.
- Keep the off-policy branch in the code (for compatibility), but it is effectively never taken.
  - Alternatively, guard the off-policy path to be skipped when `use_external_teacher_vllm` is enabled.

### 4.2 Data Flow Per Step

For a single training step:

1. **Batch of prompts**
   - Collator produces `input_ids`, `attention_mask`, and any metadata needed to reconstruct the textual prompts.
   - Complements (labels) from the original dataset are not used for loss.

2. **Student on-policy generation**
   - The student model generates completions for each prompt:

     ```python
     generated = student_model.generate(
       input_ids=prompt_ids,
       attention_mask=prompt_mask,
       max_new_tokens=max_new_tokens,
       temperature=temperature,
       top_p=top_p,
       do_sample=True,
       ...
     )
     ```

   - Get:
     - `generated_input_ids` (prompt + completion) with shape `[B, L_total]`.
     - Per-sample prompt lengths `prompt_lengths`.

   - Define completion token positions for each sample:

     ```python
     completion_positions[i] = range(prompt_len[i], L_total[i])
     ```

   - These positions define the set \( \mathcal{T} \).

3. **Teacher log-probabilities from vLLM**

   - For each sample (first version, simplest implementation):
     - Send the concatenated token IDs to the vLLM teacher endpoint using the already implemented teacher mode:

       ```json
       {
         "model": "qwen3-30b-3a-teacher",
         "prompt": [token_ids...],
         "max_tokens": 0,
         "stream": false,
         "n": 1,
         "full_logprobs": {
           "enabled": true,
           "positions": completion_positions_or_shifted_positions,
           "dtype": "fp16",
           "format": "base64_dense"
         }
       }
       ```

     - Decode the response’s base64 `data` into a `[L_sub, V_t]` float16 matrix, then convert to `torch.Tensor` (e.g., float32) on GPU.

   - The exact `positions` used should match the student’s target positions for next-token prediction (e.g., `pos-1` vs `pos`)—same as the original GOLD alignment.

   - Future optimization: if the vLLM endpoint supports batching of multiple prompts in a single call, we can replace the per-sample loop with micro-batching.

4. **Student distributions**

   - Run the student forward pass on the full concatenated sequence:

     ```python
     outputs = student_model(
       input_ids=generated_input_ids,
       attention_mask=attention_mask_full,
       use_cache=False,
     )
     logits = outputs.logits  # [B, L_total, V_s]
     ```

   - Select logits at the same positions used by the teacher:

     ```python
     student_logits_answer = gather_completion_logits(logits, completion_positions)
     logp_student = torch.log_softmax(
       student_logits_answer / T_student,
       dim=-1,
     )
     ```

   - `T_student` can be 1.0 initially, and later exposed as a temperature hyperparameter.

5. **Hybrid ULD loss computation**

   - Convert teacher log-probs to probabilities or work directly in log-space for reverse KL:
     - For reverse KL, we can compute:

       \[
       p_i(t) = \exp(\log p_s(t,i)), \quad q_i(t) = \exp(\log p_t(t,i)).
       \]

   - Use the precomputed vocab mapping to split matched/unmatched dimensions.
   - Compute:
     - \(L_{\text{matched}}\) via reverse KL on the matched sub-distribution.
     - \(L_{\text{unmatched}}\) via rank-sorted L1 for unmatched probabilities.
   - Combine with \(\alpha\) to obtain \(L_{\text{total}}\).

6. **Backpropagation**

   - Backward and optimizer step are unchanged; gradients only flow through the student.
   - Teacher side (vLLM) is stateless in terms of training and remains fixed.

---

## 5. Integration into GOLDTrainer

### 5.1 Configuration

Extend `GOLDConfig` with:

```python
@dataclass
class GOLDConfig(SFTConfig):
    # existing fields...

    use_external_teacher_vllm: bool = False

    # External teacher vLLM endpoint parameters
    teacher_vllm_base_url: str = "http://localhost:8000/v1/completions"
    teacher_vllm_model_name: str = "qwen3-30b-3a-teacher"
    teacher_vllm_timeout: float = 10.0
    teacher_vllm_max_retries: int = 3

    # Loss-related hyperparameters
    use_ce_loss: bool = False  # must be False for this setup
    hybrid_uld_alpha: Optional[float] = None  # if None, derive from vocab overlap
    distillation_temperature_student: float = 1.0
```

Notes:

- `use_ce_loss` should be **forced to False** when `use_external_teacher_vllm` is True and labels are not provided.
- If labels accidentally exist in the dataset, they must be ignored.

### 5.2 Initialization Changes

In `GOLDTrainer.__init__`:

- When `use_external_teacher_vllm` is True:

  - **Do not load a local teacher HF model.**
  - Initialize vocab mapping for Hybrid ULD using the student tokenizer and the (string-level) teacher tokenizer definition (which can be loaded from config or via a small helper that only initializes the tokenizer, not the model weights).
  - Initialize any helper needed to call the vLLM endpoint (e.g., a lightweight HTTP client wrapper), but the implementation of the teacher mode itself is out of scope and assumed to exist.

- When `use_external_teacher_vllm` is False:
  - Keep the existing behavior (local teacher model, existing GOLD logic).

### 5.3 Loss Function Changes

- Replace the current loss computation branch for ULD/hybrid with:
  - Reverse KL on the matched subspace.
  - Sorted L1 on the unmatched subspace.
  - No CE term when `use_external_teacher_vllm` and `use_ce_loss=False`.

- Ensure that the legacy JSD / CE path remains available behind config flags, but is disabled in this configuration.

---

## 6. Resource & Scaling Considerations

### 6.1 vLLM GPU (2 → 8 H100)

- Scaling from 2x to 8x H100 is handled entirely on the vLLM side (e.g., `--tensor-parallel-size` and other server flags).
- From GOLDTrainer’s perspective, the teacher is always accessed via the same HTTP endpoint and model name; no code changes required.

### 6.2 Host RAM and Full Logprobs

- Full logprobs have memory footprint `L_sub * V_t * 2 bytes` (fp16).
- To keep memory usage manageable:
  - Limit `positions` to only the answer tokens (no prompt tokens unless necessary).
  - Optionally further subsample positions if we later discover a need to reduce teacher traffic.

---

## 7. Error Handling & Logging

### 7.1 Teacher Request Failures

- For each vLLM request:
  - Use up to `teacher_vllm_max_retries` with exponential backoff.
  - If all retries fail:
    - Log a warning with batch ID and error details.
    - Recommended behavior:
      - Skip the backward for this batch (loss = 0, no optimizer step).
      - Increment a counter for “teacher unavailability”.
  - Long-term, we can add metrics to monitor vLLM teacher health (e.g., via Prometheus).

### 7.2 Metrics & Debug Logging

- Log the following per step or per logging interval:
  - `loss_matched_reverse_kl`
  - `loss_unmatched_l1`
  - `alpha_hybrid_weight`
  - vLLM request latency statistics (avg, p95)
  - Teacher entropy statistics (optional, for debugging)

- For debugging:
  - Optionally sample and log a few `(prompt, student completion, teacher top tokens)` tuples to WandB or logs.

---

## 8. Implementation Plan

1. **Config**
   - Add `use_external_teacher_vllm`, endpoint params, and new loss hyperparameters to `GOLDConfig`.
   - Enforce `use_ce_loss=False` in this configuration.

2. **Tokenizer & Vocab Mapping**
   - Ensure both teacher and student tokenizers are available.
   - Build and store the teacher→student vocab mapping and matched/unmatched ID sets.

3. **Teacher Access**
   - Add a thin wrapper in GOLDTrainer (or a small helper) to call the existing vLLM teacher endpoint:
     - Input: `token_ids`, `positions`.
     - Output: `log_probs_teacher` tensor `[L_sub, V_t]`.

4. **On-Policy Only Training**
   - Set `lmbda=1.0` and ensure off-policy branches are effectively disabled when using external teacher mode.

5. **Loss Implementation**
   - Implement hybrid ULD reverse-KL + rank-sorted L1 loss using the mapping and teacher/student log-probs.
   - Integrate into `compute_loss` / `training_step` replacing the old JSD+CE path for this config.

6. **Testing**
   - Unit test:
     - Vocab mapping correctness (matched vs unmatched sets).
     - Loss behavior on synthetic distributions.
   - Integration test:
     - End-to-end single batch: student generate → teacher logprobs → loss → backward.
   - Measure:
     - Basic throughput and teacher latency.
     - GPU memory usage before/after externalizing the teacher.

This completes the spec for plugging an external vLLM teacher (Qwen3-30B-3A) into GOLDTrainer with pure on-policy, label-free Hybrid ULD (reverse KL + rank-sorted L1) for a Mistral Small student.
