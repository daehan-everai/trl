# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trl.experimental.gold.gold_trainer import GOLDTrainer, ULDLoss
from trl.experimental.gold.teacher_client import VLLMTeacherClient, VLLMTeacherClientConfig


class ToyTokenizer:
    def __init__(self, vocab: dict[str, int], *, pad_token: str = "<pad>", eos_token: str = "<eos>"):
        self._vocab = dict(vocab)
        self.pad_token_id = self._vocab.get(pad_token)
        self.eos_token_id = self._vocab.get(eos_token)

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def __call__(self, texts: list[str], *, add_special_tokens: bool = True, **_: Any) -> dict[str, list[list[int]]]:
        input_ids: list[list[int]] = []
        for text in texts:
            ids = [self._vocab[ch] for ch in text if ch in self._vocab]
            if add_special_tokens and self.eos_token_id is not None:
                ids.append(self.eos_token_id)
            input_ids.append(ids)
        return {"input_ids": input_ids}


class DummyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


class _DummyTeacherServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, *, vocab_size: int, seed: int):
        super().__init__(server_address, handler_cls)
        self.vocab_size = vocab_size
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))


class _DummyTeacherHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence server logs during tests.
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/completions":
            self._send_json({"error": "not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        positions = (payload.get("full_logprobs") or {}).get("positions") or []
        rows = len(positions)
        vocab_size = int(self.server.vocab_size)  # type: ignore[attr-defined]
        generator = self.server.generator  # type: ignore[attr-defined]

        logits = torch.randn(rows, vocab_size, generator=generator, dtype=torch.float32)
        logprobs = torch.log_softmax(logits, dim=-1).to(torch.float16)
        data_b64 = base64.b64encode(logprobs.numpy().tobytes()).decode("utf-8")

        self._send_json(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "full_logprobs": {
                            "dtype": "fp16",
                            "format": "base64_dense",
                            "positions": positions,
                            "logprobs": data_b64,
                        },
                    }
                ]
            }
        )


@pytest.fixture()
def dummy_teacher_url():
    server = _DummyTeacherServer(("127.0.0.1", 0), _DummyTeacherHandler, vocab_size=4, seed=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1/completions"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _config(**overrides):
    base = dict(
        uld_crossentropy_weight=0.0,
        uld_distillation_weight=1.0,
        uld_student_temperature=1.0,
        uld_teacher_temperature=1.0,
        uld_skip_student_eos=True,
        uld_skip_teacher_eos=True,
        use_extended_uld=False,
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=None,
        uld_hybrid_unmatched_weight=None,
        beta=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_hybrid_uld_reverse_kl_and_sorted_l1():
    teacher_vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    student_vocab = {"b": 0, "d": 1, "x": 2}
    teacher_tok = ToyTokenizer(teacher_vocab)
    student_tok = ToyTokenizer(student_vocab)

    loss_fn = ULDLoss(
        _config(
            uld_matched_divergence="skew_kl",
            uld_matched_forward_kl_weight=0.0,
            uld_matched_reverse_kl_weight=1.0,
        ),
        student_tokenizer=student_tok,
        teacher_tokenizer=teacher_tok,
    )

    student_aligned = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]], dtype=torch.float32)
    teacher_aligned = torch.tensor(
        [[0.1, 0.25, 0.05, 0.6], [0.2, 0.05, 0.05, 0.7]],
        dtype=torch.float32,
    )

    got = loss_fn._compute_hybrid_uld_loss(student_aligned, teacher_aligned)

    # Matched: tokens "b" and "d" (teacher ids 1 and 3) map to student ids 0 and 1.
    p = student_aligned[:, torch.tensor([0, 1])]
    q = teacher_aligned[:, torch.tensor([1, 3])]
    matched = (p * (p.log() - q.log())).sum() / 2

    # Unmatched: teacher tokens (a,c), student token (x).
    t_unmatched = teacher_aligned[:, torch.tensor([0, 2])].sort(dim=-1, descending=True).values
    s_unmatched = student_aligned[:, torch.tensor([2])].sort(dim=-1, descending=True).values
    s_unmatched = torch.nn.functional.pad(s_unmatched, (0, t_unmatched.size(-1) - s_unmatched.size(-1)))
    unmatched = torch.nn.functional.l1_loss(s_unmatched, t_unmatched, reduction="sum") / 2

    alpha = 2 / 4  # matched vocab ratio (teacher side)
    expected = alpha * matched + (1 - alpha) * unmatched

    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_external_teacher_compute_loss_end_to_end(dummy_teacher_url):
    vocab = {"<pad>": 0, "a": 1, "b": 2, "<eos>": 3}
    student_tok = ToyTokenizer(vocab)
    teacher_tok = ToyTokenizer(vocab)

    model = DummyCausalLM(vocab_size=len(vocab))

    teacher_client = VLLMTeacherClient(
        VLLMTeacherClientConfig(base_url=dummy_teacher_url, model_name="dummy", timeout=5.0, max_retries=0)
    )
    loss_fn = ULDLoss(_config(), student_tokenizer=student_tok, teacher_tokenizer=teacher_tok)

    trainer_like = SimpleNamespace(
        use_external_teacher_vllm=True,
        teacher_tokenizer=teacher_tok,
        teacher_client=teacher_client,
        uld_loss_fn=loss_fn,
        processing_class=student_tok,
        args=SimpleNamespace(gradient_accumulation_steps=1),
        _matched_sum=0.0,
        _unmatched_sum=0.0,
        _matched_step_eq=0.0,
        _unmatched_step_eq=0.0,
    )

    input_ids = torch.tensor([[vocab["a"], vocab["b"], vocab["<eos>"]]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, vocab["b"], vocab["<eos>"]]], dtype=torch.long)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "original_prompt_text": ["a"],
        "original_completion_text": ["b"],
    }

    loss = GOLDTrainer.compute_loss(trainer_like, model, inputs)  # type: ignore[arg-type]

    assert loss.requires_grad
    assert torch.isfinite(loss).item()
    loss.backward()
