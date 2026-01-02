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

import argparse
import base64
from typing import Any

import torch


def _require_fastapi() -> tuple[Any, Any, Any]:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError(
            "Dummy teacher server requires `fastapi`. Install with `pip install trl[vllm]` (or install fastapi/uvicorn)."
        ) from exc
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Dummy teacher server requires `uvicorn`. Install with `pip install trl[vllm]` (or install uvicorn)."
        ) from exc
    return FastAPI, HTTPException, uvicorn


def _encode_base64_dense_fp16(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
    return base64.b64encode(tensor.numpy().tobytes()).decode("utf-8")


def build_app(*, vocab_size: int, seed: int | None = None):
    FastAPI, HTTPException, _ = _require_fastapi()
    app = FastAPI()

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    @app.get("/health/")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        full_logprobs = payload.get("full_logprobs") or {}
        if not full_logprobs.get("enabled", False):
            raise HTTPException(status_code=400, detail="`full_logprobs.enabled` must be true")

        positions = full_logprobs.get("positions")
        if not isinstance(positions, list) or not all(isinstance(p, int) for p in positions):
            raise HTTPException(status_code=400, detail="`full_logprobs.positions` must be a list[int]")

        if full_logprobs.get("format") not in (None, "base64_dense"):
            raise HTTPException(status_code=400, detail="Only `full_logprobs.format=base64_dense` is supported")
        if full_logprobs.get("dtype") not in (None, "fp16"):
            raise HTTPException(status_code=400, detail="Only `full_logprobs.dtype=fp16` is supported")

        rows = len(positions)
        logits = torch.randn(rows, vocab_size, generator=generator, dtype=torch.float32)
        logprobs = torch.log_softmax(logits, dim=-1).to(dtype=torch.float16)

        return {
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "full_logprobs": {
                        "dtype": "fp16",
                        "format": "base64_dense",
                        "positions": positions,
                        "logprobs": _encode_base64_dense_fp16(logprobs),
                    },
                }
            ]
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy vLLM teacher-mode server (full_logprobs, base64_dense fp16).")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--vocab-size", type=int, default=151936, help="Teacher vocab size to simulate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible responses.")
    args = parser.parse_args()

    _, _, uvicorn = _require_fastapi()
    app = build_app(vocab_size=args.vocab_size, seed=args.seed)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
