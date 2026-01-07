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
import math
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from ...import_utils import is_requests_available


if is_requests_available():
    import requests


def decode_base64_dense_fp16(data: str, shape: tuple[int, int]) -> torch.Tensor:
    """
    Decode a base64-encoded dense fp16 matrix into a CPU tensor of shape `shape`.

    The binary payload is expected to be raw row-major fp16 values (2 bytes per element).
    """
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D shape, got {shape}.")
    rows, cols = shape
    expected_nbytes = int(rows) * int(cols) * 2

    raw = bytearray(base64.b64decode(data))
    if len(raw) != expected_nbytes:
        raise ValueError(f"Decoded byte length {len(raw)} does not match expected {expected_nbytes} for shape={shape}.")

    # `torch.frombuffer` is only available in newer PyTorch versions; keep a fallback.
    try:
        tensor = torch.frombuffer(raw, dtype=torch.float16).clone()
    except Exception:
        import numpy as np

        tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy())

    return tensor.view(rows, cols)


@dataclass(frozen=True)
class VLLMTeacherClientConfig:
    base_url: str
    model_name: str
    timeout: float = 10.0
    max_retries: int = 3
    full_logprobs_format: str = "top_p"
    full_logprobs_top_p: float | None = 0.9999
    full_logprobs_max_top_k: int | None = 512


class VLLMTeacherClient:
    """
    Minimal HTTP client for the vLLM "full_logprobs teacher mode" endpoint.

    The endpoint is expected to accept an OpenAI-compatible `/v1/completions` request with a `full_logprobs` field and
    return a `base64_dense` fp16 matrix of log-softmax log-probabilities for each requested position. The response can
    either be a top-level `full_logprobs` payload or an OpenAI-style `choices[0].full_logprobs` payload.
    """

    def __init__(self, config: VLLMTeacherClientConfig):
        if not config.base_url:
            raise ValueError("`base_url` must be non-empty.")
        if config.timeout <= 0:
            raise ValueError("`timeout` must be > 0.")
        if config.max_retries < 0:
            raise ValueError("`max_retries` must be >= 0.")
        if config.full_logprobs_format not in {"base64_dense", "top_p"}:
            raise ValueError("`full_logprobs_format` must be 'base64_dense' or 'top_p'.")
        if config.full_logprobs_format == "top_p" and config.full_logprobs_top_p is not None:
            if not (0.0 < config.full_logprobs_top_p <= 1.0):
                raise ValueError("`full_logprobs_top_p` must be within (0, 1].")
        if config.full_logprobs_format == "top_p" and config.full_logprobs_max_top_k is not None:
            if config.full_logprobs_max_top_k <= 0:
                raise ValueError("`full_logprobs_max_top_k` must be > 0.")
        object.__setattr__(config, "base_url", config.base_url.rstrip("/"))
        self.config = config

    def _request_payload(self, token_ids: list[int], positions: list[int]) -> dict[str, Any]:
        full_logprobs = {
            "enabled": True,
            "positions": positions,
            "dtype": "fp16",
            "format": self.config.full_logprobs_format,
        }
        if self.config.full_logprobs_format == "top_p":
            if self.config.full_logprobs_top_p is not None:
                full_logprobs["top_p"] = self.config.full_logprobs_top_p
            if self.config.full_logprobs_max_top_k is not None:
                full_logprobs["max_top_k"] = self.config.full_logprobs_max_top_k
        return {
            "model": self.config.model_name,
            "prompt": token_ids,
            "max_tokens": 0,
            "stream": False,
            "n": 1,
            "full_logprobs": full_logprobs,
        }

    def fetch_full_logprobs(
        self,
        token_ids: list[int],
        positions: list[int],
        *,
        vocab_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Fetch full-vocab teacher log-probabilities for the requested positions.

        Returns:
            Tensor with shape `[len(positions), vocab_size]`.
        """
        payload = self._request_payload(token_ids, positions)

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._post_json(payload)
                full_logprobs = self._extract_full_logprobs(response)
                dtype_tag = full_logprobs.get("dtype", "fp16")
                if dtype_tag != "fp16":
                    raise ValueError(f"Unsupported full_logprobs dtype: {dtype_tag}")

                format_tag = full_logprobs.get("format", "base64_dense")
                if format_tag == "base64_dense":
                    data_b64 = full_logprobs.get("data") or full_logprobs.get("logprobs")
                    if data_b64 is None:
                        raise ValueError("Missing full_logprobs base64 payload (expected `data` or `logprobs`).")

                    shape = full_logprobs.get("shape")
                    if shape is None:
                        if vocab_size is None:
                            raise ValueError(
                                "`vocab_size` must be provided when `full_logprobs.shape` is missing from the response."
                            )
                        shape = (len(positions), int(vocab_size))
                    shape = tuple(shape)

                    tensor = decode_base64_dense_fp16(data_b64, shape=shape)  # CPU fp16
                    tensor = tensor.to(device=device, dtype=dtype) if device is not None else tensor.to(dtype=dtype)
                    return tensor

                if format_tag == "top_p":
                    if vocab_size is None:
                        raise ValueError("`vocab_size` must be provided when using top_p full_logprobs format.")
                    dense = self._dense_from_top_p(full_logprobs, vocab_size=vocab_size, dtype=dtype)
                    dense = dense.to(device=device) if device is not None else dense
                    return dense

                raise ValueError(f"Unsupported full_logprobs format: {format_tag}")
            except Exception as exc:
                pos_min = min(positions) if positions else None
                pos_max = max(positions) if positions else None
                context = (
                    f"prompt_len={len(token_ids)} positions_len={len(positions)} "
                    f"positions_min={pos_min} positions_max={pos_max}"
                )
                last_exc = RuntimeError(f"{exc} ({context})")
                if attempt >= self.config.max_retries:
                    break
                time.sleep(0.25 * (2**attempt))

        raise RuntimeError("Teacher endpoint request failed after retries.") from last_exc

    def _dense_from_top_p(
        self,
        payload: dict[str, Any],
        *,
        vocab_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        token_ids = payload.get("token_ids")
        logprobs = payload.get("logprobs")
        positions = payload.get("positions")
        if not isinstance(token_ids, list) or not isinstance(logprobs, list) or not isinstance(positions, list):
            raise ValueError("Invalid top_p payload: expected token_ids/logprobs/positions lists.")
        if len(token_ids) != len(logprobs) or len(token_ids) != len(positions):
            raise ValueError("top_p payload length mismatch between token_ids/logprobs/positions.")

        rows = []
        for ids, lps in zip(token_ids, logprobs):
            row = torch.full((int(vocab_size),), float("-inf"), dtype=dtype)
            for token_id, logp in zip(ids, lps):
                token_id = int(token_id)
                if 0 <= token_id < vocab_size:
                    row[token_id] = float(logp)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _extract_full_logprobs(self, response: dict[str, Any]) -> dict[str, Any]:
        if "full_logprobs" in response:
            return response["full_logprobs"]

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict) and "full_logprobs" in choice:
                return choice["full_logprobs"]

        raise ValueError("Teacher response missing `full_logprobs` payload.")

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if is_requests_available():
            r = requests.post(self.config.base_url, json=payload, timeout=self.config.timeout)
            if not r.ok:
                snippet = r.text.strip().replace("\n", " ")
                if len(snippet) > 800:
                    snippet = snippet[:800] + "..."
                raise RuntimeError(f"Teacher endpoint error {r.status_code}: {snippet}")
            return r.json()

        data = json.dumps(payload).encode("utf-8")
        req = Request(self.config.base_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Teacher endpoint request failed: {exc}") from exc
