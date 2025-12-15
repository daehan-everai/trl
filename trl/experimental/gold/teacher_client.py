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

    raw = base64.b64decode(data)
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


class VLLMTeacherClient:
    """
    Minimal HTTP client for the vLLM "full_logprobs teacher mode" endpoint.

    The endpoint is expected to accept an OpenAI-compatible `/v1/completions` request with a `full_logprobs` field and
    return a `base64_dense` fp16 matrix of log-softmax log-probabilities for each requested position.
    """

    def __init__(self, config: VLLMTeacherClientConfig):
        if not config.base_url:
            raise ValueError("`base_url` must be non-empty.")
        if config.timeout <= 0:
            raise ValueError("`timeout` must be > 0.")
        if config.max_retries < 0:
            raise ValueError("`max_retries` must be >= 0.")
        object.__setattr__(config, "base_url", config.base_url.rstrip("/"))
        self.config = config

    def _request_payload(self, token_ids: list[int], positions: list[int]) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "prompt": token_ids,
            "max_tokens": 0,
            "stream": False,
            "n": 1,
            "full_logprobs": {
                "enabled": True,
                "positions": positions,
                "dtype": "fp16",
                "format": "base64_dense",
            },
        }

    def fetch_full_logprobs(
        self,
        token_ids: list[int],
        positions: list[int],
        *,
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
                full_logprobs = response["full_logprobs"]
                if full_logprobs.get("dtype") != "fp16":
                    raise ValueError(f"Unsupported full_logprobs dtype: {full_logprobs.get('dtype')}")
                shape = tuple(full_logprobs["shape"])
                tensor = decode_base64_dense_fp16(full_logprobs["data"], shape=shape)  # CPU fp16
                tensor = tensor.to(device=device, dtype=dtype) if device is not None else tensor.to(dtype=dtype)
                return tensor
            except Exception as exc:
                last_exc = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(0.25 * (2**attempt))

        raise RuntimeError("Teacher endpoint request failed after retries.") from last_exc

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if is_requests_available():
            r = requests.post(self.config.base_url, json=payload, timeout=self.config.timeout)
            r.raise_for_status()
            return r.json()

        data = json.dumps(payload).encode("utf-8")
        req = Request(self.config.base_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Teacher endpoint request failed: {exc}") from exc
