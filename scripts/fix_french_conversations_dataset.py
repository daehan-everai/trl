#!/usr/bin/env python
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
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv


def load_env() -> str:
    load_dotenv(".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN in .env or environment.")
    return token


def fix_example(example: dict[str, Any]) -> dict[str, Any]:
    messages = list(example.get("messages") or [])
    while messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    example["messages"] = messages
    example["message_count"] = len(messages)
    return example


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix French conversations dataset for GOLD training.")
    parser.add_argument("--repo-id", default="EverAI-AI/french-conversations-prompt")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    token = load_env()
    dataset = load_dataset(args.repo_id, split=args.split, token=token)
    dataset = dataset.map(fix_example)
    if "prompt" in dataset.column_names:
        dataset = dataset.remove_columns(["prompt"])
    dataset.push_to_hub(args.repo_id, token=token)


if __name__ == "__main__":
    main()
