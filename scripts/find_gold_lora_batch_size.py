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
import subprocess
import sys
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    if "HUGGINGFACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def is_oom(output: str) -> bool:
    markers = [
        "CUDA out of memory",
        "CUDA error: out of memory",
        "cublas",
        "CUDNN",
        "device-side assert triggered",
    ]
    lowered = output.lower()
    return any(marker.lower() in lowered for marker in markers)


def run_trial(args: argparse.Namespace, batch_size: int) -> tuple[bool, float | None, float | None, str]:
    output_dir = f"{args.output_root}/bs{batch_size}"
    cmd = [
        sys.executable,
        "scripts/run_gold_external_teacher.py",
        "--model-id",
        args.model_id,
        "--dataset-id",
        args.dataset_id,
        "--dataset-split",
        args.dataset_split,
        "--teacher-tokenizer",
        args.teacher_tokenizer,
        "--output-dir",
        output_dir,
        "--max-steps",
        "1",
        "--per-device-train-batch-size",
        str(batch_size),
        "--max-completion-length",
        str(args.max_completion_length),
        "--max-length",
        str(args.max_length),
        "--learning-rate",
        str(args.learning_rate),
        "--teacher-port",
        "0",
        "--use-lora",
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--log-cuda-memory",
    ]
    if args.lora_target_modules:
        cmd.extend(["--lora-target-modules", args.lora_target_modules])
    if args.report_to_none:
        cmd.extend(["--report-to-none"])

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return False, None, None, stdout

    max_alloc = None
    max_reserved = None
    for line in stdout.splitlines():
        if line.startswith("max_cuda_allocated_gb="):
            try:
                max_alloc = float(line.split("=", 1)[1])
            except ValueError:
                pass
        if line.startswith("max_cuda_reserved_gb="):
            try:
                max_reserved = float(line.split("=", 1)[1])
            except ValueError:
                pass
    return True, max_alloc, max_reserved, stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Find max per-device batch size for GOLD+LoRA on a GPU.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dataset-id", default="EverAI-AI/french-conversations-prompt")
    parser.add_argument("--dataset-split", default="train[:32]")
    parser.add_argument("--teacher-tokenizer", default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default=None)
    parser.add_argument("--report-to-none", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--output-root", default="runs/gold-lora-batch-sweep")
    args = parser.parse_args()

    load_dotenv()
    if args.model_id is None:
        args.model_id = os.environ.get("STUDENT_MODEL_ID")
    if args.model_id is None:
        raise RuntimeError("Missing --model-id or STUDENT_MODEL_ID in .env.")
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("Missing WANDB_API_KEY. Add it to `.env` or export it in your shell.")
    if "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        raise RuntimeError("Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN. Add it to `.env` or export it in your shell.")

    os.makedirs(args.output_root, exist_ok=True)

    batch = 1
    last_success = 0
    last_alloc = None
    last_reserved = None

    while batch <= args.max_batch_size:
        ok, alloc, reserved, output = run_trial(args, batch)
        if ok:
            last_success = batch
            last_alloc = alloc
            last_reserved = reserved
            batch *= 2
            continue
        if is_oom(output):
            break
        raise RuntimeError(f"Trial with batch_size={batch} failed unexpectedly. Output:\n{output}")

    if last_success == 0:
        raise RuntimeError("Even batch_size=1 failed. Try reducing max_length or completion length.")

    low = last_success + 1
    high = min(args.max_batch_size, batch - 1)
    while low <= high:
        mid = (low + high) // 2
        ok, alloc, reserved, output = run_trial(args, mid)
        if ok:
            last_success = mid
            last_alloc = alloc
            last_reserved = reserved
            low = mid + 1
        elif is_oom(output):
            high = mid - 1
        else:
            raise RuntimeError(f"Trial with batch_size={mid} failed unexpectedly. Output:\n{output}")

    print(f"max_batch_size={last_success}")
    if last_alloc is not None:
        print(f"max_cuda_allocated_gb={last_alloc:.2f}")
    if last_reserved is not None:
        print(f"max_cuda_reserved_gb={last_reserved:.2f}")


if __name__ == "__main__":
    main()
