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
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import torch
from datasets import Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer


DEFAULT_MODEL_ID = "EverAI-AI/MagistSmall-Raven-DPO6-c"


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

    # Transformers/huggingface_hub use HUGGINGFACE_HUB_TOKEN; accept HF_TOKEN as a .env-friendly alias.
    if "HUGGINGFACE_HUB_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def wait_for_teacher(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except (URLError, Exception) as exc:
            last_err = exc
            time.sleep(0.25)
    raise RuntimeError(f"Teacher did not become healthy at {url!r} within {timeout_s}s.") from last_err


def start_dummy_teacher(*, port: int, vocab_size: int, seed: int = 0) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "scripts/dummy_teacher_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--vocab-size",
        str(vocab_size),
        "--seed",
        str(seed),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def build_tiny_chat_dataset() -> Dataset:
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Write one sentence about ravens."},
                {"role": "assistant", "content": "Ravens are intelligent birds known for problem-solving."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Give three synonyms for 'small'."},
                {"role": "assistant", "content": "Tiny, little, miniature."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Answer with a single word: What color is the sky on a clear day?"},
                {"role": "assistant", "content": "Blue."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Translate to French: 'Good morning'."},
                {"role": "assistant", "content": "Bonjour."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "List two prime numbers between 10 and 20."},
                {"role": "assistant", "content": "11 and 13."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a haiku about a quiet library."},
                {"role": "assistant", "content": "Dusty pages breathe / whispers drift through aisles of light / silence turns the key."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 7 * 8?"},
                {"role": "assistant", "content": "56"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain in one sentence what distillation is in ML."},
                {"role": "assistant", "content": "Distillation trains a smaller model to match a larger model's output distribution."},
            ]
        },
    ]
    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GOLD external-teacher training against a local dummy teacher.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="runs/gold-external-dummy-teacher")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--teacher-port", type=int, default=8000)
    parser.add_argument("--wandb-project", default="trl-gold-dummy-teacher")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune; may OOM).")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv()
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("Missing WANDB_API_KEY. Add it to `.env` or export it in your shell.")
    if "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        raise RuntimeError("Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN. Add it to `.env` or export it in your shell.")

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name is not None:
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_url = f"http://127.0.0.1:{args.teacher_port}/v1/completions"
    teacher_health = f"http://127.0.0.1:{args.teacher_port}/health/"
    teacher_proc = start_dummy_teacher(port=args.teacher_port, vocab_size=len(tokenizer), seed=0)
    try:
        wait_for_teacher(teacher_health, timeout_s=60.0)

        dataset = build_tiny_chat_dataset()

        model_init_kwargs = {"device_map": None, "trust_remote_code": True}
        if torch.cuda.is_available():
            model_init_kwargs["torch_dtype"] = torch.bfloat16

        train_args = GOLDConfig(
            output_dir=args.output_dir,
            report_to=["wandb"],
            logging_steps=1,
            save_steps=max(1, args.max_steps),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=1e-6,
            lr_scheduler_type="constant",
            warmup_steps=0,
            eval_strategy="no",
            max_completion_length=32,
            use_external_teacher_vllm=True,
            teacher_vllm_base_url=teacher_url,
            teacher_vllm_model_name="dummy-teacher",
            teacher_vllm_timeout=30.0,
            teacher_vllm_max_retries=2,
            teacher_tokenizer_name_or_path=args.model_id,
        )
        train_args.model_init_kwargs = model_init_kwargs

        peft_config = None
        use_lora = not args.no_lora
        if use_lora:
            try:
                from peft import LoraConfig, TaskType
            except ImportError as exc:
                raise ImportError("LoRA requested but `peft` is not installed. Install with `pip install peft`.") from exc
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            train_args.gradient_checkpointing = True
            if torch.cuda.is_available():
                train_args.bf16 = True

        trainer = GOLDTrainer(
            model=args.model_id,
            args=train_args,
            train_dataset=dataset,
            eval_dataset=None,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
    finally:
        stop_process(teacher_proc)

        # If the server printed errors, surface them to help debugging (without leaking env vars).
        try:
            if teacher_proc.stdout is not None:
                out = teacher_proc.stdout.read()
                if out.strip():
                    print("\n--- dummy teacher server output ---\n" + out, file=sys.stderr)
        except Exception:
            pass


if __name__ == "__main__":
    main()
