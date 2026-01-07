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
from typing import Any
from urllib.error import URLError

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer
from trl.experimental.gold.teacher_client import VLLMTeacherClient, VLLMTeacherClientConfig


DEFAULT_STUDENT_MODEL = "EverAI-AI/MagistSmall-Raven-DPO6-c"
DEFAULT_DATASET = "EverAI-AI/french-conversations-prompt"
DEFAULT_TEACHER_TOKENIZER = "deepseek-ai/DeepSeek-V3.1"


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


def wait_for_teacher_vllm(
    *,
    base_url: str,
    model_name: str,
    vocab_size: int,
    token_id: int,
    timeout_s: float = 30.0,
    full_logprobs_format: str,
    full_logprobs_top_p: float | None,
    full_logprobs_max_top_k: int | None,
) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    client = VLLMTeacherClient(
        VLLMTeacherClientConfig(
            base_url=base_url,
            model_name=model_name,
            timeout=5.0,
            max_retries=0,
            full_logprobs_format=full_logprobs_format,
            full_logprobs_top_p=full_logprobs_top_p,
            full_logprobs_max_top_k=full_logprobs_max_top_k,
        )
    )
    while time.time() < deadline:
        try:
            client.fetch_full_logprobs(
                [token_id, token_id],
                [1],
                vocab_size=vocab_size,
            )
            return
        except (URLError, Exception) as exc:
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(f"Teacher did not respond at {base_url!r} within {timeout_s}s.") from last_err


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


def get_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ensure_port_available(port: int) -> int:
    import socket

    if port == 0:
        return get_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return get_free_port()
    return port


def _convert_sharegpt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted = []
    for message in messages:
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        if role is None or content is None:
            continue
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        converted.append({"role": role, "content": content})
    return converted


def to_messages(example: dict[str, Any]) -> dict[str, Any]:
    if "messages" in example:
        return {"messages": example["messages"]}

    for key in ("conversations", "conversation", "dialog", "dialogue"):
        value = example.get(key)
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                if "role" in value[0] and "content" in value[0]:
                    return {"messages": value}
                if "from" in value[0] and "value" in value[0]:
                    return {"messages": _convert_sharegpt_messages(value)}
            if value and isinstance(value[0], str) and len(value) >= 2:
                return {"messages": [{"role": "user", "content": value[0]}, {"role": "assistant", "content": value[1]}]}

    for user_key, assistant_key in (
        ("prompt", "completion"),
        ("prompt", "response"),
        ("instruction", "output"),
        ("input", "output"),
        ("question", "answer"),
    ):
        if user_key in example and assistant_key in example:
            messages = []
            if example.get("system"):
                messages.append({"role": "system", "content": example["system"]})
            messages.append({"role": "user", "content": example[user_key]})
            messages.append({"role": "assistant", "content": example[assistant_key]})
            return {"messages": messages}

    raise ValueError(f"Unable to infer ChatML fields from dataset columns: {sorted(example.keys())}")


def ensure_chat_template(tokenizer: AutoTokenizer) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )


def resolve_max_length(tokenizer: AutoTokenizer, requested: int | None) -> int:
    if requested is not None:
        max_len = getattr(tokenizer, "model_max_length", None)
        if max_len is not None and max_len > 0 and max_len < 100000 and requested > max_len:
            print(f"Requested max_length={requested} exceeds model_max_length={max_len}; clamping to {max_len}.")
            return int(max_len)
        return int(requested)
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len <= 0 or max_len > 100000:
        return 8192
    return int(max_len)


def preflight_teacher(
    *,
    base_url: str,
    model_name: str,
    timeout: float,
    max_retries: int,
    vocab_size: int,
    token_ids: list[int],
    positions: list[int],
    attempts: int,
    min_success: int,
    full_logprobs_format: str,
    full_logprobs_top_p: float | None,
    full_logprobs_max_top_k: int | None,
) -> None:
    if attempts <= 0:
        return
    client = VLLMTeacherClient(
        VLLMTeacherClientConfig(
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            full_logprobs_format=full_logprobs_format,
            full_logprobs_top_p=full_logprobs_top_p,
            full_logprobs_max_top_k=full_logprobs_max_top_k,
        )
    )
    success = 0
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            client.fetch_full_logprobs(token_ids, positions, vocab_size=vocab_size)
            success += 1
        except Exception as exc:
            last_exc = exc
        if success >= min_success:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"Teacher preflight failed: {success}/{attempts} successful requests (min_success={min_success})."
    ) from last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GOLD external-teacher smoke training with a dummy teacher.")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Student model id (defaults to STUDENT_MODEL_ID env or a project default).",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default="train[:128]")
    parser.add_argument("--teacher-tokenizer", default=DEFAULT_TEACHER_TOKENIZER)
    parser.add_argument(
        "--teacher-url",
        default=None,
        help="Override teacher vLLM /v1/completions URL (defaults to TEACHER_VLLM_URL or local dummy).",
    )
    parser.add_argument(
        "--teacher-model-name",
        default=None,
        help="Teacher model name for the vLLM endpoint (defaults to TEACHER_VLLM_MODEL_NAME or dummy).",
    )
    parser.add_argument("--teacher-timeout", type=float, default=30.0, help="Teacher endpoint timeout (seconds).")
    parser.add_argument("--teacher-max-retries", type=int, default=2, help="Teacher endpoint max retries.")
    parser.add_argument(
        "--teacher-full-logprobs-format",
        default="top_p",
        choices=["top_p", "base64_dense"],
        help="Teacher full_logprobs format to request.",
    )
    parser.add_argument(
        "--teacher-full-logprobs-top-p",
        type=float,
        default=0.9999,
        help="Top-p cutoff for sparse full_logprobs format.",
    )
    parser.add_argument(
        "--teacher-full-logprobs-max-top-k",
        type=int,
        default=512,
        help="Maximum top-k per position for sparse full_logprobs format.",
    )
    parser.add_argument(
        "--teacher-max-input-tokens",
        type=int,
        default=None,
        help="Max teacher input tokens for external endpoint (truncates from left).",
    )
    parser.add_argument("--teacher-preflight-requests", type=int, default=3)
    parser.add_argument("--teacher-preflight-min-success", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/gold-external-teacher-smoke")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--teacher-port", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-new-tokens", type=int, default=None, help="Force a minimum number of generated tokens.")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA for the student model.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated list of module names to target with LoRA. If unset, attempts to infer.",
    )
    parser.add_argument("--report-to-none", action="store_true", help="Disable all report_to integrations.")
    parser.add_argument("--log-cuda-memory", action="store_true", help="Log peak CUDA memory after training.")
    parser.add_argument(
        "--log-rollouts",
        action="store_true",
        help="Log sampled (prompt, completion) rollouts to stdout and wandb tables.",
    )
    parser.add_argument(
        "--log-rollouts-steps",
        type=int,
        default=100,
        help="Steps between logging rollouts when --log-rollouts is enabled.",
    )
    parser.add_argument(
        "--rollouts-per-log",
        type=int,
        default=10,
        help="Number of rollouts to log when --log-rollouts is enabled.",
    )
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for student rollouts.")
    parser.add_argument("--vllm-mode", default="server", choices=["server", "colocate"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-sync-frequency", type=int, default=1)
    parser.add_argument("--vllm-enable-sleep-mode", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the final checkpoint to the Hub.")
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Hub repo id for push_to_hub (defaults to EverAI-AI/<model>_onpolicy_french).",
    )
    parser.add_argument("--wandb-project", default="trl-gold-external-teacher-smoke")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    load_dotenv()
    if args.model_id is None:
        args.model_id = os.environ.get("STUDENT_MODEL_ID", DEFAULT_STUDENT_MODEL)
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
    ensure_chat_template(tokenizer)

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_port = ensure_port_available(args.teacher_port)
    teacher_url = args.teacher_url or os.environ.get("TEACHER_VLLM_URL")
    teacher_model_name = args.teacher_model_name or os.environ.get("TEACHER_VLLM_MODEL_NAME") or "dummy-teacher"
    if args.teacher_max_input_tokens is None:
        env_limit = os.environ.get("TEACHER_MAX_INPUT_TOKENS")
        if env_limit:
            try:
                args.teacher_max_input_tokens = int(env_limit)
            except ValueError:
                raise ValueError("TEACHER_MAX_INPUT_TOKENS must be an integer.")
    teacher_proc = None
    if not teacher_url:
        teacher_port = ensure_port_available(args.teacher_port)
        teacher_url = f"http://127.0.0.1:{teacher_port}/v1/completions"
        teacher_proc = start_dummy_teacher(port=teacher_port, vocab_size=len(teacher_tokenizer), seed=0)
    try:
        token_id = teacher_tokenizer.eos_token_id or 0
        wait_for_teacher_vllm(
            base_url=teacher_url,
            model_name=teacher_model_name,
            vocab_size=len(teacher_tokenizer),
            token_id=int(token_id),
            timeout_s=60.0,
            full_logprobs_format=args.teacher_full_logprobs_format,
            full_logprobs_top_p=args.teacher_full_logprobs_top_p,
            full_logprobs_max_top_k=args.teacher_full_logprobs_max_top_k,
        )

        dataset = load_dataset(args.dataset_id, split=args.dataset_split)
        dataset = dataset.map(to_messages, remove_columns=dataset.column_names)

        max_length = resolve_max_length(tokenizer, args.max_length)
        if max_length <= 1:
            raise ValueError("max_length must be > 1 to leave room for generation.")
        max_completion_length = min(int(args.max_completion_length), max_length - 1)
        max_prompt_tokens = max_length - max_completion_length
        if max_prompt_tokens <= 0:
            raise ValueError("max_completion_length is too large for max_length.")

        def keep_example(example: dict[str, Any]) -> bool:
            prompt_messages = example["messages"][:-1]
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(
                formatted_prompt,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            return len(prompt_ids) <= max_prompt_tokens

        dataset = dataset.filter(keep_example)
        if len(dataset) == 0:
            raise ValueError("Filtered dataset is empty. Increase max_length or reduce max_completion_length.")

        if args.teacher_preflight_requests > 0:
            sample_messages = dataset[0]["messages"]
            teacher_text = teacher_tokenizer.apply_chat_template(
                sample_messages, tokenize=False, add_generation_prompt=False
            )
            teacher_ids = teacher_tokenizer(
                teacher_text, add_special_tokens=False, truncation=False, padding=False, return_tensors=None
            )["input_ids"]
            if args.teacher_max_input_tokens is not None and len(teacher_ids) > args.teacher_max_input_tokens:
                teacher_ids = teacher_ids[-args.teacher_max_input_tokens :]
            if not teacher_ids:
                raise ValueError("Teacher preflight failed: empty tokenized input.")
            if len(teacher_ids) == 1:
                teacher_ids = teacher_ids + teacher_ids
            last_pos = max(1, len(teacher_ids) - 1)
            preflight_teacher(
                base_url=teacher_url,
                model_name=teacher_model_name,
                timeout=args.teacher_timeout,
                max_retries=args.teacher_max_retries,
                vocab_size=len(teacher_tokenizer),
                token_ids=teacher_ids,
                positions=[last_pos],
                attempts=args.teacher_preflight_requests,
                min_success=max(1, args.teacher_preflight_min_success),
                full_logprobs_format=args.teacher_full_logprobs_format,
                full_logprobs_top_p=args.teacher_full_logprobs_top_p,
                full_logprobs_max_top_k=args.teacher_full_logprobs_max_top_k,
            )

        model_init_kwargs = {"device_map": None, "trust_remote_code": True}
        if torch.cuda.is_available():
            model_init_kwargs["torch_dtype"] = torch.bfloat16

        report_to = ["wandb"]
        if args.report_to_none:
            report_to = []

        max_steps = args.max_steps if args.max_steps is not None and args.max_steps > 0 else None
        num_train_epochs = args.num_train_epochs
        if num_train_epochs is None:
            num_train_epochs = 1.0

        save_strategy = "steps" if max_steps is not None else "epoch"
        save_steps = max(1, max_steps) if max_steps is not None else 0
        max_steps_value = max_steps if max_steps is not None else -1

        train_args = GOLDConfig(
            output_dir=args.output_dir,
            report_to=report_to,
            logging_steps=1,
            save_strategy=save_strategy,
            save_steps=save_steps,
            max_steps=max_steps_value,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type="constant",
            warmup_steps=0,
            eval_strategy="no",
            max_completion_length=max_completion_length,
            max_length=max_length,
            lmbda=1.0,
            use_external_teacher_vllm=True,
            teacher_vllm_base_url=teacher_url,
            teacher_vllm_model_name=teacher_model_name,
            teacher_vllm_timeout=args.teacher_timeout,
            teacher_vllm_max_retries=args.teacher_max_retries,
            teacher_vllm_max_input_tokens=args.teacher_max_input_tokens,
            teacher_vllm_full_logprobs_format=args.teacher_full_logprobs_format,
            teacher_vllm_full_logprobs_top_p=args.teacher_full_logprobs_top_p,
            teacher_vllm_full_logprobs_max_top_k=args.teacher_full_logprobs_max_top_k,
            teacher_vllm_fail_on_error=True,
            teacher_tokenizer_name_or_path=args.teacher_tokenizer,
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_sync_frequency=args.vllm_sync_frequency,
            vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        )
        train_args.model_init_kwargs = model_init_kwargs
        train_args.log_completions = args.log_rollouts
        train_args.log_completions_steps = args.log_rollouts_steps
        train_args.num_completions_to_print = args.rollouts_per_log

        peft_config = None
        if args.use_lora:
            try:
                from peft import LoraConfig, TaskType
            except ImportError as exc:
                raise ImportError("LoRA requested but `peft` is not installed. Install with `pip install peft`.") from exc

            target_modules = None
            if args.lora_target_modules is not None:
                target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
            else:
                config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
                model_type = getattr(config, "model_type", "")
                if model_type in {"gpt2"}:
                    target_modules = ["c_attn", "c_proj"]
                elif model_type in {"llama", "mistral", "mistral3", "mixtral", "qwen2", "qwen2_moe", "qwen3"}:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

            if not target_modules:
                raise ValueError(
                    "Unable to infer LoRA target modules. Please pass --lora-target-modules explicitly."
                )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )
            train_args.gradient_checkpointing = True
            if torch.cuda.is_available():
                train_args.bf16 = True

        trainer = GOLDTrainer(
            model=args.model_id,
            teacher_model=None,
            args=train_args,
            train_dataset=dataset,
            eval_dataset=None,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        if args.min_new_tokens is not None:
            trainer.generation_config.min_new_tokens = int(args.min_new_tokens)
        trainer.train()
        if args.push_to_hub:
            hub_model_id = args.hub_model_id
            if hub_model_id is None:
                model_slug = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in args.model_id)
                hub_model_id = f"EverAI-AI/{model_slug}_onpolicy_french"
            trainer.push_to_hub(dataset_name=args.dataset_id, hub_model_id=hub_model_id)
        if args.log_cuda_memory and torch.cuda.is_available():
            max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            print(f"max_cuda_allocated_gb={max_alloc:.2f}")
            print(f"max_cuda_reserved_gb={max_reserved:.2f}")
    finally:
        if teacher_proc is not None:
            stop_process(teacher_proc)

        # If the server printed errors, surface them to help debugging (without leaking env vars).
        try:
            if teacher_proc is not None and teacher_proc.stdout is not None:
                out = teacher_proc.stdout.read()
                if out.strip():
                    print("\n--- dummy teacher server output ---\n" + out, file=sys.stderr)
        except Exception:
            pass


if __name__ == "__main__":
    main()
