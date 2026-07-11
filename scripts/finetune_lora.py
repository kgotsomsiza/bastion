"""LoRA fine-tune of Qwen2.5-3B-Instruct for Bastion's local tier (V18).

Runs on the hackathon AMD box (MI300X, ROCm). Network there is whitelisted:
the base model comes via hf-mirror.com (HF_ENDPOINT), packages via PyPI.
Training data (train.jsonl, chat format) is uploaded via the Jupyter UI.

Pipeline: load -> LoRA SFT with assistant-only loss -> merge -> GGUF f16 ->
quantize Q4_K_M -> /workspace/bastion-v18-q4km.gguf (download via Jupyter).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE = "Qwen/Qwen2.5-3B-Instruct"
DATA = "/workspace/train.jsonl"
OUT = "/workspace/ft"
MERGED = "/workspace/merged"
LLAMA = "/opt/llama.cpp"

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

assert torch.cuda.is_available(), "GPU not visible to torch"
print("gpu:", torch.cuda.get_device_name(0), flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE)
MAXLEN = 1024


def encode(example):
    msgs = example["messages"]
    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_only = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_only, truncation=True, max_length=MAXLEN, add_special_tokens=False)["input_ids"]
    labels = list(full_ids)
    boundary = min(len(prompt_ids), len(labels))
    for i in range(boundary):
        labels[i] = -100  # loss on the assistant answer only
    pad = MAXLEN - len(full_ids)
    attention = [1] * len(full_ids) + [0] * pad
    full_ids = full_ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad
    labels = labels + [-100] * pad
    return {"input_ids": full_ids, "attention_mask": attention, "labels": labels}


rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
print("training rows:", len(rows), flush=True)
ds = Dataset.from_list(rows).map(encode, remove_columns=["messages"])

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT, num_train_epochs=2, per_device_train_batch_size=16,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="no", bf16=True, report_to=[],
    ),
    train_dataset=ds,
    data_collator=default_data_collator,
).train()

print("merging...", flush=True)
merged = model.merge_and_unload()
merged.save_pretrained(MERGED, safe_serialization=True)
tokenizer.save_pretrained(MERGED)

print("converting to GGUF f16...", flush=True)
subprocess.run([sys.executable, f"{LLAMA}/convert_hf_to_gguf.py", MERGED,
                "--outfile", "/workspace/ft-f16.gguf", "--outtype", "f16"], check=True)
print("quantizing Q4_K_M...", flush=True)
subprocess.run([f"{LLAMA}/build/bin/llama-quantize", "/workspace/ft-f16.gguf",
                "/workspace/bastion-v18-q4km.gguf", "Q4_K_M"], check=True)
print("DONE -> /workspace/bastion-v18-q4km.gguf", flush=True)
