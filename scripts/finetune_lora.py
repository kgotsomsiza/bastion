"""Conservative LoRA fine-tune of Qwen2.5-3B-Instruct for Bastion V19.

Runs on the hackathon AMD box (ROCm). Network there is whitelisted:
the base model comes via hf-mirror.com (HF_ENDPOINT), packages via PyPI.
Training and source-family holdout data are uploaded to ``/workspace``.

Pipeline: load -> LoRA SFT with assistant-only loss -> merge -> GGUF f16 ->
quantize Q4_K_M. The holdout is evaluated before and after training but is
never passed to ``Trainer.train``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = "Qwen/Qwen2.5-3B-Instruct"
DATA = os.getenv("BASTION_TRAIN_DATA", "/workspace/train_v2.jsonl")
HOLDOUT = os.getenv("BASTION_HOLDOUT_DATA", "/workspace/holdout_v2.jsonl")
VARIANT = os.getenv("BASTION_FT_VARIANT", "v19-conservative")
assert re.fullmatch(r"[a-zA-Z0-9._-]+", VARIANT), "unsafe BASTION_FT_VARIANT"
OUT = f"/workspace/{VARIANT}-lora"
MERGED = f"/workspace/{VARIANT}-merged"
F16 = f"/workspace/{VARIANT}-f16.gguf"
GGUF = f"/workspace/bastion-{VARIANT}-q4km.gguf"
METRICS = f"/workspace/{VARIANT}-training-metrics.json"
LLAMA = "/opt/llama.cpp"

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

assert torch.cuda.is_available(), "GPU not visible to torch"
print("gpu:", torch.cuda.get_device_name(0), flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
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
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


rows = [json.loads(line) for line in open(DATA, encoding="utf-8") if line.strip()]
holdout_rows = [json.loads(line) for line in open(HOLDOUT, encoding="utf-8") if line.strip()]
holdout_messages = [{"messages": row["messages"]} for row in holdout_rows]
print("training rows:", len(rows), "holdout rows:", len(holdout_messages), flush=True)
ds = Dataset.from_list(rows).map(encode, remove_columns=["messages"])
holdout_ds = Dataset.from_list(holdout_messages).map(encode, remove_columns=["messages"])
lengths = [len(row["input_ids"]) for row in ds]
print("token lengths:", min(lengths), "to", max(lengths), flush=True)
assert all(any(label != -100 for label in row["labels"]) for row in ds)
assert all(any(label != -100 for label in row["labels"]) for row in holdout_ds)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
    return_tensors="pt",
)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.10, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=16,
        learning_rate=5e-5, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01, max_grad_norm=1.0,
        logging_steps=10, save_strategy="no", bf16=True, report_to=[],
        dataloader_num_workers=2, remove_unused_columns=False,
        seed=20260712, data_seed=20260712,
    ),
    train_dataset=ds,
    eval_dataset=holdout_ds,
    data_collator=collator,
)
base_metrics = trainer.evaluate(metric_key_prefix="base_holdout")
train_metrics = trainer.train().metrics
tuned_metrics = trainer.evaluate(metric_key_prefix="tuned_holdout")
print("base holdout:", base_metrics, flush=True)
print("tuned holdout:", tuned_metrics, flush=True)

print("merging...", flush=True)
model.config.use_cache = True
merged = model.merge_and_unload()
merged.save_pretrained(MERGED, safe_serialization=True)
tokenizer.save_pretrained(MERGED)

print("converting to GGUF f16...", flush=True)
subprocess.run([sys.executable, f"{LLAMA}/convert_hf_to_gguf.py", MERGED,
                "--outfile", F16, "--outtype", "f16"], check=True)
print("quantizing Q4_K_M...", flush=True)
subprocess.run([f"{LLAMA}/build/bin/llama-quantize", F16, GGUF, "Q4_K_M"], check=True)
metadata = {
    "variant": VARIANT,
    "base_model": BASE,
    "train_rows": len(rows),
    "holdout_rows": len(holdout_messages),
    "train_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest(),
    "holdout_sha256": hashlib.sha256(open(HOLDOUT, "rb").read()).hexdigest(),
    "gguf_sha256": hashlib.sha256(open(GGUF, "rb").read()).hexdigest(),
    "training": {"epochs": 1, "learning_rate": 5e-5, "lora_rank": 8, "lora_alpha": 16},
    "base_holdout": base_metrics,
    "train_metrics": train_metrics,
    "tuned_holdout": tuned_metrics,
}
with open(METRICS, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
print(f"DONE -> {GGUF}", flush=True)
print(f"metrics -> {METRICS}", flush=True)
