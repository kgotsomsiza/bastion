"""LoRA fine-tune of Qwen2.5-3B-Instruct for Bastion's local tier (V18).

Runs on the hackathon AMD box (ROCm). Network there is whitelisted:
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
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]


def augment_ner_formats(source_rows):
    augmented = list(source_rows)
    ner_index = 0
    sentiment_labels = {"positive", "negative", "neutral"}
    for row in source_rows:
        gold = row["messages"][2]["content"].strip()
        if gold in sentiment_labels:
            continue
        items = [item.strip() for item in gold.split(",") if item.strip()]
        if not items:
            continue
        variant = ner_index % 3
        ner_index += 1
        if variant == 0:
            suffix = "Return the result as a JSON array of strings."
            answer = json.dumps(items, ensure_ascii=False)
        elif variant == 1:
            suffix = "Return a pipe-separated list with no spaces around the | characters."
            answer = "|".join(items)
        else:
            suffix = "Return one extracted item per line."
            answer = "\n".join(items)
        messages = [dict(message) for message in row["messages"]]
        messages[1]["content"] = f"{messages[1]['content']}\n{suffix}"
        messages[2]["content"] = answer
        augmented.append({"messages": messages})
    return augmented


rows = augment_ner_formats(rows)
print("training rows after NER format augmentation:", len(rows), flush=True)
ds = Dataset.from_list(rows).map(encode, remove_columns=["messages"])
lengths = [len(row["input_ids"]) for row in ds]
print("token lengths:", min(lengths), "to", max(lengths), flush=True)
assert all(any(label != -100 for label in row["labels"]) for row in ds)

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
        dataloader_num_workers=2, remove_unused_columns=False,
    ),
    train_dataset=ds,
    data_collator=collator,
).train()

print("merging...", flush=True)
model.config.use_cache = True
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
