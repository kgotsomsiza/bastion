"""Generate LoRA training data for the local-tier fine-tune (V18 phase 2).

Design: gold labels are deterministic wherever possible.
- NER: texts are TEMPLATED with known entities inserted -> gold = the spans.
- Sentiment: texts are generated CONDITIONED on a label -> gold = that label.
- Summarization: passages generated, summaries distilled from the serverless
  model, kept only when they pass the same constraint checks used at runtime.

Output: data/finetune/train.jsonl in chat format matching the runtime prompt
shape exactly (same system prompt and instructions as LocalModelProvider).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frugalrouter.providers.local_model import LOCAL_MODEL_SYSTEM, LOCAL_CATEGORY_INSTRUCTIONS  # noqa: E402
from frugalrouter.prompting import CATEGORY_INSTRUCTIONS  # noqa: E402
from frugalrouter.local_verify import verify_local_answer  # noqa: E402

API = "https://api.fireworks.ai/inference/v1/chat/completions"
KEY = os.environ["FIREWORKS_API_KEY"]
GEN_MODEL = "accounts/fireworks/models/kimi-k2p7-code"

random.seed(20260711)

FIRST = ["Sara", "Tomas", "Priya", "Kabelo", "Lena", "Marco", "Yuki", "Amara", "Devon", "Ingrid",
         "Rafael", "Zanele", "Oliver", "Mei", "Hassan", "Nadia", "Peter", "Thandi", "Jonas", "Elif"]
LAST = ["Nkosi", "Meyer", "Tanaka", "Okafor", "Silva", "Novak", "Dlamini", "Bergman", "Rossi", "Khan"]
CITIES = ["Cape Town", "Nairobi", "Oslo", "Kyoto", "Lisbon", "Austin", "Mumbai", "Toronto", "Accra", "Seoul"]
ORGS = ["Helix Labs", "BlueRiver Bank", "Northwind Logistics", "Kestrel Systems", "Ubuntu Clinic"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
CURR = ["$", "EUR ", "GBP ", "R"]

def call(prompt: str, max_tokens: int = 400) -> str:
    body = json.dumps({
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
        "User-Agent": "bastion-datagen/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""

def chat_row(category: str, task_text: str, gold: str) -> dict:
    instruction = LOCAL_CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS.get(category, ""))
    return {"messages": [
        {"role": "system", "content": LOCAL_MODEL_SYSTEM},
        {"role": "user", "content": f"{instruction}\n{task_text}"},
        {"role": "assistant", "content": gold},
    ]}

rows: list[dict] = []

# ---------- NER: fully templated, deterministic gold ----------
def gen_ner(n: int) -> None:
    made = 0
    while made < n:
        kind = random.choice(["names", "dates", "money", "emails", "cities"])
        if kind == "names":
            people = random.sample(FIRST, k=random.randint(2, 4))
            if random.random() < 0.5:
                people = [f"{p} {random.choice(LAST)}" for p in people]
            text = (f"{people[0]} presented the roadmap while "
                    + " and ".join(people[1:]) + f" took notes in {random.choice(CITIES)}.")
            ask, gold = "Extract all person names from this text", ", ".join(people)
        elif kind == "dates":
            dates = [f"{random.choice(MONTHS)} {random.randint(1,28)}" for _ in range(random.randint(2,3))]
            text = (f"The audit starts on {dates[0]}"
                    + (f", pauses on {dates[1]}" if len(dates) > 1 else "")
                    + (f", and wraps up on {dates[2]}." if len(dates) > 2 else "."))
            ask, gold = "Extract all dates mentioned", ", ".join(dates)
        elif kind == "money":
            amounts = [f"{random.choice(CURR)}{random.randint(5,900)}" for _ in range(random.randint(2,3))]
            text = (f"The subscription costs {amounts[0]} per month"
                    + (f", setup was {amounts[1]}" if len(amounts) > 1 else "")
                    + (f", and the deposit is {amounts[2]}." if len(amounts) > 2 else "."))
            ask, gold = "Extract all monetary values with their currencies", ", ".join(amounts)
        elif kind == "emails":
            emails = [f"{random.choice(FIRST).lower()}@{random.choice(['example.com','corp.io','mail.net'])}"
                      for _ in range(random.randint(1,2))]
            text = f"Contact {' or '.join(emails)} for access to the {random.choice(ORGS)} portal."
            ask, gold = "Extract all email addresses", ", ".join(emails)
        else:
            cities = random.sample(CITIES, k=random.randint(2,3))
            text = f"The tour covers {', '.join(cities[:-1])} and {cities[-1]} this quarter."
            ask, gold = "Extract all city names", ", ".join(cities)
        task = f"{ask}: {text}"
        if verify_local_answer(task, "ner", gold):
            rows.append(chat_row("ner", task, gold))
            made += 1

# ---------- Sentiment: label-conditioned generation ----------
def gen_sentiment(n: int) -> None:
    per = n // 3
    for label in ("positive", "negative", "neutral"):
        got = 0
        while got < per:
            style = random.choice(["product review", "service feedback", "app store review",
                                   "restaurant comment", "delivery feedback", "status update"])
            hint = {"positive": "clearly satisfied, no complaints",
                    "negative": "clearly dissatisfied, no praise",
                    "neutral": "a plain factual statement with no opinion"}[label]
            out = call(f"Write 8 short one-sentence {style} texts that are {hint}. "
                       f"One per line, no numbering, no quotes, varied wording and topics.")
            for line in out.splitlines():
                line = line.strip().strip('-').strip()
                if len(line) < 15 or len(line) > 220 or got >= per:
                    continue
                ask = random.choice([
                    "Classify the sentiment as positive, negative, or neutral",
                    "Label the sentiment of this text (positive/negative/neutral)",
                    "Is the sentiment positive, negative, or neutral?"])
                task = f"{ask}: {line}"
                rows.append(chat_row("sentiment", task, label))
                got += 1

# ---------- Summarization: distilled, constraint-verified ----------
def gen_summarization(n: int) -> None:
    got = 0
    while got < n:
        out = call("Write a 60-90 word workplace or news-style paragraph about a specific concrete "
                   "situation (a schedule change, an outage, a decision, a local event). "
                   "Then on a new line write 'SUMMARY:' followed by a one-sentence summary in "
                   "your own words (do not reuse 6+ consecutive words from the paragraph).", 320)
        if "SUMMARY:" not in out:
            continue
        passage, summary = out.split("SUMMARY:", 1)
        passage, summary = passage.strip(), summary.strip().splitlines()[0].strip()
        if not (40 <= len(passage.split()) <= 120 and 6 <= len(summary.split()) <= 28):
            continue
        task = f"Summarize this in one sentence: {passage}"
        if verify_local_answer(task, "summarization", summary):
            rows.append(chat_row("summarization", task, summary))
            got += 1

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    gen_ner(n)
    print(f"ner done: {len(rows)}", flush=True)
    gen_sentiment(n)
    print(f"+sentiment done: {len(rows)}", flush=True)
    gen_summarization(n // 2)
    print(f"+summarization done: {len(rows)}", flush=True)
    random.shuffle(rows)
    os.makedirs("data/finetune", exist_ok=True)
    with open("data/finetune/train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to data/finetune/train.jsonl")
