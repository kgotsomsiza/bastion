"""V19 training data: diversity + source-family holdout.

V18's fine-tune failed hidden scoring because training data and validation
tasks shared provenance. V2 fixes this structurally:

- Multiple SOURCE FAMILIES per category (distinct template pools, distinct
  generation personas/domains).
- An ENTIRE family per category is written to holdout.jsonl and never trained.
- The ship gate later compares stock vs tuned on holdout + blind tasks; the
  tuned model ships only if accepted-answer accuracy is perfect and at least
  matches stock's acceptance.

Outputs: data/finetune/train_v2.jsonl, data/finetune/holdout_v2.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frugalrouter.providers.local_model import LOCAL_MODEL_SYSTEM, LOCAL_CATEGORY_INSTRUCTIONS  # noqa: E402
from frugalrouter.prompting import CATEGORY_INSTRUCTIONS  # noqa: E402
from frugalrouter.local_verify import verify_local_answer  # noqa: E402

API = "https://api.fireworks.ai/inference/v1/chat/completions"
KEY = os.environ["FIREWORKS_API_KEY"]
GEN_MODEL = "accounts/fireworks/models/kimi-k2p7-code"

random.seed(20260712)

FIRST = ["Sara", "Tomas", "Priya", "Kabelo", "Lena", "Marco", "Yuki", "Amara", "Devon", "Ingrid",
         "Rafael", "Zanele", "Oliver", "Mei", "Hassan", "Nadia", "Peter", "Thandi", "Jonas", "Elif",
         "Bianca", "Kofi", "Astrid", "Ravi", "Carmen", "Dmitri", "Fatima", "Liam", "Noor", "Sipho"]
LAST = ["Nkosi", "Meyer", "Tanaka", "Okafor", "Silva", "Novak", "Dlamini", "Bergman", "Rossi", "Khan",
        "Petrov", "Andersson", "Moyo", "Costa", "Yamada"]
CITIES = ["Cape Town", "Nairobi", "Oslo", "Kyoto", "Lisbon", "Austin", "Mumbai", "Toronto", "Accra",
          "Seoul", "Bogota", "Vienna", "Perth", "Casablanca", "Reykjavik"]
ORGS = ["Helix Labs", "BlueRiver Bank", "Northwind Logistics", "Kestrel Systems", "Ubuntu Clinic",
        "Argon Foundry", "Silverleaf Media", "Quantia Insurance"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
CURR = ["$", "EUR ", "GBP ", "R", "CHF "]
PHONES = ["555-0192", "555-8841", "021-555-7733", "(212) 555-0007"]
REGS = ["GDPR", "CCPA", "HIPAA", "POPIA", "PCI-DSS", "SOX"]

# NER ask-phrasings split into two families; family B is HOLDOUT-only.
NER_ASKS_A = ["Extract all {kind} from this text", "List the {kind} mentioned",
              "Pull out every {kind} you can find in this passage"]
NER_ASKS_B = ["Identify each {kind} appearing below", "What {kind} does this text contain? List them",
              "Find and return the {kind} in the following"]

# Sentiment domains split into two families; family B is HOLDOUT-only.
SENT_DOMAINS_A = ["product review", "restaurant comment", "app store review", "delivery feedback",
                  "hotel review", "customer support transcript line"]
SENT_DOMAINS_B = ["gym membership feedback", "airline experience tweet", "online course review",
                  "car repair shop comment", "streaming service complaint or praise"]
SENT_STYLES = ["plain", "casual with a typo or two", "formal", "very short (under 10 words)",
               "slightly verbose", "with an emoji"]


def call(prompt: str, max_tokens: int = 400, temperature: float = 1.0) -> str:
    body = json.dumps({
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
        "User-Agent": "bastion-datagen/2.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def chat_row(category: str, task_text: str, gold: str, family: str) -> dict:
    instruction = LOCAL_CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS.get(category, ""))
    return {"family": family, "category": category, "messages": [
        {"role": "system", "content": LOCAL_MODEL_SYSTEM},
        {"role": "user", "content": f"{instruction}\n{task_text}"},
        {"role": "assistant", "content": gold},
    ]}


def make_ner_text() -> tuple[str, str, list[str]]:
    """Returns (entity kind, sentence, gold spans) - fully templated, exact gold."""
    kind = random.choice(["person names", "dates", "monetary values", "email addresses",
                          "city names", "organization names", "phone numbers", "regulations"])
    if kind == "person names":
        ents = random.sample(FIRST, k=random.randint(2, 4))
        if random.random() < 0.5:
            ents = [f"{p} {random.choice(LAST)}" for p in ents]
        text = (f"{ents[0]} chaired the meeting"
                + ("".join(f", with {p} presenting" for p in ents[1:-1]) if len(ents) > 2 else "")
                + (f" and {ents[-1]} taking minutes." if len(ents) > 1 else "."))
    elif kind == "dates":
        ents = [f"{random.choice(MONTHS)} {random.randint(1, 28)}" for _ in range(random.randint(2, 3))]
        text = f"Phase one begins on {ents[0]}" + (f", review lands on {ents[1]}" if len(ents) > 1 else "") + \
               (f", and delivery is due {ents[2]}." if len(ents) > 2 else ".")
    elif kind == "monetary values":
        ents = [f"{random.choice(CURR)}{random.randint(5, 900)}" for _ in range(random.randint(2, 3))]
        text = f"The base plan is {ents[0]}" + (f", the upgrade costs {ents[1]}" if len(ents) > 1 else "") + \
               (f", and the annual bundle is {ents[2]}." if len(ents) > 2 else ".")
    elif kind == "email addresses":
        ents = [f"{random.choice(FIRST).lower()}{random.randint(1,99)}@{random.choice(['example.com','corp.io','mail.net','firm.org'])}"
                for _ in range(random.randint(1, 3))]
        text = f"Direct queries to {', '.join(ents[:-1]) + ' or ' if len(ents) > 1 else ''}{ents[-1]} before Friday."
    elif kind == "city names":
        ents = random.sample(CITIES, k=random.randint(2, 3))
        text = f"The roadshow visits {', '.join(ents[:-1])} and {ents[-1]} next quarter."
    elif kind == "organization names":
        ents = random.sample(ORGS, k=random.randint(2, 3))
        text = f"{ents[0]} signed the agreement" + (f" alongside {' and '.join(ents[1:])}." if len(ents) > 1 else ".")
    elif kind == "phone numbers":
        ents = random.sample(PHONES, k=random.randint(1, 2))
        text = f"Call {' or '.join(ents)} for after-hours support."
    else:
        ents = random.sample(REGS, k=random.randint(2, 3))
        text = f"The audit checks compliance with {', '.join(ents[:-1])} and {ents[-1]}."
    return kind, text, ents


rows_train: list[dict] = []
rows_holdout: list[dict] = []


def gen_ner(n_train: int, n_holdout: int) -> None:
    made_t = made_h = 0
    while made_t < n_train or made_h < n_holdout:
        to_holdout = made_h < n_holdout and (made_t >= n_train or random.random() < 0.2)
        asks = NER_ASKS_B if to_holdout else NER_ASKS_A
        kind, text, ents = make_ner_text()
        ask = random.choice(asks).format(kind=kind)
        task = f"{ask}: {text}"
        gold = ", ".join(ents)
        if not verify_local_answer(task, "ner", gold):
            continue
        row = chat_row("ner", task, gold, "ner_B_holdout" if to_holdout else "ner_A_train")
        if to_holdout:
            rows_holdout.append(row); made_h += 1
        else:
            rows_train.append(row); made_t += 1


def gen_sentiment(n_train: int, n_holdout: int) -> None:
    per_label_t = n_train // 3
    per_label_h = n_holdout // 3
    for label in ("positive", "negative", "neutral"):
        for pool, domains, per, family in (
            (rows_train, SENT_DOMAINS_A, per_label_t, "sent_A_train"),
            (rows_holdout, SENT_DOMAINS_B, per_label_h, "sent_B_holdout"),
        ):
            got = 0
            attempts = 0
            while got < per and attempts < per:
                attempts += 1
                domain = random.choice(domains)
                style = random.choice(SENT_STYLES)
                hint = {"positive": "clearly satisfied, no complaints, no negation words",
                        "negative": "clearly dissatisfied, no praise, no negation words like 'not'",
                        "neutral": "a plain factual statement with no opinion and no negation words"}[label]
                out = call(f"Write 10 one-sentence {domain} texts, style: {style}. Each must be {hint}. "
                           f"One per line, no numbering, no quotes.")
                for line in out.splitlines():
                    line = line.strip().strip("-• ").strip()
                    if len(line) < 12 or len(line) > 220 or got >= per:
                        continue
                    ask = random.choice([
                        "Classify the sentiment as positive, negative, or neutral",
                        "Label this text's sentiment (positive/negative/neutral)",
                        "Is the sentiment positive, negative, or neutral?",
                        "Sentiment classification, one word (positive, negative, neutral)"])
                    task = f"{ask}: {line}"
                    # Runtime gate parity: skip anything the nuance gate would reject.
                    if not verify_local_answer(task, "sentiment", label):
                        continue
                    pool.append(chat_row("sentiment", task, label, family))
                    got += 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    gen_ner(n, n // 6)
    print(f"ner done: train={sum(1 for r in rows_train if r['category']=='ner')} "
          f"holdout={sum(1 for r in rows_holdout if r['category']=='ner')}", flush=True)
    gen_sentiment(n, n // 6)
    print(f"sentiment done: train={sum(1 for r in rows_train if r['category']=='sentiment')} "
          f"holdout={sum(1 for r in rows_holdout if r['category']=='sentiment')}", flush=True)
    random.shuffle(rows_train)
    os.makedirs("data/finetune", exist_ok=True)
    with open("data/finetune/train_v2.jsonl", "w", encoding="utf-8") as f:
        for r in rows_train:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    with open("data/finetune/holdout_v2.jsonl", "w", encoding="utf-8") as f:
        for r in rows_holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows_train)} train, {len(rows_holdout)} holdout", flush=True)
