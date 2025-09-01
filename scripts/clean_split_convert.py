import json, re, random
from pathlib import Path
from datasets import Dataset, DatasetDict

random.seed(42)
RAW = Path("data/raw/corpus.jsonl")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

def sanitize(msg: str) -> str:
    msg = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[email]", msg)
    msg = re.sub(r"\b(\+?\d[\d\- ]{7,}\d)\b", "[phone]", msg)
    return msg.strip()

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for m in row["messages"]:
                m["content"] = sanitize(m["content"])[:1000]
            yield row

rows = list(load_rows(RAW))
random.shuffle(rows)
n = len(rows)
train, val, test = rows[:int(0.9*n)], rows[int(0.9*n):int(0.95*n)], rows[int(0.95*n):]

def to_sft(rows):
    data = []
    for r in rows:
        if len(r["messages"])<2: continue
        prompt = r["messages"][:-1]
        answer = r["messages"][-1]
        if answer["role"]!="assistant": continue
        data.append({"prompt": json.dumps(prompt, ensure_ascii=False), "response": answer["content"]})
    return data

dd = DatasetDict({
    "train": Dataset.from_list(to_sft(train)),
    "validation": Dataset.from_list(to_sft(val)),
    "test": Dataset.from_list(to_sft(test)),
})

dd["train"].to_json(OUT/"train.jsonl")
dd["validation"].to_json(OUT/"val.jsonl")
dd["test"].to_json(OUT/"test.jsonl")
print("✓ 清洗/切分完成:", OUT)