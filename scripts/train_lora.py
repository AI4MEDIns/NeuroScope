import json
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer)
from peft import LoraConfig, get_peft_model
import yaml

cfg = yaml.safe_load(open("configs/model.gpt2.small.yaml"))
BASE = cfg["base_model"]
LORA = cfg["lora"]; TCFG = cfg["train"]

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(BASE)

peft_cfg = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                      target_modules=["c_attn","c_proj"] if "gpt2" in BASE else None)
mdl = get_peft_model(mdl, peft_cfg)

ds = load_dataset("json", data_files={"train":"data/processed/train.jsonl","validation":"data/processed/val.jsonl"})

def build_text(ex):
    msgs = json.loads(ex["prompt"])
    head = "".join([("用户" if m["role"]=="user" else "助手")+": "+m["content"]+"\n" for m in msgs])
    return head + "助手: " + ex["response"]

def tokenize(batch):
    texts = [build_text(x) for x in batch["prompt"]]
    return tok(texts, truncation=True, max_length=TCFG["max_length"], padding="max_length")

ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="checkpoints/neuroscope",
    num_train_epochs=TCFG["num_epochs"],
    per_device_train_batch_size=TCFG["batch_size"],
    per_device_eval_batch_size=TCFG["batch_size"],
    gradient_accumulation_steps=TCFG["grad_accum"],
    learning_rate=TCFG["lr"],
    warmup_ratio=TCFG["warmup_ratio"],
    logging_steps=TCFG["logging_steps"],
    save_steps=TCFG["save_steps"],
    evaluation_strategy="steps",
    eval_steps=TCFG["save_steps"],
    fp16=False,
    report_to="none"
)

trainer = Trainer(model=mdl, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], data_collator=collator)
trainer.train()
trainer.save_model("checkpoints/neuroscope/final")
tok.save_pretrained("checkpoints/neuroscope/final")
print("✓ 训练完成 -> checkpoints/neuroscope/final")