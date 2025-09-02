# train_full.py
# 全量微调用 medical.txt + vocab.txt + config.json 训练 GPT2LMHeadModel
import os, json, math, argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    BertTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, set_seed
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含 medical.txt / vocab.txt / config.json 的目录")
    ap.add_argument("--output_dir", default="checkpoints/med_full", help="输出目录")
    ap.add_argument("--block_size", type=int, default=512, help="拼接分块长度")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Tokenizer
    vocab_path = data_dir / "vocab.txt"
    tok = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)
    if tok.pad_token is None: tok.add_special_tokens({"pad_token":"[PAD]"})
    if tok.sep_token is None: tok.add_special_tokens({"sep_token":"[SEP]"})
    if tok.cls_token is None: tok.add_special_tokens({"cls_token":"[CLS]"})

    # 2) Config
    config = AutoConfig.from_pretrained(str(data_dir))
    config.vocab_size   = len(tok)
    config.eos_token_id = tok.sep_token_id
    config.bos_token_id = tok.cls_token_id
    config.pad_token_id = tok.pad_token_id

    # 3) Model
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tok))

    # 4) Dataset
    txt_path = data_dir / "medical.txt"
    raw = load_dataset("text", data_files={"all": str(txt_path)})
    split = raw["all"].train_test_split(test_size=0.02, seed=42)
    ds_train, ds_val = split["train"], split["test"]

    def tok_fn(examples):
        return tok(examples["text"], add_special_tokens=True, truncation=False)

    tokenized_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    tokenized_val   = ds_val.map(tok_fn,   batched=True, remove_columns=["text"])

    def group_texts(examples):
        block = args.block_size
        ids = sum(examples["input_ids"], [])
        tot = (len(ids) // block) * block
        ids = ids[:tot]
        res = {
            "input_ids": [ids[i:i+block] for i in range(0, tot, block)],
        }
        res["attention_mask"] = [[1]*block for _ in res["input_ids"]]
        res["labels"] = [x[:] for x in res["input_ids"]]
        return res

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_val   = tokenized_val.map(group_texts,   batched=True)

    # 5) Data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # 6) Training args
    args_train = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        bf16=False, fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        data_collator=collator,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        tokenizer=tok,
    )

    trainer.train()

    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", None)
    if loss is not None:
        try:
            ppl = math.exp(loss)
            print(f"Eval loss: {loss:.4f} | Perplexity: {ppl:.2f}")
        except OverflowError:
            print(f"Eval loss: {loss:.4f} | Perplexity: overflow")

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    with open(out_dir / "TRAIN_INFO.txt", "w") as f:
        f.write("Trained from scratch on medical.txt with GPT2LMHeadModel + BertTokenizer.\n")

if __name__ == "__main__":
    main()
