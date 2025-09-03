# train_full.py
# 全量微调用你的 medical.txt + vocab.txt + config.json 训练 GPT2LMHeadModel
# GPT2 架构 + BertTokenizer（WordPiece）
# 已包含：
#  - 兼容旧 API：TrainingArguments(eval_strategy=...)
#  - 防“乱码”：显式映射 BOS/CLS、EOS/SEP、PAD/[PAD]
#  - 修复 token_type_ids 长度不一致
#  - 训练后在 config.json 中写入 model_type/architectures，前端可直接识别

import os, json, math, argparse
from pathlib import Path
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

    # 1) Tokenizer（BERT 词表）
    vocab_path = data_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"缺少 {vocab_path}")
    tok = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)
    if tok.pad_token is None: tok.add_special_tokens({"pad_token":"[PAD]"})
    if tok.sep_token is None: tok.add_special_tokens({"sep_token":"[SEP]"})
    if tok.cls_token is None: tok.add_special_tokens({"cls_token":"[CLS]"})

    # 2) Config（GPT-2 架构；读取 data_dir 下的 config.json）
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少 {cfg_path}")
    config = AutoConfig.from_pretrained(str(data_dir))

    # 与 tokenizer 对齐（防乱码）
    config.vocab_size   = len(tok)
    config.eos_token_id = tok.sep_token_id
    config.bos_token_id = tok.cls_token_id
    config.pad_token_id = tok.pad_token_id

    # 关键：显式声明模型类型，确保前端可识别加载
    config.model_type = "gpt2"
    config.architectures = ["GPT2LMHeadModel"]

    # 兼容字段（若缺失则补默认值）
    if getattr(config, "n_ctx", None) is None and getattr(config, "n_positions", None) is not None:
        config.n_ctx = config.n_positions
    if getattr(config, "n_positions", None) is None and getattr(config, "n_ctx", None) is not None:
        config.n_positions = config.n_ctx
    if getattr(config, "n_ctx", None) is None and getattr(config, "n_positions", None) is None:
        config.n_ctx = config.n_positions = 1024

    # 3) Model（从配置随机初始化）
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tok))

    # 4) Dataset
    txt_path = data_dir / "medical.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"缺少 {txt_path}")

    raw = load_dataset("text", data_files={"train": str(txt_path)})
    split = raw["train"].train_test_split(test_size=0.02, seed=42)
    ds_train, ds_val = split["train"], split["test"]

    def tok_fn(examples):
        return tok(examples["text"], add_special_tokens=True, truncation=False)

    tokenized_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    tokenized_val   = ds_val.map(tok_fn,   batched=True, remove_columns=["text"])

    # 把长序列拼接后按 block_size 切块；同步裁剪 token_type_ids
    def group_texts(examples):
        block = args.block_size
        ids = sum(examples["input_ids"], [])
        tot = (len(ids) // block) * block
        ids = ids[:tot]
        res = {"input_ids": [ids[i:i+block] for i in range(0, tot, block)]}
        res["attention_mask"] = [[1]*block for _ in res["input_ids"]]
        res["labels"] = [x[:] for x in res["input_ids"]]
        if "token_type_ids" in examples:
            tids = sum(examples["token_type_ids"], [])
            tids = tids[:tot]
            res["token_type_ids"] = [tids[i:i+block] for i in range(0, tot, block)]
        return res

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_val   = tokenized_val.map(group_texts,   batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # 5) TrainingArguments（兼容旧 API 用 eval_strategy）
    log_steps = 50
    args_train = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",         # 旧 API
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=log_steps,
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

    # 6) 训练
    trainer.train()

    # 7) 评估（困惑度）
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", None)
    if loss is not None:
        try:
            import math
            ppl = math.exp(loss)
            print(f"Eval loss: {loss:.4f} | Perplexity: {ppl:.2f}")
        except OverflowError:
            print(f"Eval loss: {loss:.4f} | Perplexity: overflow")

    # 8) 保存（模型 + tokenizer + 完整 config）
    trainer.save_model(out_dir)   # 会写出 model.safetensors / config.json
    tok.save_pretrained(out_dir)  # 会写出 tokenizer_config.json / vocab.txt 等

    # 再次确保 config.json 中包含关键键位（避免某些版本 save 覆盖掉）
    cfg_out = out_dir / "config.json"
    try:
        with open(cfg_out, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["model_type"] = "gpt2"
        cfg["architectures"] = ["GPT2LMHeadModel"]
        cfg["vocab_size"]    = len(tok)
        cfg["bos_token_id"]  = tok.cls_token_id
        cfg["eos_token_id"]  = tok.sep_token_id
        cfg["pad_token_id"]  = tok.pad_token_id
        cfg.setdefault("n_ctx", config.n_ctx)
        cfg.setdefault("n_positions", config.n_positions)
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("✓ config.json 已写入 model_type/architectures 与 special token ids")
    except Exception as e:
        print("写回 config.json 失败（可忽略，如果已写入过）：", e)

    # 9) 训练信息
    with open(out_dir / "TRAIN_INFO.txt", "w", encoding="utf-8") as f:
        f.write("Trained from scratch on medical.txt with GPT2LMHeadModel + BertTokenizer.\n")
        f.write(json.dumps({
            "block_size": args.block_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "warmup_ratio": args.warmup_ratio,
        }, ensure_ascii=False, indent=2))
    print("✓ Saved to", out_dir)

if __name__ == "__main__":
    main()
