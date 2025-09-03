#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_full_bpe.py
- 使用 GPT2TokenizerFast（Byte-Level BPE）+ GPT2LMHeadModel 从零训练
- 兼容 transformers==4.56.x（使用 eval_strategy/save_strategy 字段）
- 解决中文“乱码”的根因：分词器与模型统一为 GPT-2 体系（vocab.json + merges.txt）

运行（使用默认路径）：
    python /root/workspace/train_full_bpe.py

或自定义：
    python /root/workspace/train_full_bpe.py \
      --data_txt "/root/workspace/local_med_gpt2/medical.txt" \
      --tok_dir  "/root/workspace/gpt2_bpe_tok" \
      --output_dir "/root/workspace/output/med_full_bpe" \
      --block_size 512 --epochs 2 --lr 2e-4 --batch_size 2 --grad_accum 4
"""

import argparse
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    # === 这里给了默认值，你可以不传参直接运行 ===
    ap.add_argument(
        "--data_txt",
        type=str,
        default="/root/workspace/local_med_gpt2/medical.txt",
        help="训练语料 txt 文件的绝对路径",
    )
    ap.add_argument(
        "--tok_dir",
        type=str,
        default="/root/workspace/gpt2_bpe_tok",
        help="build_tokenizer.py 生成的 GPT-2 分词器目录（含 vocab.json/merges.txt/tokenizer.json）",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/root/workspace/output/med_full_bpe",
        help="训练输出目录（保存 config.json / model.safetensors / tokenizer*）",
    )
    ap.add_argument("--block_size", type=int, default=512, help="拼接切块长度（上下文窗口）")
    ap.add_argument("--epochs", type=int, default=2, help="训练轮数，先 1 跑通再加")
    ap.add_argument("--lr", type=float, default=2e-4, help="学习率")
    ap.add_argument("--batch_size", type=int, default=2, help="每卡 batch 大小（显存不够就降到 1）")
    ap.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数（小显存可增大）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # === 路径准备 ===
    data_txt = Path(args.data_txt)
    tok_dir = Path(args.tok_dir)
    out_dir = Path(args.output_dir)
    if not data_txt.exists():
        raise FileNotFoundError(f"[data_txt] 未找到语料文件：{data_txt}")
    if not tok_dir.exists():
        raise FileNotFoundError(f"[tok_dir] 未找到分词器目录：{tok_dir}（先运行 build_tokenizer.py）")
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1) 加载数据 ===
    print(f"Loading dataset from: {data_txt}")
    # 注意：这里键名只能是“train”，不要用“all”，否则 datasets 会报保留字错误
    raw = load_dataset("text", data_files={"train": str(data_txt)})

    # === 2) 加载 GPT-2 分词器（Byte-Level BPE）===
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tok_dir))
    # 保险起见：补齐特殊 token（GPT-2 常将 pad 映射为 eos）
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
    # 设定一个较大的上限，避免 transformers 的 warning
    tokenizer.model_max_length = max(1024, args.block_size)

    # 分两步：先逐条 tokenize（不截断），再手动拼接切块，避免 token_type_ids & ArrowInvalid
    def tok_fn(batch):
        # 不返回 token_type_ids；GPT-2 不需要它
        return tokenizer(batch["text"], add_special_tokens=True, truncation=False)

    tokenized = raw["train"].map(tok_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        block = args.block_size
        # 把一批样本的 input_ids 拼接起来
        ids = sum(examples["input_ids"], [])
        total_len = (len(ids) // block) * block
        ids = ids[:total_len]
        input_ids = [ids[i:i + block] for i in range(0, total_len, block)]
        attention_mask = [[1] * block for _ in input_ids]
        labels = [x[:] for x in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    lm_dataset = tokenized.map(group_texts, batched=True, desc="Packing into blocks")

    # === 3) GPT-2 配置 & 模型（从零初始化）===
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=768,    # 可按需调整：小=384/512；大=1024
        n_layer=12,    # 可按需调整
        n_head=12,     # 可按需调整（需满足 n_embd % n_head == 0）
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(cfg)
    # 若特殊 token 数量变化，需要同步 embedding 尺寸
    model.resize_token_embeddings(len(tokenizer))

    # === 4) 训练参数（transformers 4.56.x 使用 eval_strategy/save_strategy）===
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,                 # 只训练；要评估可改 True，并额外准备 eval_dataset
        eval_strategy="no",            # 或者 "epoch" 并提供 eval_dataset
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_dir=str(out_dir / "logs"),
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",         # 每个 epoch 保存一次
        save_total_limit=2,
        save_safetensors=True,
        fp16=use_fp16,                 # 4090 可开；不稳定可改 False
        dataloader_pin_memory=True,
        remove_unused_columns=False,   # 我们已经手动准备好 columns，禁用自动删列更稳
        report_to="none",
        seed=args.seed,
    )

    # === 5) Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,  # 自回归 LM，mlm=False；默认 collator 即可
    )

    # === 6) 训练 ===
    trainer.train()

    # === 7) 保存（可被 from_pretrained 直接加载）===
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print("✓ 训练完成，模型与分词器已保存到：", out_dir)

    # === 8) 可选：简单打印困惑度（需要先做一次 eval；这里快速在训练集上评估一下）===
    try:
        metrics = trainer.evaluate(eval_dataset=lm_dataset.select(range(min(2048, len(lm_dataset)))))
        loss = metrics.get("eval_loss")
        if loss is not None:
            ppl = math.exp(loss)
            print(f"Eval (quick) loss: {loss:.4f} | PPL: {ppl:.2f}")
    except Exception as e:
        print("跳过 quick eval：", repr(e))


if __name__ == "__main__":
    main()
