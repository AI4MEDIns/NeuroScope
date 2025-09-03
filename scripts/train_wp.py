# train_wp.py
# WordPiece 版全量微调：BertTokenizer(+vocab.txt) + GPT2LMHeadModel(+config.json)
# 关键点：
#  - 特殊符号对齐：BOS=[CLS], EOS=[SEP], PAD=[PAD]
#  - 不生成 token_type_ids，loss 里忽略 PAD（-100）
#  - 单卡稳定训练，避免 NCCL 多卡坑；需要更快可以后再改分布式/AMP/ZeRO

import os
import math
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

IGNORE_INDEX = -100

def build_tokenizer(data_dir: Path):
    vocab = data_dir / "vocab.txt"
    if not vocab.exists():
        raise FileNotFoundError(f"缺少词表：{vocab}")
    tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    # 确保三大特殊 token 存在
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.sep_token is None:
        tok.add_special_tokens({"sep_token": "[SEP]"})
    if tok.cls_token is None:
        tok.add_special_tokens({"cls_token": "[CLS]"})
    return tok

class LineDataset(Dataset):
    def __init__(self, txt_path: Path, tok: BertTokenizerFast, max_len: int):
        if not txt_path.exists():
            raise FileNotFoundError(f"找不到语料：{txt_path}")
        self.tok = tok
        self.max_len = max_len
        with open(txt_path, "r", encoding="utf-8") as f:
            # 过滤空行
            self.lines = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(self.lines)} lines from: {txt_path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        text = self.lines[i]
        # add_special_tokens=True 会自动加 [CLS] ... [SEP]
        ids = self.tok.encode(text, add_special_tokens=True,
                              truncation=True, max_length=self.max_len)
        return ids

def collate_fn(batch_ids, pad_id: int):
    max_len = max(len(x) for x in batch_ids)
    input_ids, attention_mask, labels = [], [], []
    for ids in batch_ids:
        pad_len = max_len - len(ids)
        x = ids + [pad_id] * pad_len
        m = [1]*len(ids) + [0]*pad_len
        y = x.copy()
        # 忽略 padding 的 loss
        for j in range(len(ids), max_len):
            y[j] = IGNORE_INDEX
        input_ids.append(x)
        attention_mask.append(m)
        labels.append(y)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    required=True, help="包含 config.json / vocab.txt / medical.txt 的目录")
    ap.add_argument("--output_dir",  required=True, help="输出目录")
    ap.add_argument("--epochs",      type=int, default=1)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--max_len",     type=int, default=256)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--warmup_ratio",type=float, default=0.03)
    ap.add_argument("--grad_accum",  type=int, default=2)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--eval_ratio",  type=float, default=0.02, help="验证集比例（从末尾切）")
    ap.add_argument("--grad_ckpt",   action="store_true", help="开启梯度检查点以省显存")
    return ap.parse_args()

def set_seed(s):
    import random, numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) tokenizer & config & model
    tok = build_tokenizer(data_dir)
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置：{cfg_path}")
    config = AutoConfig.from_pretrained(str(data_dir))
    config.vocab_size   = len(tok)
    config.eos_token_id = tok.sep_token_id   # EOS 对齐 [SEP]
    config.bos_token_id = tok.cls_token_id   # BOS 对齐 [CLS]
    config.pad_token_id = tok.pad_token_id

    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tok))
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)

    # 2) dataset split
    txt_path = data_dir / "medical.txt"
    all_lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        all_lines = [ln.strip() for ln in f if ln.strip()]
    n_total = len(all_lines)
    n_eval  = max(1, int(n_total * args.eval_ratio))
    n_train = n_total - n_eval
    train_lines = all_lines[:n_train]
    eval_lines  = all_lines[n_train:]

    # 临时写到内存对象，不落盘
    class _MemDataset(LineDataset):
        def __init__(self, lines, tok, max_len):
            self.lines = lines
            self.tok = tok
            self.max_len = max_len

    ds_train = _MemDataset(train_lines, tok, args.max_len)
    ds_eval  = _MemDataset(eval_lines, tok, args.max_len)

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
        num_workers=2, pin_memory=True,
    )
    dl_eval = DataLoader(
        ds_eval, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
        num_workers=2, pin_memory=True,
    )

    # 3) 优化器 & 调度器
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(dl_train) * args.epochs + args.grad_accum - 1) // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 4) 训练
    global_step = 0
    model.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}")
        optim.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(pbar):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix({"loss": f"{running:.4f}"})
                running = 0.0

        # 每个 epoch 评估一次 ppl
        ppl = evaluate_ppl(model, dl_eval, device)
        print(f"[Eval] epoch={ep} perplexity={ppl:.2f}")

    # 5) 保存（确保前端可识别）
    model.config.model_type = "gpt2"
    model.config.architectures = ["GPT2LMHeadModel"]
    model.save_pretrained(out_dir, safe_serialization=True)  # model.safetensors + config.json
    tok.save_pretrained(out_dir)                              # tokenizer_config.json + vocab.txt
    with open(out_dir / "TRAIN_INFO.txt", "w", encoding="utf-8") as f:
        json.dump({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "warmup_ratio": args.warmup_ratio
        }, f, ensure_ascii=False, indent=2)
    print("✓ saved to", out_dir)

@torch.no_grad()
def evaluate_ppl(model, dl_eval, device):
    model.eval()
    losses = []
    for batch in dl_eval:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(out.loss.item())
    model.train()
    mean_loss = sum(losses)/max(1, len(losses))
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")

if __name__ == "__main__":
    main()
