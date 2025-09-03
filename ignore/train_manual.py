# -*- coding: utf-8 -*-
"""
按“Dataset -> DataLoader(batch) -> collate_fn -> 训练/验证循环 -> 保存最佳模型”的流程，
用 GPT2LMHeadModel + BertTokenizerFast 在你的 medical 语料上从零训练。

需要文件：
- /root/workspace/local_med_gpt2/config.json
- /root/workspace/local_med_gpt2/vocab.txt
- /root/workspace/local_med_gpt2/train.txt
- /root/workspace/local_med_gpt2/valid.txt
"""

import os, math, json, time, argparse, pathlib, random
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BertTokenizerFast,
    get_cosine_schedule_with_warmup,
)

# --------------- 1) 数据集 -----------------
class TextLineDataset(Dataset):
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.lines = [ln.strip() for ln in self.path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        return self.lines[idx]

# --------------- 2) 批处理 & 填充 -----------------
@dataclass
class CollateFn:
    tokenizer: BertTokenizerFast
    max_len: int

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        # 将文本转 id： [CLS] + text + [SEP]
        ids_list = []
        for text in batch:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            # 截断：为 [CLS]/[SEP] 留两个位置
            ids = ids[: self.max_len - 2]
            ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
            ids_list.append(torch.tensor(ids, dtype=torch.long))

        # 动态 padding 到本批最大长度
        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = input_ids.clone()  # 自回归 LM：标签即输入

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --------------- 3) 评估函数 -----------------
@torch.no_grad()
def evaluate(model, dataloader, device) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        out = model(**batch)
        total += out.loss.item()
        steps += 1
    return total / max(1, steps)

# --------------- 4) 训练主程序 -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/workspace/local_med_gpt2", type=str)
    ap.add_argument("--output_dir", default="/root/workspace/output/med_from_scratch_manual", type=str)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)           # 你的 4090 可先试 16；不够就降
    ap.add_argument("--max_len", type=int, default=256)             # 先 256，训练快，显存稳
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir  = pathlib.Path(args.data_dir)
    output_dir= pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer（用 vocab.txt；按截图里 BertTokenizerFast）
    vocab = str((data_dir/"vocab.txt").resolve())
    tok = BertTokenizerFast(vocab_file=vocab, do_lower_case=False)
    # 对齐特殊符号
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.sep_token is None: tok.add_special_tokens({"sep_token": "[SEP]"})
    if tok.cls_token is None: tok.add_special_tokens({"cls_token": "[CLS]"})

    # --- Config（读取你的 config.json），并写入 special token ids
    config = AutoConfig.from_pretrained(str(data_dir))
    config.vocab_size   = len(tok)
    config.pad_token_id = tok.pad_token_id
    config.bos_token_id = tok.cls_token_id
    config.eos_token_id = tok.sep_token_id
    # model_type / architectures（保证后续可被正确识别）
    config.model_type   = config.model_type or "gpt2"
    if not getattr(config, "architectures", None):
        config.architectures = ["GPT2LMHeadModel"]

    # --- 模型（从配置随机初始化）
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tok))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)   # 简单多卡

    # --- 数据
    train_set = TextLineDataset(str(data_dir/"train.txt"))
    valid_set = TextLineDataset(str(data_dir/"valid.txt"))

    collate = CollateFn(tokenizer=tok, max_len=args.max_len)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    # --- 优化器与调度器
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(train_loader)/args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # --- 训练循环（按你的截图：遍历 epoch、遍历 batch、前向、计算损失、反传、更新）
    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        t0 = time.time()
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += loss.item()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            if step % 100 == 0:
                print(f"[Epoch {epoch}] step {step}/{len(train_loader)}  lr={sched.get_last_lr()[0]:.3e}  loss={running/100:.4f}")
                running = 0.0

        # --- 验证（按截图 evaluate）
        val_loss = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch} done. valid loss = {val_loss:.4f} | time = {time.time()-t0:.1f}s")

        # 保存最优
        if val_loss < best_val:
            best_val = val_loss
            save_dir = output_dir
            if isinstance(model, torch.nn.DataParallel):
                model.module.save_pretrained(save_dir)
            else:
                model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
            # 写入一个标准的 special_tokens_map.json（避免前端乱码）
            (save_dir/"special_tokens_map.json").write_text(json.dumps({
                "bos_token": tok.cls_token, "eos_token": tok.sep_token, "pad_token": tok.pad_token
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            # 同步生成 config 中的 special token ids（保守再写一次）
            cfg_path = save_dir/"config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["model_type"] = "gpt2"
            cfg.setdefault("architectures", ["GPT2LMHeadModel"])
            cfg["pad_token_id"] = tok.pad_token_id
            cfg["bos_token_id"] = tok.cls_token_id
            cfg["eos_token_id"] = tok.sep_token_id
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

            # 训练信息
            (save_dir/"TRAIN_INFO.txt").write_text(
                json.dumps({
                    "epochs": args.epochs, "max_len": args.max_len,
                    "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                    "best_valid_loss": best_val
                }, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"✓ Best checkpoint saved to: {save_dir}")

    print("All done.")

if __name__ == "__main__":
    main()
