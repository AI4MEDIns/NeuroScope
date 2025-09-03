# build_tokenizer.py
# 训练 GPT-2 风格的 Byte-Level BPE 分词器，并导出为可被 GPT2TokenizerFast 直接加载的格式

import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

DATA_TXT = "/root/workspace/local_med_gpt2/medical.txt"
TOK_OUT   = "/root/workspace/gpt2_bpe_tok"

Path(TOK_OUT).mkdir(parents=True, exist_ok=True)

# 1) 用 HuggingFace tokenizers 训练 BPE（byte-level，覆盖任意字符，中文OK，不会有UNK）
tok = ByteLevelBPETokenizer(lowercase=False)
tok.train(files=DATA_TXT, vocab_size=30000, min_frequency=2,
          special_tokens=["<|endoftext|>", "<|pad|>"])

# 2) 保存 vocab.json / merges.txt
tok.save_model(TOK_OUT)

# 3) 用 transformers 包装成 GPT2TokenizerFast，并设置特殊token映射，然后保存为标准目录
fast_tok = GPT2TokenizerFast.from_pretrained(TOK_OUT)
# 对齐 GPT-2 习惯：把 pad 映射到 eos（避免生成时警告）
if fast_tok.pad_token is None:
    fast_tok.add_special_tokens({"pad_token": "<|pad|>"})
if fast_tok.eos_token is None:
    fast_tok.add_special_tokens({"eos_token": "<|endoftext|>"})
# 也可把 bos 设置为 eos，简化自回归生成
if fast_tok.bos_token is None:
    fast_tok.add_special_tokens({"bos_token": "<|endoftext|>"})

fast_tok.model_max_length = 1024
fast_tok.save_pretrained(TOK_OUT)

print("✅ GPT-2 分词器已生成于：", TOK_OUT)
print("vocab size =", len(fast_tok), " | eos_id=", fast_tok.eos_token_id, " | pad_id=", fast_tok.pad_token_id)
