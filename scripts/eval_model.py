import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = "checkpoints/neuroscope/final"
mdl = AutoModelForCausalLM.from_pretrained(CKPT)
tok = AutoTokenizer.from_pretrained(CKPT)

def ppl_on_texts(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        loss = mdl(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss))

print("Val PPL 示例：", ppl_on_texts(["用户: 我最近很焦虑，睡不好。\n助手: "]))
print("✓ 评估完成（建议结合人工打分表）")