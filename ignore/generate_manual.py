# -*- coding: utf-8 -*-
import torch, math
from transformers import AutoModelForCausalLM, BertTokenizerFast

CKPT = "/root/workspace/output/med_from_scratch_manual"

tok = BertTokenizerFast.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def generate(prompt, max_new_tokens=80, temperature=0.8, top_p=0.9):
    # 手写按步生成（和你截图一致）
    ids = tok.encode(prompt, add_special_tokens=False)
    ids = [tok.cls_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=ids)
            logits = out.logits[:, -1, :] / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > top_p).float().argmax(dim=-1)
            mask = torch.arange(probs.shape[-1], device=probs.device)[None, :] > cutoff[:, None]
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(sorted_probs, num_samples=1)
            next_token_id = sorted_idx.gather(-1, next_id)

            ids = torch.cat([ids, next_token_id], dim=-1)
            if next_token_id.item() == tok.sep_token_id:
                break

    text = tok.decode(ids[0].tolist()[1:], skip_special_tokens=False)
    return text

if __name__ == "__main__":
    print(generate("患者：近两周情绪低落、失眠。医生："))
