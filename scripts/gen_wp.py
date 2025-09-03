# gen_wp.py
import torch
from transformers import AutoModelForCausalLM, BertTokenizerFast

CKPT = "/root/workspace/output/med_full_wp"  # 改成你的输出路径

tok = BertTokenizerFast.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT).cuda().eval()

def chat_turn(user_text: str):
    # 构造 [CLS] 患者... [SEP] 医生：
    prompt_ids = [tok.cls_token_id] \
               + tok.encode(user_text, add_special_tokens=False) \
               + [tok.sep_token_id] \
               + tok.encode("医生：", add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids]).cuda()
    out = model.generate(
        input_ids,
        max_new_tokens=120,
        do_sample=True, top_p=0.9, temperature=0.7,
        eos_token_id=tok.sep_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

print(chat_turn("患者主诉：最近两周情绪低落、失眠。"))
