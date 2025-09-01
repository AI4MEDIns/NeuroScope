# app/app.py
# -*- coding: utf-8 -*-
"""
NeuroScope · 仅 OpenAI / 自训练本地模型 双模（无任何 distilgpt2 回退）
- .env 鲁棒读取（find_dotenv + usecwd）
- local 引擎只加载 NS_CKPT；为空/无效即报错，不回退
- 词表一致性校验，防止乱码
- 显式设置 pad_token_id
- OpenAI 路径含额度/认证/模型不存在等友好报错
- 危机词识别 + 免责声明
"""
from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# 读取 .env（无论从哪里启动都能找到）
load_dotenv(find_dotenv(usecwd=True), override=True)

# ========= 环境变量 =========
DEFAULT_ENGINE   = os.getenv("NS_ENGINE", "openai").lower()   # openai / local
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY       = os.getenv("OPENAI_API_KEY", "")
LOCAL_CKPT       = os.getenv("NS_CKPT", "").strip()           # 仅当你已训练并“合并”后的完整权重目录
# 可选：自定义本地提示标签（避免小模型被中文标签干扰）
ROLE_USER        = os.getenv("NS_ROLE_USER", "User")
ROLE_ASSISTANT   = os.getenv("NS_ROLE_ASSISTANT", "Assistant")

# ========= 页面配置 =========
st.set_page_config(page_title="NeuroScope · OpenAI/自训练本地", page_icon="🧠", layout="centered")
st.title("🧠 NeuroScope · 精神健康问答（OpenAI / 自训练本地）")
st.caption("教育与自助筛查，不替代专业诊疗。遇到自/他伤等紧急风险请立刻联系当地急救/危机热线。")

# ========= UI：引擎切换 =========
with st.expander("⚙️ 设置（本页临时切换）", expanded=False):
    engine_choice = st.radio("推理引擎", ["openai", "local"], index=0 if DEFAULT_ENGINE=="openai" else 1, horizontal=True)
    st.write(f"OpenAI 模型：`{OPENAI_MODEL}`")
    st.write(f"本地权重 NS_CKPT：`{LOCAL_CKPT or '（未设置）'}`")
ENGINE = engine_choice

# ========= 安全策略 =========
CRISIS_TERMS = [
    "自杀","轻生","结束生命","活不下去","割腕","上吊","跳楼","杀了我",
    "想死","自残","suicide","kill myself","end my life","hurt myself"
]
CRISIS_RE = re.compile("|".join(map(re.escape, CRISIS_TERMS)), re.I)

FORBID_CLAIMS = ["诊断为","处方","药量","治愈","保证痊愈","确保痊愈"]
DISCLAIMER = "（提醒：此对话不构成医学诊断或治疗；如症状明显/持续/加重，请尽快寻求线下专业帮助。）"

def crisis_hit(text: str) -> bool:
    return bool(text and CRISIS_RE.search(text))

def soft_sanitize(text: str) -> str:
    # 软替换承诺性医疗用语
    for w in FORBID_CLAIMS:
        text = text.replace(w, "（建议线下面诊评估）")
    return text

# ========= 本地模型加载（仅 NS_CKPT；无效即报错） =========
@st.cache_resource(show_spinner=False)
def load_local_model() -> Tuple[Any, Any, str]:
    """
    只加载 NS_CKPT 指定的“合并后完整权重”目录（需包含 config.json / model.safetensors / tokenizer.json 等）
    不做任何回退。
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not LOCAL_CKPT:
        raise ValueError("NS_CKPT 未设置。请在 .env 中设置你的本地权重目录（合并后的完整模型）。")
    tok = AutoTokenizer.from_pretrained(LOCAL_CKPT, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(LOCAL_CKPT)
    # 词表一致性校验
    if mdl.get_input_embeddings().num_embeddings != len(tok):
        raise ValueError(f"Tokenizer/Model vocab mismatch: {len(tok)} vs "
                         f"{mdl.get_input_embeddings().num_embeddings}（请确认 tokenizer 与模型来自同一目录）")
    # 显式 pad
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if mdl.config.pad_token_id is None:
        mdl.config.pad_token_id = tok.eos_token_id
    name = getattr(mdl.config, "_name_or_path", str(LOCAL_CKPT)) or str(LOCAL_CKPT)
    return tok, mdl, name

# 尝试加载一次以在 UI 显示“当前本地模型”
if ENGINE == "local":
    try:
        _tok_info, _mdl_info, _local_name = load_local_model()
        st.caption(f"当前本地模型：`{_local_name}`")
    except Exception as _err:
        st.error(f"本地模型不可用：{_err}")

# ========= OpenAI 路径 =========
def llm_openai(history: List[Dict[str,str]], user_text: str) -> Dict[str, Any]:
    if not (OPENAI_KEY or os.getenv("OPENAI_API_KEY")):
        return {"message": "未检测到 OPENAI_API_KEY。请在项目根目录的 .env 中配置，或切换到 local 引擎。",
                "followups": [], "risk_level": "none"}

    try:
        from openai import OpenAI
        from openai import RateLimitError, AuthenticationError, NotFoundError, APIStatusError, OpenAIError
    except Exception:
        return {"message": "未安装或无法导入 openai SDK。请在环境中执行：`pip install -U openai`。",
                "followups": [], "risk_level": "none"}

    client = OpenAI()

    # 输入审核（失败不阻断）
    try:
        mod = client.moderations.create(model="omni-moderation-latest", input=user_text)
        if mod.results and mod.results[0].flagged:
            return {"message": "你的输入触发了内容安全审核。如有紧急风险请立即联系当地急救或危机热线。",
                    "followups": [], "risk_level": "high" if crisis_hit(user_text) else "medium"}
    except Exception:
        pass

    SYSTEM_PROMPT = (
        "You are a supportive mental health triage assistant for adults.\n"
        "- You are NOT a doctor and do NOT provide diagnosis or prescriptions.\n"
        "- Warm, validating, stigma-free, and succinct.\n"
        "- If imminent danger: urge immediate local emergency help and crisis hotlines.\n"
        "- Ask 1-3 targeted follow-up questions.\n"
        "- Always end with next steps and disclaimer.\n"
        "Return JSON strictly following the provided schema."
    )
    TRIAGE_SCHEMA = {
        "name": "triage_schema",
        "schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "followups": {"type": "array", "items": {"type": "string"}},
                "risk_level": {"type": "string", "enum": ["none","low","medium","high"]}
            },
            "required": ["message","followups","risk_level"],
            "additionalProperties": False
        },
        "strict": True
    }

    msgs = [{"role":"system","content":SYSTEM_PROMPT}, *history, {"role":"user","content":user_text}]

    def _post_json():
        return client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            response_format={"type":"json_schema","json_schema":TRIAGE_SCHEMA},
            temperature=0.3
        )

    def _post_text():
        return client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.3
        )

    try:
        try:
            r = _post_json()
            data = json.loads(r.choices[0].message.content)
        except Exception:
            r = _post_text()
            txt = r.choices[0].message.content
            data = {"message": txt, "followups": [], "risk_level": "none"}

        # 输出审核（失败不阻断）
        try:
            mod2 = client.moderations.create(model="omni-moderation-latest", input=data.get("message",""))
            if mod2.results and mod2.results[0].flagged:
                data["message"] = "抱歉，我的回复被安全系统拦截。请换一种描述方式，或尽快寻求线下专业帮助。"
                data["followups"] = []
                data["risk_level"] = "low"
        except Exception:
            pass

        if crisis_hit(user_text) or crisis_hit(data.get("message","")):
            data["risk_level"] = "high"

        data["message"] = soft_sanitize(data.get("message","")).rstrip() + "\n\n" + DISCLAIMER
        return data

    except RateLimitError:
        return {"message": "你的 OpenAI 账户/项目当前没有可用额度（insufficient_quota / 429）。请到 Billing 充值或提高 Project 的月度预算，然后再试。",
                "followups": [], "risk_level": "none"}
    except AuthenticationError:
        return {"message": "OpenAI 认证失败：API Key 可能无效或没有权限。请检查 .env 的 OPENAI_API_KEY。",
                "followups": [], "risk_level": "none"}
    except NotFoundError:
        return {"message": f"指定模型 `{OPENAI_MODEL}` 不存在或无权限。请在 .env 改为 gpt-4o 或 gpt-4o-mini。",
                "followups": [], "risk_level": "none"}
    except APIStatusError as e:
        return {"message": f"OpenAI 服务暂时不可用（{getattr(e, 'status_code', '5xx')}）。请稍后再试。",
                "followups": [], "risk_level": "none"}
    except OpenAIError as e:
        return {"message": f"OpenAI 调用失败：{e}", "followups": [], "risk_level": "none"}
    except Exception as e:
        return {"message": f"调用异常：{e}", "followups": [], "risk_level": "none"}

# ========= 本地路径（只用你训练的模型） =========
def history_to_prompt(history: List[Dict[str,str]], user_text: str) -> str:
    head = ""
    for m in history:
        role = ROLE_USER if m["role"] == "user" else ROLE_ASSISTANT
        head += f"{role}: {m['content']}\n"
    head += f"{ROLE_USER}: {user_text}\n{ROLE_ASSISTANT}: "
    return head

def llm_local(history: List[Dict[str,str]], user_text: str) -> Dict[str, Any]:
    import torch
    try:
        tok, mdl, name = load_local_model()
    except Exception as e:
        return {"message": f"本地模型不可用：{e}\n\n请在 `.env` 设置 NS_CKPT 为你合并后的完整权重目录，然后重启应用。",
                "followups": [], "risk_level": "none", "local_name": None}

    inputs = tok(history_to_prompt(history, user_text), return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True, top_p=0.9, temperature=0.7,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = soft_sanitize(gen).rstrip() + "\n\n" + DISCLAIMER
    risk = "high" if crisis_hit(user_text) or crisis_hit(gen) else "none"
    followups = [
        "这些困扰持续了多久？对日常学习/工作影响多大？",
        "最近睡眠、食欲、精力、注意力有没有明显变化？",
        "你身边有没有能提供支持的家人/朋友/老师？"
    ]
    return {"message": gen, "followups": followups, "risk_level": risk, "local_name": name}

# ========= 会话状态 =========
if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str,str]] = []

# 历史
for m in st.session_state.chat:
    st.chat_message(m["role"]).markdown(m["content"])

# 输入框
user_text = st.chat_input("和我聊聊你现在最困扰的事情…（此工具不提供诊断/处方）")

# ========= 主流程 =========
if user_text:
    st.session_state.chat.append({"role":"user","content":user_text})
    st.chat_message("user").markdown(user_text)

    if crisis_hit(user_text):
        st.warning("⚠️ 你的描述可能涉及自/他伤等紧急风险。请**立即**联系当地急救或危机热线，并寻求身边可信赖的人的陪伴与专业帮助。")

    if ENGINE == "openai":
        data = llm_openai(st.session_state.chat[:-1], user_text)
    else:
        data = llm_local(st.session_state.chat[:-1], user_text)

    badge = {"none":"🟢","low":"🟡","medium":"🟠","high":"🔴"}.get(data.get("risk_level","none"), "🟢")
    reply = f"{badge} **风险评估：{data.get('risk_level','none')}**\n\n" + data.get("message","（无内容）")

    # 显示本地模型名（仅当 local 且已加载）
    if ENGINE == "local" and data.get("local_name"):
        reply = f"（本地模型：`{data['local_name']}`）\n\n" + reply

    st.session_state.chat.append({"role":"assistant","content":reply})
    st.chat_message("assistant").markdown(reply)

    if data.get("followups"):
        st.info("**进一步了解的问题：**\n- " + "\n- ".join(data["followups"]))

# ========= 侧边栏说明 =========
with st.sidebar:
    st.markdown("### 使用说明")
    st.markdown("- 顶部可在 `openai` / `local` 间切换。")
    st.markdown("- `openai` 需在 `.env` 配置 `OPENAI_API_KEY`，推荐模型 `gpt-4o-mini`。")
    st.markdown("- `local` 仅在 `.env` 的 `NS_CKPT` 指向你**合并后的完整权重目录**时可用；否则会报错提示。")
    st.markdown("- 可用 `NS_ROLE_USER` / `NS_ROLE_ASSISTANT` 自定义对话标签（默认英文以提高兼容性）。")
    st.markdown("---")
    st.markdown("**安全与合规**：本系统不提供诊断/处方；如遇紧急风险请立即联系当地急救/危机热线。")
