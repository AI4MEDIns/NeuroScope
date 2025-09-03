# app_wp.py
# Streamlit 聊天应用：本地(WordPiece) / OpenAI 双引擎，带 [CLS]/[SEP]/“医生：” 拼提示
import os
import json
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# ---------- 本地 LLM 依赖 ----------
import torch
from transformers import AutoModelForCausalLM, BertTokenizer

# ---------- OpenAI 可选 ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE = "NeuroScope · WordPiece 对话（本地/OpenAI）"
DEFAULT_LOCAL_DIR = os.environ.get("NS_CKPT", "")  # 你的本地模型目录（可在侧边栏改）
DEFAULT_ENGINE = os.environ.get("NS_ENGINE", "local")  # "local" or "openai"
DEFAULT_OPENAI_MODEL = os.environ.get("NS_OPENAI_MODEL", "gpt-4o-mini")

# 角色名（界面显示，不影响本地拼提示的 token）
ROLE_USER = os.environ.get("NS_ROLE_USER", "患者")
ROLE_ASSISTANT = os.environ.get("NS_ROLE_ASSISTANT", "医生")

# -------------------- UI 基础 --------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)
st.caption("⚠️ 本应用仅用于**心理/情绪支持**与**科普**，**不构成医疗诊断或治疗建议**。如症状明显/持续/加重，请及时就医。")

# -------------------- 缓存加载本地模型 --------------------
@st.cache_resource(show_spinner=True)
def load_local_model(model_dir: str):
    """
    加载 WordPiece 版本地模型：
      - 使用 BertTokenizer(vocab.txt)
      - 确保 pad/sep/cls 特殊符号存在
      - 将 eos 对齐为 [SEP]，pad 对齐为 [PAD]
      - resize_token_embeddings 避免 vocab size 不匹配
    """
    if not model_dir:
        raise ValueError("未提供本地模型目录。")

    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"模型目录不存在：{p}")

    # 加载 tokenizer
    tok = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
    # 兜底：确保特殊 token 存在
    add_map = {}
    if tok.pad_token is None:
        add_map["pad_token"] = "[PAD]"
    if tok.sep_token is None:
        add_map["sep_token"] = "[SEP]"
    if tok.cls_token is None:
        add_map["cls_token"] = "[CLS]"
    if add_map:
        tok.add_special_tokens(add_map)

    # 加载模型
    mdl = AutoModelForCausalLM.from_pretrained(model_dir)
    mdl.resize_token_embeddings(len(tok))

    # 同步生成配置（特别是 eos/pad）
    if mdl.config.eos_token_id is None:
        mdl.config.eos_token_id = tok.sep_token_id
    if mdl.generation_config.eos_token_id is None:
        mdl.generation_config.eos_token_id = tok.sep_token_id
    if mdl.config.pad_token_id is None:
        mdl.config.pad_token_id = tok.pad_token_id
    if mdl.generation_config.pad_token_id is None:
        mdl.generation_config.pad_token_id = tok.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device

# -------------------- 拼提示 --------------------
def build_prompt_wp(
    tok: BertTokenizer,
    history: List[Tuple[str, str]],
    user_text: str,
    sys_hint: str = "你是一位温和、共情的心理健康支持助手，提供安抚与就医建议，避免诊断与处方。"
):
    """
    将多轮对话拼接为：
      [CLS] <sys_hint> [SEP]
      患者：xxx [SEP] 医生：yyy [SEP]
      患者：zzz [SEP] 医生：... [SEP]
      患者：<当前输入> [SEP] 医生：
    注意：真正送入模型的是 token id，CLS/SEP 为 special token。
    """
    ids: List[int] = []

    # 1) System 提示
    ids += [tok.cls_token_id] + tok.encode(sys_hint, add_special_tokens=False) + [tok.sep_token_id]

    # 2) 历史轮次
    for u, a in history:
        ids += tok.encode(f"{ROLE_USER}：{u}", add_special_tokens=False) + [tok.sep_token_id]
        ids += tok.encode(f"{ROLE_ASSISTANT}：{a}", add_special_tokens=False) + [tok.sep_token_id]

    # 3) 当前轮输入 + 回答提示
    ids += tok.encode(f"{ROLE_USER}：{user_text}", add_special_tokens=False) + [tok.sep_token_id]
    ids += tok.encode(f"{ROLE_ASSISTANT}：", add_special_tokens=False)

    return ids

# -------------------- 本地推理 --------------------
@torch.no_grad()
def generate_local(
    tok: BertTokenizer,
    mdl: AutoModelForCausalLM,
    device: str,
    history: List[Tuple[str, str]],
    user_text: str,
    max_new_tokens: int = 200,
    top_p: float = 0.9,
    temperature: float = 0.7,
):
    input_ids = build_prompt_wp(tok, history, user_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    out = mdl.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tok.sep_token_id,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # 抽取最后一轮“医生：”之后的内容作为回答
    # 防止把历史也一起显示出来
    spl = text.rsplit(f"{ROLE_ASSISTANT}：", 1)
    answer = spl[-1] if len(spl) > 1 else text
    # 若生成末尾又跟了“患者：”，截断在此之前
    cut = answer.find(f"{ROLE_USER}：")
    if cut != -1:
        answer = answer[:cut]
    return answer.strip()

# -------------------- OpenAI 推理（可选） --------------------
def generate_openai(
    api_key: str,
    history: List[Tuple[str, str]],
    user_text: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = 400,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，请先 `pip install openai`。")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY。")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": "你是一位温和、共情的心理健康支持助手，提供安抚与就医建议，避免诊断与处方。"},
    ]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_text})

    # 使用 Chat Completions（兼容范围广）
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# -------------------- 会话状态 --------------------
if "chat" not in st.session_state:
    st.session_state.chat: List[Tuple[str, str]] = []  # [(user, assistant), ...]

# -------------------- 侧边栏 --------------------
with st.sidebar:
    st.header("引擎与参数")
    engine = st.radio("选择引擎", options=["local", "openai"], index=0 if DEFAULT_ENGINE=="local" else 1)
    max_new_tokens = st.slider("max_new_tokens（本地）", 32, 1024, 256, step=16)
    temperature = st.slider("temperature", 0.0, 1.5, 0.7, step=0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, step=0.05)

    st.markdown("---")
    st.subheader("本地模型设置")
    local_dir = st.text_input("本地模型目录", value=DEFAULT_LOCAL_DIR, placeholder="/path/to/your_wp_model_dir")
    local_status = st.empty()

    st.markdown("---")
    st.subheader("OpenAI 设置（可选）")
    openai_key = st.text_input("OPENAI_API_KEY", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    openai_model = st.text_input("OpenAI 模型名", value=DEFAULT_OPENAI_MODEL)

    st.markdown("---")
    if st.button("清空会话"):
        st.session_state.chat.clear()
        st.rerun()

# -------------------- 主对话区 --------------------
for u, a in st.session_state.chat:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(u)
    with st.chat_message("assistant", avatar="🩺"):
        st.markdown(a)

user_input = st.chat_input(f"{ROLE_USER}输入…")
if user_input:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # 推理
    try:
        if engine == "local":
            # 加载本地（带缓存）
            tok, mdl, device = load_local_model(local_dir)
            local_status.info(f"已加载本地模型：{local_dir}（设备：{device}）")
            ans = generate_local(
                tok=tok, mdl=mdl, device=device,
                history=st.session_state.chat,
                user_text=user_input,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            ans = generate_openai(
                api_key=openai_key,
                history=st.session_state.chat,
                user_text=user_input,
                model=openai_model,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    except Exception as e:
        ans = f"❌ 出错：{e}"

    st.session_state.chat.append((user_input, ans))
    with st.chat_message("assistant", avatar="🩺"):
        st.markdown(ans)
