# NeuroScope · M1 v3（零基础版）

> 用途声明：本项目仅用于教育与自助筛查演示，**不构成医学诊断或治疗**。如有自/他伤等紧急风险，请立即联系当地急救/危机热线。

## 快速启动
```bash
conda create -n neuroscope python=3.11 -y
conda activate neuroscope
pip install -r requirements.txt
cp .env.example .env       # 打开 .env 填入 OPENAI_API_KEY（如用 OpenAI 模式）
streamlit run app/app.py
```
- 页面右上角 **⚙️** 可在 `openai` / `local` 之间切换
- 本地模式无需 API；若没有你自己的权重，会自动回退到 distilgpt2
