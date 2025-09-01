#!/usr/bin/env bash
set -e
pip install -r requirements.txt
if [ ! -f ".env" ]; then cp .env.example .env; fi
python - <<'PY'
from pathlib import Path
p=Path(".env")
txt=p.read_text(encoding="utf-8")
txt=txt.replace("NS_ENGINE=openai","NS_ENGINE=local")
p.write_text(txt,encoding="utf-8")
print("已将 NS_ENGINE 设为 local")
PY
streamlit run app/app.py
