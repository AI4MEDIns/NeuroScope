#!/usr/bin/env bash
set -e
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "已生成 .env（请编辑填入 OPENAI_API_KEY）"
fi
pip install -r requirements.txt
streamlit run app/app.py
