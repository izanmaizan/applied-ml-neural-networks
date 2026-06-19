#!/usr/bin/env bash
# Jalankan aplikasi (asumsi setup.sh sudah dijalankan).
set -e
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py
