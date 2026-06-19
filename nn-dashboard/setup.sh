#!/usr/bin/env bash
# Setup sekali jalan: buat venv, install dependency, latih model.
set -e
cd "$(dirname "$0")"

echo "==> Membuat virtual environment (venv) ..."
python3 -m venv venv

echo "==> Mengaktifkan venv & meng-upgrade pip ..."
source venv/bin/activate
pip install --upgrade pip

echo "==> Menginstal dependency (butuh beberapa menit) ..."
pip install -r requirements.txt

echo "==> Melatih semua model ..."
python train.py

echo "==> Mengekspor gambar sampel untuk uji coba ..."
python export_samples.py

echo ""
echo "==> Selesai! Jalankan aplikasi dengan:"
echo "    source venv/bin/activate && streamlit run app.py"
