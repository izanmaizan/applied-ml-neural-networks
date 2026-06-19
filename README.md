# Applied Machine Learning — Neural Networks (Chapter 9)

Tugas Neural Networks: sebuah **notebook** pembelajaran + **aplikasi web (Streamlit)** untuk mencoba 5 model neural network pada **data nyata**.

**Kelompok:** Maizan Insani Akbar (242321011) · Muhammad Dicky Imansyah (242321030)

| Komponen | Isi |
|----------|-----|
| `NeuralNetworks_Chapter9_AML.ipynb` | Notebook utama (regresi, klasifikasi biner/multikelas, dropout, callbacks) |
| `PENJELASAN.md` | Penjelasan tiap section **+ cara membaca setiap grafik/gambar** |
| `nn-dashboard/` | Aplikasi web Streamlit (5 model: taksi, diabetes, penipuan, digit, wajah) |

5 model: 🚕 Tarif Taksi · ❤️ Risiko Diabetes · 🛡️ Deteksi Penipuan · 🔢 Pengenal Digit 0–9 · 👤 Pengenalan Wajah.

---

## A. Menjalankan Notebook

### Di Google Colab (paling mudah)
1. Buka [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → pilih `NeuralNetworks_Chapter9_AML.ipynb`
2. **Runtime → Run all**. Dataset terunduh otomatis; tidak perlu instal apa pun.

### Di Jupyter lokal (laptop)
1. Jalankan **sel Section 0** (paling atas) untuk memasang dependency:
   ```
   %pip install -q numpy pandas matplotlib seaborn scikit-learn tensorflow keras ipywidgets
   ```
2. **Restart kernel**, lalu **Run All**.

---

## B. Menjalankan Aplikasi Web (nn-dashboard)

Butuh **Python 3.10+**. Semua perintah dijalankan dari folder repo.

### Cara cepat (otomatis)
```bash
cd nn-dashboard
bash setup.sh        # buat venv + install dependency + latih model + ekspor sampel
bash run.sh          # jalankan aplikasi di browser
```

### Cara manual (langkah demi langkah)
```bash
cd nn-dashboard

# 1) Buat virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2) Install semua dependency
pip install -r requirements.txt

# 3) Latih semua model (sekali saja; output tampil langsung)
python train.py

# 4) (Opsional) Ekspor gambar sampel & lihat metrik
python export_samples.py
python eval_models.py

# 5) Jalankan aplikasi → otomatis terbuka di http://localhost:8501
streamlit run app.py
```

> **Catatan dataset:** `creditcard.csv` (±112 MB) **tidak disertakan** di repo (melebihi batas GitHub). File ini **diunduh otomatis** oleh `train.py`/`eval_models.py`/notebook saat dibutuhkan. Dataset `taxi-fares.csv` sudah ikut di repo.

---

## C. Sumber Dataset

| Dataset | Sumber |
|---------|--------|
| 🚕 NYC Taxi Fares | [Kaggle](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) |
| ❤️ Pima Indians Diabetes | [Kaggle/UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| 🛡️ Credit Card Fraud | [Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| 🔢 MNIST | bawaan Keras · [situs asli](http://yann.lecun.com/exdb/mnist/) |
| 👤 LFW (wajah) | bawaan scikit-learn · [situs asli](https://vis-www.cs.umass.edu/lfw/) |

---

## D. Struktur Singkat

```
.
├── NeuralNetworks_Chapter9_AML.ipynb   # notebook utama
├── PENJELASAN.md                       # penjelasan section + cara baca grafik
└── nn-dashboard/                       # aplikasi Streamlit
    ├── app.py                          # UI dashboard
    ├── train.py                        # latih & simpan semua model
    ├── eval_models.py                  # evaluasi metrik semua model
    ├── export_samples.py               # ekspor gambar sampel uji coba
    ├── requirements.txt
    ├── data/        (taxi-fares.csv; creditcard.csv diunduh otomatis)
    ├── models/      (model .keras + sampel)
    └── samples/     (gambar contoh digit & wajah)
```
