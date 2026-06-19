# 🧠 Dashboard Prediksi Neural Network

Aplikasi web (Streamlit) untuk mencoba 5 model neural network dari **Chapter 9 — Applied Machine Learning**.
UI bersih, satu menu samping untuk berpindah antar model.

**Kelompok:** Maizan Insani Akbar (242321011) · Muhammad Dicky Imansyah (242321030)

---

## Fitur

| Model | Tipe | Input | Output |
|-------|------|-------|--------|
| 🚕 Tarif Taksi | Regresi | Hari, jam (jam pilih), jarak | Estimasi tarif ($) |
| ❤️ Risiko Diabetes | Klasifikasi biner | 8 indikator klinis (bersatuan) | Probabilitas risiko |
| 🛡️ Deteksi Penipuan | Klasifikasi biner | Pilih transaksi dari tabel | Sah / Penipuan + **alasan** |
| 🔢 Pengenal Digit | Multikelas | **Gambar tangan**, unggah, atau data uji | Angka 0–9 + grafik |
| 👤 Pengenalan Wajah | Multikelas | Unggah foto atau data uji | Nama tokoh |

**Sorotan UX:**
- 🔢 Digit bisa **digambar langsung** di canvas, **diunggah**, atau diambil dari data uji.
- 📤 Unggah gambar menampilkan **sebelum (asli)** vs **sesudah (diproses)** berdampingan.
- 🛡️ Penipuan menampilkan **tabel transaksi lengkap** + **penjelasan fitur** kenapa diputuskan penipuan.
- ☀️ Dipaksa **light mode** (kontras teks/latar terjaga) lewat `.streamlit/config.toml`.

---

## Cara Menjalankan

### Otomatis (paling mudah)
```bash
cd nn-dashboard
bash setup.sh        # buat venv + install + latih model (sekali saja)
bash run.sh          # jalankan aplikasi
```

### Manual
```bash
cd nn-dashboard
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                   # latih & simpan model (sekali saja)
streamlit run app.py              # buka di browser
```

Aplikasi terbuka otomatis di `http://localhost:8501`.

---

## Struktur Project

```
nn-dashboard/
├── app.py              # aplikasi Streamlit (UI dashboard)
├── train.py            # latih & simpan semua model
├── export_samples.py   # ekspor gambar sampel untuk uji coba
├── requirements.txt    # daftar dependency
├── setup.sh / run.sh   # skrip bantu
├── data/
│   ├── taxi-fares.csv
│   └── creditcard.csv
├── samples/            # gambar contoh + cheat-sheet input (hasil export_samples.py)
│   ├── mnist/          # digit 0-3 (.png)
│   ├── faces/          # wajah tiap tokoh (.png)
│   └── CONTOH_INPUT.md # contoh angka untuk Taksi & Diabetes
└── models/             # model & sampel hasil train.py
    ├── best_model.keras        (MNIST — disediakan)
    ├── model_wajah.keras       (wajah — disediakan)
    ├── taxi.keras
    ├── diabetes.keras + diabetes_scaler.pkl
    ├── fraud.keras + fraud_samples.npz
    ├── mnist_samples.npz
    └── faces_samples.npz + faces_meta.json
```

---

## Catatan

- `train.py` mengunduh dataset Pima (kecil) dan LFW (via scikit-learn) saat pertama dijalankan, lalu menyimpan sampelnya. Setelah itu aplikasi **tidak butuh internet**.
- Startup aplikasi cepat karena model hanya **di-load**, bukan dilatih ulang.
- Jika sebuah model belum ada, aplikasi menampilkan pesan yang meminta menjalankan `python train.py` — tidak error.
