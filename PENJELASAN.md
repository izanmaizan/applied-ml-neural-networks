# Penjelasan Notebook — Neural Networks (Chapter 9)

Dokumen ini menjelaskan tiap section dan step di `NeuralNetworks_Chapter9_AML.ipynb`.

**Cara menjalankan:** buka [colab.research.google.com](https://colab.research.google.com) → `File > Upload notebook` → pilih file `.ipynb` → `Runtime > Run all`. Tidak perlu instalasi atau unduh CSV manual. Untuk bagian wajah (Section 7), aktifkan GPU lewat `Runtime > Change runtime type > GPU`.

---

## 1. Setup

Mengimpor semua library (TensorFlow/Keras untuk model, scikit-learn untuk data & evaluasi, pandas/numpy/matplotlib/seaborn untuk olah data & grafik) dan mencetak versinya. `np.random.seed` dan `tf.random.set_seed` dipasang agar hasil relatif konsisten tiap dijalankan — walau neural network tetap punya keacakan bawaan.

## 2. Sequential API & Functional API

- **Sequential API.** Membangun perceptron 3-layer paling dasar: input 2 nilai → hidden 3 neuron (ReLU) → output 1 neuron. Layer disusun berurutan satu arah. Model ini tidak dilatih, hanya `summary()` untuk memperlihatkan struktur layer dan jumlah parameter.
- **Functional API.** Cara yang lebih fleksibel: tiap layer dipanggil sebagai fungsi terhadap output layer sebelumnya, lalu dibungkus `keras.Model(inputs, outputs)`. Cocok untuk model multi-input/output, bercabang, atau berbagi layer. Contohnya: input 2 fitur → Dense 128 (ReLU) → Dense 1 (sigmoid).

Catatan: `keras.Input(shape=(2,))` setara dengan `input_dim=2` di buku, tapi lebih aman untuk versi Keras terbaru.

## 3. Regresi — Prediksi Tarif Taksi

Tujuan: memprediksi tarif (angka kontinu) dari 3 fitur, jadi ini masalah **regresi**.

- **Step 1 — Muat & olah data nyata.** Mengunduh `taxi-fares.csv` (NYC Taxi, ±55 rb baris) dari GitHub penulis buku. Disaring 1 penumpang, dihitung `day_of_week` (0–6) & `pickup_time` (0–23) dari waktu, lalu `distance` (mil) dari koordinat GPS. Outlier dibuang (jarak 1–10 mil, tarif $0–50) → tersisa ±23 rb baris.
- **Step 2 — Bangun model.** Input 3 → dua hidden layer @512 (ReLU) → output 1 neuron tanpa aktivasi. Loss memakai **MAE** (rata-rata selisih absolut). `summary()` memperlihatkan jumlah trainable params.
- **Step 3 — Latih.** `fit` dengan `validation_split=0.2` (80% latih, 20% validasi), 50 epoch, batch 100.
- **Step 4 — Plot MAE.** Membandingkan kurva MAE training vs validasi. Jika dua kurva berdekatan → tidak overfitting. → *lihat gambar A*.
- **Step 5 — R².** Skor 0–1; makin dekat 1 makin baik.
- **Step 6 — Prediksi.** Memprediksi tarif perjalanan 2 mil jam 17:00 di hari Jumat vs Sabtu.

## 4. Klasifikasi Biner — Pima Indians Diabetes

Tujuan: memprediksi risiko diabetes (2 kelas) dari indikator klinis — pakai **data nyata**.

- **Step 1 — Muat data.** Mengunduh `pima-indians-diabetes.csv` (768 pasien, 8 fitur). Karena CSV tak berheader, nama kolom diberi manual.
- **Step 2 — Olah.** Pisahkan fitur (x) & label (y), lalu **normalisasi** dengan `StandardScaler` (z-score).
- **Step 3 — Visualisasi.** Scatter sebaran **Glucose vs BMI** berdasarkan kelas. → *lihat gambar B*.
- **Step 4 — Model & latih.** Input 8 → hidden 128 → 64 (ReLU) → output 1 **sigmoid**. Loss `binary_crossentropy`. Split 80/20 `stratify`, 50 epoch. Lalu plot akurasi training vs validasi. → *lihat gambar C*.
- **Step 5 — Prediksi.** Probabilitas seorang pasien (input dinormalisasi dgn scaler yang sama), lalu jadi kelas dengan ambang 0,5.

## 5. Klasifikasi Biner — Deteksi Penipuan & Confusion Matrix

Tujuan: kasus nyata dengan data **sangat tidak seimbang** (penipuan < 0,3%).

- **Step 1 — Muat data.** Mengunduh `creditcard.csv` (Credit Card Fraud, mirror GitHub). Kolom `Time` dibuang, sisanya 29 fitur (V1–V28 hasil PCA + Amount). Dibagi train/test dengan `stratify` agar proporsi kelas tetap.
- **Step 2 — Model & latih.** Input 29 → hidden 128 → output 1 sigmoid. Dievaluasi pakai `validation_data`.
- **Step 3 — Confusion matrix + laporan.** Akurasi ~99,9% **menyesatkan** (menebak "semua sah" saja sudah hampir benar). Confusion matrix + `classification_report` (Precision/Recall/F1) memperlihatkan berapa penipuan yang benar-benar tertangkap. → *lihat gambar D*.

## 6. Klasifikasi Multikelas — MNIST 4 Digit

Tujuan: memprediksi 1 dari **4 kelas** (digit 0–3) dari data gambar **nyata**.

- **Step 1 — Muat data.** Memuat MNIST bawaan Keras, lalu disaring hanya digit 0–3. Gambar 28×28 di-flatten jadi 784 fitur dan dinormalisasi ke 0–1. Beberapa contoh ditampilkan sebagai grid. → *lihat gambar E*.
- **Step 2 — Model & latih.** Output **4 neuron** dengan **softmax** (jumlah probabilitas = 1), loss `sparse_categorical_crossentropy`. Plot akurasi training vs validasi.
- **Step 3 — Prediksi.** Satu gambar diprediksi; output array 4 probabilitas, kelas terpilih = `argmax`.

> Catatan: di **aplikasi Streamlit** (`nn-dashboard`), model digit diperluas menjadi **10 kelas (0–9)**; di notebook tetap 4 kelas sesuai buku.

## 7. Pengenalan Wajah (LFW)

Tujuan: multikelas pada data **nyata** — mengenali 5 wajah tokoh dari dataset Labeled Faces in the Wild.

- **Step 1 — Muat & crop.** `fetch_lfw_people` mengunduh dataset (≈200 MB saat pertama jalan) dan gambar di-crop seperti di buku.
- **Step 2 — Seimbangkan & split.** Maksimal 100 gambar per orang, lalu bagi 80% latih / 20% uji. (Typo `face_images` di buku sudah diperbaiki jadi `x_faces`.)
- **Step 3 — Model & latih.** Input = lebar×tinggi piksel → hidden 512 → output `class_count` neuron softmax.
- **Step 4 — Plot + confusion matrix.** Melihat akurasi per epoch dan wajah mana yang sering tertukar.

> Sel ini paling lama (beberapa menit). Pakai GPU agar cepat.

## 8. Dropout

Tujuan: mengurangi **overfitting**. Model wajah dibangun ulang dengan `Dropout(0.2)` yang mengabaikan 20% koneksi secara acak tiap iterasi training, lalu dilatih ulang untuk membandingkan akurasi validasinya.

## 9. Menyimpan & Memuat Model

Menyimpan model ke `model_wajah.keras`, memuatnya kembali dengan `load_model`, lalu membuktikan prediksi model asli dan model hasil-muat **identik** (`np.allclose`). Format `.keras` adalah pengganti modern untuk `.h5` di buku.

## 10. Keras Callbacks

Mengontrol proses training otomatis:

- **StopCallback (custom).** Menghentikan training begitu `val_accuracy` mencapai ambang (0,95).
- **EarlyStopping.** Berhenti jika akurasi validasi tak membaik selama 5 epoch (`patience=5`), lalu mengembalikan bobot terbaik.
- **ModelCheckpoint.** Menyimpan otomatis model dengan akurasi validasi tertinggi ke `best_model.keras`.
- **TensorBoard.** Mencatat metrik dan grafik training ke folder `logs`. Di Colab bisa dibuka dengan menjalankan `%load_ext tensorboard` lalu `%tensorboard --logdir logs` di sel terpisah.

---

## Ringkasan pola model

| Jenis masalah | Output layer | Aktivasi output | Loss function |
|---|---|---|---|
| Regresi | 1 neuron | (tidak ada) | `mae` / `mse` |
| Klasifikasi biner | 1 neuron | `sigmoid` | `binary_crossentropy` |
| Klasifikasi multikelas | N neuron (= jumlah kelas) | `softmax` | `sparse_categorical_crossentropy` |

## Catatan umum

- Beberapa contoh jumlah epoch-nya dikecilkan dari buku agar "Run all" lebih cepat (mis. taksi & wajah dari 100 → 50).
- **Semua dataset kini nyata** (bukan sintetis): NYC Taxi, Pima Diabetes, Credit Card Fraud, MNIST, dan LFW — diunduh otomatis saat notebook dijalankan.

---

# Penjelasan Gambar & Visualisasi

Bagian ini mengajarkan **cara membaca tiap gambar/grafik** yang muncul saat notebook dijalankan. Angka contoh di sini diambil dari satu kali run — **bisa sedikit berbeda** tiap kali dilatih ulang, tapi cara membacanya sama.

### Prinsip dasar dulu (penting!)
- **Kurva ERROR** (mis. MAE) → **makin RENDAH makin baik**.
- **Kurva ACCURACY** → **makin TINGGI (mendekati 1.0 / 100%) makin baik**.
- **Garis penuh** = data **training** (latih), **garis putus-putus** = data **validation** (uji).
- **Overfitting** = model hafal data latih tapi gagal di data baru → ciri: garis training makin bagus tapi garis validasi **berbalik memburuk** (gap melebar).
- **Confusion matrix:** **baris = label asli**, **kolom = tebakan model**. Sel **diagonal** (kiri-atas → kanan-bawah) = **tebakan benar**; sel di luar diagonal = salah.

---

## A. Kurva MAE Taksi (Section 3)

**Jenis:** learning curve *Mean Absolute Error* (MAE) selama 50 epoch. MAE = rata-rata selisih (dalam dolar) antara tarif tebakan vs tarif asli.

**Cara baca, langkah per langkah:**
1. **Sumbu-X** = Epoch (putaran latihan), **Sumbu-Y** = MAE (error dalam $). Ingat: **makin rendah makin baik**.
2. **Garis Training MAE** mulai tinggi (≈2,9) lalu turun cepat di epoch awal dan melandai ke ≈2,3.
3. **Garis Validation MAE** mengikuti tren turun yang sama.
4. **Cek konvergensi:** karena keduanya turun lalu **mendatar (*plateau*)**, model sudah "selesai belajar" — menambah epoch tak banyak menolong.
5. **Cek overfitting:** karena garis validasi **tidak berbalik naik** dan menempel dekat garis training → **tidak overfitting**, generalisasi baik.

**Catatan:** judul plot tertulis "…Accuracy" — itu **typo**; isi grafik ini adalah **error (MAE)**, bukan akurasi. (Kalau validasi sempat lebih rendah dari training, itu lumrah — bisa karena efek regularisasi atau data validasi kebetulan lebih mudah.)

---

## B. Scatter Glucose vs BMI — Diabetes (Section 4)

**Jenis:** scatter plot (diagram tebar) sebaran pasien pada 2 fitur paling informatif: **Glukosa (sumbu-X)** vs **BMI (sumbu-Y)**. Sudah dinormalisasi → skala *z-score* (pusat 0, bukan satuan mg/dL atau kg/m²).

**Cara baca, langkah per langkah:**
1. **Baca warna (colorbar kanan):** **biru = tidak diabetes (0)**, **merah = diabetes (1)**. Teks di atas grafik: 768 pasien, 268 (34,9%) positif.
2. **Lihat lokasi tiap warna:** titik **biru menumpuk di kiri-bawah** (glukosa & BMI rendah). Titik **merah dominan di kanan-atas** (glukosa & BMI tinggi).
3. **Tarik kesimpulan hubungan:** makin ke kanan (glukosa↑) dan makin ke atas (BMI↑), kecenderungan diabetes makin besar → kedua fitur ini memang "penanda" diabetes.
4. **Cari pencilan (outlier):** ada barisan titik rapi di **tepi kiri (Glucose ≈ −4)** dan **tepi bawah (BMI ≈ −4)**. Itu pasien dengan **nilai 0 di data asli** (glukosa/BMI 0 mustahil secara medis → *missing value* yang belum dibersihkan).
5. **Nilai untuk model:** ada area tengah tempat biru & merah **bertumpuk** (tidak bisa dipisah sempurna), tapi secara umum **daya pisahnya (*separability*) baik** → model bisa menarik *decision boundary* yang masuk akal.

---

## C. Kurva Akurasi Diabetes (Section 4)

**Jenis:** learning curve **akurasi** (makin tinggi makin baik). Ini contoh **klasik OVERFITTING**.

**Cara baca, langkah per langkah:**
1. **Garis Training Accuracy** naik terus & stabil dari ≈69% sampai mendekati ≈90% → model makin "hafal" data latih.
2. **Garis Validation Accuracy** naik bagus sampai **puncak ≈85% di sekitar epoch 12–22**, lalu **berbalik turun** ke ≈80% di epoch 50.
3. **Kenali tanda overfitting:** setelah ±epoch 25, **gap antara dua garis melebar** (training terus naik, validasi turun). Inilah tanda utama overfitting.
4. **Pahami penyebabnya:** arsitektur cukup besar (128→64) untuk data yang kecil (768 baris) + dilatih 50 epoch **tanpa penahan** → model mulai menghafal noise.
5. **Solusi (lihat Section 8 & 10):**
   - **EarlyStopping** → hentikan di puncak validasi (≈epoch 15–20).
   - **Dropout** → matikan sebagian neuron acak biar tak menghafal.
   - **Sederhanakan arsitektur** → mis. 64→32 atau 1 hidden layer saja.

---

## D. Confusion Matrix + Laporan — Penipuan (Section 5)

**Jenis:** confusion matrix 2×2 + *classification report*, untuk data **sangat tidak seimbang** (fraud < 0,3%).

**Langkah 1 — Baca 4 sel matriks** (baris = asli, kolom = tebakan):

| | Prediksi: Legit | Prediksi: Fraud |
|---|---|---|
| **Asli: Legit** | **TN** ≈56.856 — sah, ditebak sah (benar) | **FP** ≈8 — sah, dituduh fraud (*alarm palsu*) |
| **Asli: Fraud** | **FN** ≈27 — fraud, lolos dikira sah (**paling berbahaya**) | **TP** ≈71 — fraud, tertangkap (benar) |

**Langkah 2 — Sadari kenapa "accuracy" menipu:** akurasi ≈100%, tapi kalau ada model malas yang menebak **"semua sah"** saja, ia sudah ≈99,8% benar (karena fraud cuma segelintir). Jadi accuracy **tidak boleh** jadi patokan di sini.

**Langkah 3 — Baca laporan (fokus baris Fraud):**
- **Precision ≈0,90** → dari semua yang **dituduh** fraud (TP+FP = 71+8 = 79), **90% benar** fraud. Artinya jarang salah blokir.
- **Recall ≈0,72** → dari semua fraud **yang sebenarnya ada** (TP+FN = 71+27 = 98), model **menangkap 72%**; sisanya 28% (27 transaksi) **lolos**.
- **F1 ≈0,80** → rata-rata seimbang precision & recall; 0,80 tergolong **kuat**.

**Langkah 4 — Tahu cara memperbaiki:** untuk finansial, fraud yang lolos (FN) itu mahal. Turunkan ambang dari `> 0.5` ke mis. `> 0.3`:
```python
y_pred = (model_fraud.predict(x_test) > 0.3).astype(int)
```
Efek: model lebih sensitif → **Recall naik** (fraud lolos berkurang), tapi **Precision sedikit turun** (alarm palsu bertambah). Ini soal *trade-off*.

---

## E. Grid Digit + Kurva Akurasi MNIST (Section 6)

**Tiga gambar muncul di section ini:**
1. **Grid digit:** beberapa contoh angka tulisan tangan dari data uji — sekadar memperlihatkan seperti apa input modelnya (gambar 28×28).
2. **Kurva Akurasi:** **cara baca** → training & validasi sama-sama **naik cepat ke level tinggi (±97–99%) dan menempel berdekatan**. Gap kecil = **sehat, tanpa overfitting berarti**. MNIST memang relatif "mudah" sehingga akurasi cepat tinggi & stabil.
3. **Prediksi satuan:** satu gambar ditampilkan dengan judul **"Asli: x | Prediksi: y"** — kalau x = y berarti tebakan benar.

---

## F. Kurva Akurasi & Confusion Matrix Wajah (Section 7)

**Gambar 1 — Kurva Akurasi:** akurasi naik tiap epoch, tapi **gap training–validasi yang lebar** menandakan model wajah **cenderung overfitting** (wajar: data per orang sedikit & fitur piksel mentah).

**Gambar 2 — Confusion Matrix 5×5** (5 tokoh). **Cara baca, langkah per langkah:**
1. **Baris = wajah asli, Kolom = tebakan model.** Warna makin gelap = jumlah makin banyak.
2. **Lihat diagonal (kiri-atas → kanan-bawah) = tebakan BENAR.** Contoh satu run: Colin Powell 13, Donald Rumsfeld 15, George W Bush 18, Gerhard Schroeder 19, Tony Blair 9.
3. **Baca antar-baris untuk temukan kesalahan:** mis. baris **Tony Blair** menunjukkan **10 fotonya salah ditebak jadi Gerhard Schroeder** — kesalahan terbesar. Artinya model sering menganggap wajah Tony Blair mirip Schroeder.
4. **Baca per-kolom untuk temukan bias:** kolom **Gerhard Schroeder** kemasukan banyak wajah orang lain (Powell, Rumsfeld, Bush, Blair) → model **bias** cenderung menebak "Schroeder" saat ragu.
5. **Identifikasi kelas terbaik & terlemah:** **George W Bush & Gerhard Schroeder** paling akurat (data terbanyak); **Tony Blair** paling lemah.
6. **Solusi:** seimbangkan jumlah data tiap orang, *data augmentation* (rotasi/crop/kecerahan), atau pakai arsitektur lebih kuat (CNN) — sebab di sini cuma 1 hidden layer dari piksel mentah.

---

> **Catatan judul plot:** beberapa grafik berjudul "Training and Validation Accuracy" walau untuk Section 3 isinya **MAE (error)**. Itu hanya label; perhatikan **sumbu-Y** untuk tahu metrik sebenarnya (Mean Absolute Error vs Accuracy).
