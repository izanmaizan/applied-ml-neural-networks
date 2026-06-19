"""
Ekspor gambar sampel untuk uji coba / demo.
Jalankan SETELAH train.py:  python export_samples.py

Hasil di folder samples/:
  samples/mnist/   -> gambar digit 0-9 (beberapa per angka, + grid)
  samples/faces/   -> gambar wajah tiap tokoh (beberapa per orang, + grid)
  samples/CONTOH_INPUT.md -> daftar input angka untuk Taksi/Diabetes
"""
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
OUT = os.path.join(BASE, "samples")


def to_u8(arr):
    a = arr.astype("float32")
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return (a * 255).astype("uint8")


def save_raw(arr, path, scale):
    """Simpan gambar mentah (tanpa judul/padding) — cocok untuk diunggah & diuji."""
    im = Image.fromarray(to_u8(arr))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def bersihkan(folder):
    os.makedirs(folder, exist_ok=True)
    for f in glob.glob(os.path.join(folder, "*.png")):
        os.remove(f)


def export_mnist():
    npz = os.path.join(MODELS, "mnist_samples.npz")
    if not os.path.exists(npz):
        print("[skip] mnist_samples.npz belum ada — jalankan train.py dulu")
        return
    d = np.load(npz, allow_pickle=True)
    img, y = d["img"], d["y"]
    folder = os.path.join(OUT, "mnist")
    bersihkan(folder)

    # 3 contoh per angka 0-9
    count = {}
    for i in range(len(img)):
        lbl = int(y[i])
        count.setdefault(lbl, 0)
        if count[lbl] >= 3:
            continue
        count[lbl] += 1
        save_raw(img[i], os.path.join(folder, f"digit_{lbl}_{count[lbl]}.png"), scale=6)

    # grid satu contoh per angka 0-9
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, d_ in zip(axes.flat, range(10)):
        i = np.where(y == d_)[0][0]
        ax.imshow(img[i], cmap="gray")
        ax.set_title(f"Angka {d_}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Contoh Digit MNIST (0-9)")
    fig.savefig(os.path.join(folder, "_grid.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    n = len(glob.glob(os.path.join(folder, "digit_*.png")))
    print(f"[ok] MNIST -> {folder} ({n} gambar)")


def export_faces():
    npz = os.path.join(MODELS, "faces_samples.npz")
    meta = os.path.join(MODELS, "faces_meta.json")
    mpath = os.path.join(MODELS, "model_wajah.keras")
    if not (os.path.exists(npz) and os.path.exists(meta)):
        print("[skip] faces_samples.npz/meta belum ada — jalankan train.py dulu")
        return
    d = np.load(npz, allow_pickle=True)
    img, y, x = d["img"], d["y"], d["x"]
    names = json.load(open(meta))["target_names"]
    folder = os.path.join(OUT, "faces")
    bersihkan(folder)

    # Prediksi model -> utamakan contoh yang ditebak BENAR agar demo cocok
    preds = (keras.models.load_model(mpath).predict(x, verbose=0).argmax(1)
             if os.path.exists(mpath) else y)

    grid_pick = {}
    for lbl in np.unique(y):
        idxs = np.where(y == lbl)[0]
        benar = [i for i in idxs if preds[i] == lbl]
        chosen = (benar[:3] if benar else [idxs[0]])
        grid_pick[int(lbl)] = chosen[0]
        safe = names[int(lbl)].replace(" ", "_")
        for k, i in enumerate(chosen, 1):
            save_raw(img[i], os.path.join(folder, f"{safe}_{k}.png"), scale=4)

    # grid satu wajah (yang benar) per tokoh
    uniq = sorted(grid_pick)
    fig, axes = plt.subplots(1, len(uniq), figsize=(2.4 * len(uniq), 3))
    axes = np.atleast_1d(axes)
    for ax, lbl in zip(axes, uniq):
        ax.imshow(img[grid_pick[lbl]], cmap="gray")
        ax.set_title(names[lbl], fontsize=8)
        ax.axis("off")
    fig.suptitle("Contoh Wajah (LFW)")
    fig.savefig(os.path.join(folder, "_grid.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    n = len(glob.glob(os.path.join(folder, "*.png"))) - 1
    print(f"[ok] Wajah -> {folder} ({n} gambar)")


def export_doc():
    os.makedirs(OUT, exist_ok=True)
    doc = """# Contoh Input untuk Uji Coba

Gambar siap pakai ada di `samples/mnist/` (angka 0-9) dan `samples/faces/` (5 tokoh).
Nama file wajah = identitas asli (ground-truth). Untuk model angka, pakai nilai berikut.

## 🚕 Tarif Taksi
| Hari   | Jam | Jarak (mil) | Perkiraan      |
|--------|-----|-------------|----------------|
| Jumat  | 17  | 2.0         | tarif sedang   |
| Sabtu  | 2   | 8.5         | tarif tinggi   |
| Senin  | 9   | 1.5         | tarif rendah   |

## ❤️ Risiko Diabetes
(urutan: Kehamilan, Glukosa, Tekanan, Tebal Kulit, Insulin, BMI, Riwayat, Usia)
| Contoh        | Nilai                              | Harapan    |
|---------------|------------------------------------|------------|
| Risiko tinggi | 6, 148, 72, 35, 0, 33.6, 0.627, 50 | Berisiko   |
| Risiko rendah | 1, 85, 66, 29, 0, 26.6, 0.351, 31  | Rendah     |

## 🛡️ Deteksi Penipuan
Fitur sudah ter-anonim (PCA). Pilih transaksi langsung dari tabel di aplikasi.

## 🔢 Pengenal Digit & 👤 Pengenalan Wajah
Bisa digambar, diunggah (pakai gambar di `samples/`), atau diuji dari data uji.
"""
    with open(os.path.join(OUT, "CONTOH_INPUT.md"), "w") as f:
        f.write(doc)
    print(f"[ok] Cheat-sheet -> {os.path.join(OUT, 'CONTOH_INPUT.md')}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    export_mnist()
    export_faces()
    export_doc()
    print("Selesai.")
