"""
Latih & simpan semua model + sampel data untuk dashboard.
Cukup jalankan SEKALI:  python train.py

Setiap tahap menampilkan jumlah data latih/uji, fitur, arsitektur, epoch,
dan hasil akhir secara langsung di terminal.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

keras.utils.set_random_seed(42)


def judul(emoji, nama):
    print("\n" + "─" * 60)
    print(f"{emoji}  {nama}")
    print("─" * 60)


def baris(label, nilai):
    print(f"  {label:<18}: {nilai}")


# ---------------------------------------------------------------- 1) TAKSI
def train_taxi():
    judul("🚕", "TAKSI — Regresi Tarif")
    df = pd.read_csv(os.path.join(DATA, "taxi-fares.csv"),
                     parse_dates=["pickup_datetime"])
    df = df[df["passenger_count"] == 1].copy()
    df = df.drop(["key", "passenger_count"], axis=1)
    df["day_of_week"] = df["pickup_datetime"].dt.weekday.astype(float)
    df["pickup_time"] = df["pickup_datetime"].dt.hour.astype(float)
    dx = (df["dropoff_longitude"] - df["pickup_longitude"]) * 54.6
    dy = (df["dropoff_latitude"] - df["pickup_latitude"]) * 69.0
    df["distance"] = np.sqrt(dx ** 2 + dy ** 2)
    df = df.drop(["pickup_datetime", "pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"], axis=1)
    df = df[(df["distance"] > 1.0) & (df["distance"] < 10.0)]
    df = df[(df["fare_amount"] > 0.0) & (df["fare_amount"] < 50.0)]

    x = df.drop("fare_amount", axis=1).values.astype("float32")
    y = df["fare_amount"].values.astype("float32")
    n_val = int(len(x) * 0.2)
    epochs, batch = 30, 100
    baris("Data total", f"{len(x):,} baris (setelah bersih outlier)")
    baris("Data latih", f"{len(x) - n_val:,} (80%)")
    baris("Data validasi", f"{n_val:,} (20%)")
    baris("Fitur (3)", "hari, jam, jarak")
    baris("Arsitektur", "512 → 512 → 1 (relu), optimizer=adam, loss=MAE")
    baris("Epoch / batch", f"{epochs} / {batch}")
    print()

    m = Sequential([keras.Input((3,)),
                    Dense(512, activation="relu"),
                    Dense(512, activation="relu"),
                    Dense(1)])
    m.compile(optimizer="adam", loss="mae", metrics=["mae"])
    h = m.fit(x, y, validation_split=0.2, epochs=epochs, batch_size=batch, verbose=2)
    m.save(os.path.join(MODELS, "taxi.keras"))
    baris("Hasil", f"MAE latih ${h.history['mae'][-1]:.3f} | "
                    f"MAE validasi ${h.history['val_mae'][-1]:.3f}")
    baris("Tersimpan", "models/taxi.keras")


# ---------------------------------------------------------------- 2) DIABETES
def train_diabetes():
    judul("❤️", "DIABETES — Klasifikasi Biner")
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.csv")
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]
    df = pd.read_csv(url, names=cols)
    x = df.drop("outcome", axis=1).values.astype("float32")
    y = df["outcome"].values.astype("float32")
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    xtr, xte, ytr, yte = train_test_split(xs, y, test_size=0.2,
                                          stratify=y, random_state=0)
    epochs, batch = 50, 16
    baris("Data total", f"{len(x)} pasien")
    baris("Data latih", f"{len(xtr)} (80%)")
    baris("Data uji", f"{len(xte)} (20%)")
    baris("Positif diabetes", f"{int(y.sum())} dari {len(y)} "
                              f"({100*y.mean():.1f}%)")
    baris("Fitur (8)", "glukosa, BMI, tekanan, insulin, usia, dll")
    baris("Arsitektur", "128 → 64 → 1 (sigmoid), loss=binary_crossentropy")
    baris("Epoch / batch", f"{epochs} / {batch}")
    print()

    m = Sequential([keras.Input((8,)),
                    Dense(128, activation="relu"),
                    Dense(64, activation="relu"),
                    Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    h = m.fit(xtr, ytr, validation_data=(xte, yte),
              epochs=epochs, batch_size=batch, verbose=2)
    m.save(os.path.join(MODELS, "diabetes.keras"))
    joblib.dump(scaler, os.path.join(MODELS, "diabetes_scaler.pkl"))
    baris("Hasil", f"Akurasi uji {h.history['val_accuracy'][-1]*100:.1f}%")
    baris("Tersimpan", "models/diabetes.keras + diabetes_scaler.pkl")


# ---------------------------------------------------------------- 3) FRAUD
def _ensure_creditcard():
    """Unduh creditcard.csv bila belum ada (tidak ikut di repo karena >100 MB)."""
    csv = os.path.join(DATA, "creditcard.csv")
    if not os.path.exists(csv):
        import urllib.request
        os.makedirs(DATA, exist_ok=True)
        print("  creditcard.csv tidak ada — mengunduh (~144 MB) ...", flush=True)
        url = ("https://raw.githubusercontent.com/nsethi31/"
               "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")
        urllib.request.urlretrieve(url, csv)
    return csv


def train_fraud():
    judul("🛡️", "PENIPUAN — Klasifikasi Biner (imbalanced)")
    df = pd.read_csv(_ensure_creditcard())
    feat_names = [c for c in df.columns if c not in ("Time", "Class")]
    x = df[feat_names].values.astype("float32")
    y = df["Class"].values.astype("int")
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2,
                                          stratify=y, random_state=0)
    epochs, batch = 10, 100
    baris("Data total", f"{len(x):,} transaksi")
    baris("Data latih", f"{len(xtr):,} (80%)")
    baris("Data uji", f"{len(xte):,} (20%)")
    baris("Penipuan", f"{int(y.sum())} dari {len(y):,} "
                      f"({100*y.mean():.3f}% — sangat tidak seimbang)")
    baris("Fitur", f"{x.shape[1]} (V1–V28 PCA + Amount)")
    baris("Arsitektur", "128 → 1 (sigmoid), loss=binary_crossentropy")
    baris("Epoch / batch", f"{epochs} / {batch}")
    print()

    m = Sequential([keras.Input((x.shape[1],)),
                    Dense(128, activation="relu"),
                    Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    h = m.fit(xtr, ytr, validation_data=(xte, yte),
              epochs=epochs, batch_size=batch, verbose=2)
    m.save(os.path.join(MODELS, "fraud.keras"))

    rng = np.random.default_rng(0)
    f_idx = np.where(yte == 1)[0][:25]
    l_idx = rng.choice(np.where(yte == 0)[0], 25, replace=False)
    idx = np.concatenate([f_idx, l_idx])
    rng.shuffle(idx)
    np.savez_compressed(os.path.join(MODELS, "fraud_samples.npz"),
                        x=xte[idx], y=yte[idx], names=np.array(feat_names))
    baris("Hasil", f"Akurasi uji {h.history['val_accuracy'][-1]*100:.2f}%")
    baris("Tersimpan", f"models/fraud.keras + fraud_samples.npz ({len(idx)} contoh)")


# ---------------------------------------------------------------- 4) DIGIT
def train_digit():
    judul("🔢", "DIGIT 0-9 — Klasifikasi Multikelas (MNIST)")
    (xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
    xtr = xtr.reshape(-1, 784).astype("float32") / 255.0
    xte = xte.reshape(-1, 784).astype("float32") / 255.0
    epochs, batch = 12, 128
    baris("Data latih", f"{len(xtr):,} gambar")
    baris("Data uji", f"{len(xte):,} gambar")
    baris("Kelas (10)", "angka 0 sampai 9")
    baris("Fitur", "784 (gambar 28×28 di-flatten)")
    baris("Arsitektur", "256 → 128 → 10 (softmax)")
    baris("Epoch / batch", f"{epochs} / {batch}")
    print()

    m = Sequential([keras.Input((784,)),
                    Dense(256, activation="relu"),
                    Dense(128, activation="relu"),
                    Dense(10, activation="softmax")])
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(xtr, ytr, validation_data=(xte, yte),
              epochs=epochs, batch_size=batch, verbose=2)
    m.save(os.path.join(MODELS, "digit.keras"))

    rng = np.random.default_rng(0)
    img = xte.reshape(-1, 28, 28)
    sel = []
    for d in range(10):
        sel += list(rng.choice(np.where(yte == d)[0], 10, replace=False))
    sel = np.array(sel)
    np.savez_compressed(os.path.join(MODELS, "mnist_samples.npz"),
                        x=xte[sel], img=img[sel], y=yte[sel])
    baris("Hasil", f"Akurasi uji {h.history['val_accuracy'][-1]*100:.2f}%")
    baris("Tersimpan", f"models/digit.keras + mnist_samples.npz ({len(sel)} contoh)")


# ---------------------------------------------------------------- 5) WAJAH
def prep_faces():
    judul("👤", "WAJAH — Siapkan Data (LFW)")
    from sklearn.datasets import fetch_lfw_people
    faces = fetch_lfw_people(min_faces_per_person=100, slice_=None)
    faces.images = faces.images[:, 35:97, 39:86]
    h, w = faces.images.shape[1], faces.images.shape[2]
    data = faces.images.reshape(faces.images.shape[0], h * w)
    names = [str(n) for n in faces.target_names]
    baris("Total gambar", f"{faces.images.shape[0]:,}")
    baris("Ukuran wajah", f"{w}×{h} piksel → {h*w} fitur")
    baris("Kelas (5)", ", ".join(names))
    baris("Model wajah", "model_wajah.keras (sudah dilatih di notebook)")

    rng = np.random.default_rng(0)
    sel = []
    for t in np.unique(faces.target):
        sel += list(rng.choice(np.where(faces.target == t)[0], 8, replace=False))
    sel = np.array(sel)
    np.savez_compressed(os.path.join(MODELS, "faces_samples.npz"),
                        x=data[sel], img=faces.images[sel], y=faces.target[sel])
    with open(os.path.join(MODELS, "faces_meta.json"), "w") as f:
        json.dump({"target_names": names, "h": int(h), "w": int(w)}, f)
    baris("Tersimpan", f"models/faces_samples.npz + faces_meta.json ({len(sel)} contoh)")


def main():
    print("=" * 60)
    print(" MELATIH SEMUA MODEL — Dashboard Neural Network")
    print("=" * 60)
    train_taxi()
    train_diabetes()
    train_fraud()
    train_digit()
    prep_faces()
    print("\n" + "=" * 60)
    print(" ✅ SELESAI. Jalankan:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
