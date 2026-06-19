"""Evaluasi lengkap semua model terlatih. Jalankan: python eval_models.py"""
import os
import numpy as np
import pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score,
                             precision_score, recall_score, f1_score)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS = os.path.join(BASE, "data"), os.path.join(BASE, "models")


def taxi():
    df = pd.read_csv(os.path.join(DATA, "taxi-fares.csv"),
                     parse_dates=["pickup_datetime"])
    df = df[df["passenger_count"] == 1].copy().drop(["key", "passenger_count"], axis=1)
    df["day_of_week"] = df["pickup_datetime"].dt.weekday.astype(float)
    df["pickup_time"] = df["pickup_datetime"].dt.hour.astype(float)
    dx = (df["dropoff_longitude"] - df["pickup_longitude"]) * 54.6
    dy = (df["dropoff_latitude"] - df["pickup_latitude"]) * 69.0
    df["distance"] = np.sqrt(dx ** 2 + dy ** 2)
    df = df.drop(["pickup_datetime", "pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"], axis=1)
    df = df[(df["distance"] > 1) & (df["distance"] < 10)]
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 50)]
    x = df.drop("fare_amount", axis=1).values.astype("float32")
    y = df["fare_amount"].values.astype("float32")
    m = keras.models.load_model(os.path.join(MODELS, "taxi.keras"))
    pred = m.predict(x, verbose=0).flatten()
    print(f"🚕 TAKSI (regresi)   : MAE ${mean_absolute_error(y, pred):.3f} | "
          f"R² {r2_score(y, pred):.3f} | n={len(y):,}")


def diabetes():
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.csv")
    df = pd.read_csv(url, header=None)
    x = df.iloc[:, :8].values.astype("float32")
    y = df.iloc[:, 8].values.astype("int")
    sc = joblib.load(os.path.join(MODELS, "diabetes_scaler.pkl"))
    x = sc.transform(x)
    _, xte, _, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
    m = keras.models.load_model(os.path.join(MODELS, "diabetes.keras"))
    pred = (m.predict(xte, verbose=0).flatten() > 0.5).astype(int)
    print(f"❤️ DIABETES (biner)   : Akurasi {accuracy_score(yte, pred)*100:.1f}% | "
          f"Presisi {precision_score(yte, pred)*100:.1f}% | "
          f"Recall {recall_score(yte, pred)*100:.1f}% | "
          f"F1 {f1_score(yte, pred)*100:.1f}% | n_uji={len(yte)}")


def fraud():
    csv = os.path.join(DATA, "creditcard.csv")
    if not os.path.exists(csv):
        import urllib.request
        os.makedirs(DATA, exist_ok=True)
        print("  creditcard.csv tidak ada — mengunduh (~144 MB) ...", flush=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/nsethi31/"
            "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv", csv)
    df = pd.read_csv(csv)
    feat = [c for c in df.columns if c not in ("Time", "Class")]
    x = df[feat].values.astype("float32")
    y = df["Class"].values.astype("int")
    _, xte, _, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
    m = keras.models.load_model(os.path.join(MODELS, "fraud.keras"))
    pred = (m.predict(xte, verbose=0).flatten() > 0.5).astype(int)
    print(f"🛡️ PENIPUAN (biner)   : Akurasi {accuracy_score(yte, pred)*100:.2f}% | "
          f"Presisi {precision_score(yte, pred, zero_division=0)*100:.1f}% | "
          f"Recall {recall_score(yte, pred)*100:.1f}% | "
          f"F1 {f1_score(yte, pred)*100:.1f}% | n_uji={len(yte):,}")


def digit():
    (_, _), (xte, yte) = keras.datasets.mnist.load_data()
    xte = xte.reshape(-1, 784).astype("float32") / 255.0
    m = keras.models.load_model(os.path.join(MODELS, "digit.keras"))
    pred = m.predict(xte, verbose=0).argmax(1)
    print(f"🔢 DIGIT 0-9 (multi)  : Akurasi {accuracy_score(yte, pred)*100:.2f}% | "
          f"n_uji={len(yte):,}")


def faces():
    from sklearn.datasets import fetch_lfw_people
    f = fetch_lfw_people(min_faces_per_person=100, slice_=None)
    f.images = f.images[:, 35:97, 39:86]
    data = f.images.reshape(f.images.shape[0], -1)
    mask = np.zeros(f.target.shape, bool)
    for t in np.unique(f.target):
        mask[np.where(f.target == t)[0][:100]] = 1
    x, y = data[mask], f.target[mask]
    _, xte, _, yte = train_test_split(x, y, train_size=0.8, stratify=y, random_state=0)
    m = keras.models.load_model(os.path.join(MODELS, "model_wajah.keras"))
    pred = m.predict(xte, verbose=0).argmax(1)
    print(f"👤 WAJAH (multi 5)    : Akurasi {accuracy_score(yte, pred)*100:.1f}% | "
          f"n_uji={len(yte)}")


if __name__ == "__main__":
    print("=" * 60)
    print(" HASIL EVALUASI SEMUA MODEL")
    print("=" * 60)
    taxi(); diabetes(); fraud(); digit(); faces()
    print("=" * 60)
