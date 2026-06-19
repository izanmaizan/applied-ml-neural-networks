"""
Dashboard Prediksi Neural Network — Chapter 9 (Applied Machine Learning)
Jalankan:  streamlit run app.py

Sebelum menjalankan, latih model dulu sekali:  python train.py
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # diamkan log TF
import keras  # noqa: E402
import joblib  # noqa: E402

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except Exception:
    CANVAS_OK = False

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
PRIMARY = "#2563eb"

# Matplotlib mengikuti tema terang (teks gelap di latar putih)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": "#0f172a",
    "axes.labelcolor": "#0f172a", "axes.edgecolor": "#cbd5e1",
    "xtick.color": "#334155", "ytick.color": "#334155",
})

# ----------------------------------------------------------------- Konfigurasi
st.set_page_config(page_title="Dashboard Neural Network",
                   page_icon="🧠", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1080px;}
  .hero {
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      color: #ffffff; padding: 1.4rem 1.7rem; border-radius: 16px; margin-bottom: 1.4rem;
  }
  .hero h1 {margin: 0; font-size: 1.55rem; color: #ffffff;}
  .hero p  {margin: .35rem 0 0; color: #e2e8f0;}
  .stButton>button {
      border-radius: 10px; padding: .5rem 1.2rem; font-weight: 600;
      border: 1px solid #cbd5e1;
  }
  /* tombol primary (type="primary") */
  .stButton>button[kind="primary"] {background: #2563eb; color: #ffffff; border: none;}
  .stButton>button[kind="primary"]:hover {background: #1d4ed8; color: #ffffff;}
  [data-testid="stMetricValue"] {color: #2563eb;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- Pemuat (cache)
@st.cache_resource(show_spinner=False)
def load_model(name):
    path = os.path.join(MODELS, name)
    return keras.models.load_model(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_scaler(name):
    path = os.path.join(MODELS, name)
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_npz(name):
    path = os.path.join(MODELS, name)
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_json(name):
    path = os.path.join(MODELS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def butuh_train(label):
    st.warning(f"Model **{label}** belum tersedia. Jalankan dulu di terminal:\n\n"
               "```bash\npython train.py\n```")


def to_u8(arr):
    """Normalisasi array gambar ke uint8 0-255 agar aman ditampilkan."""
    a = arr.astype("float32")
    rng = a.max() - a.min()
    a = (a - a.min()) / (rng + 1e-9)
    return (a * 255).astype("uint8")


def proses_upload(file, mode):
    """Ubah gambar unggahan jadi format input model.
    Mengembalikan (gambar_asli_RGB, array_terproses)."""
    orig = Image.open(file).convert("RGB")
    gray = orig.convert("L")
    if mode == "mnist":
        gray = gray.resize((28, 28))
        arr = np.array(gray).astype("float32")
        if arr.mean() > 127:          # latar terang -> balik (MNIST: angka terang)
            arr = 255.0 - arr
        return orig, arr / 255.0      # 28x28, skala 0-1
    gray = gray.resize((47, 62))      # wajah: (lebar, tinggi)
    return orig, np.array(gray).astype("float32") / 255.0  # 62x47, skala 0-1


def tampil_sebelum_sesudah(orig, arr_proc, judul_kanan):
    """Dua kolom: kiri gambar asli, kanan gambar hasil proses."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**⬅️ Sebelum (asli)**")
        st.image(orig, use_container_width=True)
    with col2:
        st.markdown("**➡️ Sesudah (diproses)**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(arr_proc, cmap="gray")
        ax.axis("off")
        ax.set_title(judul_kanan)
        st.pyplot(fig)


def kontrol_sampel(prefix, n):
    """Baris tombol Acak + Pilih. Mengembalikan (indeks_aktif, klik_pilih)."""
    key = f"{prefix}_idx"
    c1, c2 = st.columns(2)
    if c1.button("🎲 Acak", key=f"{prefix}_rand", type="primary",
                 use_container_width=True):
        st.session_state[key] = int(np.random.randint(n))
    pilih = c2.button("🖼️ Pilih", key=f"{prefix}_pick", use_container_width=True)
    return st.session_state.get(key), pilih


# ----------------------------------------------------------------- Modal pilih
@st.dialog("Pilih digit untuk diuji", width="large")
def dlg_mnist(s):
    st.caption("Klik salah satu gambar di bawah.")
    n = len(s["img"])
    per = 6
    for r in range(0, n, per):
        cols = st.columns(per)
        for c, i in enumerate(range(r, min(r + per, n))):
            with cols[c]:
                st.image(to_u8(s["img"][i]), use_container_width=True)
                if st.button(f"#{i}", key=f"mnpick{i}", use_container_width=True):
                    st.session_state["mnist_idx"] = i
                    st.rerun()


@st.dialog("Pilih wajah untuk diuji", width="large")
def dlg_faces(s, names):
    st.caption("Klik foto orang yang ingin diuji.")
    n = len(s["img"])
    per = 5
    for r in range(0, n, per):
        cols = st.columns(per)
        for c, i in enumerate(range(r, min(r + per, n))):
            with cols[c]:
                st.image(to_u8(s["img"][i]), use_container_width=True)
                st.caption(f"#{i} · {names[int(s['y'][i])]}")
                if st.button("Pilih", key=f"fcpick{i}", use_container_width=True):
                    st.session_state["faces_idx"] = i
                    st.rerun()


def canvas_to_mnist(rgba):
    """Ubah hasil coretan canvas (RGBA) jadi citra 28x28 ala MNIST (0-1)."""
    gray = rgba[..., :3].mean(axis=2)          # coretan putih -> nilai tinggi
    ys, xs = np.where(gray > 30)
    if len(xs) == 0:
        return None
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(crop.shape)
    sq = np.zeros((s, s), dtype="uint8")
    h, w = crop.shape
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    small = np.array(Image.fromarray(sq).resize((20, 20)))   # ala MNIST: 20x20
    out = np.zeros((28, 28), dtype="float32")
    out[4:24, 4:24] = small                                  # tengahkan di 28x28
    return out / 255.0


# ----------------------------------------------------------------- Halaman
def page_taxi():
    st.subheader("🚕 Prediksi Tarif Taksi")
    st.caption("Estimasi tarif taksi NYC dari hari, jam, dan jarak tempuh.")
    model = load_model("taxi.keras")
    if model is None:
        return butuh_train("Taksi")

    hari_map = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3,
                "Jumat": 4, "Sabtu": 5, "Minggu": 6}
    with st.container(border=True):
        st.markdown("**Detail perjalanan**")
        c1, c2, c3 = st.columns(3)
        hari = c1.selectbox("Hari", list(hari_map))
        jam_obj = c2.time_input("Jam penjemputan", datetime.time(17, 0), step=3600)
        jarak = c3.number_input("Jarak tempuh (mil)", 1.0, 10.0, 2.0, 0.5)
        hitung = st.button("Hitung Tarif", type="primary")

    if hitung:
        X = np.array([[hari_map[hari], jam_obj.hour, jarak]], dtype="float32")
        fare = float(model.predict(X, verbose=0)[0][0])
        with st.container(border=True):
            st.markdown("**Hasil**")
            st.metric("Estimasi Tarif", f"${fare:.2f}",
                      help=f"{hari}, pukul {jam_obj.strftime('%H:%M')}, {jarak} mil")


def page_diabetes():
    st.subheader("❤️ Prediksi Risiko Diabetes")
    st.caption("Isi indikator klinis pasien (dataset Pima Indians).")
    model = load_model("diabetes.keras")
    scaler = load_scaler("diabetes_scaler.pkl")
    if model is None or scaler is None:
        return butuh_train("Diabetes")

    # (label, satuan, min, max, default, step)  — urutan = urutan fitur model
    fields = [("Jumlah Kehamilan", "kali", 0, 17, 3, 1),
              ("Glukosa", "mg/dL", 0, 200, 120, 1),
              ("Tekanan Darah", "mmHg", 0, 122, 70, 1),
              ("Tebal Lipatan Kulit", "mm", 0, 99, 20, 1),
              ("Insulin", "µU/mL", 0, 846, 79, 1),
              ("BMI", "kg/m²", 0.0, 67.0, 32.0, 0.1),
              ("Riwayat Keluarga", "skor", 0.0, 2.5, 0.47, 0.01),
              ("Usia", "tahun", 21, 81, 33, 1)]
    with st.form("form_diabetes", border=True):
        st.markdown("**Indikator klinis pasien**")
        cols = st.columns(2)
        vals = []
        for i, (label, unit, lo, hi, default, step) in enumerate(fields):
            vals.append(cols[i % 2].number_input(
                f"{label} ({unit})", float(lo), float(hi),
                float(default), float(step)))
        submit = st.form_submit_button("Prediksi Risiko", type="primary")

    if submit:
        arr = scaler.transform(np.array([vals], dtype="float32"))
        p = float(model.predict(arr, verbose=0)[0][0])
        with st.container(border=True):
            st.markdown("**Hasil**")
            c1, c2 = st.columns([1, 2])
            c1.metric("Probabilitas", f"{p*100:.1f}%")
            if p > 0.5:
                c2.error("⚠ Berisiko Diabetes — disarankan pemeriksaan lanjutan.")
            else:
                c2.success("✓ Risiko Rendah.")
            st.progress(min(p, 1.0))


def _saliency(model, x):
    """Kontribusi tiap fitur ke skor penipuan = nilai fitur × gradien output."""
    import tensorflow as tf
    xt = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        out = model(xt)
    grad = tape.gradient(out, xt).numpy()[0]
    return grad * x[0]  # arah & besar pengaruh tiap fitur


def page_fraud():
    st.subheader("🛡️ Deteksi Penipuan Kartu Kredit")
    st.caption("Pilih sebuah transaksi dari data uji, lalu lihat detail dan "
               "alasan keputusan model.")
    model = load_model("fraud.keras")
    samples = load_npz("fraud_samples.npz")
    if model is None or samples is None:
        return butuh_train("Penipuan")

    names = [str(n) for n in samples["names"]]

    # ---------- 1) Pilih transaksi (detail langsung terlihat di tabel) ----------
    with st.container(border=True):
        st.markdown("**📄 Daftar Transaksi (data uji)**")
        df_all = pd.DataFrame(np.round(samples["x"], 4), columns=names)
        df_all.insert(0, "ID", range(len(df_all)))
        df_all.insert(1, "Nominal ($)", np.round(samples["x"][:, -1], 2))
        st.dataframe(df_all, use_container_width=True, hide_index=True, height=240)
        st.caption("V1–V28 dianonimkan dengan PCA demi privasi pemegang kartu; "
                   "hanya *Amount* nilai asli. Pilih satu transaksi di bawah.")
        idx = st.selectbox(
            "Transaksi yang dianalisis", range(len(samples["x"])),
            format_func=lambda i: f"Transaksi #{i} · Nominal ${samples['x'][i, -1]:,.2f}")

    x = samples["x"][idx:idx + 1]
    asli = int(samples["y"][idx])
    p = float(model.predict(x, verbose=0)[0][0])
    pred = int(p > 0.5)
    nominal = float(samples["x"][idx, -1])

    # ---------- 2) Hasil analisis ----------
    with st.container(border=True):
        st.markdown("**🔎 Hasil Analisis**")
        c1, c2, c3 = st.columns(3)
        c1.metric("ID Transaksi", f"#{idx}")
        c2.metric("Nominal", f"${nominal:,.2f}")
        c3.metric("Skor Penipuan", f"{p*100:.2f}%")
        if pred:
            st.error("⚠ Transaksi ini diprediksi **PENIPUAN**.")
        else:
            st.success("✓ Transaksi ini diprediksi **SAH**.")
        st.progress(min(p, 1.0))
        benar = pred == asli
        st.caption(("✓ Sesuai label asli." if benar else "✗ Berbeda dari label asli.")
                   + f" Label sebenarnya: **{'Penipuan' if asli else 'Sah'}**.")

    # ---------- 3) Penjelasan: mengapa? ----------
    with st.container(border=True):
        st.markdown("**💡 Mengapa diputuskan begitu?**")
        contrib = _saliency(model, x)
        order = np.argsort(np.abs(contrib))[::-1][:6]
        fitur = [names[i] for i in order]
        nilai = contrib[order]

        fig, ax = plt.subplots(figsize=(6, 2.8))
        warna = ["#dc2626" if v > 0 else "#16a34a" for v in nilai]
        ax.barh(range(len(order)), nilai, color=warna)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(fitur)
        ax.invert_yaxis()
        ax.axvline(0, color="#94a3b8", linewidth=.8)
        ax.set_title("6 fitur paling berpengaruh pada keputusan")
        st.pyplot(fig)

        naik = [names[i] for i in order if contrib[i] > 0][:3]
        turun = [names[i] for i in order if contrib[i] < 0][:3]
        pesan = f"Model memberi skor penipuan **{p*100:.2f}%**. "
        if naik:
            pesan += f"Pola pada fitur **{', '.join(naik)}** paling **menaikkan** kecurigaan. "
        if turun:
            pesan += f"Sementara **{', '.join(turun)}** justru **menurunkan** kecurigaan. "
        pesan += ("Karena fitur sudah dianonimkan (PCA), nama fitur tak bermakna langsung, "
                  "tetapi model mempelajari bahwa kombinasi nilai inilah yang menyimpang "
                  "dari pola transaksi normal.")
        st.markdown(
            f"<div style='background:#f8fafc;border-left:5px solid {PRIMARY};"
            f"padding:12px 16px;border-radius:10px;color:#0f172a'>{pesan}</div>",
            unsafe_allow_html=True)
        st.markdown("<span style='color:#dc2626'>■</span> menaikkan kecurigaan &nbsp;&nbsp; "
                    "<span style='color:#16a34a'>■</span> menurunkan kecurigaan",
                    unsafe_allow_html=True)


def _bar_probs(probs, n):
    fig, ax = plt.subplots(figsize=(3.6, 3))
    ax.bar(range(n), probs, color=PRIMARY)
    ax.set_xticks(range(n))
    ax.set_ylim(0, 1)
    ax.set_title("Probabilitas tiap angka")
    return fig


def page_mnist():
    st.subheader("🔢 Pengenal Digit (MNIST)")
    st.caption("Kenali angka 0–9: gambar sendiri, unggah gambar, atau uji data uji.")
    model = load_model("digit.keras")
    samples = load_npz("mnist_samples.npz")
    if model is None or samples is None:
        return butuh_train("Digit")

    tab_draw, tab_up, tab_test = st.tabs(["✏️ Gambar", "📤 Unggah", "🎲 Data Uji"])

    # ---- Tab: gambar tangan ----
    with tab_draw:
        if not CANVAS_OK:
            st.info("Fitur gambar perlu paket tambahan. Jalankan:\n\n"
                    "```bash\npip install streamlit-drawable-canvas\n```")
        else:
            with st.container(border=True):
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**⬅️ Sebelum (coretan)**")
                    canvas = st_canvas(
                        stroke_width=18, stroke_color="#FFFFFF",
                        background_color="#000000", height=260, width=260,
                        drawing_mode="freedraw", key="mnist_canvas",
                        update_streamlit=True)
                    draw_btn = st.button("Prediksi Gambar", type="primary",
                                         key="mnist_draw_btn")
                if draw_btn:
                    arr = (canvas_to_mnist(canvas.image_data)
                           if canvas.image_data is not None else None)
                    if arr is None:
                        st.warning("Belum ada coretan. Gambar dulu sebuah angka 0–9.")
                    else:
                        probs = model.predict(arr.reshape(1, 784), verbose=0)[0]
                        pred = int(np.argmax(probs))
                        with cc2:
                            st.markdown("**➡️ Sesudah (diproses 28×28)**")
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.imshow(arr, cmap="gray")
                            ax.axis("off")
                            ax.set_title(f"Prediksi: {pred}")
                            st.pyplot(fig)
                            st.metric("Hasil", f"Angka {pred}",
                                      help=f"Keyakinan {probs[pred]*100:.1f}%")
            st.caption("Tip: gambar angka besar & tebal di tengah kotak.")

    # ---- Tab: unggah ----
    with tab_up:
        with st.container(border=True):
            up = st.file_uploader("Pilih gambar (png/jpg)",
                                  type=["png", "jpg", "jpeg"], key="mnist_up")
            if up is not None:
                orig, arr = proses_upload(up, "mnist")
                probs = model.predict(arr.reshape(1, 784), verbose=0)[0]
                pred = int(np.argmax(probs))
                tampil_sebelum_sesudah(orig, arr, f"Prediksi: {pred}")
                st.metric("Hasil Prediksi", f"Angka {pred}",
                          help=f"Keyakinan {probs[pred]*100:.1f}%")
        st.caption("Gambar contoh tersedia di folder `samples/mnist/`.")

    # ---- Tab: data uji ----
    with tab_test:
        with st.container(border=True):
            idx, pilih = kontrol_sampel("mnist", len(samples["x"]))
            if pilih:
                dlg_mnist(samples)
            if idx is None:
                st.info("Tekan **Acak** atau **Pilih** untuk menguji dari data uji.")
            else:
                probs = model.predict(samples["x"][idx:idx + 1], verbose=0)[0]
                pred = int(np.argmax(probs))
                asli = int(samples["y"][idx])
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(samples["img"][idx], cmap="gray")
                    ax.axis("off")
                    ax.set_title(f"Asli: {asli} | Prediksi: {pred}",
                                 color="#16a34a" if pred == asli else "#dc2626")
                    st.pyplot(fig)
                with c2:
                    st.pyplot(_bar_probs(probs, 10))


def page_faces():
    st.subheader("👤 Pengenalan Wajah (LFW)")
    st.caption("Kenali 5 tokoh dunia: unggah foto atau uji dari data uji.")
    model = load_model("model_wajah.keras")
    samples = load_npz("faces_samples.npz")
    meta = load_json("faces_meta.json")
    if model is None or samples is None or meta is None:
        return butuh_train("Wajah")

    names = meta["target_names"]
    tab_up, tab_test = st.tabs(["📤 Unggah", "🎲 Data Uji"])

    # ---- Tab: unggah ----
    with tab_up:
        with st.container(border=True):
            up = st.file_uploader("Pilih gambar (png/jpg)",
                                  type=["png", "jpg", "jpeg"], key="faces_up")
            if up is not None:
                orig, arr = proses_upload(up, "faces")
                probs = model.predict(arr.reshape(1, 47 * 62), verbose=0)[0]
                pred = int(np.argmax(probs))
                tampil_sebelum_sesudah(orig, arr, names[pred])
                st.metric("Hasil Prediksi", names[pred],
                          help=f"Keyakinan {probs[pred]*100:.1f}%")
        st.caption("Model mengenali: " + ", ".join(names)
                   + ". Foto lain akan dipaksa ke salah satunya. "
                   "Contoh di folder `samples/faces/`.")

    # ---- Tab: data uji ----
    with tab_test:
        with st.container(border=True):
            idx, pilih = kontrol_sampel("faces", len(samples["x"]))
            if pilih:
                dlg_faces(samples, names)
            if idx is None:
                st.info("Tekan **Acak** atau **Pilih** untuk menguji dari data uji.")
            else:
                pred = int(model.predict(samples["x"][idx:idx + 1], verbose=0).argmax())
                asli = int(samples["y"][idx])
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig, ax = plt.subplots(figsize=(3, 4))
                    ax.imshow(samples["img"][idx], cmap="gray")
                    ax.axis("off")
                    st.pyplot(fig)
                with c2:
                    st.metric("Prediksi", names[pred])
                    st.metric("Asli", names[asli])
                    if pred == asli:
                        st.success("✓ Tebakan benar.")
                    else:
                        st.error("✗ Tebakan salah.")


# ----------------------------------------------------------------- Tata Letak
st.markdown(
    "<div class='hero'><h1>🧠 Dashboard Prediksi Neural Network</h1>"
    "<p>Chapter 9 — Applied Machine Learning · Maizan &amp; Dicky</p></div>",
    unsafe_allow_html=True)

HALAMAN = {
    "🚕 Tarif Taksi": page_taxi,
    "❤️ Risiko Diabetes": page_diabetes,
    "🛡️ Deteksi Penipuan": page_fraud,
    "🔢 Pengenal Digit": page_mnist,
    "👤 Pengenalan Wajah": page_faces,
}

with st.sidebar:
    st.header("Menu")
    pilihan = st.radio("Pilih model:", list(HALAMAN), label_visibility="collapsed")
    st.divider()
    st.caption("Tiap model dilatih dari data nyata. "
               "Jalankan `python train.py` sekali sebelum memakai aplikasi.")

HALAMAN[pilihan]()
