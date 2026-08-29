"""
app.py
Web App Prediksi Kelayakan Air Minum (Water Potability) menggunakan Streamlit
Jalankan dengan: streamlit run app.py
"""
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import plotly.graph_objects as go

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Kelayakan Air Minum",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Font & background umum */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Poppins', sans-serif;
    }

    /* Latar belakang: gradasi biru + pola gelembung air lembut */
    .stApp {
        background:
            radial-gradient(circle at 8% 15%, rgba(14,165,233,0.10) 0px, rgba(14,165,233,0.10) 40px, transparent 41px),
            radial-gradient(circle at 85% 25%, rgba(56,189,248,0.12) 0px, rgba(56,189,248,0.12) 55px, transparent 56px),
            radial-gradient(circle at 20% 75%, rgba(2,132,199,0.08) 0px, rgba(2,132,199,0.08) 30px, transparent 31px),
            radial-gradient(circle at 92% 80%, rgba(14,165,233,0.10) 0px, rgba(14,165,233,0.10) 45px, transparent 46px),
            radial-gradient(circle at 55% 92%, rgba(56,189,248,0.10) 0px, rgba(56,189,248,0.10) 25px, transparent 26px),
            linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 40%, #f8fafc 100%);
        background-attachment: fixed;
    }

    /* Bola air dekoratif melayang */
    @keyframes float-bubble {
        0%   { transform: translateY(0px) scale(1); opacity: 0.55; }
        50%  { transform: translateY(-18px) scale(1.05); opacity: 0.85; }
        100% { transform: translateY(0px) scale(1); opacity: 0.55; }
    }
    .bubble {
        position: fixed;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.9), rgba(56,189,248,0.35) 60%, rgba(3,105,161,0.25) 100%);
        z-index: 0;
        pointer-events: none;
        animation: float-bubble 6s ease-in-out infinite;
    }
    .b1 { width: 46px; height: 46px; top: 8%;  left: 4%;  animation-delay: 0s; }
    .b2 { width: 26px; height: 26px; top: 18%; right: 6%; animation-delay: 1.2s; }
    .b3 { width: 60px; height: 60px; bottom: 12%; left: 3%; animation-delay: 0.6s; }
    .b4 { width: 32px; height: 32px; bottom: 22%; right: 4%; animation-delay: 1.8s; }

    /* ============================================================
       DEKORASI KARTUN ELEMENTAL — "Forces of Nature"
       Ikon melayang bergaya kartun: matahari (api/energi), daun
       (tanah/alam), awan & angin (udara), dan petir (energi alam).
       ============================================================ */
    @keyframes float-elemental {
        0%   { transform: translateY(0px) rotate(-4deg) scale(1); }
        50%  { transform: translateY(-22px) rotate(4deg) scale(1.08); }
        100% { transform: translateY(0px) rotate(-4deg) scale(1); }
    }
    @keyframes drift-cloud {
        0%   { transform: translateX(0px) translateY(0px); }
        50%  { transform: translateX(14px) translateY(-10px); }
        100% { transform: translateX(0px) translateY(0px); }
    }
    @keyframes flicker-bolt {
        0%, 100% { opacity: 0.8; filter: drop-shadow(0 0 6px rgba(250,204,21,0.6)); }
        50%      { opacity: 1;   filter: drop-shadow(0 0 14px rgba(250,204,21,0.9)); }
    }
    .elemental {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        line-height: 1;
        filter: drop-shadow(0 4px 6px rgba(15,23,42,0.15));
        user-select: none;
    }
    .elem-sun {
        top: 6%;
        right: 10%;
        font-size: 3rem;
        animation: float-elemental 7s ease-in-out infinite;
        opacity: 0.85;
    }
    .elem-leaf {
        bottom: 8%;
        left: 7%;
        font-size: 2.4rem;
        animation: float-elemental 5.5s ease-in-out infinite;
        animation-delay: 0.8s;
        opacity: 0.8;
    }
    .elem-cloud {
        top: 14%;
        left: 6%;
        font-size: 2.6rem;
        animation: drift-cloud 8s ease-in-out infinite;
        opacity: 0.75;
    }
    .elem-wind {
        bottom: 30%;
        right: 5%;
        font-size: 2.2rem;
        animation: drift-cloud 6.5s ease-in-out infinite;
        animation-delay: 1s;
        opacity: 0.7;
    }
    .elem-bolt {
        top: 42%;
        left: 2.5%;
        font-size: 2rem;
        animation: flicker-bolt 2.4s ease-in-out infinite;
        opacity: 0.85;
    }

    /* ============================================================
       NAVBAR — tombol profil kelompok (tertutup secara default)
       ============================================================ */
    .navbar-wrap {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0.2rem;
        margin-bottom: 0.6rem;
    }
    .navbar-brand {
        font-weight: 800;
        font-size: 1.05rem;
        color: #0369a1;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    div.stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.9);
        color: #0369a1;
        font-weight: 700;
        border: 1.5px solid #7dd3fc;
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        box-shadow: 0 3px 10px rgba(2,132,199,0.12);
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #e0f2fe;
        border-color: #0ea5e9;
        color: #075985;
        transform: translateY(-1px);
    }

    @keyframes slide-down {
        from { opacity: 0; transform: translateY(-14px); }
        to   { opacity: 1; transform: translateY(0px); }
    }
    .profile-panel {
        position: relative;
        z-index: 2;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(4px);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        border: 1px solid #bae6fd;
        box-shadow: 0 8px 24px rgba(2,132,199,0.14);
        margin-bottom: 1.4rem;
        animation: slide-down 0.28s ease-out;
    }
    .profile-header {
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px dashed #bae6fd;
    }
    .profile-kelompok {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
    }
    .profile-prodi {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.2rem;
    }
    .member-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem;
        text-align: center;
        height: 100%;
    }
    .member-card img {
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }
    .member-photo-placeholder {
        background: linear-gradient(135deg, #e0f2fe, #bae6fd);
        border-radius: 12px;
        padding: 1.8rem 0.5rem;
        margin-bottom: 0.6rem;
        font-size: 2.2rem;
        color: #0369a1;
    }
    .member-photo-placeholder span {
        display: block;
        font-size: 0.72rem;
        color: #0369a1;
        margin-top: 0.3rem;
        font-weight: 600;
    }
    .member-name {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.95rem;
    }
    .member-nim {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.1rem;
    }

    /* Header hero dengan foto air sebagai latar */
    .hero {
        position: relative;
        text-align: center;
        padding: 3rem 1.5rem 3.4rem 1.5rem;
        border-radius: 22px;
        overflow: hidden;
        color: white;
        margin-bottom: 0;
        box-shadow: 0 10px 30px rgba(2, 132, 199, 0.25);
        background-image:
            linear-gradient(135deg, rgba(2,60,90,0.72) 0%, rgba(3,105,161,0.75) 55%, rgba(2,132,199,0.65) 100%),
            url('https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
    }
    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    .hero p {
        font-size: 0.98rem;
        opacity: 0.95;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.5;
        text-shadow: 0 1px 6px rgba(0,0,0,0.2);
    }
    /* Gelombang di bagian bawah hero */
    .wave-wrap {
        margin-top: -18px;
        margin-bottom: 1.6rem;
        line-height: 0;
    }

    /* Kartu section */
    .section-card {
        position: relative;
        z-index: 1;
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(4px);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #0f172a;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Kartu info kecil dengan ikon */
    .info-strip {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }
    .info-chip {
        flex: 1;
        min-width: 140px;
        background: rgba(255,255,255,0.85);
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(2,132,199,0.06);
    }
    .info-chip .icon { font-size: 1.4rem; }
    .info-chip .label { font-size: 0.78rem; color: #475569; margin-top: 2px; }

    /* Tombol utama */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0369a1);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 0;
        box-shadow: 0 6px 16px rgba(3, 105, 161, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(3, 105, 161, 0.45);
        color: white;
    }

    /* Input styling */
    div[data-baseweb="input"] input, .stNumberInput input {
        border-radius: 10px !important;
    }

    /* Hasil card */
    .result-card {
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
    }
    .result-good {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #86efac;
    }
    .result-bad {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #fca5a5;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-good .result-title { color: #15803d; }
    .result-bad .result-title { color: #b91c1c; }
    .result-sub {
        font-size: 1rem;
        color: #334155;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .custom-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL, SCALER, & NILAI IMPUTASI
# ============================================================
@st.cache_resource
def load_artifacts():
    # Model disimpan dalam format native XGBoost (.json), bukan pickle,
    # supaya tidak bermasalah kalau versi xgboost di komputer ini berbeda
    # dengan versi yang dipakai saat training.
    model = XGBClassifier()
    model.load_model("model.json")
    scaler = joblib.load("scaler.pkl")
    median_imputasi = joblib.load("median_imputasi.pkl")
    return model, scaler, median_imputasi

model, scaler, median_imputasi = load_artifacts()

# ============================================================
# DEKORASI: gelembung air + ikon kartun elemental (forces of nature)
# ============================================================
st.markdown("""
<div class="bubble b1"></div>
<div class="bubble b2"></div>
<div class="bubble b3"></div>
<div class="bubble b4"></div>

<div class="elemental elem-sun">☀️</div>
<div class="elemental elem-cloud">☁️</div>
<div class="elemental elem-leaf">🍃</div>
<div class="elemental elem-wind">🌬️</div>
<div class="elemental elem-bolt">⚡</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVBAR — tombol Profil Kelompok (tertutup secara default)
# ============================================================
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False

st.markdown('<div class="navbar-wrap">', unsafe_allow_html=True)
nav_col1, nav_col2 = st.columns([3, 1.3])
with nav_col1:
    st.markdown('<div class="navbar-brand">💧 Water Potability 💧</div>', unsafe_allow_html=True)
with nav_col2:
    label_tombol = "▲ Tutup Profil" if st.session_state.show_profile else "👥 Profil Kelompok"
    if st.button(label_tombol, use_container_width=True, type="secondary", key="btn_toggle_profile"):
        st.session_state.show_profile = not st.session_state.show_profile
st.markdown('</div>', unsafe_allow_html=True)

# Panel profil kelompok — hanya tampil kalau tombol di atas ditekan
if st.session_state.show_profile:
    st.markdown('<div class="profile-panel">', unsafe_allow_html=True)
    st.markdown("""
        <div class="profile-header">
            <div class="profile-kelompok">Kelompok 4 - 3AEC2</div>
            <div class="profile-prodi">Program Studi Teknologi Rekayasa Informatika Industri</div>
        </div>
    """, unsafe_allow_html=True)

    # Daftar anggota kelompok. Taruh foto masing-masing anggota di folder
    # "assets/" dengan nama file sesuai path di bawah ini. Kalau foto belum
    # ada, otomatis ditampilkan placeholder supaya app tetap jalan normal.
    anggota = [
        {"nama": "Dien Putri Alexa", "nim": "224443028", "foto": "assets/dien.jpg"},
        {"nama": "Salsabila Ramadhani Gusmi", "nim": "224443044", "foto": "assets/salsabila.jpg"},
    ]

    member_cols = st.columns(len(anggota))
    for col, m in zip(member_cols, anggota):
        with col:
            st.markdown('<div class="member-card">', unsafe_allow_html=True)
            if os.path.exists(m["foto"]):
                st.image(m["foto"], use_container_width=True)
            else:
                st.markdown(
                    '<div class="member-photo-placeholder">🧑‍🎓'
                    '<span>Foto belum ditambahkan</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"""
                <div class="member-name">{m['nama']}</div>
                <div class="member-nim">NIM: {m['nim']}</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# HERO HEADER (dengan foto latar & gelombang)
# ============================================================
st.markdown("""
<div class="hero">
    <h1>💧 Prediksi Kelayakan Air Minum</h1>
    <p>Masukkan hasil uji kualitas air pada form di bawah, dan model Machine Learning
    (XGBoost) akan memprediksi apakah air tersebut layak diminum atau tidak.</p>
</div>
<div class="wave-wrap">
<svg viewBox="0 0 1440 90" xmlns="http://www.w3.org/2000/svg">
    <path fill="#e0f2fe" d="M0,32L60,42.7C120,53,240,75,360,74.7C480,75,600,53,720,42.7C840,32,960,32,1080,37.3C1200,43,1320,53,1380,58.7L1440,64L1440,100L1380,100C1320,100,1200,100,1080,100C960,100,840,100,720,100C600,100,480,100,360,100C240,100,120,100,60,100L0,100Z"></path>
</svg>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STRIP INFO SINGKAT
# ============================================================
st.markdown("""
<div class="info-strip">
    <div class="info-chip"><div class="icon">🧫</div><div class="label">9 Parameter Kualitas Air</div></div>
    <div class="info-chip"><div class="icon">🤖</div><div class="label">Model XGBoost</div></div>
    <div class="info-chip"><div class="icon">⚡</div><div class="label">Hasil Instan</div></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FORM INPUT
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Parameter Fisika & Kimia</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH Air", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    ph_unknown = st.checkbox("Tidak tahu nilai pH")
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=400.0, value=196.0, step=1.0)
    solids = st.number_input("Solids / TDS (ppm)", min_value=0.0, max_value=61000.0, value=20000.0, step=100.0)
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=333.0, step=1.0)
    sulfate_unknown = st.checkbox("Tidak tahu nilai Sulfate")

with col2:
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=800.0, value=425.0, step=1.0)
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=14.0, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=130.0, value=66.0, step=0.1)
    trihalomethanes_unknown = st.checkbox("Tidak tahu nilai Trihalomethanes")
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=7.0, value=4.0, step=0.1)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TOMBOL PREDIKSI
# ============================================================
predict_clicked = st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary")

if predict_clicked:
    input_data = pd.DataFrame([{
        "ph": median_imputasi["ph"] if ph_unknown else ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": median_imputasi["Sulfate"] if sulfate_unknown else sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": median_imputasi["Trihalomethanes"] if trihalomethanes_unknown else trihalomethanes,
        "Turbidity": turbidity,
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-good">
            <div class="result-title">✅ Air Layak Diminum</div>
            <div class="result-sub">Probabilitas layak minum: <b>{proba*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        gauge_color = "#22c55e"
    else:
        st.markdown(f"""
        <div class="result-card result-bad">
            <div class="result-title">⚠️ Air Tidak Layak Diminum</div>
            <div class="result-sub">Probabilitas layak minum: <b>{proba*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        gauge_color = "#ef4444"

    # Gauge chart untuk memvisualisasikan probabilitas
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": gauge_color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 3},
                "thickness": 0.8,
                "value": proba * 100,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Poppins, sans-serif"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grafik batang perbandingan nilai input vs ambang batas umum (WHO/Permenkes)
    ref_values = {
        "pH": (input_data["ph"].iloc[0], 8.5),
        "Hardness": (input_data["Hardness"].iloc[0], 300),
        "Chloramines": (input_data["Chloramines"].iloc[0], 4),
        "Sulfate": (input_data["Sulfate"].iloc[0], 250),
        "Turbidity": (input_data["Turbidity"].iloc[0], 5),
    }
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Nilai Input Anda",
        x=list(ref_values.keys()),
        y=[v[0] for v in ref_values.values()],
        marker_color="#0ea5e9",
    ))
    fig_bar.add_trace(go.Bar(
        name="Ambang Batas Umum",
        x=list(ref_values.keys()),
        y=[v[1] for v in ref_values.values()],
        marker_color="#cbd5e1",
    ))
    fig_bar.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title="📊 Perbandingan Nilai Input vs Ambang Batas Umum",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font={"family": "Segoe UI, Poppins, sans-serif"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        "Catatan: Hasil ini adalah prediksi model machine learning berdasarkan data statistik, "
        "bukan hasil uji laboratorium resmi. Untuk kepastian, gunakan uji kualitas air standar."
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="custom-footer">
    Dibuat dengan ❤️ 
</div>
""", unsafe_allow_html=True)