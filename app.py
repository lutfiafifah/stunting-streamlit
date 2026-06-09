import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ====================================================
# CONFIG
# ====================================================
st.set_page_config(
    page_title="Dashboard Stunting - Naive Bayes",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================
# CSS
# ====================================================
CUSTOM_CSS = """
<style>
.main {
    background: linear-gradient(180deg, #f8fbff 0%, #ffffff 45%, #f7f9fc 100%);
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}

.hero-card {
    background: linear-gradient(135deg, #155e75 0%, #2563eb 52%, #7c3aed 100%);
    color: white;
    padding: 28px 34px;
    border-radius: 26px;
    box-shadow: 0 18px 45px rgba(37, 99, 235, 0.22);
    margin-bottom: 22px;
}

.metric-card {
    background: #ffffff;
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 700;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
}

[data-testid="stSidebar"] {
    background: #f8fafc;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====================================================
# KONSTANTA
# ====================================================
REQUIRED_COLS = [
    "Umur (bulan)",
    "Jenis Kelamin",
    "Tinggi Badan (cm)",
    "Status Gizi"
]

MONTHS = [
    "Januari", "Februari", "Maret", "April",
    "Mei", "Juni", "Juli", "Agustus",
    "September", "Oktober", "November", "Desember"
]

# ====================================================
# LOAD DATA
# ====================================================
@st.cache_data
def load_default_data():
    return pd.read_csv("data_stunting.csv")

# ====================================================
# CLEAN DATA
# ====================================================
def clean_data(df):

    if df.empty:
        return df

    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    df["Umur (bulan)"] = pd.to_numeric(
        df["Umur (bulan)"],
        errors="coerce"
    )

    df["Tinggi Badan (cm)"] = pd.to_numeric(
        df["Tinggi Badan (cm)"],
        errors="coerce"
    )

    df["Jenis Kelamin"] = (
        df["Jenis Kelamin"]
        .astype(str)
        .str.upper()
    )

    df["Status Gizi"] = (
        df["Status Gizi"]
        .astype(str)
    )

    df = df.dropna().reset_index(drop=True)

    if "Baris" not in df.columns:
        df.insert(0, "Baris", range(1, len(df) + 1))

    # otomatis tambah bulan
    if "Bulan" not in df.columns:
        df["Bulan"] = [
            MONTHS[i % 12]
            for i in range(len(df))
        ]

    return df

# ====================================================
# METRIC CARD
# ====================================================
def metric_card(label, value):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ====================================================
# SESSION STATE
# ====================================================
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "data_deleted" not in st.session_state:
    st.session_state["data_deleted"] = False

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4320/4320337.png",
        width=100
    )

    st.title("Pengaturan Data")

    uploaded = st.file_uploader(
        "Upload CSV baru",
        type=["csv"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded is not None:

        try:
            uploaded_df = pd.read_csv(uploaded)

            missing = [
                c for c in REQUIRED_COLS
                if c not in uploaded_df.columns
            ]

            if missing:
                st.error(
                    "Kolom kurang: "
                    + ", ".join(missing)
                )

            else:
                st.session_state["uploaded_data"] = uploaded_df
                st.session_state["data_deleted"] = False

                st.success(
                    "Dataset berhasil diupload!"
                )

        except Exception as e:
            st.error(e)

    if st.button("🗑️ Hapus / Ganti Data"):

        st.session_state["uploaded_data"] = (
            pd.DataFrame(columns=REQUIRED_COLS)
        )

        st.session_state["data_deleted"] = True

        st.session_state["uploader_key"] += 1

        st.rerun()

# ====================================================
# PILIH DATA
# ====================================================
if st.session_state["uploaded_data"] is not None:

    raw_df = st.session_state["uploaded_data"]

elif st.session_state["data_deleted"]:

    raw_df = pd.DataFrame(columns=REQUIRED_COLS)

else:

    raw_df = load_default_data()

df = clean_data(raw_df)

# ====================================================
# DATA KOSONG
# ====================================================
if df.empty:

    st.warning(
        "Dataset kosong. Upload data baru."
    )

    st.stop()

# ====================================================
# FILTER DASHBOARD
# ====================================================
with st.sidebar:

    st.divider()

    st.subheader("Filter Dashboard")

    # FILTER BULAN
    selected_month = st.selectbox(
        "Pilih Bulan",
        ["Semua Bulan"] + MONTHS
    )

    status_options = sorted(
        df["Status Gizi"]
        .dropna()
        .unique()
        .tolist()
    )

    gender_options = sorted(
        df["Jenis Kelamin"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_status = st.multiselect(
        "Status Gizi",
        status_options,
        default=status_options
    )

    selected_gender = st.multiselect(
        "Jenis Kelamin",
        gender_options,
        default=gender_options
    )

    # FILTER UMUR
    umur_min = int(df["Umur (bulan)"].min())
    umur_max = int(df["Umur (bulan)"].max())

    if umur_min == umur_max:
        umur_max += 1

    umur_range = st.slider(
        "Rentang Umur (bulan)",
        min_value=umur_min,
        max_value=umur_max,
        value=(umur_min, umur_max)
    )

    # FILTER TINGGI
    tinggi_min = float(
        np.floor(
            df["Tinggi Badan (cm)"].min()
        )
    )

    tinggi_max = float(
        np.ceil(
            df["Tinggi Badan (cm)"].max()
        )
    )

    if tinggi_min == tinggi_max:
        tinggi_max += 1.0

    tinggi_range = st.slider(
        "Rentang Tinggi Badan (cm)",
        min_value=tinggi_min,
        max_value=tinggi_max,
        value=(tinggi_min, tinggi_max)
    )

# ====================================================
# FILTER DATA
# ====================================================
df = df[
    (
        df["Status Gizi"]
        .isin(selected_status)
    )
    &
    (
        df["Jenis Kelamin"]
        .isin(selected_gender)
    )
    &
    (
        df["Umur (bulan)"]
        .between(
            umur_range[0],
            umur_range[1]
        )
    )
    &
    (
        df["Tinggi Badan (cm)"]
        .between(
            tinggi_range[0],
            tinggi_range[1]
        )
    )
]

# FILTER BULAN
if selected_month != "Semua Bulan":
    df = df[
        df["Bulan"] == selected_month
    ]

# ====================================================
# HEADER
# ====================================================
st.markdown("""
<div class="hero-card">
<h1>Dashboard Grafik Klasifikasi Stunting</h1>
<p>
Visualisasi data balita stunting dan simulasi
klasifikasi menggunakan metode Naive Bayes.
</p>
</div>
""", unsafe_allow_html=True)

# ====================================================
# DATA FILTER KOSONG
# ====================================================
if df.empty:

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Data", "0")

    with c2:
        metric_card("Rata-rata Umur", "-")

    with c3:
        metric_card("Rata-rata Tinggi", "-")

    with c4:
        metric_card("Status Terbanyak", "-")

    st.warning(
        "Tidak ada data yang sesuai filter."
    )

    st.stop()

# ====================================================
# METRICS
# ====================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "Total Data",
        len(df)
    )

with c2:
    metric_card(
        "Rata-rata Umur",
        round(
            df["Umur (bulan)"].mean(),
            1
        )
    )

with c3:
    metric_card(
        "Rata-rata Tinggi",
        round(
            df["Tinggi Badan (cm)"].mean(),
            1
        )
    )

with c4:
    metric_card(
        "Status Terbanyak",
        df["Status Gizi"].mode()[0]
    )

# ====================================================
# TABS
# ====================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Grafik",
    "🤖 Naive Bayes",
    "📄 Data"
])

# ====================================================
# TAB GRAFIK
# ====================================================
with tab1:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names="Status Gizi",
            title="Distribusi Status Gizi"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig2 = px.scatter(
            df,
            x="Umur (bulan)",
            y="Tinggi Badan (cm)",
            color="Status Gizi",
            symbol="Jenis Kelamin"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # GRAFIK PER BULAN
    st.subheader("Grafik Status Gizi Per Bulan")

    monthly_data = (
        df.groupby(["Bulan", "Status Gizi"])
        .size()
        .reset_index(name="Jumlah")
    )

    fig_month = px.bar(
        monthly_data,
        x="Bulan",
        y="Jumlah",
        color="Status Gizi",
        barmode="group",
        category_orders={
            "Bulan": MONTHS
        },
        title="Jumlah Status Gizi Per Bulan"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

# ====================================================
# TAB NAIVE BAYES
# ====================================================
with tab2:

    X = df[[
        "Umur (bulan)",
        "Jenis Kelamin",
        "Tinggi Badan (cm)"
    ]]

    y = df["Status Gizi"]

    if len(df) < 5:

        st.warning(
            "Data terlalu sedikit untuk training model."
        )

    else:

        preprocessor = ColumnTransformer([
            (
                "num",
                StandardScaler(),
                [
                    "Umur (bulan)",
                    "Tinggi Badan (cm)"
                ]
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                ["Jenis Kelamin"]
            )
        ])

        model = Pipeline([
            ("prep", preprocessor),
            ("model", GaussianNB())
        ])

        stratify = (
            y
            if y.value_counts().min() >= 2
            else None
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify
            )
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(
            y_test,
            pred
        )

        st.success(
            f"Akurasi Model: {acc*100:.2f}%"
        )

        cm = confusion_matrix(
            y_test,
            pred
        )

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

        report = classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )

        # PREDIKSI MANUAL
        st.subheader("Prediksi Manual")

        c1, c2, c3 = st.columns(3)

        with c1:
            umur_input = st.number_input(
                "Umur",
                0,
                72,
                24
            )

        with c2:
            jk_input = st.selectbox(
                "Jenis Kelamin",
                sorted(
                    df["Jenis Kelamin"]
                    .unique()
                )
            )

        with c3:
            tinggi_input = st.number_input(
                "Tinggi",
                30.0,
                130.0,
                80.0
            )

        input_df = pd.DataFrame({
            "Umur (bulan)": [umur_input],
            "Jenis Kelamin": [jk_input],
            "Tinggi Badan (cm)": [tinggi_input]
        })

        hasil = model.predict(input_df)[0]

        st.success(
            f"Hasil Prediksi: {hasil}"
        )

        prob = model.predict_proba(
            input_df
        )[0]

        prob_df = pd.DataFrame({
            "Status Gizi": model.classes_,
            "Probabilitas": prob
        })

        fig_prob = px.bar(
            prob_df,
            x="Status Gizi",
            y="Probabilitas",
            text_auto=True
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )

# ====================================================
# TAB DATA
# ====================================================
with tab3:

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="data_stunting_filter.csv",
        mime="text/csv"
    )