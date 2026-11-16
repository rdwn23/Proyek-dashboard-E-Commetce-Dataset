from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")
sns.set_style("whitegrid")


@st.cache_data
def load_data():
    path = Path(__file__).parent / "main_data.csv"
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def build_rfm(df):
    ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (ref - rfm["last_purchase"]).dt.days
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    conditions = [
        rfm["frequency"] >= 2,
        (rfm["frequency"] == 1) & (rfm["r_score"] >= 3) & (rfm["m_score"] >= 3),
        (rfm["frequency"] == 1) & (rfm["r_score"] >= 3) & (rfm["m_score"] < 3),
        (rfm["frequency"] == 1) & (rfm["r_score"] < 3) & (rfm["m_score"] >= 3),
    ]
    segments = ["Loyal Customers", "New High Spenders", "Recent Low Spenders", "At Risk High Spenders"]
    rfm["segment"] = np.select(conditions, segments, default="Lost / Low Value")

    seg = (
        rfm.groupby("segment")
        .agg(customers=("customer_unique_id", "count"), total_monetary=("monetary", "sum"))
        .reset_index()
    )
    seg["pct_customers"] = seg["customers"] / seg["customers"].sum() * 100
    seg["pct_monetary"] = seg["total_monetary"] / seg["total_monetary"].sum() * 100
    return rfm, seg


df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Filter")
min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Rentang tanggal pesanan", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
top_n = st.sidebar.slider("Jumlah kategori ditampilkan (Top/Bottom)", 3, 10, 5)

if len(date_range) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir pada filter di sidebar.")
    st.stop()

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
data = df[(df["order_purchase_timestamp"] >= start) & (df["order_purchase_timestamp"] < end)]

if data.empty:
    st.warning("Tidak ada data pada rentang tanggal tersebut.")
    st.stop()

# ---------------- Header & metrik ----------------
st.title("Dashboard Analisis E-Commerce")
st.caption("Pesanan berstatus delivered, Januari-Agustus 2018. Seluruh nilai dalam Real Brasil (BRL).")

total_revenue = data["price"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("Total Revenue (BRL)", f"{total_revenue:,.0f}")
c2.metric("Jumlah Pesanan", f"{data['order_id'].nunique():,}")
c3.metric("Jumlah Pelanggan", f"{data['customer_unique_id'].nunique():,}")

# ---------------- Pertanyaan 1 ----------------
st.header("Pertanyaan 1: Revenue per Kategori Produk")

category_revenue = (
    data[data["product_category_name_english"] != "unknown"]
    .groupby("product_category_name_english")["price"]
    .sum()
    .sort_values(ascending=False)
    .reset_index(name="revenue")
)

top = category_revenue.head(top_n)
bottom = category_revenue.tail(top_n).sort_values("revenue", ascending=False)
top5_share = category_revenue.head(5)["revenue"].sum() / total_revenue * 100

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=top, x="revenue", y="product_category_name_english", color="#1f77b4", ax=ax)
    ax.set_title(f"{top_n} Kategori Revenue Tertinggi")
    ax.set_xlabel("Revenue (BRL)")
    ax.set_ylabel("")
    st.pyplot(fig)
with col_b:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=bottom, x="revenue", y="product_category_name_english", color="#d62728", ax=ax)
    ax.set_title(f"{top_n} Kategori Revenue Terendah")
    ax.set_xlabel("Revenue (BRL)")
    ax.set_ylabel("")
    st.pyplot(fig)

col_c, col_d = st.columns([1, 2])
with col_c:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(
        [top5_share, 100 - top5_share],
        labels=["Top 5 kategori", "Lainnya"],
        autopct="%1.1f%%",
        colors=["#1f77b4", "#c7c7c7"],
        startangle=90,
        wedgeprops={"width": 0.45},
    )
    ax.set_title("Kontribusi Top 5 terhadap Total Revenue")
    st.pyplot(fig)
with col_d:
    st.subheader("Ringkasan")
    st.write(
        f"Lima kategori teratas menyumbang **{top5_share:.2f}%** dari total revenue pada periode yang dipilih. "
        "Kategori `unknown` (produk tanpa kategori) tidak diikutkan dalam ranking."
    )
    st.dataframe(category_revenue.head(10).rename(columns={"product_category_name_english": "kategori"}))

# ---------------- Pertanyaan 2 ----------------
st.header("Pertanyaan 2: Segmentasi Pelanggan (RFM)")

if data["customer_unique_id"].nunique() < 4:
    st.warning("Jumlah pelanggan terlalu sedikit untuk segmentasi RFM. Perlebar rentang tanggal.")
else:
    rfm_df, segment_df = build_rfm(data)
    order = segment_df.sort_values("pct_monetary", ascending=False)["segment"].tolist()
    palette = {s: ("#1f77b4" if s in order[:2] else "#c7c7c7") for s in order}

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for a, col, title in zip(
        ax, ["pct_customers", "pct_monetary"], ["Persentase Pelanggan per Segmen", "Persentase Total Monetary per Segmen"]
    ):
        sns.barplot(data=segment_df, x=col, y="segment", order=order, hue="segment", palette=palette, legend=False, ax=a)
        a.set_title(title)
        a.set_xlabel("%")
        a.set_ylabel("")
        a.set_xlim(0, 100 if segment_df[col].max() > 50 else 50)
        for i, seg in enumerate(order):
            v = segment_df.loc[segment_df["segment"] == seg, col].values[0]
            a.text(v, i, f" {v:.1f}%", va="center", fontsize=9)
    st.pyplot(fig)

    best = segment_df.sort_values("pct_monetary", ascending=False).iloc[0]
    st.write(
        f"Segmen dengan kontribusi monetary terbesar adalah **{best['segment']}** "
        f"({best['pct_monetary']:.1f}% monetary dari {best['pct_customers']:.1f}% pelanggan). "
        "Dua segmen berwarna biru adalah prioritas program retensi."
    )
    st.dataframe(segment_df.round(2))