import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Dashboard Analisis Angket Siswa",
    layout="wide"
)

st.title("📊 Dashboard Analisis Angket Siswa")
st.markdown("Analisis hasil angket pembelajaran berbasis data")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal (1).xlsx")

# Pastikan semua numerik
df = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# KPI UTAMA
# ==========================================================
mean_all = df.mean().mean()
indeks_kepuasan = (mean_all / 5) * 100

def kategori(x):
    if x >= 85:
        return "Sangat Baik"
    elif x >= 70:
        return "Baik"
    elif x >= 55:
        return "Cukup"
    else:
        return "Kurang"

col1, col2, col3 = st.columns(3)
col1.metric("📈 Rata-rata Skor", f"{mean_all:.2f}")
col2.metric("📊 Indeks Kepuasan", f"{indeks_kepuasan:.2f}%")
col3.metric("🏷️ Kategori", kategori(indeks_kepuasan))

st.divider()

# ==========================================================
# RATA-RATA PER SOAL
# ==========================================================
st.header("1️⃣ Rata-rata Skor Tiap Soal")

mean_per_item = df.mean()

fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.bar(mean_per_item.index.astype(str), mean_per_item.values)
ax1.set_ylim(0,5)
ax1.set_ylabel("Skor Rata-rata")
ax1.set_xlabel("Butir Soal")
ax1.grid(axis="y", linestyle="--", alpha=0.6)

st.pyplot(fig1)

st.divider()

# ==========================================================
# ANALISIS GAP
# ==========================================================
st.header("2️⃣ Analisis GAP")

gap = 5 - mean_per_item
prioritas = gap.idxmax()

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(gap.index.astype(str), gap.values)
ax2.set_ylabel("Nilai GAP")
ax2.set_xlabel("Butir Soal")
ax2.grid(axis="y", linestyle="--", alpha=0.6)

st.pyplot(fig2)
st.info(f"📌 Soal prioritas perbaikan: **{prioritas}**")

st.divider()

# ==========================================================
# KORELASI ANTAR SOAL
# ==========================================================
st.header("3️⃣ Korelasi Antar Soal")

corr = df.corr()

fig3, ax3 = plt.subplots(figsize=(7,6))
im = ax3.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax3)

ax3.set_xticks(range(len(corr.columns)))
ax3.set_yticks(range(len(corr.columns)))
ax3.set_xticklabels(corr.columns, rotation=90)
ax3.set_yticklabels(corr.columns)

st.pyplot(fig3)

st.divider()

# ==========================================================
# SEGMENTASI SISWA
# ==========================================================
st.header("4️⃣ Segmentasi Kepuasan Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.fillna(df.mean()))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster = kmeans.fit_predict(X_scaled)

df_cluster = df.copy()
df_cluster["Cluster"] = cluster

cluster_mean = df_cluster.groupby("Cluster").mean().mean(axis=1)
cluster_mean = cluster_mean.sort_values(ascending=False)

segment_label = ["Sangat Puas", "Cukup Puas", "Kurang Puas"]

cluster_map = dict(zip(cluster_mean.index, segment_label))
df_cluster["Segment"] = df_cluster["Cluster"].map(cluster_map)

st.subheader("📊 Distribusi Segmentasi Siswa")
st.bar_chart(df_cluster["Segment"].value_counts())

st.success("✅ Dashboard berhasil dibuat dan siap digunakan")