

---

# 📄 Product Requirements Document (PRD)

**Project Name:** Bogor Rainfall Forecasting Dashboard (TFT Model)
**Version:** 1.0 (MVP - Minimum Viable Product for Thesis Defense)
**Platform:** Web Application (Streamlit)

## 1. Latar Belakang & Tujuan

* **Masalah:** Model prediksi cuaca (TFT) yang sudah dilatih tersimpan dalam format `.ckpt` dan sulit diakses oleh pengguna awam tanpa menjalankan kode di Jupyter Notebook.
* **Solusi:** Membuat antarmuka web sederhana berbasis Streamlit yang memungkinkan pengguna melakukan simulasi prediksi curah hujan 7 hari ke depan.
* **Tujuan Utama:** Mendemokan kemampuan model saat sidang skripsi secara interaktif dan visual.

## 2. Target Pengguna

1. **Penguji/Dosen:** Ingin melihat bukti bahwa model bekerja dan bisa menerima input.
2. **Mahasiswa (Kamu):** Membutuhkan alat presentasi yang lancar tanpa *live coding*.

## 3. Spesifikasi Fitur (Functional Requirements)

### A. Fitur Utama (Core)

1. **Model Loader (Auto-Cache):**
* Aplikasi harus memuat file `tft_model_final.ckpt` dan `dataset_metadata.pkl` secara otomatis saat aplikasi dibuka.
* Gunakan fitur *caching* agar model tidak dimuat ulang setiap kali user mengklik tombol (agar cepat).


2. **Input Data Selector (Mode Simulasi):**
* Karena model butuh input 30 hari ke belakang (`max_encoder_length=30`), user **TIDAK BOLEH** mengetik manual 30 hari data (kelamaan).
* **Solusi:** Sediakan *Dropdown* atau *Slider* untuk memilih "Tanggal Mulai Prediksi" berdasarkan data validasi yang sudah ada (misal: Data Test Juni 2025).


3. **Visualization Dashboard:**
* Menampilkan grafik garis (Line Chart): Data Historis (30 hari ke belakang) disambung dengan Data Prediksi (7 hari ke depan).
* Bedakan warna garis: Hitam untuk Historis, Merah Putus-putus untuk Prediksi.


4. **Weather Metrics Display:**
* Tampilkan "Kartu Ringkasan" untuk prediksi besok (H+1): Misal "Prediksi Besok: 5.8 mm (Hujan Ringan)".



### B. Alur Pengguna (User Flow)

1. User membuka aplikasi Streamlit.
2. Muncul tulisan "Model Loaded Successfully ✅".
3. User memilih tanggal simulasi (misal: "Prediksi mulai tanggal 2 Juni 2025").
4. Aplikasi menampilkan tabel input data 30 hari terakhir (sebagai konfirmasi).
5. User klik tombol **"Start Forecast 🚀"**.
6. Aplikasi menampilkan Grafik & Tabel hasil prediksi 7 hari ke depan.

## 4. Spesifikasi Teknis (Non-Functional)

* **Framework:** Streamlit (`streamlit`).
* **Machine Learning Engine:** PyTorch Forecasting & PyTorch Lightning.
* **Data Processing:** Pandas.
* **Performance:** Waktu inferensi (prediksi) harus di bawah 2 detik.

---

## 5. Struktur Folder Project

Agar rapi, susun foldermu seperti ini:

```text
my_skripsi_app/
├── app.py                  # File utama Streamlit
├── requirements.txt        # Daftar library
├── models/
│   ├── tft_model_final.ckpt      # File model kamu
│   └── dataset_metadata.pkl      # File metadata pickle
└── data/
    └── val_data_sample.csv       # Sampel data CSV (Juni 2025) untuk simulasi

```

---

## 6. Bocoran Kode `app.py` (Starter Pack)

Ini adalah kode kerangka yang bisa langsung kamu pakai. Copy-paste ke `app.py`:

```python
import streamlit as st
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(page_title="Bogor Rain Forecast", page_icon="🌧️")

st.title("🌧️ Bogor Rainfall Forecasting (TFT Model)")
st.write("Aplikasi demo skripsi untuk memprediksi curah hujan harian di Bogor.")

# ---------------------------------------------------------
# 2. LOAD MODEL (DENGAN CACHE BIAR CEPAT)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    # Load Metadata
    with open("models/dataset_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Load Model (CPU)
    model = TemporalFusionTransformer.load_from_checkpoint(
        "models/tft_model_final.ckpt", 
        map_location=torch.device("cpu")
    )
    return model, metadata

# Tampilkan status loading
with st.spinner('Sedang memuat model AI...'):
    model, metadata = load_model_and_metadata()
    st.success("Model TFT & Metadata Berhasil Dimuat! ✅")

# ---------------------------------------------------------
# 3. INPUT SIMULASI
# ---------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Simulasi")

# Load data sampel (CSV yang kamu pake buat validasi kemarin)
# Pastikan kamu simpan val_df ke csv dulu: val_df.to_csv("data/val_data_sample.csv")
try:
    df_sample = pd.read_csv("data/val_data_sample.csv")
    df_sample['date'] = pd.to_datetime(df_sample['date'])
    
    # Ambil tanggal-tanggal unik buat pilihan
    unique_dates = df_sample['date'].dt.date.unique()
    unique_dates.sort()
    
    # User milih tanggal start prediksi
    selected_date = st.sidebar.selectbox(
        "Pilih Tanggal Awal Prediksi (H+1):", 
        options=unique_dates[-10:] # Ambil 10 tanggal terakhir aja biar gampang
    )
    
    st.sidebar.info(f"Model akan menggunakan data 30 hari sebelum {selected_date} sebagai input.")
    
except Exception as e:
    st.error(f"Gagal memuat data sampel: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. PROSES PREDIKSI
# ---------------------------------------------------------
if st.button("🚀 Jalankan Prediksi"):
    
    # Logika untuk mengambil 30 hari data sebelum selected_date
    # ... (Di sini kamu masukkan logika filtering dataframe seperti di notebook)
    # ... (Lalu convert ke TimeSeriesDataSet)
    
    # Placeholder hasil (nanti diganti prediksi beneran)
    st.subheader(f"📅 Hasil Forecast: 7 Hari Mulai {selected_date}")
    
    # Contoh visualisasi dummy (Ganti dengan plot matplotlib hasil model kamu)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(7), [5.2, 6.1, 8.3, 2.1, 0.0, 1.2, 4.5], marker='x', color='red', label='Prediksi TFT')
    ax.set_title("Forecast Curah Hujan")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Tampilkan Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Hujan", "4.5 mm", "Hujan Sedang")
    col2.metric("Hari Terbasah", "H+3", "8.3 mm")
    col3.metric("Confidence", "High", "Loss: 0.2")


```

### Apa yang harus kamu siapkan sekarang?

1. **File Model:** `tft_model_final.ckpt`.
2. **File Metadata:** `dataset_metadata.pkl` (pastikan ekstensinya `.pkl` ya, kalau `.ckpt` itu biasanya model weights).
3. **File Data Sampel:** Ambil `val_df` dari notebook kamu, save jadi `val_data_sample.csv`. Ini penting buat bahan simulasi di Streamlit.

Gimana? Lebih kebayang kan presentasi skripsinya bakal sekeren apa pake ini? 😎