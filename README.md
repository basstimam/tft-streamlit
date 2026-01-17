# 🌧️ Bogor Rainfall Forecasting Dashboard

Dashboard interaktif untuk prediksi curah hujan harian di Bogor menggunakan model Temporal Fusion Transformer (TFT). Dibuat untuk keperluan presentasi skripsi dan demo kemampuan model.

## 📋 Fitur Utama

- ✅ **Model Auto-Load**: Model TFT dimuat otomatis dengan caching untuk performa cepat
- 📅 **Date Selection**: Pilih tanggal simulasi dengan mudah melalui dropdown
- 📊 **Visualization Dashboard**: Grafik prediksi dengan visualisasi data historis dan forecast
- 📈 **Weather Metrics**: Ringkasan statistik prediksi untuk 7 hari ke depan
- 💡 **Smart Recommendations**: Rekomendasi aktivitas berdasarkan prediksi cuaca
- 🎯 **User-Friendly Interface**: Antarmuka yang mudah digunakan untuk presentasi

## 🏗️ Struktur Project

```
my_skripsi_app/
├── app.py                       # File utama aplikasi Streamlit
├── requirements.txt             # Daftar library Python yang dibutuhkan
├── README.md                   # Dokumentasi project ini
├── prd.md                      # Product Requirements Document
├── models/
│   ├── tft_model_final.ckpt    # Model TFT yang sudah dilatih
│   └── dataset_metadata.pkl    # Metadata dataset untuk model
└── data/
    └── val_data_sample.csv     # Data sampel untuk simulasi (Juni-Juli 2025)
```

## 🚀 Cara Install dan Menjalankan

### 1. Install Dependencies

Buka terminal di folder project dan jalankan:

```bash
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser dengan alamat: `http://localhost:8501`

## 📖 Cara Penggunaan

### Langkah 1: Buka Aplikasi

Setelah menjalankan perintah di atas, aplikasi akan terbuka di browser.

### Langkah 2: Pilih Tanggal Simulasi

- Lihat sidebar di sebelah kiri
- Pilih tanggal awal prediksi dari dropdown
- Model akan menggunakan 30 hari data sebelum tanggal yang dipilih sebagai input

### Langkah 3: Lihat Data Input

- Klik "Lihat Data Input (30 Hari Terakhir)" untuk melihat data yang akan digunakan
- Data akan ditampilkan dalam tabel dengan highlight gradient

### Langkah 4: Jalankan Prediksi

- Klik tombol "🚀 Jalankan Prediksi"
- Tunggu proses prediksi selesai (biasanya < 2 detik)

### Langkah 5: Analisis Hasil

Dashboard akan menampilkan:

1. **Grafik Prediksi**: Line chart dengan garis hitam (historis) dan merah putus-putus (prediksi)
2. **Ringkasan Prediksi**: Statistik besok dan rata-rata 7 hari
3. **Tabel Detail**: Prediksi harian lengkap dengan kategori hujan
4. **Rekomendasi**: Saran aktivitas berdasarkan prediksi cuaca

## 🎨 Komponen Dashboard

### Visualisasi Grafik

- **Garis Hitam Solid**: Data historis 30 hari terakhir
- **Garis Merah Putus-putus**: Prediksi 7 hari ke depan
- **Garis Titik-titik**: Batas antara data aktual dan prediksi

### Kategori Hujan

| Curah Hujan | Kategori | Emoji |
|-------------|----------|-------|
| 0 mm | Tidak Hujan | 🌞 |
| < 1 mm | Sangat Ringan | 🌤️ |
| 1-5 mm | Ringan | 🌥️ |
| 5-10 mm | Sedang | 🌧️ |
| 10-20 mm | Lebat | ⛈️ |
| > 20 mm | Sangat Lebat | 🌊 |

## 🔧 Teknologi

- **Framework**: Streamlit
- **Machine Learning**: PyTorch Forecasting & PyTorch Lightning
- **Data Processing**: Pandas & NumPy
- **Visualization**: Matplotlib & Seaborn
- **Model**: Temporal Fusion Transformer (TFT)

## 📊 Model Performance

Model TFT dilatih menggunakan:
- **Encoder Length**: 30 hari
- **Decoder Length**: 7 hari
- **Dataset**: Data curah hujan harian Bogor
- **Optimization**: Adam optimizer dengan learning rate scheduling

### Catatan Penting

- Model ini dilatih pada data historis dan akurasi prediksi dapat bervariasi
- Prediksi berdasarkan pattern data masa lalu dan tidak memperhitungkan perubahan ekstrem mendadak
- Aplikasi ini untuk keperluan akademis dan presentasi skripsi
- Tidak untuk keperluan operasional resmi

## 🐛 Troubleshooting

### Error: "No module named 'pytorch_forecasting'"

**Solusi**: Install ulang dependencies:
```bash
pip install --upgrade pytorch-forecasting
```

### Error: "Gagal memuat model"

**Solusi**: Pastikan file model ada di folder `models/`:
- `tft_model_final.ckpt`
- `dataset_metadata.pkl`

### Aplikasi berjalan lambat

**Solusi**: Pastikan menjalankan dengan CPU (default). Caching sudah diaktifkan untuk percepatan.

### Tidak ada tanggal yang tersedia

**Solusi**: Pastikan file `data/val_data_sample.csv` ada dan memiliki minimal 37 baris data (30 hari encoder + 7 hari prediksi).

## 📝 Catatan untuk Presentasi Skripsi

### Tips Presentasi:

1. **Siapkan Data**: Pastikan data sampel mencakup periode yang relevan dengan skripsi
2. **Highlight Fitur**: Tunjukkan auto-load model, caching, dan visualisasi
3. **Jelaskan Alur**: Dari input tanggal → prediksi → hasil → rekomendasi
4. **Jawab Pertanyaan**: Siapkan jawaban untuk pertanyaan teknis tentang model

### Demonya:

```
"Selamat pagi, saya akan mendemokan aplikasi prediksi curah hujan
menggunakan model Temporal Fusion Transformer. Aplikasi ini
memudahkan presentasi hasil penelitian tanpa perlu live coding..."

(Lanjutkan dengan demo interaktif)
```

## 📞 Kontak & Dukungan

Untuk informasi lebih lanjut atau dukungan teknis:
- **Project**: Tesis - Prediksi Curah Hujan dengan Deep Learning
- **Author**: [Nama Mahasiswa]
- **Institution**: [Nama Universitas]

## 📄 Lisensi

Project ini dibuat untuk keperluan akademis (Tesis S1).

---

**Happy Coding & Good Luck for Your Thesis Defense! 🎓**
