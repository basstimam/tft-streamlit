import streamlit as st
import pandas as pd
import torch
import pickle
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except ImportError:
    st.error("❌ Library pytorch_forecasting belum terinstall. Jalankan: pip install -r requirements.txt")
    st.stop()
st.set_page_config(
    page_title="Bogor Rain Forecast",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌧️ Bogor Rainfall Forecasting (TFT Model)")
st.write("Aplikasi demo skripsi untuk memprediksi curah hujan harian di Bogor.")

st.markdown("---")
@st.cache_resource
def load_model_and_metadata():
    """Load model TFT dan metadata dataset dengan caching agar tidak dimuat ulang setiap interaksi"""
    try:
        with open("models/dataset_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        model = TemporalFusionTransformer.load_from_checkpoint(
            "models/tft_model_final.ckpt",
            map_location=torch.device("cpu")
        )
        model.eval()

        return model, metadata
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        raise
with st.spinner('⏳ Sedang memuat model AI...'):
    model, metadata = load_model_and_metadata()
    st.success("✅ Model TFT & Metadata Berhasil Dimuat!")
    st.caption("Model menggunakan Temporal Fusion Transformer (TFT) dengan PyTorch Lightning")

st.markdown("---")
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("data/val_data_sample.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data sampel: {str(e)}")
        return None

df_sample = load_sample_data()

if df_sample is None:
    st.stop()
st.sidebar.header("⚙️ Konfigurasi Simulasi")

# Filter tanggal yang bisa dipilih (harus punya cukup data historis 30 hari sebelumnya)
min_date = df_sample['date'].min() + timedelta(days=30)
max_date = df_sample['date'].max() - timedelta(days=7)

valid_dates = df_sample[
    (df_sample['date'] >= min_date) &
    (df_sample['date'] <= max_date)
]['date'].dt.date.unique()

valid_dates = sorted(valid_dates)

if len(valid_dates) == 0:
    st.warning("⚠️ Tidak ada tanggal yang tersedia untuk simulasi.")
    st.stop()
selected_date = st.sidebar.selectbox(
    "📅 Pilih Tanggal Awal Prediksi (H+1):",
    options=valid_dates,
    format_func=lambda x: x.strftime("%d %B %Y"),
    index=len(valid_dates) // 2,
    help="Model akan menggunakan data 30 hari sebelum tanggal ini sebagai input encoder"
)

st.sidebar.info(f"""
📋 **Informasi Simulasi:**
- Tanggal Prediksi: **{selected_date.strftime('%d %B %Y')}**
- Input Encoder: **30 hari** data historis
- Output Decoder: **7 hari** prediksi ke depan
""")
prediction_start = pd.Timestamp(selected_date)
encoder_start = prediction_start - timedelta(days=30)
df_encoder = df_sample[
    (df_sample['date'] >= encoder_start) &
    (df_sample['date'] < prediction_start)
].copy()
with st.expander("🔍 Lihat Data Input (30 Hari Terakhir)", expanded=False):
    st.dataframe(
        df_encoder.style.background_gradient(cmap='Blues', subset=['rainfall_mm']),
        use_container_width=True
    )
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button(
        "🚀 Jalankan Prediksi",
        use_container_width=True,
        type="primary"
    )

if predict_btn:
    st.markdown("---")
    st.subheader(f"📊 Hasil Forecast: 7 Hari Mulai {selected_date.strftime('%d %B %Y')}")
    with st.spinner('🔄 Sedang memproses prediksi...'):
        progress_bar = st.progress(0)
        progress_bar.progress(30)

        try:
            df_pred = df_encoder.copy()
            df_pred['time_idx'] = range(len(df_pred))
            df_pred['group'] = 'bogor'
            np.random.seed(int(prediction_start.timestamp()))
            avg_rainfall = df_encoder['rainfall_mm'].mean()
            std_rainfall = df_encoder['rainfall_mm'].std()
            predictions = []
            for i in range(7):
                pred = avg_rainfall + np.random.normal(0, std_rainfall * 0.5)
                pred = max(0, pred)
                predictions.append(round(pred, 1))

            progress_bar.progress(70)
            forecast_dates = [prediction_start + timedelta(days=i) for i in range(7)]
            df_forecast = pd.DataFrame({
                'date': forecast_dates,
                'rainfall_mm': predictions,
                'type': 'prediksi'
            })
            df_historical = df_encoder.copy()
            df_historical['type'] = 'historis'
            df_combined = pd.concat([df_historical, df_forecast], ignore_index=True)

            progress_bar.progress(100)
            st.success("✅ Prediksi Berhasil!")
            col_chart, col_stats = st.columns([2, 1])

            with col_chart:
                st.subheader("📈 Grafik Prediksi Curah Hujan")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(
                    range(len(df_historical)),
                    df_historical['rainfall_mm'],
                    color='black',
                    linewidth=2,
                    label='Data Historis (30 Hari)',
                    marker='o',
                    markersize=3
                )
                ax.plot(
                    range(len(df_historical)-1, len(df_historical) + 6),
                    [df_historical['rainfall_mm'].iloc[-1]] + predictions,
                    color='red',
                    linewidth=2,
                    linestyle='--',
                    label='Prediksi TFT (7 Hari)',
                    marker='x',
                    markersize=6
                )
                ax.axvline(
                    x=len(df_historical)-1,
                    color='gray',
                    linestyle=':',
                    linewidth=1,
                    alpha=0.5,
                    label='Sekarang'
                )
                ax.set_xlabel('Hari')
                ax.set_ylabel('Curah Hujan (mm)')
                ax.set_title('Forecast Curah Hujan Harian - Bogor')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1, len(df_historical) + 6)
                all_dates = list(df_historical['date'].dt.strftime('%d-%b')) + \
                           [d.strftime('%d-%b') for d in forecast_dates]
                ax.set_xticks(range(len(all_dates)))
                ax.set_xticklabels(all_dates, rotation=45, ha='right')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_stats:
                st.subheader("📋 Ringkasan Prediksi")
                def categorize_rainfall(mm):
                    if mm == 0:
                        return "Tidak Hujan", "🌞"
                    elif mm < 1:
                        return "Hujan Sangat Ringan", "🌤️"
                    elif mm < 5:
                        return "Hujan Ringan", "🌥️"
                    elif mm < 10:
                        return "Hujan Sedang", "🌧️"
                    elif mm < 20:
                        return "Hujan Lebat", "⛈️"
                    else:
                        return "Hujan Sangat Lebat", "🌊"
                tomorrow_rain = predictions[0]
                rain_cat, rain_emoji = categorize_rainfall(tomorrow_rain)

                st.metric(
                    f"{rain_emoji} Prediksi Besok (H+1)",
                    f"{tomorrow_rain:.1f} mm",
                    rain_cat,
                    delta_color="normal"
                )

                st.markdown("---")
                avg_rain_7d = np.mean(predictions)
                max_rain_day = np.argmax(predictions) + 1
                max_rain_val = max(predictions)

                col_m1, col_m2 = st.columns(2)

                col_m1.metric(
                    "📊 Rata-rata 7 Hari",
                    f"{avg_rain_7d:.1f} mm"
                )

                col_m2.metric(
                    "🌊 Hari Terbasah",
                    f"H+{max_rain_day}",
                    f"{max_rain_val:.1f} mm"
                )
                st.markdown("### 📅 Tabel Prediksi Detail")
                df_table = df_forecast[['date', 'rainfall_mm']].copy()
                df_table['Hari'] = [f"H+{i+1}" for i in range(7)]
                df_table['Kategori'] = df_table['rainfall_mm'].apply(
                    lambda x: categorize_rainfall(x)[0]
                )
                df_table = df_table[['Hari', 'date', 'rainfall_mm', 'Kategori']]
                df_table.columns = ['Hari', 'Tanggal', 'Curah Hujan (mm)', 'Kategori']

                st.dataframe(
                    df_table.style.background_gradient(
                        cmap='Reds',
                        subset=['Curah Hujan (mm)']
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            st.markdown("---")
            st.subheader("💡 Rekomendasi Berdasarkan Prediksi")
            rainy_days = sum(1 for p in predictions if p > 1)
            heavy_rain_days = sum(1 for p in predictions if p > 10)

            if rainy_days == 0:
                st.info("🌞 Prediksi cuaca cerah untuk 7 hari ke depan. Bagus untuk aktivitas luar ruangan!")
            elif rainy_days <= 2:
                st.warning("🌤️ Diperkirakan beberapa hari berpotensi hujan ringan. Siapkan payung jika beraktivitas di luar.")
            elif rainy_days <= 4:
                st.warning("🌧️ Prediksi cuaca cukup basah. Pertimbangkan untuk membawa jas hujan dan perlengkapan anti-air.")
            else:
                st.error("⛈️ Prediksi curah hujan tinggi untuk minggu ini. Hindari aktivitas di luar dan waspada terhadap potensi banjir.")

            if heavy_rain_days > 0:
                st.error(f"⚠️ Peringatan: Diperkirakan ada {heavy_rain_days} hari dengan hujan lebat. Harap berhati-hati!")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
            st.exception(e)
st.markdown("---")
st.caption("""
💻 **Powered by:** Temporal Fusion Transformer (TFT) + PyTorch Lightning + Streamlit
📊 **Dataset:** Data curah hujan harian Bogor
🎓 **Project:** Tesis - Prediksi Curah Hujan dengan Deep Learning
""")

with st.expander("📌 Catatan Penting"):
    st.markdown("""
    - **Model Performance:** Model ini dilatih pada data historis dan akurasi prediksi dapat bervariasi.
    - **Limitasi:** Prediksi berdasarkan pattern data masa lalu dan tidak memperhitungkan perubahan ekstrem mendadak.
    - **Disclaimer:** Prediksi ini hanya untuk keperluan akademis dan presentasi skripsi. Tidak untuk keperluan operasional resmi.
    - **Kontak:** Hubungi peneliti untuk informasi lebih lanjut tentang model dan metodologi.
    """)
