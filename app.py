import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Prediksi Harga Pangan 2025-2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: #ffffff;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        background-color: #ffffff;
        border-bottom: 3px solid #2c3e50;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
        padding: 0.5rem;
        background-color: #ffffff;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-card h3 {
        color: #2c3e50;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        color: #34495e;
        font-size: 2rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .metric-card p {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stButton>button:hover {
        background-color: #34495e;
        border: none;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #3498db;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .excellent-box {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #ffffff;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    
    .prediction-result {
        background-color: #e8f5e9;
        padding: 2rem;
        border-radius: 8px;
        border: 2px solid #4caf50;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .prediction-result h2 {
        color: #2e7d32;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .prediction-result p {
        color: #558b2f;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# FUNGSI EVALUASI METRIK
# =====================================================================
def get_metric_interpretation(mape_value):
    """Memberikan interpretasi kualitas prediksi berdasarkan MAPE"""
    if mape_value < 5:
        return "Sangat Baik", "success-box", "Model memiliki akurasi prediksi yang sangat tinggi"
    elif mape_value < 10:
        return "Baik", "excellent-box", "Model memiliki akurasi prediksi yang baik"
    elif mape_value < 20:
        return "Cukup Baik", "info-box", "Model memiliki akurasi prediksi yang cukup memadai"
    else:
        return "Perlu Perbaikan", "warning-box", "Model perlu ditingkatkan untuk akurasi yang lebih baik"

# =====================================================================
# FUNGSI LOAD MODEL
# =====================================================================
@st.cache_resource
def load_trained_model(model_path='best_lstm_model.h5'):
    """Load model yang sudah di-train dari file lokal"""
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# =====================================================================
# FUNGSI PREPROCESSING
# =====================================================================
def preprocess_data(df_raw):
    """Preprocessing data dari Excel yang diupload"""
    try:
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        df_data = df_raw.iloc[:, 2:]
        df_transposed = df_data.T
        df_transposed.columns = komoditas_list
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
        
        df_transposed['Tanggal'] = pd.to_datetime(
            df_transposed['Tanggal'], 
            format='%d/ %m/ %Y', 
            errors='coerce'
        )
        df_transposed = df_transposed.dropna(subset=['Tanggal'])
        df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
        
        for kolom in komoditas_list:
            if df_transposed[kolom].dtype == 'object':
                df_transposed[kolom] = df_transposed[kolom].str.replace(',', '').str.replace('"', '')
            df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
        
        df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(
            method='linear', 
            limit_direction='both'
        )
        df_transposed[komoditas_list] = df_transposed[komoditas_list].bfill().ffill()
        
        return df_transposed, komoditas_list
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam preprocessing: {str(e)}")
        return None, None

def create_scalers(df_processed, komoditas_list):
    """Membuat scaler untuk setiap komoditas"""
    scalers = {}
    for kolom in komoditas_list:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_processed[[kolom]].values)
        scalers[kolom] = scaler
    return scalers

# =====================================================================
# FUNGSI PREDIKSI
# =====================================================================
def predict_single_commodity(model, last_sequence, scaler, commodity_index, bulan_target, time_steps=20):
    """Prediksi harga untuk satu komoditas tertentu"""
    tanggal_sekarang = datetime.now()
    tanggal_target = datetime(bulan_target.year, bulan_target.month, 15)
    
    minggu_prediksi = int((tanggal_target - tanggal_sekarang).days / 7)
    
    if minggu_prediksi <= 0:
        minggu_prediksi = 1
    
    current_sequence = last_sequence.copy()
    predictions = []
    
    for _ in range(minggu_prediksi):
        pred_norm = model.predict(
            current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), 
            verbose=0
        )
        predictions.append(pred_norm[0])
        current_sequence = np.vstack([current_sequence[1:], pred_norm[0]])
    
    # Denormalisasi prediksi untuk komoditas yang dipilih
    pred_value = scaler.inverse_transform([[predictions[-1][commodity_index]]])[0][0]
    
    return pred_value, minggu_prediksi

def prepare_last_sequence(df_processed, komoditas_list, scalers, time_steps=20):
    """Menyiapkan sequence terakhir untuk prediksi"""
    data_normalized = np.zeros((len(df_processed), len(komoditas_list)))
    
    for i, kolom in enumerate(komoditas_list):
        data_normalized[:, i] = scalers[kolom].transform(
            df_processed[[kolom]].values
        ).flatten()
    
    return data_normalized[-time_steps:]

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("### Model Prediksi Harga Pangan")
st.sidebar.markdown("---")

# Upload dataset
uploaded_dataset = st.sidebar.file_uploader(
    "Upload Dataset Excel",
    type=['xlsx', 'xls'],
    help="Upload file Excel dengan format yang sama dengan dataset training"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Format Dataset:**
- Kolom 1: Nomor urut
- Kolom 2: Nama Komoditas
- Kolom 3 dst: Tanggal dengan harga
- Format tanggal: DD/ MM/ YYYY
""")

st.sidebar.markdown("---")

# Cek model
model_exists = os.path.exists('best_lstm_model.h5')
st.sidebar.markdown("**Status Model:**")
if model_exists:
    st.sidebar.success("Model tersedia")
else:
    st.sidebar.error("Model tidak ditemukan")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Penjelasan Metrik Evaluasi:**

**MAPE (Mean Absolute Percentage Error)**
- < 5%: Sangat Baik
- 5-10%: Baik
- 10-20%: Cukup Baik
- > 20%: Perlu Perbaikan

**MAE (Mean Absolute Error)**
Rata-rata selisih antara nilai prediksi dan aktual dalam satuan Rupiah

**RMSE (Root Mean Squared Error)**
Akar dari rata-rata kuadrat selisih prediksi dan aktual
""")

# =====================================================================
# MAIN CONTENT
# =====================================================================
st.markdown('<h1 class="main-header">Model Prediksi Harga Pangan Indonesia 2025-2026</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Prediksi Harga Multi-Komoditas Menggunakan Long Short-Term Memory (LSTM)</p>', unsafe_allow_html=True)

# Cek model dan dataset
if model_exists and uploaded_dataset is not None:
    try:
        # Load model
        with st.spinner('Memuat model...'):
            model = load_trained_model()
        
        if model is not None:
            st.markdown('<div class="success-box">Model berhasil dimuat dari file best_lstm_model.h5</div>', unsafe_allow_html=True)
            
            # Load dataset yang diupload
            with st.spinner('Memuat dataset...'):
                df_raw = pd.read_excel(uploaded_dataset)
            
            st.markdown('<div class="success-box">Dataset berhasil diupload</div>', unsafe_allow_html=True)
            
            # Preprocessing
            with st.spinner('Memproses dataset...'):
                df_processed, komoditas_list = preprocess_data(df_raw)
            
            if df_processed is not None and komoditas_list is not None:
                # Buat scalers
                scalers = create_scalers(df_processed, komoditas_list)
                
                # Tampilkan info dataset
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Data</h3>
                        <h2>{len(df_processed)}</h2>
                        <p>Baris Data</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Jumlah Komoditas</h3>
                        <h2>{len(komoditas_list)}</h2>
                        <p>Jenis Komoditas</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Periode Data</h3>
                        <h2>{df_processed['Tanggal'].min().strftime('%Y')}-{df_processed['Tanggal'].max().strftime('%Y')}</h2>
                        <p>Rentang Waktu</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Tab untuk evaluasi dan prediksi
                tab1, tab2 = st.tabs(["Evaluasi Model", "Prediksi Harga Masa Depan"])
                
                # =====================================================================
                # TAB 1: EVALUASI - TAMPILKAN SEMUA KOMODITAS
                # =====================================================================
                with tab1:
                    st.markdown("### Hasil Evaluasi Model")
                    st.markdown('<div class="info-box">Berikut adalah hasil evaluasi performa model LSTM untuk <strong>semua komoditas</strong> yang telah dilatih menggunakan data historis harga pangan</div>', unsafe_allow_html=True)
                    
                    # Load hasil evaluasi jika ada
                    if os.path.exists('hasil_evaluasi_lstm_100epochs.csv'):
                        df_eval = pd.read_csv('hasil_evaluasi_lstm_100epochs.csv')
                        
                        # Statistik keseluruhan
                        st.markdown("#### Ringkasan Statistik Evaluasi")
                        col1, col2, col3 = st.columns(3)
                        
                        avg_mape = df_eval['MAPE (%)'].mean()
                        interpretation, box_class, description = get_metric_interpretation(avg_mape)
                        
                        with col1:
                            st.metric("Rata-rata MAPE", f"{avg_mape:.2f}%", delta=interpretation)
                        with col2:
                            mae_val = df_eval['MAE'].mean() if 'MAE' in df_eval.columns else 0
                            st.metric("Rata-rata MAE", f"Rp {mae_val:,.0f}")
                        with col3:
                            rmse_val = df_eval['RMSE'].mean() if 'RMSE' in df_eval.columns else 0
                            st.metric("Rata-rata RMSE", f"Rp {rmse_val:,.0f}")
                        
                        # Interpretasi keseluruhan
                        st.markdown(f'<div class="{box_class}"><strong>Interpretasi:</strong> {description}. Rata-rata MAPE sebesar {avg_mape:.2f}% menunjukkan performa model secara keseluruhan.</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("#### Tabel Evaluasi Metrik - Semua Komoditas")
                        
                        # Format tabel dengan interpretasi
                        df_eval_display = df_eval.copy()
                        df_eval_display['Interpretasi'] = df_eval_display['MAPE (%)'].apply(
                            lambda x: get_metric_interpretation(x)[0]
                        )
                        
                        # Urutkan berdasarkan MAPE
                        df_eval_display = df_eval_display.sort_values('MAPE (%)')
                        
                        # Format kolom
                        if 'MAE' in df_eval_display.columns:
                            df_eval_display['MAE (Rp)'] = df_eval_display['MAE'].apply(lambda x: f"Rp {x:,.0f}")
                        if 'RMSE' in df_eval_display.columns:
                            df_eval_display['RMSE (Rp)'] = df_eval_display['RMSE'].apply(lambda x: f"Rp {x:,.0f}")
                        
                        # Pilih kolom untuk ditampilkan
                        display_cols = ['Komoditas', 'MAPE (%)', 'MAE (Rp)', 'RMSE (Rp)', 'Interpretasi']
                        st.dataframe(
                            df_eval_display[display_cols], 
                            use_container_width=True, 
                            height=600
                        )
                        
                        # Download button
                        csv = df_eval_display[display_cols].to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="Unduh Tabel Evaluasi Lengkap (CSV)",
                            data=csv,
                            file_name="evaluasi_model_lengkap.csv",
                            mime="text/csv"
                        )
                        
                        # Visualisasi
                        st.markdown("---")
                        st.markdown("#### Visualisasi Distribusi Metrik Evaluasi")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Grafik MAPE untuk 15 komoditas terbaik
                            df_sorted = df_eval_display.sort_values('MAPE (%)').head(15)
                            fig_mape = go.Figure()
                            
                            colors = ['#28a745' if x < 5 else '#17a2b8' if x < 10 else '#ffc107' if x < 20 else '#dc3545' 
                                     for x in df_sorted['MAPE (%)']]
                            
                            fig_mape.add_trace(go.Bar(
                                x=df_sorted['MAPE (%)'],
                                y=df_sorted['Komoditas'],
                                orientation='h',
                                marker=dict(color=colors),
                                text=df_sorted['MAPE (%)'].apply(lambda x: f'{x:.2f}%'),
                                textposition='auto'
                            ))
                            fig_mape.update_layout(
                                title='Top 15 Komoditas - MAPE Terbaik',
                                xaxis_title='MAPE (%)',
                                yaxis_title='Komoditas',
                                height=500,
                                template='plotly_white',
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                        
                        with col2:
                            # Grafik RMSE untuk 15 komoditas terbaik
                            rmse_col = 'RMSE' if 'RMSE' in df_eval.columns else 'RMSE (Rp)'
                            df_sorted_rmse = df_eval.sort_values(rmse_col).head(15)
                            fig_rmse = go.Figure()
                            fig_rmse.add_trace(go.Bar(
                                x=df_sorted_rmse[rmse_col],
                                y=df_sorted_rmse['Komoditas'],
                                orientation='h',
                                marker=dict(color='#34495e'),
                                text=df_sorted_rmse[rmse_col].apply(lambda x: f'Rp {x:,.0f}'),
                                textposition='auto'
                            ))
                            fig_rmse.update_layout(
                                title='Top 15 Komoditas - RMSE Terbaik',
                                xaxis_title='RMSE (Rp)',
                                yaxis_title='Komoditas',
                                height=500,
                                template='plotly_white',
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            st.plotly_chart(fig_rmse, use_container_width=True)
                        
                        # Distribusi kategori performa
                        st.markdown("---")
                        st.markdown("#### Distribusi Kategori Performa Model")
                        
                        category_counts = df_eval_display['Interpretasi'].value_counts()
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=category_counts.index,
                            values=category_counts.values,
                            marker=dict(colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545']),
                            hole=0.4
                        )])
                        fig_pie.update_layout(
                            title='Distribusi Kategori Performa',
                            height=400,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    else:
                        st.markdown('<div class="info-box">File hasil evaluasi tidak ditemukan. Model tetap dapat digunakan untuk prediksi.</div>', unsafe_allow_html=True)
                
                # =====================================================================
                # TAB 2: PREDIKSI - PILIH TAHUN, BULAN, KOMODITAS
                # =====================================================================
                with tab2:
                    st.markdown("### Prediksi Harga Komoditas Masa Depan")
                    st.markdown('<div class="info-box">Pilih tahun, bulan, dan komoditas yang ingin diprediksi, kemudian klik tombol "Mulai Prediksi"</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        tahun_selected = st.selectbox("Pilih Tahun", [2025, 2026, 2027, 2028])
                    
                    with col2:
                        bulan_options = {
                            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
                            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
                            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
                        }
                        bulan_selected = st.selectbox("Pilih Bulan", list(bulan_options.keys()))
                    
                    with col3:
                        komoditas_selected = st.selectbox(
                            "Pilih Komoditas",
                            komoditas_list,
                            help="Pilih komoditas yang ingin diprediksi harganya"
                        )
                    
                    if st.button("Mulai Prediksi", key="predict_btn"):
                        bulan_target = datetime(tahun_selected, bulan_options[bulan_selected], 15)
                        
                        # Siapkan sequence terakhir
                        last_sequence = prepare_last_sequence(df_processed, komoditas_list, scalers)
                        
                        # Cari index komoditas yang dipilih
                        commodity_index = komoditas_list.index(komoditas_selected)
                        
                        with st.spinner(f'Memprediksi harga {komoditas_selected} untuk {bulan_selected} {tahun_selected}...'):
                            predicted_price, weeks_ahead = predict_single_commodity(
                                model,
                                last_sequence,
                                scalers[komoditas_selected],
                                commodity_index,
                                bulan_target
                            )
                        
                        st.markdown(f'<div class="success-box">Prediksi harga untuk <strong>{komoditas_selected}</strong> pada <strong>{bulan_selected} {tahun_selected}</strong> berhasil dibuat</div>', unsafe_allow_html=True)
                        
                        # Tampilkan hasil prediksi dengan styling khusus
                        st.markdown("---")
                        st.markdown("### Hasil Prediksi Harga")
                        
                        st.markdown(f"""
                        <div class="prediction-result">
                            <p style="margin: 0; font-size: 1.2rem; color: #555;">Prediksi Harga</p>
                            <h2 style="margin: 0.5rem 0;">Rp {predicted_price:,.0f}</h2>
                            <p style="margin: 0;"><strong>{komoditas_selected}</strong></p>
                            <p style="margin: 0; font-size: 0.9rem;">{bulan_selected} {tahun_selected}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Informasi tambahan
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Komoditas", komoditas_selected)
                        with col2:
                            st.metric("Periode Prediksi", f"{bulan_selected} {tahun_selected}")
                        with col3:
                            st.metric("Prediksi Minggu ke Depan", f"{weeks_ahead} minggu")
                        
                        # Bandingkan dengan harga terakhir
                        st.markdown("---")
                        st.markdown("### Analisis Perbandingan Harga")
                        
                        last_actual_price = df_processed[komoditas_selected].iloc[-1]
                        price_change = predicted_price - last_actual_price
                        price_change_pct = (price_change / last_actual_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Harga Terakhir (Data Aktual)", 
                                f"Rp {last_actual_price:,.0f}",
                                help="Harga terakhir dari dataset yang diupload"
                            )
                        
                        with col2:
                            st.metric(
                                "Harga Prediksi", 
                                f"Rp {predicted_price:,.0f}",
                                delta=f"Rp {price_change:,.0f}",
                                help="Prediksi harga untuk periode yang dipilih"
                            )
                        
                        with col3:
                            st.metric(
                                "Perubahan Harga", 
                                f"{price_change_pct:+.2f}%",
                                delta=f"{'Naik' if price_change > 0 else 'Turun'}",
                                help="Persentase perubahan harga dibanding harga terakhir"
                            )
                        
                        # Interpretasi perubahan harga
                        if price_change_pct > 10:
                            st.markdown('<div class="warning-box"><strong>Peringatan:</strong> Prediksi menunjukkan kenaikan harga yang signifikan (>10%). Perlu antisipasi dalam pengelolaan stok dan distribusi.</div>', unsafe_allow_html=True)
                        elif price_change_pct > 5:
                            st.markdown('<div class="info-box"><strong>Informasi:</strong> Prediksi menunjukkan kenaikan harga moderat (5-10%). Pantau terus perkembangan harga.</div>', unsafe_allow_html=True)
                        elif price_change_pct < -10:
                            st.markdown('<div class="excellent-box"><strong>Informasi:</strong> Prediksi menunjukkan penurunan harga yang signifikan. Kondisi menguntungkan bagi konsumen.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box"><strong>Informasi:</strong> Prediksi menunjukkan harga relatif stabil dengan perubahan minimal.</div>', unsafe_allow_html=True)
                        
                        # Visualisasi trend harga
                        st.markdown("---")
                        st.markdown("### Visualisasi Trend Harga")
                        
                        # Ambil data historis komoditas
                        historical_data = df_processed[['Tanggal', komoditas_selected]].tail(52)  # 1 tahun terakhir
                        
                        fig_trend = go.Figure()
                        
                        # Data historis
                        fig_trend.add_trace(go.Scatter(
                            x=historical_data['Tanggal'],
                            y=historical_data[komoditas_selected],
                            mode='lines+markers',
                            name='Data Historis',
                            line=dict(color='#2c3e50', width=2),
                            marker=dict(size=5)
                        ))
                        
                        # Titik prediksi
                        fig_trend.add_trace(go.Scatter(
                            x=[bulan_target],
                            y=[predicted_price],
                            mode='markers',
                            name='Prediksi',
                            marker=dict(
                                size=15,
                                color='#e74c3c',
                                symbol='star',
                                line=dict(color='#c0392b', width=2)
                            )
                        ))
                        
                        fig_trend.update_layout(
                            title=f'Trend Harga {komoditas_selected}',
                            xaxis_title='Tanggal',
                            yaxis_title='Harga (Rp)',
                            height=500,
                            template='plotly_white',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True)
                        
                        # Download hasil prediksi
                        st.markdown("---")
                        st.markdown("### Unduh Hasil Prediksi")
                        
                        # Buat dataframe hasil
                        result_df = pd.DataFrame({
                            'Komoditas': [komoditas_selected],
                            'Tahun': [tahun_selected],
                            'Bulan': [bulan_selected],
                            'Harga Terakhir (Rp)': [f"Rp {last_actual_price:,.0f}"],
                            'Prediksi Harga (Rp)': [f"Rp {predicted_price:,.0f}"],
                            'Perubahan (Rp)': [f"Rp {price_change:,.0f}"],
                            'Perubahan (%)': [f"{price_change_pct:+.2f}%"]
                        })
                        
                        csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="Unduh Hasil Prediksi (CSV)",
                            data=csv,
                            file_name=f"prediksi_{komoditas_selected.replace(' ', '_')}_{bulan_selected}_{tahun_selected}.csv",
                            mime="text/csv"
                        )
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

elif not model_exists:
    st.markdown("""
    <div class="warning-box">
        <h3>Model Tidak Ditemukan</h3>
        <p>File <strong>best_lstm_model.h5</strong> tidak ditemukan di direktori. Pastikan file model ada di direktori yang sama dengan app.py</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
        <h3>Panduan Penggunaan Aplikasi</h3>
        <ol style="padding-left: 1.5rem;">
            <li style="margin-bottom: 0.5rem;">Upload dataset Excel di sidebar</li>
            <li style="margin-bottom: 0.5rem;">Pilih tab <strong>Evaluasi Model</strong> untuk melihat performa model semua komoditas</li>
            <li style="margin-bottom: 0.5rem;">Pilih tab <strong>Prediksi Harga Masa Depan</strong> untuk melakukan prediksi</li>
            <li style="margin-bottom: 0.5rem;">Pilih <strong>tahun, bulan, dan komoditas</strong> yang ingin diprediksi</li>
            <li style="margin-bottom: 0.5rem;">Klik tombol <strong>Mulai Prediksi</strong></li>
            <li style="margin-bottom: 0.5rem;">Lihat hasil prediksi, analisis perbandingan, dan visualisasi trend</li>
            <li style="margin-bottom: 0.5rem;">Unduh hasil prediksi dalam format CSV</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1.5rem;'>
    <p style="margin: 0;">Model Prediksi Harga Pangan Indonesia</p>
    <p style="margin: 0.5rem 0 0 0;">Dibuat menggunakan Streamlit dan TensorFlow</p>
</div>
""", unsafe_allow_html=True)
