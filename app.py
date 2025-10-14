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
    /* Background dan padding utama */
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
    
    /* Header styling */
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
    
    /* Card untuk metrik */
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
    
    /* Button styling */
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
    
    /* Info box */
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
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    
    /* Tab styling */
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
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Metric container */
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
</style>
""", unsafe_allow_html=True)

# =====================================================================
# FUNGSI LOAD FILES LOKAL
# =====================================================================
@st.cache_resource
def load_trained_model(model_path='best_lstm_model.h5'):
    """Load model yang sudah di-train dari file lokal"""
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            st.error(f"File model tidak ditemukan: {model_path}")
            return None
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

@st.cache_data
def load_dataset(dataset_path='dataset.xlsx'):
    """Load dataset dari file lokal"""
    try:
        if os.path.exists(dataset_path):
            df = pd.read_excel(dataset_path)
            return df
        else:
            st.error(f"File dataset tidak ditemukan: {dataset_path}")
            return None
    except Exception as e:
        st.error(f"Gagal memuat dataset: {str(e)}")
        return None

@st.cache_data
def load_evaluation_results(eval_path='hasil_evaluasi_lstm_100epochs.csv'):
    """Load hasil evaluasi dari file lokal"""
    try:
        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path)
            return df
        else:
            return None
    except Exception as e:
        return None

# =====================================================================
# FUNGSI PREPROCESSING
# =====================================================================
def preprocess_data(df_raw):
    """Preprocessing data dari Excel yang diupload"""
    try:
        # Ambil nama komoditas dari kolom kedua
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        # Transpose data (mulai dari kolom ke-3)
        df_data = df_raw.iloc[:, 2:]
        df_transposed = df_data.T
        df_transposed.columns = komoditas_list
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
        
        # Konversi tanggal
        df_transposed['Tanggal'] = pd.to_datetime(
            df_transposed['Tanggal'], 
            format='%d/ %m/ %Y', 
            errors='coerce'
        )
        df_transposed = df_transposed.dropna(subset=['Tanggal'])
        df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
        
        # Konversi ke numeric
        for kolom in komoditas_list:
            if df_transposed[kolom].dtype == 'object':
                df_transposed[kolom] = df_transposed[kolom].str.replace(',', '').str.replace('"', '')
            df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
        
        # Interpolasi missing values
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
# FUNGSI PREDIKSI MASA DEPAN
# =====================================================================
def predict_future(model, last_sequence, scalers, komoditas_list, bulan_target, time_steps=20):
    """Prediksi harga untuk bulan tertentu di masa depan"""
    # Hitung jumlah minggu yang perlu diprediksi
    tanggal_sekarang = datetime.now()
    tanggal_target = datetime(bulan_target.year, bulan_target.month, 15)
    
    # Hitung jumlah minggu (asumsi data mingguan)
    minggu_prediksi = int((tanggal_target - tanggal_sekarang).days / 7)
    
    if minggu_prediksi <= 0:
        minggu_prediksi = 1
    
    # Prediksi iteratif
    current_sequence = last_sequence.copy()
    predictions = []
    
    for _ in range(minggu_prediksi):
        # Prediksi normalized
        pred_norm = model.predict(
            current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), 
            verbose=0
        )
        predictions.append(pred_norm[0])
        
        # Update sequence
        current_sequence = np.vstack([current_sequence[1:], pred_norm[0]])
    
    # Denormalisasi prediksi terakhir
    final_prediction = {}
    for i, komoditas in enumerate(komoditas_list):
        pred_value = scalers[komoditas].inverse_transform([[predictions[-1][i]]])[0][0]
        final_prediction[komoditas] = pred_value
    
    return final_prediction

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

st.sidebar.markdown("""
**Status File:**
""")

# Cek keberadaan file
model_exists = os.path.exists('best_lstm_model.h5')
dataset_exists = os.path.exists('dataset.xlsx')
eval_exists = os.path.exists('hasil_evaluasi_lstm_100epochs.csv')

if model_exists:
    st.sidebar.success("✓ Model tersedia")
else:
    st.sidebar.error("✗ Model tidak ditemukan")

if dataset_exists:
    st.sidebar.success("✓ Dataset tersedia")
else:
    st.sidebar.error("✗ Dataset tidak ditemukan")

if eval_exists:
    st.sidebar.success("✓ File evaluasi tersedia")
else:
    st.sidebar.info("○ File evaluasi opsional")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Informasi Model:**
- Model: LSTM Bidirectional
- Time Steps: 20
- Trained Epochs: 100
- Optimizer: Adam
- Loss: Huber
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**File yang Diperlukan:**
- best_lstm_model.h5
- dataset.xlsx
- hasil_evaluasi_lstm_100epochs.csv (opsional)
""")

# =====================================================================
# MAIN CONTENT
# =====================================================================
st.markdown('<h1 class="main-header">Model Prediksi Harga Pangan Indonesia 2025-2026</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Prediksi Harga Multi-Komoditas Menggunakan Long Short-Term Memory (LSTM)</p>', unsafe_allow_html=True)

# Load files dari direktori lokal
if model_exists and dataset_exists:
    try:
        # Load model
        with st.spinner('Memuat model dari direktori...'):
            model = load_trained_model()
        
        if model is not None:
            st.markdown('<div class="success-box">Model berhasil dimuat dari file best_lstm_model.h5</div>', unsafe_allow_html=True)
            
            # Load dataset
            with st.spinner('Memuat dataset dari direktori...'):
                df_raw = load_dataset()
            
            if df_raw is not None:
                st.markdown('<div class="success-box">Dataset berhasil dimuat dari file dataset.xlsx</div>', unsafe_allow_html=True)
                
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
                    # TAB 1: EVALUASI MODEL
                    # =====================================================================
                    with tab1:
                        st.markdown("### Hasil Evaluasi Model")
                        st.markdown('<div class="info-box">Berikut adalah hasil evaluasi performa model LSTM yang telah dilatih menggunakan data historis harga pangan</div>', unsafe_allow_html=True)
                        
                        # Load hasil evaluasi
                        df_eval = load_evaluation_results()
                        
                        if df_eval is not None:
                            st.markdown("#### Tabel Evaluasi Metrik per Komoditas")
                            
                            # Format kolom untuk tampilan yang lebih baik
                            df_eval_display = df_eval.copy()
                            if 'MAE' in df_eval_display.columns:
                                df_eval_display['MAE (Rp)'] = df_eval_display['MAE'].apply(lambda x: f"Rp {x:,.0f}")
                            if 'RMSE' in df_eval_display.columns:
                                df_eval_display['RMSE (Rp)'] = df_eval_display['RMSE'].apply(lambda x: f"Rp {x:,.0f}")
                            
                            # Pilih kolom yang akan ditampilkan
                            display_cols = ['Komoditas', 'MAPE (%)', 'MAE (Rp)', 'RMSE (Rp)']
                            st.dataframe(df_eval_display[display_cols], use_container_width=True, height=400)
                            
                            # Statistik
                            st.markdown("#### Ringkasan Statistik Evaluasi")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rata-rata MAPE", f"{df_eval['MAPE (%)'].mean():.2f}%")
                            with col2:
                                mae_col = 'MAE' if 'MAE' in df_eval.columns else 'MAE (Rp)'
                                if mae_col in df_eval.columns:
                                    mae_val = df_eval['MAE'].mean() if 'MAE' in df_eval.columns else df_eval['MAE (Rp)'].mean()
                                    st.metric("Rata-rata MAE", f"Rp {mae_val:,.0f}")
                            with col3:
                                rmse_col = 'RMSE' if 'RMSE' in df_eval.columns else 'RMSE (Rp)'
                                if rmse_col in df_eval.columns:
                                    rmse_val = df_eval['RMSE'].mean() if 'RMSE' in df_eval.columns else df_eval['RMSE (Rp)'].mean()
                                    st.metric("Rata-rata RMSE", f"Rp {rmse_val:,.0f}")
                            
                            # Visualisasi
                            st.markdown("---")
                            st.markdown("#### Visualisasi Metrik Evaluasi")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Grafik MAPE
                                df_sorted = df_eval.sort_values('MAPE (%)')
                                fig_mape = go.Figure()
                                fig_mape.add_trace(go.Bar(
                                    x=df_sorted['MAPE (%)'].head(10),
                                    y=df_sorted['Komoditas'].head(10),
                                    orientation='h',
                                    marker=dict(color='#2c3e50'),
                                    text=df_sorted['MAPE (%)'].head(10).apply(lambda x: f'{x:.2f}%'),
                                    textposition='auto'
                                ))
                                fig_mape.update_layout(
                                    title='Top 10 Komoditas - MAPE Terbaik',
                                    xaxis_title='MAPE (%)',
                                    yaxis_title='Komoditas',
                                    height=400,
                                    template='plotly_white',
                                    plot_bgcolor='white',
                                    paper_bgcolor='white'
                                )
                                st.plotly_chart(fig_mape, use_container_width=True)
                            
                            with col2:
                                # Grafik RMSE
                                mae_col = 'MAE' if 'MAE' in df_eval.columns else 'MAE (Rp)'
                                df_sorted_mae = df_eval.sort_values(mae_col)
                                fig_rmse = go.Figure()
                                
                                rmse_values = df_sorted_mae['RMSE'].head(10) if 'RMSE' in df_sorted_mae.columns else df_sorted_mae['RMSE (Rp)'].head(10)
                                
                                fig_rmse.add_trace(go.Bar(
                                    x=rmse_values,
                                    y=df_sorted_mae['Komoditas'].head(10),
                                    orientation='h',
                                    marker=dict(color='#34495e'),
                                    text=rmse_values.apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='auto'
                                ))
                                fig_rmse.update_layout(
                                    title='Top 10 Komoditas - RMSE Terbaik',
                                    xaxis_title='RMSE (Rp)',
                                    yaxis_title='Komoditas',
                                    height=400,
                                    template='plotly_white',
                                    plot_bgcolor='white',
                                    paper_bgcolor='white'
                                )
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            # Tampilkan gambar evaluasi metrik jika ada
                            if os.path.exists('evaluasi_metrik.png'):
                                st.markdown("---")
                                st.markdown("#### Visualisasi Lengkap Evaluasi Metrik")
                                st.image('evaluasi_metrik.png', use_container_width=True)
                            
                            # Tampilkan gambar prediksi vs aktual jika ada
                            if os.path.exists('prediksi_vs_aktual.png'):
                                st.markdown("---")
                                st.markdown("#### Grafik Prediksi vs Aktual")
                                st.image('prediksi_vs_aktual.png', use_container_width=True)
                        
                        else:
                            st.markdown('<div class="info-box">File hasil evaluasi tidak ditemukan. Model tetap dapat digunakan untuk prediksi.</div>', unsafe_allow_html=True)
                    
                    # =====================================================================
                    # TAB 2: PREDIKSI MASA DEPAN
                    # =====================================================================
                    with tab2:
                        st.markdown("### Prediksi Harga Komoditas Masa Depan")
                        st.markdown('<div class="info-box">Pilih bulan dan tahun yang ingin diprediksi, kemudian klik tombol "Mulai Prediksi" untuk melihat hasil prediksi harga seluruh komoditas</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            bulan_options = {
                                'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
                                'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
                                'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
                            }
                            bulan_selected = st.selectbox("Pilih Bulan", list(bulan_options.keys()))
                        
                        with col2:
                            tahun_selected = st.selectbox("Pilih Tahun", [2025, 2026, 2027])
                        
                        if st.button("Mulai Prediksi", key="predict_btn"):
                            bulan_target = datetime(tahun_selected, bulan_options[bulan_selected], 15)
                            
                            # Siapkan sequence terakhir
                            last_sequence = prepare_last_sequence(df_processed, komoditas_list, scalers)
                            
                            with st.spinner(f'Memprediksi harga untuk {bulan_selected} {tahun_selected}...'):
                                predictions = predict_future(
                                    model,
                                    last_sequence,
                                    scalers,
                                    komoditas_list,
                                    bulan_target
                                )
                            
                            st.markdown(f'<div class="success-box">Prediksi harga untuk <strong>{bulan_selected} {tahun_selected}</strong> berhasil dibuat</div>', unsafe_allow_html=True)
                            
                            # Tampilkan hasil prediksi
                            st.markdown("---")
                            st.markdown("### Hasil Prediksi Harga")
                            
                            # Buat DataFrame hasil
                            df_predictions = pd.DataFrame([
                                {
                                    'Komoditas': k,
                                    'Prediksi Harga (Rp)': f"Rp {v:,.0f}",
                                    'Nilai': v
                                }
                                for k, v in predictions.items()
                            ])
                            
                            # Tampilkan dalam 2 kolom
                            col1, col2 = st.columns(2)
                            
                            mid = len(df_predictions) // 2
                            
                            with col1:
                                st.markdown("#### Komoditas 1 - {}".format(mid))
                                st.dataframe(
                                    df_predictions[['Komoditas', 'Prediksi Harga (Rp)']].iloc[:mid],
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            with col2:
                                st.markdown("#### Komoditas {} - {}".format(mid+1, len(df_predictions)))
                                st.dataframe(
                                    df_predictions[['Komoditas', 'Prediksi Harga (Rp)']].iloc[mid:],
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            # Visualisasi top 10 komoditas termahal
                            st.markdown("---")
                            st.markdown("### Visualisasi Harga Prediksi")
                            
                            sorted_predictions = df_predictions.sort_values('Nilai', ascending=False).head(10)
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=sorted_predictions['Nilai'],
                                    y=sorted_predictions['Komoditas'],
                                    orientation='h',
                                    marker=dict(color='#2c3e50'),
                                    text=sorted_predictions['Nilai'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='auto'
                                )
                            ])
                            
                            fig.update_layout(
                                title=f'Top 10 Komoditas dengan Harga Tertinggi - {bulan_selected} {tahun_selected}',
                                xaxis_title='Harga (Rp)',
                                yaxis_title='Komoditas',
                                height=500,
                                template='plotly_white',
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Visualisasi top 10 komoditas termurah
                            st.markdown("---")
                            sorted_predictions_cheap = df_predictions.sort_values('Nilai', ascending=True).head(10)
                            
                            fig2 = go.Figure(data=[
                                go.Bar(
                                    x=sorted_predictions_cheap['Nilai'],
                                    y=sorted_predictions_cheap['Komoditas'],
                                    orientation='h',
                                    marker=dict(color='#34495e'),
                                    text=sorted_predictions_cheap['Nilai'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='auto'
                                )
                            ])
                            
                            fig2.update_layout(
                                title=f'Top 10 Komoditas dengan Harga Terendah - {bulan_selected} {tahun_selected}',
                                xaxis_title='Harga (Rp)',
                                yaxis_title='Komoditas',
                                height=500,
                                template='plotly_white',
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Download button
                            st.markdown("---")
                            st.markdown("### Unduh Hasil Prediksi")
                            csv = df_predictions[['Komoditas', 'Prediksi Harga (Rp)']].to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="Unduh Hasil Prediksi (CSV)",
                                data=csv,
                                file_name=f"prediksi_harga_{bulan_selected}_{tahun_selected}.csv",
                                mime="text/csv"
                            )
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

else:
    # Tampilan jika file tidak ditemukan
    st.markdown("""
    <div class="warning-box">
        <h3>File yang Diperlukan Tidak Ditemukan</h3>
        <p style="margin: 0.5rem 0;">Pastikan file-file berikut ada di direktori yang sama dengan app.py:</p>
        <ul style="padding-left: 1.5rem; margin-top: 0.5rem;">
            <li style="margin-bottom: 0.5rem;"><strong>best_lstm_model.h5</strong> - Model LSTM yang sudah dilatih</li>
            <li style="margin-bottom: 0.5rem;"><strong>dataset.xlsx</strong> - Dataset historis harga pangan</li>
            <li style="margin-bottom: 0.5rem;"><strong>hasil_evaluasi_lstm_100epochs.csv</strong> - File evaluasi (opsional)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>Struktur Direktori yang Benar</h3>
        <pre style="background-color: #f8f9fa; padding: 1rem; border-radius: 4px; margin-top: 0.5rem;">
project/
├── app.py
├── best_lstm_model.h5
├── dataset.xlsx
├── hasil_evaluasi_lstm_100epochs.csv
├── evaluasi_metrik.png (opsional)
├── prediksi_vs_aktual.png (opsional)
└── requirements.txt
        </pre>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1.5rem;'>
    <p style="margin: 0;">Model Prediksi Harga Pangan Indonesia</p>
    <p style="margin: 0.5rem 0 0 0;">Dibuat menggunakan Streamlit dan TensorFlow</p>
</div>
""", unsafe_allow_html=True)
