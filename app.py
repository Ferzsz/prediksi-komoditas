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

st.set_page_config(
    page_title="Prediksi Harga Pangan 2025-2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    .score-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .score-card h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def get_metric_interpretation(mape_value):
    """Memberikan interpretasi kualitas prediksi berdasarkan MAPE"""
    if mape_value < 5:
        return "Sangat Baik", "success-box", "Model memiliki akurasi prediksi yang sangat tinggi dengan error di bawah 5%"
    elif mape_value < 10:
        return "Baik", "excellent-box", "Model memiliki akurasi prediksi yang baik dengan error 5-10%"
    elif mape_value < 20:
        return "Cukup Baik", "info-box", "Model memiliki akurasi prediksi yang cukup memadai dengan error 10-20%"
    else:
        return "Perlu Perbaikan", "warning-box", "Model perlu ditingkatkan untuk akurasi yang lebih baik (error >20%)"

@st.cache_resource
def load_trained_model(model_path='best_lstm_model.h5'):
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

def preprocess_data(df_raw):
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
    scalers = {}
    for kolom in komoditas_list:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_processed[[kolom]].values)
        scalers[kolom] = scaler
    return scalers

def predict_single_commodity(model, last_sequence, scaler, commodity_index, bulan_target, time_steps=20):
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
    
    pred_value = scaler.inverse_transform([[predictions[-1][commodity_index]]])[0][0]
    return pred_value, minggu_prediksi

def prepare_last_sequence(df_processed, komoditas_list, scalers, time_steps=20):
    data_normalized = np.zeros((len(df_processed), len(komoditas_list)))
    
    for i, kolom in enumerate(komoditas_list):
        data_normalized[:, i] = scalers[kolom].transform(
            df_processed[[kolom]].values
        ).flatten()
    
    return data_normalized[-time_steps:]

# SIDEBAR
st.sidebar.markdown("### Model Prediksi Harga Pangan")
st.sidebar.markdown("---")

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

model_exists = os.path.exists('best_lstm_model.h5')
st.sidebar.markdown("**Status Model:**")
if model_exists:
    st.sidebar.success("Model tersedia")
else:
    st.sidebar.error("Model tidak ditemukan")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Penjelasan Metrik:**

**MAPE (Mean Absolute Percentage Error)**
- < 5%: Sangat Baik
- 5-10%: Baik
- 10-20%: Cukup Baik
- > 20%: Perlu Perbaikan

**MAE (Mean Absolute Error)**
Rata-rata selisih absolut antara nilai prediksi dan aktual (Rupiah)

**RMSE (Root Mean Squared Error)**
Akar dari rata-rata kuadrat error (lebih sensitif terhadap outlier)
""")

# MAIN CONTENT
st.markdown('<h1 class="main-header">Model Prediksi Harga Pangan Indonesia 2025-2026</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Prediksi Harga Multi-Komoditas Menggunakan Long Short-Term Memory (LSTM)</p>', unsafe_allow_html=True)

if model_exists and uploaded_dataset is not None:
    try:
        with st.spinner('Memuat model...'):
            model = load_trained_model()
        
        if model is not None:
            st.markdown('<div class="success-box">Model berhasil dimuat dari file best_lstm_model.h5</div>', unsafe_allow_html=True)
            
            with st.spinner('Memuat dataset...'):
                df_raw = pd.read_excel(uploaded_dataset)
            
            st.markdown('<div class="success-box">Dataset berhasil diupload</div>', unsafe_allow_html=True)
            
            with st.spinner('Memproses dataset...'):
                df_processed, komoditas_list = preprocess_data(df_raw)
            
            if df_processed is not None and komoditas_list is not None:
                scalers = create_scalers(df_processed, komoditas_list)
                
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
                
                # 3 TAB TERPISAH
                tab1, tab2, tab3 = st.tabs(["Evaluasi Model", "Prediksi Harga", "Evaluasi Keseluruhan"])
                
                # =====================================================================
                # TAB 1: EVALUASI MODEL - PILIH KOMODITAS
                # =====================================================================
                with tab1:
                    st.markdown("### Evaluasi Model Per Komoditas")
                    st.markdown('<div class="info-box">Pilih komoditas untuk melihat skor evaluasi model (MAPE, MAE, RMSE)</div>', unsafe_allow_html=True)
                    
                    if os.path.exists('hasil_evaluasi_lstm_100epochs.csv'):
                        df_eval = pd.read_csv('hasil_evaluasi_lstm_100epochs.csv')
                        
                        # Pilih komoditas
                        selected_commodity = st.selectbox(
                            "Pilih Komoditas untuk Evaluasi",
                            df_eval['Komoditas'].tolist(),
                            key="eval_commodity"
                        )
                        
                        if selected_commodity:
                            # Ambil data komoditas yang dipilih
                            commodity_data = df_eval[df_eval['Komoditas'] == selected_commodity].iloc[0]
                            
                            mape_val = commodity_data['MAPE (%)']
                            mae_val = commodity_data['MAE'] if 'MAE' in commodity_data else 0
                            rmse_val = commodity_data['RMSE'] if 'RMSE' in commodity_data else 0
                            
                            interpretation, box_class, description = get_metric_interpretation(mape_val)
                            
                            st.markdown("---")
                            st.markdown(f"### Hasil Evaluasi: {selected_commodity}")
                            
                            # Tampilkan skor dalam card
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="score-card">
                                    <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">MAPE</h3>
                                    <h2 style="color: {'#28a745' if mape_val < 5 else '#17a2b8' if mape_val < 10 else '#ffc107' if mape_val < 20 else '#dc3545'};">{mape_val:.2f}%</h2>
                                    <p style="color: #7f8c8d; margin-top: 0.5rem;">Mean Absolute Percentage Error</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="score-card">
                                    <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">MAE</h3>
                                    <h2 style="color: #34495e;">Rp {mae_val:,.0f}</h2>
                                    <p style="color: #7f8c8d; margin-top: 0.5rem;">Mean Absolute Error</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="score-card">
                                    <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">RMSE</h3>
                                    <h2 style="color: #34495e;">Rp {rmse_val:,.0f}</h2>
                                    <p style="color: #7f8c8d; margin-top: 0.5rem;">Root Mean Squared Error</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Interpretasi
                            st.markdown("---")
                            st.markdown("### Interpretasi Hasil")
                            
                            st.markdown(f"""
                            <div class="{box_class}">
                                <h4>Kategori: {interpretation}</h4>
                                <p>{description}</p>
                                <ul style="margin-top: 1rem;">
                                    <li><strong>MAPE {mape_val:.2f}%</strong> menunjukkan rata-rata error persentase prediksi</li>
                                    <li><strong>MAE Rp {mae_val:,.0f}</strong> adalah rata-rata selisih harga prediksi dengan aktual</li>
                                    <li><strong>RMSE Rp {rmse_val:,.0f}</strong> memberikan bobot lebih pada error yang besar</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Perbandingan dengan rata-rata
                            st.markdown("---")
                            st.markdown("### Perbandingan dengan Rata-rata Model")
                            
                            avg_mape = df_eval['MAPE (%)'].mean()
                            avg_mae = df_eval['MAE'].mean() if 'MAE' in df_eval.columns else 0
                            avg_rmse = df_eval['RMSE'].mean() if 'RMSE' in df_eval.columns else 0
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                delta_mape = mape_val - avg_mape
                                st.metric(
                                    "MAPE vs Rata-rata",
                                    f"{mape_val:.2f}%",
                                    f"{delta_mape:+.2f}%",
                                    delta_color="inverse"
                                )
                            
                            with col2:
                                delta_mae = mae_val - avg_mae
                                st.metric(
                                    "MAE vs Rata-rata",
                                    f"Rp {mae_val:,.0f}",
                                    f"Rp {delta_mae:+,.0f}",
                                    delta_color="inverse"
                                )
                            
                            with col3:
                                delta_rmse = rmse_val - avg_rmse
                                st.metric(
                                    "RMSE vs Rata-rata",
                                    f"Rp {rmse_val:,.0f}",
                                    f"Rp {delta_rmse:+,.0f}",
                                    delta_color="inverse"
                                )
                    
                    else:
                        st.markdown('<div class="warning-box">File hasil evaluasi tidak ditemukan</div>', unsafe_allow_html=True)
                
                # =====================================================================
                # TAB 2: PREDIKSI HARGA - TANPA GRAFIK
                # =====================================================================
                with tab2:
                    st.markdown("### Prediksi Harga Komoditas")
                    st.markdown('<div class="info-box">Pilih komoditas, tahun, dan bulan untuk memprediksi harga</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        komoditas_selected = st.selectbox(
                            "Pilih Komoditas",
                            komoditas_list,
                            key="pred_commodity"
                        )
                    
                    with col2:
                        tahun_selected = st.selectbox("Pilih Tahun", [2025, 2026, 2027, 2028])
                    
                    with col3:
                        bulan_options = {
                            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
                            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
                            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
                        }
                        bulan_selected = st.selectbox("Pilih Bulan", list(bulan_options.keys()))
                    
                    if st.button("Mulai Prediksi", key="predict_btn"):
                        bulan_target = datetime(tahun_selected, bulan_options[bulan_selected], 15)
                        last_sequence = prepare_last_sequence(df_processed, komoditas_list, scalers)
                        commodity_index = komoditas_list.index(komoditas_selected)
                        
                        with st.spinner(f'Memprediksi harga {komoditas_selected}...'):
                            predicted_price, weeks_ahead = predict_single_commodity(
                                model,
                                last_sequence,
                                scalers[komoditas_selected],
                                commodity_index,
                                bulan_target
                            )
                        
                        st.markdown(f'<div class="success-box">Prediksi berhasil dibuat untuk <strong>{komoditas_selected}</strong></div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("### Hasil Prediksi")
                        
                        st.markdown(f"""
                        <div class="prediction-result">
                            <p style="margin: 0; font-size: 1.2rem; color: #555;">Prediksi Harga</p>
                            <h2 style="margin: 0.5rem 0;">Rp {predicted_price:,.0f}</h2>
                            <p style="margin: 0;"><strong>{komoditas_selected}</strong></p>
                            <p style="margin: 0; font-size: 0.9rem;">{bulan_selected} {tahun_selected}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Analisis Perbandingan
                        st.markdown("---")
                        st.markdown("### Analisis Perbandingan Harga")
                        
                        last_actual_price = df_processed[komoditas_selected].iloc[-1]
                        price_change = predicted_price - last_actual_price
                        price_change_pct = (price_change / last_actual_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Harga Terakhir", 
                                f"Rp {last_actual_price:,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Harga Prediksi", 
                                f"Rp {predicted_price:,.0f}",
                                delta=f"Rp {price_change:,.0f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Perubahan", 
                                f"{price_change_pct:+.2f}%",
                                delta=f"{'Naik' if price_change > 0 else 'Turun'}"
                            )
                        
                        # Interpretasi
                        if price_change_pct > 10:
                            st.markdown('<div class="warning-box"><strong>Peringatan:</strong> Prediksi menunjukkan kenaikan harga signifikan (>10%). Perlu antisipasi dalam pengelolaan stok.</div>', unsafe_allow_html=True)
                        elif price_change_pct > 5:
                            st.markdown('<div class="info-box"><strong>Informasi:</strong> Prediksi menunjukkan kenaikan harga moderat (5-10%).</div>', unsafe_allow_html=True)
                        elif price_change_pct < -10:
                            st.markdown('<div class="excellent-box"><strong>Informasi:</strong> Prediksi menunjukkan penurunan harga signifikan (>10%).</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box"><strong>Informasi:</strong> Prediksi menunjukkan harga relatif stabil.</div>', unsafe_allow_html=True)
                        
                        # Download
                        st.markdown("---")
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
                
                # =====================================================================
                # TAB 3: EVALUASI KESELURUHAN
                # =====================================================================
                with tab3:
                    st.markdown("### Evaluasi Keseluruhan Model")
                    st.markdown('<div class="info-box">Ringkasan performa model untuk semua komoditas</div>', unsafe_allow_html=True)
                    
                    if os.path.exists('hasil_evaluasi_lstm_100epochs.csv'):
                        df_eval = pd.read_csv('hasil_evaluasi_lstm_100epochs.csv')
                        
                        # Statistik keseluruhan
                        st.markdown("#### Ringkasan Statistik")
                        
                        avg_mape = df_eval['MAPE (%)'].mean()
                        interpretation, box_class, description = get_metric_interpretation(avg_mape)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Rata-rata MAPE", f"{avg_mape:.2f}%")
                        with col2:
                            mae_val = df_eval['MAE'].mean() if 'MAE' in df_eval.columns else 0
                            st.metric("Rata-rata MAE", f"Rp {mae_val:,.0f}")
                        with col3:
                            rmse_val = df_eval['RMSE'].mean() if 'RMSE' in df_eval.columns else 0
                            st.metric("Rata-rata RMSE", f"Rp {rmse_val:,.0f}")
                        
                        st.markdown(f'<div class="{box_class}"><strong>Interpretasi Keseluruhan:</strong> {description}</div>', unsafe_allow_html=True)
                        
                        # Tabel lengkap
                        st.markdown("---")
                        st.markdown("#### Tabel Evaluasi Semua Komoditas")
                        
                        df_eval_display = df_eval.copy()
                        df_eval_display['Interpretasi'] = df_eval_display['MAPE (%)'].apply(
                            lambda x: get_metric_interpretation(x)[0]
                        )
                        df_eval_display = df_eval_display.sort_values('MAPE (%)')
                        
                        if 'MAE' in df_eval_display.columns:
                            df_eval_display['MAE (Rp)'] = df_eval_display['MAE'].apply(lambda x: f"Rp {x:,.0f}")
                        if 'RMSE' in df_eval_display.columns:
                            df_eval_display['RMSE (Rp)'] = df_eval_display['RMSE'].apply(lambda x: f"Rp {x:,.0f}")
                        
                        display_cols = ['Komoditas', 'MAPE (%)', 'MAE (Rp)', 'RMSE (Rp)', 'Interpretasi']
                        st.dataframe(df_eval_display[display_cols], use_container_width=True, height=600)
                        
                        csv = df_eval_display[display_cols].to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="Unduh Evaluasi Keseluruhan (CSV)",
                            data=csv,
                            file_name="evaluasi_keseluruhan.csv",
                            mime="text/csv"
                        )
                        
                        # Visualisasi
                        st.markdown("---")
                        st.markdown("#### Visualisasi Performa Model")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            df_sorted = df_eval_display.sort_values('MAPE (%)').head(15)
                            colors = ['#28a745' if x < 5 else '#17a2b8' if x < 10 else '#ffc107' if x < 20 else '#dc3545' 
                                     for x in df_sorted['MAPE (%)']]
                            
                            fig_mape = go.Figure()
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
                            category_counts = df_eval_display['Interpretasi'].value_counts()
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=category_counts.index,
                                values=category_counts.values,
                                marker=dict(colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545']),
                                hole=0.4
                            )])
                            fig_pie.update_layout(
                                title='Distribusi Kategori Performa',
                                height=500,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    else:
                        st.markdown('<div class="warning-box">File hasil evaluasi tidak ditemukan</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

elif not model_exists:
    st.markdown("""
    <div class="warning-box">
        <h3>Model Tidak Ditemukan</h3>
        <p>File <strong>best_lstm_model.h5</strong> tidak ditemukan. Pastikan file ada di direktori.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
        <h3>Panduan Penggunaan Aplikasi</h3>
        <ol style="padding-left: 1.5rem;">
            <li style="margin-bottom: 0.5rem;">Upload dataset Excel di sidebar</li>
            <li style="margin-bottom: 0.5rem;"><strong>Tab Evaluasi Model:</strong> Pilih komoditas untuk melihat skor MAE, RMSE, MAPE</li>
            <li style="margin-bottom: 0.5rem;"><strong>Tab Prediksi Harga:</strong> Pilih komoditas, tahun, bulan untuk prediksi</li>
            <li style="margin-bottom: 0.5rem;"><strong>Tab Evaluasi Keseluruhan:</strong> Lihat performa semua komoditas</li>
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
