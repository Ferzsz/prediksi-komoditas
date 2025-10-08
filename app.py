import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================================
# KONFIGURASI HALAMAN
# ===========================================================================================

st.set_page_config(
    page_title="Prediksi Harga Komoditas Pangan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (sama seperti sebelumnya)
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .title-text {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================================================================================
# FUNGSI HELPER
# ===========================================================================================

def calculate_target_date(year, month, week=None):
    """
    Hitung tanggal target berdasarkan tahun, bulan, dan minggu (opsional)
    
    Args:
        year: Tahun target
        month: Bulan target (1-12)
        week: Minggu ke- (1-4) atau None untuk prediksi bulanan
    
    Returns:
        datetime object untuk tanggal target
    """
    if week is None:
        # Prediksi bulanan - gunakan pertengahan bulan (hari ke-15)
        return datetime(year, month, 15)
    else:
        # Prediksi mingguan - hitung tanggal berdasarkan minggu
        # Minggu 1 = hari ke-7, Minggu 2 = hari ke-14, Minggu 3 = hari ke-21, Minggu 4 = hari ke-28
        day = week * 7
        # Pastikan tidak melebihi hari dalam bulan
        import calendar
        max_day = calendar.monthrange(year, month)[1]
        day = min(day, max_day)
        return datetime(year, month, day)

def load_and_validate_metrics(komoditas_list_from_dataset):
    """Load pre-computed metrics dan validasi dengan dataset"""
    try:
        df_eval = pd.read_csv('hasil_evaluasi_lstm_100epochs.csv')
        csv_komoditas = set(df_eval['Komoditas'].tolist())
        dataset_komoditas = set(komoditas_list_from_dataset)
        
        if csv_komoditas != dataset_komoditas:
            st.warning(f"""
            ⚠️ **Dataset berbeda terdeteksi!**
            
            - Komoditas di file evaluasi: {len(csv_komoditas)} items
            - Komoditas di dataset upload: {len(dataset_komoditas)} items
            
            Sistem akan menghitung metrik evaluasi secara **real-time** berdasarkan dataset yang baru diupload.
            """)
            return None, "different_dataset"
        
        st.info("✅ Menggunakan metrik evaluasi pre-computed dari hasil training (100 epochs optimal)")
        return df_eval, "same_dataset"
        
    except FileNotFoundError:
        st.info("ℹ️ File evaluasi pre-computed tidak ditemukan. Menghitung metrik secara real-time...")
        return None, "file_not_found"
    except Exception as e:
        st.warning(f"⚠️ Error loading evaluation file: {str(e)}. Menghitung metrik secara real-time...")
        return None, "error"

def calculate_metrics_realtime(model, data_normalized, scalers, komoditas_list, TIME_STEPS=20):
    """Hitung metrik evaluasi secara real-time"""
    split_idx = int(len(data_normalized) * 0.90)
    test_data = data_normalized[split_idx:]
    
    all_metrics = []
    X_test, y_test = [], []
    
    for i in range(TIME_STEPS, len(test_data)):
        X_test.append(test_data[i-TIME_STEPS:i])
        y_test.append(test_data[i])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    if len(X_test) > 0:
        y_pred = model.predict(X_test, verbose=0)
        
        for idx, commodity_name in enumerate(komoditas_list):
            y_test_commodity = y_test[:, idx]
            y_pred_commodity = y_pred[:, idx]
            
            y_test_orig = scalers[commodity_name].inverse_transform(y_test_commodity.reshape(-1, 1))
            y_pred_orig = scalers[commodity_name].inverse_transform(y_pred_commodity.reshape(-1, 1))
            
            rmse_val = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            mae_val = mean_absolute_error(y_test_orig, y_pred_orig)
            mape_val = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
            
            all_metrics.append({
                'Komoditas': commodity_name,
                'RMSE': rmse_val,
                'MAE': mae_val,
                'MAPE': mape_val
            })
    
    return all_metrics

# ===========================================================================================
# SIDEBAR
# ===========================================================================================

with st.sidebar:
    st.markdown("### Informasi Model")
    st.markdown("---")
    
    st.markdown("**Arsitektur Model**")
    st.markdown("""
    - Bidirectional LSTM (128 units)
    - LSTM (64 units)
    - Dense Layers (64, 32)
    - Regularisasi: L2 + Dropout
    """)
    
    st.markdown("---")
    st.markdown("**Hyperparameter**")
    st.markdown("""
    - Epochs: 100
    - Batch Size: 32
    - Learning Rate: 0.001
    - Optimizer: Adam
    - Loss: Huber Loss
    """)
    
    st.markdown("---")
    st.markdown("**Preprocessing**")
    st.markdown("""
    - Time Steps: 20
    - Normalisasi: MinMaxScaler
    - Train/Test Split: 90/10
    - Interpolasi: Linear
    """)
    
    st.markdown("---")
    st.markdown("**Performa Model**")
    st.markdown("""
    - Target MAPE: < 10%
    - Early Stopping: Patience 20
    - ReduceLR: Patience 8
    """)

# ===========================================================================================
# MAIN CONTENT
# ===========================================================================================

st.markdown('<p class="title-text">Prediksi Harga Komoditas Pangan</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Sistem Prediksi Harga Menggunakan LSTM Neural Network</p>', unsafe_allow_html=True)

# ===========================================================================================
# UPLOAD DATASET
# ===========================================================================================

st.markdown("### Upload Dataset")
st.markdown("Upload file Excel (.xlsx) yang berisi data harga komoditas historis")

uploaded_file = st.file_uploader(
    "Pilih file dataset",
    type=['xlsx'],
    help="Format: Kolom 1 = No, Kolom 2 = Komoditas, Kolom 3+ = Data harga dengan header tanggal"
)

if uploaded_file is not None:
    try:
        # Load dataset
        df_raw = pd.read_excel(uploaded_file)
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        st.markdown('<div class="success-box">Dataset berhasil dimuat</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Komoditas", len(komoditas_list))
        with col2:
            st.metric("Total Data Points", df_raw.shape[1] - 2)
        with col3:
            st.metric("Rentang Waktu", f"{df_raw.shape[1] - 2} minggu")
        
        st.markdown("---")
        
        # ===========================================================================================
        # FORM PREDIKSI - DENGAN OPSI MINGGUAN
        # ===========================================================================================
        
        st.markdown("### Prediksi Harga")
        
        col_form1, col_form2, col_form3, col_form4 = st.columns(4)
        
        with col_form1:
            selected_commodity = st.selectbox(
                "Pilih Komoditas",
                options=komoditas_list,
                help="Pilih komoditas yang ingin diprediksi"
            )
        
        with col_form2:
            selected_year = st.selectbox(
                "Pilih Tahun",
                options=[2025, 2026],
                help="Pilih tahun untuk prediksi"
            )
        
        with col_form3:
            selected_month = st.selectbox(
                "Pilih Bulan",
                options=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                help="Pilih bulan untuk prediksi"
            )
        
        with col_form4:
            selected_week = st.selectbox(
                "Minggu (Opsional)",
                options=['Bulanan (default)', 'Minggu 1', 'Minggu 2', 'Minggu 3', 'Minggu 4'],
                help="Pilih 'Bulanan' untuk prediksi pertengahan bulan, atau pilih minggu spesifik"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Prediksi Harga", use_container_width=True):
            with st.spinner("Memproses prediksi..."):
                try:
                    # ===========================================================================================
                    # PREPROCESSING
                    # ===========================================================================================
                    
                    # Transpose data
                    df_data = df_raw.iloc[:, 2:]
                    df_transposed = df_data.T
                    df_transposed.columns = komoditas_list
                    df_transposed.reset_index(inplace=True)
                    df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
                    
                    # Convert tanggal
                    df_transposed['Tanggal'] = pd.to_datetime(df_transposed['Tanggal'], format='%d/ %m/ %Y', errors='coerce')
                    df_transposed = df_transposed.dropna(subset=['Tanggal'])
                    df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
                    
                    # Konversi ke numeric
                    for kolom in komoditas_list:
                        if df_transposed[kolom].dtype == 'object':
                            df_transposed[kolom] = df_transposed[kolom].str.replace(',', '').str.replace('"', '')
                        df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
                    
                    # Interpolasi
                    df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(method='linear', limit_direction='both')
                    df_transposed[komoditas_list] = df_transposed[komoditas_list].fillna(method='bfill').fillna(method='ffill')
                    
                    # Normalisasi
                    scalers = {}
                    data_normalized = np.zeros((len(df_transposed), len(komoditas_list)))
                    
                    for i, kolom in enumerate(komoditas_list):
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        data_normalized[:, i] = scaler.fit_transform(df_transposed[[kolom]].values).flatten()
                        scalers[kolom] = scaler
                    
                    # ===========================================================================================
                    # HITUNG TARGET DATE BERDASARKAN PILIHAN MINGGUAN
                    # ===========================================================================================
                    
                    month_dict = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
                                 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
                    
                    target_month = month_dict[selected_month]
                    last_date = df_transposed['Tanggal'].iloc[-1]
                    
                    # Tentukan minggu berdasarkan pilihan user
                    if selected_week == 'Bulanan (default)':
                        week_num = None
                        week_label = "Pertengahan Bulan"
                    else:
                        week_num = int(selected_week.split()[1])
                        week_label = selected_week
                    
                    target_date = calculate_target_date(selected_year, target_month, week_num)
                    weeks_to_predict = max(1, int((target_date - last_date).days / 7))
                    
                    # ===========================================================================================
                    # LOAD MODEL DAN PREDIKSI
                    # ===========================================================================================
                    
                    model = load_model('best_lstm_model.h5', compile=False)
                    TIME_STEPS = 20
                    predictions = []
                    
                    current_sequence = data_normalized[-TIME_STEPS:].reshape(1, TIME_STEPS, len(komoditas_list))
                    
                    for _ in range(weeks_to_predict):
                        pred = model.predict(current_sequence, verbose=0)
                        predictions.append(pred[0])
                        
                        new_pred = pred.reshape(1, 1, len(komoditas_list))
                        current_sequence = np.append(current_sequence[:, 1:, :], new_pred, axis=1)
                    
                    # Ambil prediksi untuk komoditas yang dipilih
                    commodity_idx = komoditas_list.index(selected_commodity)
                    predicted_price_norm = predictions[-1][commodity_idx]
                    predicted_price = scalers[selected_commodity].inverse_transform([[predicted_price_norm]])[0, 0]
                    
                    # ===========================================================================================
                    # LOAD/HITUNG METRIK
                    # ===========================================================================================
                    
                    df_eval_metrics, status = load_and_validate_metrics(komoditas_list)
                    
                    if df_eval_metrics is not None and status == "same_dataset":
                        all_metrics = df_eval_metrics.rename(columns={'MAPE (%)': 'MAPE'}).to_dict('records')
                        metric_source = "pre-computed (100 epochs optimal)"
                    else:
                        with st.spinner("Menghitung metrik evaluasi untuk dataset baru..."):
                            all_metrics = calculate_metrics_realtime(model, data_normalized, scalers, komoditas_list, TIME_STEPS)
                        metric_source = "real-time calculation"
                    
                    if len(all_metrics) > 0:
                        selected_metrics = [m for m in all_metrics if m['Komoditas'] == selected_commodity][0]
                        rmse = selected_metrics['RMSE']
                        mae = selected_metrics['MAE']
                        mape = selected_metrics['MAPE']
                    else:
                        rmse, mae, mape = 0, 0, 0
                    
                    # ===========================================================================================
                    # TAMPILKAN HASIL
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown("### Hasil Prediksi")
                    
                    # Info periode prediksi
                    period_info = f"{selected_month} {selected_year} - {week_label}"
                    st.markdown(f'<div class="info-box">📅 <strong>Periode Prediksi:</strong> {period_info} | <strong>Tanggal Target:</strong> {target_date.strftime("%d %B %Y")}</div>', unsafe_allow_html=True)
                    
                    # Info sumber metrik
                    if metric_source == "pre-computed (100 epochs optimal)":
                        st.markdown('<div class="info-box">📊 <strong>Metrik Evaluasi:</strong> Menggunakan hasil pre-computed dari training 100 epochs</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">📊 <strong>Metrik Evaluasi:</strong> Dihitung secara real-time dari dataset yang baru diupload</div>', unsafe_allow_html=True)
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                    
                    with col_result1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="Harga Prediksi",
                            value=f"Rp {predicted_price:,.0f}",
                            help="Prediksi harga untuk periode yang dipilih"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_result2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(label="RMSE", value=f"Rp {rmse:,.0f}", help="Root Mean Squared Error")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_result3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(label="MAE", value=f"Rp {mae:,.0f}", help="Mean Absolute Error")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_result4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(label="MAPE", value=f"{mape:.2f}%", help="Mean Absolute Percentage Error")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ===========================================================================================
                    # VISUALISASI
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown("### Visualisasi Prediksi")
                    
                    tab1, tab2, tab3 = st.tabs(["Grafik Prediksi Harga", "Grafik Metrik Evaluasi Keseluruhan", "Grafik Metrik Evaluasi"])
                    
                    with tab1:
                        historical_dates = df_transposed['Tanggal'].tolist()
                        historical_prices = df_transposed[selected_commodity].tolist()
                        
                        future_dates = pd.date_range(start=last_date, periods=weeks_to_predict + 1, freq='W')[1:]
                        future_prices = [scalers[selected_commodity].inverse_transform([[p[commodity_idx]]])[0, 0] for p in predictions]
                        
                        fig1 = go.Figure()
                        
                        fig1.add_trace(go.Scatter(
                            x=historical_dates, y=historical_prices,
                            mode='lines+markers', name='Data Historis',
                            line=dict(color='#2E86AB', width=3), marker=dict(size=6)
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=future_dates, y=future_prices,
                            mode='lines+markers', name='Prediksi',
                            line=dict(color='#E63946', width=3, dash='dash'), marker=dict(size=8, symbol='square')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date], y=[predicted_price],
                            mode='markers', name=f'Target ({period_info})',
                            marker=dict(size=15, color='#27ae60', symbol='star')
                        ))
                        
                        fig1.update_layout(
                            title=dict(text=f'Prediksi Harga {selected_commodity}',
                                     font=dict(size=20, color='#2c3e50', family='Arial Black')),
                            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                            hovermode='x unified', template='plotly_white', height=500,
                            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                    with tab2:
                        # [Kode Tab 2 sama seperti sebelumnya - Grafik Metrik Evaluasi Keseluruhan]
                        st.markdown(f"#### Evaluasi Performa Model untuk Semua Komoditas")
                        st.markdown(f"*Sumber metrik: {metric_source}*")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            # [Sisanya sama seperti kode sebelumnya untuk Tab 2]
                            # ... (copy dari kode sebelumnya)
                            
                    with tab3:
                        # [Kode Tab 3 sama seperti sebelumnya - Grafik Metrik Evaluasi per Komoditas]
                        st.markdown(f"#### Evaluasi Metrik - {selected_commodity}")
                        
                        # [Sisanya sama seperti kode sebelumnya untuk Tab 3]
                        # ... (copy dari kode sebelumnya)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error saat membaca dataset: {str(e)}")

else:
    st.markdown('<div class="info-box">Silakan upload dataset untuk memulai prediksi</div>', unsafe_allow_html=True)
    st.markdown("#### Format Dataset yang Diharapkan:")
    st.markdown("""
    - **Kolom 1**: No (1, 2, 3, ...)
    - **Kolom 2**: Nama Komoditas
    - **Kolom 3+**: Data harga dengan header tanggal
    - **File format**: Excel (.xlsx)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)
