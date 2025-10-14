import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime
import warnings
import streamlit_shadcn_ui as ui
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

# Custom CSS with Tailwind-inspired styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 3rem 5rem;
        max-width: 1800px;
        background: linear-gradient(to bottom, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Sidebar Modern Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem 1.5rem;
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #334155;
        margin: 1.5rem 0;
    }
    
    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 3rem;
        border-radius: 24px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="white" stroke-width="0.5" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .title-text {
        color: white;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        letter-spacing: -0.04em;
        line-height: 1.1;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 1;
    }
    
    .subtitle-text {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-bottom: 0;
        font-weight: 400;
        letter-spacing: -0.01em;
        position: relative;
        z-index: 1;
    }
    
    /* Card Container */
    .card-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .card-container:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Section Header */
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 3rem 0 2rem 0;
        padding-left: 1.5rem;
        border-left: 6px solid #667eea;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
    }
    
    .section-header::before {
        content: '';
        width: 12px;
        height: 12px;
        background: #667eea;
        border-radius: 50%;
        margin-right: 1rem;
        margin-left: -1.9rem;
    }
    
    /* Custom Alert Boxes - Tailwind Style */
    .alert-modern {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        border-left: 6px solid;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .alert-modern::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(209, 250, 229, 0.8) 0%, rgba(167, 243, 208, 0.8) 100%);
        border-color: #10b981;
        color: #065f46;
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(219, 234, 254, 0.8) 0%, rgba(191, 219, 254, 0.8) 100%);
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.8) 0%, rgba(253, 230, 138, 0.8) 100%);
        border-color: #f59e0b;
        color: #92400e;
    }
    
    /* Metric Cards Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    /* File Uploader Modern */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 3px dashed #cbd5e1;
        border-radius: 20px;
        padding: 4rem 3rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 2rem 0;
        position: relative;
    }
    
    [data-testid="stFileUploader"]::before {
        content: '📁';
        font-size: 4rem;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        opacity: 0.1;
        pointer-events: none;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8fafc 0%, #ede9fe 100%);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
        transform: translateY(-4px);
    }
    
    /* Selectbox Modern */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
    }
    
    /* Tab Styling - Ultra Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 12px;
        margin-bottom: 2.5rem;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 14px;
        padding: 16px 32px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.02em;
        position: relative;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"]::before {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 50%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        border-radius: 2px;
    }
    
    .stTabs [aria-selected="false"] {
        background: transparent;
        color: #64748b;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background: white;
        color: #475569;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Dataframe Modern */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Table Headers Gradient */
    thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1.25rem 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.85rem;
        border: none !important;
    }
    
    tbody tr {
        transition: all 0.2s ease;
    }
    
    tbody tr:hover {
        background-color: #f8fafc !important;
        transform: scale(1.01);
    }
    
    tbody td {
        padding: 1rem !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 0.75rem;
    }
    
    /* Spinner Modern */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
        border-width: 4px !important;
    }
    
    /* Button enhancement */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Loading animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card-container {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Badge styling in sidebar */
    [data-testid="stSidebar"] ul {
        list-style: none;
        padding-left: 0;
    }
    
    [data-testid="stSidebar"] li {
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    [data-testid="stSidebar"] li::before {
        content: '▸';
        position: absolute;
        left: 0;
        color: #667eea;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================================================================================
# FUNGSI UNTUK LOAD DAN VALIDASI METRIK
# ===========================================================================================

def load_and_validate_metrics(komoditas_list_from_dataset):
    """Load pre-computed metrics dan validasi apakah cocok dengan dataset yang di-upload"""
    try:
        df_eval = pd.read_csv('hasil_evaluasi_lstm_100epochs.csv')
        csv_komoditas = set(df_eval['Komoditas'].tolist())
        dataset_komoditas = set(komoditas_list_from_dataset)
        
        if csv_komoditas != dataset_komoditas:
            st.markdown("""
            <div class="alert-modern alert-warning">
                <strong style="font-size: 1.1rem;">Dataset Berbeda Terdeteksi</strong><br><br>
                Komoditas di file evaluasi: <strong>{}</strong> items<br>
                Komoditas di dataset upload: <strong>{}</strong> items<br><br>
                Sistem akan menghitung metrik evaluasi secara <strong>real-time</strong> berdasarkan dataset yang baru diupload.
            </div>
            """.format(len(csv_komoditas), len(dataset_komoditas)), unsafe_allow_html=True)
            return None, "different_dataset"
        
        st.markdown("""
        <div class="alert-modern alert-success">
            <strong style="font-size: 1.1rem;">Metrik Pre-computed Berhasil Dimuat</strong><br>
            Menggunakan hasil evaluasi dari training 100 epochs optimal
        </div>
        """, unsafe_allow_html=True)
        return df_eval, "same_dataset"
        
    except FileNotFoundError:
        st.markdown("""
        <div class="alert-modern alert-info">
            <strong style="font-size: 1.1rem;">File Evaluasi Tidak Ditemukan</strong><br>
            Sistem akan menghitung metrik secara real-time dari dataset yang diupload
        </div>
        """, unsafe_allow_html=True)
        return None, "file_not_found"
    except Exception as e:
        st.markdown(f"""
        <div class="alert-modern alert-warning">
            <strong style="font-size: 1.1rem;">Error Loading Evaluation File</strong><br>
            {str(e)}<br><br>
            Menghitung metrik secara real-time...
        </div>
        """, unsafe_allow_html=True)
        return None, "error"

def calculate_metrics_realtime(model, data_normalized, scalers, komoditas_list, TIME_STEPS=20):
    """Hitung metrik evaluasi secara real-time dari dataset yang di-upload"""
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
# SIDEBAR - DARK THEME
# ===========================================================================================

with st.sidebar:
    st.markdown("### INFORMASI MODEL")
    st.markdown("---")
    
    st.markdown("**Arsitektur Model**")
    ui.badges(badge_list=[("LSTM", "default"), ("Bidirectional", "secondary"), ("Dense", "outline")], key="arch_badges")
    st.markdown("""
    - Bidirectional LSTM (128 units)
    - LSTM (64 units)
    - Dense Layers (64, 32)
    - Regularisasi: L2 + Dropout
    """)
    
    st.markdown("---")
    st.markdown("**Hyperparameter**")
    
    col1, col2 = st.columns(2)
    with col1:
        ui.metric_card(title="Epochs", content="100", key="metric_epochs")
    with col2:
        ui.metric_card(title="Batch Size", content="32", key="metric_batch")
    
    st.markdown("""
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
    ui.badges(badge_list=[("Target MAPE < 10%", "destructive"), ("Early Stopping", "default")], key="performance_badges")

# ===========================================================================================
# MAIN CONTENT - HERO SECTION
# ===========================================================================================

st.markdown("""
<div class="hero-section">
    <p class="title-text">Prediksi Harga Komoditas Pangan</p>
    <p class="subtitle-text">Sistem Prediksi Harga Menggunakan LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)

# ===========================================================================================
# UPLOAD DATASET
# ===========================================================================================

st.markdown('<div class="section-header">Upload Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih file dataset Excel (.xlsx)",
    type=['xlsx'],
    help="Format: Kolom 1 = No, Kolom 2 = Komoditas, Kolom 3+ = Data harga dengan header tanggal"
)

if uploaded_file is not None:
    try:
        # Load dataset
        df_raw = pd.read_excel(uploaded_file)
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        st.markdown(f"""
        <div class="alert-modern alert-success">
            <strong style="font-size: 1.1rem;">Dataset Berhasil Dimuat</strong><br><br>
            Total <strong>{len(komoditas_list)}</strong> komoditas dengan <strong>{df_raw.shape[1] - 2}</strong> data points
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ui.metric_card(title="Total Komoditas", content=str(len(komoditas_list)), key="metric_total_komoditas")
        with col2:
            ui.metric_card(title="Data Points", content=str(df_raw.shape[1] - 2), key="metric_data_points")
        with col3:
            ui.metric_card(title="Rentang Waktu", content=f"{df_raw.shape[1] - 2} minggu", key="metric_rentang")
        
        st.markdown("---")
        
        # ===========================================================================================
        # FORM PREDIKSI
        # ===========================================================================================
        
        st.markdown('<div class="section-header">Prediksi Harga</div>', unsafe_allow_html=True)
        
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            selected_commodity = st.selectbox(
                "Pilih Komoditas",
                options=komoditas_list,
                key="select_commodity"
            )
        
        with col_form2:
            selected_year = st.selectbox(
                "Pilih Tahun",
                options=[2025, 2026],
                key="select_year"
            )
        
        with col_form3:
            selected_month = st.selectbox(
                "Pilih Bulan",
                options=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                key="select_month"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        predict_btn = ui.button(text="Prediksi Harga Sekarang", key="predict_button", variant="default")
        
        if predict_btn:
            with st.spinner("Memproses prediksi..."):
                try:
                    # ===========================================================================================
                    # PREPROCESSING
                    # ===========================================================================================
                    
                    df_data = df_raw.iloc[:, 2:]
                    df_transposed = df_data.T
                    df_transposed.columns = komoditas_list
                    df_transposed.reset_index(inplace=True)
                    df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
                    
                    df_transposed['Tanggal'] = pd.to_datetime(df_transposed['Tanggal'], format='%d/ %m/ %Y', errors='coerce')
                    df_transposed = df_transposed.dropna(subset=['Tanggal'])
                    df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
                    
                    for kolom in komoditas_list:
                        if df_transposed[kolom].dtype == 'object':
                            df_transposed[kolom] = df_transposed[kolom].str.replace(',', '').str.replace('"', '')
                        df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
                    
                    df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(method='linear', limit_direction='both')
                    df_transposed[komoditas_list] = df_transposed[komoditas_list].fillna(method='bfill').fillna(method='ffill')
                    
                    scalers = {}
                    data_normalized = np.zeros((len(df_transposed), len(komoditas_list)))
                    
                    for i, kolom in enumerate(komoditas_list):
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        data_normalized[:, i] = scaler.fit_transform(df_transposed[[kolom]].values).flatten()
                        scalers[kolom] = scaler
                    
                    # ===========================================================================================
                    # LOAD MODEL DAN PREDIKSI
                    # ===========================================================================================
                    
                    model = load_model('best_lstm_model.h5', compile=False)
                    
                    month_dict = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
                                 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
                    
                    target_month = month_dict[selected_month]
                    last_date = df_transposed['Tanggal'].iloc[-1]
                    target_date = datetime(selected_year, target_month, 15)
                    weeks_to_predict = max(1, int((target_date - last_date).days / 7))
                    
                    TIME_STEPS = 20
                    predictions = []
                    
                    current_sequence = data_normalized[-TIME_STEPS:].reshape(1, TIME_STEPS, len(komoditas_list))
                    
                    for _ in range(weeks_to_predict):
                        pred = model.predict(current_sequence, verbose=0)
                        predictions.append(pred[0])
                        
                        new_pred = pred.reshape(1, 1, len(komoditas_list))
                        current_sequence = np.append(current_sequence[:, 1:, :], new_pred, axis=1)
                    
                    commodity_idx = komoditas_list.index(selected_commodity)
                    predicted_price_norm = predictions[-1][commodity_idx]
                    predicted_price = scalers[selected_commodity].inverse_transform([[predicted_price_norm]])[0, 0]
                    
                    # ===========================================================================================
                    # VALIDASI DAN LOAD/HITUNG METRIK
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
                    st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)
                    
                    # Info sumber metrik
                    if metric_source == "pre-computed (100 epochs optimal)":
                        st.markdown("""
                        <div class="alert-modern alert-info">
                            <strong style="font-size: 1.1rem;">Sumber Metrik Evaluasi</strong><br>
                            Menggunakan hasil pre-computed dari training 100 epochs (dataset original)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert-modern alert-warning">
                            <strong style="font-size: 1.1rem;">Sumber Metrik Evaluasi</strong><br>
                            Dihitung secara real-time dari dataset yang baru diupload
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                    
                    with col_result1:
                        ui.metric_card(
                            title="Harga Prediksi",
                            content=f"Rp {predicted_price:,.0f}",
                            description=f"{selected_month} {selected_year}",
                            key="metric_prediction"
                        )
                    
                    with col_result2:
                        ui.metric_card(
                            title="RMSE",
                            content=f"Rp {rmse:,.0f}",
                            description="Root Mean Squared Error",
                            key="metric_rmse"
                        )
                    
                    with col_result3:
                        ui.metric_card(
                            title="MAE",
                            content=f"Rp {mae:,.0f}",
                            description="Mean Absolute Error",
                            key="metric_mae"
                        )
                    
                    with col_result4:
                        mape_status = "Excellent" if mape < 5 else "Good" if mape < 10 else "Fair" if mape < 20 else "Poor"
                        ui.metric_card(
                            title="MAPE",
                            content=f"{mape:.2f}%",
                            description=mape_status,
                            key="metric_mape"
                        )
                    
                    # ===========================================================================================
                    # VISUALISASI
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">Visualisasi Prediksi</div>', unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["Grafik Prediksi Harga", "Metrik Evaluasi Detail", "Metrik Evaluasi Keseluruhan"])
                    
                    with tab1:
                        historical_dates = df_transposed['Tanggal'].tolist()
                        historical_prices = df_transposed[selected_commodity].tolist()
                        
                        future_dates = pd.date_range(start=last_date, periods=weeks_to_predict + 1, freq='W')[1:]
                        future_prices = [scalers[selected_commodity].inverse_transform([[p[commodity_idx]]])[0, 0] for p in predictions]
                        
                        fig1 = go.Figure()
                        
                        fig1.add_trace(go.Scatter(
                            x=historical_dates, y=historical_prices,
                            mode='lines+markers', name='Data Historis',
                            line=dict(color='#667eea', width=4), 
                            marker=dict(size=8, line=dict(width=2, color='white'))
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=future_dates, y=future_prices,
                            mode='lines+markers', name='Prediksi',
                            line=dict(color='#f43f5e', width=4, dash='dash'), 
                            marker=dict(size=10, symbol='diamond', line=dict(width=2, color='white'))
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date], y=[predicted_price],
                            mode='markers', name=f'Target ({selected_month} {selected_year})',
                            marker=dict(size=22, color='#10b981', symbol='star', line=dict(width=3, color='white'))
                        ))
                        
                        fig1.update_layout(
                            title=dict(text=f'Prediksi Harga {selected_commodity}',
                                     font=dict(size=24, color='#1e293b', family='Inter', weight=800)),
                            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                            hovermode='x unified', template='plotly_white', height=650,
                            showlegend=True, 
                            legend=dict(
                                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(255,255,255,0.95)", 
                                bordercolor="#e2e8f0", 
                                borderwidth=2,
                                font=dict(size=12, weight=600)
                            ),
                            margin=dict(t=100, b=80, l=80, r=80),
                            plot_bgcolor='rgba(248, 250, 252, 0.5)'
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="alert-modern alert-info">
                            <strong style="font-size: 1.1rem;">Informasi Prediksi</strong><br><br>
                            <strong>Komoditas:</strong> {selected_commodity}<br>
                            <strong>Periode Target:</strong> {selected_month} {selected_year}<br>
                            <strong>Minggu Prediksi:</strong> {weeks_to_predict} minggu<br>
                            <strong>Tanggal Data Terakhir:</strong> {last_date.strftime('%d %B %Y')}<br>
                            <strong>Harga Prediksi:</strong> Rp {predicted_price:,.0f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab2:
                        st.markdown(f"<h4 style='font-weight: 700; color: #1e293b;'>Evaluasi Metrik - {selected_commodity}</h4>", unsafe_allow_html=True)
                        st.markdown("")
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'], 
                                y=[rmse, mae],
                                marker=dict(
                                    color=['#667eea', '#f43f5e'],
                                    line=dict(color='white', width=3)
                                ),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside',
                                textfont=dict(size=18, color='#1e293b', family='Inter', weight=800)
                            ))
                            fig3.update_layout(
                                title=dict(text='RMSE & MAE', font=dict(size=20, weight=800)),
                                yaxis_title='Nilai (Rp)', 
                                template='plotly_white', 
                                height=550,
                                margin=dict(t=100, b=80, l=80, r=80),
                                showlegend=False,
                                plot_bgcolor='rgba(248, 250, 252, 0.5)'
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col_chart2:
                            mape_color = '#10b981' if mape < 5 else '#f59e0b' if mape < 10 else '#f97316' if mape < 20 else '#ef4444'
                            
                            fig4 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=mape,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "MAPE (%)", 'font': {'size': 24, 'family': 'Inter', 'color': '#1e293b', 'weight': 800}},
                                number={'font': {'size': 52, 'family': 'Inter', 'color': mape_color, 'weight': 900}, 'suffix': '%'},
                                gauge={
                                    'axis': {'range': [0, 30], 'tickwidth': 3, 'tickcolor': "#cbd5e1"},
                                    'bar': {'color': mape_color, 'thickness': 0.85},
                                    'steps': [
                                        {'range': [0, 5], 'color': '#d1fae5'},
                                        {'range': [5, 10], 'color': '#fef3c7'},
                                        {'range': [10, 20], 'color': '#fed7aa'},
                                        {'range': [20, 30], 'color': '#fecaca'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "#1e293b", 'width': 6},
                                        'thickness': 0.85,
                                        'value': mape
                                    }
                                }
                            ))
                            fig4.update_layout(
                                height=550,
                                margin=dict(t=100, b=80, l=80, r=80)
                            )
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="alert-modern alert-info">
                            <strong style="font-size: 1.1rem;">Status Performa - {selected_commodity}</strong><br><br>
                            <strong>RMSE:</strong> Rp {rmse:,.2f}<br>
                            <strong>MAE:</strong> Rp {mae:,.2f}<br>
                            <strong>MAPE:</strong> {mape:.2f}%<br><br>
                            <strong>Kategori Evaluasi:</strong> {mape_status}<br><br>
                            <strong>Interpretasi:</strong> MAPE {mape:.2f}% berarti prediksi rata-rata meleset {mape:.2f}% dari nilai aktual
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown(f"<h4 style='font-weight: 700; color: #1e293b;'>Evaluasi Performa Model untuk Semua Komoditas</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #64748b; font-style: italic;'>Sumber metrik: {metric_source}</p>", unsafe_allow_html=True)
                        st.markdown("")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_rmse = go.Figure()
                                fig_rmse.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['RMSE'],
                                    marker=dict(
                                        color='#667eea',
                                        line=dict(color='white', width=2)
                                    ),
                                    text=df_metrics['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=9, family='Inter', weight=600)
                                ))
                                fig_rmse.update_layout(
                                    title=dict(text='Root Mean Squared Error (RMSE)', font=dict(size=18, weight=800)),
                                    xaxis_title='Komoditas',
                                    yaxis_title='RMSE (Rp)',
                                    height=600,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=100, b=170, l=80, r=40),
                                    plot_bgcolor='rgba(248, 250, 252, 0.5)'
                                )
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            with col2:
                                fig_mae = go.Figure()
                                fig_mae.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['MAE'],
                                    marker=dict(
                                        color='#f43f5e',
                                        line=dict(color='white', width=2)
                                    ),
                                    text=df_metrics['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=9, family='Inter', weight=600)
                                ))
                                fig_mae.update_layout(
                                    title=dict(text='Mean Absolute Error (MAE)', font=dict(size=18, weight=800)),
                                    xaxis_title='Komoditas',
                                    yaxis_title='MAE (Rp)',
                                    height=600,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=100, b=170, l=80, r=40),
                                    plot_bgcolor='rgba(248, 250, 252, 0.5)'
                                )
                                st.plotly_chart(fig_mae, use_container_width=True)
                            
                            colors = []
                            for val in df_metrics['MAPE']:
                                if val < 5:
                                    colors.append('#10b981')
                                elif val < 10:
                                    colors.append('#f59e0b')
                                elif val < 20:
                                    colors.append('#f97316')
                                else:
                                    colors.append('#ef4444')
                            
                            fig_mape = go.Figure()
                            fig_mape.add_trace(go.Bar(
                                x=df_metrics['Komoditas'],
                                y=df_metrics['MAPE'],
                                marker=dict(
                                    color=colors,
                                    line=dict(color='white', width=2)
                                ),
                                text=df_metrics['MAPE'].apply(lambda x: f'{x:.2f}%'),
                                textposition='outside',
                                textfont=dict(size=11, family='Inter', weight=700)
                            ))
                            fig_mape.update_layout(
                                title=dict(text='Mean Absolute Percentage Error (MAPE)', font=dict(size=20, weight=800)),
                                xaxis_title='Komoditas',
                                yaxis_title='MAPE (%)',
                                height=600,
                                template='plotly_white',
                                xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                margin=dict(t=100, b=170, l=80, r=80),
                                plot_bgcolor='rgba(248, 250, 252, 0.5)'
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                            
                            excellent_count = len(df_metrics[df_metrics['MAPE'] < 5])
                            good_count = len(df_metrics[(df_metrics['MAPE'] >= 5) & (df_metrics['MAPE'] < 10)])
                            fair_count = len(df_metrics[(df_metrics['MAPE'] >= 10) & (df_metrics['MAPE'] < 20)])
                            poor_count = len(df_metrics[df_metrics['MAPE'] >= 20])
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                            
                            with col_perf1:
                                ui.metric_card(
                                    title="Excellent",
                                    content=str(excellent_count),
                                    description=f"{excellent_count/len(df_metrics)*100:.1f}%",
                                    key="perf_excellent"
                                )
                            
                            with col_perf2:
                                ui.metric_card(
                                    title="Good",
                                    content=str(good_count),
                                    description=f"{good_count/len(df_metrics)*100:.1f}%",
                                    key="perf_good"
                                )
                            
                            with col_perf3:
                                ui.metric_card(
                                    title="Fair",
                                    content=str(fair_count),
                                    description=f"{fair_count/len(df_metrics)*100:.1f}%",
                                    key="perf_fair"
                                )
                            
                            with col_perf4:
                                ui.metric_card(
                                    title="Poor",
                                    content=str(poor_count),
                                    description=f"{poor_count/len(df_metrics)*100:.1f}%",
                                    key="perf_poor"
                                )
                            
                            st.markdown("---")
                            st.markdown("<h4 style='font-weight: 700; color: #1e293b;'>Tabel Detail Metrik Evaluasi</h4>", unsafe_allow_html=True)
                            st.markdown("")
                            
                            df_display = df_metrics.copy()
                            df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAE'] = df_display['MAE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(df_display, use_container_width=True, height=500)
                        
                        else:
                            st.warning("Tidak cukup data test untuk evaluasi")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error saat membaca dataset: {str(e)}")

else:
    st.markdown("""
    <div class="alert-modern alert-info">
        <strong style="font-size: 1.1rem;">Upload Dataset untuk Memulai</strong><br><br>
        <strong>Format Dataset yang Diharapkan:</strong><br>
        <strong>Kolom 1:</strong> No (1, 2, 3, ...)<br>
        <strong>Kolom 2:</strong> Nama Komoditas<br>
        <strong>Kolom 3+:</strong> Data harga dengan header tanggal<br>
        <strong>File format:</strong> Excel (.xlsx)
    </div>
    """, unsafe_allow_html=True)
    
    ui.badges(badge_list=[("Excel", "default"), (".xlsx", "secondary"), ("Required", "destructive")], key="format_badges")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 3rem 1rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 20px; margin-top: 3rem;'>
    <p style='font-size: 1.1rem; font-weight: 600; letter-spacing: 0.03em; margin-bottom: 0.75rem;'>Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM Neural Network</p>
    <p style='font-size: 0.9rem; color: #94a3b8; font-weight: 500;'>Built with Streamlit & Shadcn UI • Tailwind CSS Design System</p>
</div>
""", unsafe_allow_html=True)
