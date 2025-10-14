import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime
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

# Custom CSS untuk styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 3rem 4rem;
        max-width: 1600px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        padding: 2rem 1.5rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }
    
    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    
    h1 {
        color: #1e40af;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    h3 {
        color: #1e40af;
        font-weight: 700;
        font-size: 1.5rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    h4 {
        color: #1e40af;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 6px;
        margin-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
    }
    
    .stAlert {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border: none;
        margin: 1.5rem 0;
    }
    
    [data-testid="stFileUploader"] {
        padding: 2.5rem 2rem;
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        background-color: #f8fafc;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem 1rem;
        background-color: white;
    }
    
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }
    
    thead tr th {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    tbody tr:hover {
        background-color: #f8fafc !important;
    }
    
    .js-plotly-plot {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    div[data-testid="column"] {
        padding: 0.5rem;
    }
    
    hr {
        margin: 2rem 0;
        border-color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================================================================================
# FUNGSI HELPER
# ===========================================================================================

def load_and_validate_metrics(komoditas_list_from_dataset):
    """Load pre-computed metrics dan validasi dataset"""
    try:
        df_eval = pd.read_csv('hasil_evaluasi_lstm_100epochs.csv')
        csv_komoditas = set(df_eval['Komoditas'].tolist())
        dataset_komoditas = set(komoditas_list_from_dataset)
        
        if csv_komoditas != dataset_komoditas:
            st.warning("Dataset berbeda terdeteksi. Menghitung metrik secara real-time...")
            return None, "different_dataset"
        
        st.success("Menggunakan metrik pre-computed dari training 100 epochs")
        return df_eval, "same_dataset"
        
    except FileNotFoundError:
        st.info("File evaluasi tidak ditemukan. Menghitung metrik secara real-time...")
        return None, "file_not_found"
    except Exception as e:
        st.warning(f"Error: {str(e)}. Menghitung metrik secara real-time...")
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
    st.markdown("**Target Performa**")
    st.markdown("""
    - MAPE Target: < 10%
    - Early Stopping: Patience 20
    - ReduceLR: Patience 8
    """)

# ===========================================================================================
# MAIN CONTENT
# ===========================================================================================

st.title("Prediksi Harga Komoditas Pangan")
st.markdown("Sistem Prediksi Harga Menggunakan LSTM Neural Network")
st.markdown("---")

# ===========================================================================================
# UPLOAD DATASET
# ===========================================================================================

st.markdown("### Upload Dataset")
uploaded_file = st.file_uploader(
    "Pilih file dataset Excel (.xlsx)",
    type=['xlsx'],
    help="Format: Kolom 1 = No, Kolom 2 = Komoditas, Kolom 3+ = Data harga"
)

if uploaded_file is not None:
    try:
        # Load dataset
        df_raw = pd.read_excel(uploaded_file)
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        st.success(f"Dataset berhasil dimuat: {len(komoditas_list)} komoditas, {df_raw.shape[1] - 2} data points")
        
        st.markdown("")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komoditas", len(komoditas_list))
        col2.metric("Data Points", df_raw.shape[1] - 2)
        col3.metric("Rentang Waktu", f"{df_raw.shape[1] - 2} minggu")
        
        st.markdown("---")
        
        # ===========================================================================================
        # FORM PREDIKSI
        # ===========================================================================================
        
        st.markdown("### Prediksi Harga")
        st.markdown("")
        
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            selected_commodity = st.selectbox("Pilih Komoditas", komoditas_list)
        
        with col_form2:
            selected_year = st.selectbox("Pilih Tahun", [2025, 2026])
        
        with col_form3:
            selected_month = st.selectbox(
                "Pilih Bulan",
                ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
            )
        
        # Button prediksi
        if st.button("Prediksi Harga Sekarang", type="primary"):
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
                    else:
                        all_metrics = calculate_metrics_realtime(model, data_normalized, scalers, komoditas_list, TIME_STEPS)
                    
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
                    st.markdown("")
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                    
                    with col_result1:
                        st.metric("Harga Prediksi", f"Rp {predicted_price:,.0f}", f"{selected_month} {selected_year}")
                    
                    with col_result2:
                        st.metric("RMSE", f"Rp {rmse:,.0f}", "Root Mean Squared Error")
                    
                    with col_result3:
                        st.metric("MAE", f"Rp {mae:,.0f}", "Mean Absolute Error")
                    
                    with col_result4:
                        mape_status = "Excellent" if mape < 5 else "Good" if mape < 10 else "Fair" if mape < 20 else "Poor"
                        st.metric("MAPE", f"{mape:.2f}%", mape_status)
                    
                    # ===========================================================================================
                    # VISUALISASI
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown("### Visualisasi Prediksi")
                    
                    tab1, tab2, tab3 = st.tabs(["Grafik Prediksi Harga", "Metrik Evaluasi Detail", "Metrik Evaluasi Keseluruhan"])
                    
                    with tab1:
                        st.markdown("")
                        historical_dates = df_transposed['Tanggal'].tolist()
                        historical_prices = df_transposed[selected_commodity].tolist()
                        
                        future_dates = pd.date_range(start=last_date, periods=weeks_to_predict + 1, freq='W')[1:]
                        future_prices = [scalers[selected_commodity].inverse_transform([[p[commodity_idx]]])[0, 0] for p in predictions]
                        
                        fig1 = go.Figure()
                        
                        fig1.add_trace(go.Scatter(
                            x=historical_dates, y=historical_prices,
                            mode='lines+markers', name='Data Historis',
                            line=dict(color='#3b82f6', width=3), 
                            marker=dict(size=6, color='#3b82f6')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=future_dates, y=future_prices,
                            mode='lines+markers', name='Prediksi',
                            line=dict(color='#ef4444', width=3, dash='dash'), 
                            marker=dict(size=8, symbol='diamond', color='#ef4444')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date], y=[predicted_price],
                            mode='markers', name=f'Target ({selected_month} {selected_year})',
                            marker=dict(size=16, color='#10b981', symbol='star')
                        ))
                        
                        fig1.update_layout(
                            title=dict(text=f'Prediksi Harga {selected_commodity}', font=dict(size=18, color='#1e40af', family='Inter')),
                            xaxis_title='Tanggal', 
                            yaxis_title='Harga (Rp)',
                            hovermode='x unified', 
                            template='plotly_white', 
                            height=550,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(t=80, b=60, l=60, r=60)
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        st.markdown("")
                        st.info(f"""
                        **Informasi Prediksi:**
                        - Komoditas: {selected_commodity}
                        - Periode Target: {selected_month} {selected_year}
                        - Minggu Prediksi: {weeks_to_predict} minggu
                        - Tanggal Data Terakhir: {last_date.strftime('%d %B %Y')}
                        - Harga Prediksi: Rp {predicted_price:,.0f}
                        """)
                    
                    with tab2:
                        st.markdown("")
                        st.markdown(f"#### Evaluasi Metrik - {selected_commodity}")
                        st.markdown("")
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'], 
                                y=[rmse, mae],
                                marker=dict(color=['#3b82f6', '#ef4444']),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside',
                                textfont=dict(size=14, family='Inter')
                            ))
                            fig3.update_layout(
                                title=dict(text='RMSE & MAE', font=dict(size=16)),
                                yaxis_title='Nilai (Rp)', 
                                template='plotly_white', 
                                height=480,
                                showlegend=False,
                                margin=dict(t=60, b=60, l=60, r=60)
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col_chart2:
                            mape_color = '#10b981' if mape < 5 else '#f59e0b' if mape < 10 else '#f97316' if mape < 20 else '#ef4444'
                            
                            fig4 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=mape,
                                title={'text': "MAPE (%)", 'font': {'size': 18, 'family': 'Inter'}},
                                number={'font': {'size': 40}},
                                gauge={
                                    'axis': {'range': [0, 30], 'tickwidth': 2},
                                    'bar': {'color': mape_color},
                                    'steps': [
                                        {'range': [0, 5], 'color': '#d1fae5'},
                                        {'range': [5, 10], 'color': '#fef3c7'},
                                        {'range': [10, 20], 'color': '#fed7aa'},
                                        {'range': [20, 30], 'color': '#fecaca'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "#1e293b", 'width': 4},
                                        'thickness': 0.75,
                                        'value': mape
                                    }
                                }
                            ))
                            fig4.update_layout(height=480, margin=dict(t=60, b=60, l=60, r=60))
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        st.markdown("")
                        st.info(f"""
                        **Status Performa - {selected_commodity}:**
                        - RMSE: Rp {rmse:,.2f}
                        - MAE: Rp {mae:,.2f}
                        - MAPE: {mape:.2f}%
                        - Kategori: {mape_status}
                        - Interpretasi: MAPE {mape:.2f}% berarti prediksi rata-rata meleset {mape:.2f}% dari nilai aktual
                        """)
                    
                    with tab3:
                        st.markdown("")
                        st.markdown("#### Evaluasi Performa Model untuk Semua Komoditas")
                        st.markdown("")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_rmse = go.Figure()
                                fig_rmse.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['RMSE'],
                                    marker=dict(color='#3b82f6'),
                                    text=df_metrics['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=9)
                                ))
                                fig_rmse.update_layout(
                                    title=dict(text='Root Mean Squared Error (RMSE)', font=dict(size=15)),
                                    xaxis_title='Komoditas',
                                    yaxis_title='RMSE (Rp)',
                                    height=550,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 8}},
                                    margin=dict(t=70, b=140, l=70, r=30)
                                )
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            with col2:
                                fig_mae = go.Figure()
                                fig_mae.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['MAE'],
                                    marker=dict(color='#ef4444'),
                                    text=df_metrics['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=9)
                                ))
                                fig_mae.update_layout(
                                    title=dict(text='Mean Absolute Error (MAE)', font=dict(size=15)),
                                    xaxis_title='Komoditas',
                                    yaxis_title='MAE (Rp)',
                                    height=550,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 8}},
                                    margin=dict(t=70, b=140, l=70, r=30)
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
                                marker=dict(color=colors),
                                text=df_metrics['MAPE'].apply(lambda x: f'{x:.2f}%'),
                                textposition='outside',
                                textfont=dict(size=10)
                            ))
                            fig_mape.update_layout(
                                title=dict(text='Mean Absolute Percentage Error (MAPE)', font=dict(size=16)),
                                xaxis_title='Komoditas',
                                yaxis_title='MAPE (%)',
                                height=550,
                                template='plotly_white',
                                xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                margin=dict(t=70, b=140, l=70, r=70)
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                            
                            excellent_count = len(df_metrics[df_metrics['MAPE'] < 5])
                            good_count = len(df_metrics[(df_metrics['MAPE'] >= 5) & (df_metrics['MAPE'] < 10)])
                            fair_count = len(df_metrics[(df_metrics['MAPE'] >= 10) & (df_metrics['MAPE'] < 20)])
                            poor_count = len(df_metrics[df_metrics['MAPE'] >= 20])
                            
                            st.markdown("")
                            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                            col_perf1.metric("Excellent", excellent_count, f"{excellent_count/len(df_metrics)*100:.1f}%")
                            col_perf2.metric("Good", good_count, f"{good_count/len(df_metrics)*100:.1f}%")
                            col_perf3.metric("Fair", fair_count, f"{fair_count/len(df_metrics)*100:.1f}%")
                            col_perf4.metric("Poor", poor_count, f"{poor_count/len(df_metrics)*100:.1f}%")
                            
                            st.markdown("---")
                            st.markdown("#### Tabel Detail Metrik Evaluasi")
                            st.markdown("")
                            
                            df_display = df_metrics.copy()
                            df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAE'] = df_display['MAE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(df_display, use_container_width=True, height=450)
                        else:
                            st.warning("Tidak cukup data test untuk evaluasi")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error saat membaca dataset: {str(e)}")

else:
    st.info("""
    **Silakan upload dataset untuk memulai**
    
    Format Dataset yang Diharapkan:
    - Kolom 1: No (1, 2, 3, ...)
    - Kolom 2: Nama Komoditas
    - Kolom 3+: Data harga dengan header tanggal
    - File format: Excel (.xlsx)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p style='font-size: 0.95rem; margin: 0;'>Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)
