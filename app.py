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

# Custom CSS - Enhanced Modern Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1400px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem 1rem;
        border-right: 1px solid #e2e8f0;
    }
    
    .title-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle-text {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Streamlit native components enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
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
            st.warning(f"""
            ⚠️ **Dataset berbeda terdeteksi!**
            
            - Komoditas di file evaluasi: {len(csv_komoditas)} items
            - Komoditas di dataset upload: {len(dataset_komoditas)} items
            
            Sistem akan menghitung metrik evaluasi secara **real-time** berdasarkan dataset yang baru diupload.
            """)
            return None, "different_dataset"
        
        st.success("✅ Menggunakan metrik evaluasi pre-computed dari hasil training (100 epochs optimal)")
        return df_eval, "same_dataset"
        
    except FileNotFoundError:
        st.info("ℹ️ File evaluasi pre-computed tidak ditemukan. Menghitung metrik secara real-time...")
        return None, "file_not_found"
    except Exception as e:
        st.warning(f"⚠️ Error loading evaluation file: {str(e)}. Menghitung metrik secara real-time...")
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
# SIDEBAR - SHADCN UI ENHANCED
# ===========================================================================================

with st.sidebar:
    st.markdown("### 📊 Informasi Model")
    st.markdown("---")
    
    with ui.card(key="model_arch_card"):
        st.markdown("**🏗️ Arsitektur Model**")
        ui.badges(badge_list=[("LSTM", "default"), ("Bidirectional", "secondary"), ("Dense", "outline")], key="arch_badges")
        st.markdown("""
        - Bidirectional LSTM (128 units)
        - LSTM (64 units)
        - Dense Layers (64, 32)
        - Regularisasi: L2 + Dropout
        """)
    
    st.markdown("---")
    
    with ui.card(key="hyperparams_card"):
        st.markdown("**⚙️ Hyperparameter**")
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
    
    with ui.card(key="preprocess_card"):
        st.markdown("**🔧 Preprocessing**")
        st.markdown("""
        - Time Steps: 20
        - Normalisasi: MinMaxScaler
        - Train/Test Split: 90/10
        - Interpolasi: Linear
        """)
    
    st.markdown("---")
    
    ui.badges(badge_list=[("Target MAPE < 10%", "destructive"), ("Early Stopping", "default")], key="performance_badges")

# ===========================================================================================
# MAIN CONTENT
# ===========================================================================================

st.markdown('<p class="title-text">📈 Prediksi Harga Komoditas Pangan</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Sistem Prediksi Harga Menggunakan LSTM Neural Network</p>', unsafe_allow_html=True)

# ===========================================================================================
# UPLOAD DATASET - SHADCN UI
# ===========================================================================================

st.markdown('<div class="section-header">📁 Upload Dataset</div>', unsafe_allow_html=True)
st.markdown("")

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
        
        ui.alert(key="dataset_success", title="Dataset Berhasil Dimuat!", description=f"Total {len(komoditas_list)} komoditas dengan {df_raw.shape[1] - 2} data points", variant="default")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ui.metric_card(title="Total Komoditas", content=str(len(komoditas_list)), key="metric_total_komoditas")
        with col2:
            ui.metric_card(title="Data Points", content=str(df_raw.shape[1] - 2), key="metric_data_points")
        with col3:
            ui.metric_card(title="Rentang Waktu", content=f"{df_raw.shape[1] - 2} minggu", key="metric_rentang")
        
        st.markdown("---")
        
        # ===========================================================================================
        # FORM PREDIKSI - SHADCN UI
        # ===========================================================================================
        
        st.markdown('<div class="section-header">🎯 Prediksi Harga</div>', unsafe_allow_html=True)
        st.markdown("")
        
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            selected_commodity = ui.select(
                options=komoditas_list,
                key="select_commodity",
                label="Pilih Komoditas"
            )
        
        with col_form2:
            selected_year = ui.select(
                options=[2025, 2026],
                key="select_year",
                label="Pilih Tahun"
            )
        
        with col_form3:
            selected_month = ui.select(
                options=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                key="select_month",
                label="Pilih Bulan"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        predict_btn = ui.button(text="🚀 Prediksi Harga Sekarang", key="predict_button", variant="default", size="lg")
        
        if predict_btn:
            with st.spinner("🔄 Memproses prediksi..."):
                try:
                    # ===========================================================================================
                    # PREPROCESSING - SEMUA KOMODITAS
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
                    # TAMPILKAN HASIL - SHADCN UI
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">✨ Hasil Prediksi</div>', unsafe_allow_html=True)
                    st.markdown("")
                    
                    # Info sumber metrik
                    if metric_source == "pre-computed (100 epochs optimal)":
                        st.info("📊 **Metrik Evaluasi:** Menggunakan hasil pre-computed dari training 100 epochs (dataset original)")
                    else:
                        st.warning("📊 **Metrik Evaluasi:** Dihitung secara real-time dari dataset yang baru diupload")
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                    
                    with col_result1:
                        ui.metric_card(
                            title="💰 Harga Prediksi",
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
                        mape_description = "🟢 Excellent" if mape < 5 else "🟡 Good" if mape < 10 else "🟠 Fair" if mape < 20 else "🔴 Poor"
                        ui.metric_card(
                            title="MAPE",
                            content=f"{mape:.2f}%",
                            description=mape_description,
                            key="metric_mape"
                        )
                    
                    # ===========================================================================================
                    # VISUALISASI
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">📊 Visualisasi Prediksi</div>', unsafe_allow_html=True)
                    st.markdown("")
                    
                    tab1, tab2, tab3 = st.tabs(["📈 Grafik Prediksi Harga", "📊 Metrik Evaluasi Keseluruhan", "🎯 Metrik Evaluasi Detail"])
                    
                    with tab1:
                        historical_dates = df_transposed['Tanggal'].tolist()
                        historical_prices = df_transposed[selected_commodity].tolist()
                        
                        future_dates = pd.date_range(start=last_date, periods=weeks_to_predict + 1, freq='W')[1:]
                        future_prices = [scalers[selected_commodity].inverse_transform([[p[commodity_idx]]])[0, 0] for p in predictions]
                        
                        fig1 = go.Figure()
                        
                        fig1.add_trace(go.Scatter(
                            x=historical_dates, y=historical_prices,
                            mode='lines+markers', name='Data Historis',
                            line=dict(color='#667eea', width=3), marker=dict(size=6)
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=future_dates, y=future_prices,
                            mode='lines+markers', name='Prediksi',
                            line=dict(color='#f43f5e', width=3, dash='dash'), marker=dict(size=8, symbol='diamond')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date], y=[predicted_price],
                            mode='markers', name=f'Target ({selected_month} {selected_year})',
                            marker=dict(size=18, color='#10b981', symbol='star', line=dict(width=2, color='white'))
                        ))
                        
                        fig1.update_layout(
                            title=dict(text=f'Prediksi Harga {selected_commodity}',
                                     font=dict(size=22, color='#1e293b', family='Inter')),
                            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                            hovermode='x unified', template='plotly_white', height=550,
                            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        with ui.card(key="prediction_info_card"):
                            st.markdown(f"""
                            **📋 Informasi Prediksi:**
                            - **Komoditas:** {selected_commodity}
                            - **Periode Target:** {selected_month} {selected_year}
                            - **Minggu Prediksi:** {weeks_to_predict} minggu
                            - **Tanggal Data Terakhir:** {last_date.strftime('%d %B %Y')}
                            - **Harga Prediksi:** Rp {predicted_price:,.0f}
                            """)
                    
                    with tab2:
                        st.markdown(f"#### Evaluasi Performa Model untuk Semua Komoditas")
                        st.markdown(f"*Sumber metrik: {metric_source}*")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_rmse = go.Figure()
                                fig_rmse.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['RMSE'],
                                    marker=dict(color='#667eea'),
                                    text=df_metrics['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=10)
                                ))
                                fig_rmse.update_layout(
                                    title='Root Mean Squared Error (RMSE)',
                                    xaxis_title='Komoditas',
                                    yaxis_title='RMSE (Rp)',
                                    height=550,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=80, b=150, l=80, r=40)
                                )
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            with col2:
                                fig_mae = go.Figure()
                                fig_mae.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['MAE'],
                                    marker=dict(color='#f43f5e'),
                                    text=df_metrics['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=10)
                                ))
                                fig_mae.update_layout(
                                    title='Mean Absolute Error (MAE)',
                                    xaxis_title='Komoditas',
                                    yaxis_title='MAE (Rp)',
                                    height=550,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=80, b=150, l=80, r=40)
                                )
                                st.plotly_chart(fig_mae, use_container_width=True)
                            
                            # MAPE Chart
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
                                textfont=dict(size=11, family='Inter')
                            ))
                            fig_mape.update_layout(
                                title='Mean Absolute Percentage Error (MAPE)',
                                xaxis_title='Komoditas',
                                yaxis_title='MAPE (%)',
                                height=550,
                                template='plotly_white',
                                xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                margin=dict(t=80, b=150, l=80, r=80)
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                            
                            # Performance Summary
                            excellent_count = len(df_metrics[df_metrics['MAPE'] < 5])
                            good_count = len(df_metrics[(df_metrics['MAPE'] >= 5) & (df_metrics['MAPE'] < 10)])
                            fair_count = len(df_metrics[(df_metrics['MAPE'] >= 10) & (df_metrics['MAPE'] < 20)])
                            poor_count = len(df_metrics[df_metrics['MAPE'] >= 20])
                            
                            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                            
                            with col_perf1:
                                ui.metric_card(
                                    title="🟢 Excellent",
                                    content=str(excellent_count),
                                    description=f"{excellent_count/len(df_metrics)*100:.1f}%",
                                    key="perf_excellent"
                                )
                            
                            with col_perf2:
                                ui.metric_card(
                                    title="🟡 Good",
                                    content=str(good_count),
                                    description=f"{good_count/len(df_metrics)*100:.1f}%",
                                    key="perf_good"
                                )
                            
                            with col_perf3:
                                ui.metric_card(
                                    title="🟠 Fair",
                                    content=str(fair_count),
                                    description=f"{fair_count/len(df_metrics)*100:.1f}%",
                                    key="perf_fair"
                                )
                            
                            with col_perf4:
                                ui.metric_card(
                                    title="🔴 Poor",
                                    content=str(poor_count),
                                    description=f"{poor_count/len(df_metrics)*100:.1f}%",
                                    key="perf_poor"
                                )
                            
                            st.markdown("---")
                            st.markdown("#### 📋 Tabel Detail Metrik Evaluasi")
                            
                            df_display = df_metrics.copy()
                            df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAE'] = df_display['MAE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(df_display, use_container_width=True, height=400)
                        
                        else:
                            st.warning("Tidak cukup data test untuk evaluasi")
                    
                    with tab3:
                        st.markdown(f"#### Evaluasi Metrik - {selected_commodity}")
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'], 
                                y=[rmse, mae],
                                marker=dict(color=['#667eea', '#f43f5e']),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside',
                                textfont=dict(size=16, color='#1e293b', family='Inter')
                            ))
                            fig3.update_layout(
                                title=dict(text='RMSE & MAE', font=dict(size=18)),
                                yaxis_title='Nilai (Rp)', 
                                template='plotly_white', 
                                height=450,
                                margin=dict(t=80, b=60, l=80, r=60),
                                showlegend=False
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col_chart2:
                            mape_color = '#10b981' if mape < 5 else '#f59e0b' if mape < 10 else '#f97316' if mape < 20 else '#ef4444'
                            
                            fig4 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=mape,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "MAPE (%)", 'font': {'size': 20}},
                                number={'font': {'size': 40}},
                                gauge={
                                    'axis': {'range': [0, 30], 'tickwidth': 1},
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
                            fig4.update_layout(
                                height=450,
                                margin=dict(t=80, b=60, l=60, r=60)
                            )
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        with ui.card(key="metric_detail_card"):
                            st.markdown(f"""
                            **📊 Status Performa - {selected_commodity}:**
                            - **RMSE:** Rp {rmse:,.2f}
                            - **MAE:** Rp {mae:,.2f}
                            - **MAPE:** {mape:.2f}%
                            
                            **Evaluasi:** {'🟢 Excellent' if mape < 5 else '🟡 Good' if mape < 10 else '🟠 Fair' if mape < 20 else '🔴 Poor'}
                            
                            **Interpretasi:** MAPE {mape:.2f}% berarti prediksi rata-rata meleset {mape:.2f}% dari nilai aktual
                            """)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Error saat membaca dataset: {str(e)}")

else:
    with ui.card(key="upload_instruction_card"):
        st.markdown("### 📤 Silakan Upload Dataset")
        st.markdown("#### Format Dataset yang Diharapkan:")
        st.markdown("""
        - **Kolom 1**: No (1, 2, 3, ...)
        - **Kolom 2**: Nama Komoditas
        - **Kolom 3+**: Data harga dengan header tanggal
        - **File format**: Excel (.xlsx)
        """)
        
        ui.badges(badge_list=[("Excel", "default"), (".xlsx", "secondary"), ("Required", "destructive")], key="format_badges")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p style='font-size: 0.9rem;'>⚡ Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM Neural Network</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Built with Streamlit & Shadcn UI</p>
</div>
""", unsafe_allow_html=True)
