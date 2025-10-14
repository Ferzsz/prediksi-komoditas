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
    page_title="LSTM Price Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================================
# MODERN CSS STYLING - GLASSMORPHISM & GRADIENT DESIGN
# ===========================================================================================

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Block Container Styling */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Sidebar Modern Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Sidebar Text Color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.85);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Remove metric padding */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Info Box Styling */
    .modern-info {
        background: rgba(52, 211, 153, 0.15);
        backdrop-filter: blur(10px);
        border-left: 4px solid #34D399;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: white;
    }
    
    .modern-warning {
        background: rgba(251, 191, 36, 0.15);
        backdrop-filter: blur(10px);
        border-left: 4px solid #FBBF24;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: white;
    }
    
    .modern-success {
        background: rgba(34, 197, 94, 0.15);
        backdrop-filter: blur(10px);
        border-left: 4px solid #22C55E;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: white;
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.25);
        color: white;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Remove default padding/margin */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    .stVerticalBlock {
        gap: 0;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly chart background */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
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
            st.markdown(f"""
            <div class="modern-warning">
                ⚠️ <strong>Dataset Berbeda Terdeteksi</strong><br>
                Sistem akan menghitung metrik evaluasi secara real-time berdasarkan dataset yang baru diupload.
            </div>
            """, unsafe_allow_html=True)
            return None, "different_dataset"
        
        st.markdown("""
        <div class="modern-info">
            ✅ <strong>Menggunakan metrik evaluasi pre-computed dari hasil training (100 epochs optimal)</strong>
        </div>
        """, unsafe_allow_html=True)
        return df_eval, "same_dataset"
        
    except FileNotFoundError:
        st.markdown("""
        <div class="modern-info">
            ℹ️ <strong>File evaluasi tidak ditemukan. Menghitung metrik secara real-time...</strong>
        </div>
        """, unsafe_allow_html=True)
        return None, "file_not_found"
    except Exception as e:
        st.markdown(f"""
        <div class="modern-warning">
            ⚠️ <strong>Error loading file: {str(e)}. Menghitung metrik secara real-time...</strong>
        </div>
        """, unsafe_allow_html=True)
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
    st.markdown("### 🤖 Model Information")
    st.markdown("---")
    
    st.markdown("#### Architecture")
    st.markdown("""
    • Bidirectional LSTM (128 units)  
    • LSTM (64 units)  
    • Dense Layers (64, 32)  
    • L2 Regularization + Dropout
    """)
    
    st.markdown("---")
    st.markdown("#### Hyperparameters")
    st.markdown("""
    • **Epochs:** 100  
    • **Batch Size:** 32  
    • **Learning Rate:** 0.001  
    • **Optimizer:** Adam  
    • **Loss Function:** Huber Loss
    """)
    
    st.markdown("---")
    st.markdown("#### Preprocessing")
    st.markdown("""
    • **Time Steps:** 20  
    • **Normalization:** MinMaxScaler  
    • **Train/Test Split:** 90/10  
    • **Interpolation:** Linear
    """)
    
    st.markdown("---")
    st.markdown("#### Performance Target")
    st.markdown("""
    • **MAPE Target:** < 10%  
    • **Early Stopping:** Patience 20  
    • **ReduceLR:** Patience 8
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;'>
        <p style='margin: 0; font-size: 0.85rem; opacity: 0.8;'>
            © 2025 LSTM Forecasting System<br>
            Powered by TensorFlow & Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================================
# MAIN HEADER
# ===========================================================================================

st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📈 LSTM Price Forecasting System</h1>
        <p class="main-subtitle">Advanced Commodity Price Prediction using Deep Learning Neural Networks</p>
    </div>
""", unsafe_allow_html=True)

# ===========================================================================================
# UPLOAD SECTION
# ===========================================================================================

st.markdown('<h2 class="section-header">📁 Dataset Upload</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your dataset (Excel format .xlsx)",
    type=['xlsx'],
    help="Format: Column 1 = No, Column 2 = Commodity Name, Column 3+ = Price data with date headers"
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        st.markdown("""
        <div class="modern-success">
            ✅ <strong>Dataset loaded successfully!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Commodities", len(komoditas_list))
        with col2:
            st.metric("📈 Total Data Points", df_raw.shape[1] - 2)
        with col3:
            st.metric("🗓️ Time Range", f"{df_raw.shape[1] - 2} weeks")
        
        # ===========================================================================================
        # PREDICTION FORM
        # ===========================================================================================
        
        st.markdown('<h2 class="section-header">🎯 Price Prediction</h2>', unsafe_allow_html=True)
        
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            selected_commodity = st.selectbox(
                "🛒 Select Commodity",
                options=komoditas_list,
                help="Choose the commodity you want to predict"
            )
        
        with col_form2:
            selected_year = st.selectbox(
                "📅 Select Year",
                options=[2025, 2026],
                help="Choose the prediction year"
            )
        
        with col_form3:
            selected_month = st.selectbox(
                "📆 Select Month",
                options=['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'],
                help="Choose the prediction month"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Generate Prediction", use_container_width=True):
            with st.spinner("🔄 Processing prediction..."):
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
                    # MODEL PREDICTION
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
                    # METRICS CALCULATION
                    # ===========================================================================================
                    
                    df_eval_metrics, status = load_and_validate_metrics(komoditas_list)
                    
                    if df_eval_metrics is not None and status == "same_dataset":
                        all_metrics = df_eval_metrics.rename(columns={'MAPE (%)': 'MAPE'}).to_dict('records')
                        metric_source = "pre-computed (100 epochs optimal)"
                    else:
                        with st.spinner("📊 Calculating evaluation metrics..."):
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
                    # RESULTS DISPLAY
                    # ===========================================================================================
                    
                    st.markdown('<h2 class="section-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4, gap="medium")
                    
                    with col_result1:
                        st.metric("💰 Predicted Price", f"Rp {predicted_price:,.0f}")
                    with col_result2:
                        st.metric("📉 RMSE", f"Rp {rmse:,.0f}")
                    with col_result3:
                        st.metric("📊 MAE", f"Rp {mae:,.0f}")
                    with col_result4:
                        mape_emoji = "🟢" if mape < 5 else "🟡" if mape < 10 else "🟠" if mape < 20 else "🔴"
                        st.metric(f"{mape_emoji} MAPE", f"{mape:.2f}%")
                    
                    # ===========================================================================================
                    # VISUALIZATION
                    # ===========================================================================================
                    
                    st.markdown('<h2 class="section-header">📈 Data Visualization</h2>', unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "📊 Overall Metrics", "🎯 Commodity Metrics"])
                    
                    with tab1:
                        historical_dates = df_transposed['Tanggal'].tolist()
                        historical_prices = df_transposed[selected_commodity].tolist()
                        
                        future_dates = pd.date_range(start=last_date, periods=weeks_to_predict + 1, freq='W')[1:]
                        future_prices = [scalers[selected_commodity].inverse_transform([[p[commodity_idx]]])[0, 0] for p in predictions]
                        
                        fig1 = go.Figure()
                        
                        fig1.add_trace(go.Scatter(
                            x=historical_dates, y=historical_prices,
                            mode='lines+markers', name='Historical Data',
                            line=dict(color='#60A5FA', width=3),
                            marker=dict(size=6, color='#3B82F6')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=future_dates, y=future_prices,
                            mode='lines+markers', name='Prediction',
                            line=dict(color='#F472B6', width=3, dash='dash'),
                            marker=dict(size=8, symbol='square', color='#EC4899')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date], y=[predicted_price],
                            mode='markers', name=f'Target ({selected_month} {selected_year})',
                            marker=dict(size=15, color='#10B981', symbol='star')
                        ))
                        
                        fig1.update_layout(
                            title=dict(
                                text=f'Price Forecast - {selected_commodity}',
                                font=dict(size=22, color='white', family='Inter')
                            ),
                            xaxis_title='Date',
                            yaxis_title='Price (Rp)',
                            hovermode='x unified',
                            template='plotly_dark',
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                bgcolor='rgba(255,255,255,0.1)',
                                bordercolor='rgba(255,255,255,0.2)',
                                borderwidth=1
                            )
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with tab2:
                        st.markdown(f"#### 📊 Model Performance - All Commodities")
                        st.markdown(f"*Metric source: {metric_source}*")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_rmse = go.Figure()
                                fig_rmse.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['RMSE'],
                                    marker=dict(
                                        color=df_metrics['RMSE'],
                                        colorscale='Blues',
                                        showscale=False
                                    ),
                                    text=df_metrics['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=10, color='white')
                                ))
                                fig_rmse.update_layout(
                                    title='Root Mean Squared Error (RMSE)',
                                    xaxis_title='Commodity',
                                    yaxis_title='RMSE (Rp)',
                                    height=500,
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=60, b=120, l=60, r=20)
                                )
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            with col2:
                                fig_mae = go.Figure()
                                fig_mae.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['MAE'],
                                    marker=dict(
                                        color=df_metrics['MAE'],
                                        colorscale='Reds',
                                        showscale=False
                                    ),
                                    text=df_metrics['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=10, color='white')
                                ))
                                fig_mae.update_layout(
                                    title='Mean Absolute Error (MAE)',
                                    xaxis_title='Commodity',
                                    yaxis_title='MAE (Rp)',
                                    height=500,
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=60, b=120, l=60, r=20)
                                )
                                st.plotly_chart(fig_mae, use_container_width=True)
                            
                            # MAPE Chart
                            colors = []
                            for val in df_metrics['MAPE']:
                                if val < 5:
                                    colors.append('#10B981')
                                elif val < 10:
                                    colors.append('#F59E0B')
                                elif val < 20:
                                    colors.append('#F97316')
                                else:
                                    colors.append('#EF4444')
                            
                            fig_mape = go.Figure()
                            fig_mape.add_trace(go.Bar(
                                x=df_metrics['Komoditas'],
                                y=df_metrics['MAPE'],
                                marker=dict(color=colors),
                                text=df_metrics['MAPE'].apply(lambda x: f'{x:.2f}%'),
                                textposition='outside',
                                textfont=dict(size=11, color='white', family='Inter')
                            ))
                            fig_mape.update_layout(
                                title='Mean Absolute Percentage Error (MAPE)',
                                xaxis_title='Commodity',
                                yaxis_title='MAPE (%)',
                                height=500,
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                margin=dict(t=60, b=120, l=60, r=60)
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                            
                            # Performance Summary
                            excellent_count = len(df_metrics[df_metrics['MAPE'] < 5])
                            good_count = len(df_metrics[(df_metrics['MAPE'] >= 5) & (df_metrics['MAPE'] < 10)])
                            fair_count = len(df_metrics[(df_metrics['MAPE'] >= 10) & (df_metrics['MAPE'] < 20)])
                            poor_count = len(df_metrics[df_metrics['MAPE'] >= 20])
                            
                            st.markdown(f"""
                            <div class="glass-card">
                                <h4 style='margin-top: 0;'>Performance Summary</h4>
                                <p>🟢 <strong>Excellent (< 5%):</strong> {excellent_count} ({excellent_count/len(df_metrics)*100:.1f}%)</p>
                                <p>🟡 <strong>Good (5-10%):</strong> {good_count} ({good_count/len(df_metrics)*100:.1f}%)</p>
                                <p>🟠 <strong>Fair (10-20%):</strong> {fair_count} ({fair_count/len(df_metrics)*100:.1f}%)</p>
                                <p>🔴 <strong>Poor (> 20%):</strong> {poor_count} ({poor_count/len(df_metrics)*100:.1f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.warning("Not enough test data for evaluation")
                    
                    with tab3:
                        st.markdown(f"#### 🎯 Performance Metrics - {selected_commodity}")
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'],
                                y=[rmse, mae],
                                marker=dict(color=['#3B82F6', '#EF4444']),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside',
                                textfont=dict(size=16, color='white', family='Inter')
                            ))
                            fig3.update_layout(
                                title='RMSE & MAE Comparison',
                                yaxis_title='Value (Rp)',
                                template='plotly_dark',
                                height=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                showlegend=False,
                                margin=dict(t=60, b=40, l=60, r=40)
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col_chart2:
                            mape_color = '#10B981' if mape < 5 else '#F59E0B' if mape < 10 else '#F97316' if mape < 20 else '#EF4444'
                            
                            fig4 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=mape,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "MAPE (%)", 'font': {'size': 20, 'color': 'white'}},
                                number={'font': {'size': 40, 'color': 'white'}},
                                gauge={
                                    'axis': {'range': [0, 30], 'tickwidth': 1, 'tickcolor': 'white'},
                                    'bar': {'color': mape_color},
                                    'steps': [
                                        {'range': [0, 5], 'color': 'rgba(16, 185, 129, 0.3)'},
                                        {'range': [5, 10], 'color': 'rgba(245, 158, 11, 0.3)'},
                                        {'range': [10, 20], 'color': 'rgba(249, 115, 22, 0.3)'},
                                        {'range': [20, 30], 'color': 'rgba(239, 68, 68, 0.3)'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "white", 'width': 4},
                                        'thickness': 0.75,
                                        'value': mape
                                    }
                                }
                            ))
                            fig4.update_layout(
                                height=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                margin=dict(t=60, b=40, l=40, r=40)
                            )
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        # Performance status
                        status_color = "🟢 Excellent" if mape < 5 else "🟡 Good" if mape < 10 else "🟠 Fair" if mape < 20 else "🔴 Poor"
                        
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style='margin-top: 0;'>Performance Status - {selected_commodity}</h4>
                            <p><strong>RMSE:</strong> Rp {rmse:,.2f}</p>
                            <p><strong>MAE:</strong> Rp {mae:,.2f}</p>
                            <p><strong>MAPE:</strong> {mape:.2f}%</p>
                            <p><strong>Status:</strong> {status_color}</p>
                            <p style='margin-bottom: 0;'><strong>Interpretation:</strong> MAPE {mape:.2f}% means predictions deviate {mape:.2f}% from actual values on average</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error reading dataset: {str(e)}")

else:
    st.markdown("""
    <div class="glass-card">
        <h3>📁 Getting Started</h3>
        <p>Please upload your dataset to begin the prediction process.</p>
        
        <h4>Expected Dataset Format:</h4>
        <ul>
            <li><strong>Column 1:</strong> No (1, 2, 3, ...)</li>
            <li><strong>Column 2:</strong> Commodity Name</li>
            <li><strong>Column 3+:</strong> Price data with date headers</li>
            <li><strong>File format:</strong> Excel (.xlsx)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
