import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
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
# SIDEBAR - INFO MODEL
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
    
    st.markdown("---")
    st.markdown("**Versi**")
    st.markdown("""
    - TensorFlow 2.20.0+
    - Python 3.11+
    - Streamlit 1.x
    """)

# ===========================================================================================
# MAIN CONTENT - HEADER
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
        # FORM PREDIKSI
        # ===========================================================================================
        
        st.markdown("### Prediksi Harga")
        
        col_form1, col_form2, col_form3 = st.columns(3)
        
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Button prediksi
        if st.button("Prediksi Harga", use_container_width=True):
            with st.spinner("Memproses prediksi..."):
                try:
                    # ===========================================================================================
                    # PREPROCESSING DATA
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
                    
                    # PERBAIKAN: Normalisasi HANYA untuk komoditas yang dipilih
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    commodity_data = df_transposed[selected_commodity].values.reshape(-1, 1)
                    normalized_data = scaler.fit_transform(commodity_data)
                    
                    # ===========================================================================================
                    # LOAD MODEL DAN PREDIKSI
                    # ===========================================================================================
                    
                    # Load model
                    model = load_model('best_lstm_model.h5', compile=False)
                    
                    # Prediksi untuk bulan yang dipilih
                    month_dict = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
                                 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
                    
                    target_month = month_dict[selected_month]
                    last_date = df_transposed['Tanggal'].iloc[-1]
                    target_date = datetime(selected_year, target_month, 15)
                    
                    weeks_to_predict = max(1, int((target_date - last_date).days / 7))
                    
                    # Generate prediksi
                    TIME_STEPS = 20
                    predictions = []
                    
                    # PERBAIKAN: Input shape (1, TIME_STEPS, 1) - SINGLE FEATURE
                    current_sequence = normalized_data[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)
                    
                    for _ in range(weeks_to_predict):
                        pred = model.predict(current_sequence, verbose=0)
                        predictions.append(pred[0, 0])  # Ambil prediksi untuk single commodity
                        
                        # Update sequence dengan prediksi terbaru
                        new_pred = np.array([[[pred[0, 0]]]])  # Shape (1, 1, 1)
                        current_sequence = np.append(current_sequence[:, 1:, :], new_pred, axis=1)
                    
                    # Inverse transform
                    predicted_price = scaler.inverse_transform([[predictions[-1]]])[0, 0]
                    
                    # ===========================================================================================
                    # HITUNG METRIK EVALUASI
                    # ===========================================================================================
                    
                    split_idx = int(len(normalized_data) * 0.90)
                    test_data = normalized_data[split_idx:]
                    
                    # Generate sequences untuk testing
                    X_test, y_test = [], []
                    for i in range(TIME_STEPS, len(test_data)):
                        X_test.append(test_data[i-TIME_STEPS:i])
                        y_test.append(test_data[i])
                    
                    X_test = np.array(X_test).reshape(-1, TIME_STEPS, 1)  # PERBAIKAN: Reshape ke (samples, TIME_STEPS, 1)
                    y_test = np.array(y_test)
                    
                    # Prediksi pada test set
                    if len(X_test) > 0:
                        y_pred = model.predict(X_test, verbose=0)
                        
                        # Inverse transform
                        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
                        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))
                        
                        # Hitung metrik
                        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
                        mae = mean_absolute_error(y_test_orig, y_pred_orig)
                        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
                    else:
                        rmse, mae, mape = 0, 0, 0
                        y_test_orig = np.array([])
                        y_pred_orig = np.array([])
                    
                    # ===========================================================================================
                    # TAMPILKAN HASIL (Grafik visualization code tetap sama)
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown("### Hasil Prediksi")
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                    
                    with col_result1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="Harga Prediksi",
                            value=f"Rp {predicted_price:,.0f}",
                            help="Prediksi harga untuk bulan yang dipilih"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_result2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="RMSE",
                            value=f"Rp {rmse:,.0f}",
                            help="Root Mean Squared Error"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_result3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="MAE",
                            value=f"Rp {mae:,.0f}",
                            help="Mean Absolute Error"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_result4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="MAPE",
                            value=f"{mape:.2f}%",
                            help="Mean Absolute Percentage Error"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualisasi (kode grafik tetap sama seperti sebelumnya)
                    st.markdown("---")
                    st.markdown("### Visualisasi Prediksi")
                    
                    tab1, tab2, tab3 = st.tabs(["Grafik Prediksi Harga", "Grafik Prediksi vs Aktual (Test Data)", "Grafik Metrik Evaluasi"])
                    
                    with tab1:
                        historical_dates = df_transposed['Tanggal'].tolist()
                        historical_prices = df_transposed[selected_commodity].tolist()
                        
                        future_dates = pd.date_range(start=last_date, periods=weeks_to_predict + 1, freq='W')[1:]
                        future_prices = [scaler.inverse_transform([[p]])[0, 0] for p in predictions]
                        
                        fig1 = go.Figure()
                        
                        fig1.add_trace(go.Scatter(
                            x=historical_dates,
                            y=historical_prices,
                            mode='lines+markers',
                            name='Data Historis',
                            line=dict(color='#2E86AB', width=3),
                            marker=dict(size=6)
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_prices,
                            mode='lines+markers',
                            name='Prediksi',
                            line=dict(color='#E63946', width=3, dash='dash'),
                            marker=dict(size=8, symbol='square')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date],
                            y=[predicted_price],
                            mode='markers',
                            name=f'Target ({selected_month} {selected_year})',
                            marker=dict(size=15, color='#27ae60', symbol='star')
                        ))
                        
                        fig1.update_layout(
                            title=dict(
                                text=f'Prediksi Harga {selected_commodity}',
                                font=dict(size=20, color='#2c3e50', family='Arial Black')
                            ),
                            xaxis_title='Tanggal',
                            yaxis_title='Harga (Rp)',
                            hovermode='x unified',
                            template='plotly_white',
                            height=500,
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Informasi Prediksi:</strong><br>
                            Komoditas: {selected_commodity}<br>
                            Periode Target: {selected_month} {selected_year}<br>
                            Minggu Prediksi: {weeks_to_predict} minggu dari data terakhir<br>
                            Tanggal Data Terakhir: {last_date.strftime('%d %B %Y')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab2:
                        if len(y_test_orig) > 0:
                            fig2 = go.Figure()
                            
                            fig2.add_trace(go.Scatter(
                                x=list(range(len(y_test_orig))),
                                y=y_test_orig.flatten(),
                                mode='lines+markers',
                                name='Aktual',
                                line=dict(color='#2E86AB', width=3),
                                marker=dict(size=6, symbol='circle')
                            ))
                            
                            fig2.add_trace(go.Scatter(
                                x=list(range(len(y_pred_orig))),
                                y=y_pred_orig.flatten(),
                                mode='lines+markers',
                                name='Prediksi',
                                line=dict(color='#E63946', width=3, dash='dash'),
                                marker=dict(size=6, symbol='square')
                            ))
                            
                            fig2.update_layout(
                                title=dict(
                                    text=f'Prediksi vs Aktual - {selected_commodity} (Test Data)',
                                    font=dict(size=20, color='#2c3e50', family='Arial Black')
                                ),
                                xaxis_title='Time Step',
                                yaxis_title='Harga (Rp)',
                                hovermode='x unified',
                                template='plotly_white',
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.warning("Tidak cukup data test untuk visualisasi")
                    
                    with tab3:
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'],
                                y=[rmse, mae],
                                marker=dict(
                                    color=['#3498db', '#e74c3c'],
                                    line=dict(color='white', width=2)
                                ),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside',
                                textfont=dict(size=14, color='#2c3e50', family='Arial Black')
                            ))
                            
                            fig3.update_layout(
                                title=dict(
                                    text='RMSE & MAE',
                                    font=dict(size=18, color='#2c3e50', family='Arial Black')
                                ),
                                yaxis_title='Nilai (Rp)',
                                template='plotly_white',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col_chart2:
                            mape_color = '#27ae60' if mape < 5 else '#f39c12' if mape < 10 else '#e67e22' if mape < 20 else '#c0392b'
                            
                            fig4 = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=mape,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "MAPE (%)", 'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'}},
                                delta={'reference': 10, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                                gauge={
                                    'axis': {'range': [None, 25], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': mape_color},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, 5], 'color': '#d4edda'},
                                        {'range': [5, 10], 'color': '#fff3cd'},
                                        {'range': [10, 20], 'color': '#f8d7da'},
                                        {'range': [20, 25], 'color': '#f5c6cb'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 10
                                    }
                                }
                            ))
                            
                            fig4.update_layout(
                                height=400,
                                margin=dict(l=20, r=20, t=50, b=20)
                            )
                            
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        if mape < 5:
                            performance = "Excellent"
                            color = "#27ae60"
                            message = "Model memiliki performa sangat baik dengan error prediksi sangat rendah."
                        elif mape < 10:
                            performance = "Good"
                            color = "#f39c12"
                            message = "Model memiliki performa baik dengan error prediksi yang dapat diterima."
                        elif mape < 20:
                            performance = "Fair"
                            color = "#e67e22"
                            message = "Model memiliki performa cukup, masih dapat digunakan namun perlu perbaikan."
                        else:
                            performance = "Poor"
                            color = "#c0392b"
                            message = "Model memiliki performa kurang baik, perlu optimasi lebih lanjut."
                        
                        st.markdown(f"""
                        <div style="background-color: {color}15; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {color}; margin-top: 1rem;">
                            <h3 style="color: {color}; margin-bottom: 0.5rem;">Performa Model: {performance}</h3>
                            <p style="color: #2c3e50; font-size: 1rem; margin: 0;">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error saat memproses prediksi: {str(e)}")
                    st.error("Pastikan file 'best_lstm_model.h5' tersedia di direktori yang sama")
    
    except Exception as e:
        st.error(f"Error saat membaca dataset: {str(e)}")

else:
    st.markdown('<div class="info-box">Silakan upload dataset untuk memulai prediksi</div>', unsafe_allow_html=True)
    
    st.markdown("#### Format Dataset yang Diharapkan:")
    st.markdown("""
    - **Kolom 1**: No (1, 2, 3, ...)
    - **Kolom 2**: Nama Komoditas
    - **Kolom 3+**: Data harga dengan header tanggal (format: DD/ MM/ YYYY)
    - **File format**: Excel (.xlsx)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM Neural Network</p>
    <p>Dikembangkan dengan TensorFlow/Keras dan Streamlit</p>
</div>
""", unsafe_allow_html=True)
