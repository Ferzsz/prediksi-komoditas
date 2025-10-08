import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Custom CSS
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
        
        if st.button("Prediksi Harga", use_container_width=True):
            with st.spinner("Memproses prediksi..."):
                try:
                    # ===========================================================================================
                    # PREPROCESSING - SEMUA KOMODITAS
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
                    
                    # Normalisasi SEMUA komoditas
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
                    
                    # Ambil prediksi untuk komoditas yang dipilih
                    commodity_idx = komoditas_list.index(selected_commodity)
                    predicted_price_norm = predictions[-1][commodity_idx]
                    predicted_price = scalers[selected_commodity].inverse_transform([[predicted_price_norm]])[0, 0]
                    
                    # ===========================================================================================
                    # HITUNG METRIK UNTUK SEMUA KOMODITAS
                    # ===========================================================================================
                    
                    split_idx = int(len(data_normalized) * 0.90)
                    test_data = data_normalized[split_idx:]
                    
                    # Dictionary untuk menyimpan metrik dan data prediksi semua komoditas
                    all_metrics = []
                    all_predictions_data = {}
                    
                    X_test, y_test = [], []
                    for i in range(TIME_STEPS, len(test_data)):
                        X_test.append(test_data[i-TIME_STEPS:i])
                        y_test.append(test_data[i])
                    
                    X_test = np.array(X_test)
                    y_test = np.array(y_test)
                    
                    if len(X_test) > 0:
                        y_pred = model.predict(X_test, verbose=0)
                        
                        # Hitung metrik untuk setiap komoditas
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
                            
                            # Simpan data prediksi vs aktual untuk grafik
                            all_predictions_data[commodity_name] = {
                                'actual': y_test_orig.flatten(),
                                'predicted': y_pred_orig.flatten()
                            }
                        
                        # Ambil metrik untuk komoditas yang dipilih
                        selected_metrics = [m for m in all_metrics if m['Komoditas'] == selected_commodity][0]
                        rmse = selected_metrics['RMSE']
                        mae = selected_metrics['MAE']
                        mape = selected_metrics['MAPE']
                        
                        # Data untuk grafik prediksi vs aktual (komoditas terpilih)
                        y_test_orig = all_predictions_data[selected_commodity]['actual']
                        y_pred_orig = all_predictions_data[selected_commodity]['predicted']
                    else:
                        rmse, mae, mape = 0, 0, 0
                        y_test_orig = np.array([])
                        y_pred_orig = np.array([])
                        all_metrics = []
                        all_predictions_data = {}
                    
                    # ===========================================================================================
                    # TAMPILKAN HASIL
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
                    
                    tab1, tab2, tab3 = st.tabs(["Grafik Prediksi Harga", "Prediksi vs Aktual - Semua Komoditas", "Grafik Metrik Evaluasi"])
                    
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
                            mode='markers', name=f'Target ({selected_month} {selected_year})',
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
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Informasi Prediksi:</strong><br>
                            Komoditas: {selected_commodity}<br>
                            Periode Target: {selected_month} {selected_year}<br>
                            Minggu Prediksi: {weeks_to_predict} minggu<br>
                            Tanggal Data Terakhir: {last_date.strftime('%d %B %Y')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab2:
                        st.markdown("#### Prediksi vs Aktual - Semua Komoditas (100 Epochs Optimal - Urut dari MAPE Terbaik)")
                        
                        if len(all_predictions_data) > 0:
                            # Urutkan komoditas berdasarkan MAPE (terbaik ke terburuk)
                            df_metrics = pd.DataFrame(all_metrics)
                            df_metrics_sorted = df_metrics.sort_values('MAPE')
                            
                            # Buat subplot 8x4 = 32 grafik (untuk 31 komoditas)
                            rows = 8
                            cols = 4
                            
                            fig = make_subplots(
                                rows=rows, cols=cols,
                                subplot_titles=[f"{row['Komoditas']}<br>MAPE: {row['MAPE']:.2f}%" 
                                              for _, row in df_metrics_sorted.iterrows()],
                                vertical_spacing=0.08,
                                horizontal_spacing=0.06
                            )
                            
                            # Tambahkan grafik untuk setiap komoditas
                            for idx, (_, row) in enumerate(df_metrics_sorted.iterrows()):
                                commodity_name = row['Komoditas']
                                mape_val = row['MAPE']
                                
                                actual_data = all_predictions_data[commodity_name]['actual']
                                predicted_data = all_predictions_data[commodity_name]['predicted']
                                
                                # Tentukan warna MAPE
                                if mape_val < 5:
                                    mape_color = '#27ae60'  # Green
                                elif mape_val < 10:
                                    mape_color = '#f39c12'  # Yellow
                                elif mape_val < 20:
                                    mape_color = '#e67e22'  # Orange
                                else:
                                    mape_color = '#c0392b'  # Red
                                
                                row_num = idx // cols + 1
                                col_num = idx % cols + 1
                                
                                # Line Aktual (biru)
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(range(len(actual_data))),
                                        y=actual_data,
                                        mode='lines',
                                        name='Aktual',
                                        line=dict(color='#2E86AB', width=1.5),
                                        showlegend=(idx == 0),
                                        legendgroup='actual'
                                    ),
                                    row=row_num, col=col_num
                                )
                                
                                # Line Prediksi (merah putus-putus)
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(range(len(predicted_data))),
                                        y=predicted_data,
                                        mode='lines',
                                        name='Prediksi',
                                        line=dict(color='#E63946', width=1.5, dash='dash'),
                                        showlegend=(idx == 0),
                                        legendgroup='predicted'
                                    ),
                                    row=row_num, col=col_num
                                )
                                
                                # Update axes untuk subplot ini
                                fig.update_xaxes(title_text="Time Step", row=row_num, col=col_num, title_font=dict(size=9))
                                fig.update_yaxes(title_text="Harga (Rp)", row=row_num, col=col_num, title_font=dict(size=9))
                            
                            # Update layout keseluruhan
                            fig.update_layout(
                                height=2800,
                                showlegend=True,
                                template='plotly_white',
                                title_text="",
                                legend=dict(
                                    orientation="h",
                                    yanchor="top",
                                    y=1.01,
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(size=12)
                                )
                            )
                            
                            # Update font size untuk subtitle
                            for annotation in fig['layout']['annotations']:
                                annotation['font'] = dict(size=9)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Tampilkan tabel metrik
                            st.markdown("#### Tabel Metrik Evaluasi (Urut dari MAPE Terbaik)")
                            
                            # Format tabel
                            df_display = df_metrics_sorted.copy()
                            df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"Rp {x:,.0f}")
                            df_display['MAE'] = df_display['MAE'].apply(lambda x: f"Rp {x:,.0f}")
                            df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(df_display, use_container_width=True, height=400)
                            
                        else:
                            st.warning("Tidak cukup data test untuk visualisasi")
                    
                    with tab3:
                        st.markdown(f"#### Evaluasi Metrik - {selected_commodity}")
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'], 
                                y=[rmse, mae],
                                marker=dict(color=['#3498db', '#e74c3c']),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside',
                                textfont=dict(size=16, color='#2c3e50', family='Arial Black')
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
                            # Gauge MAPE dengan indikator yang jelas
                            mape_color = '#27ae60' if mape < 5 else '#f39c12' if mape < 10 else '#e67e22' if mape < 20 else '#c0392b'
                            
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
                                        {'range': [0, 5], 'color': '#d5f4e6'},
                                        {'range': [5, 10], 'color': '#fcf3cf'},
                                        {'range': [10, 20], 'color': '#fae5d3'},
                                        {'range': [20, 30], 'color': '#fadbd8'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
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
                        
                        # Tambahkan penjelasan score
                        st.markdown(f"""
                        **Status Performa Model untuk {selected_commodity}:**
                        - **RMSE:** Rp {rmse:,.0f} (nilai absolut error dalam Rupiah)
                        - **MAE:** Rp {mae:,.0f} (rata-rata kesalahan prediksi)
                        - **MAPE:** {mape:.2f}% (persentase error relatif)
                        
                        **Evaluasi:** {'🟢 Excellent - Model sangat akurat!' if mape < 5 else '🟡 Good - Model akurat' if mape < 10 else '🟠 Fair - Model cukup baik' if mape < 20 else '🔴 Poor - Model perlu ditingkatkan'}
                        
                        **Interpretasi Standar:**
                        
                        | Kategori | Range MAPE | Kualitas Model |
                        |----------|------------|----------------|
                        | 🟢 Excellent | < 5% | Model sangat akurat |
                        | 🟡 Good | 5% - 10% | Model akurat |
                        | 🟠 Fair | 10% - 20% | Model cukup baik |
                        | 🔴 Poor | > 20% | Model perlu perbaikan |
                        
                        **Catatan:**
                        - RMSE dan MAE lebih rendah menunjukkan prediksi lebih akurat dalam satuan Rupiah
                        - MAPE memberikan perspektif persentase error yang mudah dipahami
                        - Untuk prediksi harga komoditas, MAPE < 10% dianggap hasil yang sangat baik
                        """)
                    
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
    - **Kolom 3+**: Data harga dengan header tanggal (format: DD/ MM/ YYYY)
    - **File format**: Excel (.xlsx)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)
