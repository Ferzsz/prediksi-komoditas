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
                    
                    # Dictionary untuk menyimpan metrik semua komoditas
                    all_metrics = []
                    
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
                        
                        # Ambil metrik untuk komoditas yang dipilih
                        selected_metrics = [m for m in all_metrics if m['Komoditas'] == selected_commodity][0]
                        rmse = selected_metrics['RMSE']
                        mae = selected_metrics['MAE']
                        mape = selected_metrics['MAPE']
                        
                        # Data untuk grafik prediksi vs aktual (komoditas terpilih)
                        y_test_commodity = y_test[:, commodity_idx]
                        y_pred_commodity = y_pred[:, commodity_idx]
                        y_test_orig = scalers[selected_commodity].inverse_transform(y_test_commodity.reshape(-1, 1))
                        y_pred_orig = scalers[selected_commodity].inverse_transform(y_pred_commodity.reshape(-1, 1))
                    else:
                        rmse, mae, mape = 0, 0, 0
                        y_test_orig = np.array([])
                        y_pred_orig = np.array([])
                        all_metrics = []
                    
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
                        st.markdown("#### Evaluasi Performa Model untuk Semua Komoditas")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            # Visualisasi metrik keseluruhan
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Grafik RMSE untuk semua komoditas
                                fig_rmse = go.Figure()
                                fig_rmse.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['RMSE'],
                                    marker=dict(color='#3498db'),
                                    text=df_metrics['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=10)
                                ))
                                fig_rmse.update_layout(
                                    title='Root Mean Squared Error (RMSE) - Semua Komoditas',
                                    xaxis_title='Komoditas',
                                    yaxis_title='RMSE (Rp)',
                                    height=550,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=80, b=150, l=80, r=40)
                                )
                                st.plotly_chart(fig_rmse, use_container_width=True)
                                
                                # Indikator RMSE
                                st.markdown("""
                                **Interpretasi RMSE:**
                                - Semakin **rendah** semakin baik
                                - Mengukur rata-rata kesalahan prediksi
                                - Sensitif terhadap outlier (penalti lebih besar untuk error besar)
                                - Satuan: Rupiah (Rp)
                                - **Score bagus:** Relatif terhadap rentang harga komoditas (semakin kecil % dari harga rata-rata, semakin baik)
                                """)
                            
                            with col2:
                                # Grafik MAE untuk semua komoditas
                                fig_mae = go.Figure()
                                fig_mae.add_trace(go.Bar(
                                    x=df_metrics['Komoditas'],
                                    y=df_metrics['MAE'],
                                    marker=dict(color='#e74c3c'),
                                    text=df_metrics['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                                    textposition='outside',
                                    textfont=dict(size=10)
                                ))
                                fig_mae.update_layout(
                                    title='Mean Absolute Error (MAE) - Semua Komoditas',
                                    xaxis_title='Komoditas',
                                    yaxis_title='MAE (Rp)',
                                    height=550,
                                    template='plotly_white',
                                    xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                    margin=dict(t=80, b=150, l=80, r=40)
                                )
                                st.plotly_chart(fig_mae, use_container_width=True)
                                
                                # Indikator MAE
                                st.markdown("""
                                **Interpretasi MAE:**
                                - Semakin **rendah** semakin baik
                                - Lebih mudah diinterpretasi (rata-rata error absolut)
                                - Kurang sensitif terhadap outlier dibanding RMSE
                                - Satuan: Rupiah (Rp)
                                - **Score bagus:** MAE yang mendekati 0 atau < 5% dari nilai rata-rata
                                """)
                            
                            # Grafik MAPE - Full width
                            # Tentukan warna berdasarkan MAPE
                            colors = []
                            for val in df_metrics['MAPE']:
                                if val < 5:
                                    colors.append('#27ae60')  # Excellent
                                elif val < 10:
                                    colors.append('#f39c12')  # Good
                                elif val < 20:
                                    colors.append('#e67e22')  # Fair
                                else:
                                    colors.append('#c0392b')  # Poor
                            
                            fig_mape = go.Figure()
                            fig_mape.add_trace(go.Bar(
                                x=df_metrics['Komoditas'],
                                y=df_metrics['MAPE'],
                                marker=dict(color=colors),
                                text=df_metrics['MAPE'].apply(lambda x: f'{x:.2f}%'),
                                textposition='outside',
                                textfont=dict(size=11, family='Arial Black')
                            ))
                            fig_mape.update_layout(
                                title='Mean Absolute Percentage Error (MAPE) - Semua Komoditas',
                                xaxis_title='Komoditas',
                                yaxis_title='MAPE (%)',
                                height=550,
                                template='plotly_white',
                                xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
                                margin=dict(t=80, b=150, l=80, r=80)
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                            
                            # Indikator MAPE dengan kategori dan score
                            st.markdown("""
                            **Interpretasi MAPE & Indikator Score yang Bagus:**
                            
                            MAPE (Mean Absolute Percentage Error) mengukur rata-rata persentase kesalahan prediksi relatif terhadap nilai aktual. 
                            Nilai yang lebih rendah menunjukkan model lebih akurat.
                            
                            | Kategori | Range MAPE | Kualitas Model | Interpretasi |
                            |----------|------------|----------------|--------------|
                            | 🟢 **Excellent** | **< 5%** | Model sangat akurat | Prediksi sangat dekat dengan nilai aktual, error rata-rata < 5% |
                            | 🟡 **Good** | **5% - 10%** | Model akurat | Prediksi akurat dengan error yang dapat diterima untuk forecasting |
                            | 🟠 **Fair** | **10% - 20%** | Model cukup baik | Model masih dapat digunakan, namun perlu monitoring |
                            | 🔴 **Poor** | **> 20%** | Model perlu perbaikan | Error terlalu besar, model memerlukan optimasi atau re-training |
                            
                            **Catatan Penting:**
                            - **Target ideal untuk prediksi harga komoditas: MAPE < 10%** (kategori Excellent atau Good)
                            - MAPE < 5% dianggap **performa luar biasa** dalam forecasting
                            - Nilai mendekati 0% = prediksi hampir sempurna
                            - MAPE > 20% mengindikasikan model perlu ditingkatkan dengan:
                              - Hyperparameter tuning lebih lanjut
                              - Penambahan fitur atau data training
                              - Arsitektur model yang berbeda
                            
                            **Contoh Interpretasi:**
                            - MAPE 2.71% = Prediksi rata-rata meleset hanya 2.71% dari nilai aktual (**Excellent**)
                            - MAPE 7.5% = Prediksi rata-rata meleset 7.5% dari nilai aktual (**Good**)
                            - MAPE 15% = Prediksi rata-rata meleset 15% dari nilai aktual (**Fair**)
                            """)
                            
                            # Tampilkan ringkasan performa
                            excellent_count = len(df_metrics[df_metrics['MAPE'] < 5])
                            good_count = len(df_metrics[(df_metrics['MAPE'] >= 5) & (df_metrics['MAPE'] < 10)])
                            fair_count = len(df_metrics[(df_metrics['MAPE'] >= 10) & (df_metrics['MAPE'] < 20)])
                            poor_count = len(df_metrics[df_metrics['MAPE'] >= 20])
                            
                            st.markdown(f"""
                            **Ringkasan Performa Model Keseluruhan:**
                            - 🟢 Excellent (< 5%): **{excellent_count} komoditas** ({excellent_count/len(df_metrics)*100:.1f}%)
                            - 🟡 Good (5-10%): **{good_count} komoditas** ({good_count/len(df_metrics)*100:.1f}%)
                            - 🟠 Fair (10-20%): **{fair_count} komoditas** ({fair_count/len(df_metrics)*100:.1f}%)
                            - 🔴 Poor (> 20%): **{poor_count} komoditas** ({poor_count/len(df_metrics)*100:.1f}%)
                            
                            **Kesimpulan:** {'Model memiliki performa sangat baik pada mayoritas komoditas!' if (excellent_count + good_count) / len(df_metrics) >= 0.8 else 'Model memiliki performa baik, namun beberapa komoditas perlu optimasi lebih lanjut.' if (excellent_count + good_count) / len(df_metrics) >= 0.6 else 'Model perlu perbaikan signifikan untuk meningkatkan akurasi prediksi.'}
                            """)
                            
                        else:
                            st.warning("Tidak cukup data test untuk menghitung metrik evaluasi")
                    
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
                        - Score MAPE {mape:.2f}% berarti prediksi rata-rata meleset {mape:.2f}% dari nilai aktual
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
