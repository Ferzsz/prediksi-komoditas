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
            st.warning(f"⚠️ Dataset berbeda terdeteksi. Menghitung metrik secara real-time...")
            return None, "different_dataset"
        
        st.success("✅ Menggunakan metrik pre-computed dari training 100 epochs")
        return df_eval, "same_dataset"
        
    except FileNotFoundError:
        st.info("ℹ️ File evaluasi tidak ditemukan. Menghitung metrik secara real-time...")
        return None, "file_not_found"
    except Exception as e:
        st.warning(f"⚠️ Error: {str(e)}. Menghitung metrik secara real-time...")
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
    st.markdown("### 📊 Informasi Model")
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

# ===========================================================================================
# MAIN CONTENT
# ===========================================================================================

st.title("📈 Prediksi Harga Komoditas Pangan")
st.markdown("Sistem Prediksi Harga Menggunakan LSTM Neural Network")
st.markdown("---")

# ===========================================================================================
# UPLOAD DATASET
# ===========================================================================================

st.markdown("### 📁 Upload Dataset")
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
        
        st.success(f"✅ Dataset berhasil dimuat: {len(komoditas_list)} komoditas, {df_raw.shape[1] - 2} data points")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komoditas", len(komoditas_list))
        col2.metric("Data Points", df_raw.shape[1] - 2)
        col3.metric("Rentang Waktu", f"{df_raw.shape[1] - 2} minggu")
        
        st.markdown("---")
        
        # ===========================================================================================
        # FORM PREDIKSI
        # ===========================================================================================
        
        st.markdown("### 🎯 Prediksi Harga")
        
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
        
        if st.button("🚀 Prediksi Harga", type="primary", use_container_width=True):
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
                    st.markdown("### ✨ Hasil Prediksi")
                    
                    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                    
                    with col_result1:
                        st.metric("💰 Harga Prediksi", f"Rp {predicted_price:,.0f}", f"{selected_month} {selected_year}")
                    
                    with col_result2:
                        st.metric("RMSE", f"Rp {rmse:,.0f}")
                    
                    with col_result3:
                        st.metric("MAE", f"Rp {mae:,.0f}")
                    
                    with col_result4:
                        mape_status = "🟢 Excellent" if mape < 5 else "🟡 Good" if mape < 10 else "🟠 Fair" if mape < 20 else "🔴 Poor"
                        st.metric("MAPE", f"{mape:.2f}%", mape_status)
                    
                    # ===========================================================================================
                    # VISUALISASI
                    # ===========================================================================================
                    
                    st.markdown("---")
                    st.markdown("### 📊 Visualisasi Prediksi")
                    
                    tab1, tab2, tab3 = st.tabs(["📈 Grafik Prediksi", "🎯 Evaluasi Detail", "📊 Evaluasi Keseluruhan"])
                    
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
                            line=dict(color='#E63946', width=3, dash='dash'), marker=dict(size=8, symbol='diamond')
                        ))
                        
                        fig1.add_trace(go.Scatter(
                            x=[target_date], y=[predicted_price],
                            mode='markers', name=f'Target ({selected_month} {selected_year})',
                            marker=dict(size=15, color='#27ae60', symbol='star')
                        ))
                        
                        fig1.update_layout(
                            title=f'Prediksi Harga {selected_commodity}',
                            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                            hovermode='x unified', template='plotly_white', height=500
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        st.info(f"""
                        **Informasi Prediksi:**
                        - Komoditas: {selected_commodity}
                        - Target: {selected_month} {selected_year}
                        - Minggu Prediksi: {weeks_to_predict}
                        - Harga: Rp {predicted_price:,.0f}
                        """)
                    
                    with tab2:
                        st.markdown(f"#### Evaluasi Metrik - {selected_commodity}")
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=['RMSE', 'MAE'], 
                                y=[rmse, mae],
                                marker=dict(color=['#3498db', '#e74c3c']),
                                text=[f'Rp {rmse:,.0f}', f'Rp {mae:,.0f}'],
                                textposition='outside'
                            ))
                            fig3.update_layout(
                                title='RMSE & MAE',
                                yaxis_title='Nilai (Rp)', 
                                template='plotly_white', 
                                height=450
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col_chart2:
                            mape_color = '#27ae60' if mape < 5 else '#f39c12' if mape < 10 else '#e67e22' if mape < 20 else '#c0392b'
                            
                            fig4 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=mape,
                                title={'text': "MAPE (%)"},
                                gauge={
                                    'axis': {'range': [0, 30]},
                                    'bar': {'color': mape_color},
                                    'steps': [
                                        {'range': [0, 5], 'color': '#d5f4e6'},
                                        {'range': [5, 10], 'color': '#fcf3cf'},
                                        {'range': [10, 20], 'color': '#fae5d3'},
                                        {'range': [20, 30], 'color': '#fadbd8'}
                                    ],
                                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': mape}
                                }
                            ))
                            fig4.update_layout(height=450)
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        st.info(f"""
                        **Status Performa:**
                        - RMSE: Rp {rmse:,.2f}
                        - MAE: Rp {mae:,.2f}
                        - MAPE: {mape:.2f}% ({mape_status})
                        """)
                    
                    with tab3:
                        st.markdown("#### Evaluasi Semua Komoditas")
                        
                        if len(all_metrics) > 0:
                            df_metrics = pd.DataFrame(all_metrics)
                            
                            # MAPE Chart
                            colors = ['#27ae60' if val < 5 else '#f39c12' if val < 10 else '#e67e22' if val < 20 else '#c0392b' 
                                     for val in df_metrics['MAPE']]
                            
                            fig_mape = go.Figure()
                            fig_mape.add_trace(go.Bar(
                                x=df_metrics['Komoditas'],
                                y=df_metrics['MAPE'],
                                marker=dict(color=colors),
                                text=df_metrics['MAPE'].apply(lambda x: f'{x:.2f}%'),
                                textposition='outside'
                            ))
                            fig_mape.update_layout(
                                title='Mean Absolute Percentage Error (MAPE)',
                                xaxis_title='Komoditas',
                                yaxis_title='MAPE (%)',
                                height=500,
                                template='plotly_white',
                                xaxis={'tickangle': -45}
                            )
                            st.plotly_chart(fig_mape, use_container_width=True)
                            
                            # Performance Summary
                            excellent_count = len(df_metrics[df_metrics['MAPE'] < 5])
                            good_count = len(df_metrics[(df_metrics['MAPE'] >= 5) & (df_metrics['MAPE'] < 10)])
                            fair_count = len(df_metrics[(df_metrics['MAPE'] >= 10) & (df_metrics['MAPE'] < 20)])
                            poor_count = len(df_metrics[df_metrics['MAPE'] >= 20])
                            
                            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                            col_perf1.metric("🟢 Excellent", excellent_count, f"{excellent_count/len(df_metrics)*100:.1f}%")
                            col_perf2.metric("🟡 Good", good_count, f"{good_count/len(df_metrics)*100:.1f}%")
                            col_perf3.metric("🟠 Fair", fair_count, f"{fair_count/len(df_metrics)*100:.1f}%")
                            col_perf4.metric("🔴 Poor", poor_count, f"{poor_count/len(df_metrics)*100:.1f}%")
                            
                            st.markdown("---")
                            st.markdown("#### Tabel Detail Metrik")
                            
                            df_display = df_metrics.copy()
                            df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAE'] = df_display['MAE'].apply(lambda x: f"Rp {x:,.2f}")
                            df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(df_display, use_container_width=True, height=400)
                        else:
                            st.warning("Tidak cukup data test untuk evaluasi")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Error saat membaca dataset: {str(e)}")

else:
    st.info("""
    📤 **Silakan upload dataset untuk memulai**
    
    **Format Dataset:**
    - Kolom 1: No (1, 2, 3, ...)
    - Kolom 2: Nama Komoditas
    - Kolom 3+: Data harga dengan header tanggal
    - File format: Excel (.xlsx)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>Sistem Prediksi Harga Komoditas Pangan • LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)
