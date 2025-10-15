import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================================
# KONFIGURASI HALAMAN
# ===========================================================================================

st.set_page_config(
    page_title="LSTM - Prediksi Harga Pangan",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2, h3 {
        color: #34495e;
    }
    .metric-excellent {
        background-color: #27ae60;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-good {
        background-color: #f39c12;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-fair {
        background-color: #e67e22;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-poor {
        background-color: #c0392b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================================================================================
# SIDEBAR - UPLOAD & MODEL INFO
# ===========================================================================================

with st.sidebar:
    st.title("LSTM Model Dashboard")
    st.markdown("---")
    
    st.markdown("### Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload file Excel dataset",
        type=['xlsx', 'xls'],
        help="Upload file dataset dengan format Excel (.xlsx atau .xls)"
    )
    
    st.markdown("---")
    
    st.markdown("### Informasi Model")
    with st.expander("Arsitektur Model"):
        st.write("""
        **Layers:**
        - Bidirectional LSTM (64 units)
        - LSTM (32 units)
        - Dense (32, 16)
        - Dropout & Batch Normalization
        
        **Loss Function:** Huber
        **Optimizer:** Adam
        """)
    
    with st.expander("Hyperparameters"):
        st.write("""
        - Time Steps: 15
        - Batch Size: 16
        - Learning Rate: 0.0005
        - Max Epochs: 150
        - Early Stopping: Enabled
        """)
    
    with st.expander("Preprocessing"):
        st.write("""
        - Detrending: Linear
        - Normalization: MinMax (0-1)
        - Interpolation: Polynomial Order 2
        - Train/Test Split: 85/15
        """)

# ===========================================================================================
# FUNGSI HELPER
# ===========================================================================================

@st.cache_data
def load_and_process_dataset(uploaded_file):
    """Load dan proses dataset dari Excel"""
    try:
        df_raw = pd.read_excel(uploaded_file)
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        # Transpose data
        df_data = df_raw.iloc[:, 2:]
        df_transposed = df_data.T
        df_transposed.columns = komoditas_list
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
        
        # Convert tanggal
        df_transposed['Tanggal'] = pd.to_datetime(
            df_transposed['Tanggal'], 
            format='%d/ %m/ %Y', 
            errors='coerce'
        )
        df_transposed = df_transposed.dropna(subset=['Tanggal'])
        df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
        
        # Convert to numeric
        for kolom in komoditas_list:
            if df_transposed[kolom].dtype == 'object':
                df_transposed[kolom] = df_transposed[kolom].str.replace(',', '').str.replace('"', '')
            df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
        
        # Interpolasi
        df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(
            method='polynomial', order=2, limit_direction='both'
        )
        df_transposed[komoditas_list] = df_transposed[komoditas_list].bfill().ffill()
        
        return df_transposed, komoditas_list
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

@st.cache_resource
def load_model_and_artifacts():
    """Load model, trends, dan scalers"""
    try:
        # Load model
        model = load_model('best_lstm_model.h5', compile=False)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
        
        # Load trends dan scalers
        with open('trends_scalers.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        return model, artifacts['trends'], artifacts['scalers']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_evaluation_results():
    """Load hasil evaluasi"""
    try:
        df = pd.read_csv('hasil_evaluasi_lstm_enhanced.csv')
        return df
    except Exception as e:
        st.warning(f"File evaluasi tidak ditemukan: {str(e)}")
        return None

def get_mape_category(mape):
    """Get kategori MAPE"""
    if mape < 5:
        return "Excellent", "#27ae60"
    elif mape < 10:
        return "Good", "#f39c12"
    elif mape < 20:
        return "Fair", "#e67e22"
    else:
        return "Poor", "#c0392b"

def predict_future(model, last_sequence, trends, scalers, komoditas, n_months, last_date):
    """Prediksi untuk n bulan ke depan"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for i in range(n_months):
        # Predict next step
        pred_norm = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        
        # Inverse transform
        pred_descaled = scalers[komoditas].inverse_transform(pred_norm[:, 0].reshape(-1, 1)).flatten()[0]
        
        # Restore trend
        trend_info = trends[komoditas]
        last_x = trend_info['last_x']
        future_x = last_x + i + 1
        trend_value = np.polyval(trend_info['coeffs'], future_x)
        pred_final = pred_descaled + trend_value
        
        predictions.append(pred_final)
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred_norm[0]
    
    # Generate dates
    future_dates = [last_date + relativedelta(months=i+1) for i in range(n_months)]
    
    return predictions, future_dates

# ===========================================================================================
# MAIN APP
# ===========================================================================================

st.title("Prediksi Harga Pangan dengan LSTM")
st.markdown("### Dashboard Evaluasi dan Forecasting Model")
st.markdown("---")

# Check if dataset uploaded
if uploaded_file is None:
    st.info("Silakan upload dataset Excel di sidebar untuk memulai")
    st.stop()

# Load dataset
df_data, komoditas_list = load_and_process_dataset(uploaded_file)

if df_data is None or komoditas_list is None:
    st.error("Gagal memuat dataset. Pastikan format file sesuai.")
    st.stop()

# Load model dan artifacts
model, trends, scalers = load_model_and_artifacts()

if model is None:
    st.error("Model tidak ditemukan. Pastikan file 'best_lstm_model.h5' dan 'trends_scalers.pkl' tersedia.")
    st.stop()

# Load evaluation results
df_results = load_evaluation_results()

# ===========================================================================================
# TABS
# ===========================================================================================

tab1, tab2, tab3 = st.tabs([
    "Evaluasi Model (Per Komoditas)",
    "Prediksi Harga",
    "Evaluasi Keseluruhan"
])

# ===========================================================================================
# TAB 1: EVALUASI MODEL (PER KOMODITAS)
# ===========================================================================================

with tab1:
    st.header("Evaluasi Model per Komoditas")
    st.markdown("Analisis detail performa model untuk komoditas terpilih")
    
    # Pilih komoditas
    if df_results is not None:
        default_komoditas = df_results.sort_values('MAPE (%)').iloc[0]['Komoditas']
    else:
        default_komoditas = komoditas_list[0]
    
    selected_komoditas = st.selectbox(
        "Pilih Komoditas:",
        komoditas_list,
        index=komoditas_list.index(default_komoditas) if default_komoditas in komoditas_list else 0
    )
    
    if df_results is not None and selected_komoditas in df_results['Komoditas'].values:
        komoditas_data = df_results[df_results['Komoditas'] == selected_komoditas].iloc[0]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mape_val = komoditas_data['MAPE (%)']
            category, color = get_mape_category(mape_val)
            st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'>"
                       f"<h3 style='color:white; margin:0;'>{mape_val:.2f}%</h3>"
                       f"<p style='color:white; margin:5px 0 0 0;'>MAPE - {category}</p></div>",
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("MAE", f"Rp {komoditas_data['MAE']:,.2f}")
        
        with col3:
            st.metric("RMSE", f"Rp {komoditas_data['RMSE']:,.2f}")
        
        with col4:
            rank = df_results.sort_values('MAPE (%)').reset_index(drop=True).index[
                df_results.sort_values('MAPE (%)['Komoditas'] == selected_komoditas
            ].tolist()[0] + 1
            st.metric("Ranking", f"#{rank} dari {len(komoditas_list)}")
        
        st.markdown("---")
        
        # Historical data plot
        st.subheader("Data Historis Harga")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df_data['Tanggal'],
            y=df_data[selected_komoditas],
            mode='lines+markers',
            name='Harga Aktual',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
        
        fig_hist.update_layout(
            title=f"Trend Harga {selected_komoditas}",
            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)",
            hovermode='x unified',
            height=500,
            plot_bgcolor='#f8f9fa'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistik deskriptif
        st.subheader("Statistik Deskriptif")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"Rp {df_data[selected_komoditas].mean():,.2f}")
        
        with col2:
            st.metric("Median", f"Rp {df_data[selected_komoditas].median():,.2f}")
        
        with col3:
            st.metric("Min", f"Rp {df_data[selected_komoditas].min():,.2f}")
        
        with col4:
            st.metric("Max", f"Rp {df_data[selected_komoditas].max():,.2f}")
    
    else:
        st.warning("Data evaluasi untuk komoditas ini tidak tersedia.")

# ===========================================================================================
# TAB 2: PREDIKSI HARGA
# ===========================================================================================

with tab2:
    st.header("Prediksi Harga Future")
    st.markdown("Forecasting harga untuk periode 2025-2026")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_komoditas = st.selectbox(
            "Pilih Komoditas:",
            komoditas_list,
            key="pred_komoditas"
        )
    
    with col2:
        pred_year = st.selectbox(
            "Pilih Tahun:",
            [2025, 2026],
            key="pred_year"
        )
    
    with col3:
        pred_month = st.selectbox(
            "Pilih Bulan:",
            list(range(1, 13)),
            format_func=lambda x: datetime(2025, x, 1).strftime('%B'),
            key="pred_month"
        )
    
    if st.button("Prediksi Harga", type="primary"):
        with st.spinner("Melakukan prediksi..."):
            try:
                # Hitung jumlah bulan dari data terakhir
                last_date = df_data['Tanggal'].max()
                target_date = datetime(pred_year, pred_month, 1)
                months_ahead = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
                
                if months_ahead <= 0:
                    st.error("Tanggal prediksi harus lebih besar dari data terakhir")
                    st.stop()
                
                # Prepare last sequence untuk prediksi
                komoditas_idx = komoditas_list.index(pred_komoditas)
                
                # Detrend data
                x = np.arange(len(df_data))
                y = df_data[pred_komoditas].values
                coeffs = np.polyfit(x, y, 1)
                trend = np.polyval(coeffs, x)
                y_detrended = y - trend
                
                # Normalize
                from sklearn.preprocessing import MinMaxScaler
                scaler_temp = MinMaxScaler(feature_range=(0, 1))
                y_normalized = scaler_temp.fit_transform(y_detrended.reshape(-1, 1)).flatten()
                
                # Get last 15 time steps
                TIME_STEPS = 15
                last_sequence = y_normalized[-TIME_STEPS:]
                
                # Update trends untuk prediksi
                trends_temp = {
                    pred_komoditas: {
                        'coeffs': coeffs,
                        'last_x': len(df_data) - 1,
                        'last_trend': trend[-1]
                    }
                }
                
                scalers_temp = {pred_komoditas: scaler_temp}
                
                # Predict
                predictions, future_dates = predict_future(
                    model,
                    last_sequence,
                    trends_temp,
                    scalers_temp,
                    pred_komoditas,
                    months_ahead,
                    last_date
                )
                
                # Get predicted value untuk bulan target
                predicted_price = predictions[-1]
                
                # Display hasil
                st.success("Prediksi Berhasil!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Harga Prediksi",
                        f"Rp {predicted_price:,.2f}"
                    )
                
                with col2:
                    last_actual = df_data[pred_komoditas].iloc[-1]
                    change = ((predicted_price - last_actual) / last_actual) * 100
                    st.metric(
                        "Perubahan dari Data Terakhir",
                        f"{change:+.2f}%",
                        delta=f"Rp {predicted_price - last_actual:,.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Tanggal Prediksi",
                        target_date.strftime('%B %Y')
                    )
                
                st.markdown("---")
                
                # Plot historical + prediction
                st.subheader("Visualisasi Prediksi")
                
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=df_data['Tanggal'],
                    y=df_data[pred_komoditas],
                    mode='lines+markers',
                    name='Data Historis',
                    line=dict(color='#2E86AB', width=2),
                    marker=dict(size=4)
                ))
                
                # Predicted data
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Prediksi',
                    line=dict(color='#E63946', width=2, dash='dash'),
                    marker=dict(size=6, symbol='star')
                ))
                
                # Highlight target date
                fig_forecast.add_trace(go.Scatter(
                    x=[target_date],
                    y=[predicted_price],
                    mode='markers',
                    name='Target Prediksi',
                    marker=dict(size=15, color='#27ae60', symbol='circle')
                ))
                
                fig_forecast.update_layout(
                    title=f"Forecast Harga {pred_komoditas} hingga {target_date.strftime('%B %Y')}",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    hovermode='x unified',
                    height=600,
                    plot_bgcolor='#f8f9fa',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Tabel prediksi per bulan
                st.subheader("Detail Prediksi per Bulan")
                
                df_forecast = pd.DataFrame({
                    'Tanggal': [d.strftime('%B %Y') for d in future_dates],
                    'Harga Prediksi (Rp)': [f"Rp {p:,.2f}" for p in predictions]
                })
                
                st.dataframe(df_forecast, use_container_width=True, hide_index=True)
                
                # Download prediksi
                csv = df_forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Hasil Prediksi (CSV)",
                    data=csv,
                    file_name=f'prediksi_{pred_komoditas}_{target_date.strftime("%Y%m")}.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Error saat prediksi: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ===========================================================================================
# TAB 3: EVALUASI KESELURUHAN
# ===========================================================================================

with tab3:
    st.header("Evaluasi Keseluruhan Model")
    st.markdown("Performa model untuk semua komoditas")
    
    if df_results is not None:
        # Overview metrics
        st.subheader("Ringkasan Performa")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_mape = df_results['MAPE (%)'].mean()
            st.metric("MAPE Rata-rata", f"{avg_mape:.2f}%")
        
        with col2:
            avg_mae = df_results['MAE'].mean()
            st.metric("MAE Rata-rata", f"Rp {avg_mae:,.0f}")
        
        with col3:
            avg_rmse = df_results['RMSE'].mean()
            st.metric("RMSE Rata-rata", f"Rp {avg_rmse:,.0f}")
        
        with col4:
            excellent_count = len(df_results[df_results['MAPE (%)'] < 5])
            st.metric("MAPE < 5%", f"{excellent_count}")
        
        with col5:
            good_count = len(df_results[df_results['MAPE (%)'] < 10])
            st.metric("MAPE < 10%", f"{good_count}")
        
        st.markdown("---")
        
        # Filter options
        st.subheader("Tabel Hasil Evaluasi")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            filter_option = st.selectbox(
                "Filter Kategori:",
                ["Semua", "Excellent (<5%)", "Good (<10%)", "Fair (<20%)", "Poor (>=20%)"],
                key="filter_eval"
            )
        
        # Apply filter
        if filter_option == "Excellent (<5%)":
            df_filtered = df_results[df_results['MAPE (%)'] < 5].copy()
        elif filter_option == "Good (<10%)":
            df_filtered = df_results[df_results['MAPE (%)'] < 10].copy()
        elif filter_option == "Fair (<20%)":
            df_filtered = df_results[(df_results['MAPE (%)'] >= 10) & (df_results['MAPE (%)'] < 20)].copy()
        elif filter_option == "Poor (>=20%)":
            df_filtered = df_results[df_results['MAPE (%)'] >= 20].copy()
        else:
            df_filtered = df_results.copy()
        
        df_filtered = df_filtered.sort_values('MAPE (%)')
        
        # Color coding function
        def color_mape(val):
            if val < 5:
                color = '#27ae60'
            elif val < 10:
                color = '#f39c12'
            elif val < 20:
                color = '#e67e22'
            else:
                color = '#c0392b'
            return f'background-color: {color}; color: white; font-weight: bold;'
        
        # Format and display table
        styled_df = df_filtered.style.format({
            'RMSE': 'Rp {:,.2f}',
            'MAE': 'Rp {:,.2f}',
            'MAPE (%)': '{:.2f}%'
        }).applymap(color_mape, subset=['MAPE (%)'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown(f"**Menampilkan {len(df_filtered)} dari {len(df_results)} komoditas**")
        
        st.markdown("---")
        
        # Visualisasi metrik
        st.subheader("Visualisasi Performa")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["MAPE", "MAE", "RMSE"])
        
        with viz_tab1:
            df_sorted_mape = df_results.sort_values('MAPE (%)')
            colors = ['#27ae60' if x < 5 else '#f39c12' if x < 10 else '#e67e22' if x < 20 else '#c0392b' 
                      for x in df_sorted_mape['MAPE (%)']]
            
            fig_mape = go.Figure(data=[go.Bar(
                x=df_sorted_mape['MAPE (%)'],
                y=df_sorted_mape['Komoditas'],
                orientation='h',
                marker=dict(color=colors),
                text=df_sorted_mape['MAPE (%)'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
            )])
            
            fig_mape.add_vline(x=5, line_dash="dash", line_color="#27ae60")
            fig_mape.add_vline(x=10, line_dash="dash", line_color="#f39c12")
            fig_mape.add_vline(x=20, line_dash="dash", line_color="#e67e22")
            
            fig_mape.update_layout(
                title="Mean Absolute Percentage Error (MAPE)",
                xaxis_title="MAPE (%)",
                yaxis_title="Komoditas",
                height=900,
                showlegend=False,
                plot_bgcolor='#f8f9fa'
            )
            
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with viz_tab2:
            df_sorted_mae = df_results.sort_values('MAE')
            
            fig_mae = go.Figure(data=[go.Bar(
                x=df_sorted_mae['MAE'],
                y=df_sorted_mae['Komoditas'],
                orientation='h',
                marker=dict(color='#e74c3c'),
                text=df_sorted_mae['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                textposition='outside'
            )])
            
            fig_mae.update_layout(
                title="Mean Absolute Error (MAE)",
                xaxis_title="MAE (Rp)",
                yaxis_title="Komoditas",
                height=900,
                showlegend=False,
                plot_bgcolor='#f8f9fa'
            )
            
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with viz_tab3:
            df_sorted_rmse = df_results.sort_values('RMSE')
            
            fig_rmse = go.Figure(data=[go.Bar(
                x=df_sorted_rmse['RMSE'],
                y=df_sorted_rmse['Komoditas'],
                orientation='h',
                marker=dict(color='#3498db'),
                text=df_sorted_rmse['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                textposition='outside'
            )])
            
            fig_rmse.update_layout(
                title="Root Mean Squared Error (RMSE)",
                xaxis_title="RMSE (Rp)",
                yaxis_title="Komoditas",
                height=900,
                showlegend=False,
                plot_bgcolor='#f8f9fa'
            )
            
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        st.markdown("---")
        
        # Distribusi performa
        st.subheader("Distribusi Performa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            categories = ['Excellent (<5%)', 'Good (5-10%)', 'Fair (10-20%)', 'Poor (>=20%)']
            counts = [
                len(df_results[df_results['MAPE (%)'] < 5]),
                len(df_results[(df_results['MAPE (%)'] >= 5) & (df_results['MAPE (%)'] < 10)]),
                len(df_results[(df_results['MAPE (%)'] >= 10) & (df_results['MAPE (%)'] < 20)]),
                len(df_results[df_results['MAPE (%)'] >= 20])
            ]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=categories,
                values=counts,
                marker=dict(colors=['#27ae60', '#f39c12', '#e67e22', '#c0392b']),
                hole=0.4
            )])
            
            fig_pie.update_layout(title="Distribusi Kategori MAPE", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_hist = go.Figure(data=[go.Histogram(
                x=df_results['MAPE (%)'],
                nbinsx=15,
                marker=dict(color='#3498db', line=dict(color='white', width=1))
            )])
            
            fig_hist.update_layout(
                title="Distribusi Nilai MAPE",
                xaxis_title="MAPE (%)",
                yaxis_title="Jumlah Komoditas",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Download section
        st.subheader("Download Hasil Evaluasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_all = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Semua Data (CSV)",
                data=csv_all,
                file_name='hasil_evaluasi_lengkap.csv',
                mime='text/csv'
            )
        
        with col2:
            if filter_option != "Semua":
                csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {filter_option} (CSV)",
                    data=csv_filtered,
                    file_name=f'hasil_evaluasi_{filter_option.replace(" ", "_")}.csv',
                    mime='text/csv'
                )
    
    else:
        st.warning("File evaluasi tidak ditemukan. Jalankan training model terlebih dahulu.")

# ===========================================================================================
# FOOTER
# ===========================================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Prediksi Harga Pangan dengan LSTM</strong></p>
    <p>Enhanced Model - October 2025</p>
</div>
""", unsafe_allow_html=True)
