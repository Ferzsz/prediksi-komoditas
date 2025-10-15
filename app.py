# app.py - Streamlit Application untuk Prediksi Harga Pangan

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import h5py
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ===================================================================
# KONFIGURASI HALAMAN
# ===================================================================

st.set_page_config(
    page_title="Prediksi Harga Pangan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# FUNGSI LOAD MODEL DAN DATA
# ===================================================================

@st.cache_resource
def load_model_from_h5(commodity_name):
    """Load model tertentu dari all_models.h5"""
    try:
        with h5py.File('all_models.h5', 'r') as h5file:
            safe_name = commodity_name.replace('/', '_').replace(' ', '_')
            group = h5file[safe_name]
            
            # Load arsitektur
            model_config = group.attrs['model_config']
            model = model_from_json(model_config)
            
            # Load weights
            weight_group = group['weights']
            for i, layer in enumerate(model.layers):
                if f'layer_{i}' in weight_group:
                    layer_group = weight_group[f'layer_{i}']
                    weights = []
                    j = 0
                    while f'weight_{j}' in layer_group:
                        weights.append(layer_group[f'weight_{j}'][:])
                        j += 1
                    if weights:
                        layer.set_weights(weights)
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scalers():
    """Load semua scaler"""
    try:
        with open('all_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        return scalers
    except Exception as e:
        st.error(f"Error loading scalers: {str(e)}")
        return None

@st.cache_data
def load_forecasting_data():
    """Load hasil forecasting"""
    try:
        df = pd.read_csv('hasil_forecasting.csv')
        df = df.set_index('Komoditas')
        return df
    except Exception as e:
        st.error(f"Error loading forecasting data: {str(e)}")
        return None

@st.cache_data
def load_evaluation_data():
    """Load hasil evaluasi"""
    try:
        df = pd.read_csv('hasil_evaluasi.csv')
        return df
    except Exception as e:
        st.error(f"Error loading evaluation data: {str(e)}")
        return None

@st.cache_data
def load_dataset():
    """Load dataset asli"""
    try:
        df = pd.read_excel('dataset.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# ===================================================================
# FUNGSI HELPER
# ===================================================================

def get_month_column(year, month):
    """Convert tahun dan bulan ke format kolom CSV"""
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    return f"{month_names[month]}-{year}"

def get_performance_indicator(mape):
    """Menentukan indikator performa berdasarkan MAPE"""
    if mape < 10:
        return "Sangat Baik", "green"
    elif mape < 20:
        return "Baik", "blue"
    elif mape < 30:
        return "Cukup", "orange"
    else:
        return "Perlu Perbaikan", "red"

def format_currency(value):
    """Format angka ke format Rupiah"""
    return f"Rp {value:,.0f}"

# ===================================================================
# SIDEBAR - INFORMASI MODEL
# ===================================================================

st.sidebar.title("Informasi Model")
st.sidebar.markdown("---")

st.sidebar.subheader("Arsitektur LSTM")
st.sidebar.markdown("""
**Model:** Multi-layer LSTM

**Struktur Layer:**
- Layer 1: LSTM (128 units) + Dropout (0.2)
- Layer 2: LSTM (64 units) + Dropout (0.2)
- Layer 3: LSTM (32 units) + Dropout (0.2)
- Output: Dense (1 unit)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")
st.sidebar.markdown("""
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** MSE
- **Batch Size:** 32
- **Epochs:** 100
- **Early Stopping:** Patience 10
- **Time Steps:** 7 periode
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Training")
st.sidebar.markdown("""
- **Periode Data:** 2020-2025
- **Jumlah Komoditas:** 31
- **Split Ratio:** 80% Train / 20% Test
- **Normalisasi:** MinMaxScaler (0-1)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Metrik Evaluasi")
st.sidebar.markdown("""
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error
- **Target MAPE:** < 20%
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Periode Forecasting")
st.sidebar.markdown("""
- **Periode:** Nov 2025 - Des 2026
- **Total Bulan:** 14 bulan
- **Metode:** Iterative Forecasting
""")

# ===================================================================
# HEADER
# ===================================================================

st.title("Prediksi Harga Pangan Multi Komoditas")
st.markdown("---")
st.markdown("""
Aplikasi ini menggunakan model **LSTM (Long Short-Term Memory)** untuk memprediksi 
harga 31 komoditas pangan di Indonesia untuk periode **2025-2026**.
""")

# ===================================================================
# MENU TAB
# ===================================================================

tab1, tab2 = st.tabs(["Prediksi Harga", "Evaluasi Model"])

# ===================================================================
# TAB 1: PREDIKSI HARGA
# ===================================================================

with tab1:
    st.header("Prediksi Harga Komoditas")
    st.markdown("---")
    
    # Load data
    forecast_df = load_forecasting_data()
    
    if forecast_df is not None:
        # Daftar komoditas
        commodities = forecast_df.index.tolist()
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_commodity = st.selectbox(
                "Pilih Komoditas:",
                commodities,
                help="Pilih komoditas yang ingin diprediksi"
            )
        
        with col2:
            selected_year = st.selectbox(
                "Pilih Tahun:",
                [2025, 2026],
                help="Pilih tahun prediksi"
            )
        
        with col3:
            if selected_year == 2025:
                months = list(range(11, 13))  # Nov-Des 2025
                month_labels = ["November", "Desember"]
            else:
                months = list(range(1, 13))  # Jan-Des 2026
                month_labels = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                               "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
            
            selected_month = st.selectbox(
                "Pilih Bulan:",
                months,
                format_func=lambda x: month_labels[months.index(x)],
                help="Pilih bulan prediksi"
            )
        
        # Button prediksi
        if st.button("Prediksi Harga", type="primary", use_container_width=True):
            # Get column name
            col_name = get_month_column(selected_year, selected_month)
            
            try:
                # Get predicted price
                predicted_price = forecast_df.loc[selected_commodity, col_name]
                
                # Display result
                st.markdown("---")
                st.subheader("Hasil Prediksi")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        label=f"Harga Prediksi - {month_labels[months.index(selected_month)]} {selected_year}",
                        value=format_currency(predicted_price)
                    )
                    
                    st.info(f"""
                    **Detail Prediksi:**
                    - Komoditas: {selected_commodity}
                    - Periode: {month_labels[months.index(selected_month)]} {selected_year}
                    - Model: LSTM 3-Layer
                    """)
                
                with col2:
                    # Plot trend forecasting untuk komoditas terpilih
                    commodity_data = forecast_df.loc[selected_commodity]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=commodity_data.index,
                        y=commodity_data.values,
                        mode='lines+markers',
                        name='Prediksi Harga',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ))
                    
                    # Highlight selected month
                    fig.add_trace(go.Scatter(
                        x=[col_name],
                        y=[predicted_price],
                        mode='markers',
                        name='Bulan Terpilih',
                        marker=dict(size=15, color='red', symbol='star')
                    ))
                    
                    fig.update_layout(
                        title=f'Trend Forecasting: {selected_commodity}',
                        xaxis_title='Periode',
                        yaxis_title='Harga (Rp)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tabel forecasting lengkap
                st.markdown("---")
                st.subheader("Tabel Prediksi Lengkap")
                
                # Format tabel
                display_data = commodity_data.to_frame(name='Harga Prediksi (Rp)')
                display_data['Harga Prediksi (Rp)'] = display_data['Harga Prediksi (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
                
                st.dataframe(display_data, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.error("Data forecasting tidak tersedia. Pastikan file hasil_forecasting.csv ada di repository.")

# ===================================================================
# TAB 2: EVALUASI MODEL
# ===================================================================

with tab2:
    st.header("Evaluasi Model LSTM")
    st.markdown("---")
    
    # Load evaluation data
    eval_df = load_evaluation_data()
    
    if eval_df is not None:
        # Summary metrics
        st.subheader("Ringkasan Performa Model")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_rmse = eval_df['RMSE_Test'].mean()
            st.metric("Rata-rata RMSE", f"{avg_rmse:,.2f}")
        
        with col2:
            avg_mae = eval_df['MAE_Test'].mean()
            st.metric("Rata-rata MAE", f"{avg_mae:,.2f}")
        
        with col3:
            avg_mape = eval_df['MAPE_Test (%)'].mean()
            st.metric("Rata-rata MAPE", f"{avg_mape:.2f}%")
        
        with col4:
            success_count = len(eval_df[eval_df['MAPE_Test (%)'] < 20])
            st.metric("Target Tercapai", f"{success_count}/31 komoditas")
        
        # Indikator performa
        st.markdown("---")
        status, color = get_performance_indicator(avg_mape)
        
        if status == "Sangat Baik":
            st.success(f"Status Model: **{status}** - MAPE < 10%")
        elif status == "Baik":
            st.info(f"Status Model: **{status}** - MAPE < 20%")
        elif status == "Cukup":
            st.warning(f"Status Model: **{status}** - MAPE 20-30%")
        else:
            st.error(f"Status Model: **{status}** - MAPE > 30%")
        
        # Detail evaluasi per komoditas
        st.markdown("---")
        st.subheader("Detail Evaluasi per Komoditas")
        
        # Add color coding untuk status
        def color_mape(val):
            if val < 10:
                color = 'background-color: #d4edda'
            elif val < 20:
                color = 'background-color: #d1ecf1'
            elif val < 30:
                color = 'background-color: #fff3cd'
            else:
                color = 'background-color: #f8d7da'
            return color
        
        # Format dataframe
        display_df = eval_df.copy()
        display_df['RMSE_Test'] = display_df['RMSE_Test'].apply(lambda x: f"{x:,.2f}")
        display_df['MAE_Test'] = display_df['MAE_Test'].apply(lambda x: f"{x:,.2f}")
        display_df['MAPE_Test (%)'] = display_df['MAPE_Test (%)'].round(2)
        
        # Tampilkan tabel dengan styling
        st.dataframe(
            display_df[['Komoditas', 'RMSE_Test', 'MAE_Test', 'MAPE_Test (%)', 'Status_MAPE']],
            use_container_width=True,
            height=600
        )
        
        # Visualisasi MAPE
        st.markdown("---")
        st.subheader("Visualisasi MAPE per Komoditas")
        
        # Sort by MAPE
        eval_sorted = eval_df.sort_values('MAPE_Test (%)')
        
        # Color based on performance
        colors = []
        for mape in eval_sorted['MAPE_Test (%)']:
            if mape < 10:
                colors.append('green')
            elif mape < 20:
                colors.append('blue')
            elif mape < 30:
                colors.append('orange')
            else:
                colors.append('red')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=eval_sorted['MAPE_Test (%)'],
            y=eval_sorted['Komoditas'],
            orientation='h',
            marker=dict(color=colors),
            text=eval_sorted['MAPE_Test (%)'].round(2),
            textposition='outside'
        ))
        
        # Add target line
        fig.add_vline(x=20, line_dash="dash", line_color="black", 
                     annotation_text="Target: 20%", annotation_position="top right")
        
        fig.update_layout(
            title='Mean Absolute Percentage Error (MAPE) - 31 Komoditas',
            xaxis_title='MAPE (%)',
            yaxis_title='Komoditas',
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Keterangan Warna:**
        - Hijau: MAPE < 10% (Sangat Baik)
        - Biru: MAPE 10-20% (Baik)
        - Oranye: MAPE 20-30% (Cukup)
        - Merah: MAPE > 30% (Perlu Perbaikan)
        """)
        
        # Comparison chart
        st.markdown("---")
        st.subheader("Perbandingan Metrik Evaluasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Bar(
                x=eval_sorted['Komoditas'],
                y=eval_sorted['RMSE_Test'],
                name='RMSE',
                marker_color='steelblue'
            ))
            fig_rmse.update_layout(
                title='Root Mean Squared Error (RMSE)',
                xaxis_title='Komoditas',
                yaxis_title='RMSE',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # MAE comparison
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(
                x=eval_sorted['Komoditas'],
                y=eval_sorted['MAE_Test'],
                name='MAE',
                marker_color='coral'
            ))
            fig_mae.update_layout(
                title='Mean Absolute Error (MAE)',
                xaxis_title='Komoditas',
                yaxis_title='MAE',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Download button
        st.markdown("---")
        csv = eval_df.to_csv(index=False)
        st.download_button(
            label="Download Hasil Evaluasi (CSV)",
            data=csv,
            file_name="hasil_evaluasi_model.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.error("Data evaluasi tidak tersedia. Pastikan file hasil_evaluasi.csv ada di repository.")

# ===================================================================
# FOOTER
# ===================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Prediksi Harga Pangan menggunakan LSTM | Data: 2020-2025 | Forecasting: 2025-2026</p>
</div>
""", unsafe_allow_html=True)
