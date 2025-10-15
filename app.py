import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px

# ===========================================================================================
# KONFIGURASI HALAMAN
# ===========================================================================================

st.set_page_config(
    page_title="LSTM Evaluasi Model - Prediksi Harga Pangan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling modern
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================================================================================
# SIDEBAR - INFORMASI MODEL
# ===========================================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/combo-chart.png", width=80)
    st.title("📊 Model LSTM")
    st.markdown("---")
    
    st.markdown("### 🔧 Informasi Model")
    st.info("""
    **Arsitektur:**
    - Bidirectional LSTM (64 units)
    - LSTM (32 units)
    - Dense Layers (32→16)
    - Dropout & Batch Normalization
    
    **Hyperparameters:**
    - Time Steps: 15
    - Batch Size: 16
    - Learning Rate: 0.0005
    - Loss: Huber
    - Optimizer: Adam
    
    **Preprocessing:**
    - Detrending (Linear)
    - MinMax Normalization
    - Polynomial Interpolation
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Training Info")
    st.success("""
    - Max Epochs: 150
    - Early Stopping: ✓
    - Data Split: 85/15
    - Total Komoditas: 31
    """)

# ===========================================================================================
# HEADER
# ===========================================================================================

st.title("🎯 Evaluasi Model LSTM - Prediksi Harga Pangan")
st.markdown("### Dashboard Analisis Performa Model untuk 31 Komoditas")
st.markdown("---")

# ===========================================================================================
# LOAD DATA
# ===========================================================================================

@st.cache_data
def load_evaluation_results():
    """Load hasil evaluasi dari CSV"""
    try:
        df = pd.read_csv('hasil_evaluasi_lstm_enhanced.csv')
        return df
    except FileNotFoundError:
        st.error("❌ File 'hasil_evaluasi_lstm_enhanced.csv' tidak ditemukan!")
        return None

df_results = load_evaluation_results()

if df_results is not None:
    
    # ===========================================================================================
    # OVERVIEW METRICS
    # ===========================================================================================
    
    st.markdown("## 📊 Overview Performa Model")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_mape = df_results['MAPE (%)'].mean()
        st.metric(
            label="📉 MAPE Rata-rata",
            value=f"{avg_mape:.2f}%",
            delta=f"{len(df_results[df_results['MAPE (%)'] < 5])} excellent"
        )
    
    with col2:
        avg_mae = df_results['MAE'].mean()
        st.metric(
            label="💰 MAE Rata-rata",
            value=f"Rp {avg_mae:,.0f}",
            delta=None
        )
    
    with col3:
        avg_rmse = df_results['RMSE'].mean()
        st.metric(
            label="📐 RMSE Rata-rata",
            value=f"Rp {avg_rmse:,.0f}",
            delta=None
        )
    
    with col4:
        excellent_count = len(df_results[df_results['MAPE (%)'] < 5])
        st.metric(
            label="⭐ MAPE < 5%",
            value=f"{excellent_count}",
            delta="Excellent"
        )
    
    with col5:
        good_count = len(df_results[df_results['MAPE (%)'] < 10])
        st.metric(
            label="✅ MAPE < 10%",
            value=f"{good_count}",
            delta="Good"
        )
    
    st.markdown("---")
    
    # ===========================================================================================
    # TABEL HASIL EVALUASI
    # ===========================================================================================
    
    st.markdown("## 📋 Hasil Evaluasi Lengkap")
    
    # Filter berdasarkan kategori MAPE
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_option = st.selectbox(
            "Filter Kategori MAPE:",
            ["Semua", "Excellent (<5%)", "Good (<10%)", "Fair (<20%)", "Poor (≥20%)"]
        )
    
    # Apply filter
    if filter_option == "Excellent (<5%)":
        df_filtered = df_results[df_results['MAPE (%)'] < 5].copy()
    elif filter_option == "Good (<10%)":
        df_filtered = df_results[df_results['MAPE (%)'] < 10].copy()
    elif filter_option == "Fair (<20%)":
        df_filtered = df_results[(df_results['MAPE (%)'] >= 10) & (df_results['MAPE (%)'] < 20)].copy()
    elif filter_option == "Poor (≥20%)":
        df_filtered = df_results[df_results['MAPE (%)'] >= 20].copy()
    else:
        df_filtered = df_results.copy()
    
    # Sort berdasarkan MAPE
    df_filtered = df_filtered.sort_values('MAPE (%)')
    
    # Add color coding for MAPE
    def color_mape(val):
        if val < 5:
            color = '#27ae60'  # Green
        elif val < 10:
            color = '#f39c12'  # Orange
        elif val < 20:
            color = '#e67e22'  # Dark Orange
        else:
            color = '#c0392b'  # Red
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    # Format table
    styled_df = df_filtered.style.format({
        'RMSE': 'Rp {:,.2f}',
        'MAE': 'Rp {:,.2f}',
        'MAPE (%)': '{:.2f}%'
    }).applymap(color_mape, subset=['MAPE (%)'])
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.markdown(f"**Menampilkan {len(df_filtered)} dari {len(df_results)} komoditas**")
    
    st.markdown("---")
    
    # ===========================================================================================
    # VISUALISASI METRIK
    # ===========================================================================================
    
    st.markdown("## 📊 Visualisasi Performa Metrik")
    
    tab1, tab2, tab3 = st.tabs(["📉 MAPE", "💰 MAE", "📐 RMSE"])
    
    # Sort untuk visualisasi
    df_sorted_mape = df_results.sort_values('MAPE (%)')
    df_sorted_mae = df_results.sort_values('MAE')
    df_sorted_rmse = df_results.sort_values('RMSE')
    
    with tab1:
        # MAPE Bar Chart
        colors = ['#27ae60' if x < 5 else '#f39c12' if x < 10 else '#e67e22' if x < 20 else '#c0392b' 
                  for x in df_sorted_mape['MAPE (%)']]
        
        fig_mape = go.Figure(data=[
            go.Bar(
                x=df_sorted_mape['MAPE (%)'],
                y=df_sorted_mape['Komoditas'],
                orientation='h',
                marker=dict(color=colors),
                text=df_sorted_mape['MAPE (%)'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside',
            )
        ])
        
        fig_mape.add_vline(x=5, line_dash="dash", line_color="#27ae60", 
                           annotation_text="Excellent (5%)", annotation_position="top right")
        fig_mape.add_vline(x=10, line_dash="dash", line_color="#f39c12", 
                           annotation_text="Good (10%)", annotation_position="top right")
        fig_mape.add_vline(x=20, line_dash="dash", line_color="#e67e22", 
                           annotation_text="Fair (20%)", annotation_position="top right")
        
        fig_mape.update_layout(
            title="Mean Absolute Percentage Error (MAPE) - Semua Komoditas",
            xaxis_title="MAPE (%)",
            yaxis_title="Komoditas",
            height=900,
            showlegend=False,
            plot_bgcolor='#f8f9fa'
        )
        
        st.plotly_chart(fig_mape, use_container_width=True)
    
    with tab2:
        # MAE Bar Chart
        fig_mae = go.Figure(data=[
            go.Bar(
                x=df_sorted_mae['MAE'],
                y=df_sorted_mae['Komoditas'],
                orientation='h',
                marker=dict(color='#e74c3c'),
                text=df_sorted_mae['MAE'].apply(lambda x: f'Rp {x:,.0f}'),
                textposition='outside',
            )
        ])
        
        fig_mae.update_layout(
            title="Mean Absolute Error (MAE) - Semua Komoditas",
            xaxis_title="MAE (Rp)",
            yaxis_title="Komoditas",
            height=900,
            showlegend=False,
            plot_bgcolor='#f8f9fa'
        )
        
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with tab3:
        # RMSE Bar Chart
        fig_rmse = go.Figure(data=[
            go.Bar(
                x=df_sorted_rmse['RMSE'],
                y=df_sorted_rmse['Komoditas'],
                orientation='h',
                marker=dict(color='#3498db'),
                text=df_sorted_rmse['RMSE'].apply(lambda x: f'Rp {x:,.0f}'),
                textposition='outside',
            )
        ])
        
        fig_rmse.update_layout(
            title="Root Mean Squared Error (RMSE) - Semua Komoditas",
            xaxis_title="RMSE (Rp)",
            yaxis_title="Komoditas",
            height=900,
            showlegend=False,
            plot_bgcolor='#f8f9fa'
        )
        
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    st.markdown("---")
    
    # ===========================================================================================
    # DISTRIBUSI PERFORMA
    # ===========================================================================================
    
    st.markdown("## 📈 Distribusi Performa Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart kategori MAPE
        categories = ['Excellent (<5%)', 'Good (5-10%)', 'Fair (10-20%)', 'Poor (≥20%)']
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
        
        fig_pie.update_layout(
            title="Distribusi Kategori MAPE",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Histogram MAPE
        fig_hist = go.Figure(data=[go.Histogram(
            x=df_results['MAPE (%)'],
            nbinsx=15,
            marker=dict(color='#3498db', line=dict(color='white', width=1))
        )])
        
        fig_hist.update_layout(
            title="Distribusi Nilai MAPE",
            xaxis_title="MAPE (%)",
            yaxis_title="Jumlah Komoditas",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # ===========================================================================================
    # DETAIL KOMODITAS INDIVIDUAL
    # ===========================================================================================
    
    st.markdown("## 🔍 Analisis Detail per Komoditas")
    
    selected_komoditas = st.selectbox(
        "Pilih Komoditas untuk Analisis Detail:",
        df_results.sort_values('MAPE (%)['Komoditas'].tolist()
    )
    
    if selected_komoditas:
        komoditas_data = df_results[df_results['Komoditas'] == selected_komoditas].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mape_val = komoditas_data['MAPE (%)']
            if mape_val < 5:
                status = "⭐ Excellent"
                color = "#27ae60"
            elif mape_val < 10:
                status = "✅ Good"
                color = "#f39c12"
            elif mape_val < 20:
                status = "⚠️ Fair"
                color = "#e67e22"
            else:
                status = "❌ Poor"
                color = "#c0392b"
            
            st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; color:white;'>"
                       f"<h2 style='color:white; margin:0;'>{selected_komoditas}</h2>"
                       f"<h3 style='color:white; margin:10px 0 0 0;'>{status}</h3></div>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("MAPE", f"{komoditas_data['MAPE (%)']:.2f}%")
            st.metric("MAE", f"Rp {komoditas_data['MAE']:,.2f}")
        
        with col3:
            st.metric("RMSE", f"Rp {komoditas_data['RMSE']:,.2f}")
            rank = df_results.sort_values('MAPE (%)').reset_index(drop=True).index[
                df_results.sort_values('MAPE (%)['Komoditas'] == selected_komoditas
            ].tolist()[0] + 1
            st.metric("Ranking", f"#{rank} dari 31")
    
    st.markdown("---")
    
    # ===========================================================================================
    # DOWNLOAD SECTION
    # ===========================================================================================
    
    st.markdown("## 💾 Download Hasil Evaluasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='hasil_evaluasi_lstm_enhanced.csv',
            mime='text/csv',
        )
    
    with col2:
        # Download filtered data
        if filter_option != "Semua":
            csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {filter_option} CSV",
                data=csv_filtered,
                file_name=f'hasil_evaluasi_{filter_option.replace(" ", "_")}.csv',
                mime='text/csv',
            )

# ===========================================================================================
# FOOTER
# ===========================================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>🎓 Evaluasi Model LSTM - Prediksi Harga Pangan</strong></p>
    <p>Enhanced Model with Detrending & Optimized Hyperparameters</p>
    <p style='font-size: 0.9em;'>Built with Streamlit | October 2025</p>
</div>
""", unsafe_allow_html=True)
