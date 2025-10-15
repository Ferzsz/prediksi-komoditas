import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Prediksi Harga Pangan 2025-2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df_eval = pd.read_csv('hasil_evaluasi_lstm.csv')
    df_forecast = pd.read_csv('forecast_2025_2026.csv')
    return df_eval, df_forecast

df_eval, df_forecast = load_data()

# Fungsi kategori performa
def get_performance_category(mape):
    if mape < 5:
        return "Excellent", "#27ae60"
    elif mape < 10:
        return "Good", "#f39c12"
    elif mape < 20:
        return "Fair", "#e67e22"
    else:
        return "Poor", "#c0392b"

# Header
st.title("📊 Sistem Prediksi Harga Komoditas Pangan 2025-2026")
st.markdown("**Model:** LSTM dengan Log Transform & RobustScaler | **Dataset:** 31 Komoditas Pangan")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Evaluasi Model per Komoditas",
    "🔮 Prediksi Harga",
    "📋 Evaluasi Semua Komoditas"
])

# TAB 1: EVALUASI MODEL PER KOMODITAS
with tab1:
    st.header("Evaluasi Model per Komoditas")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_commodity = st.selectbox(
            "Pilih Komoditas:",
            options=df_eval['Komoditas'].tolist(),
            key="eval_commodity"
        )
    
    commodity_data = df_eval[df_eval['Komoditas'] == selected_commodity].iloc[0]
    
    rmse = commodity_data['RMSE']
    mae = commodity_data['MAE']
    mape = commodity_data['MAPE (%)']
    
    category, color = get_performance_category(mape)
    
    st.subheader(f"Hasil Evaluasi: {selected_commodity}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="RMSE (Rp)", value=f"{rmse:,.2f}")
    
    with col2:
        st.metric(label="MAE (Rp)", value=f"{mae:,.2f}")
    
    with col3:
        st.metric(label="MAPE (%)", value=f"{mape:.2f}%")
    
    with col4:
        st.markdown(f"**Kategori Performa**")
        st.markdown(f"<h2 style='color: {color};'>{category}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Interpretasi Score")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        **Kategori Performa Model:**
        - 🟢 **Excellent** (MAPE < 5%)
        - 🟡 **Good** (MAPE 5-10%)
        - 🟠 **Fair** (MAPE 10-20%)
        - 🔴 **Poor** (MAPE > 20%)
        
        **Model Anda:** <span style='color: {color}; font-weight: bold;'>{category}</span>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        **Penjelasan Metrik:**
        - **RMSE**: Rata-rata kesalahan prediksi. Semakin kecil semakin baik. 
          Nilai: **Rp {rmse:,.0f}**
        
        - **MAE**: Rata-rata selisih absolut prediksi dengan aktual. 
          Nilai: **Rp {mae:,.0f}**
        
        - **MAPE**: Persentase kesalahan prediksi (metrik utama). 
          Nilai: **{mape:.2f}%**
        """)
    
    st.markdown("---")
    st.subheader("Perbandingan dengan Komoditas Lain")
    
    df_sorted = df_eval.sort_values('MAPE (%)')
    colors = [get_performance_category(x)[1] for x in df_sorted['MAPE (%)']]
    highlight_colors = ['#3498db' if x == selected_commodity else c 
                       for x, c in zip(df_sorted['Komoditas'], colors)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted['Komoditas'],
        x=df_sorted['MAPE (%)'],
        orientation='h',
        marker=dict(color=highlight_colors),
        text=df_sorted['MAPE (%)'].round(2),
        textposition='outside',
        texttemplate='%{text}%'
    ))
    
    fig.update_layout(
        title=f"MAPE Semua Komoditas ({selected_commodity} ditandai biru)",
        xaxis_title="MAPE (%)",
        yaxis_title="",
        height=800,
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    fig.add_vline(x=5, line_dash="dash", line_color="#27ae60", 
                  annotation_text="Excellent", annotation_position="top")
    fig.add_vline(x=10, line_dash="dash", line_color="#f39c12", 
                  annotation_text="Good", annotation_position="top")
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: PREDIKSI HARGA
with tab2:
    st.header("Prediksi Harga Komoditas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        commodity_forecast = st.selectbox(
            "Pilih Komoditas:",
            options=df_forecast.columns[1:].tolist(),
            key="forecast_commodity"
        )
    
    with col2:
        year = st.selectbox("Pilih Tahun:", options=[2025, 2026], key="forecast_year")
    
    with col3:
        month = st.selectbox(
            "Pilih Bulan:",
            options=list(range(1, 13)),
            format_func=lambda x: [
                "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                "Juli", "Agustus", "September", "Oktober", "November", "Desember"
            ][x-1],
            key="forecast_month"
        )
    
    month_str = f"{year}-{month:02d}"
    
    try:
        predicted_price = df_forecast[df_forecast['Bulan'] == month_str][commodity_forecast].values[0]
        
        min_price = df_forecast[commodity_forecast].min()
        max_price = df_forecast[commodity_forecast].max()
        avg_price = df_forecast[commodity_forecast].mean()
        change_pct = ((df_forecast[commodity_forecast].iloc[-1] - df_forecast[commodity_forecast].iloc[0]) / df_forecast[commodity_forecast].iloc[0]) * 100
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label=f"Harga Prediksi - {month_str}", value=f"Rp {predicted_price:,.2f}")
        
        with col2:
            st.metric(label="Harga Minimum 2025-2026", value=f"Rp {min_price:,.2f}")
        
        with col3:
            st.metric(label="Harga Maximum 2025-2026", value=f"Rp {max_price:,.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Harga Rata-rata 2025-2026", value=f"Rp {avg_price:,.2f}")
        
        with col2:
            st.metric(
                label="Perubahan Harga (Jan 2025 - Des 2026)",
                value=f"{abs(change_pct):.2f}%",
                delta=f"{'Naik' if change_pct > 0 else 'Turun'}"
            )
        
        st.markdown("---")
        st.subheader(f"Tren Harga {commodity_forecast} (2025-2026)")
        
        forecast_data = df_forecast[['Bulan', commodity_forecast]].copy()
        forecast_data.columns = ['Bulan', 'Harga']
        forecast_data['Selected'] = forecast_data['Bulan'] == month_str
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Bulan'],
            y=forecast_data['Harga'],
            mode='lines+markers',
            name='Prediksi Harga',
            line=dict(color='#9B59B6', width=3),
            marker=dict(size=8)
        ))
        
        selected_data = forecast_data[forecast_data['Selected']]
        fig.add_trace(go.Scatter(
            x=selected_data['Bulan'],
            y=selected_data['Harga'],
            mode='markers',
            name='Bulan Dipilih',
            marker=dict(size=15, color='#e74c3c', symbol='star')
        ))
        
        fig.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Harga (Rp)",
            height=500,
            hovermode='x unified',
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Informasi Prediksi:**
        - Periode: {month_str}
        - Harga Prediksi: Rp {predicted_price:,.2f}
        - Posisi: {'Di atas' if predicted_price > avg_price else 'Di bawah'} harga rata-rata
        - Selisih dari rata-rata: Rp {abs(predicted_price - avg_price):,.2f}
        """)
        
    except IndexError:
        st.error(f"Data untuk bulan {month_str} tidak tersedia.")

# TAB 3: EVALUASI SEMUA KOMODITAS
with tab3:
    st.header("Evaluasi Model Semua Komoditas")
    
    st.subheader("Ringkasan Performa Model")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Rata-rata MAPE", value=f"{df_eval['MAPE (%)'].mean():.2f}%")
    
    with col2:
        excellent_count = len(df_eval[df_eval['MAPE (%)'] < 5])
        st.metric(label="Komoditas Excellent", value=f"{excellent_count}/31")
    
    with col3:
        good_count = len(df_eval[df_eval['MAPE (%)'] < 10])
        st.metric(label="Komoditas Good+", value=f"{good_count}/31")
    
    with col4:
        fair_count = len(df_eval[df_eval['MAPE (%)'] < 20])
        st.metric(label="Komoditas Fair+", value=f"{fair_count}/31")
    
    st.markdown("---")
    st.subheader("Tabel Evaluasi Lengkap")
    
    df_display = df_eval.copy()
    df_display['Kategori'] = df_display['MAPE (%)'].apply(
        lambda x: get_performance_category(x)[0]
    )
    
    df_display = df_display.sort_values('MAPE (%)')
    
    df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"Rp {x:,.2f}")
    df_display['MAE'] = df_display['MAE'].apply(lambda x: f"Rp {x:,.2f}")
    df_display['MAPE (%)'] = df_display['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(df_display, use_container_width=True, height=600, hide_index=True)
    
    st.markdown("---")
    st.subheader("Visualisasi 3 Metrik Evaluasi")
    
    metric_choice = st.selectbox(
        "Pilih Metrik untuk Divisualisasikan:",
        options=['RMSE', 'MAE', 'MAPE (%)'],
        key="metric_visualization"
    )
    
    df_sorted = df_eval.sort_values(metric_choice)
    
    if metric_choice == 'MAPE (%)':
        colors = [get_performance_category(x)[1] for x in df_sorted['MAPE (%)']]
    else:
        colors = '#3498db'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted['Komoditas'],
        x=df_sorted[metric_choice],
        orientation='h',
        marker=dict(color=colors),
        text=df_sorted[metric_choice].round(2),
        textposition='outside'
    ))
    
    unit = "%" if metric_choice == 'MAPE (%)' else "Rp"
    
    fig.update_layout(
        title=f"{metric_choice} untuk Semua Komoditas",
        xaxis_title=f"{metric_choice} ({unit})",
        yaxis_title="",
        height=800,
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    if metric_choice == 'MAPE (%)':
        fig.add_vline(x=5, line_dash="dash", line_color="#27ae60")
        fig.add_vline(x=10, line_dash="dash", line_color="#f39c12")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Download Data Evaluasi")
    
    csv = df_eval.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="evaluasi_model_lstm.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sistem Prediksi Harga Komoditas Pangan 2025-2026</p>
    <p>Model: LSTM dengan Log Transform & RobustScaler | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)

