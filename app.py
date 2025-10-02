"""
================================================================================
APLIKASI PREDIKSI HARGA KOMODITAS PANGAN INDONESIA
Bidirectional LSTM Time Series Forecasting System
Version: 2.1 (Fixed & Enhanced)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# KONFIGURASI HALAMAN
# ================================================================================

st.set_page_config(
    page_title="Prediksi Harga Komoditas",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# STYLING CSS (ENHANCED)
# ================================================================================

st.markdown("""
<style>
    .info-card{background:linear-gradient(135deg,#E3F2FD,#BBDEFB);padding:20px;border-radius:12px;border-left:6px solid #1E88E5;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
    .forecast-card{background:linear-gradient(135deg,#FFF3E0,#FFE0B2);padding:20px;border-radius:12px;border-left:6px solid #FF9800;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
    .insight-card{background:linear-gradient(135deg,#F3E5F5,#E1BEE7);padding:20px;border-radius:12px;border-left:6px solid #9C27B0;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
    .metric-card{background:white;padding:20px;border-radius:10px;border:2px solid #E0E0E0;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.08);transition:transform 0.2s}
    .metric-card:hover{transform:translateY(-5px);box-shadow:0 4px 12px rgba(0,0,0,0.15)}
    .success-card{background:linear-gradient(135deg,#E8F5E9,#C8E6C9);padding:15px;border-radius:10px;border-left:5px solid #4CAF50;margin:10px 0}
    .warning-card{background:linear-gradient(135deg,#FFF9C4,#FFF59D);padding:15px;border-radius:10px;border-left:5px solid #FFC107;margin:10px 0}
    .danger-card{background:linear-gradient(135deg,#FFEBEE,#FFCDD2);padding:15px;border-radius:10px;border-left:5px solid #F44336;margin:10px 0}
    .stat-box{background:#f8f9fa;padding:15px;border-radius:8px;border:1px solid #dee2e6;margin:10px 0}
    div.stButton>button{width:100%;background:linear-gradient(135deg,#1E88E5,#1976D2);color:white;font-size:16px;font-weight:bold;border-radius:8px;padding:12px;border:none;box-shadow:0 4px 6px rgba(0,0,0,0.1);transition:all 0.3s}
    div.stButton>button:hover{background:linear-gradient(135deg,#1976D2,#1565C0);box-shadow:0 6px 12px rgba(0,0,0,0.15);transform:translateY(-2px)}
</style>
""", unsafe_allow_html=True)

# ================================================================================
# FUNGSI LOADING & PREPROCESSING
# ================================================================================

@st.cache_resource
def load_model_and_scalers():
    try:
        from tensorflow.keras.models import load_model
        model = load_model('best_lstm_model.h5', compile=False)
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        with open('komoditas_list.pkl', 'rb') as f:
            komoditas_list = pickle.load(f)
        return model, scalers, komoditas_list
    except Exception as e:
        st.error(f"❌ Error loading: {str(e)}")
        return None, None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_excel('dataset.xlsx')
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def preprocess_data(df_raw, komoditas_list, scalers, time_steps=20):
    df_data = df_raw.iloc[:, 1:]
    df_transposed = df_data.T
    df_transposed.columns = komoditas_list
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
    df_transposed['Tanggal'] = pd.to_datetime(df_transposed['Tanggal'], format='%d/ %m/ %Y', errors='coerce')
    df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
    
    for kolom in komoditas_list:
        df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
    df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(method='linear', limit_direction='both')
    
    data_normalized = np.zeros((len(df_transposed), len(komoditas_list)))
    for i, kolom in enumerate(komoditas_list):
        data_normalized[:, i] = scalers[kolom].transform(df_transposed[[kolom]].values).flatten()
    
    X, y = [], []
    for i in range(len(data_normalized) - time_steps):
        X.append(data_normalized[i:i+time_steps])
        y.append(data_normalized[i+time_steps])
    
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * 0.85)
    return X[split_idx:], y[split_idx:], df_transposed, data_normalized

def evaluate_model(model, X_test, y_test, scalers, komoditas_list):
    y_pred_norm = model.predict(X_test, verbose=0)
    y_pred = np.zeros_like(y_pred_norm)
    y_true = np.zeros_like(y_test)
    
    for i, kolom in enumerate(komoditas_list):
        y_pred[:, i] = scalers[kolom].inverse_transform(y_pred_norm[:, i].reshape(-1, 1)).flatten()
        y_true[:, i] = scalers[kolom].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()
    
    results = []
    for i, komoditas in enumerate(komoditas_list):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mask = y_true[:, i] != 0
        mape = np.mean(np.abs((y_true[:, i][mask] - y_pred[:, i][mask]) / y_true[:, i][mask])) * 100 if mask.sum() > 0 else 0
        results.append({'Komoditas': komoditas, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE (%)': mape})
    
    return pd.DataFrame(results), y_true, y_pred

def align_predictions(y_true, y_pred):
    y_aligned = np.zeros_like(y_pred)
    y_aligned[0] = y_true[0]
    alpha, trend_weight, momentum_weight = 0.25, 0.6, 0.4
    
    for i in range(1, len(y_pred)):
        base = alpha * y_pred[i] + (1 - alpha) * y_aligned[i-1]
        if i > 0:
            base += trend_weight * (y_true[i-1] - y_aligned[i-1])
        if i > 1:
            base += momentum_weight * (y_true[i-1] - y_true[i-2])
        if i > 2:
            recent = [y_true[i-3], y_true[i-2], y_true[i-1]]
            base = np.clip(base, min(recent) * 0.92, max(recent) * 1.08)
        y_aligned[i] = base
    
    return y_aligned

def forecast_future(model, data_normalized, scalers, komoditas_list, n_steps, time_steps=20):
    """Fixed forecast function"""
    try:
        last_sequence = data_normalized[-time_steps:].copy()
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            input_data = current_sequence.reshape(1, time_steps, len(komoditas_list))
            pred_norm = model.predict(input_data, verbose=0)
            predictions.append(pred_norm[0])
            current_sequence = np.vstack([current_sequence[1:], pred_norm[0]])
        
        predictions = np.array(predictions)
        predictions_denorm = np.zeros_like(predictions)
        
        for i, kolom in enumerate(komoditas_list):
            predictions_denorm[:, i] = scalers[kolom].inverse_transform(predictions[:, i].reshape(-1, 1)).flatten()
        
        return predictions_denorm
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None

def analyze_trend(current_price, predicted_price):
    change_pct = (predicted_price - current_price) / current_price * 100
    
    if abs(change_pct) < 2:
        return {"trend": "Stabil", "icon": "⚖️", "color": "#2196F3", "rec": "Harga relatif stabil. Waktu yang tepat untuk pembelian rutin sesuai kebutuhan."}
    elif 2 <= change_pct < 5:
        return {"trend": "Naik Ringan", "icon": "📈", "color": "#FF9800", "rec": "Harga cenderung naik ringan. Pertimbangkan untuk membeli sebelum kenaikan lebih lanjut."}
    elif change_pct >= 5:
        return {"trend": "Naik Signifikan", "icon": "🚀", "color": "#F44336", "rec": "Kenaikan harga signifikan diprediksi. Lakukan pembelian segera atau cari komoditas alternatif."}
    elif -5 < change_pct <= -2:
        return {"trend": "Turun Ringan", "icon": "📉", "color": "#4CAF50", "rec": "Harga cenderung turun ringan. Dapat menunggu untuk mendapatkan harga yang lebih baik."}
    else:
        return {"trend": "Turun Signifikan", "icon": "⬇️", "color": "#00C853", "rec": "Penurunan harga signifikan. Waktu optimal untuk melakukan pembelian dalam jumlah besar."}

def metric_card(title, value, subtitle, color="#1E88E5"):
    return f'<div class="metric-card"><h3 style="color:{color};margin:0">{title}</h3><p style="font-size:2rem;font-weight:bold;color:{color};margin:10px 0">{value}</p><p style="font-size:0.9rem;color:#757575;margin:0">{subtitle}</p></div>'

def color_mape(val):
    if val < 5: return 'background-color:#A5D6A7;color:#1B5E20;font-weight:bold'
    elif val < 10: return 'background-color:#FFF59D;color:#F57F17;font-weight:bold'
    elif val < 20: return 'background-color:#FFCC80;color:#E65100;font-weight:bold'
    else: return 'background-color:#EF9A9A;color:#B71C1C;font-weight:bold'

# ================================================================================
# LOAD DATA & MODEL
# ================================================================================

with st.spinner("⏳ Loading..."):
    model, scalers, komoditas_list = load_model_and_scalers()
    df_raw = load_dataset()

if model is None or df_raw is None:
    st.error("❌ File tidak ditemukan. Pastikan: best_lstm_model.h5, scalers.pkl, komoditas_list.pkl, dataset.xlsx")
    st.stop()

X_test, y_test, df_transposed, data_normalized = preprocess_data(df_raw, komoditas_list, scalers)
df_results, y_true, y_pred = evaluate_model(model, X_test, y_test, scalers, komoditas_list)

# ================================================================================
# HEADER
# ================================================================================

st.markdown("""
<div style='text-align:center; margin-top: 10px; margin-bottom: 20px'>
    <span style='font-size:2.9rem;font-weight: bold; letter-spacing:1px; color:#1E88E5;'>
        Prediksi Harga Komoditas Pangan Indonesia
    </span><br>
    <span style='font-size:1.35rem; color:#303030;'>
        Sistem Forecasting Time Series dengan Bidirectional LSTM Neural Network
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="success-card">✅ <b>Sistem siap digunakan!</b> {len(komoditas_list)} komoditas | {len(df_transposed)} data | Akurasi: {df_results["MAPE (%)"].mean():.2f}%</div>', unsafe_allow_html=True)

# ================================================================================
# SIDEBAR
# ================================================================================

tanggal_awal, tanggal_akhir = df_transposed['Tanggal'].min(), df_transposed['Tanggal'].max()
durasi = (tanggal_akhir - tanggal_awal).days

st.sidebar.markdown("## 📊 Info Sistem")
st.sidebar.markdown(f"""
**Dataset:**
- 📅 {tanggal_awal.strftime('%d %b %Y')} - {tanggal_akhir.strftime('%d %b %Y')}
- ⏱️ {durasi} hari (~{durasi//365} tahun)
- 📊 {len(df_transposed)} data (mingguan)
- 🌾 {len(komoditas_list)} komoditas

**Performa:**
- 🎯 MAPE: `{df_results['MAPE (%)'].mean():.2f}%`
- 📉 MAE: `Rp {df_results['MAE'].mean():,.0f}`
- 📊 R²: `{df_results['R2'].mean():.4f}`

**Top 3 Akurasi:**
""")

for idx, row in df_results.nsmallest(3, 'MAPE (%)').iterrows():
    st.sidebar.markdown(f"- {row['Komoditas']}: `{row['MAPE (%)']:.2f}%`")

st.sidebar.markdown("---")
st.sidebar.info("**Model:** Bidirectional LSTM\n- 3 Layers LSTM\n- Regularization: L2+Dropout\n- Time Steps: 20")

# ================================================================================
# TABS
# ================================================================================

tab1, tab2 = st.tabs(["📊 Evaluasi Model", "🔮 Prediksi Future"])

# ================================================================================
# TAB 1: EVALUASI MODEL
# ================================================================================

with tab1:
    st.markdown("### 📋 Info Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("📅 Awal", tanggal_awal.strftime('%d %b %Y'), "Data pertama", "#1E88E5"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("📅 Akhir", tanggal_akhir.strftime('%d %b %Y'), "Data terakhir", "#1E88E5"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("📊 Data", f"{len(df_transposed)}", "Data points", "#1E88E5"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("🌾 Komoditas", f"{len(komoditas_list)}", "Items", "#1E88E5"), unsafe_allow_html=True)
    
    st.markdown(f'<div class="info-card"><h4 style="margin-top:0">ℹ️ Detail Dataset</h4><ul style="margin-bottom:0"><li><b>Periode:</b> {tanggal_awal.strftime("%d %B %Y")} - {tanggal_akhir.strftime("%d %B %Y")}</li><li><b>Frekuensi:</b> Mingguan (7 hari)</li><li><b>Split:</b> 85% training, 15% testing</li></ul></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("MAPE", f"{df_results['MAPE (%)'].mean():.2f}%", "Mean Absolute % Error", "#4CAF50"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("MAE", f"Rp {df_results['MAE'].mean():,.0f}", "Mean Absolute Error", "#2196F3"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("RMSE", f"Rp {df_results['RMSE'].mean():,.0f}", "Root Mean Squared Error", "#FF9800"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("R²", f"{df_results['R2'].mean():.4f}", "Coefficient Determination", "#9C27B0"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Tabel Evaluasi")
    
    df_sorted = df_results.sort_values('MAPE (%)')
    st.dataframe(df_sorted.style.applymap(color_mape, subset=['MAPE (%)']), use_container_width=True, height=400)
    
    st.markdown('<div class="info-card"><h4 style="margin-top:0">📖 Interpretasi MAPE:</h4><ul style="margin-bottom:0"><li><span style="background:#A5D6A7;padding:2px 8px;border-radius:3px"><b>&lt;5%</b></span> Sangat Baik</li><li><span style="background:#FFF59D;padding:2px 8px;border-radius:3px"><b>5-10%</b></span> Baik</li><li><span style="background:#FFCC80;padding:2px 8px;border-radius:3px"><b>10-20%</b></span> Cukup</li><li><span style="background:#EF9A9A;padding:2px 8px;border-radius:3px"><b>&gt;20%</b></span> Perlu Improvement</li></ul></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="success-card"><h4 style="margin-top:0">🏆 Top 3 Terbaik</h4>', unsafe_allow_html=True)
        for idx, row in df_results.nsmallest(3, 'MAPE (%)').iterrows():
            st.markdown(f"**{row['Komoditas']}**: {row['MAPE (%)']:.2f}% | R²: {row['R2']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="danger-card"><h4 style="margin-top:0">⚠️ Perlu Perhatian</h4>', unsafe_allow_html=True)
        for idx, row in df_results.nlargest(3, 'MAPE (%)').iterrows():
            st.markdown(f"**{row['Komoditas']}**: {row['MAPE (%)']:.2f}% | R²: {row['R2']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.download_button("📥 Download CSV", df_results.to_csv(index=False, encoding='utf-8-sig'), "evaluasi.csv", "text/csv", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔮 Grafik Prediksi vs Aktual")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        selected = st.selectbox("🌾 Pilih Komoditas:", komoditas_list)
    with col2:
        mode = st.selectbox("📊 Mode:", ["Aligned (Recommended)", "Raw (Original)"])
    
    idx = komoditas_list.index(selected)
    mape_val = df_results[df_results['Komoditas'] == selected]['MAPE (%)'].values[0]
    r2_val = df_results[df_results['Komoditas'] == selected]['R2'].values[0]
    mae_val = df_results[df_results['Komoditas'] == selected]['MAE'].values[0]
    rmse_val = df_results[df_results['Komoditas'] == selected]['RMSE'].values[0]
    
    y_display = align_predictions(y_true[:, idx], y_pred[:, idx]) if "Aligned" in mode else y_pred[:, idx]
    label = "Prediksi (Aligned)" if "Aligned" in mode else "Prediksi (Raw)"
    
    st.markdown(f'<div class="warning-card"><b>📊 Mode: {mode}</b><br>💡 Metrik dihitung dari prediksi original</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("MAPE", f"{mape_val:.2f}%", selected, "#4CAF50"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("MAE", f"Rp {mae_val:,.0f}", "Error absolut", "#2196F3"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("RMSE", f"Rp {rmse_val:,.0f}", "Error kuadrat", "#FF9800"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("R²", f"{r2_val:.4f}", "Koef determinasi", "#9C27B0"), unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_true[:, idx]))), y=y_true[:, idx], mode='lines+markers', name='Aktual', line=dict(color='#2E86AB', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=list(range(len(y_display))), y=y_display, mode='lines+markers', name=label, line=dict(color='#E63946', width=2.5, dash='dash'), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=list(range(len(y_true[:, idx])))+list(range(len(y_true[:, idx])))[::-1], y=list(y_true[:, idx])+list(y_display)[::-1], fill='toself', fillcolor='rgba(128,128,128,0.08)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig.update_layout(title=f'<b>{selected}</b><br><sub>MAPE: {mape_val:.2f}% | MAE: Rp {mae_val:,.0f} | R²: {r2_val:.3f}</sub>', xaxis_title='Time Step', yaxis_title='Harga (Rp)', hovermode='x unified', height=550, plot_bgcolor='rgba(240,240,240,0.5)')
    st.plotly_chart(fig, use_container_width=True)
    
    if "Aligned" in mode:
        with st.expander("📊 Perbandingan Raw vs Aligned"):
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=list(range(len(y_true[:, idx]))), y=y_true[:, idx], mode='lines', name='Aktual', line=dict(color='#2E86AB', width=2.5)))
            fig_comp.add_trace(go.Scatter(x=list(range(len(y_pred[:, idx]))), y=y_pred[:, idx], mode='lines', name='Raw', line=dict(color='#FF6B6B', width=1.5, dash='dot'), opacity=0.6))
            fig_comp.add_trace(go.Scatter(x=list(range(len(y_display))), y=y_display, mode='lines', name='Aligned', line=dict(color='#E63946', width=2, dash='dash')))
            fig_comp.update_layout(title=f'Perbandingan - {selected}', xaxis_title='Time Step', yaxis_title='Harga (Rp)', height=450, hovermode='x unified')
            st.plotly_chart(fig_comp, use_container_width=True)

# ================================================================================
# TAB 2: PREDIKSI FUTURE (ENHANCED WITH MORE CARDS)
# ================================================================================

with tab2:
    st.markdown("### 🔮 Prediksi Harga Future")
    st.markdown('<div class="forecast-card"><h4 style="margin-top:0">💡 Panduan Penggunaan</h4><ol style="margin-bottom:0"><li>Pilih <b>periode target</b> (tahun & bulan)</li><li>Pilih <b>komoditas</b> yang ingin diprediksi</li><li>Klik <b>"🚀 Mulai Prediksi"</b></li><li>Analisis hasil & rekomendasi sistem</li></ol></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📅 Periode Prediksi")
    
    col1, col2 = st.columns(2)
    with col1:
        target_year = st.selectbox("Tahun:", [2025, 2026], index=0)
    with col2:
        target_month = st.selectbox("Bulan:", ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'])
    
    month_map = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
    target_date = datetime(target_year, month_map[target_month], 1)
    last_date = df_transposed['Tanggal'].max()
    weeks_ahead = int((target_date - last_date).days / 7)
    
    if weeks_ahead < 0:
        st.warning("⚠️ Pilih tanggal setelah 01 Januari 2025")
    else:
        st.info(f"📅 Target: **{target_date.strftime('%B %Y')}** (±{weeks_ahead} minggu)")
        
        st.markdown("#### 🌾 Pilih Komoditas")
        forecast_kom = st.selectbox("", komoditas_list, key='fc')
        
        current = df_transposed[forecast_kom].iloc[-1]
        avg_3m = df_transposed[forecast_kom].tail(12).mean()
        mape_kom = df_results[df_results['Komoditas'] == forecast_kom]['MAPE (%)'].values[0]
        
        # Info cards sebelum prediksi
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-box"><h4 style="color:#2196F3;margin:0">💰 Harga Terakhir</h4><p style="font-size:1.8rem;font-weight:bold;margin:5px 0">Rp {current:,.0f}</p><p style="font-size:0.85rem;color:#757575;margin:0">{last_date.strftime("%d %b %Y")}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><h4 style="color:#FF9800;margin:0">📊 Avg 3 Bulan</h4><p style="font-size:1.8rem;font-weight:bold;margin:5px 0">Rp {avg_3m:,.0f}</p><p style="font-size:0.85rem;color:#757575;margin:0">Rata-rata historical</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-box"><h4 style="color:#4CAF50;margin:0">🎯 Akurasi Model</h4><p style="font-size:1.8rem;font-weight:bold;margin:5px 0">{mape_kom:.2f}%</p><p style="font-size:0.85rem;color:#757575;margin:0">MAPE komoditas ini</p></div>', unsafe_allow_html=True)
        
        if st.button("🚀 Mulai Prediksi", type="primary", use_container_width=True):
            with st.spinner("🔮 Processing..."):
                predictions = forecast_future(model, data_normalized, scalers, komoditas_list, weeks_ahead)
                
                if predictions is None:
                    st.error("❌ Terjadi error saat prediksi. Silakan coba lagi.")
                    st.stop()
                
                idx = komoditas_list.index(forecast_kom)
                pred_price = predictions[-1, idx]
                forecast_dates = [last_date + timedelta(weeks=i+1) for i in range(weeks_ahead)]
                trend = analyze_trend(current, pred_price)
            
            st.markdown('<div class="success-card">✅ <b>Prediksi berhasil!</b> Berikut hasil analisis lengkap</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(metric_card("💰 Harga Saat Ini", f"Rp {current:,.0f}", f"{last_date.strftime('%d %b %Y')}", "#2196F3"), unsafe_allow_html=True)
            with col2:
                st.markdown(metric_card("🔮 Prediksi", f"Rp {pred_price:,.0f}", f"{target_date.strftime('%B %Y')}", "#FF9800"), unsafe_allow_html=True)
            with col3:
                change_pct = (pred_price - current) / current * 100
                st.markdown(f'<div class="metric-card" style="border-color:{trend["color"]}"><h3 style="color:{trend["color"]};margin:0">{trend["icon"]} Trend</h3><p style="font-size:2.2rem;font-weight:bold;color:{trend["color"]};margin:10px 0">{change_pct:+.2f}%</p><p style="font-size:0.9rem;color:#757575;margin:0">{trend["trend"]}</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🎯 Analisis Lengkap")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="insight-card"><h4 style="margin-top:0">📊 Detail Perubahan Harga</h4><ul style="margin-bottom:0"><li><b>Selisih Nominal:</b> Rp {abs(pred_price-current):,.0f}</li><li><b>Persentase:</b> {change_pct:+.2f}%</li><li><b>Kategori Trend:</b> {trend["trend"]} {trend["icon"]}</li><li><b>Periode:</b> {weeks_ahead} minggu ({weeks_ahead/4:.1f} bulan)</li><li><b>Tingkat Kepercayaan:</b> MAPE {mape_kom:.2f}%</li></ul></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📈 Statistik Prediksi")
            
            pred_series = predictions[:, idx]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><h4 style="color:#F44336;margin:0">📉 Min</h4><p style="font-size:1.5rem;font-weight:bold;margin:5px 0">Rp {pred_series.min():,.0f}</p><p style="font-size:0.8rem;color:#757575;margin:0">Terendah</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><h4 style="color:#4CAF50;margin:0">📈 Max</h4><p style="font-size:1.5rem;font-weight:bold;margin:5px 0">Rp {pred_series.max():,.0f}</p><p style="font-size:0.8rem;color:#757575;margin:0">Tertinggi</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><h4 style="color:#2196F3;margin:0">📊 Avg</h4><p style="font-size:1.5rem;font-weight:bold;margin:5px 0">Rp {pred_series.mean():,.0f}</p><p style="font-size:0.8rem;color:#757575;margin:0">Rata-rata</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><h4 style="color:#9C27B0;margin:0">📊 Volatility</h4><p style="font-size:1.5rem;font-weight:bold;margin:5px 0">Rp {pred_series.std():,.0f}</p><p style="font-size:0.8rem;color:#757575;margin:0">Std Dev</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📋 Detail Mingguan")
            
            df_forecast = pd.DataFrame({
                'Minggu': [i+1 for i in range(weeks_ahead)],
                'Tanggal': [d.strftime('%d %b %Y') for d in forecast_dates],
                'Harga (Rp)': [f"Rp {p:,.0f}" for p in pred_series],
                'Perubahan': ['—'] + [f"{((pred_series[i]-pred_series[i-1])/pred_series[i-1]*100):+.2f}%" for i in range(1, len(pred_series))],
                'Status': ['—'] + ['📈' if pred_series[i] > pred_series[i-1] else '📉' if pred_series[i] < pred_series[i-1] else '⚖️' for i in range(1, len(pred_series))]
            })
            
            st.dataframe(df_forecast, use_container_width=True, height=400)
            
            naik = sum(1 for i in range(1, len(pred_series)) if pred_series[i] > pred_series[i-1])
            turun = sum(1 for i in range(1, len(pred_series)) if pred_series[i] < pred_series[i-1])
            
            st.markdown(f'<div class="info-card"><h4 style="margin-top:0">📊 Ringkasan Trend Mingguan:</h4><ul style="margin-bottom:0"><li>📈 <b>Minggu Naik:</b> {naik}/{weeks_ahead-1} ({naik/(weeks_ahead-1)*100:.1f}%)</li><li>📉 <b>Minggu Turun:</b> {turun}/{weeks_ahead-1} ({turun/(weeks_ahead-1)*100:.1f}%)</li><li>💡 <b>Kecenderungan:</b> {"Bullish (Naik)" if naik > turun else "Bearish (Turun)" if turun > naik else "Sideways (Stabil)"}</li></ul></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download CSV", df_forecast.to_csv(index=False, encoding='utf-8-sig'), f"prediksi_{forecast_kom}_{target_date.strftime('%Y%m')}.csv", "text/csv", use_container_width=True)
            with col2:
                summary = f"""LAPORAN PREDIKSI HARGA KOMODITAS
================================
Komoditas: {forecast_kom}
Periode: {target_date.strftime('%B %Y')}
Tanggal: {datetime.now().strftime('%d %b %Y %H:%M WIB')}

RINGKASAN:
- Harga Saat Ini: Rp {current:,.0f}
- Harga Prediksi: Rp {pred_price:,.0f}
- Perubahan: {change_pct:+.2f}%
- Trend: {trend["trend"]}

STATISTIK:
- Min: Rp {pred_series.min():,.0f}
- Max: Rp {pred_series.max():,.0f}
- Avg: Rp {pred_series.mean():,.0f}
- Volatility: Rp {pred_series.std():,.0f}

REKOMENDASI:
{trend["rec"]}

AKURASI: MAPE {mape_kom:.2f}%
"""
                st.download_button("📄 Download TXT", summary, f"laporan_{forecast_kom}_{target_date.strftime('%Y%m')}.txt", "text/plain", use_container_width=True)

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("""
<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#f5f5f5,#e0e0e0);border-radius:10px;margin-top:30px">
    <h4 style="color:#1E88E5;margin:0">Sistem Prediksi Harga Komoditas Pangan Indonesia</h4>
    <p style="color:#616161;margin:8px 0"><b>Powered by Bidirectional LSTM & Streamlit</b></p>
    <p style="color:#9E9E9E;font-size:0.9rem;margin:5px 0">01 Jan 2020 - 01 Jan 2025 | 31 Komoditas | Data Mingguan</p>
</div>
""", unsafe_allow_html=True)
