import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
from datetime import datetime, timedelta
import io

warnings.filterwarnings('ignore')

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Prediksi Harga Pangan 2025-2026",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #3498db;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# FUNGSI PREPROCESSING
# =====================================================================
def preprocess_data(df_raw):
    """Preprocessing data dari Excel yang diupload"""
    try:
        # Ambil nama komoditas dari kolom kedua
        komoditas_list = df_raw.iloc[:, 1].tolist()
        
        # Transpose data (mulai dari kolom ke-3)
        df_data = df_raw.iloc[:, 2:]
        df_transposed = df_data.T
        df_transposed.columns = komoditas_list
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)
        
        # Konversi tanggal
        df_transposed['Tanggal'] = pd.to_datetime(
            df_transposed['Tanggal'], 
            format='%d/ %m/ %Y', 
            errors='coerce'
        )
        df_transposed = df_transposed.dropna(subset=['Tanggal'])
        df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)
        
        # Konversi ke numeric
        for kolom in komoditas_list:
            if df_transposed[kolom].dtype == 'object':
                df_transposed[kolom] = df_transposed[kolom].str.replace(',', '').str.replace('"', '')
            df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')
        
        # Interpolasi missing values
        df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(
            method='linear', 
            limit_direction='both'
        )
        df_transposed[komoditas_list] = df_transposed[komoditas_list].bfill().ffill()
        
        return df_transposed, komoditas_list
    
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        return None, None

# =====================================================================
# FUNGSI MEMBUAT SEQUENCES
# =====================================================================
def create_sequences(data, time_steps=20):
    """Membuat sequences untuk LSTM"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# =====================================================================
# FUNGSI BUILD MODEL LSTM
# =====================================================================
def build_lstm_model(input_shape):
    """Build model LSTM dengan arsitektur optimal"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True,
                          kernel_regularizer=l2(0.0005),
                          recurrent_regularizer=l2(0.0005),
                          input_shape=input_shape)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False,
             kernel_regularizer=l2(0.0005),
             recurrent_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(input_shape[1])
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    
    return model

# =====================================================================
# FUNGSI TRAINING MODEL
# =====================================================================
def train_model(X_train, y_train, X_test, y_test):
    """Training model LSTM"""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=0
    )
    
    with st.spinner('🔄 Model sedang dilatih... Mohon tunggu (estimasi 2-5 menit)'):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
    
    return model, history

# =====================================================================
# FUNGSI PREDIKSI MASA DEPAN
# =====================================================================
def predict_future(model, last_sequence, scalers, komoditas_list, bulan_target):
    """Prediksi harga untuk bulan tertentu di masa depan"""
    # Hitung jumlah minggu yang perlu diprediksi
    tanggal_sekarang = datetime.now()
    tanggal_target = datetime(bulan_target.year, bulan_target.month, 15)  # Tengah bulan
    
    # Hitung jumlah minggu (asumsi data mingguan)
    minggu_prediksi = int((tanggal_target - tanggal_sekarang).days / 7)
    
    if minggu_prediksi <= 0:
        minggu_prediksi = 1
    
    # Prediksi iteratif
    current_sequence = last_sequence.copy()
    predictions = []
    
    for _ in range(minggu_prediksi):
        # Prediksi normalized
        pred_norm = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
        predictions.append(pred_norm[0])
        
        # Update sequence
        current_sequence = np.vstack([current_sequence[1:], pred_norm[0]])
    
    # Denormalisasi prediksi terakhir
    final_prediction = {}
    for i, komoditas in enumerate(komoditas_list):
        pred_value = scalers[komoditas].inverse_transform([[predictions[-1][i]]])[0][0]
        final_prediction[komoditas] = pred_value
    
    return final_prediction

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("### 📊 Model Prediksi Harga Pangan")
st.sidebar.markdown("---")

# Upload file
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Dataset Excel",
    type=['xlsx', 'xls'],
    help="Upload file Excel dengan format yang sama dengan contoh dataset"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Format Dataset:**
- Kolom 1: No
- Kolom 2: Nama Komoditas
- Kolom 3+: Tanggal dengan harga
- Format tanggal: DD/ MM/ YYYY
""")

# =====================================================================
# MAIN CONTENT
# =====================================================================
st.markdown('<h1 class="main-header">🌾 Prediksi Harga Pangan Indonesia 2025-2026</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Model LSTM untuk Prediksi Multi-Komoditas Pangan</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Load data
        df_raw = pd.read_excel(uploaded_file)
        
        st.success("✅ Dataset berhasil diupload!")
        
        # Preprocessing
        with st.spinner('⚙️ Memproses dataset...'):
            df_processed, komoditas_list = preprocess_data(df_raw)
        
        if df_processed is not None and komoditas_list is not None:
            # Tampilkan info dataset
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📦 Total Data</h3>
                    <h2>{len(df_processed)}</h2>
                    <p>Baris Data</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏷️ Komoditas</h3>
                    <h2>{len(komoditas_list)}</h2>
                    <p>Jenis Komoditas</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📅 Periode</h3>
                    <h2>{df_processed['Tanggal'].min().strftime('%Y')}-{df_processed['Tanggal'].max().strftime('%Y')}</h2>
                    <p>Rentang Data</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tab untuk pilihan mode
            tab1, tab2 = st.tabs(["📈 Training & Evaluasi", "🔮 Prediksi Masa Depan"])
            
            # =====================================================================
            # TAB 1: TRAINING & EVALUASI
            # =====================================================================
            with tab1:
                st.markdown("### 🎯 Training Model & Evaluasi")
                
                if st.button("🚀 Mulai Training Model", key="train_btn"):
                    # Normalisasi data
                    scalers = {}
                    data_normalized = np.zeros((len(df_processed), len(komoditas_list)))
                    
                    for i, kolom in enumerate(komoditas_list):
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        data_normalized[:, i] = scaler.fit_transform(
                            df_processed[[kolom]].values
                        ).flatten()
                        scalers[kolom] = scaler
                    
                    # Buat sequences
                    TIME_STEPS = 20
                    X, y = create_sequences(data_normalized, TIME_STEPS)
                    
                    # Split data
                    split_idx = int(len(X) * 0.90)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Training
                    model, history = train_model(X_train, y_train, X_test, y_test)
                    
                    # Simpan ke session state
                    st.session_state['model'] = model
                    st.session_state['scalers'] = scalers
                    st.session_state['komoditas_list'] = komoditas_list
                    st.session_state['last_sequence'] = data_normalized[-TIME_STEPS:]
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    
                    st.success("✅ Model berhasil dilatih!")
                    
                    # Plot training history
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Training Loss',
                        line=dict(color='#3498db', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    fig.update_layout(
                        title='Training History',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Evaluasi model
                    st.markdown("### 📊 Evaluasi Model")
                    
                    y_pred_norm = model.predict(X_test, verbose=0)
                    
                    # Denormalisasi
                    y_pred = np.zeros_like(y_pred_norm)
                    y_true = np.zeros_like(y_test)
                    
                    for i, kolom in enumerate(komoditas_list):
                        y_pred[:, i] = scalers[kolom].inverse_transform(
                            y_pred_norm[:, i].reshape(-1, 1)
                        ).flatten()
                        y_true[:, i] = scalers[kolom].inverse_transform(
                            y_test[:, i].reshape(-1, 1)
                        ).flatten()
                    
                    # Hitung metrik
                    results = []
                    for i, komoditas in enumerate(komoditas_list):
                        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
                        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
                        
                        mask = y_true[:, i] != 0
                        if mask.sum() > 0:
                            mape = np.mean(np.abs((y_true[:, i][mask] - y_pred[:, i][mask]) / y_true[:, i][mask])) * 100
                        else:
                            mape = 0
                        
                        results.append({
                            'Komoditas': komoditas,
                            'MAPE (%)': round(mape, 2),
                            'MAE (Rp)': round(mae, 2),
                            'RMSE (Rp)': round(rmse, 2)
                        })
                    
                    df_results = pd.DataFrame(results).sort_values('MAPE (%)')
                    
                    # Tampilkan tabel hasil
                    st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # Statistik
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Rata-rata MAPE", f"{df_results['MAPE (%)'].mean():.2f}%")
                    with col2:
                        st.metric("📈 Rata-rata MAE", f"Rp {df_results['MAE (Rp)'].mean():,.0f}")
                    with col3:
                        st.metric("📉 Rata-rata RMSE", f"Rp {df_results['RMSE (Rp)'].mean():,.0f}")
            
            # =====================================================================
            # TAB 2: PREDIKSI MASA DEPAN
            # =====================================================================
            with tab2:
                st.markdown("### 🔮 Prediksi Harga Masa Depan")
                
                # Check apakah model sudah di-train
                if 'model' not in st.session_state:
                    st.warning("⚠️ Silakan training model terlebih dahulu di tab 'Training & Evaluasi'")
                else:
                    st.markdown('<div class="info-box">Pilih bulan dan tahun untuk melihat prediksi harga komoditas</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        bulan_options = {
                            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
                            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
                            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
                        }
                        bulan_selected = st.selectbox("📅 Pilih Bulan", list(bulan_options.keys()))
                    
                    with col2:
                        tahun_selected = st.selectbox("📆 Pilih Tahun", [2025, 2026, 2027])
                    
                    if st.button("🔍 Prediksi Harga", key="predict_btn"):
                        bulan_target = datetime(tahun_selected, bulan_options[bulan_selected], 15)
                        
                        with st.spinner(f'🔮 Memprediksi harga untuk {bulan_selected} {tahun_selected}...'):
                            predictions = predict_future(
                                st.session_state['model'],
                                st.session_state['last_sequence'],
                                st.session_state['scalers'],
                                st.session_state['komoditas_list'],
                                bulan_target
                            )
                        
                        st.success(f"✅ Prediksi harga untuk **{bulan_selected} {tahun_selected}** berhasil!")
                        
                        # Tampilkan hasil prediksi
                        st.markdown("### 💰 Hasil Prediksi Harga")
                        
                        # Buat DataFrame hasil
                        df_predictions = pd.DataFrame([
                            {
                                'Komoditas': k,
                                'Prediksi Harga (Rp)': f"Rp {v:,.0f}"
                            }
                            for k, v in predictions.items()
                        ])
                        
                        # Tampilkan dalam 2 kolom
                        col1, col2 = st.columns(2)
                        
                        mid = len(df_predictions) // 2
                        
                        with col1:
                            st.dataframe(
                                df_predictions.iloc[:mid],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            st.dataframe(
                                df_predictions.iloc[mid:],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        # Visualisasi top 10 komoditas termahal
                        st.markdown("### 📊 Top 10 Komoditas Termahal")
                        
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[v for k, v in sorted_predictions],
                                y=[k for k, v in sorted_predictions],
                                orientation='h',
                                marker=dict(
                                    color=np.arange(10),
                                    colorscale='Viridis'
                                )
                            )
                        ])
                        
                        fig.update_layout(
                            title=f'Top 10 Komoditas Termahal - {bulan_selected} {tahun_selected}',
                            xaxis_title='Harga (Rp)',
                            yaxis_title='Komoditas',
                            height=500,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        csv = df_predictions.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Download Hasil Prediksi (CSV)",
                            data=csv,
                            file_name=f"prediksi_harga_{bulan_selected}_{tahun_selected}.csv",
                            mime="text/csv"
                        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Pastikan format file Excel sesuai dengan contoh dataset")

else:
    # Tampilan awal sebelum upload
    st.markdown("""
    <div class="info-box">
        <h3>📋 Cara Menggunakan Aplikasi:</h3>
        <ol>
            <li>Upload dataset Excel di sidebar (format sesuai contoh)</li>
            <li>Pilih tab <b>Training & Evaluasi</b> untuk melatih model</li>
            <li>Setelah training selesai, pilih tab <b>Prediksi Masa Depan</b></li>
            <li>Pilih bulan dan tahun yang ingin diprediksi</li>
            <li>Klik tombol <b>Prediksi Harga</b> untuk melihat hasil</li>
        </ol>
        <br>
        <h3>📊 Fitur Aplikasi:</h3>
        <ul>
            <li>✅ Training model LSTM dengan arsitektur optimal</li>
            <li>✅ Evaluasi performa model (MAPE, MAE, RMSE)</li>
            <li>✅ Prediksi harga untuk 31 komoditas pangan</li>
            <li>✅ Visualisasi interaktif dengan Plotly</li>
            <li>✅ Download hasil prediksi dalam format CSV</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contoh screenshot atau demo
    st.markdown("---")
    st.markdown("### 📸 Preview Aplikasi")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📊 **Tab Training & Evaluasi**\n\nLatih model LSTM dan lihat performa evaluasi untuk setiap komoditas")
    with col2:
        st.info("🔮 **Tab Prediksi Masa Depan**\n\nPilih bulan & tahun untuk prediksi harga di masa depan")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🌾 Model Prediksi Harga Pangan Indonesia | Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
