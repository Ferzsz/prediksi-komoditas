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

warnings.filterwarnings('ignore')

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Prediksi Harga Pangan 2025-2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    /* Background dan padding utama */
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        background-color: #ffffff;
        border-bottom: 3px solid #2c3e50;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
        padding: 0.5rem;
        background-color: #ffffff;
    }
    
    /* Card untuk metrik */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-card h3 {
        color: #2c3e50;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        color: #34495e;
        font-size: 2rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .metric-card p {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stButton>button:hover {
        background-color: #34495e;
        border: none;
    }
    
    /* Info box */
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #3498db;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #ffffff;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Upload area */
    .uploadedFile {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border: 2px dashed #cccccc;
    }
    
    /* Metric container */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
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
        st.error(f"Terjadi kesalahan dalam preprocessing: {str(e)}")
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
    
    with st.spinner('Sedang melatih model... Mohon tunggu (estimasi 2-5 menit)'):
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
st.sidebar.markdown("### Model Prediksi Harga Pangan")
st.sidebar.markdown("---")

# Upload file
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset Excel",
    type=['xlsx', 'xls'],
    help="Upload file Excel dengan format yang sama dengan contoh dataset"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Format Dataset:**
- Kolom 1: Nomor urut
- Kolom 2: Nama Komoditas
- Kolom 3 dan seterusnya: Tanggal dengan harga
- Format tanggal: DD/ MM/ YYYY
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Informasi Model:**
- Model: LSTM Bidirectional
- Optimizer: Adam
- Loss Function: Huber
- Metrics: MAE, RMSE, MAPE
""")

# =====================================================================
# MAIN CONTENT
# =====================================================================
st.markdown('<h1 class="main-header">Model Prediksi Harga Pangan Indonesia 2025-2026</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Prediksi Harga Multi-Komoditas Menggunakan Long Short-Term Memory (LSTM)</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Load data
        df_raw = pd.read_excel(uploaded_file)
        
        st.markdown('<div class="success-box">Dataset berhasil diupload dan siap diproses</div>', unsafe_allow_html=True)
        
        # Preprocessing
        with st.spinner('Memproses dataset...'):
            df_processed, komoditas_list = preprocess_data(df_raw)
        
        if df_processed is not None and komoditas_list is not None:
            # Tampilkan info dataset
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Data</h3>
                    <h2>{len(df_processed)}</h2>
                    <p>Baris Data</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Jumlah Komoditas</h3>
                    <h2>{len(komoditas_list)}</h2>
                    <p>Jenis Komoditas</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Periode Data</h3>
                    <h2>{df_processed['Tanggal'].min().strftime('%Y')}-{df_processed['Tanggal'].max().strftime('%Y')}</h2>
                    <p>Rentang Waktu</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tab untuk pilihan mode
            tab1, tab2 = st.tabs(["Training dan Evaluasi Model", "Prediksi Harga Masa Depan"])
            
            # =====================================================================
            # TAB 1: TRAINING & EVALUASI
            # =====================================================================
            with tab1:
                st.markdown("### Training Model dan Evaluasi Performa")
                st.markdown('<div class="info-box">Klik tombol di bawah untuk memulai proses training model LSTM. Proses ini akan memakan waktu beberapa menit tergantung spesifikasi komputer Anda.</div>', unsafe_allow_html=True)
                
                if st.button("Mulai Training Model", key="train_btn"):
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
                    
                    st.markdown('<div class="success-box">Training model berhasil diselesaikan</div>', unsafe_allow_html=True)
                    
                    # Plot training history
                    st.markdown("#### Riwayat Training Model")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Training Loss',
                        line=dict(color='#2c3e50', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    fig.update_layout(
                        title='Grafik Loss Training dan Validasi',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        height=400,
                        template='plotly_white',
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Evaluasi model
                    st.markdown("---")
                    st.markdown("### Hasil Evaluasi Model")
                    
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
                    st.markdown("#### Tabel Evaluasi Metrik per Komoditas")
                    st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # Statistik
                    st.markdown("#### Ringkasan Statistik Evaluasi")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rata-rata MAPE", f"{df_results['MAPE (%)'].mean():.2f}%")
                    with col2:
                        st.metric("Rata-rata MAE", f"Rp {df_results['MAE (Rp)'].mean():,.0f}")
                    with col3:
                        st.metric("Rata-rata RMSE", f"Rp {df_results['RMSE (Rp)'].mean():,.0f}")
                    
                    # Visualisasi metrik
                    st.markdown("---")
                    st.markdown("#### Visualisasi Metrik Evaluasi")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Grafik MAPE
                        fig_mape = go.Figure()
                        fig_mape.add_trace(go.Bar(
                            x=df_results['MAPE (%)'].head(10),
                            y=df_results['Komoditas'].head(10),
                            orientation='h',
                            marker=dict(color='#2c3e50')
                        ))
                        fig_mape.update_layout(
                            title='Top 10 Komoditas - MAPE Terbaik',
                            xaxis_title='MAPE (%)',
                            yaxis_title='Komoditas',
                            height=400,
                            template='plotly_white',
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        st.plotly_chart(fig_mape, use_container_width=True)
                    
                    with col2:
                        # Grafik MAE
                        fig_mae = go.Figure()
                        fig_mae.add_trace(go.Bar(
                            x=df_results.sort_values('MAE (Rp)')['MAE (Rp)'].head(10),
                            y=df_results.sort_values('MAE (Rp)')['Komoditas'].head(10),
                            orientation='h',
                            marker=dict(color='#34495e')
                        ))
                        fig_mae.update_layout(
                            title='Top 10 Komoditas - MAE Terbaik',
                            xaxis_title='MAE (Rp)',
                            yaxis_title='Komoditas',
                            height=400,
                            template='plotly_white',
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        st.plotly_chart(fig_mae, use_container_width=True)
            
            # =====================================================================
            # TAB 2: PREDIKSI MASA DEPAN
            # =====================================================================
            with tab2:
                st.markdown("### Prediksi Harga Komoditas Masa Depan")
                
                # Check apakah model sudah di-train
                if 'model' not in st.session_state:
                    st.markdown('<div class="warning-box">Silakan lakukan training model terlebih dahulu di tab "Training dan Evaluasi Model"</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">Pilih bulan dan tahun yang ingin diprediksi, kemudian klik tombol "Mulai Prediksi" untuk melihat hasil prediksi harga seluruh komoditas</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        bulan_options = {
                            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
                            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
                            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
                        }
                        bulan_selected = st.selectbox("Pilih Bulan", list(bulan_options.keys()))
                    
                    with col2:
                        tahun_selected = st.selectbox("Pilih Tahun", [2025, 2026, 2027])
                    
                    if st.button("Mulai Prediksi", key="predict_btn"):
                        bulan_target = datetime(tahun_selected, bulan_options[bulan_selected], 15)
                        
                        with st.spinner(f'Memprediksi harga untuk {bulan_selected} {tahun_selected}...'):
                            predictions = predict_future(
                                st.session_state['model'],
                                st.session_state['last_sequence'],
                                st.session_state['scalers'],
                                st.session_state['komoditas_list'],
                                bulan_target
                            )
                        
                        st.markdown(f'<div class="success-box">Prediksi harga untuk <strong>{bulan_selected} {tahun_selected}</strong> berhasil dibuat</div>', unsafe_allow_html=True)
                        
                        # Tampilkan hasil prediksi
                        st.markdown("---")
                        st.markdown("### Hasil Prediksi Harga")
                        
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
                            st.markdown("#### Komoditas 1 - {}".format(mid))
                            st.dataframe(
                                df_predictions.iloc[:mid],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            st.markdown("#### Komoditas {} - {}".format(mid+1, len(df_predictions)))
                            st.dataframe(
                                df_predictions.iloc[mid:],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        # Visualisasi top 10 komoditas termahal
                        st.markdown("---")
                        st.markdown("### Visualisasi Harga Prediksi")
                        
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[v for k, v in sorted_predictions],
                                y=[k for k, v in sorted_predictions],
                                orientation='h',
                                marker=dict(color='#2c3e50')
                            )
                        ])
                        
                        fig.update_layout(
                            title=f'Top 10 Komoditas dengan Harga Tertinggi - {bulan_selected} {tahun_selected}',
                            xaxis_title='Harga (Rp)',
                            yaxis_title='Komoditas',
                            height=500,
                            template='plotly_white',
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualisasi top 10 komoditas termurah
                        st.markdown("---")
                        sorted_predictions_cheap = sorted(predictions.items(), key=lambda x: x[1])[:10]
                        
                        fig2 = go.Figure(data=[
                            go.Bar(
                                x=[v for k, v in sorted_predictions_cheap],
                                y=[k for k, v in sorted_predictions_cheap],
                                orientation='h',
                                marker=dict(color='#34495e')
                            )
                        ])
                        
                        fig2.update_layout(
                            title=f'Top 10 Komoditas dengan Harga Terendah - {bulan_selected} {tahun_selected}',
                            xaxis_title='Harga (Rp)',
                            yaxis_title='Komoditas',
                            height=500,
                            template='plotly_white',
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Download button
                        st.markdown("---")
                        st.markdown("### Unduh Hasil Prediksi")
                        csv = df_predictions.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="Unduh Hasil Prediksi (CSV)",
                            data=csv,
                            file_name=f"prediksi_harga_{bulan_selected}_{tahun_selected}.csv",
                            mime="text/csv"
                        )
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.error("Pastikan format file Excel sesuai dengan contoh dataset yang diberikan")

else:
    # Tampilan awal sebelum upload
    st.markdown("""
    <div class="info-box">
        <h3>Panduan Penggunaan Aplikasi</h3>
        <ol style="padding-left: 1.5rem;">
            <li style="margin-bottom: 0.5rem;">Upload dataset Excel di sidebar dengan format yang sesuai</li>
            <li style="margin-bottom: 0.5rem;">Pilih tab <strong>Training dan Evaluasi Model</strong> untuk melatih model LSTM</li>
            <li style="margin-bottom: 0.5rem;">Tunggu hingga proses training selesai dan lihat hasil evaluasi</li>
            <li style="margin-bottom: 0.5rem;">Pindah ke tab <strong>Prediksi Harga Masa Depan</strong></li>
            <li style="margin-bottom: 0.5rem;">Pilih bulan dan tahun yang ingin diprediksi</li>
            <li style="margin-bottom: 0.5rem;">Klik tombol <strong>Mulai Prediksi</strong> untuk melihat hasil</li>
            <li style="margin-bottom: 0.5rem;">Unduh hasil prediksi dalam format CSV jika diperlukan</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>Fitur Aplikasi</h3>
        <ul style="padding-left: 1.5rem;">
            <li style="margin-bottom: 0.5rem;">Training model LSTM dengan arsitektur Bidirectional</li>
            <li style="margin-bottom: 0.5rem;">Evaluasi performa model menggunakan metrik MAPE, MAE, dan RMSE</li>
            <li style="margin-bottom: 0.5rem;">Prediksi harga untuk 31 jenis komoditas pangan</li>
            <li style="margin-bottom: 0.5rem;">Visualisasi interaktif hasil prediksi dan evaluasi</li>
            <li style="margin-bottom: 0.5rem;">Export hasil prediksi ke format CSV</li>
            <li style="margin-bottom: 0.5rem;">Antarmuka yang responsif dan mudah digunakan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Informasi tambahan
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Tab Training dan Evaluasi</h4>
            <p style="margin: 0.5rem 0;">Pada tab ini Anda dapat melatih model LSTM dan melihat hasil evaluasi performa model untuk setiap komoditas. Tersedia visualisasi grafik loss dan metrik evaluasi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Tab Prediksi Masa Depan</h4>
            <p style="margin: 0.5rem 0;">Setelah model dilatih, Anda dapat memilih bulan dan tahun untuk memprediksi harga komoditas di masa depan dengan visualisasi yang informatif.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1.5rem;'>
    <p style="margin: 0;">Model Prediksi Harga Pangan Indonesia</p>
    <p style="margin: 0.5rem 0 0 0;">Dibuat menggunakan Streamlit dan TensorFlow</p>
</div>
""", unsafe_allow_html=True)
