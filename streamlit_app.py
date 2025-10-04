import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Prediksi Bahan Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for padding and styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .sidebar .sidebar-content {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #0277bd;
    }
    .result-box {
        background-color: #f3e5f5;
        padding: 2rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border-left: 5px solid #7b1fa2;
    }
    .upload-box {
        background-color: #fff3e0;
        padding: 2rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border-left: 5px solid #ef6c00;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Model Information
st.sidebar.markdown("## Model Information")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### LSTM Model untuk Prediksi Harga Pangan")
    st.markdown("")
    st.markdown("**Model Architecture:**")
    st.markdown("- Bidirectional LSTM (200 units)")
    st.markdown("- Bidirectional LSTM (150 units)")
    st.markdown("- LSTM (100 units)")
    st.markdown("- Dense layers (100, 50)")
    st.markdown("")
    st.markdown("**Training Configuration:**")
    st.markdown("- Time Steps: 20")
    st.markdown("- Train/Test Split: 85/15")
    st.markdown("- Optimizer: Adam (lr=0.0003)")
    st.markdown("- Loss Function: Huber")
    st.markdown("- Early Stopping: 40 patience")
    st.markdown("")
    st.markdown("**Data Processing:**")
    st.markdown("- MinMax Normalization (0-1)")
    st.markdown("- Linear Interpolation")
    st.markdown("- Weekly data frequency")
    st.markdown("")
    st.markdown("**Performance Metrics:**")
    st.markdown("- RMSE: Root Mean Squared Error")
    st.markdown("- MAE: Mean Absolute Error")
    st.markdown("- MAPE: Mean Absolute Percentage Error")
    st.markdown('</div>', unsafe_allow_html=True)

# Main page
st.title("Prediksi Bahan Pangan")
st.markdown("### Sistem Prediksi Harga Komoditas Pangan Menggunakan LSTM")
st.markdown("---")

# Dataset upload section
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
st.subheader("Upload Dataset")
st.markdown("")
uploaded_file = st.file_uploader("Pilih file dataset (Excel)", type=['xlsx', 'xls'])
st.markdown("")
if uploaded_file is not None:
    st.success("Dataset berhasil diupload!")
else:
    st.info("Silakan upload dataset untuk memulai prediksi")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction configuration
if uploaded_file is not None:
    # Load and process dataset
    @st.cache_data
    def load_and_process_data(file):
        df_raw = pd.read_excel(file)
        komoditas_list = df_raw['Komoditas (Rp)'].tolist()

        # Transpose data
        df_data = df_raw.iloc[:, 2:]
        df_transposed = df_data.T
        df_transposed.columns = komoditas_list
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={'index': 'Tanggal'}, inplace=True)

        # Convert date and sort
        df_transposed['Tanggal'] = pd.to_datetime(df_transposed['Tanggal'], format='%d/ %m/ %Y', errors='coerce')
        df_transposed = df_transposed.sort_values('Tanggal').reset_index(drop=True)

        # Convert to numeric
        for kolom in komoditas_list:
            df_transposed[kolom] = df_transposed[kolom].astype(str).str.replace(',', '')
            df_transposed[kolom] = pd.to_numeric(df_transposed[kolom], errors='coerce')

        df_transposed[komoditas_list] = df_transposed[komoditas_list].interpolate(method='linear', limit_direction='both')

        return df_transposed, komoditas_list

    try:
        df_processed, komoditas_list = load_and_process_data(uploaded_file)

        # Configuration section
        st.markdown("")
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("Pilih Komoditas")
            st.markdown("")
            selected_commodity = st.selectbox(
                "Komoditas:",
                komoditas_list,
                help="Pilih komoditas yang ingin diprediksi"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("Tahun Prediksi")
            st.markdown("")
            selected_year = st.selectbox(
                "Tahun:",
                [2025, 2026],
                help="Pilih tahun untuk prediksi"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("Bulan Prediksi")
            st.markdown("")
            month_names = {
                1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
                5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
                9: "September", 10: "Oktober", 11: "November", 12: "Desember"
            }
            selected_month = st.selectbox(
                "Bulan:",
                list(month_names.keys()),
                format_func=lambda x: month_names[x],
                help="Pilih bulan untuk prediksi"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Prediction button
        st.markdown("")
        st.markdown("")
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            predict_button = st.button(
                "🔮 Mulai Prediksi",
                use_container_width=True,
                type="primary"
            )

        # Prediction logic
        if predict_button:
            try:
                with st.spinner("Memproses prediksi..."):
                    # Load model
                    @st.cache_resource
                    def load_lstm_model():
                        return load_model('best_lstm_model.h5', compile=False)

                    model = load_lstm_model()

                    # Load evaluation results
                    @st.cache_data
                    def load_evaluation():
                        return pd.read_csv('hasil_evaluasi_lstm_optimal.csv')

                    df_eval = load_evaluation()

                    # Prepare data for prediction
                    def prepare_prediction_data(df, commodity, target_date):
                        # Get data for selected commodity
                        commodity_data = df[commodity].values

                        # Normalize data
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        normalized_data = scaler.fit_transform(commodity_data.reshape(-1, 1)).flatten()

                        # Create sequence for prediction (last 20 points)
                        TIME_STEPS = 20
                        if len(normalized_data) >= TIME_STEPS:
                            sequence = normalized_data[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)
                            return sequence, scaler
                        else:
                            return None, None

                    # Create target date
                    target_date = datetime(selected_year, selected_month, 1)

                    # Prepare data
                    sequence, scaler = prepare_prediction_data(df_processed, selected_commodity, target_date)

                    if sequence is not None and scaler is not None:
                        # Make prediction
                        prediction_normalized = model.predict(sequence, verbose=0)

                        # Get prediction for selected commodity
                        commodity_idx = komoditas_list.index(selected_commodity)
                        predicted_price = scaler.inverse_transform(
                            prediction_normalized[0][commodity_idx].reshape(-1, 1)
                        )[0][0]

                        # Get evaluation metrics
                        eval_row = df_eval[df_eval['Komoditas'] == selected_commodity]
                        if not eval_row.empty:
                            rmse = eval_row['RMSE'].values[0]
                            mae = eval_row['MAE'].values[0]
                            mape = eval_row['MAPE (%)'].values[0]
                        else:
                            rmse = mae = mape = 0

                        # Display results
                        st.markdown("")
                        st.markdown("")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader(f"Hasil Prediksi - {selected_commodity}")
                        st.markdown(f"**Periode:** {month_names[selected_month]} {selected_year}")
                        st.markdown("")

                        # Metrics display
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                label="Harga Prediksi",
                                value=f"Rp {predicted_price:,.0f}",
                                help="Prediksi harga untuk periode yang dipilih"
                            )

                        with col2:
                            st.metric(
                                label="RMSE",
                                value=f"Rp {rmse:,.0f}",
                                help="Root Mean Squared Error dari model"
                            )

                        with col3:
                            st.metric(
                                label="MAE",
                                value=f"Rp {mae:,.0f}",
                                help="Mean Absolute Error dari model"
                            )

                        with col4:
                            st.metric(
                                label="MAPE",
                                value=f"{mape:.2f}%",
                                help="Mean Absolute Percentage Error dari model"
                            )

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Chart section
                        st.markdown("")
                        st.markdown("")
                        st.subheader("Grafik Harga Historis dan Prediksi")
                        st.markdown("")

                        # Prepare chart data
                        historical_data = df_processed[['Tanggal', selected_commodity]].copy()
                        historical_data = historical_data.dropna()

                        # Add prediction point
                        prediction_data = pd.DataFrame({
                            'Tanggal': [target_date],
                            selected_commodity: [predicted_price]
                        })

                        # Create chart
                        fig = go.Figure()

                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_data['Tanggal'],
                            y=historical_data[selected_commodity],
                            mode='lines+markers',
                            name='Data Historis',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))

                        # Prediction point
                        fig.add_trace(go.Scatter(
                            x=prediction_data['Tanggal'],
                            y=prediction_data[selected_commodity],
                            mode='markers',
                            name='Prediksi',
                            marker=dict(
                                size=12,
                                color='#ff7f0e',
                                symbol='diamond',
                                line=dict(width=2, color='white')
                            )
                        ))

                        # Update layout
                        fig.update_layout(
                            title=f'Harga {selected_commodity} - Historis dan Prediksi',
                            xaxis_title='Tanggal',
                            yaxis_title='Harga (Rp)',
                            height=500,
                            hovermode='x unified',
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            ),
                            template='plotly_white'
                        )

                        # Format y-axis
                        fig.update_yaxis(tickformat=',')

                        st.plotly_chart(fig, use_container_width=True)

                        # Additional information
                        st.markdown("")
                        st.markdown("")
                        with st.expander("Informasi Tambahan"):
                            st.markdown(f"**Dataset terakhir:** {historical_data['Tanggal'].max().strftime('%d %B %Y')}")
                            st.markdown(f"**Jumlah data historis:** {len(historical_data)} record")
                            st.markdown(f"**Rata-rata harga historis:** Rp {historical_data[selected_commodity].mean():,.0f}")
                            st.markdown(f"**Harga minimum:** Rp {historical_data[selected_commodity].min():,.0f}")
                            st.markdown(f"**Harga maksimum:** Rp {historical_data[selected_commodity].max():,.0f}")

                            # Performance interpretation
                            st.markdown("")
                            st.markdown("**Interpretasi Performa Model:**")
                            if mape < 5:
                                st.success(f"Model memiliki akurasi sangat baik (MAPE: {mape:.2f}%)")
                            elif mape < 10:
                                st.info(f"Model memiliki akurasi baik (MAPE: {mape:.2f}%)")
                            elif mape < 20:
                                st.warning(f"Model memiliki akurasi cukup (MAPE: {mape:.2f}%)")
                            else:
                                st.error(f"Model memiliki akurasi rendah (MAPE: {mape:.2f}%)")

                    else:
                        st.error("Data tidak mencukupi untuk melakukan prediksi. Minimal diperlukan 20 data point.")

            except Exception as e:
                st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")
                st.info("Pastikan file model 'best_lstm_model.h5' dan 'hasil_evaluasi_lstm_optimal.csv' tersedia.")

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses dataset: {str(e)}")
        st.info("Pastikan format dataset sesuai dengan yang diharapkan.")

else:
    # Show sample information when no file uploaded
    st.markdown("")
    st.markdown("")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("Tentang Aplikasi")
    st.markdown("")
    st.markdown("Aplikasi ini menggunakan model LSTM (Long Short-Term Memory) untuk memprediksi harga komoditas pangan.")
    st.markdown("")
    st.markdown("**Cara Penggunaan:**")
    st.markdown("1. Upload dataset dalam format Excel")
    st.markdown("2. Pilih komoditas yang ingin diprediksi")
    st.markdown("3. Tentukan tahun dan bulan prediksi")
    st.markdown("4. Klik tombol prediksi untuk melihat hasil")
    st.markdown("")
    st.markdown("**Format Dataset:**")
    st.markdown("- Kolom pertama: No")
    st.markdown("- Kolom kedua: Komoditas (Rp)")
    st.markdown("- Kolom ketiga dan seterusnya: Data harga mingguan")
    st.markdown("")
    st.markdown("**Output:**")
    st.markdown("- Harga prediksi untuk periode yang dipilih")
    st.markdown("- Metrik evaluasi model (RMSE, MAE, MAPE)")
    st.markdown("- Grafik perbandingan data historis dan prediksi")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("")
st.markdown("")
st.markdown("---")
st.markdown("**Sistem Prediksi Bahan Pangan** | Powered by LSTM Deep Learning")
