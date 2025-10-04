import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

st.set_page_config(
    page_title="Prediksi Harga Pangan LSTM",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS untuk styling kotak abu-abu selaras
st.markdown("""
<style>
    .metric-box-1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-2 {
        background: linear-gradient(135deg, #868f96 0%, #596164 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-3 {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-4 {
        background: linear-gradient(135deg, #7f8c8d 0%, #95a5a6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #485563 0%, #29323c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    model = load_model('lstm_food_price_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def preprocess_commodity_data(df, commodity_index):
    commodity_name = df.iloc[commodity_index, 1]
    prices = df.iloc[commodity_index, 2:].values
    
    prices_clean = []
    for price in prices:
        if price == '-' or price == '' or pd.isna(price):
            prices_clean.append(np.nan)
        else:
            prices_clean.append(float(str(price).replace(',', '')))
    
    prices_clean = np.array(prices_clean)
    prices_series = pd.Series(prices_clean)
    prices_series = prices_series.fillna(method='ffill').fillna(method='bfill')
    prices_final = prices_series.values
    
    return commodity_name, prices_final

def predict_future_prices_monthly(model, scaler, last_data, target_month, target_year, look_back=12):
    """Prediksi harga sampai bulan dan tahun tertentu"""
    # Hitung jumlah minggu dari sekarang sampai target
    current_date = datetime.now()
    target_date = datetime(target_year, target_month, 1)
    
    # Hitung selisih minggu
    weeks_diff = (target_date.year - current_date.year) * 52 + \
                 (target_date.month - current_date.month) * 4
    
    if weeks_diff <= 0:
        weeks_diff = 1
    
    predictions = []
    current_data = last_data[-look_back:].reshape(-1, 1)
    current_data_scaled = scaler.transform(current_data)
    
    for _ in range(weeks_diff):
        X = current_data_scaled[-look_back:].reshape(1, look_back, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred_price)
        
        current_data_scaled = np.append(current_data_scaled, pred_scaled).reshape(-1, 1)
    
    return predictions, weeks_diff

def calculate_accuracy_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return rmse, mae, mape, r2

def main():
    st.title("Aplikasi Prediksi Harga Pangan dengan LSTM")
    st.markdown("---")
    
    try:
        model, scaler = load_model_and_scaler()
        
        # SIDEBAR - INFO ONLY
        with st.sidebar:
            st.header("Informasi Model")
            
            with st.container():
                st.markdown("### Algoritma LSTM")
                st.markdown("""
                **Long Short-Term Memory (LSTM)** adalah jenis Recurrent Neural Network 
                yang dirancang untuk mempelajari pola ketergantungan jangka panjang dalam 
                data time series. LSTM mampu mengingat informasi penting dan melupakan 
                informasi yang tidak relevan melalui mekanisme gate.
                """)
            
            st.markdown("---")
            
            with st.container():
                st.markdown("### Metrik Evaluasi")
                st.markdown("""
                **RMSE (Root Mean Squared Error)**  
                Akar dari rata-rata kuadrat kesalahan. Semakin kecil nilai RMSE, 
                semakin akurat prediksi model.
                
                **MAE (Mean Absolute Error)**  
                Rata-rata nilai absolut dari kesalahan prediksi. Memberikan gambaran 
                rata-rata kesalahan dalam satuan harga.
                
                **MAPE (Mean Absolute Percentage Error)**  
                Persentase rata-rata kesalahan absolut. Menunjukkan akurasi model 
                dalam bentuk persentase.
                """)
            
            st.markdown("---")
            
            with st.container():
                st.markdown("### Informasi Dataset")
                st.markdown("""
                Dataset yang digunakan berisi data harga mingguan komoditas pangan 
                dari periode 2020-2025. Model menggunakan 80% data untuk training 
                dan 20% untuk testing dengan look-back window 12 minggu.
                """)
        
        # Initialize session state for uploaded data
        if 'uploaded_df' not in st.session_state:
            st.session_state.uploaded_df = None
        if 'eval_results' not in st.session_state:
            st.session_state.eval_results = None
        
        # MAIN CONTENT - 2 TABS
        tab1, tab2 = st.tabs(["Dataset", "Evaluasi"])
        
        # TAB 1: Dataset (Upload dan Prediksi)
        with tab1:
            st.markdown('<div class="info-box"><h3>Upload Dataset dan Prediksi Harga</h3></div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown("### Upload File Dataset")
                uploaded_file = st.file_uploader(
                    "Upload File Excel (.xlsx)",
                    type=['xlsx'],
                    key="upload_file"
                )
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_excel(uploaded_file)
                    st.session_state.uploaded_df = df_upload
                    
                    st.success("File berhasil diupload!")
                    
                    with st.expander("Preview Data"):
                        st.dataframe(df_upload.head(10), use_container_width=True)
                    
                    st.markdown("---")
                    
                    commodity_list = df_upload.iloc[:, 1].tolist()
                    
                    # Pengaturan Prediksi
                    with st.container():
                        st.markdown("### Pengaturan Prediksi")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            selected_commodity = st.selectbox(
                                "Pilih Komoditas:",
                                commodity_list,
                                key="pred_commodity"
                            )
                        
                        with col2:
                            target_year = st.selectbox(
                                "Pilih Tahun:",
                                options=[2025, 2026],
                                index=0,
                                key="pred_year"
                            )
                        
                        with col3:
                            months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                                     'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
                            target_month_name = st.selectbox(
                                "Pilih Bulan:",
                                options=months,
                                index=0,
                                key="pred_month"
                            )
                            target_month = months.index(target_month_name) + 1
                    
                    if st.button("Mulai Prediksi", key="btn_predict"):
                        st.markdown("---")
                        
                        commodity_idx = commodity_list.index(selected_commodity)
                        commodity_name, prices_final = preprocess_commodity_data(df_upload, commodity_idx)
                        predictions, n_weeks = predict_future_prices_monthly(
                            model, scaler, prices_final, target_month, target_year
                        )
                        
                        # Hasil Prediksi
                        with st.container():
                            st.markdown("### Hasil Prediksi")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-box-2">
                                    <h4>Harga Terakhir</h4>
                                    <h2>Rp {prices_final[-1]:,.0f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-box-3">
                                    <h4>Target Prediksi</h4>
                                    <h2>{target_month_name} {target_year}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                change_pct = ((predictions[-1] - prices_final[-1]) / prices_final[-1] * 100)
                                arrow = "↑" if change_pct > 0 else "↓"
                                st.markdown(f"""
                                <div class="metric-box-4">
                                    <h4>Harga Prediksi</h4>
                                    <h2>Rp {predictions[-1]:,.0f}</h2>
                                    <p>{arrow} {abs(change_pct):.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                avg_prediction = np.mean(predictions)
                                st.markdown(f"""
                                <div class="metric-box-1">
                                    <h4>Rata-rata Prediksi</h4>
                                    <h2>Rp {avg_prediction:,.0f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Tabel Prediksi Mingguan
                        with st.container():
                            st.markdown("### Tabel Prediksi Mingguan")
                            
                            prediction_dates = []
                            start_date = datetime.now()
                            for i in range(n_weeks):
                                prediction_dates.append((start_date + timedelta(weeks=i+1)).strftime("%d/%m/%Y"))
                            
                            pred_df = pd.DataFrame({
                                'Minggu': [f"Minggu {i+1}" for i in range(n_weeks)],
                                'Tanggal Prediksi': prediction_dates,
                                'Harga Prediksi (Rp)': [f"{p:,.0f}" for p in predictions]
                            })
                            
                            st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # Grafik
                        with st.container():
                            st.markdown("### Visualisasi Tren Harga")
                            
                            historical_weeks = min(20, len(prices_final))
                            historical_prices = prices_final[-historical_weeks:]
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(historical_prices))),
                                y=historical_prices,
                                mode='lines+markers',
                                name='Data Historis',
                                line=dict(color='#485563', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(historical_prices), len(historical_prices) + n_weeks)),
                                y=predictions,
                                mode='lines+markers',
                                name='Prediksi',
                                line=dict(color='#667eea', width=3, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                title=f"Tren Harga {selected_commodity} - Prediksi sampai {target_month_name} {target_year}",
                                xaxis_title="Minggu",
                                yaxis_title="Harga (Rp)",
                                hovermode='x unified',
                                height=500,
                                showlegend=True,
                                template='plotly_white',
                                font=dict(size=12)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Simpan hasil evaluasi untuk tab 2
                        commodity_idx_eval = commodity_list.index(selected_commodity)
                        commodity_name_eval, prices_final_eval = preprocess_commodity_data(df_upload, commodity_idx_eval)
                        
                        train_size = int(len(prices_final_eval) * 0.8)
                        train_data_eval = prices_final_eval[:train_size]
                        test_data_eval = prices_final_eval[train_size:]
                        
                        from sklearn.preprocessing import MinMaxScaler
                        scaler_eval = MinMaxScaler(feature_range=(0, 1))
                        train_scaled_eval = scaler_eval.fit_transform(train_data_eval.reshape(-1, 1))
                        test_scaled_eval = scaler_eval.transform(test_data_eval.reshape(-1, 1))
                        
                        look_back = 12
                        X_test_eval = []
                        y_test_eval = []
                        for i in range(look_back, len(test_scaled_eval)):
                            X_test_eval.append(test_scaled_eval[i-look_back:i, 0])
                            y_test_eval.append(test_scaled_eval[i, 0])
                        
                        X_test_eval = np.array(X_test_eval)
                        y_test_eval = np.array(y_test_eval)
                        X_test_eval = X_test_eval.reshape(X_test_eval.shape[0], X_test_eval.shape[1], 1)
                        
                        predictions_eval = model.predict(X_test_eval, verbose=0)
                        predictions_eval = scaler_eval.inverse_transform(predictions_eval)
                        y_test_actual_eval = scaler_eval.inverse_transform(y_test_eval.reshape(-1, 1))
                        
                        st.session_state.eval_results = {
                            'commodity_name': selected_commodity,
                            'predictions': predictions_eval,
                            'actual': y_test_actual_eval
                        }
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Silakan upload file Excel untuk melakukan prediksi")
        
        # TAB 2: Evaluasi (Otomatis dari Upload Dataset)
        with tab2:
            st.markdown('<div class="info-box"><h3>Evaluasi Model</h3></div>', unsafe_allow_html=True)
            
            if st.session_state.uploaded_df is not None:
                df_eval = st.session_state.uploaded_df
                commodity_list_eval = df_eval.iloc[:, 1].tolist()
                
                # Pilih komoditas untuk evaluasi
                with st.container():
                    st.markdown("### Pilih Komoditas untuk Evaluasi")
                    selected_eval = st.selectbox(
                        "Pilih Komoditas:",
                        commodity_list_eval,
                        key="eval_commodity"
                    )
                
                if st.button("Evaluasi Model", key="btn_eval"):
                    st.markdown("---")
                    
                    commodity_idx_eval = commodity_list_eval.index(selected_eval)
                    commodity_name_eval, prices_final_eval = preprocess_commodity_data(df_eval, commodity_idx_eval)
                    
                    # Split data 80/20
                    train_size = int(len(prices_final_eval) * 0.8)
                    train_data_eval = prices_final_eval[:train_size]
                    test_data_eval = prices_final_eval[train_size:]
                    
                    # Prediksi untuk evaluasi
                    from sklearn.preprocessing import MinMaxScaler
                    scaler_eval = MinMaxScaler(feature_range=(0, 1))
                    train_scaled_eval = scaler_eval.fit_transform(train_data_eval.reshape(-1, 1))
                    test_scaled_eval = scaler_eval.transform(test_data_eval.reshape(-1, 1))
                    
                    look_back = 12
                    X_test_eval = []
                    y_test_eval = []
                    for i in range(look_back, len(test_scaled_eval)):
                        X_test_eval.append(test_scaled_eval[i-look_back:i, 0])
                        y_test_eval.append(test_scaled_eval[i, 0])
                    
                    X_test_eval = np.array(X_test_eval)
                    y_test_eval = np.array(y_test_eval)
                    X_test_eval = X_test_eval.reshape(X_test_eval.shape[0], X_test_eval.shape[1], 1)
                    
                    predictions_eval = model.predict(X_test_eval, verbose=0)
                    predictions_eval = scaler_eval.inverse_transform(predictions_eval)
                    y_test_actual_eval = scaler_eval.inverse_transform(y_test_eval.reshape(-1, 1))
                    
                    # Hitung metrik
                    rmse_eval, mae_eval, mape_eval, r2_eval = calculate_accuracy_metrics(
                        y_test_actual_eval.flatten(),
                        predictions_eval.flatten()
                    )
                    accuracy_eval = 100 - mape_eval
                    
                    # Tampilkan metrik
                    with st.container():
                        st.markdown(f"### Hasil Evaluasi: {selected_eval}")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-box-2">
                                <h4>RMSE</h4>
                                <h2>{rmse_eval:.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-box-3">
                                <h4>MAE</h4>
                                <h2>{mae_eval:.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-box-4">
                                <h4>MAPE</h4>
                                <h2>{mape_eval:.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-box-1">
                                <h4>R²</h4>
                                <h2>{r2_eval:.4f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col5:
                            st.markdown(f"""
                            <div class="metric-box-2">
                                <h4>Akurasi</h4>
                                <h2>{accuracy_eval:.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Grafik Perbandingan
                    with st.container():
                        st.markdown("### Grafik Actual vs Predicted")
                        
                        fig_eval = go.Figure()
                        
                        fig_eval.add_trace(go.Scatter(
                            x=list(range(len(y_test_actual_eval))),
                            y=y_test_actual_eval.flatten(),
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='#485563', width=2),
                            marker=dict(size=6)
                        ))
                        
                        fig_eval.add_trace(go.Scatter(
                            x=list(range(len(predictions_eval))),
                            y=predictions_eval.flatten(),
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#667eea', width=2),
                            marker=dict(size=6)
                        ))
                        
                        fig_eval.update_layout(
                            title=f"Perbandingan Harga Actual vs Predicted - {selected_eval}",
                            xaxis_title="Minggu",
                            yaxis_title="Harga (Rp)",
                            hovermode='x unified',
                            height=500,
                            showlegend=True,
                            template='plotly_white',
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig_eval, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Scatter Plot
                    with st.container():
                        st.markdown("### Scatter Plot Actual vs Predicted")
                        
                        fig_scatter = go.Figure()
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=y_test_actual_eval.flatten(),
                            y=predictions_eval.flatten(),
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='#7f8c8d',
                                opacity=0.6
                            ),
                            name='Data Points'
                        ))
                        
                        min_val = min(y_test_actual_eval.min(), predictions_eval.min())
                        max_val = max(y_test_actual_eval.max(), predictions_eval.max())
                        fig_scatter.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(color='#667eea', width=2, dash='dash'),
                            name='Perfect Prediction'
                        ))
                        
                        fig_scatter.update_layout(
                            title="Scatter Plot: Actual vs Predicted",
                            xaxis_title="Actual Price (Rp)",
                            yaxis_title="Predicted Price (Rp)",
                            height=500,
                            template='plotly_white',
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Silakan upload dataset pada tab Dataset terlebih dahulu untuk melakukan evaluasi")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Pastikan file model tersedia: lstm_food_price_model.h5 dan scaler.pkl")

if __name__ == "__main__":
    main()
