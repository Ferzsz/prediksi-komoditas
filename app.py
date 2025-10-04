import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime
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
        background: linear-gradient(135deg, #868f96 0%, #596164 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-2 {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-3 {
        background: linear-gradient(135deg, #7f8c8d 0%, #95a5a6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-4 {
        background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-5 {
        background: linear-gradient(135deg, #596164 0%, #868f96 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box-6 {
        background: linear-gradient(135deg, #2c3e50 0%, #bdc3c7 100%);
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

def predict_until_target_date(model, scaler, last_data, target_month, target_year, look_back=12):
    """Prediksi harga sampai bulan dan tahun tertentu"""
    current_date = datetime.now()
    target_date = datetime(target_year, target_month, 1)
    
    # Hitung jumlah bulan dari sekarang sampai target
    months_diff = (target_date.year - current_date.year) * 12 + (target_date.month - current_date.month)
    
    if months_diff <= 0:
        months_diff = 1
    
    n_weeks = months_diff * 4  # Konversi ke minggu
    
    predictions_weekly = []
    current_data = last_data[-look_back:].reshape(-1, 1)
    current_data_scaled = scaler.transform(current_data)
    
    for _ in range(n_weeks):
        X = current_data_scaled[-look_back:].reshape(1, look_back, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        predictions_weekly.append(pred_price)
        
        current_data_scaled = np.append(current_data_scaled, pred_scaled).reshape(-1, 1)
    
    # Konversi ke bulanan (ambil rata-rata per 4 minggu)
    predictions_monthly = []
    for i in range(0, len(predictions_weekly), 4):
        chunk = predictions_weekly[i:i+4]
        monthly_avg = np.mean(chunk)
        predictions_monthly.append(monthly_avg)
    
    return predictions_monthly, predictions_weekly, months_diff

def calculate_accuracy_metrics(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return rmse, mae, mape

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
        
        # MAIN CONTENT
        st.markdown('<div class="info-box"><h3>Prediksi Harga Pangan Bulanan</h3></div>', unsafe_allow_html=True)
        
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
                
                st.markdown("---")
                
                # Otomatis proses prediksi dan evaluasi
                commodity_idx = commodity_list.index(selected_commodity)
                commodity_name, prices_final = preprocess_commodity_data(df_upload, commodity_idx)
                
                # Prediksi
                predictions_monthly, predictions_weekly, months_diff = predict_until_target_date(
                    model, scaler, prices_final, target_month, target_year
                )
                
                # Evaluasi
                train_size = int(len(prices_final) * 0.8)
                train_data = prices_final[:train_size]
                test_data = prices_final[train_size:]
                
                from sklearn.preprocessing import MinMaxScaler
                scaler_eval = MinMaxScaler(feature_range=(0, 1))
                train_scaled = scaler_eval.fit_transform(train_data.reshape(-1, 1))
                test_scaled = scaler_eval.transform(test_data.reshape(-1, 1))
                
                look_back = 12
                X_test = []
                y_test = []
                for i in range(look_back, len(test_scaled)):
                    X_test.append(test_scaled[i-look_back:i, 0])
                    y_test.append(test_scaled[i, 0])
                
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                predictions_eval = model.predict(X_test, verbose=0)
                predictions_eval = scaler_eval.inverse_transform(predictions_eval)
                y_test_actual = scaler_eval.inverse_transform(y_test.reshape(-1, 1))
                
                rmse_eval, mae_eval, mape_eval = calculate_accuracy_metrics(
                    y_test_actual.flatten(),
                    predictions_eval.flatten()
                )
                
                # Tampilkan Hasil
                with st.container():
                    st.markdown("### Hasil Prediksi dan Evaluasi")
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box-1">
                            <h4>Harga Terakhir</h4>
                            <h2>Rp {prices_final[-1]:,.0f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box-2">
                            <h4>Harga Prediksi</h4>
                            <h2>Rp {predictions_monthly[-1]:,.0f}</h2>
                            <p>{target_month_name} {target_year}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-box-3">
                            <h4>Rata-rata</h4>
                            <h2>Rp {np.mean(predictions_monthly):,.0f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-box-4">
                            <h4>RMSE</h4>
                            <h2>{rmse_eval:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col5:
                        st.markdown(f"""
                        <div class="metric-box-5">
                            <h4>MAE</h4>
                            <h2>{mae_eval:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col6:
                        st.markdown(f"""
                        <div class="metric-box-6">
                            <h4>MAPE</h4>
                            <h2>{mape_eval:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Tabel Prediksi Bulanan
                with st.container():
                    st.markdown("### Tabel Prediksi Bulanan")
                    
                    current_date = datetime.now()
                    prediction_months = []
                    for i in range(months_diff):
                        future_date = current_date + relativedelta(months=i+1)
                        month_name = months[future_date.month - 1]
                        prediction_months.append(f"{month_name} {future_date.year}")
                    
                    pred_df = pd.DataFrame({
                        'Bulan': prediction_months,
                        'Harga Prediksi (Rp)': [f"{p:,.0f}" for p in predictions_monthly]
                    })
                    
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Grafik Prediksi
                with st.container():
                    st.markdown("### Grafik Prediksi Bulanan")
                    
                    historical_weeks = min(20, len(prices_final))
                    historical_prices = prices_final[-historical_weeks:]
                    
                    fig_pred = go.Figure()
                    
                    fig_pred.add_trace(go.Scatter(
                        x=list(range(len(historical_prices))),
                        y=historical_prices,
                        mode='lines+markers',
                        name='Data Historis (Mingguan)',
                        line=dict(color='#596164', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Plot prediksi mingguan
                    fig_pred.add_trace(go.Scatter(
                        x=list(range(len(historical_prices), len(historical_prices) + len(predictions_weekly))),
                        y=predictions_weekly,
                        mode='lines',
                        name='Prediksi Mingguan',
                        line=dict(color='#95a5a6', width=2, dash='dot'),
                        opacity=0.5
                    ))
                    
                    # Plot prediksi bulanan
                    monthly_x = list(range(len(historical_prices), len(historical_prices) + len(predictions_weekly), 4))
                    fig_pred.add_trace(go.Scatter(
                        x=monthly_x,
                        y=predictions_monthly,
                        mode='lines+markers',
                        name='Prediksi Bulanan',
                        line=dict(color='#2c3e50', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig_pred.update_layout(
                        title=f"Prediksi Harga {selected_commodity} sampai {target_month_name} {target_year}",
                        xaxis_title="Periode",
                        yaxis_title="Harga (Rp)",
                        hovermode='x unified',
                        height=500,
                        showlegend=True,
                        template='plotly_white',
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                st.markdown("---")
                
                # Grafik Evaluasi
                with st.container():
                    st.markdown("### Grafik Evaluasi Model (Actual vs Predicted)")
                    
                    fig_eval = go.Figure()
                    
                    fig_eval.add_trace(go.Scatter(
                        x=list(range(len(y_test_actual))),
                        y=y_test_actual.flatten(),
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#596164', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig_eval.add_trace(go.Scatter(
                        x=list(range(len(predictions_eval))),
                        y=predictions_eval.flatten(),
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='#95a5a6', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig_eval.update_layout(
                        title=f"Evaluasi Model - {selected_commodity}",
                        xaxis_title="Data Testing",
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
                    st.markdown("### Scatter Plot Evaluasi")
                    
                    fig_scatter = go.Figure()
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=y_test_actual.flatten(),
                        y=predictions_eval.flatten(),
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='#7f8c8d',
                            opacity=0.6
                        ),
                        name='Data Points'
                    ))
                    
                    min_val = min(y_test_actual.min(), predictions_eval.min())
                    max_val = max(y_test_actual.max(), predictions_eval.max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='#2c3e50', width=2, dash='dash'),
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
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Silakan upload file Excel untuk memulai prediksi")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Pastikan file model tersedia: lstm_food_price_model.h5 dan scaler.pkl")

if __name__ == "__main__":
    main()
