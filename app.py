import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Pangan LSTM",
    page_icon="🌾",
    layout="wide"
)

# Load model dan scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('lstm_food_price_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel('dataset.xlsx')
    return df

# Load hasil evaluasi
@st.cache_data
def load_evaluation():
    eval_df = pd.read_csv('lstm_evaluation_results_all_commodities.csv')
    return eval_df

# Fungsi preprocessing
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

# Fungsi prediksi
def predict_future_prices(model, scaler, last_data, n_weeks=4, look_back=12):
    predictions = []
    current_data = last_data[-look_back:].reshape(-1, 1)
    current_data_scaled = scaler.transform(current_data)
    
    for _ in range(n_weeks):
        X = current_data_scaled[-look_back:].reshape(1, look_back, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred_price)
        
        current_data_scaled = np.append(current_data_scaled, pred_scaled).reshape(-1, 1)
    
    return predictions

# Main App
def main():
    st.title("🌾 Aplikasi Prediksi Harga Pangan dengan LSTM")
    st.markdown("---")
    
    # Load data
    try:
        model, scaler = load_model_and_scaler()
        df = load_data()
        eval_df = load_evaluation()
        
        # Sidebar
        st.sidebar.header("⚙️ Pengaturan")
        
        # Pilih komoditas
        commodity_list = df.iloc[:, 1].tolist()
        selected_commodity = st.sidebar.selectbox(
            "Pilih Komoditas:",
            commodity_list
        )
        
        # Jumlah minggu prediksi
        n_weeks = st.sidebar.slider(
            "Jumlah Minggu Prediksi:",
            min_value=1,
            max_value=12,
            value=4
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 Model LSTM dengan 3 layer dan dropout 0.2")
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["📈 Prediksi", "📊 Evaluasi Model", "ℹ️ Info"])
        
        with tab1:
            st.header(f"Prediksi Harga: {selected_commodity}")
            
            # Get commodity index
            commodity_idx = commodity_list.index(selected_commodity)
            commodity_name, prices_final = preprocess_commodity_data(df, commodity_idx)
            
            # Prediksi
            predictions = predict_future_prices(model, scaler, prices_final, n_weeks)
            
            # Tampilkan hasil prediksi
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Harga Terakhir",
                    f"Rp {prices_final[-1]:,.0f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    f"Prediksi {n_weeks} Minggu",
                    f"Rp {predictions[-1]:,.0f}",
                    delta=f"{((predictions[-1] - prices_final[-1]) / prices_final[-1] * 100):.2f}%"
                )
            
            with col3:
                avg_prediction = np.mean(predictions)
                st.metric(
                    "Rata-rata Prediksi",
                    f"Rp {avg_prediction:,.0f}",
                    delta=None
                )
            
            st.markdown("---")
            
            # Tabel prediksi
            st.subheader("📋 Tabel Prediksi Mingguan")
            
            prediction_dates = []
            start_date = datetime.now()
            for i in range(n_weeks):
                prediction_dates.append((start_date + timedelta(weeks=i+1)).strftime("%d/%m/%Y"))
            
            pred_df = pd.DataFrame({
                'Minggu': [f"Minggu {i+1}" for i in range(n_weeks)],
                'Tanggal Prediksi': prediction_dates,
                'Harga Prediksi (Rp)': [f"{p:,.0f}" for p in predictions]
            })
            
            st.dataframe(pred_df, use_container_width=True)
            
            # Grafik
            st.subheader("📊 Visualisasi Harga")
            
            # Data historis (20 minggu terakhir)
            historical_weeks = min(20, len(prices_final))
            historical_prices = prices_final[-historical_weeks:]
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(len(historical_prices))),
                y=historical_prices,
                mode='lines+markers',
                name='Data Historis',
                line=dict(color='blue', width=2)
            ))
            
            # Prediction data
            fig.add_trace(go.Scatter(
                x=list(range(len(historical_prices), len(historical_prices) + n_weeks)),
                y=predictions,
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Tren Harga {selected_commodity}",
                xaxis_title="Minggu",
                yaxis_title="Harga (Rp)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("📊 Evaluasi Performa Model")
            
            # Filter evaluasi untuk komoditas terpilih
            commodity_eval = eval_df[eval_df['Komoditas'] == selected_commodity]
            
            if not commodity_eval.empty:
                st.subheader(f"Metrik Evaluasi: {selected_commodity}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RMSE Testing", f"{commodity_eval['Test_RMSE'].values[0]:.2f}")
                
                with col2:
                    st.metric("MAE Testing", f"{commodity_eval['Test_MAE'].values[0]:.2f}")
                
                with col3:
                    st.metric("MAPE Testing", f"{commodity_eval['Test_MAPE'].values[0]:.2f}%")
                
                st.markdown("---")
            
            # Tabel semua komoditas
            st.subheader("📋 Evaluasi Semua Komoditas")
            st.dataframe(
                eval_df[['No', 'Komoditas', 'Test_RMSE', 'Test_MAE', 'Test_MAPE']],
                use_container_width=True
            )
            
            # Statistik keseluruhan
            st.subheader("📈 Statistik Keseluruhan")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg RMSE", f"{eval_df['Test_RMSE'].mean():.2f}")
            
            with col2:
                st.metric("Avg MAE", f"{eval_df['Test_MAE'].mean():.2f}")
            
            with col3:
                st.metric("Avg MAPE", f"{eval_df['Test_MAPE'].mean():.2f}%")
            
            # Top 10 komoditas terbaik
            st.subheader("🏆 Top 10 Komoditas dengan MAPE Terendah")
            top_10 = eval_df.nsmallest(10, 'Test_MAPE')[['Komoditas', 'Test_MAPE']]
            
            fig_top10 = go.Figure(go.Bar(
                x=top_10['Test_MAPE'],
                y=top_10['Komoditas'],
                orientation='h',
                marker=dict(color='green')
            ))
            
            fig_top10.update_layout(
                title="Top 10 Komoditas - MAPE Terendah",
                xaxis_title="MAPE (%)",
                yaxis_title="Komoditas",
                height=400
            )
            
            st.plotly_chart(fig_top10, use_container_width=True)
        
        with tab3:
            st.header("ℹ️ Informasi Aplikasi")
            
            st.markdown("""
            ### Tentang Aplikasi
            Aplikasi ini menggunakan **Long Short-Term Memory (LSTM)** neural network untuk memprediksi harga pangan berdasarkan data historis mingguan.
            
            ### Fitur Utama
            - 🔮 Prediksi harga 1-12 minggu ke depan
            - 📊 Visualisasi tren harga historis dan prediksi
            - 📈 Evaluasi performa model dengan metrik RMSE, MAE, MAPE
            - 🌾 31 komoditas pangan tersedia
            
            ### Arsitektur Model
            - **3 Layer LSTM** dengan units: 100, 100, 50
            - **Dropout**: 0.2 untuk setiap layer
            - **Optimizer**: Adam
            - **Loss Function**: Mean Squared Error
            - **Look-back Window**: 12 minggu
            
            ### Metrik Evaluasi
            - **RMSE** (Root Mean Squared Error): Mengukur rata-rata kesalahan prediksi
            - **MAE** (Mean Absolute Error): Rata-rata selisih absolut
            - **MAPE** (Mean Absolute Percentage Error): Persentase kesalahan rata-rata
            
            ### Dataset
            Data harga mingguan dari 31 komoditas pangan periode 2020-2025
            
            ### Developer
            Developed for thesis research | Indonesia
            """)
            
            st.markdown("---")
            st.info("💡 Tip: Pilih komoditas dan jumlah minggu prediksi di sidebar")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Pastikan semua file model sudah tersedia (lstm_food_price_model.h5, scaler.pkl, dataset.xlsx, lstm_evaluation_results_all_commodities.csv)")

if __name__ == "__main__":
    main()
