import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Prediksi Harga Pangan LSTM",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS untuk styling kotak
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-box-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-box-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-box-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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

@st.cache_data
def load_evaluation():
    eval_df = pd.read_csv('lstm_evaluation_results_all_commodities.csv')
    return eval_df

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

def calculate_accuracy_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return rmse, mae, mape, r2

def main():
    st.title("Aplikasi Prediksi Harga Pangan dengan LSTM")
    st.markdown("---")
    
    try:
        model, scaler = load_model_and_scaler()
        eval_df = load_evaluation()
        
        # SIDEBAR - INFO ONLY
        with st.sidebar:
            st.header("Informasi Model")
            
            with st.container():
                st.markdown("### Arsitektur LSTM")
                st.markdown("""
                - Layer 1: LSTM 100 units
                - Layer 2: LSTM 100 units
                - Layer 3: LSTM 50 units
                - Dropout: 0.2 per layer
                - Optimizer: Adam
                - Loss: MSE
                - Look-back: 12 minggu
                """)
            
            st.markdown("---")
            
            with st.container():
                st.markdown("### Metrik Evaluasi")
                st.markdown("""
                **RMSE** (Root Mean Squared Error)  
                Mengukur rata-rata kesalahan prediksi
                
                **MAE** (Mean Absolute Error)  
                Rata-rata selisih absolut
                
                **MAPE** (Mean Absolute Percentage Error)  
                Persentase kesalahan rata-rata
                
                **R²** (R-Squared)  
                Koefisien determinasi model
                """)
            
            st.markdown("---")
            
            with st.container():
                st.markdown("### Dataset")
                st.markdown("""
                - 31 komoditas pangan
                - Periode: 2020-2025
                - Frekuensi: Mingguan
                - Split: 80% training, 20% testing
                """)
            
            st.markdown("---")
            
            with st.container():
                st.markdown("### Format Upload File")
                st.markdown("""
                **Excel (.xlsx)**
                - Kolom 1: No
                - Kolom 2: Komoditas (Rp)
                - Kolom 3+: Tanggal mingguan
                - Format harga: gunakan koma untuk ribuan
                """)
        
        # MAIN CONTENT - 4 TABS
        tab1, tab2, tab3, tab4 = st.tabs([
            "Prediksi Dataset Bawaan",
            "Evaluasi Dataset Bawaan",
            "Prediksi Dataset Input",
            "Evaluasi Dataset Input"
        ])
        
        # TAB 1: Prediksi Dataset Bawaan
        with tab1:
            st.markdown('<div class="info-box"><h3>Prediksi Menggunakan Dataset Bawaan</h3></div>', unsafe_allow_html=True)
            
            df_default = pd.read_excel('dataset.xlsx')
            commodity_list = df_default.iloc[:, 1].tolist()
            
            # Pengaturan
            with st.container():
                st.markdown("### Pengaturan Prediksi")
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_commodity = st.selectbox(
                        "Pilih Komoditas:",
                        commodity_list,
                        key="pred_default"
                    )
                
                with col2:
                    n_weeks = st.selectbox(
                        "Jumlah Minggu Prediksi:",
                        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        index=3,
                        key="weeks_default"
                    )
            
            st.markdown("---")
            
            # Proses prediksi
            commodity_idx = commodity_list.index(selected_commodity)
            commodity_name, prices_final = preprocess_commodity_data(df_default, commodity_idx)
            predictions = predict_future_prices(model, scaler, prices_final, n_weeks)
            
            # Hasil Prediksi dalam kotak warna
            with st.container():
                st.markdown("### Hasil Prediksi")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box-blue">
                        <h4>Harga Terakhir</h4>
                        <h2>Rp {prices_final[-1]:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    change_pct = ((predictions[-1] - prices_final[-1]) / prices_final[-1] * 100)
                    arrow = "↑" if change_pct > 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-box-green">
                        <h4>Prediksi {n_weeks} Minggu</h4>
                        <h2>Rp {predictions[-1]:,.0f}</h2>
                        <p>{arrow} {abs(change_pct):.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_prediction = np.mean(predictions)
                    st.markdown(f"""
                    <div class="metric-box-orange">
                        <h4>Rata-rata Prediksi</h4>
                        <h2>Rp {avg_prediction:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tabel Prediksi
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
                    line=dict(color='#2193b0', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(historical_prices), len(historical_prices) + n_weeks)),
                    y=predictions,
                    mode='lines+markers',
                    name='Prediksi',
                    line=dict(color='#f5576c', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"Tren Harga {selected_commodity}",
                    xaxis_title="Minggu",
                    yaxis_title="Harga (Rp)",
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    template='plotly_white',
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Evaluasi Dataset Bawaan
        with tab2:
            st.markdown('<div class="info-box"><h3>Evaluasi Model Dataset Bawaan</h3></div>', unsafe_allow_html=True)
            
            # Pilih komoditas
            with st.container():
                st.markdown("### Pilih Komoditas untuk Evaluasi")
                selected_eval = st.selectbox(
                    "Pilih Komoditas:",
                    commodity_list,
                    key="eval_default"
                )
            
            st.markdown("---")
            
            # Metrik Evaluasi Komoditas Terpilih
            commodity_eval = eval_df[eval_df['Komoditas'] == selected_eval]
            
            if not commodity_eval.empty:
                with st.container():
                    st.markdown(f"### Metrik Evaluasi: {selected_eval}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box-blue">
                            <h4>RMSE Training</h4>
                            <h2>{commodity_eval['Train_RMSE'].values[0]:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box-green">
                            <h4>MAE Training</h4>
                            <h2>{commodity_eval['Train_MAE'].values[0]:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-box-orange">
                            <h4>MAPE Training</h4>
                            <h2>{commodity_eval['Train_MAPE'].values[0]:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        accuracy = 100 - commodity_eval['Train_MAPE'].values[0]
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>Akurasi Training</h4>
                            <h2>{accuracy:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.markdown(f"""
                        <div class="metric-box-blue">
                            <h4>RMSE Testing</h4>
                            <h2>{commodity_eval['Test_RMSE'].values[0]:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col6:
                        st.markdown(f"""
                        <div class="metric-box-green">
                            <h4>MAE Testing</h4>
                            <h2>{commodity_eval['Test_MAE'].values[0]:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col7:
                        st.markdown(f"""
                        <div class="metric-box-orange">
                            <h4>MAPE Testing</h4>
                            <h2>{commodity_eval['Test_MAPE'].values[0]:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col8:
                        accuracy_test = 100 - commodity_eval['Test_MAPE'].values[0]
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>Akurasi Testing</h4>
                            <h2>{accuracy_test:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Statistik Keseluruhan
            with st.container():
                st.markdown("### Statistik Keseluruhan Model")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box-blue">
                        <h4>Avg RMSE Testing</h4>
                        <h2>{eval_df['Test_RMSE'].mean():.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box-green">
                        <h4>Avg MAE Testing</h4>
                        <h2>{eval_df['Test_MAE'].mean():.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box-orange">
                        <h4>Avg MAPE Testing</h4>
                        <h2>{eval_df['Test_MAPE'].mean():.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    avg_accuracy = 100 - eval_df['Test_MAPE'].mean()
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>Avg Akurasi Testing</h4>
                        <h2>{avg_accuracy:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tabel Semua Komoditas
            with st.container():
                st.markdown("### Evaluasi Semua Komoditas")
                st.dataframe(
                    eval_df[['No', 'Komoditas', 'Test_RMSE', 'Test_MAE', 'Test_MAPE']],
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("---")
            
            # Grafik Perbandingan
            with st.container():
                st.markdown("### Grafik Perbandingan Metrik")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rmse = go.Figure()
                    fig_rmse.add_trace(go.Bar(
                        x=eval_df['Komoditas'],
                        y=eval_df['Test_RMSE'],
                        marker=dict(color='#2193b0')
                    ))
                    fig_rmse.update_layout(
                        title="RMSE Testing per Komoditas",
                        xaxis_title="Komoditas",
                        yaxis_title="RMSE",
                        height=400,
                        template='plotly_white',
                        xaxis={'tickangle': 45}
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                with col2:
                    fig_mae = go.Figure()
                    fig_mae.add_trace(go.Bar(
                        x=eval_df['Komoditas'],
                        y=eval_df['Test_MAE'],
                        marker=dict(color='#11998e')
                    ))
                    fig_mae.update_layout(
                        title="MAE Testing per Komoditas",
                        xaxis_title="Komoditas",
                        yaxis_title="MAE",
                        height=400,
                        template='plotly_white',
                        xaxis={'tickangle': 45}
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)
            
            # Grafik MAPE
            with st.container():
                fig_mape = go.Figure()
                fig_mape.add_trace(go.Bar(
                    x=eval_df['Komoditas'],
                    y=eval_df['Test_MAPE'],
                    marker=dict(color='#f5576c')
                ))
                fig_mape.update_layout(
                    title="MAPE Testing per Komoditas",
                    xaxis_title="Komoditas",
                    yaxis_title="MAPE (%)",
                    height=400,
                    template='plotly_white',
                    xaxis={'tickangle': 45}
                )
                st.plotly_chart(fig_mape, use_container_width=True)
        
        # TAB 3: Prediksi Dataset Input
        with tab3:
            st.markdown('<div class="info-box"><h3>Prediksi Menggunakan Dataset Input</h3></div>', unsafe_allow_html=True)
            
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
                    
                    commodity_list_upload = df_upload.iloc[:, 1].tolist()
                    
                    with st.container():
                        st.markdown("### Pengaturan Prediksi")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            selected_commodity_upload = st.selectbox(
                                "Pilih Komoditas:",
                                commodity_list_upload,
                                key="pred_upload"
                            )
                        
                        with col2:
                            n_weeks_upload = st.selectbox(
                                "Jumlah Minggu Prediksi:",
                                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                index=3,
                                key="weeks_upload"
                            )
                    
                    if st.button("Mulai Prediksi", key="btn_predict"):
                        st.markdown("---")
                        
                        commodity_idx_upload = commodity_list_upload.index(selected_commodity_upload)
                        commodity_name_upload, prices_final_upload = preprocess_commodity_data(df_upload, commodity_idx_upload)
                        predictions_upload = predict_future_prices(model, scaler, prices_final_upload, n_weeks_upload)
                        
                        # Hasil Prediksi
                        with st.container():
                            st.markdown("### Hasil Prediksi")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-box-blue">
                                    <h4>Harga Terakhir</h4>
                                    <h2>Rp {prices_final_upload[-1]:,.0f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                change_pct_upload = ((predictions_upload[-1] - prices_final_upload[-1]) / prices_final_upload[-1] * 100)
                                arrow = "↑" if change_pct_upload > 0 else "↓"
                                st.markdown(f"""
                                <div class="metric-box-green">
                                    <h4>Prediksi {n_weeks_upload} Minggu</h4>
                                    <h2>Rp {predictions_upload[-1]:,.0f}</h2>
                                    <p>{arrow} {abs(change_pct_upload):.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                avg_prediction_upload = np.mean(predictions_upload)
                                st.markdown(f"""
                                <div class="metric-box-orange">
                                    <h4>Rata-rata Prediksi</h4>
                                    <h2>Rp {avg_prediction_upload:,.0f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Tabel
                        with st.container():
                            st.markdown("### Tabel Prediksi Mingguan")
                            
                            prediction_dates_upload = []
                            start_date_upload = datetime.now()
                            for i in range(n_weeks_upload):
                                prediction_dates_upload.append((start_date_upload + timedelta(weeks=i+1)).strftime("%d/%m/%Y"))
                            
                            pred_df_upload = pd.DataFrame({
                                'Minggu': [f"Minggu {i+1}" for i in range(n_weeks_upload)],
                                'Tanggal Prediksi': prediction_dates_upload,
                                'Harga Prediksi (Rp)': [f"{p:,.0f}" for p in predictions_upload]
                            })
                            
                            st.dataframe(pred_df_upload, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # Grafik
                        with st.container():
                            st.markdown("### Visualisasi Tren Harga")
                            
                            historical_weeks_upload = min(20, len(prices_final_upload))
                            historical_prices_upload = prices_final_upload[-historical_weeks_upload:]
                            
                            fig_upload = go.Figure()
                            
                            fig_upload.add_trace(go.Scatter(
                                x=list(range(len(historical_prices_upload))),
                                y=historical_prices_upload,
                                mode='lines+markers',
                                name='Data Historis',
                                line=dict(color='#2193b0', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig_upload.add_trace(go.Scatter(
                                x=list(range(len(historical_prices_upload), len(historical_prices_upload) + n_weeks_upload)),
                                y=predictions_upload,
                                mode='lines+markers',
                                name='Prediksi',
                                line=dict(color='#f5576c', width=3, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            fig_upload.update_layout(
                                title=f"Tren Harga {selected_commodity_upload}",
                                xaxis_title="Minggu",
                                yaxis_title="Harga (Rp)",
                                hovermode='x unified',
                                height=500,
                                showlegend=True,
                                template='plotly_white',
                                font=dict(size=12)
                            )
                            
                            st.plotly_chart(fig_upload, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Silakan upload file Excel untuk melakukan prediksi")
        
        # TAB 4: Evaluasi Dataset Input
        with tab4:
            st.markdown('<div class="info-box"><h3>Evaluasi Model Dataset Input</h3></div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown("### Upload File Dataset")
                uploaded_file_eval = st.file_uploader(
                    "Upload File Excel (.xlsx)",
                    type=['xlsx'],
                    key="upload_eval"
                )
            
            if uploaded_file_eval is not None:
                try:
                    df_upload_eval = pd.read_excel(uploaded_file_eval)
                    st.success("File berhasil diupload!")
                    
                    with st.expander("Preview Data"):
                        st.dataframe(df_upload_eval.head(10), use_container_width=True)
                    
                    st.markdown("---")
                    
                    commodity_list_eval_upload = df_upload_eval.iloc[:, 1].tolist()
                    
                    with st.container():
                        st.markdown("### Pilih Komoditas untuk Evaluasi")
                        selected_eval_upload = st.selectbox(
                            "Pilih Komoditas:",
                            commodity_list_eval_upload,
                            key="eval_upload"
                        )
                    
                    if st.button("Evaluasi Model", key="btn_eval"):
                        st.markdown("---")
                        
                        commodity_idx_eval = commodity_list_eval_upload.index(selected_eval_upload)
                        commodity_name_eval, prices_final_eval = preprocess_commodity_data(df_upload_eval, commodity_idx_eval)
                        
                        # Split data 80/20
                        train_size = int(len(prices_final_eval) * 0.8)
                        train_data_eval = prices_final_eval[:train_size]
                        test_data_eval = prices_final_eval[train_size:]
                        
                        # Prediksi untuk evaluasi
                        from sklearn.preprocessing import MinMaxScaler
                        scaler_eval = MinMaxScaler(feature_range=(0, 1))
                        train_scaled_eval = scaler_eval.fit_transform(train_data_eval.reshape(-1, 1))
                        test_scaled_eval = scaler_eval.transform(test_data_eval.reshape(-1, 1))
                        
                        # Create sequences
                        look_back = 12
                        X_test_eval = []
                        y_test_eval = []
                        for i in range(look_back, len(test_scaled_eval)):
                            X_test_eval.append(test_scaled_eval[i-look_back:i, 0])
                            y_test_eval.append(test_scaled_eval[i, 0])
                        
                        X_test_eval = np.array(X_test_eval)
                        y_test_eval = np.array(y_test_eval)
                        X_test_eval = X_test_eval.reshape(X_test_eval.shape[0], X_test_eval.shape[1], 1)
                        
                        # Prediksi
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
                            st.markdown(f"### Hasil Evaluasi: {selected_eval_upload}")
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-box-blue">
                                    <h4>RMSE</h4>
                                    <h2>{rmse_eval:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-box-green">
                                    <h4>MAE</h4>
                                    <h2>{mae_eval:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-box-orange">
                                    <h4>MAPE</h4>
                                    <h2>{mape_eval:.2f}%</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h4>R²</h4>
                                    <h2>{r2_eval:.4f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col5:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h4>Akurasi</h4>
                                    <h2>{accuracy_eval:.2f}%</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Grafik Perbandingan Actual vs Predicted
                        with st.container():
                            st.markdown("### Grafik Actual vs Predicted")
                            
                            fig_eval = go.Figure()
                            
                            fig_eval.add_trace(go.Scatter(
                                x=list(range(len(y_test_actual_eval))),
                                y=y_test_actual_eval.flatten(),
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='#2193b0', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig_eval.add_trace(go.Scatter(
                                x=list(range(len(predictions_eval))),
                                y=predictions_eval.flatten(),
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='#f5576c', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig_eval.update_layout(
                                title=f"Perbandingan Harga Actual vs Predicted - {selected_eval_upload}",
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
                                    color='#667eea',
                                    opacity=0.6
                                ),
                                name='Data Points'
                            ))
                            
                            # Perfect prediction line
                            min_val = min(y_test_actual_eval.min(), predictions_eval.min())
                            max_val = max(y_test_actual_eval.max(), predictions_eval.max())
                            fig_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', width=2, dash='dash'),
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
                st.warning("Silakan upload file Excel untuk melakukan evaluasi")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Pastikan semua file tersedia: lstm_food_price_model.h5, scaler.pkl, dataset.xlsx, lstm_evaluation_results_all_commodities.csv")

if __name__ == "__main__":
    main()
