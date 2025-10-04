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
        
        # MAIN CONTENT - TABS
        tab1, tab2, tab3 = st.tabs(["Prediksi dari Dataset", "Prediksi dari Upload File", "Evaluasi Model"])
        
        # TAB 1: Prediksi dari Dataset Default
        with tab1:
            with st.container():
                st.subheader("Prediksi Menggunakan Dataset Default")
                st.info("Gunakan dataset yang sudah tersedia dalam sistem untuk melakukan prediksi")
                
                col1, col2 = st.columns(2)
                
                # Load default dataset
                df_default = pd.read_excel('dataset.xlsx')
                commodity_list = df_default.iloc[:, 1].tolist()
                
                with col1:
                    selected_commodity = st.selectbox(
                        "Pilih Komoditas:",
                        commodity_list,
                        key="default_commodity"
                    )
                
                with col2:
                    n_weeks = st.slider(
                        "Jumlah Minggu Prediksi:",
                        min_value=1,
                        max_value=12,
                        value=4,
                        key="default_weeks"
                    )
            
            st.markdown("---")
            
            commodity_idx = commodity_list.index(selected_commodity)
            commodity_name, prices_final = preprocess_commodity_data(df_default, commodity_idx)
            predictions = predict_future_prices(model, scaler, prices_final, n_weeks)
            
            with st.container():
                st.subheader("Hasil Prediksi")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### Harga Terakhir")
                    st.markdown(f"### Rp {prices_final[-1]:,.0f}")
                
                with col2:
                    change_pct = ((predictions[-1] - prices_final[-1]) / prices_final[-1] * 100)
                    st.markdown(f"##### Prediksi {n_weeks} Minggu")
                    st.markdown(f"### Rp {predictions[-1]:,.0f}")
                    if change_pct > 0:
                        st.markdown(f"<span style='color: green;'>↑ {change_pct:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color: red;'>↓ {change_pct:.2f}%</span>", unsafe_allow_html=True)
                
                with col3:
                    avg_prediction = np.mean(predictions)
                    st.markdown("##### Rata-rata Prediksi")
                    st.markdown(f"### Rp {avg_prediction:,.0f}")
            
            st.markdown("---")
            
            with st.container():
                st.subheader("Tabel Prediksi Mingguan")
                
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
            
            with st.container():
                st.subheader("Visualisasi Tren Harga")
                
                historical_weeks = min(20, len(prices_final))
                historical_prices = prices_final[-historical_weeks:]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(historical_prices))),
                    y=historical_prices,
                    mode='lines+markers',
                    name='Data Historis',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(historical_prices), len(historical_prices) + n_weeks)),
                    y=predictions,
                    mode='lines+markers',
                    name='Prediksi',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"Tren Harga {selected_commodity}",
                    xaxis_title="Minggu",
                    yaxis_title="Harga (Rp)",
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Prediksi dari Upload File
        with tab2:
            with st.container():
                st.subheader("Prediksi Menggunakan File Upload")
                st.info("Upload file Excel dengan format yang sama dengan dataset untuk melakukan prediksi custom")
                
                uploaded_file = st.file_uploader(
                    "Upload File Excel (.xlsx)",
                    type=['xlsx'],
                    help="Format: Kolom 1=No, Kolom 2=Komoditas (Rp), Kolom 3+=Tanggal mingguan"
                )
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_excel(uploaded_file)
                    
                    st.success("File berhasil diupload!")
                    
                    with st.expander("Lihat Preview Data"):
                        st.dataframe(df_upload.head(), use_container_width=True)
                    
                    st.markdown("---")
                    
                    with st.container():
                        col1, col2 = st.columns(2)
                        
                        commodity_list_upload = df_upload.iloc[:, 1].tolist()
                        
                        with col1:
                            selected_commodity_upload = st.selectbox(
                                "Pilih Komoditas:",
                                commodity_list_upload,
                                key="upload_commodity"
                            )
                        
                        with col2:
                            n_weeks_upload = st.slider(
                                "Jumlah Minggu Prediksi:",
                                min_value=1,
                                max_value=12,
                                value=4,
                                key="upload_weeks"
                            )
                    
                    if st.button("Prediksi Harga", key="btn_predict_upload"):
                        st.markdown("---")
                        
                        commodity_idx_upload = commodity_list_upload.index(selected_commodity_upload)
                        commodity_name_upload, prices_final_upload = preprocess_commodity_data(df_upload, commodity_idx_upload)
                        predictions_upload = predict_future_prices(model, scaler, prices_final_upload, n_weeks_upload)
                        
                        with st.container():
                            st.subheader("Hasil Prediksi")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("##### Harga Terakhir")
                                st.markdown(f"### Rp {prices_final_upload[-1]:,.0f}")
                            
                            with col2:
                                change_pct_upload = ((predictions_upload[-1] - prices_final_upload[-1]) / prices_final_upload[-1] * 100)
                                st.markdown(f"##### Prediksi {n_weeks_upload} Minggu")
                                st.markdown(f"### Rp {predictions_upload[-1]:,.0f}")
                                if change_pct_upload > 0:
                                    st.markdown(f"<span style='color: green;'>↑ {change_pct_upload:.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: red;'>↓ {change_pct_upload:.2f}%</span>", unsafe_allow_html=True)
                            
                            with col3:
                                avg_prediction_upload = np.mean(predictions_upload)
                                st.markdown("##### Rata-rata Prediksi")
                                st.markdown(f"### Rp {avg_prediction_upload:,.0f}")
                        
                        st.markdown("---")
                        
                        with st.container():
                            st.subheader("Tabel Prediksi Mingguan")
                            
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
                        
                        with st.container():
                            st.subheader("Visualisasi Tren Harga")
                            
                            historical_weeks_upload = min(20, len(prices_final_upload))
                            historical_prices_upload = prices_final_upload[-historical_weeks_upload:]
                            
                            fig_upload = go.Figure()
                            
                            fig_upload.add_trace(go.Scatter(
                                x=list(range(len(historical_prices_upload))),
                                y=historical_prices_upload,
                                mode='lines+markers',
                                name='Data Historis',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig_upload.add_trace(go.Scatter(
                                x=list(range(len(historical_prices_upload), len(historical_prices_upload) + n_weeks_upload)),
                                y=predictions_upload,
                                mode='lines+markers',
                                name='Prediksi',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            fig_upload.update_layout(
                                title=f"Tren Harga {selected_commodity_upload}",
                                xaxis_title="Minggu",
                                yaxis_title="Harga (Rp)",
                                hovermode='x unified',
                                height=500,
                                showlegend=True,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_upload, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error membaca file: {str(e)}")
                    st.info("Pastikan format file sesuai dengan contoh dataset")
            else:
                st.warning("Silakan upload file Excel untuk melakukan prediksi")
        
        # TAB 3: Evaluasi Model
        with tab3:
            with st.container():
                st.subheader("Evaluasi Model")
                
                df_default = pd.read_excel('dataset.xlsx')
                commodity_list_eval = df_default.iloc[:, 1].tolist()
                
                selected_eval = st.selectbox(
                    "Pilih Komoditas untuk Evaluasi:",
                    commodity_list_eval,
                    key="eval_select"
                )
                
                commodity_eval = eval_df[eval_df['Komoditas'] == selected_eval]
                
                if not commodity_eval.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("##### RMSE Testing")
                        st.markdown(f"### {commodity_eval['Test_RMSE'].values[0]:.2f}")
                    
                    with col2:
                        st.markdown("##### MAE Testing")
                        st.markdown(f"### {commodity_eval['Test_MAE'].values[0]:.2f}")
                    
                    with col3:
                        st.markdown("##### MAPE Testing")
                        st.markdown(f"### {commodity_eval['Test_MAPE'].values[0]:.2f}%")
            
            st.markdown("---")
            
            with st.container():
                st.subheader("Statistik Keseluruhan Model")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### Rata-rata RMSE")
                    st.markdown(f"### {eval_df['Test_RMSE'].mean():.2f}")
                
                with col2:
                    st.markdown("##### Rata-rata MAE")
                    st.markdown(f"### {eval_df['Test_MAE'].mean():.2f}")
                
                with col3:
                    st.markdown("##### Rata-rata MAPE")
                    st.markdown(f"### {eval_df['Test_MAPE'].mean():.2f}%")
            
            st.markdown("---")
            
            with st.container():
                st.subheader("Evaluasi Semua Komoditas")
                st.dataframe(
                    eval_df[['No', 'Komoditas', 'Test_RMSE', 'Test_MAE', 'Test_MAPE']],
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("---")
            
            with st.container():
                st.subheader("Top 10 Komoditas dengan MAPE Terendah")
                
                top_10 = eval_df.nsmallest(10, 'Test_MAPE')[['Komoditas', 'Test_MAPE']]
                
                fig_top10 = go.Figure(go.Bar(
                    x=top_10['Test_MAPE'],
                    y=top_10['Komoditas'],
                    orientation='h',
                    marker=dict(color='#2ca02c')
                ))
                
                fig_top10.update_layout(
                    title="Top 10 Komoditas Berdasarkan MAPE",
                    xaxis_title="MAPE (%)",
                    yaxis_title="Komoditas",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_top10, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Pastikan semua file model tersedia: lstm_food_price_model.h5, scaler.pkl, dataset.xlsx, lstm_evaluation_results_all_commodities.csv")

if __name__ == "__main__":
    main()
