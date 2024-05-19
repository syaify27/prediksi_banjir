import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import catboost
# Definisikan path ke folder model yang disimpan
#output_dir = "/content/drive/MyDrive/kuliah/urban analitik/tugas besar kelompok/HASIL"

# Nama model yang tersedia
model_names = ['RandomForest', 'GradientBoosting', 'AdaBoost', 'DecisionTree', 'XGBoost', 'SVR', 'KNeighbors', 'Linear', 'CatBoost']

# Membuat dictionary untuk menyimpan model yang dimuat
loaded_models = {}

# Memuat model yang telah disimpan
for name in model_names:
    if name == 'Linear':#tidak menampilkan hasil dari Linear karna sangat berbeda dengan yang lain
        pass
    else:
        model_path = os.path.join(f'{name}_model.pkl')
        loaded_models[name] = joblib.load(model_path)

# Membuat antarmuka Streamlit
st.title("Flood Prediction App")
st.title("URBAN ANALITIK KEL 5 SIKC 2024")
st.write("https://github.com/syaify27/prediksi_banjir/")

st.header("Input Features")
# Membuat input fields untuk setiap fitur
CH = st.number_input("CURAH HUJAN", value=0.0)
HH = st.number_input("HARI HUJAN", value=0.0)
KA = st.number_input("KECEPATAN ANGIN", value=0.0)
KU = st.number_input("KELEMBABAN UDARA", value=0.0)
PM = st.number_input("PENYINARAN MATAHARI", value=0.0)
TU = st.number_input("TEKANN UDARA", value=0.0)
TEMP = st.number_input("TEMPERATUR", value=0.0)

# Menyimpan input ke dalam DataFrame
input_data = pd.DataFrame({
    'CH': [CH],
    'HH': [HH],
    'KA': [KA],
    'KU': [KU],
    'PM': [PM],
    'TU': [TU],
    'TEMP': [TEMP]
})

# Menampilkan DataFrame input
st.write("Input Data:")
st.write(input_data)

# Tombol untuk melakukan prediksi
if st.button("Predict"):
    predictions = {}
    
    # Membuat prediksi menggunakan setiap model yang dimuat
    for name, model in loaded_models.items():
        predictions[name] = model.predict(input_data)[0]
    
    # Menampilkan hasil prediksi
    st.write("Prediksi Kejadian Banjir Kota Bandung dalam 1 Bulan")
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])
    st.write(predictions_df)

    # Menyimpan prediksi ke dalam file CSV
   # predictions_output_path = os.path.join(output_dir, 'predictions_streamlit.csv')
   # predictions_df.to_csv(predictions_output_path, index=False)
   # st.write(f'Predictions have been saved to {predictions_output_path}')
