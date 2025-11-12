import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo de VINO
try:
    with open('modelo_wine.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Archivo 'modelo_wine.pkl' no encontrado. Asegúrate de ejecutar train_wine.py primero.")
    st.stop()

# Título y subtítulo
st.title("🍷 AI Factory: Predictor de Vinos")
st.markdown("Esta app (usando el dataset de Vinos) demuestra que el 'AI Factory' puede desplegar CUALQUIER modelo.")

# --- UI de Entrada (Sliders) ---
st.sidebar.header("Introduce las características del Vino:")

# Estos son los 4 sliders NUEVOS. 
# Los valores min/max/default se basan en la salida de train_wine.py
def user_inputs():
    proline = st.sidebar.slider('Prolina', 250, 1700, 750)
    flavanoids = st.sidebar.slider('Flavonoides', 0.3, 5.1, 2.0)
    color_intensity = st.sidebar.slider('Intensidad de Color', 1.0, 13.0, 5.0)
    alcohol = st.sidebar.slider('Alcohol (%)', 11.0, 15.0, 13.0)

    data = {
        'proline': proline,
        'flavanoids': flavanoids,
        'color_intensity': color_intensity,
        'alcohol': alcohol
    }
    # Asegurar el orden correcto de las columnas para el modelo
    features = pd.DataFrame(data, index=[0])
    features = features[['proline', 'flavanoids', 'color_intensity', 'alcohol']]
    return features

input_df = user_inputs()

# Mostrar las entradas del usuario
st.subheader('Características seleccionadas:')
st.dataframe(input_df, use_container_width=True)

# --- Predicción y Salida ---
if st.sidebar.button('¡Predecir tipo de Vino!'):
    # Convertir el dataframe a un array numpy para el modelo
    features_array = np.array(input_df)
    
    # Hacer la predicción
    prediction = model.predict(features_array)
    prediction_proba = model.predict_proba(features_array)
    
    # Mapear el resultado (3 clases, igual que Iris)
    wine_map = {0: 'Clase 0', 1: 'Clase 1', 2: 'Clase 2'}
    species = wine_map[prediction[0]]
    
    # Mostrar el resultado
    st.subheader('Resultado de la Predicción')
    st.success(f'El vino pertenece a la **{species}**.')
    
    # Mostrar confianza (probabilidades)
    st.subheader('Confianza de la Predicción')
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    proba_df = proba_df.rename(columns=wine_map).T
    proba_df.columns = ['Probabilidad']
    st.bar_chart(proba_df)