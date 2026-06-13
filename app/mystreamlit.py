import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción de Depósitos", page_icon="🏦")
st.title("🏦 Predicción de Subscripción a Depósito a Plazo")
st.write("Introduce los datos del cliente para predecir si contratará el producto bancario.")

# Cargamos el modelo y los datos
@st.cache_resource
def load_model():
    return joblib.load('modelo_final.joblib')

@st.cache_data
def load_data():
    return pd.read_pickle('bank_08.pkl')

try:
    modelo = load_model()
    df_base = load_data()
except FileNotFoundError:
    st.error("Falta un archivo necesario ('modelo_final.joblib' o 'bank_08.pkl').")
    st.stop()

# Usamos una función auxiliar para obtener las categorías de los datos
def get_options(column_name):
    options = list(df_base[column_name].dropna().unique())
    if 'unknown' not in options:
        options.append('unknown')
    return sorted(options)

# Definimos la interfaz de usuario a utilizar
with st.sidebar:
    st.header("Datos del Cliente")

    age = st.slider("Edad (age)", int(df_base['age'].min()), int(df_base['age'].max()), 35)
    job = st.selectbox("Trabajo (job)", get_options('job'))
    marital = st.selectbox("Estado civil (marital)", get_options('marital'))
    education = st.selectbox("Educación (education)", get_options('education'))
    default = st.selectbox("¿Crédito en mora? (default)", get_options('default'))
    balance = st.number_input("Balance (balance)", value=1500)
    housing = st.selectbox("¿Préstamo hipotecario? (housing)", get_options('housing'))
    loan = st.selectbox("¿Préstamo personal? (loan)", get_options('loan'))
    contact = st.selectbox("Tipo de contacto (contact)", get_options('contact'))
    day = st.number_input("Día del mes (day)", min_value=1, max_value=31, value=15)
    month = st.selectbox("Mes de contacto (month)", df_base['month'].dropna().unique())
    duration = st.number_input("Duración último contacto (seg)", value=250)
    campaign = st.number_input("Contactos esta campaña", min_value=1, value=1)
    # pdays: el usuario introduce -1 para "sin contacto previo" [cite: 1]
    pdays = st.number_input("Días desde contacto anterior (-1 = nunca)", value=-1)
    previous = st.number_input("Contactos previos", min_value=0, value=0)
    poutcome = st.selectbox("Resultado campaña anterior (poutcome)", get_options('poutcome'))

# Definimos la lógica de predicción
if st.button("Realizar Predicción", type="primary"):
    input_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing,
        'loan': loan, 'contact': contact, 'day': day, 'month': month,
        'duration': duration, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }

    df_input = pd.DataFrame([input_data])

    # Realizamos el preproceso manual, idéntico al realizado en eda.ipynb
    # Creamos la variable indicadora 'previous_contact'
    df_input['previous_contact'] = np.where(df_input['pdays'] == -1, 0, 1)
    # Convertimos -1 a NaN
    df_input.loc[df_input['pdays'] == -1, 'pdays'] = np.nan

    # Realizamos la predicción con el mejor modelo guardado
    prediccion = modelo.predict(df_input)[0]

    st.divider()
    if prediccion == 1 or prediccion == 'yes':
        st.success("✅ ¡El cliente SÍ contratará el depósito!")
        st.balloons()
    else:
        st.error("❌ El cliente NO contratará el depósito.")