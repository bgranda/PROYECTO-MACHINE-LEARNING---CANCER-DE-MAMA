import streamlit as st
import pandas as pd
import joblib

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Predicción Supervivencia Cáncer de Mama",
    page_icon="🎀",
    layout="wide"
)

# =====================================================
# CARGAR MODELO
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("../models/trained_model.pkl")

model = load_model()

# =====================================================
# TÍTULO
# =====================================================

st.title("🎀 Predicción de Supervivencia en Cáncer de Mama")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning entrenado con datos del registro SEER
para estimar los meses de supervivencia de una paciente.
""")

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.radio(
    "Menú",
    [
        "Predicción",
        "Información del Proyecto"
    ]
)

# =====================================================
# PÁGINA PREDICCIÓN
# =====================================================

if menu == "Predicción":

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Edad",
            min_value=20,
            max_value=100,
            value=60
        )

        tumor_size = st.number_input(
            "Tamaño del tumor (mm)",
            min_value=1,
            max_value=300,
            value=30
        )

        regional_examined = st.number_input(
            "Ganglios examinados",
            min_value=0,
            max_value=100,
            value=15
        )

        regional_positive = st.number_input(
            "Ganglios positivos",
            min_value=0,
            max_value=80,
            value=2
        )

        race = st.selectbox(
            "Raza",
            ["White", "Black", "Other"]
        )

        marital_status = st.selectbox(
            "Estado civil",
            [
                "Married",
                "Single",
                "Divorced",
                "Widowed",
                "Separated"
            ]
        )

    with col2:

        t_stage = st.selectbox(
            "T Stage",
            ["T1", "T2", "T3", "T4"]
        )

        n_stage = st.selectbox(
            "N Stage",
            ["N1", "N2", "N3"]
        )

        grade = st.selectbox(
            "Grade",
            [
                "Grade I",
                "Grade II",
                "Grade III",
                "Grade IV"
            ]
        )

        estrogen_status = st.selectbox(
            "Estado receptor estrógeno",
            ["Positive", "Negative"]
        )

        progesterone_status = st.selectbox(
            "Estado receptor progesterona",
            ["Positive", "Negative"]
        )

    # =================================================
    # BOTÓN PREDICCIÓN
    # =================================================

    if st.button("🔍 Calcular Predicción"):

        paciente = pd.DataFrame({
            "Age": [age],
            "Tumor Size": [tumor_size],
            "Regional Node Examined": [regional_examined],
            "Regional Node Positive": [regional_positive],
            "Race": [race],
            "Marital Status": [marital_status],
            "T Stage": [t_stage],
            "N Stage": [n_stage],
            "Grade": [grade],
            "Estrogen Status": [estrogen_status],
            "Progesterone Status": [progesterone_status]
        })

        prediction = model.predict(paciente)[0]

        st.divider()

        st.subheader("Resultado")

        st.metric(
            "Supervivencia estimada",
            f"{prediction:.1f} meses"
        )

        years = prediction / 12

        st.metric(
            "Equivalente aproximado",
            f"{years:.1f} años"
        )

        if prediction < 50:
            st.error("Pronóstico estimado inferior a la media del dataset.")
        elif prediction < 85:
            st.warning("Pronóstico estimado intermedio.")
        else:
            st.success("Pronóstico estimado favorable.")

# =====================================================
# INFORMACIÓN
# =====================================================

else:

    st.header("📖 Información del proyecto")

    st.markdown("""
### Objetivo

Predecir los meses de supervivencia de pacientes con cáncer de mama
utilizando técnicas de Machine Learning.

### Dataset

Fuente: SEER (Surveillance, Epidemiology and End Results Program).

### Variables utilizadas

- Edad
- Tamaño tumoral
- Estadificación TNM
- Ganglios regionales
- Estado hormonal
- Grado tumoral

### Variable objetivo

**Survival Months**
    """)