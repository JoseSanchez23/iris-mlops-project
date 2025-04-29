import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Clasificador de Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Clasificación de Especies de Flores Iris")
st.write("""
Esta aplicación predice la especie de una flor Iris basándose en las medidas de su sépalo y pétalo.
Proporciona las medidas y el modelo de machine learning predicirá si la flor es **Setosa**, **Versicolor** o **Virginica**.
""")


def predict_species(features):
    api_url = "http://localhost:8000/predict"

    try:
        response = requests.post(api_url, json=features)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error conectando con la API: {str(e)}")
        return None


def plot_distribution():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(url, header=None, names=column_names)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribución de Características por Especie', fontsize=16)

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for i, feature in enumerate(features):
        row, col = i // 2, i % 2
        sns.violinplot(x='species', y=feature, data=df, ax=axes[row, col])
        axes[row, col].set_title(f'{feature.replace("_", " ").title()} por Especie')
        axes[row, col].set_xlabel('Especie')
        axes[row, col].set_ylabel(f'{feature.replace("_", " ").title()} (cm)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def get_species_image(species):
    species_images = {
        "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/800px-Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg",
        "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/800px-Iris_virginica.jpg"
    }

    if species in species_images:
        return species_images[species]
    else:
        return None


st.sidebar.header("Parámetros de entrada")

sepal_length = st.sidebar.slider("Longitud del sépalo (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width = st.sidebar.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Longitud del pétalo (cm)", 1.0, 7.0, 4.7, 0.1)
petal_width = st.sidebar.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.3, 0.1)

predict_button = st.sidebar.button("Predecir Especie")

st.sidebar.subheader("Valores Seleccionados:")
st.sidebar.write(f"Longitud del sépalo: {sepal_length} cm")
st.sidebar.write(f"Ancho del sépalo: {sepal_width} cm")
st.sidebar.write(f"Longitud del pétalo: {petal_length} cm")
st.sidebar.write(f"Ancho del pétalo: {petal_width} cm")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Visualización de Datos")
    fig = plot_distribution()
    st.pyplot(fig)

    st.subheader("Tu Flor en el Contexto")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(url, header=None, names=column_names)

    user_point = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width],
        'species': ['Tu Flor']
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x='petal_length', y='petal_width', hue='species',
                    data=df, ax=ax, alpha=0.6, s=80)

    sns.scatterplot(x='petal_length', y='petal_width', data=user_point,
                    color='red', s=200, label='Tu Flor', marker='*')

    ax.set_title('Tu Flor Comparada con el Dataset de Iris')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Resultado de la Predicción")

    if predict_button:
        with st.spinner("Obteniendo predicción..."):
            features = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }

            result = predict_species(features)

            if result:
                st.success(
                    f"La flor es **{result['species'].upper()}** con una probabilidad del {round(result['probability'] * 100, 2)}%")

                image_url = get_species_image(result['species'])
                if image_url:
                    st.image(image_url, caption=f"Iris {result['species']}", width=300)

                st.subheader("Probabilidades por Especie")
                probs = result.get('species_probabilities', {})
                if probs:
                    probabilities_df = pd.DataFrame({
                        'Especie': list(probs.keys()),
                        'Probabilidad': list(probs.values())
                    })

                    fig, ax = plt.subplots(figsize=(8, 5))
                    bars = sns.barplot(x='Especie', y='Probabilidad', data=probabilities_df, ax=ax)

                    for i, p in enumerate(bars.patches):
                        height = p.get_height()
                        bars.text(p.get_x() + p.get_width() / 2., height + 0.01,
                                  f'{height:.1%}', ha="center")

                    plt.ylim(0, 1.1)
                    plt.title('Probabilidad por Especie')
                    plt.tight_layout()
                    st.pyplot(fig)

                st.subheader("Características Típicas")

                species_data = {
                    "Iris-setosa": [5.1, 3.5, 1.4, 0.2],
                    "Iris-versicolor": [5.9, 2.8, 4.3, 1.3],
                    "Iris-virginica": [6.6, 3.0, 5.6, 2.0]
                }

                if result['species'] in species_data:
                    comparison_data = {
                        'Característica': ['Longitud del Sépalo', 'Ancho del Sépalo', 'Longitud del Pétalo',
                                           'Ancho del Pétalo'],
                        'Tu Flor': [sepal_length, sepal_width, petal_length, petal_width],
                        f'Promedio {result["species"]}': species_data[result['species']]
                    }

                    comparison_df = pd.DataFrame(comparison_data)
                    st.table(comparison_df)
    else:
        st.info("Ajusta los parámetros y haz clic en 'Predecir Especie' para obtener una predicción.")

        st.image("https://cdn.pixabay.com/photo/2019/04/20/11/39/flower-4141580_1280.jpg", caption="Especies de Iris",
                 width=300)

st.markdown("---")
st.header("Información Adicional")

tab1, tab2, tab3 = st.tabs(["Acerca del Proyecto", "Dataset", "Modelo"])

with tab1:
    st.markdown("""
    ## Proyecto MLOps para Clasificación de Flores Iris

    Este proyecto implementa un pipeline completo de MLOps para la clasificación de flores Iris basado en sus características.

    ### Características del Proyecto:

    - **API REST**: Construida con FastAPI para consumir el modelo
    - **Versionamiento de Datos y Modelos**: Usando DVC
    - **CI/CD**: Implementado con GitHub Actions
    - **Conteneirización**: Docker y Docker Compose
    - **Almacenamiento**: AWS S3 para datos y modelos
    - **Despliegue**: AWS EC2
    """)

with tab2:
    st.markdown("""
    ## Conjunto de Datos de Iris

    El conjunto de datos de Iris es uno de los más famosos en machine learning. Contiene 150 muestras de flores Iris de tres especies:

    - **Setosa**
    - **Versicolor**
    - **Virginica**

    Cada muestra tiene cuatro características:

    1. Longitud del sépalo (cm)
    2. Ancho del sépalo (cm)
    3. Longitud del pétalo (cm)
    4. Ancho del pétalo (cm)
    """)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(url, header=None, names=column_names)
    st.dataframe(df.head(10))

with tab3:
    st.markdown("""
    ## Modelo de Clasificación

    Para este proyecto se utiliza un modelo de **Random Forest** con las siguientes características:

    - **Algoritmo**: Random Forest Classifier
    - **Número de Estimadores**: 100
    - **Preprocesamiento**: StandardScaler
    - **Características Utilizadas**: Las 4 medidas de la flor
    """)

    st.subheader("Matriz de Confusión (Ejemplo)")
    confusion_matrix = np.array([
        [16, 0, 0],
        [0, 17, 1],
        [0, 1, 15]
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    st.pyplot(fig)

st.markdown("---")
st.markdown("<div style='text-align: center'><p>Desarrollado para el Proyecto de MLOps | 2025</p></div>",
            unsafe_allow_html=True)