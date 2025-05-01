# Proyecto MLOps - Clasificación de Iris

Este proyecto implementa un sistema de clasificación de especies de flores Iris utilizando machine learning, siguiendo un enfoque de MLOps completo y moderno.

## Integrantes del Equipo y Roles

**Científico de Datos**  
_Responsable_: Jose Ignacio Sánchez  
- Desarrollo y exploración de datos  
- Entrenamiento y evaluación del modelo de machine learning

**Ingeniera MLOps**  
_Responsable_: Ariana Víquez Solano  
- Versionamiento de datos y modelos con **DVC**  
- Integración de almacenamiento remoto en **AWS S3**  
- Contenerización con **Docker**  
- Documentación del flujo reproducible  
- Base para CI/CD

**Ingeniera de Software**  
_Responsable_: Maria Fernanda Moroney Sole  
- Desarrollo de la API (**FastAPI**)  
- Interfaz web (**Streamlit**)  
- CI/CD con **GitHub Actions**  
- Despliegue en **AWS EC2**  
- Documentación técnica

---

## Descripción del Proyecto

Se utiliza el dataset clásico de Iris para clasificar flores en tres especies (setosa, versicolor, virginica) a partir de cuatro características morfológicas.

---

## Tecnologías Utilizadas

- **Python**
- **Random Forest** (modelo ML)
- **FastAPI** (API)
- **Streamlit** (UI)
- **Docker & Docker Compose**
- **Git & GitHub**
- **DVC** (versionamiento de datos/modelos)
- **AWS S3** (`s3://iris-mlops-project/iris-mlops/`)
- **GitHub Actions** (CI/CD)
- **AWS EC2** (`i-0adbb2e8fbf50c4e5`)

---

## Estructura del Proyecto

iris-mlops-project/
├── .github/workflows/ # Workflows de GitHub Actions
├── api/ # Código de la API (FastAPI)
├── data/ # Datos (versionados con DVC)
├── models/ # Modelos entrenados (versionados con DVC)
├── notebooks/ # Jupyter notebooks
│ └── iris_model_dev.ipynb
├── scripts/ # Scripts utilitarios
│ └── retrain.py
├── tests/ # Pruebas unitarias
├── .dvc/ # Configuración de DVC
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── streamlit_app.py


---

## Instrucciones de Ejecución

### Requisitos Previos

- Python 3.9+
- Docker y Docker Compose
- Git
- AWS CLI (opcional, para acceso a S3)

### Clonar el Repositorio

```bash
git clone https://github.com/JoseSanchez23/iris-mlops-project.git
cd iris-mlops-project
Configuración del Entorno Local
bash
Copy Code
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
Obtener Datos y Modelos (DVC + S3)
bash
Copy Code
pip install dvc dvc[s3]
aws configure  # Ingresa tus credenciales AWS
dvc pull
Los datos y modelos se almacenan en: s3://iris-mlops-project/iris-mlops/

Ejecutar la API Localmente
bash
Copy Code
uvicorn api.main:app --reload
API: http://localhost:8000
Docs: http://localhost:8000/docs
Ejecutar la Interfaz Web
bash
Copy Code
streamlit run streamlit_app.py
UI: http://localhost:8501
Reentrenar el Modelo
bash
Copy Code
python scripts/retrain.py
Ejecutar con Docker Compose
bash
Copy Code
docker-compose up -d
Documentación del Modelo y API
Inputs del Modelo
sepal_length: Longitud del sépalo (cm, float)
sepal_width: Ancho del sépalo (cm, float)
petal_length: Longitud del pétalo (cm, float)
petal_width: Ancho del pétalo (cm, float)
Ejemplo de entrada:

json
Copy Code
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Outputs del Modelo
species: Especie predicha
probability: Probabilidad de la predicción
species_probabilities: Probabilidades por especie
Ejemplo de salida:

json
Copy Code
{
  "species": "Iris-setosa",
  "probability": 0.98,
  "species_probabilities": {
    "Iris-setosa": 0.98,
    "Iris-versicolor": 0.01,
    "Iris-virginica": 0.01
  }
}
Endpoints de la API
GET /: Mensaje de bienvenida
POST /predict: Predicción de especie
GET /health: Estado de la API
GET /metadata: Info del modelo
Interfaz Gráfica (Streamlit)
Permite predicciones interactivas, visualización de probabilidades y comparación con datos típicos.

Flujo de trabajo MLOps
Versionamiento de datos y modelos con DVC + S3
bash
Copy Code
dvc add data/new_data.csv
dvc add models/new_model.pkl
dvc push
git add data/new_data.csv.dvc models/new_model.pkl.dvc
git commit -m "Añadir nuevos datos y modelo"
git push
Proceso de Reentrenamiento
bash
Copy Code
python scripts/retrain.py
dvc add models/iris_model.pkl
dvc add models/scaler.pkl
dvc push
git add models/iris_model.pkl.dvc models/scaler.pkl.dvc
git commit -m "Actualizar modelo"
git push
CI/CD con GitHub Actions
Pruebas automáticas en cada push/PR
Reentrenamiento periódico
Despliegue automático a EC2
Despliegue
La aplicación está desplegada en AWS EC2:

API: http://18.117.151.103:8000
Docs: http://18.117.151.103:8000/docs
Web: http://18.117.151.103:8501
Instancia EC2: i-0adbb2e8fbf50c4e5
Bucket S3: s3://iris-mlops-project/iris-mlops/

Licencia
MIT

Licencia
MIT
