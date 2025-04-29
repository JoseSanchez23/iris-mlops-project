# Proyecto MLOps - Clasificación de Iris

## Científico de Datos
Responsable: [Jose Ignacio Sánchez]

Este proyecto utiliza el dataset Iris para construir un modelo de clasificación.

## Estructura
- `notebooks/iris_model_dev.ipynb`: desarrollo, exploración y entrenamiento del modelo.
- `scripts/retrain.py`: reentrenamiento automático del modelo.
- `models/`: carpeta donde se guarda el modelo entrenado.
- `data/`: contiene el dataset original.

## Cómo ejecutar el reentrenamiento

```bash
python scripts/retrain.py


## Ingeniero MLOps  
Responsable: Ariana Víquez Solano

El Ingeniero MLOps se encargó de:

- Configurar el versionamiento de datos y modelos usando **DVC**.
- Integrar almacenamiento remoto en **AWS S3** para los archivos versionados.
- Crear y probar el **Dockerfile** para contenerizar la aplicación y facilitar su despliegue.
- Documentar el flujo de trabajo reproducible para todo el equipo.
- Preparar la base para la integración continua y despliegue automatizado (CI/CD).

### Flujo de trabajo MLOps

1. **Versionamiento de datos y modelos con DVC**
   - Los archivos grandes (datasets, modelos) se versionan con DVC y se almacenan en AWS S3.
   - Para descargar los datos y modelos versionados:
     ```bash
     dvc pull
     ```
2. **Contenerización**
   - El proyecto incluye un `Dockerfile` para facilitar la ejecución en cualquier entorno.
   - Para construir y correr la imagen Docker:
     ```bash
     docker build -t iris-mlops-app .
     docker run -p 8000:8000 iris-mlops-app
     ```
   - (Ajusta el puerto y el comando según tu API o notebook.)

3. **Automatización y despliegue**
   - El proyecto está preparado para integrarse con pipelines de CI/CD (por ejemplo, GitHub Actions) para pruebas, construcción y despliegue automático.

---

## Estructura

- `notebooks/iris_model_dev.ipynb`: desarrollo, exploración y entrenamiento del modelo.
- `scripts/retrain.py`: reentrenamiento automático del modelo.
- `models/`: carpeta donde se guarda el modelo entrenado.
- `data/`: contiene el dataset original.
- `Dockerfile`: definición de la imagen para contenerización.
- `.dvc/`, `.dvc` y archivos relacionados: control de versiones de datos y modelos.

---

## Cómo ejecutar el reentrenamiento

```bash
python scripts/retrain.py
