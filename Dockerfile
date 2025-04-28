FROM ubuntu:latest
LABEL authors="vique"

ENTRYPOINT ["top", "-b"]
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código del proyecto
COPY . .

# Comando para ejecutar la API (ajusta la ruta y el nombre del archivo y la app)
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]