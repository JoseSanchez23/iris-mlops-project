from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
import os
from typing import Dict, List, Optional
import joblib

app = FastAPI(
    title="Iris Classification API",
    description="API para clasificar especies de flores Iris",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

    @validator('*')
    def check_positive(cls, v, values, **kwargs):
        if v <= 0:
            raise ValueError('Los valores deben ser positivos')
        return v


class IrisPrediction(BaseModel):
    species: str
    probability: float
    species_probabilities: Dict[str, float]


MODEL_PATH = os.getenv("MODEL_PATH", "../models/iris_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "../models/scaler.pkl")


def get_model_and_scaler():
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail=f"Modelo no encontrado en {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise HTTPException(status_code=500, detail=f"Scaler no encontrado en {SCALER_PATH}")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return {"model": model, "scaler": scaler}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando el modelo o scaler: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de clasificación de Iris", "status": "online"}


@app.post("/predict", response_model=IrisPrediction)
async def predict(features: IrisFeatures, models=Depends(get_model_and_scaler)):
    try:
        model = models["model"]
        scaler = models["scaler"]

        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])

        # Escalar los datos de entrada
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)[0]
        probabilities = model.predict_proba(input_data_scaled)[0]

        # Mapear las clases numéricas a nombres
        species_list = model.classes_
        species_name = prediction

        # Crear el diccionario de probabilidades
        prob_dict = {class_name: float(prob) for class_name, prob in zip(species_list, probabilities)}

        return {
            "species": species_name,
            "probability": float(max(probabilities)),
            "species_probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@app.get("/health")
async def health(models=Depends(get_model_and_scaler)):
    return {"status": "ok", "model_loaded": models["model"] is not None, "scaler_loaded": models["scaler"] is not None}


@app.get("/metadata")
async def metadata():
    return {
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_names": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "description": "Dataset de flores Iris para clasificación de especies"
    }