# app.py
# Backend FastAPI para el modelo Q-MTL-XAI

import os
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

from models import MultiTaskEncoder
from preprocessing import preprocess_ecg
from sqi import predict_sqi_ecg


# === Configuración ===
DEVICE = torch.device('cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'c3_encoder_unificado_best.pt')

LATIDO_CLASSES = ['normal', 'paced', 'PVC', 'SPB']
VENTANA_CLASSES = ['normal', 'paced', 'PVC', 'SPB']
ISC_CLASSES = ['normal', 'ischemia']


# === Cargar modelo ===
print("Cargando modelo...")
model = MultiTaskEncoder()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Modelo cargado")


# === API ===
app = FastAPI(
    title="Q-MTL-XAI Backend",
    description="API para clasificacion de ECG (arritmias + isquemia)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    signal: Union[List[float], List[List[float]]]
    task: str = "latido"
    sampling_rate: int = 250


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    sqi: str
    task: str


@app.get("/")
def root():
    return {
        "message": "Q-MTL-XAI Backend",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        if request.task not in ['latido', 'ventana', 'isc']:
            raise HTTPException(status_code=400, detail=f"Task invalido: {request.task}")

        signal = np.array(request.signal, dtype=np.float32)

        if request.task == 'isc':
            if signal.ndim == 1:
                # Reconstruir 2D si tiene el tamaño correcto (12 leads x 2500)
                if len(signal) == 12 * 2500:
                    signal = signal.reshape(12, 2500)
                else:
                    signal = signal.reshape(1, -1)
            processed = preprocess_ecg(signal, request.sampling_rate,
                                       fs_target=250, window_size=2500)
            if processed.shape[0] != 12:
                if processed.shape[0] < 12:
                    processed = np.tile(processed, (12 // processed.shape[0] + 1, 1))[:12]
                else:
                    processed = processed[:12]
            x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
            classes = ISC_CLASSES
        else:
            window_size = 250 if request.task == 'latido' else 750
            processed = preprocess_ecg(signal, request.sampling_rate,
                                       fs_target=250, window_size=window_size)
            x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            classes = LATIDO_CLASSES if request.task == 'latido' else VENTANA_CLASSES

        # SQI con modelo entrenado
        if request.task != 'isc':
            sqi_value = predict_sqi_ecg(processed.flatten())
        else:
            sqi_value = predict_sqi_ecg(processed[0])  # Lead II

        with torch.no_grad():
            out = model(x, task=request.task)
            probs = F.softmax(out, dim=1).numpy()[0]
            pred_idx = int(probs.argmax())

        return PredictResponse(
            prediction=classes[pred_idx],
            confidence=float(probs[pred_idx]),
            probabilities={cls: float(p) for cls, p in zip(classes, probs)},
            sqi=sqi_value,
            task=request.task,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
