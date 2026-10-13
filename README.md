# Q-MTL-XAI Backend

Backend API para clasificacion de ECG (arritmias + isquemia).

## Endpoints

- GET / - Info
- GET /health - Estado
- POST /predict - Prediccion

## Tasks

- latido: 250 muestras (normal, paced, PVC, SPB)
- ventana: 750 muestras (normal, paced, PVC, SPB)
- isc: 12 leads x 2500 muestras (normal, ischemia)

## SQI

Se usa el modelo entrenado c3_sqi_ecg.pkl (F1 macro 0.92).
