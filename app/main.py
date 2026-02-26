from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import onnxruntime as ort
import numpy as np
import json
import os
import logging
from datetime import datetime
from pathlib import Path

try:
    from .schemas import CarPredictionRequest, CarPredictionResponse, HealthResponse
except ImportError:
    from schemas import CarPredictionRequest, CarPredictionResponse, HealthResponse

# Paths
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODEL_PATH = os.getenv('MODEL_PATH', str(BASE_DIR / 'model' / 'model.onnx'))
METADATA_PATH = os.getenv('METADATA_PATH', str(BASE_DIR / 'model' / 'metadata.json'))
FRONTEND_PATH = BASE_DIR / 'frontend' / 'index.html'

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EcoPriceAPI')

app = FastAPI(title='Eco-Price Estimator')

# State
session = None
metadata = None
logs = []

def load_resources():
    global session, metadata
    try:
        session = ort.InferenceSession(MODEL_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        logger.info('Model loaded successfully.')
    except Exception as e:
        logger.error(f'Failed to load model: {e}')

@app.on_event('startup')
async def startup():
    load_resources()

@app.get('/health', response_model=HealthResponse)
async def health():
    if not session: raise HTTPException(status_code=503)
    return {'status': 'healthy', 'model_loaded': True, 'r2_score': metadata.get('r2_score', 0)}

@app.post('/predict', response_model=CarPredictionResponse)
async def predict(req: CarPredictionRequest):
    if not session: raise HTTPException(status_code=503)
    
    inputs = {
        'brand': np.array([[req.brand]], dtype=object),
        'mileage': np.array([[float(req.mileage)]], dtype=np.float32),
        'year': np.array([[float(req.year)]], dtype=np.float32),
        'fuel_type': np.array([[req.fuel_type]], dtype=object)
    }
    
    res = session.run(None, inputs)
    price = float(res[0][0][0])
    
    # Simple drift detection / outlier check
    is_outlier = req.year < 2000 or req.mileage > 500000 or req.brand not in metadata.get('brands', [])
    
    log_entry = {
        'ts': datetime.now().isoformat(), 
        'request': {'brand': req.brand, 'mileage': req.mileage, 'year': req.year},
        'price': round(price, 2), 
        'is_outlier': is_outlier
    }
    logs.append(log_entry)
    if len(logs) > 100: logs.pop(0)
    
    return {
        'predicted_price': round(price, 2), 
        'model_version': '1.0', 
        'confidence_score': 0.95 if not is_outlier else 0.5
    }

@app.get('/monitoring')
async def get_monitoring():
    outlier_count = sum(1 for log in logs if log['is_outlier'])
    return {
        'total_requests': len(logs),
        'outlier_count': outlier_count,
        'drift_detected': outlier_count > (len(logs) * 0.2) if len(logs) > 0 else False,
        'logs': logs[-10:]
    }

@app.get('/')
async def index():
    return FileResponse(FRONTEND_PATH)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
