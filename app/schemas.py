from pydantic import BaseModel, Field, validator
from typing import List, Optional

class CarPredictionRequest(BaseModel):
    brand: str = Field(..., example="Toyota")
    mileage: float = Field(..., gt=-1, example=50000.0)
    year: int = Field(..., ge=1900, le=2026, example=2020)
    fuel_type: str = Field(..., example="Gasoline")

    @validator('brand')
    def validate_brand(cls, v):
        # We'll load valid brands in the app, but here we just check for basic sanity
        if len(v) < 2:
            raise ValueError('Brand name must be at least 2 characters')
        return v.title()

class CarPredictionResponse(BaseModel):
    predicted_price: float
    model_version: str = "v1.0"
    confidence_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    r2_score: float

class MonitoringLog(BaseModel):
    timestamp: str
    request: CarPredictionRequest
    prediction: float
    is_outlier: bool
