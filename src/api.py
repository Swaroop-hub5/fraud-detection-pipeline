import logging
import torch
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from src.model import FraudAutoencoder
from src.features import FeaturePipeline
from src.config import settings

# 1. Setup Structured Logging (JSON format for Splunk/Datadog)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fraud-api")

# 2. Global State (dictionary to hold model/pipeline)
ml_models = {}

# 3. Lifespan Event (The correct way to load models)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources on startup
    logger.info("Loading ML Artifacts...")
    try:
        # Load Pipeline
        pipeline = FeaturePipeline()
        pipeline.load_artifacts()
        ml_models["pipeline"] = pipeline
        
        # Load Model
        # In real life, calculate input_dim dynamically from encoder
        #input_dim = settings.INPUT_DIM 
        # 1. Get numeric count
        num_features = len(pipeline.numeric_cols) 

        # 2. Get categorical count (from the loaded encoder categories)
        # OneHotEncoder.categories_ is a list of arrays (one per column)
        cat_features = sum(len(x) for x in pipeline.encoder.categories_)

        # 3. Total input dim
        input_dim = num_features + cat_features
        logger.info(f"Dynamically determined Input Dimension: {input_dim}")
        model = FraudAutoencoder(input_dim)
        
        # Safe loading: map_location handles CPU/GPU mismatch automatically
        state_dict = torch.load(settings.MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval() # CRITICAL: Set to eval mode for inference
        ml_models["model"] = model
        
        logger.info("Successfully loaded Autoencoder and Feature Pipeline.")
    except Exception as e:
        logger.critical(f"Failed to load model artifacts: {e}")
        # In K8s, this causes the pod to crash/restart (which is what we want)
        raise e
        
    yield
    
    # Clean up on shutdown
    ml_models.clear()
    logger.info("Cleaned up resources.")

# Initialize App with Lifespan
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# 4. Strict Input Validation (Pydantic)
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in Euros")
    category: str = Field(..., description="Merchant category code")
    seconds_since_last_txn: int = Field(..., ge=0, description="Time delta in seconds")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour 0-23")
    
    # Custom Validator example
    @validator('category')
    def validate_category(cls, v):
        allowed = ['groceries', 'dining', 'travel', 'tech', 'utilities']
        if v not in allowed:
            # In production, we might map unknown to 'other' instead of crashing
            # For this demo, we enforce strict schema
            raise ValueError(f"Category must be one of {allowed}")
        return v

class PredictionResponse(BaseModel):
    is_fraud: bool
    anomaly_score: float
    threshold: float
    processing_time_ms: float = 0.0

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(txn: Transaction):
    import time
    start_time = time.time()
    
    try:
        # Get models from global state
        pipeline = ml_models["pipeline"]
        model = ml_models["model"]
        
        # Convert Pydantic -> DataFrame
        data = pd.DataFrame([txn.model_dump()])
        
        # Transform
        features = pipeline.transform(data)
        features_tensor = torch.FloatTensor(features)
        
        # Inference
        with torch.no_grad():
            reconstruction = model(features_tensor)
            
        # Logic
        mse = np.mean(np.power(features - reconstruction.numpy(), 2), axis=1)
        anomaly_score = float(mse[0])
        is_fraud = anomaly_score > settings.ANOMALY_THRESHOLD
        
        # Log the prediction (Audit Trail)
        logger.info(f"Transaction processed. Score: {anomaly_score:.4f} | Fraud: {is_fraud}")
        
        return {
            "is_fraud": is_fraud,
            "anomaly_score": anomaly_score,
            "threshold": settings.ANOMALY_THRESHOLD,
            "processing_time_ms": (time.time() - start_time) * 1000
        }

    except ValueError as ve:
        # 400 Bad Request (Client Fault)
        logger.warning(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # 500 Internal Error (Server Fault)
        logger.error(f"Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Model Error")