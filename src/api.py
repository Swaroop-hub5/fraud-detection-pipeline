import logging
import torch
import pandas as pd
import numpy as np
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from src.model import FraudAutoencoder
from src.features import FeaturePipeline
from src.config import settings

# 1. Setup Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fraud-api")

# 2. Global State
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML Artifacts...")
    try:
        # --- A. Load Feature Pipeline ---
        pipeline = FeaturePipeline()
        pipeline.load_artifacts()
        ml_models["pipeline"] = pipeline
        
        # --- B. Load Model ---
        # Calculate dynamic input dimension
        num_features = len(pipeline.numeric_cols) 
        cat_features = sum(len(x) for x in pipeline.encoder.categories_)
        input_dim = num_features + cat_features
        logger.info(f"Dynamically determined Input Dimension: {input_dim}")
        
        model = FraudAutoencoder(input_dim)
        
        # Load Weights (CRITICAL: Do this BEFORE initializing SHAP)
        state_dict = torch.load(settings.MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval() # Set to eval mode
        ml_models["model"] = model

        # --- C. Initialize SHAP Explainer ---
        # We need a wrapper that returns the ANOMALY SCORE (MSE), not the reconstruction
        def predict_anomaly_score(data_numpy):
            # 1. Convert to Tensor
            data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
            with torch.no_grad():
                reconstruction = model(data_tensor)
            
            # 2. Calculate MSE (The "Fraud Score")
            # Axis 1 means calculate error per row (per transaction)
            mse = np.mean(np.power(data_numpy - reconstruction.numpy(), 2), axis=1)
            return mse

        # Use a small background dataset (zeros is a simple baseline for autoencoders)
        # In a real app, use: shap.kmeans(X_train, 10)
        background_data = np.zeros((10, input_dim))
        
        explainer = shap.KernelExplainer(
            model=predict_anomaly_score,
            data=background_data
        )
        ml_models["explainer"] = explainer
        
        logger.info("Successfully loaded Autoencoder, Pipeline, and SHAP Explainer.")
        
    except Exception as e:
        logger.critical(f"Failed to load artifacts: {e}")
        raise e
        
    yield
    ml_models.clear()
    logger.info("Cleaned up resources.")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# --- Pydantic Models ---
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in Euros")
    category: str = Field(..., description="Merchant category code")
    seconds_since_last_txn: int = Field(..., ge=0, description="Time delta")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour 0-23")
    
    @validator('category')
    def validate_category(cls, v):
        allowed = ['groceries', 'dining', 'travel', 'tech', 'utilities']
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v

class PredictionResponse(BaseModel):
    is_fraud: bool
    anomaly_score: float
    threshold: float
    processing_time_ms: float

class ExplanationResponse(BaseModel):
    transaction_id: str
    anomaly_score: float
    top_contributors: dict[str, float]

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME}

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(txn: Transaction):
    import time
    start_time = time.time()
    
    try:
        pipeline = ml_models["pipeline"]
        model = ml_models["model"]
        
        data = pd.DataFrame([txn.model_dump()])
        features = pipeline.transform(data)
        features_tensor = torch.FloatTensor(features)
        
        with torch.no_grad():
            reconstruction = model(features_tensor)
            
        mse = np.mean(np.power(features - reconstruction.numpy(), 2), axis=1)
        anomaly_score = float(mse[0])
        is_fraud = anomaly_score > settings.ANOMALY_THRESHOLD
        
        logger.info(f"Txn processed. Score: {anomaly_score:.4f} | Fraud: {is_fraud}")
        
        return {
            "is_fraud": is_fraud,
            "anomaly_score": anomaly_score,
            "threshold": settings.ANOMALY_THRESHOLD,
            "processing_time_ms": (time.time() - start_time) * 1000
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Model Error")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_transaction(txn: Transaction):
    """
    Calculates SHAP values to explain WHY the anomaly score is high.
    """
    try:
        pipeline = ml_models["pipeline"]
        explainer = ml_models["explainer"]
        
        # 1. Transform Data
        data = pd.DataFrame([txn.model_dump()])
        features = pipeline.transform(data)
        
        # 2. Run SHAP (Calculates impact on MSE)
        # silent=True suppresses the progress bar in logs
        shap_values = explainer.shap_values(features, silent=True)
        
        # 3. Map to Feature Names
        feature_names = pipeline.numeric_cols + list(pipeline.encoder.get_feature_names_out())
        
        # explainer.shap_values returns an array of shape (1, n_features)
        # We want the first (and only) row
        contributions = shap_values[0]
        
        # Create dictionary: { "amount": 0.05, "category_tech": 0.02, ... }
        explanation = dict(zip(feature_names, contributions.tolist()))
        
        # Sort by impact (highest contribution to error first)
        sorted_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
        
        # Calculate score again just for the response
        model = ml_models["model"]
        features_tensor = torch.FloatTensor(features)
        with torch.no_grad():
            rec = model(features_tensor)
        score = float(np.mean(np.power(features - rec.numpy(), 2)))

        return {
            "transaction_id": "simulated_id",
            "anomaly_score": score,
            "top_contributors": sorted_explanation
        }
    except Exception as e:
        logger.error(f"Explanation Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation failed")