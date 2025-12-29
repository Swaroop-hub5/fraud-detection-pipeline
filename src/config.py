from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Fraud Sentinel"
    DEBUG_MODE: bool = False
    
    # Paths
    MODEL_PATH: str = "models/autoencoder.pth"
    SCALER_PATH: str = "models/scaler.joblib"
    ENCODER_PATH: str = "models/encoder.joblib"
    
    # Model Hyperparameters (Can be tuned without redeploying code)
    ANOMALY_THRESHOLD: float = 0.005
    INPUT_DIM: int = 8
    
    class Config:
        env_file = ".env"

# Singleton instance
settings = Settings()