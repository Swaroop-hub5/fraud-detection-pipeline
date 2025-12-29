import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import settings

class FeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.numeric_cols = ['amount', 'seconds_since_last_txn', 'hour_of_day']
        self.cat_cols = ['category']
        self.is_fitted = False

    def fit(self, df):
        self.scaler.fit(df[self.numeric_cols])
        self.encoder.fit(df[self.cat_cols])
        self.is_fitted = True
        
        # Save with versioning capability in mind
        os.makedirs(os.path.dirname(settings.SCALER_PATH), exist_ok=True)
        joblib.dump(self.scaler, settings.SCALER_PATH)
        joblib.dump(self.encoder, settings.ENCODER_PATH)

    def load_artifacts(self):
        """Robust loading with checks"""
        if not os.path.exists(settings.SCALER_PATH) or not os.path.exists(settings.ENCODER_PATH):
            raise FileNotFoundError(f"Artifacts not found at {settings.SCALER_PATH} or {settings.ENCODER_PATH}. Run training first.")
            
        self.scaler = joblib.load(settings.SCALER_PATH)
        self.encoder = joblib.load(settings.ENCODER_PATH)
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted or artifacts loaded before transform.")
            
        try:
            scaled_nums = self.scaler.transform(df[self.numeric_cols])
            encoded_cats = self.encoder.transform(df[self.cat_cols])
            return np.hstack([scaled_nums, encoded_cats])
        except Exception as e:
            # Re-raise with context so API knows it's a data issue
            raise ValueError(f"Feature transformation failed: {str(e)}")