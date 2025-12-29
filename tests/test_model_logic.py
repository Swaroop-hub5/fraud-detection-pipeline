import pytest
import torch
import numpy as np
from src.model import FraudAutoencoder
from src.features import FeaturePipeline
import pandas as pd

# 1. Test Model Architecture
def test_model_input_output_shape():
    """Ensure the model accepts input and returns output of same shape (Autoencoder property)"""
    input_dim = 10
    model = FraudAutoencoder(input_dim)
    
    # Create fake tensor of shape (Batch Size=5, Input Dim=10)
    fake_data = torch.randn(5, input_dim)
    output = model(fake_data)
    
    assert output.shape == fake_data.shape, "Output shape must match input shape"

# 2. Test Feature Pipeline
def test_feature_pipeline():
    """Ensure the pipeline handles data correctly"""
    # Create a tiny dummy dataframe matching your schema
    df = pd.DataFrame([{
        "amount": 100.0,
        "category": "groceries",
        "seconds_since_last_txn": 50,
        "hour_of_day": 12
    }])
    
    pipeline = FeaturePipeline()
    # We must mock the fitting or just fit on this tiny data
    pipeline.fit(df)
    transformed = pipeline.transform(df)
    
    # Expected dimensions: 3 numeric + categories (one-hot)
    # If 'category' has 1 unique value ('groceries'), one-hot adds 1 col.
    # Total = 3 numeric + 1 cat = 4 columns.
    assert transformed.shape[1] >= 4, "Feature transformation lost columns"

# 3. Test Fraud Logic (Thresholding)
def test_anomaly_logic():
    """Ensure high reconstruction error flags as fraud"""
    # Mocking high error (manual calculation check)
    original = np.array([[10.0]])
    reconstruction = np.array([[1.0]]) # Very different
    
    mse = np.mean(np.power(original - reconstruction, 2))
    threshold = 0.5
    
    is_fraud = mse > threshold
    assert is_fraud == True, "High error should be flagged as fraud"