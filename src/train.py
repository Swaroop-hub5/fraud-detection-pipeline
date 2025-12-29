import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from src.model import FraudAutoencoder
from src.features import FeaturePipeline

# Configuration
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

def train():
    print("1. Loading Data...")
    # Load the synthetic data we generated earlier
    df = pd.read_csv("data/synthetic_transactions.csv")
    
    # Filter for NORMAL transactions only for training
    # Autoencoders learn "normality" so they fail to reconstruct anomalies
    normal_df = df[df['is_anomaly'] == 0].copy()
    
    print("2. Feature Engineering...")
    pipeline = FeaturePipeline()
    pipeline.fit(normal_df) # Fit scalers on normal data
    X = pipeline.transform(normal_df)
    
    # Convert to PyTorch Tensors
    tensor_x = torch.Tensor(X)
    dataset = TensorDataset(tensor_x, tensor_x) # Input = Target for Autoencoder
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("3. Initialize Model...")
    input_dim = X.shape[1]
    model = FraudAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("4. Training Loop...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_features, _ in dataloader:
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
        
    print("5. Saving Artifacts...")
    # Save Model state
    torch.save(model.state_dict(), "models/autoencoder.pth")
    print("Training Complete. Model and Scalers saved to 'models/'")

if __name__ == "__main__":
    train()