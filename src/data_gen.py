import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def generate_financial_data(n_users=1000, n_transactions=50000):
    """
    Generates a realistic stream of transactions.
    Normal behavior: Small coffee buys, regular groceries.
    Anomalies: High amounts, odd hours, rapid burst transactions.
    """
    data = []
    
    # 1. Define Merchant Categories
    categories = ['groceries', 'dining', 'travel', 'tech', 'utilities']
    
    print(f"Generating {n_transactions} transactions for {n_users} users...")
    
    for _ in range(n_transactions):
        user_id = random.randint(1, n_users)
        category = random.choice(categories)
        
        # Base logic for amounts
        if category == 'groceries':
            amount = round(random.uniform(20, 150), 2)
        elif category == 'tech':
            amount = round(random.uniform(100, 2000), 2)
        else:
            amount = round(random.uniform(5, 50), 2)
            
        # Timestamp simulation
        date_time = fake.date_time_between(start_date='-1y', end_date='now')
        hour = date_time.hour
        
        # --- INJECT ANOMALIES (The "Fraud") ---
        is_anomaly = 0
        
        # Anomaly Type 1: High Value at 3 AM
        if random.random() < 0.02: 
            amount = amount * 10  # Spike the amount
            # Force time to be roughly 3 AM
            date_time = date_time.replace(hour=3, minute=random.randint(0, 59))
            is_anomaly = 1
            
        # Anomaly Type 2: Rapid Burst (Simulated by high frequency flag for this feature set)
        # In a real system, this would be a calculated feature 'seconds_since_last_txn'
        seconds_since_last = random.randint(300, 86400)
        if random.random() < 0.02:
            seconds_since_last = random.randint(1, 10) # 1 to 10 seconds ago
            is_anomaly = 1

        data.append({
            "transaction_id": fake.uuid4(),
            "user_id": user_id,
            "timestamp": date_time,
            "amount": amount,
            "category": category,
            "seconds_since_last_txn": seconds_since_last,
            "hour_of_day": date_time.hour,
            "is_anomaly": is_anomaly # Ground truth for validation
        })
        
    df = pd.DataFrame(data)
    print("Data generation complete.")
    return df

if __name__ == "__main__":
    df = generate_financial_data()
    df.to_csv("data/synthetic_transactions.csv", index=False)