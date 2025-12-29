import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_URL = "http://127.0.0.1:8000"  # Address of your local FastAPI
st.set_page_config(page_title="N26 Fraud Sentinel", layout="wide", page_icon="🛡️")

# --- HEADER ---
st.title("🛡️Fraud Sentinel: Real-time Anomaly Detection")
st.markdown("""
This dashboard simulates the **Fraud Review Analyst** view. 
It uses an **Autoencoder** to detect anomalies and **SHAP** to explain *why* a transaction was flagged.
""")

# --- SIDEBAR (INPUTS) ---
st.sidebar.header("📝 Transaction Details")

# Helper to generate random test cases
if st.sidebar.button("🎲 Generate Random Test Case"):
    import random
    st.session_state['amount'] = float(random.randint(10, 5000))
    st.session_state['category'] = random.choice(['groceries', 'dining', 'travel', 'tech', 'utilities'])
    st.session_state['hour'] = random.randint(0, 23)
    st.session_state['seconds'] = random.randint(1, 50000)
else:
    # Set defaults if not in session state
    if 'amount' not in st.session_state: st.session_state['amount'] = 50.0
    if 'category' not in st.session_state: st.session_state['category'] = 'dining'
    if 'hour' not in st.session_state: st.session_state['hour'] = 14
    if 'seconds' not in st.session_state: st.session_state['seconds'] = 3600

# Input Form
amount = st.sidebar.number_input("Amount (€)", min_value=0.0, value=st.session_state['amount'])
category = st.sidebar.selectbox("Merchant Category", 
                                ['groceries', 'dining', 'travel', 'tech', 'utilities'],
                                index=['groceries', 'dining', 'travel', 'tech', 'utilities'].index(st.session_state['category']))
hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, st.session_state['hour'])
seconds = st.sidebar.number_input("Seconds since last Transaction", min_value=0, value=st.session_state['seconds'])

# --- MAIN ACTION ---
if st.button("🚀 Analyze Transaction", type="primary"):
    
    # 1. Construct Payload
    payload = {
        "amount": amount,
        "category": category,
        "seconds_since_last_txn": int(seconds),
        "hour_of_day": hour
    }
    
    col1, col2 = st.columns([1, 2])
    
    # --- API CALL 1: PREDICT ---
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Display Prediction
        with col1:
            st.subheader("Risk Assessment")
            
            score = result['anomaly_score']
            threshold = result['threshold']
            is_fraud = result['is_fraud']
            
            # Gauge Chart for Score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score,
                title = {'text': "Anomaly Score (MSE)"},
                delta = {'reference': threshold, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [0, max(0.1, score * 1.5)]},
                    'bar': {'color': "red" if is_fraud else "green"},
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            if is_fraud:
                st.error(f"🚨 FRAUD DETECTED (Score > {threshold})")
            else:
                st.success("✅ Transaction Approved")
                
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        st.stop()

    # --- API CALL 2: EXPLAIN (Only if user wants deep dive) ---
    with col2:
        st.subheader("🔍 Explainability (SHAP Values)")
        with st.spinner("Calculating feature contributions..."):
            try:
                exp_response = requests.post(f"{API_URL}/explain", json=payload)
                exp_response.raise_for_status()
                explanation = exp_response.json()
                
                # Prepare data for chart
                contributors = explanation['top_contributors']
                df_shap = pd.DataFrame(list(contributors.items()), columns=['Feature', 'Impact'])
                
                # Sort for visualization
                df_shap = df_shap.sort_values(by='Impact', ascending=True)
                
                # Bar Chart
                fig_shap = px.bar(
                    df_shap, 
                    x='Impact', 
                    y='Feature', 
                    orientation='h',
                    title="Why was this score given?",
                    labels={'Impact': 'Contribution to Anomaly Error'},
                    color='Impact',
                    color_continuous_scale=px.colors.diverging.Tealrose
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                
                st.info("💡 **Interpretation:** Higher positive values mean this feature pushed the transaction towards being an 'Anomaly'.")
                
            except Exception as e:
                st.warning(f"Could not load explanation: {e}")