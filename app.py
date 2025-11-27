import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime, timedelta
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 1. INITIALIZE APP
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Crime Forecasting API",
    description="Deep Learning Microservice for predicting daily crime counts.",
    version="1.0"
)

# Register the Rate Limit Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# GLOBAL VARIABLES
model = None
scaler = None
history_db = None
district_map = {}
crime_map = {}

# 2. LOAD MODEL AND DATA
print("🚀 Loading LSTM model and data...")
try:
    # A. Load Scaler & Data 
    scaler = joblib.load('models/scaler.pkl')
    history_db = pd.read_csv('data/clean_crime_data.csv', parse_dates=['Date'])
    
    # B. Create Mappings
    district_map = {d: i for i, d in enumerate(history_db['District'].unique())}
    crime_map = {c: i for i, c in enumerate(history_db['Crime_Type'].unique())}

    # C. Load Model 
    model = load_model('models/crime_lstm_model.h5', compile=False)
    
    print("✅ System Ready.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    print("   -> Verify 'crime_lstm_model.h5', 'scaler.pkl', and 'clean_crime_data.csv' exist.")

class PredictionRequest(BaseModel):
    district: str
    crime_type: str
    date: str

# 3. FEATURE ENGINEERING 
def build_features(district, crime, query_date):
    if history_db is None:
        raise HTTPException(status_code=503, detail="Server is starting or failed to load data.")
    
    subset = history_db[
        (history_db['District'] == district) & 
        (history_db['Crime_Type'] == crime)
    ].sort_values('Date')
    
    query_dt = pd.to_datetime(query_date)
    
    try:
        # Lag Features
        lag_1_date = query_dt - timedelta(days=1)
        lag_1 = subset[subset['Date'] == lag_1_date]['Count'].values[0]
        
        lag_7_date = query_dt - timedelta(days=7)
        lag_7 = subset[subset['Date'] == lag_7_date]['Count'].values[0]
        
        lag_30_date = query_dt - timedelta(days=30)
        lag_30 = subset[subset['Date'] == lag_30_date]['Count'].values[0]
        
        # Rolling Mean
        start_roll = query_dt - timedelta(days=7)
        rolling_mean = subset[
            (subset['Date'] >= start_roll) & 
            (subset['Date'] < query_dt)
        ]['Count'].mean()
        
    except IndexError:
        raise HTTPException(status_code=400, detail="Not enough history for this date.")

    dist_code = district_map.get(district, -1)
    crime_code = crime_map.get(crime, -1)
    day_of_week = query_dt.dayofweek
    month = query_dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Create DataFrame for features
    features = pd.DataFrame([{
        'District_Code': dist_code,
        'Crime_Code': crime_code,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_30': lag_30,
        'rolling_mean_7': rolling_mean,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend
    }])
    
    return features

# 4. PREDICTION ENDPOINT
@app.post("/predict")
@limiter.limit("5/minute") # Allow only 5 requests per minute per IP
def predict_crime(request: PredictionRequest, web_request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    
    if request.district not in district_map:
        # Show available districts to help the user
        valid_districts = list(district_map.keys())
        raise HTTPException(status_code=404, detail=f"District '{request.district}' not found. Valid: {valid_districts}")
    
    if request.crime_type not in crime_map:
        valid_crimes = list(crime_map.keys())
        raise HTTPException(status_code=404, detail=f"Crime '{request.crime_type}' not found. Valid: {valid_crimes}")

    # A. Get Raw Features
    features_df = build_features(request.district, request.crime_type, request.date)
    
    # B. SCALE FEATURES
    features_scaled = scaler.transform(features_df)
    
    # C. RESHAPE
    features_lstm = features_scaled.reshape((1, 1, features_scaled.shape[1]))
    
    # D. PREDICT
    prediction = model.predict(features_lstm, verbose=0)
    
    return {
        "model_used": "LSTM",
        "district": request.district,
        "date": request.date,
        "predicted_count": max(0.0, float(prediction[0][0])),
        "risk_level": "High" if prediction[0][0] > 5 else "Low"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)