import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import shap
import matplotlib.pyplot as plt
from storage_manager import StorageManager


# 1. FEATURE ENGINEERING
def create_features(df):
    df = df.copy()
    # Lag Features (Past context)
    df['lag_1'] = df.groupby(['District', 'Crime_Type'])['Count'].shift(1)
    df['lag_7'] = df.groupby(['District', 'Crime_Type'])['Count'].shift(7)
    df['lag_30'] = df.groupby(['District', 'Crime_Type'])['Count'].shift(30)
    
    # Rolling Mean (Trend context)
    df['rolling_mean_7'] = df.groupby(['District', 'Crime_Type'])['Count'] \
                             .transform(lambda x: x.rolling(7).mean())
    
    # Seasonality
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df.dropna()

# 2. LOAD DATA
print("⏳ Loading data...")
df = pd.read_csv('data/clean_crime_data.csv', parse_dates=['Date'])

# Encode Districts & Crimes
le_dist = LabelEncoder()
le_crime = LabelEncoder()
df['District_Code'] = le_dist.fit_transform(df['District'])
df['Crime_Code'] = le_crime.fit_transform(df['Crime_Type'])

# Create Features
df_model = create_features(df)

# Define X and y
features = ['District_Code', 'Crime_Code', 'lag_1', 'lag_7', 'lag_30', 
            'rolling_mean_7', 'day_of_week', 'month', 'is_weekend']
target = 'Count'

# 3. SPLIT DATA (Temporal Split)
split_date = '2018-06-01'
train = df_model[df_model['Date'] < split_date]
test = df_model[df_model['Date'] >= split_date]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print(f"   Training Set: {len(X_train)} rows")
print(f"   Test Set:     {len(X_test)} rows")

# ==========================================
# MODEL A: XGBoost (The Gradient Boosting Approach)
# ==========================================
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

# METRICS 
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print(f"   XGBoost -> MAE: {xgb_mae:.4f} | RMSE: {xgb_rmse:.4f} | R²: {xgb_r2:.4f}")

# SHAP INTERPRETATION 
print("   📊 Generating SHAP values...")
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test.iloc[:100])

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title("XGBoost Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig('shap_summary.png')
print("   ✅ SHAP plot saved to 'shap_summary.png'")

# ==========================================
# MODEL B: LSTM (The Deep Learning Approach)
# ==========================================
print("\n🧠 Training LSTM Network...")

# A. Scaling 
scaler = MinMaxScaler()
# Fit on train, transform both 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# B. Reshaping for LSTM [Samples, TimeSteps, Features]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# C. Architecture
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', return_sequences=False, 
                    input_shape=(1, X_train_scaled.shape[1])))
lstm_model.add(Dropout(0.2)) # Prevent overfitting
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(1)) # Output layer 

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# D. Train
history = lstm_model.fit(
    X_train_lstm, y_train, 
    epochs=20, 
    batch_size=64, 
    validation_data=(X_test_lstm, y_test),
    verbose=0 
)

lstm_preds = lstm_model.predict(X_test_lstm).flatten()
# METRICS
lstm_mae = mean_absolute_error(y_test, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
lstm_r2 = r2_score(y_test, lstm_preds)

print(f"   LSTM    -> MAE: {lstm_mae:.4f} | RMSE: {lstm_rmse:.4f} | R²: {lstm_r2:.4f}")

# ==========================================
# COMPARISON 
# ==========================================
print("\n🏆 Final Results:")
if xgb_mae < lstm_mae:
    print(f"   XGBoost is better than LSTM! (Lower Error by {lstm_mae - xgb_mae:.4f})")
    best_model = xgb_model
else:
    print(f"   LSTM is better than XGBoost! (Lower Error by {xgb_mae - lstm_mae:.4f})")
    best_model = lstm_model

# Visualization of Predictions (First 100 days)
plt.figure(figsize=(15, 5))
plt.plot(y_test.values[:100], label='Actual', color='black', alpha=0.5)
plt.plot(xgb_preds[:100], label='XGBoost', linestyle='--')
plt.plot(lstm_preds[:100], label='LSTM', linestyle='-.')
plt.title("Model Forecast Comparison (First 100 Test Samples)")
plt.legend()
plt.savefig('model_comparison.png')
print("   Saved comparison plot.")

# Save the modesl
StorageManager.save_model(xgb_model, 'crime_xgb_model.pkl')
StorageManager.save_model(scaler, 'scaler.pkl')
StorageManager.save_model(lstm_model, 'crime_lstm_model.h5')
print("   Models saved successfully.")