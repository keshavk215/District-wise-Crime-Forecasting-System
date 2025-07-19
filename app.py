from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models + scalers
rf_model = joblib.load('models/murder_rf_model.pkl')
lstm_model = load_model('models/murder_lstm_model.h5', compile=False)
scaler_crime = joblib.load('models/murder_lstm_scaler_crime.pkl')
scaler_target = joblib.load('models/murder_lstm_scaler_target.pkl')
scaler_district = joblib.load('models/murder_lstm_scaler_district.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    crime_series = np.array(data['crime_series']).reshape(1, -1)   # shape (1, window)
    district_code = np.array([[data['district_code']]])

    # Scale and reshape
    crime_scaled = scaler_crime.transform(crime_series).reshape((1, crime_series.shape[1], 1))
    district_scaled = scaler_district.transform(district_code)

    # Predict
    y_pred = lstm_model.predict([crime_scaled, district_scaled])
    y_pred_orig = scaler_target.inverse_transform(y_pred)

    return jsonify({'predicted_value': int(y_pred_orig[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
