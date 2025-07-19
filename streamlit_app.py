import streamlit as st
import requests
import numpy as np
import pandas as pd

# -------------------------------
# Config
API_URL = "http://127.0.0.1:5000/predict"  # Localhost Flask API
WINDOW = 3

# -------------------------------
# Sidebar
st.sidebar.title("Crime Forecast App")
district = st.sidebar.selectbox("Select District Code", list(range(1, 730)))  
crime_type = st.sidebar.selectbox("Select Crime Type", ["murder"]) 

# -------------------------------
# Main
st.title("District-wise Crime Forecast")

st.markdown("### Enter past {} years of '{}' cases".format(WINDOW, crime_type))
past_values = []

for i in range(WINDOW):
    year = 2022 - WINDOW + i + 1
    value = st.number_input(f"{year}:", min_value=0, max_value=1000, value=25)
    past_values.append(value)

if st.button("Predict Next Year"):
    payload = {
        "crime_series": past_values,
        "district_code": district
    }

    with st.spinner("Fetching prediction..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                prediction = response.json()['predicted_value']
                st.success(f"🔮 Predicted {crime_type.title()} cases in {2023}: **{prediction}**")
                # Optional plot
                st.line_chart(past_values + [prediction])
            else:
                st.error("❌ Error from API.")
        except Exception as e:
            st.error(f"Failed to reach API. Error: {e}")
