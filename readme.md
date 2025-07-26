# 🧠 District-wise Crime Forecasting System (India, 2017–2022)

A full-stack machine learning project for forecasting district-level IPC crimes in India using historical data. Combines statistical and deep learning models with an interactive Streamlit frontend and Flask backend API for real-time, dynamic predictions.


## 🚀 Project Overview

This project predicts district-wise crime counts across different crime types (e.g., murder, robbery) using structured NCRB data from 2017 to 2022. It uses both classical ML (Random Forest) and deep learning (LSTM) models with window-based time series forecasting. The trained models are deployed through a Flask REST API and integrated into a dynamic Streamlit UI for real-time inference.

## 🛠️ Tech Stack

* **Languages & Tools** : Python, NumPy, Pandas, Matplotlib, Seaborn
* **ML/DL Libraries** : scikit-learn, TensorFlow, Keras
* **Web Frameworks** : Flask (REST API), Streamlit (frontend UI)

## 🧩 Features

* 📈 **District-wise crime trend prediction**
* 🔄 **Supports multiple models** (classical ML and deep learning)
* 🧠 **LSTM time-series forecasting** using sliding windows
* ⚙️  **Modular model architecture** , extendable to more crime categories
* 🌐 **REST API interface** built with Flask
* 🎛️ **Frontend dashboard** built in Streamlit

## How to Run Locally

### 1. Clone the repo

```
git clone https://github.com/keshavk215/District-wise-Crime-Forecasting-System.git
cd District-wise-Crime-Forecasting-System
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Start the Flask API

```
python app.py
```

### 4. Start the Streamlit Dashboard

```
streamlit run streamlit_app.py
```



## 📦 Future Extensions

* 🔐 Cybercrime category modeling
* 🗺️ Interactive map visualization using Folium or Leaflet.js
* 📬 Email alerts / SMS notifications for predicted crime spikes
* 🧠 Use CNN+RNN for more complex spatial-temporal patterns
* 🧩 Serve model on cloud (Render, Railway, or Heroku)
