# 🚓 District-wise Crime Forecasting System

## 📌 Project Overview

This project is an end-to-end **Machine Learning system** designed to predict daily crime trends. It features a hybrid forecasting engine (LSTM & XGBoost), a fully containerized **FastAPI** backend with rate-limiting, and an automated CI/CD pipeline.

The system is designed with a **Cloud-First** mindset. It utilizes an **Adapter Pattern** to abstract storage logic, allowing the application to switch seamlessly between local development (Filesystem) and production cloud infrastructure (AWS S3 & RDS) via configuration flags.

## 📊 Dataset

The model is built upon the [**Crimes in Boston**](https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston "null") dataset (2015-2018), provided by Analyze Boston.

* **Source:** Real-world incident reports including time, location, and offense descriptions.
* **Preprocessing Strategy:**
  * **Filtering:** Focused analysis on the **Top 5** most frequent crime types and districts to ensure statistical significance.
  * **Imputation:** Implemented logic to handle missing dates (days with zero reported crimes), creating a continuous time-series essential for LSTM performance.
  * **Normalization:** Applied MinMax scaling to normalize feature distributions for deep learning optimization.

## 🚀 Key Features

### 🏗️ SDE & DevOps Architecture

* **Decoupled Microservices:** Separate containers for Data Ingestion (ETL), Model Training, and Inference, orchestrated via  **Docker Compose** .
* **Cloud Infrastructure Simulation:** Implements a custom `StorageManager` to mimic **AWS S3** (artifact storage) and **RDS** (structured data) locally, enabling zero-cost cloud-native development.
* **Infrastructure as Code (IaC):** Includes **Terraform** configuration (`main.tf`) to provision ECS Clusters, ECR Repositories, and API Gateways.
* **Secure API:** Implements **Rate Limiting** (via `slowapi`) to simulate API Gateway throttling policies.
* **CI/CD Pipeline:** GitHub Actions workflow (`deploy.yml`) that automates linting (`pylint`), unit testing (`pytest`), and Docker image builds.

### 🧠 Data Science & Machine Learning

* **Hybrid Forecasting Engine:** Benchmarks **XGBoost** vs.  **LSTM** , achieving an MAE of  **~1.27** .
* **Advanced Feature Engineering:** Automated generation of Lag features (1-day, 7-day), Rolling Means, and Temporal embedding.
* **Model Interpretability:** Integrated **SHAP** (SHapley Additive exPlanations) to interpret model decision

## 🛠️ Tech Stack

* **Language:** Python 3.9
* **ML Frameworks:** TensorFlow (Keras), XGBoost, Scikit-Learn, SHAP
* **Backend:** FastAPI, Uvicorn, SlowAPI
* **Visualization:** Streamlit, Seaborn, Matplotlib
* **Cloud & DevOps:** Docker, Docker Compose, GitHub Actions, Terraform
* **Testing:** Pytest, Pylint

## ⚙️ How to Run

## 📂 Project Structure

```
District-wise-Crime-Forecasting-System/
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD pipeline (Linting, Testing, Build)
├── data/
│   └── (Place your .csv files here)
├── infrastructure/         # Terraform IaC (AWS Config)
│   └── main.tf
├── models/
│   ├── crime_lstm_model.h5 # Trained Deep Learning Model
│   ├── crime_xgb_model.pkl # Trained XGBoost Model
│   └── scaler.pkl          # Scaler for data normalization
├── tests/                  # Unit Tests
│   └── test_api.py
├── storage_manager.py      # AWS/Local Storage Adapter
├── venv/                   # Virtual Environment
├── analytics_dashboard.py  # Streamlit Dashboard script
├── app.py                  # Inference API (FastAPI)
├── data_processor.py       # Data cleaning & preprocessing pipeline
├── Dockerfile              # Docker container configuration
├── docker-compose.yml      # Multi-container orchestration
├── model_engine.py         # Model training script
├── README.md               # Project Documentation
└── requirements.txt        # Python dependencies
```

### 1. Setup Environment

```
# Clone the repository
git clone https://github.com/keshavk215/District-wise-Crime-Forecasting-System.git
cd District-wise-Crime-Forecasting-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Processing & Training

Run the pipeline to generate clean data and train the models.

```
# Clean the raw data
python data_processor.py

# Train LSTM & XGBoost models (Generates .h5 and .pkl files)
python model_engine.py
```

### 3. Start the Inference API

```
python app.py
```

* Access Swagger UI at: `http://localhost:8000/docs`
* Test Endpoint: `POST /predict`

### 4. Run the Analytics Dashboard

```
streamlit run analytics_dashboard.py
```

### 5. Run Tests (CI/CD Simulation)

```
# Run Unit Tests
pytest tests/

# Run Linter
pylint --disable=R,C app.py
```

### 6. Run with Docker Compose (Full System Simulation)

This command spins up the API, Database, and Worker simulations.

```
docker-compose up --build
```

## 🧠 Model Performance

The system evaluates multiple architectures to ensure optimal accuracy:

* **Metric:** Mean Absolute Error (MAE)
* **Result:** The **LSTM (Long Short-Term Memory)** network outperformed the XGBoost baseline, successfully capturing complex temporal dependencies and seasonality in the crime data.
