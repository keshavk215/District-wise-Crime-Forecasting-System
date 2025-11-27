# 1. Base Image
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements
COPY requirements.txt .

# 5. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code and Models
COPY app.py .
COPY data/clean_crime_data.csv .
COPY models/scaler.pkl .
COPY models/crime_lstm_model.h5 .

# 7. Expose API Port
EXPOSE 8000

# 8. Run the Application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]