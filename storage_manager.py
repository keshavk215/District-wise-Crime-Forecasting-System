import os
import pandas as pd
import joblib

# CONFIGURATION:
USE_AWS_S3 = False 
USE_RDS = False

class StorageManager:
    """
    Adapter Class: Handles the switch between Local File System (Dev) and AWS (Prod).
    """
    
    @staticmethod
    def save_model(model, filename):
        """
        Saves models to local disk or S3 based on config.
        Auto-detects Keras models (.h5) vs Scikit-Learn/XGBoost (.pkl).
        """
        if USE_AWS_S3:
            import boto3
            # In a real scenario:
            # s3 = boto3.client('s3')
            # s3.upload_file(local_path, 'my-model-bucket', filename)
            print(f"☁️ [AWS S3] Uploading {filename} to bucket 'crime-models'...")
        else:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)
            path = os.path.join("models", filename)
            
            # Check if it's a Keras model (has a .save method)
            if hasattr(model, 'save') and filename.endswith('.h5'):
                model.save(path)
            else:
                joblib.dump(model, path)
                
            print(f"💾 [Local] Saved {filename} to /models folder.")

    @staticmethod
    def save_data(df, filename):
        """
        Saves data to local CSV or RDS Table.
        """
        if USE_RDS:
            # from sqlalchemy import create_engine
            # engine = create_engine("postgresql://user:pass@host/db")
            # df.to_sql(filename, engine, if_exists='replace')
            print(f"☁️ [AWS RDS] Writing {len(df)} rows to PostgreSQL table '{filename}'...")
        else:
            os.makedirs("data", exist_ok=True)
            path = os.path.join("data", f"{filename}.csv")
            df.to_csv(path, index=False)
            print(f"💾 [Local] Saved data to {path} (Simulating RDS).")