import pandas as pd
import os

def clean_and_prepare_data():
    file_path = 'data/boston_crime.csv' 
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        print("   Please rename your downloaded file to 'boston_crime.csv' and place it in this folder.")
        return

    print("⏳ Loading raw data...")
    
    # 1. LOAD DATA
    cols = ['OFFENSE_CODE_GROUP', 'DISTRICT', 'OCCURRED_ON_DATE']
    
    try:
        df = pd.read_csv(file_path, usecols=cols, encoding='latin-1')
    except ValueError as e:
        print(f"⚠️ Column Error: {e}")
        print("   Columns in CSV might have different names. Please check the CSV header.")
        return
    
    # Show original data stats
    original_rows = len(df)
    original_mem = df.memory_usage(deep=True).sum() / (1024 * 1024) # MB
    print(f"   📊 RAW Data: {original_rows:,} rows | {original_mem:.2f} MB")

    # 2. CLEANING DATE
    print("   Parsing dates...")
    df['Date'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
    df['Date'] = df['Date'].dt.normalize() # Keep only the date part
    
    # 3. FILTERING
    # Focus on Top 5 Crimes and Top 5 Districts
    top_crimes = df['OFFENSE_CODE_GROUP'].value_counts().nlargest(5).index.tolist()
    top_districts = df['DISTRICT'].value_counts().nlargest(5).index.tolist()
    
    print(f"   Focusing on Top 5 Crimes: {top_crimes}")
    
    df_filtered = df[
        (df['OFFENSE_CODE_GROUP'].isin(top_crimes)) & 
        (df['DISTRICT'].isin(top_districts))
    ].copy()
    
    # 4. AGGREGATION
    df_filtered = df_filtered.rename(columns={
        'OFFENSE_CODE_GROUP': 'Crime_Type',
        'DISTRICT': 'District'
    })
    
    # Count crimes per day
    daily_counts = df_filtered.groupby(['Date', 'District', 'Crime_Type']).size().reset_index(name='Count')
    
    # 5. IMPUTATION (Handling Zero Days)
    print("   Imputing missing days (Zero-crime days)...")
    
    # Create a full grid of dates to fill in the blanks
    all_dates = pd.date_range(start=daily_counts['Date'].min(), end=daily_counts['Date'].max(), freq='D')
    
    # Create MultiIndex of all possible combinations
    idx = pd.MultiIndex.from_product(
        [all_dates, top_districts, top_crimes], 
        names=['Date', 'District', 'Crime_Type']
    )
    
    # Reindex to force 0s where data is missing
    daily_counts = daily_counts.set_index(['Date', 'District', 'Crime_Type']).reindex(idx, fill_value=0).reset_index()
    
    # --- CALCULATE REDUCTION ---
    final_rows = len(daily_counts)
    final_mem = daily_counts.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction_pct = (1 - (final_rows / original_rows)) * 100
    
    print(f"   📉 PROCESSED Data: {final_rows:,} rows | {final_mem:.2f} MB")
    print(f"   ✅ DATA REDUCTION: {reduction_pct:.2f}% (Aggregated incident-level data to daily time-series)")
    
    # Save
    daily_counts.to_csv('data/clean_crime_data.csv', index=False)
    print(f"✅ Data cleaned! Saved {len(daily_counts)} rows.")

if __name__ == "__main__":
    clean_and_prepare_data()