import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Boston Crime Analytics", layout="wide")
st.title("🚓 Boston Crime Analytics Dashboard")

# Load Clean Data
@st.cache_data
def load_data():
    return pd.read_csv('data/clean_crime_data.csv', parse_dates=['Date'])

try:
    df = load_data()
except FileNotFoundError:
    st.error("Run 'data_processor.py' first!")
    st.stop()

# Sidebar
selected_district = st.sidebar.selectbox("Select District", df['District'].unique())
filtered_df = df[df['District'] == selected_district]

# 1. TIME SERIES
st.subheader(f"Daily Crime Trend in {selected_district}")
daily_total = filtered_df.groupby('Date')['Count'].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(data=daily_total, x='Date', y='Count', ax=ax, color='navy', alpha=0.6)
ax.set_title("Total Crimes over Time")
st.pyplot(fig)

# 2. CRIME BREAKDOWN (Pie Chart)
st.subheader("Crime Type Distribution")
crime_counts = filtered_df.groupby('Crime_Type')['Count'].sum()
fig2, ax2 = plt.subplots()
ax2.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=90)
st.pyplot(fig2)

# 3. HEATMAP (Seasonality)
st.subheader("📅 Heatmap: Crime Intensity by Month & Day")
# Extract Month and Day
filtered_df['Month'] = filtered_df['Date'].dt.month_name()
filtered_df['Day'] = filtered_df['Date'].dt.day_name()

# Order for correct plotting
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

heatmap_data = filtered_df.pivot_table(index='Day', columns='Month', values='Count', aggfunc='sum')
# Reindex to sort days/months correctly
heatmap_data = heatmap_data.reindex(index=days_order, columns=months_order)

fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=.5, ax=ax3)
st.pyplot(fig3)