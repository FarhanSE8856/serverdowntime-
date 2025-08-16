import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom CSS for intense, professional design
st.markdown("""
<style>
    /* Dark theme */
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    /* Title styling */
    h1 {
        color: #FF4B4B;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    /* Header styling */
    h2, h3 {
        color: #00FFFF;
        font-family: 'Arial', sans-serif;
    }
    /* Input fields */
    .stNumberInput input {
        background-color: #2A2A2A;
        color: #FFFFFF;
        border: 2px solid #FF4B4B;
        border-radius: 5px;
        padding: 8px;
    }
    /* Button styling */
    .stButton button {
        background-color: #FF4B4B;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #00FFFF;
        color: #1E1E1E;
        transform: scale(1.05);
    }
    /* Table styling */
    .stTable table {
        background-color: #2A2A2A;
        border: 1px solid #FF4B4B;
        border-radius: 5px;
    }
    .stTable th, .stTable td {
        color: #FFFFFF;
        border: 1px solid #FF4B4B;
        padding: 8px;
    }
    /* Success and error messages */
    .stSuccess {
        background-color: #00FF00 !important;
        color: #1E1E1E !important;
        border-radius: 5px;
        padding: 10px;
    }
    .stError {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("🔥 Data Center Anomaly Detection Dashboard")

# Load the dataset
try:
    data = pd.read_csv("/Users/farhanjafar/Desktop/data_center_downtime_50days.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    logging.info("Dataset loaded successfully")
except FileNotFoundError:
    st.error("Error: Could not find 'data_center_downtime_50days.csv' in /Users/farhanjafar/Desktop/")
    logging.error("Dataset file not found")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    logging.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Select features for anomaly detection
features = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_traffic', 'temperature', 'power_consumption']
X = data[features]

# Scale features
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Features scaled successfully")
except Exception as e:
    st.error(f"Error scaling features: {str(e)}")
    logging.error(f"Error scaling features: {str(e)}")
    st.stop()

# Train Isolation Forest
try:
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_scaled)
    data['anomaly'] = iso_forest.predict(X_scaled)
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})  # 1 = anomaly, 0 = normal
    logging.info("Isolation Forest model trained successfully")
except Exception as e:
    st.error(f"Error training model: {str(e)}")
    logging.error(f"Error training model: {str(e)}")
    st.stop()

# Create input form
st.header("Enter Data Center Metrics")
with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    with col1:
        cpu_usage = st.number_input("CPU Usage (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        memory_usage = st.number_input("Memory Usage (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        disk_usage = st.number_input("Disk Usage (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    with col2:
        network_traffic = st.number_input("Network Traffic (MB/s)", min_value=0.0, max_value=500.0, value=150.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        power_consumption = st.number_input("Power Consumption (kW)", min_value=0.0, max_value=10.0, value=6.0, step=0.01)
    submit_button = st.form_submit_button(label="🔍 Check for Anomaly")

# Process input and predict anomaly
if submit_button:
    with st.spinner("Analyzing metrics..."):
        try:
            # Prepare input data
            input_data = np.array([[cpu_usage, memory_usage, disk_usage, network_traffic, temperature, power_consumption]])
            input_scaled = scaler.transform(input_data)
            
            # Predict anomaly
            prediction = iso_forest.predict(input_scaled)
            is_anomaly = prediction[0] == -1
            
            # Display result
            if is_anomaly:
                st.error("🚨 Anomaly Detected!")
                st.write("The input metrics indicate an unusual state in the data center.")
            else:
                st.success("✅ Normal")
                st.write("The input metrics are within normal operating conditions.")
            
            # Plot input point relative to historical CPU usage
            st.header("Historical CPU Usage with Input Point")
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='#2A2A2A')
            ax.set_facecolor('#2A2A2A')
            ax.plot(data['timestamp'][-48:], data['cpu_usage'][-48:], label='Historical CPU Usage', color='#00FFFF', linewidth=2)
            ax.scatter(data['timestamp'][-1] + pd.Timedelta(hours=1), cpu_usage, color='#FF4B4B', label='Input Point', marker='x', s=150, linewidths=2)
            ax.set_title('CPU Usage with Input Point', color='#FFFFFF', fontsize=14, fontweight='bold')
            ax.set_xlabel('Timestamp', color='#FFFFFF')
            ax.set_ylabel('CPU Usage (%)', color='#FFFFFF')
            ax.legend(facecolor='#2A2A2A', edgecolor='#FF4B4B', labelcolor='#FFFFFF')
            ax.grid(True, color='#555555')
            ax.tick_params(colors='#FFFFFF')
            for spine in ax.spines.values():
                spine.set_edgecolor('#FF4B4B')
            st.pyplot(fig)
            logging.info("Prediction and plot generated successfully")
            
            # Display input values
            st.header("Input Values")
            input_df = pd.DataFrame({
                'Metric': features,
                'Value': input_data[0]
            })
            st.table(input_df)
            
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            logging.error(f"Error processing input: {str(e)}")

# Show model insights
st.header("Model Insights")
correlation = data[['anomaly', 'downtime']].corr().iloc[0, 1]
st.write(f"**Correlation between detected anomalies and downtime**: {correlation:.4f}")
st.write("This indicates how often anomalies align with downtime events in the historical data.")