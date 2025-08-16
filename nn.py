import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import logging
import sklearn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom CSS for industrial design with dynamic lighting
st.markdown("""
<style>
    /* Industrial dark theme with metallic texture */
    .stApp {
        background: linear-gradient(135deg, #1C2526 0%, #2E3B3E 100%);
        color: #E0E0E0;
        font-family: 'Roboto', sans-serif;
    }
    /* Title styling */
    h1 {
        color: #00B7EB;
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        text-shadow: 0 0 10px #00B7EB, 0 0 20px #00B7EB;
        animation: glow 2s ease-in-out infinite alternate;
    }
    /* Header styling */
    h2, h3 {
        color: #FF5733;
        font-family: 'Roboto', sans-serif;
        text-shadow: 0 0 5px #FF5733;
    }
    /* Input fields with neon glow */
    .stNumberInput input {
        background-color: #2E3B3E;
        color: #E0E0E0;
        border: 2px solid #00B7EB;
        border-radius: 5px;
        padding: 8px;
        box-shadow: 0 0 8px #00B7EB;
        transition: box-shadow 0.3s ease;
    }
    .stNumberInput input:focus {
        box-shadow: 0 0 15px #00B7EB, 0 0 25px #00B7EB;
    }
    /* Button styling with pulsating effect */
    .stButton button {
        background-color: #FF5733;
        color: #E0E0E0;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        font-family: 'Roboto', sans-serif;
        box-shadow: 0 0 10px #FF5733;
        transition: all 0.3s ease;
        animation: pulse 1.5s infinite;
    }
    .stButton button:hover {
        background-color: #00B7EB;
        box-shadow: 0 0 15px #00B7EB, 0 0 25px #00B7EB;
        transform: scale(1.05);
    }
    /* Table styling with metallic look */
    .stTable table {
        background-color: #2E3B3E;
        border: 1px solid #00B7EB;
        border-radius: 5px;
        box-shadow: 0 0 10px #00B7EB;
    }
    .stTable th, .stTable td {
        color: #E0E0E0;
        border: 1px solid #00B7EB;
        padding: 8px;
    }
    /* Success and error messages with glow */
    .stSuccess {
        background-color: #00FF00 !important;
        color: #1C2526 !important;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 10px #00FF00;
    }
    .stError {
        background-color: #FF5733 !important;
        color: #E0E0E0 !important;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 10px #FF5733;
    }
    /* Warning styling */
    .stWarning {
        background-color: #FFA500 !important;
        color: #1C2526 !important;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 10px #FFA500;
    }
    /* Animation keyframes */
    @keyframes glow {
        from { text-shadow: 0 0 5px #00B7EB, 0 0 10px #00B7EB; }
        to { text-shadow: 0 0 10px #00B7EB, 0 0 20px #00B7EB; }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 5px #FF5733; }
        50% { box-shadow: 0 0 15px #FF5733, 0 0 25px #FF5733; }
        100% { box-shadow: 0 0 5px #FF5733; }
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("⚡ Data Center Downtime Prediction Control Panel")

# Check scikit-learn version and display warning
st.sidebar.text(f"scikit-learn version: {sklearn.__version__}")
if int(sklearn.__version__.split('.')[0]) < 1:
    st.warning("⚠️ Warning: Your scikit-learn version is outdated (<1.0). This may limit model performance on imbalanced data. Update with 'conda install scikit-learn=1.3.2' for full support of sample weights.")

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

# Select features and target
features = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_traffic', 'temperature', 'power_consumption']
X = data[features]
y = data['downtime']

# Split data into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Data split into train and test sets")
except Exception as e:
    st.error(f"Error splitting data: {str(e)}")
    logging.error(f"Error splitting data: {str(e)}")
    st.stop()

# Scale features
try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Features scaled successfully")
except Exception as e:
    st.error(f"Error scaling features: {str(e)}")
    logging.error(f"Error scaling features: {str(e)}")
    st.stop()

# Train MLP Classifier with fallback to RandomForest if needed
try:
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    try:
        # Attempt to use sample weights for imbalanced data
        class_weights = {0: 1.0, 1: len(y_train) / (2 * np.sum(y_train == 1))}
        sample_weights = np.array([class_weights[yi] for yi in y_train])
        mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        logging.info("MLP model trained with sample weights successfully")
        model_used = "MLP"
    except TypeError:
        # Fallback to RandomForest if sample_weight fails
        st.warning("⚠️ sample_weight not supported. Switching to RandomForestClassifier.")
        logging.warning("sample_weight not supported, switching to RandomForest")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        mlp = rf  # Use RandomForest as the model
        logging.info("RandomForest model trained successfully")
        model_used = "RandomForest"
except Exception as e:
    st.error(f"Error training model: {str(e)}")
    logging.error(f"Error training model: {str(e)}")
    st.stop()

# Calculate model metrics
try:
    y_pred = mlp.predict(X_test_scaled)
    y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1] if model_used == "MLP" else mlp.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info("Model metrics calculated successfully")
except Exception as e:
    st.error(f"Error calculating metrics: {str(e)}")
    logging.error(f"Error calculating metrics: {str(e)}")
    st.stop()

# Create input form
st.header("Input System Metrics")
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
    submit_button = st.form_submit_button(label="⚙️ Predict Downtime")

# Process input and predict downtime
if submit_button:
    with st.spinner("Processing metrics..."):
        try:
            # Prepare input data
            input_data = np.array([[cpu_usage, memory_usage, disk_usage, network_traffic, temperature, power_consumption]])
            input_scaled = scaler.transform(input_data)
            
            # Predict downtime
            prediction = mlp.predict(input_scaled)
            prediction_proba = mlp.predict_proba(input_scaled)[0] if model_used == "MLP" else mlp.predict_proba(input_scaled)[0]
            is_downtime = prediction[0] == 1
            
            # Display result
            if is_downtime:
                st.error("🚨 Downtime Predicted!")
                st.write(f"**Probability of Downtime**: {prediction_proba[1]:.4f}")
                st.write("The input metrics suggest a high likelihood of system downtime.")
            else:
                st.success("✅ No Downtime Predicted")
                st.write(f"**Probability of Downtime**: {prediction_proba[1]:.4f}")
                st.write("The input metrics indicate normal system operation.")
            
            # Plot input point relative to historical CPU usage
            st.header("System CPU Usage Monitor")
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1C2526')
            ax.set_facecolor('#1C2526')
            ax.plot(data['timestamp'][-48:], data['cpu_usage'][-48:], label='Historical CPU Usage', color='#00B7EB', linewidth=2)
            ax.scatter(data['timestamp'][-1] + pd.Timedelta(hours=1), cpu_usage, color='#FF5733', label='Input Point', marker='x', s=150, linewidths=2)
            ax.set_title('CPU Usage Monitor', color='#E0E0E0', fontsize=14, fontweight='bold')
            ax.set_xlabel('Timestamp', color='#E0E0E0')
            ax.set_ylabel('CPU Usage (%)', color='#E0E0E0')
            ax.legend(facecolor='#1C2526', edgecolor='#00B7EB', labelcolor='#E0E0E0')
            ax.grid(True, color='#3A4A4D')
            ax.tick_params(colors='#E0E0E0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#00B7EB')
            st.pyplot(fig)
            logging.info("Prediction and plot generated successfully")
            
            # Display input values
            st.header("Input Metrics")
            input_df = pd.DataFrame({
                'Metric': features,
                'Value': input_data[0]
            })
            st.table(input_df)
            
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            logging.error(f"Error processing input: {str(e)}")

# Show model metrics
st.header("Model Performance Metrics")
st.write("The following metrics evaluate the model's performance on the test set:")
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
})
st.table(metrics_df)
st.write("""
- **Accuracy**: Proportion of correct predictions (may be high due to imbalance).
- **Precision**: Proportion of predicted downtime cases that are correct.
- **Recall**: Proportion of actual downtime cases correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, ideal for imbalanced data.
- **ROC-AUC**: Area under the ROC curve, measuring overall classification performance.
""")
if model_used == "RandomForest":
    st.warning("⚠️ Note: Model switched to RandomForest due to scikit-learn limitations. Update scikit-learn for MLP with weights.")
logging.info("Metrics displayed successfully")