import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import sklearn  # Added to fix NameError
try:
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is not installed. Please install it using 'pip install plotly' or 'conda install plotly'.")
    st.stop()
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom CSS for industrial design
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1C2526 0%, #2E3B3E 100%);
        color: #E0E0E0;
        font-family: 'Roboto', sans-serif;
    }
    h1 {
        color: #00B7EB;
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        text-shadow: 0 0 10px #00B7EB, 0 0 20px #00B7EB;
        animation: glow 2s ease-in-out infinite alternate;
    }
    h2, h3 {
        color: #FF5733;
        font-family: 'Roboto', sans-serif;
        text-shadow: 0 0 5px #FF5733;
    }
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
    .stWarning {
        background-color: #FFA500 !important;
        color: #1C2526 !important;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 10px #FFA500;
    }
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
st.title("⚡ Data Center Downtime Prediction & ECG Monitoring Dashboard")

# Check scikit-learn version
st.sidebar.text(f"scikit-learn version: {sklearn.__version__}")
if int(sklearn.__version__.split('.')[0]) < 1 or (int(sklearn.__version__.split('.')[0]) == 1 and int(sklearn.__version__.split('.')[1]) < 5):
    st.warning("⚠️ Warning: scikit-learn version <1.5 may lack newer features. Update with 'pip install --upgrade scikit-learn' or 'conda install -c conda-forge scikit-learn'.")

# Load dataset
uploaded_file = st.file_uploader("Upload dataset (CSV)", type="csv")
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        logging.info("Dataset loaded successfully")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        logging.error(f"Error loading dataset: {str(e)}")
        st.stop()
else:
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

# Initialize session state for dynamic updates
if 'data_index' not in st.session_state:
    st.session_state.data_index = 0
if 'stream_data' not in st.session_state:
    st.session_state.stream_data = pd.DataFrame(columns=data.columns)

# Simulate real-time data by adding one row at a time
if st.session_state.data_index < len(data):
    new_row = data.iloc[[st.session_state.data_index]]
    st.session_state.stream_data = pd.concat([st.session_state.stream_data, new_row], ignore_index=True)
    st.session_state.data_index += 1
else:
    # Reset to start for continuous simulation
    st.session_state.data_index = 0
    st.session_state.stream_data = pd.DataFrame(columns=data.columns)

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

# Train MLP Classifier
try:
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    sample_weights = np.array([class_weight_dict[yi] for yi in y_train])
    mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    logging.info("MLP model trained successfully")
    if mlp.n_iter_ < mlp.max_iter:
        logging.info(f"MLP converged in {mlp.n_iter_} iterations")
    else:
        logging.warning("MLP did not converge within max iterations")
except Exception as e:
    st.error(f"Error training MLP model: {str(e)}")
    logging.error(f"Error training MLP model: {str(e)}")
    st.stop()

# Calculate model metrics
try:
    y_pred = mlp.predict(X_test_scaled)
    y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
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

# Downtime Alerts Section
st.header("Downtime Alerts")
downtime_events = st.session_state.stream_data[st.session_state.stream_data['downtime'] == 1]
if not downtime_events.empty:
    st.warning(f"⚠️ {len(downtime_events)} Downtime Events Detected in Current Window!")
    st.table(downtime_events[['timestamp'] + features].style.format({'timestamp': lambda t: t.strftime('%Y-%m-%d %H:%M:%S')}))
else:
    st.success("✅ No Downtime Events in Current Window")

# ECG-Like Time Series Visualizations
st.header("Dynamic ECG-Like Monitoring Graphs")
st.write("Real-time, scrolling graphs for each variable, resembling ECG displays. Red markers indicate downtime events.")

# Fixed time window for scrolling (48 hours)
window_hours = 48
window_delta = pd.Timedelta(hours=window_hours)
latest_time = st.session_state.stream_data['timestamp'].max() if not st.session_state.stream_data.empty else pd.Timestamp.now()
start_time = latest_time - window_delta

# Filter data for the current time window
window_data = st.session_state.stream_data[st.session_state.stream_data['timestamp'] >= start_time]
downtime_data = window_data[window_data['downtime'] == 1]

# Create ECG-like graphs for each feature
for feature in features:
    fig = go.Figure()
    
    # Main line (ECG-like)
    fig.add_trace(go.Scatter(
        x=window_data['timestamp'],
        y=window_data[feature],
        mode='lines+markers',
        name=feature.replace('_', ' ').title(),
        line=dict(color='#00B7EB', width=2),
        marker=dict(size=4)
    ))
    
    # Downtime markers
    if not downtime_data.empty:
        fig.add_trace(go.Scatter(
            x=downtime_data['timestamp'],
            y=downtime_data[feature],
            mode='markers',
            name='Downtime Event',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # Customize layout for ECG-like appearance
    fig.update_layout(
        title=f"{feature.replace('_', ' ').title()} (ECG-Style)",
        xaxis_title="Timestamp",
        yaxis_title=feature.replace('_', ' ').title(),
        plot_bgcolor='#1C2526',
        paper_bgcolor='#1C2526',
        font=dict(color='#E0E0E0'),
        xaxis=dict(
            gridcolor='#3A4A4D',
            tickfont=dict(color='#E0E0E0'),
            range=[start_time, latest_time],
            rangeslider=dict(visible=True),
            tickformat="%Y-%m-%d %H:%M"
        ),
        yaxis=dict(
            gridcolor='#3A4A4D',
            tickfont=dict(color='#E0E0E0')
        ),
        legend=dict(bgcolor='#1C2526', bordercolor='#00B7EB', font=dict(color='#E0E0E0')),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    logging.info(f"ECG-like graph for {feature} generated successfully")

# Input form for predictions
st.header("Input System Metrics for Prediction")
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
            input_data = np.array([[cpu_usage, memory_usage, disk_usage, network_traffic, temperature, power_consumption]])
            input_scaled = scaler.transform(input_data)
            
            prediction = mlp.predict(input_scaled)
            prediction_proba = mlp.predict_proba(input_scaled)[0]
            is_downtime = prediction[0] == 1
            
            if is_downtime:
                st.error(f"🚨 Downtime Predicted! Probability: {prediction_proba[1]:.4f}")
                st.write("The input metrics suggest a high likelihood of system downtime.")
            else:
                st.success(f"✅ No Downtime Predicted. Probability: {prediction_proba[1]:.4f}")
                st.write("The input metrics indicate normal system operation.")
            
            # Display input values
            st.header("Input Metrics")
            input_df = pd.DataFrame({'Metric': features, 'Value': input_data[0]})
            st.table(input_df)
            
            logging.info("Prediction generated successfully")
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            logging.error(f"Error processing input: {str(e)}")

# Model Performance Metrics
st.header("Model Performance Metrics")
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
})
st.table(metrics_df)

# Chart.js bar chart for metrics
st.header("Model Metrics Visualization")
st.markdown("```chartjs\n"
            "{\n"
            "    \"type\": \"bar\",\n"
            "    \"data\": {\n"
            "        \"labels\": [\"Accuracy\", \"Precision\", \"Recall\", \"F1-Score\", \"ROC-AUC\"],\n"
            "        \"datasets\": [{\n"
            "            \"label\": \"Model Metrics\",\n"
            "            \"data\": [" + f"{accuracy}, {precision}, {recall}, {f1}, {roc_auc}" + "],\n"
            "            \"backgroundColor\": [\"#00B7EB\", \"#FF5733\", \"#00FF00\", \"#FFA500\", \"#C70039\"],\n"
            "            \"borderColor\": [\"#E0E0E0\"],\n"
            "            \"borderWidth\": 1\n"
            "        }]\n"
            "    },\n"
            "    \"options\": {\n"
            "        \"scales\": {\n"
            "            \"y\": {\n"
            "                \"beginAtZero\": true,\n"
            "                \"max\": 1,\n"
            "                \"title\": {\n"
            "                    \"display\": true,\n"
            "                    \"text\": \"Score\",\n"
            "                    \"color\": \"#E0E0E0\"\n"
            "                },\n"
            "                \"ticks\": { \"color\": \"#E0E0E0\" },\n"
            "                \"grid\": { \"color\": \"#3A4A4D\" }\n"
            "            },\n"
            "            \"x\": {\n"
            "                \"title\": {\n"
            "                    \"display\": true,\n"
            "                    \"text\": \"Metric\",\n"
            "                    \"color\": \"#E0E0E0\"\n"
            "                },\n"
            "                \"ticks\": { \"color\": \"#E0E0E0\" },\n"
            "                \"grid\": { \"color\": \"#3A4A4D\" }\n"
            "            }\n"
            "        },\n"
            "        \"plugins\": {\n"
            "            \"legend\": {\n"
            "                \"labels\": { \"color\": \"#E0E0E0\" }\n"
            "            },\n"
            "            \"title\": {\n"
            "                \"display\": true,\n"
            "                \"text\": \"Model Performance Metrics\",\n"
            "                \"color\": \"#E0E0E0\",\n"
            "                \"font\": { \"size\": 16 }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "```")

# ROC Curve
st.header("ROC Curve")
try:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.4f})', line=dict(color='#00B7EB')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#FF5733', dash='dash')))
    fig.update_layout(
        title="Receiver Operating Characteristic",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor='#1C2526',
        paper_bgcolor='#1C2526',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='#3A4A4D', tickfont=dict(color='#E0E0E0')),
        yaxis=dict(gridcolor='#3A4A4D', tickfont=dict(color='#E0E0E0')),
        legend=dict(bgcolor='#1C2526', bordercolor='#00B7EB', font=dict(color='#E0E0E0'))
    )
    st.plotly_chart(fig, use_container_width=True)
    logging.info("ROC curve generated successfully")
except Exception as e:
    st.error(f"Error generating ROC curve: {str(e)}")
    logging.error(f"Error generating ROC curve: {str(e)}")

# Metrics explanation
st.write("""
- **Accuracy**: Proportion of correct predictions (may be high due to imbalance).
- **Precision**: Proportion of predicted downtime cases that are correct.
- **Recall**: Proportion of actual downtime cases correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, ideal for imbalanced data.
- **ROC-AUC**: Area under the ROC curve, measuring overall classification performance.
""")
logging.info("Metrics and ROC curve displayed successfully")

# Auto-refresh for dynamic updates
time.sleep(2)  # Simulate 2-second interval for new data
st.experimental_rerun()