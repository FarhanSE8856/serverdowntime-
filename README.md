Data Center Monitoring and Prediction Project
Project Overview
The Data Center Monitoring and Prediction Project is a comprehensive suite of Python-based tools designed to enhance the operational efficiency and reliability of data centers. Leveraging advanced machine learning, deep learning, and statistical techniques, the project addresses three critical aspects of data center management: anomaly detection, downtime prediction, and CPU usage forecasting. The project consists of four distinct scripts that analyze the dataset data_center_downtime_50days.csv, providing actionable insights through interactive dashboards, robust visualizations, and predictive models. 
The primary goal is to empower data center operators with tools to monitor system health in real-time, predict potential disruptions, and anticipate resource demands. Two scripts utilize Streamlit to deliver user-friendly, interactive dashboards, while the other two focus on standalone time-series forecasting with detailed console outputs and plots. The project integrates modern data science libraries, robust error handling, and visually appealing interfaces to ensure practical utility and reliability in production environments.
Figure: Workflow of the Data Center Monitoring and Prediction Project, illustrating data processing, modeling, and output generation.
Objectives
The project is structured to achieve the following objectives:

Anomaly Detection: Identify unusual patterns in data center metrics (e.g., spikes in CPU usage or temperature) that may indicate potential issues such as hardware failures or performance bottlenecks.
Downtime Prediction: Predict the likelihood of downtime events based on system metrics, enabling proactive maintenance to minimize service disruptions.
CPU Usage Forecasting: Forecast CPU usage for the next 24 hours to support capacity planning and resource allocation.
Interactive Monitoring: Provide intuitive, real-time interfaces for operators to input metrics, view predictions, and analyze historical trends.
Robust Insights: Deliver detailed model performance metrics, visualizations, and statistical analyses to guide decision-making.

Dataset
The project relies on the dataset data_center_downtime_50days.csv, which contains 50 days of data center performance metrics. The dataset includes the following columns:

timestamp: Date and time of the measurement (datetime format).
cpu_usage: CPU usage percentage (0–100%).
memory_usage: Memory usage percentage (0–100%).
disk_usage: Disk usage percentage (0–100%).
network_traffic: Network traffic in MB/s (0–500 MB/s).
temperature: System temperature in °C (0–100°C).
power_consumption: Power consumption in kW (0–10 kW).
downtime: Binary indicator (0 = no downtime, 1 = downtime).

The dataset is assumed to be located at /Users/farhanjafar/Desktop/serverprediction/. Scripts 1 and 2 utilize all features for analysis, while Scripts 3 and 4 focus exclusively on cpu_usage for time-series forecasting.
Scripts
1. Data Center Anomaly Detection Dashboard (anomaly_detection.py)
Purpose:This script creates an interactive Streamlit dashboard to detect anomalies in data center metrics using the Isolation Forest algorithm, an unsupervised machine learning method optimized for identifying outliers in high-dimensional data.
Key Features:

Data Preprocessing:
Loads the dataset and converts the timestamp column to datetime format.
Selects six key features: cpu_usage, memory_usage, disk_usage, network_traffic, temperature, and power_consumption.
Scales features using StandardScaler to ensure consistent input for the Isolation Forest model.


Model:
Trains an Isolation Forest model with a contamination rate of 0.01, assuming 1% of data points are anomalies.
Labels data points as normal (0) or anomalous (1) based on the model’s predictions.


User Interface:
Features a visually striking Streamlit dashboard with a dark theme, accented by vibrant red (#FF4B4B) and cyan (#00FFFF) colors for a professional, high-tech aesthetic.
Includes a form with number input fields for the six metrics, allowing operators to input real-time data.
Upon submission, the dashboard predicts whether the input metrics represent an anomaly, displaying the result as a success message ("Normal") or an error message ("Anomaly Detected").


Visualization:
Generates a Matplotlib plot showing the last 48 hours of historical CPU usage (cyan line) with the user’s input point marked as a red 'x'.
Displays a styled table of input metrics for quick reference.


Insights:
Computes the correlation between detected anomalies and actual downtime events, providing a quantitative measure of how well anomalies align with system failures.


Error Handling:
Implements comprehensive logging using Python’s logging module to track data loading, scaling, model training, and prediction steps.
Handles errors gracefully for file not found, data scaling issues, and model training failures, displaying user-friendly error messages in the dashboard.



Figure: Screenshot of the Anomaly Detection Dashboard, showcasing the input form, prediction result, and CPU usage plot.
Use Case:This script is designed for real-time monitoring of data center health. Operators can input current system metrics to detect anomalies that may indicate issues like overheating, resource exhaustion, or hardware malfunctions, enabling rapid response to prevent downtime.
Technologies: Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib.

2. Data Center Downtime Prediction Control Panel (downtime_prediction.py)
Purpose:This script develops a Streamlit-based control panel to predict data center downtime using supervised machine learning, leveraging either a Multi-Layer Perceptron (MLP) or a RandomForestClassifier to handle imbalanced data.
Key Features:

Data Preprocessing:
Loads the dataset and splits it into training (80%) and test (20%) sets using stratified sampling to address the imbalance in the downtime target variable (downtime events are rare).
Scales the six features (cpu_usage, memory_usage, disk_usage, network_traffic, temperature, power_consumption) using StandardScaler.


Model:
Attempts to train an MLPClassifier with sample weights to prioritize rare downtime events, using a hidden layer architecture of (100, 50) and 500 iterations.
Falls back to a RandomForestClassifier (100 estimators) if the scikit-learn version does not support sample weights (e.g., versions <1.0).
Evaluates model performance using a comprehensive set of metrics: accuracy, precision, recall, F1-score, and ROC-AUC, which are critical for imbalanced datasets.


User Interface:
Features a futuristic Streamlit dashboard with an industrial design, incorporating neon glow effects, pulsating buttons, and a metallic texture (gradient background: #1C2526 to #2E3B3E).
Provides a form for inputting the six metrics, with real-time prediction of downtime probability upon submission.
Displays results as a success message ("No Downtime Predicted") or an error message ("Downtime Predicted") with the associated probability score.


Visualization:
Plots the last 48 hours of CPU usage (cyan line) with the user’s input point marked in orange (#FF5733).
Presents a styled table of input metrics for clarity.


Insights:
Displays a table of model performance metrics with detailed explanations:
Accuracy: Proportion of correct predictions (may be skewed by imbalance).
Precision: Proportion of predicted downtime cases that are correct.
Recall: Proportion of actual downtime cases correctly identified.
F1-Score: Harmonic mean of precision and recall, ideal for imbalanced data.
ROC-AUC: Area under the ROC curve, measuring overall classification performance.


Warns users if an outdated scikit-learn version is detected, recommending an update to version 1.3.2 for full MLP functionality.


Error Handling:
Includes extensive logging for data loading, splitting, scaling, model training, and prediction.
Handles errors for data processing and model training, displaying user-friendly messages in the dashboard.



Figure: Screenshot of the Downtime Prediction Control Panel, displaying prediction results, CPU usage plot, and performance metrics.
Use Case:This script enables proactive downtime prediction, allowing operators to take preventive measures based on real-time metrics. The fallback to RandomForest ensures compatibility across different environments, making it robust for deployment.
Technologies: Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib.

3. LSTM-Based CPU Usage Forecasting (lstm_forecast.py)
Purpose:This script uses a Long Short-Term Memory (LSTM) neural network to forecast CPU usage for the next 24 hours, leveraging deep learning for accurate time-series prediction.
Key Features:

Data Preprocessing:
Loads the dataset and extracts the cpu_usage column as a univariate time-series.
Normalizes data using MinMaxScaler to scale values between 0 and 1.
Creates sequences of 24 hours of CPU usage to predict the next hour, generating input sequences (X) and target values (y).
Splits data into training (80%) and test (20%) sets.


Model:
Implements an LSTM model using PyTorch with 1 input feature (CPU usage), 32 hidden units, and 1 output (predicted CPU usage).
Trains the model for 30 epochs with a batch size of 16, using Mean Squared Error (MSE) loss and the Adam optimizer (learning rate = 0.001).


Forecasting:
Generates a 24-hour forecast by iteratively feeding the last test sequence into the model, appending each prediction to the input sequence.
Inverse-transforms predictions to the original CPU usage scale using MinMaxScaler.


Visualization:
Plots the last 48 hours of historical CPU usage (blue line) and the 24-hour forecast (red dashed line) using Matplotlib.


Evaluation:
Computes the Root Mean Squared Error (RMSE) on the test set to quantify forecast accuracy.
Outputs a table of forecasted CPU usage values with corresponding timestamps.


Error Handling:
Uses Python’s logging module to track PyTorch version, data loading, tensor conversion, training, and forecasting steps.
Handles errors for missing PyTorch dependencies, data loading, tensor conversion, and training failures.



Figure: LSTM forecast plot showing historical CPU usage (blue) and 24-hour prediction (red, dashed).
Use Case:This script supports predictive maintenance by forecasting CPU usage trends, helping operators plan resource allocation and avoid overloading. The deep learning approach is particularly effective for capturing complex patterns in time-series data.
Technologies: Pandas, NumPy, PyTorch, Matplotlib.

4. ARIMA-Based CPU Usage Forecasting (arima_forecast.py)
Purpose:This script employs the ARIMA (AutoRegressive Integrated Moving Average) model to forecast CPU usage for the next 24 hours, offering a statistical alternative to deep learning.
Key Features:

Data Preprocessing:
Loads the dataset, sets timestamp as the index, and extracts the cpu_usage series.
Performs an Augmented Dickey-Fuller (ADF) test to check for stationarity, with an option for differencing if the series is non-stationary (commented out but available).


Model:
Fits an ARIMA(2,1,2) model, where:
p=2: Autoregressive terms.
d=1: Differencing to achieve stationarity.
q=2: Moving average terms.


Note: These parameters are example values and may require tuning via grid search for optimal performance.


Forecasting:
Forecasts CPU usage for the next 24 hours.
Creates a forecast index starting from the last timestamp in the dataset, with hourly frequency.


Visualization:
Plots the last 48 hours of historical CPU usage (blue line) and the 24-hour forecast (red dashed line) using Matplotlib.


Output:
Prints a table of forecasted CPU usage values with timestamps.
Displays the ARIMA model summary, including coefficients, p-values, and statistical metrics for model diagnostics.


Error Handling:
Relies on standard Python exception handling for data loading and model fitting, with minimal explicit error management.



Figure: ARIMA forecast plot showing historical CPU usage (blue) and 24-hour prediction (red, dashed).
Use Case:This script provides a statistical approach to CPU usage forecasting, suitable for environments where traditional time-series models are preferred over neural networks. It is ideal for scenarios requiring interpretable models with statistical rigor.
Technologies: Pandas, Statsmodels, Matplotlib.

Installation
Prerequisites

Python 3.9 or higher.
Conda package manager (recommended for environment management).
Access to the dataset data_center_downtime_50days.csv.

Steps

Clone the Repository:
git clone <repository_url>
cd <repository_directory>


Set Up Conda Environment:Create and activate a Conda environment:
conda create -n data_center python=3.9
conda activate data_center


Install Dependencies:Install required libraries:
conda install pandas numpy scikit-learn pytorch matplotlib streamlit statsmodels


Optional scikit-learn Update:For Script 2, update scikit-learn to version 1.3.2 to support sample weights:
conda install scikit-learn=1.3.2


Dataset:Ensure data_center_downtime_50days.csv is placed at /Users/farhanjafar/Desktop/serverprediction/. Update the file path in the scripts if necessary.

Images:To include images in the README, create an images/ folder in the repository and add the following:

project_workflow.png: A diagram illustrating the project workflow (optional, create using Draw.io or Canva).
anomaly_detection_dashboard.png: Screenshot of Script 1’s Streamlit dashboard.
downtime_prediction_dashboard.png: Screenshot of Script 2’s Streamlit dashboard.
lstm_forecast_plot.png: CPU usage forecast plot from Script 3.
arima_forecast_plot.png: CPU usage forecast plot from Script 4.Save plots by adding plt.savefig('images/plot_name.png') before plt.show() in the scripts. Commit images to the repository:

git add images/*
git commit -m "Add images for README"
git push origin main



Usage
Running Streamlit Dashboards (Scripts 1 and 2)
Launch the interactive dashboards:
streamlit run anomaly_detection.py
streamlit run downtime_prediction.py

Access the dashboards in a web browser (typically at http://localhost:8501). Input metrics to receive real-time predictions and visualizations.
Running Standalone Scripts (Scripts 3 and 4)
Execute the forecasting scripts:
python lstm_forecast.py
python arima_forecast.py

These scripts generate console outputs (forecasted values and metrics) and Matplotlib plots.
Outputs

Script 1 (Anomaly Detection):
Streamlit dashboard with anomaly detection results ("Normal" or "Anomaly Detected").
Matplotlib plot of the last 48 hours of CPU usage with the input point.
Table of input metrics.
Correlation between detected anomalies and downtime events.


Script 2 (Downtime Prediction):
Streamlit dashboard with downtime prediction and probability score.
Matplotlib plot of the last 48 hours of CPU usage with the input point.
Table of input metrics.
Model performance metrics table (accuracy, precision, recall, F1-score, ROC-AUC).


Script 3 (LSTM Forecasting):
Console output with a table of 24-hour forecasted CPU usage values and timestamps.
Matplotlib plot comparing historical and forecasted CPU usage.
Test set RMSE for model evaluation.


Script 4 (ARIMA Forecasting):
Console output with a table of 24-hour forecasted CPU usage values and timestamps.
Matplotlib plot comparing historical and forecasted CPU usage.
ARIMA model summary with statistical metrics.



Potential Improvements

Unified Interface:
Combine all scripts into a single Streamlit app with tabs for anomaly detection, downtime prediction, and forecasting.


Model Optimization:
Script 1: Implement additional anomaly detection algorithms (e.g., DBSCAN, One-Class SVM) for comparison.
Script 2: Use grid search to optimize MLP or RandomForest hyperparameters.
Script 3: Experiment with LSTM architectures (e.g., stacked or bidirectional LSTM) or adjust sequence lengths.
Script 4: Perform grid search to optimize ARIMA parameters (p, d, q) or use auto-ARIMA.


Real-Time Data Integration:
Add support for real-time data ingestion via APIs or streaming platforms for continuous monitoring.


Extended Forecasting:
Forecast additional metrics (e.g., memory_usage, temperature) to provide a more comprehensive view.


Model Comparison:
Develop a script to compare LSTM and ARIMA forecasting performance using metrics like RMSE and MAE.


Enhanced Visualizations:
Add interactive plots using Plotly for Streamlit dashboards.
Include more detailed visualizations, such as feature importance for Script 2 or anomaly heatmaps for Script 1.



Notes

Dataset Path: Update the dataset path in all scripts if different from /Users/farhanjafar/Desktop/serverprediction/.
Streamlit Requirements: Scripts 1 and 2 require an internet connection for CSS and font rendering in Streamlit.
PyTorch Dependency: Script 3 requires PyTorch; install it with conda install pytorch -c pytorch if not present.
Scikit-learn Version: Script 2 may fall back to RandomForest if scikit-learn is outdated (<1.0). Update to version 1.3.2 for full MLP functionality.
ARIMA Parameters: The ARIMA model in Script 4 uses example parameters (2,1,2). Optimize these parameters for better performance using tools like pmdarima.auto_arima.
Image Hosting: For GitHub-hosted repositories, use raw URLs for images (e.g., https://raw.githubusercontent.com/your-username/your-repo/main/images/image_name.png). Alternatively, host images on external platforms like Imgur.

License
This project is licensed under the MIT License.
Contact
For questions, bug reports, or contributions, please open an issue on the repository or contact the project maintainer.
Acknowledgments

Built with Python, Streamlit, Scikit-learn, PyTorch, and Statsmodels.
Inspired by the need for robust, data-driven solutions for data center management.
Special thanks to open-source communities for providing the tools and libraries used in this project.
