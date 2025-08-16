import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps (50 days, hourly)
start_time = datetime(2025, 1, 1, 0, 0)
timestamps = [start_time + timedelta(hours=i) for i in range(1200)]

# Initialize dataset
n_rows = 1200
data = {
    'timestamp': timestamps,
    'cpu_usage': np.zeros(n_rows),
    'memory_usage': np.zeros(n_rows),
    'disk_usage': np.zeros(n_rows),
    'network_traffic': np.zeros(n_rows),
    'temperature': np.zeros(n_rows),
    'power_consumption': np.zeros(n_rows),
    'downtime': np.zeros(n_rows, dtype=int)
}

# Simulate realistic patterns
for i in range(n_rows):
    hour = timestamps[i].hour
    # Diurnal pattern: higher usage during 8 AM - 6 PM
    base_load = 50 + 30 * np.sin(np.pi * hour / 12)  # Peaks around noon
    # CPU and Memory Usage (0-100%)
    data['cpu_usage'][i] = np.clip(base_load + np.random.normal(0, 5), 30, 95)
    data['memory_usage'][i] = np.clip(data['cpu_usage'][i] + np.random.normal(0, 3), 30, 95)
    # Disk Usage: slow increase with noise (70-95%)
    data['disk_usage'][i] = np.clip(70 + i * 0.012 + np.random.normal(0, 2), 70, 95)
    # Network Traffic: 100-250 Mbps, correlated with CPU
    data['network_traffic'][i] = np.clip(100 + data['cpu_usage'][i] * 1.5 + np.random.normal(0, 10), 100, 250)
    # Temperature: 21-35°C, loosely follows CPU usage
    data['temperature'][i] = np.clip(21 + data['cpu_usage'][i] * 0.15 + np.random.normal(0, 1), 21, 35)
    # Power Consumption: 4.8-7.5 kW, proportional to CPU
    data['power_consumption'][i] = np.clip(4.8 + data['cpu_usage'][i] * 0.03 + np.random.normal(0, 0.2), 4.8, 7.5)
    # Downtime: rare, triggered by high CPU (>90%), disk (>90%), or temp (>30°C)
    if (data['cpu_usage'][i] > 90 or data['disk_usage'][i] > 90 or data['temperature'][i] > 30) and np.random.random() < 0.05:
        data['downtime'][i] = 1

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data_center_downtime_50days.csv', index=False)

# Confirm dataset size
print(f"Dataset generated with {df.shape[0]} rows and {df.shape[1]} columns")
print("Saved as 'data_center_downtime_50days.csv'")
print("Preview of first 5 rows:")
print(df.head())