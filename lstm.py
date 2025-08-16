import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging

# Set up logging to debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if PyTorch is installed
try:
    import torch
    logging.info(f"PyTorch version: {torch.__version__}")
except ImportError:
    raise ImportError("PyTorch is not installed. Install it using: conda install pytorch -c pytorch")

# Load the dataset
try:
    data = pd.read_csv("/Users/farhanjafar/Desktop/data_center_downtime_50days.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    logging.info("Dataset loaded successfully")
except FileNotFoundError:
    logging.error("Could not find 'data_center_downtime_50days.csv' in /Users/farhanjafar/Desktop/")
    raise
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# Select CPU usage for forecasting
series = data['cpu_usage'].values.reshape(-1, 1)
logging.info(f"Series shape: {series.shape}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

# Create sequences for LSTM (use last 24 hours to predict next hour)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24  # Use last 24 hours to predict the next
X, y = create_sequences(series_scaled, seq_length)
logging.info(f"Sequence shapes: X={X.shape}, y={y.shape}")

# Convert to PyTorch tensors
try:
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    logging.info("Converted to PyTorch tensors")
except Exception as e:
    logging.error(f"Error converting to tensors: {str(e)}")
    raise

# Split into train and test (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
logging.info(f"Train size: {train_size}, Test size: {len(X_test)}")

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Initialize model, loss, optimizer
model = LSTMModel(hidden_layer_size=32)  # Reduced hidden size for efficiency
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model with batch processing
epochs = 30  # Reduced epochs for faster training
batch_size = 16  # Smaller batch size for MacBook
logging.info("Starting LSTM training...")
try:
    for i in range(epochs):
        model.train()
        total_loss = 0
        for j in range(0, len(X_train), batch_size):
            batch_X = X_train[j:j+batch_size]
            batch_y = y_train[j:j+batch_size]
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_function(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i % 10 == 0:
            print(f'Epoch {i} loss: {total_loss / (len(X_train) / batch_size):.4f}')
            logging.info(f'Epoch {i} loss: {total_loss / (len(X_train) / batch_size):.4f}')
except Exception as e:
    logging.error(f"Error during training: {str(e)}")
    raise

# Forecast next 24 hours
forecast_steps = 24
forecast_input = X_test[-1].unsqueeze(0)  # Start with the last sequence from test
forecasts = []

logging.info("Starting forecasting...")
model.eval()
try:
    with torch.no_grad():
        for _ in range(forecast_steps):
            forecast = model(forecast_input)
            forecasts.append(forecast.item())
            forecast_input = torch.cat((forecast_input[:, 1:, :], forecast.unsqueeze(1).unsqueeze(0)), dim=1)
    logging.info("Forecasting completed")
except Exception as e:
    logging.error(f"Error during forecasting: {str(e)}")
    raise

# Inverse transform forecasts
forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

# Create forecast index
last_timestamp = data.index[-1]
try:
    forecast_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_steps, freq='h')
    logging.info("Forecast index created")
except Exception as e:
    logging.error(f"Error creating forecast index: {str(e)}")
    raise

# Plot historical data and forecast
try:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-48:], data['cpu_usage'][-48:], label='Historical CPU Usage', color='blue')
    plt.plot(forecast_index, forecasts, label='Forecasted CPU Usage', color='red', linestyle='--')
    plt.title('LSTM Forecast for CPU Usage (Next 24 Hours)')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    logging.info("Plot generated successfully")
except Exception as e:
    logging.error(f"Error generating plot: {str(e)}")
    raise

# Print forecasted values
forecast_df = pd.DataFrame({'Timestamp': forecast_index, 'Forecasted CPU Usage': forecasts.flatten()})
print("\nForecasted CPU Usage for Next 24 Hours:")
print(forecast_df)

# Evaluate on test set
model.eval()
try:
    with torch.no_grad():
        test_predictions = model(X_test).numpy()
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    rmse = np.sqrt(((y_test_actual - test_predictions) ** 2).mean())
    print(f"\nTest RMSE: {rmse:.4f}")
    logging.info(f"Test RMSE: {rmse:.4f}")
except Exception as e:
    logging.error(f"Error evaluating model: {str(e)}")
    raise