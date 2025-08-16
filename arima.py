import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load the dataset
data = pd.read_csv("data_center_downtime_50days.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Select CPU usage for forecasting
series = data['cpu_usage']

# Check stationarity with Augmented Dickey-Fuller test
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] <= 0.05:
        print("Stationary (reject null hypothesis)")
    else:
        print("Non-stationary (fail to reject null hypothesis)")

print("Stationarity Test for CPU Usage:")
check_stationarity(series)

# If non-stationary, apply differencing (uncomment if needed)
# series = series.diff().dropna()

# Fit ARIMA model (example parameters: p=2, d=1, q=2)
# Adjust (p, d, q) based on your analysis or grid search
model = ARIMA(series, order=(2, 1, 2))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast next 24 hours
forecast_steps = 24
forecast = model_fit.forecast(steps=forecast_steps)

# Create forecast index
last_timestamp = series.index[-1]
forecast_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_steps, freq='H')

# Plot historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(series[-48:], label='Historical CPU Usage', color='blue')
plt.plot(forecast_index, forecast, label='Forecasted CPU Usage', color='red', linestyle='--')
plt.title('ARIMA Forecast for CPU Usage (Next 24 Hours)')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.grid(True)
plt.show()

# Print forecasted values
forecast_df = pd.DataFrame({'Timestamp': forecast_index, 'Forecasted CPU Usage': forecast})
print("\nForecasted CPU Usage for Next 24 Hours:")
print(forecast_df)