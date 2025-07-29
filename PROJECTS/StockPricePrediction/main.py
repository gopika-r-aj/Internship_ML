# STEP 1: Import Libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# STEP 2: Download Data
ticker = "AAPL"  # You can change this to "RELIANCE.NS", "TSLA", etc.
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
print(data.head())

# STEP 3: Data Preprocessing
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

X, y = [], []
sequence_length = 60

for i in range(sequence_length, len(scaled_close)):
    X.append(scaled_close[i-sequence_length:i, 0])
    y.append(scaled_close[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# STEP 4: Build LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# STEP 5: Train the Model
history = model.fit(X, y, epochs=25, batch_size=32)

# STEP 6: Make Predictions
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# STEP 7: Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual Price', color='blue')
plt.plot(predicted_prices, label='Predicted Price', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# STEP 8: Add Moving Average and RSI Indicators
data['MA50'] = data['Close'].rolling(window=50).mean()

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# STEP 9: Plot MA and RSI
plt.figure(figsize=(14,6))

plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='50-Day MA')
plt.title(f'{ticker} - Close Price & MA50')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', color='red')
plt.axhline(30, linestyle='--', color='green')
plt.title(f'{ticker} - Relative Strength Index (RSI)')
plt.legend()

plt.tight_layout()
plt.show()

# STEP 10: Save the model
model.save("lstm_model.h5")
print("Model saved as 'lstm_model.h5'")