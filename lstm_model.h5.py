import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Data Collection
stock_symbols = ["BMW.DE", "TSLA", "AMZN", "MSFT", "AAPL", "MCD", "GOOG"]  # Adding multiple stock symbols
start_date = "2010-01-01"
end_date = "2023-01-01"

# Collecting stock data for each symbol
stock_data = {symbol: yf.download(symbol, start=start_date, end=end_date) for symbol in stock_symbols}

# Feature Engineering
def add_features(data):
    data["Close_Lag_1"] = data["Close"].shift(1)  # Create a new column with the shifted close values
    data["Volume_Lag_1"] = data["Volume"].shift(1)  # Shifted volume
    data.dropna(inplace=True)  # Drop rows with NaNs after shifting
    return data

# Apply feature engineering to all stocks
stock_data = {symbol: add_features(data) for symbol, data in stock_data.items()}

# Normalize the features
scaler = MinMaxScaler()
scaled_data = {symbol: scaler.fit_transform(data[["Close", "Volume", "Close_Lag_1", "Volume_Lag_1"]])
               for symbol, data in stock_data.items()}

# Time-Series Split for training and testing
tscv = TimeSeriesSplit(n_splits=5)

# LSTM Model Configuration
def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(100, activation="relu", return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1)  # Single output for regression
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Build, train, and save LSTM models for each stock
for symbol in stock_symbols:
    data = scaled_data[symbol]  # Get the normalized data for the stock
    X = data[:, 1:]  # All but the target feature
    y = data[:, 0]   # Target feature (close price)

    # Prepare data for LSTM (reshape to 3D)
    X_train, y_train = [], []
    for train_index, _ in tscv.split(X):
        X_train.append(np.reshape(X[train_index], (X[train_index].shape[0], X[train_index].shape[1], 1)))  # Reshape for LSTM
        y_train.append(y[train_index])

    # Build the LSTM model
    input_shape = (X_train[-1].shape[1], 1)
    model = build_lstm_model(input_shape)

    # Train the model
    model.fit(
        X_train[-1],
        y_train[-1],
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)],
        verbose=2
    )

    # Save the trained LSTM model
    model.save(f"{symbol}_lstm_model.h5")  # Save with the stock symbol name
