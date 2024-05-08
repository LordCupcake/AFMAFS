# app.py
from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained LSTM model (assume it's pre-trained and saved)
model = tf.keras.models.load_model("lstm_model.h5.py")

@app.route("/predict", methods=["GET"])
def predict():
    stock_symbol = request.args.get("symbol", "AAPL")
    data = yf.download(stock_symbol, start="2015-01-01", end="2023-01-01")
    data["Close_Lag_1"] = data["Close"].shift(1)
    data.dropna(inplace=True)

    # Normalize the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[["Close", "Volume", "Close_Lag_1"]])

    # Reshape for LSTM (3D: samples, time steps, features)
    X = scaled_data[:, 1:]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Get predictions
    predictions = model.predict(X).flatten().tolist()

    return jsonify(predictions)

if _name_ == "_main_":
    app.run(debug=True)
