import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def forecast_with_lstm(ticker):
    try:
        # Fetch historical stock data
        logging.info(f"Fetching data for ticker: {ticker}")
        df = yf.download(ticker, period='6mo', interval='1d', auto_adjust=False)

        # Check if 'Close' column exists
        if 'Close' not in df.columns:
            raise ValueError("No 'Close' column in the fetched data.")

        # Drop missing values
        df = df[['Close']].dropna()

        # Ensure enough data after cleaning
        if df.empty or len(df) < 60:
            raise ValueError("Not enough valid 'Close' data available. At least 60 rows are required.")

        # Prepare and scale the data
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(60, len(data_scaled)):
            X.append(data_scaled[i-60:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Model path
        model_path = f"models/{ticker}_lstm_model.h5"

        # Load or train model
        if os.path.exists(model_path):
            logging.info(f"Loading existing model for ticker: {ticker}")
            model = load_model(model_path)
        else:
            logging.info(f"Training new model for ticker: {ticker}")
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=20, batch_size=32)
            os.makedirs("models", exist_ok=True)
            model.save(model_path)
            logging.info(f"Model saved at {model_path}")

        # Forecast
        logging.info(f"Generating predictions for ticker: {ticker}")
        future_prices = model.predict(X[-30:])
        predictions = scaler.inverse_transform(future_prices).flatten().tolist()

        # Return future dates and predictions
        forecast_dates = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]
        return [{"ds": str(date), "yhat": price} for date, price in zip(forecast_dates, predictions)]

    except Exception as e:
        logging.error(f"Error in LSTM forecasting for ticker {ticker}: {e}")
        return {"error": str(e)}
