import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_with_lstm(ticker):
    df = yf.download(ticker, period='6mo', interval='1d')
    df = df[['Close']]
    data = df.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:train_data_len, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))


    model = create_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=20, verbose=0)

    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    future_predictions = []

    for _ in range(30):
        next_pred = model.predict(last_60_days)[0][0]
        future_predictions.append(next_pred)
  
        new_input = np.append(last_60_days[0, 1:, 0], next_pred).reshape(1, 60, 1)
        last_60_days = new_input

    predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    result = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})
    return result.to_dict(orient='records')
