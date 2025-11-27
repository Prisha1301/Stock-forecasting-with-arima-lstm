import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def forecast_with_arima(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y', interval='1d')
        if data.empty:
            return {"error": "No data available for the selected stock"}

        close_prices = data['Close']

        model = ARIMA(close_prices, order=(5, 1, 0))  
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=30)

        forecast_dates = pd.date_range(start=close_prices.index[-1], periods=31, freq='D')[1:]
        forecast_data = [{"ds": date, "yhat": price} for date, price in zip(forecast_dates, forecast)]

        return forecast_data

    except Exception as e:
        return {"error": str(e)}