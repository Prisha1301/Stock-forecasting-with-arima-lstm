from flask import Flask, render_template, request, jsonify, redirect, url_for,flash
from flask_cors import CORS
from forecast_with_lstm import forecast_with_lstm
from forecast_arima import forecast_with_arima
import yfinance as yf
import pandas as pd
import logging
import numpy as np

app = Flask(__name__)
CORS(app) 
app.secret_key = 'your-secret-key' 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
@app.route('/')
def home():
    return render_template('index.html')

STOCKS = [
    {"ticker": "AAPL", "name": "Apple (AAPL)"},
    {"ticker": "GOOGL", "name": "Google (GOOGL)"},
    {"ticker": "MSFT", "name": "Microsoft (MSFT)"},
    {"ticker": "AMZN", "name": "Amazon (AMZN)"},
    {"ticker": "TSLA", "name": "Tesla (TSLA)"},
    {"ticker": "ASIANPAINT.NS", "name": "Asian Paints Ltd"},
    {"ticker": "CIPLA.NS", "name": "Cipla Ltd"},
    {"ticker": "EICHERMOT.NS", "name": "Eicher Motors Ltd"},
    {"ticker": "NESTLEIND.NS", "name": "Nestle India Ltd"},
    {"ticker": "GRASIM.NS", "name": "Grasim Industries Ltd"},
    {"ticker": "HEROMOTOCO.NS", "name": "Hero MotoCorp Ltd"},
    {"ticker": "HINDALCO.NS", "name": "Hindalco Industries Ltd"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd"},
    {"ticker": "ITC.NS", "name": "ITC Ltd"},
    {"ticker": "TRENT.NS", "name": "Trent Ltd"},
    {"ticker": "LT.NS", "name": "Larsen & Toubro Ltd"},
    {"ticker": "M&M.NS", "name": "Mahindra & Mahindra Ltd"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd"},
    {"ticker": "TATACONSUM.NS", "name": "Tata Consumer Products Ltd"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Ltd"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel Ltd"},
    {"ticker": "WIPRO.NS", "name": "Wipro Ltd"},
    {"ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals Enterprise Ltd"},
    {"ticker": "DRREDDY.NS", "name": "Dr Reddys Laboratories Ltd"},
    {"ticker": "TITAN.NS", "name": "Titan Company Ltd"},
    {"ticker": "SBIN.NS", "name": "State Bank of India"},
    {"ticker": "SHRIRAMFIN.NS", "name": "Shriram Finance Ltd"},
    {"ticker": "BEL.NS", "name": "Bharat Electronics Ltd"},
    {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd"},
    {"ticker": "INFY.NS", "name": "Infosys Ltd"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance Ltd"},
    {"ticker": "ADANIENT.NS", "name": "Adani Enterprises Ltd"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Industries Ltd"},
    {"ticker": "JSWSTEEL.NS", "name": "JSW Steel Ltd"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services Ltd"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd"},
    {"ticker": "POWERGRID.NS", "name": "Power Grid Corporation of India Ltd"},
    {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India Ltd"},
    {"ticker": "INDUSINDBK.NS", "name": "Indusind Bank Ltd"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank Ltd"},
    {"ticker": "HCLTECH.NS", "name": "HCL Technologies Ltd"},
    {"ticker": "ONGC.NS", "name": "Oil & Natural Gas Corpn Ltd"},
    {"ticker": "NTPC.NS", "name": "NTPC Ltd"},
    {"ticker": "COALINDIA.NS", "name": "Coal India Ltd"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd"},
    {"ticker": "TECHM.NS", "name": "Tech Mahindra Ltd"},
    {"ticker": "JIOFIN.NS", "name": "Jio Financial Services Ltd"},
    {"ticker": "ADANIPORTS.NS", "name": "Adani Ports & Special Economic Zone Ltd"},
    {"ticker": "HDFCLIFE.NS", "name": "HDFC Life Insurance Company Ltd"},
    {"ticker": "SBILIFE.NS", "name": "SBI Life Insurance Company Ltd"},
    {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd"},
    {"ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto Ltd"},
    {"ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv Ltd"},
    {"ticker": "DIVISLAB.NS", "name": "Divi's Laboratories Ltd"}
]

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    selected_ticker = None
    data = None
    error = None
    if request.method == 'POST':
        selected_ticker = request.form['ticker']
        try:
            df = yf.download(selected_ticker, period='1d', interval='1m')
            if df.empty:
                df = yf.download(selected_ticker, period='5d', interval='5m')
                logging.info(f"Fallback to 5-day 5-minute data for {selected_ticker}")

            if df.empty:
                raise ValueError("No valid data available for the selected stock.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.dropna(subset=['Close'], inplace=True)

            if df.empty:
                raise ValueError("No valid price data available for the selected stock.")

            df.reset_index(inplace=True)
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.rename(columns=lambda x: str(x).capitalize(), inplace=True)

            data = df.to_dict(orient='records')

            logging.info(f"{selected_ticker} data fetched. Last record: {data[-1]}")
        except Exception as e:
            logging.error(f"Error fetching data for {selected_ticker}: {e}")
            error = str(e)

    return render_template('dashboard.html', stocks=STOCKS, ticker=selected_ticker, data=data, error=error)

# AI-Based Stock Forecasting
from forecast_arima import forecast_with_arima

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        ticker = request.form['ticker']
        model = request.form['model']

        if model == 'arima':
            forecast = forecast_with_arima(ticker)
        elif model == 'lstm':
            forecast = forecast_with_lstm(ticker)
        else:
            return render_template('forecast.html', stocks=STOCKS, ticker=ticker, forecast=None, model=model, error="Invalid model selected.")

        if "error" in forecast:
            return render_template('forecast.html', stocks=STOCKS, ticker=ticker, forecast=None, model=model, error=forecast["error"])

        print(f"Forecast data for {ticker} using {model}: {forecast}")

        return render_template('forecast.html', stocks=STOCKS, ticker=ticker, forecast=forecast, model=model, error=None)
    return render_template('forecast.html', stocks=STOCKS, ticker=None, forecast=None, model=None, error=None)

# Search & Track Stocks
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['search']

        # Use the global STOCKS list
        results = [stock for stock in STOCKS if query.lower() in stock['name'].lower() or query.lower() in stock['ticker'].lower()]

        for stock in results:
            try:
                ticker = stock['ticker']
                stock_info = yf.Ticker(ticker).info
                stock['current_price'] = stock_info.get('currentPrice', 'N/A')
                stock['market_cap'] = stock_info.get('marketCap', 'N/A')
                stock['pe_ratio'] = stock_info.get('trailingPE', 'N/A')
                stock['dividend_yield'] = stock_info.get('dividendYield', 'N/A')
            except Exception as e:
                logging.error(f"Error fetching details for {ticker}: {e}")
                stock['current_price'] = 'N/A'
                stock['market_cap'] = 'N/A'
                stock['pe_ratio'] = 'N/A'
                stock['dividend_yield'] = 'N/A'

        return render_template('search.html', query=query, results=results, stocks=STOCKS)

    return render_template('search.html', query=None, results=None, stocks=STOCKS)

@app.route('/technical', methods=['GET', 'POST'])
def technical():
    if request.method == 'POST':
        ticker = request.form['ticker']

        try:
            df = yf.download(ticker, period='1y', interval='1d')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            logging.info(f"Downloaded data for {ticker}:\n{df.head()}")

            if df.empty:
                flash("No data available for the selected ticker.")
                return render_template('technical.html', stocks=STOCKS, ticker=ticker, data=None)

            if 'Close' not in df.columns:
                flash(f"Data for {ticker} missing 'Close' prices.")
                return render_template('technical.html', stocks=STOCKS, ticker=ticker, data=None)

            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            rolling_std = df['Close'].rolling(window=20).std()
            df['Upper_Band'] = df['MA_20'] + (rolling_std * 2)
            df['Lower_Band'] = df['MA_20'] - (rolling_std * 2)

            x = np.arange(len(df))
            coefficients = np.polyfit(x, df['Close'], 1)
            df['Trendline'] = coefficients[0] * x + coefficients[1]

            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

            df.reset_index(inplace=True)
            df['Date'] = df['Date'].astype(str)

            data = df.to_dict(orient='records')

            logging.info(f"Technical data for {ticker} processed.")
            return render_template('technical.html', 
                                stocks=STOCKS,
                                ticker=ticker, 
                                data=data,
                                dates=[item['Date'] for item in data],
                                closes=[item['Close'] for item in data])

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}", exc_info=True)
            flash(f"Error processing {ticker}: {str(e)}")
            return render_template('technical.html', stocks=STOCKS, ticker=ticker, data=None)

    return render_template('technical.html', stocks=STOCKS, ticker=None, data=None)

if __name__ == '__main__':
    app.run(debug=True)