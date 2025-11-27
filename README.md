StockVision: AI-Powered Stock Price Forecasting
Project Overview
StockVision is a web-based stock forecasting application that leverages advanced machine learning models—ARIMA and LSTM—to predict future stock prices and analyze market trends. The platform provides traders and investors with accurate price forecasts, technical analysis indicators, and comprehensive stock data visualization across global and Indian markets.

Key Features
Dual Forecasting Models

ARIMA (AutoRegressive Integrated Moving Average): Statistical time-series model ideal for capturing temporal patterns and trends in historical stock data
LSTM (Long Short-Term Memory): Deep learning neural network that captures complex non-linear relationships and long-term dependencies in price movements
Real-Time Stock Data

Live price data fetching from Yahoo Finance API
Support for intraday (1-minute and 5-minute) and daily intervals
Automatic fallback mechanisms for data availability
Comprehensive Stock Universe

50+ stocks covering major US tech companies (AAPL, GOOGL, MSFT, AMZN, TSLA)
40+ Indian stock market leaders (NSE stocks) from diverse sectors including banking, IT, pharmaceuticals, and manufacturing
Technical Analysis Dashboard

Moving Averages (20, 50, 200-day)
Bollinger Bands for volatility assessment
Relative Strength Index (RSI) for momentum analysis
MACD (Moving Average Convergence Divergence) for trend identification
Trendline analysis and volatility calculations
Interactive visualization of technical indicators
Smart Stock Search

Search stocks by ticker symbol or company name
Real-time market data including current price, market cap, P/E ratio, and dividend yield
Filtered results from pre-curated stock list
Live Dashboard

View current stock prices with OHLCV data
Real-time market updates
Quick access to multiple stock quotes
Technology Stack
Backend:

Flask (Web framework)
Python 3.x
Data & APIs:

yfinance (Stock data retrieval)
Pandas (Data manipulation)
NumPy (Numerical computations)
Machine Learning:

ARIMA (Statsmodels)
LSTM (TensorFlow/Keras)
Frontend:

HTML5, CSS3
JavaScript (for interactive charts and visualizations)
Additional Libraries:

Flask-CORS (Cross-origin support)
Logging (Application monitoring)
Core Modules
app.py: Main Flask application with routes for dashboard, forecasting, search, and technical analysis
forecast_arima.py: ARIMA model implementation for statistical time-series forecasting
forecast_with_lstm.py: LSTM neural network model for deep learning-based price predictions
Templates: HTML templates for dashboard, forecast results, search interface, and technical analysis
Use Cases
Short-term Trading: Use LSTM model for capturing market volatility and rapid price changes
Long-term Investing: Use ARIMA model for identifying trends and seasonal patterns
Risk Management: Technical indicators help identify support/resistance levels and market momentum
Portfolio Analysis: Compare multiple stocks and their forecasts simultaneously
Market Research: Analyze historical price movements and technical patterns
How It Works
Select a stock ticker from the available list
Choose between ARIMA or LSTM forecasting model
The application fetches historical price data
Machine learning model processes the data and generates predictions
View forecast results with visualizations
Analyze technical indicators for informed decision-making
Getting Started
Install dependencies: pip install -r requirements.txt
Run the Flask application: python app.py
Access the web interface at http://localhost:5000
Navigate to the Forecast section to select a stock and model
View predictions and technical analysis on the dashboard
Future Enhancements
Real-time prediction updates
Custom date range forecasting
Model performance comparison metrics
Portfolio optimization recommendations
