import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime, timedelta

def get_asset_history(ticker, period="1y"):
    """
    Download historical data for the given ticker.
    """
    df = yf.download(ticker, period=period)
    return df

def forecast_arima(ticker, forecast_days=30, order=(5,1,0)):
    """
    Forecast the asset's closing price using ARIMA.
    Returns the historical series and a forecast series.
    """
    df = get_asset_history(ticker)
    if df.empty:
        return None, None
    series = df['Close']
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast_values, index=forecast_index)
    return series, forecast_series

def forecast_prophet(ticker, forecast_days=30):
    """
    Forecast the asset's closing price using Prophet.
    Returns the historical series and a forecast series.
    """
    df = get_asset_history(ticker)
    if df.empty:
        return None, None
    # Prophet expects columns: ds (date) and y (value)
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    # Return the historical portion and the forecast for the future period
    hist = df_prophet.set_index('ds')['y']
    forecast_series = forecast.set_index('ds')['yhat'][-forecast_days:]
    return hist, forecast_series

def plot_forecast(history, forecast, title="Forecast"):
    """
    Create a simple matplotlib line chart that shows historical data and forecast.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.index, history.values, label="Historical")
    ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
    ax.set_title(title)
    ax.legend()
    return fig
