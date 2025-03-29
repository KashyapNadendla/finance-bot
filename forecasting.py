import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
def get_asset_history(ticker, asset_type="stock", period="1y"):
    """
    Download historical daily data for the given ticker using the Alpha Vantage API.
    For stocks, uses TIME_SERIES_DAILY_ADJUSTED.
    For crypto, uses DIGITAL_CURRENCY_DAILY.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = "https://www.alphavantage.co/query"
    
    if asset_type.lower() == "stock":
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": api_key
        }
    elif asset_type.lower() == "crypto":
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": ticker,
            "market": "USD",
            "apikey": api_key
        }
    else:
        return pd.DataFrame()  # unsupported type

    r = requests.get(url, params=params)
    data = r.json()
    
    # Parse JSON according to asset type
    if asset_type.lower() == "stock":
        ts = data.get("Time Series (Daily)", {})
    else:
        ts = data.get("Time Series (Digital Currency Daily)", {})
    
    if not ts:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    if asset_type.lower() == "stock":
        df = df.rename(columns={"4. close": "Close"})
    else:
        df = df.rename(columns={"4a. close (USD)": "Close"})
    df["Close"] = pd.to_numeric(df["Close"])
    
    # Filter by period
    if period != "full":
        now = datetime.now()
        if period.endswith("y"):
            years = int(period[:-1])
            start_date = now - pd.DateOffset(years=years)
        elif period.endswith("m"):
            months = int(period[:-1])
            start_date = now - pd.DateOffset(months=months)
        else:
            start_date = now - pd.DateOffset(years=1)
        df = df[df.index >= start_date]
    return df


def forecast_arima(ticker, forecast_days=30, order=(5,1,0), asset_type="stock"):
    """
    Forecast the asset's closing price using ARIMA.
    Returns the historical series and a forecast series.
    """
    df = get_asset_history(ticker, asset_type=asset_type)
    if df.empty:
        return None, None
    series = df["Close"]
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast_values, index=forecast_index)
    return series, forecast_series

def forecast_prophet(ticker, forecast_days=30, asset_type="stock"):
    """
    Forecast the asset's closing price using Prophet.
    Returns the historical series and a forecast series.
    """
    # Pass the asset_type to get_asset_history
    df = get_asset_history(ticker, asset_type=asset_type)
    if df.empty:
        return None, None
    # Prophet requires columns: ds (date) and y (value)
    df_prophet = df.reset_index()[["index", "Close"]].rename(columns={"index": "ds", "Close": "y"})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    hist = df_prophet.set_index("ds")["y"]
    forecast_series = forecast.set_index("ds")["yhat"][-forecast_days:]
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
