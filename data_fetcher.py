import os
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Load Alpha Vantage API Key from environment variables
AV_API_KEY = os.getenv("AV_API_KEY")  # Alpha Vantage API key

# Error if API key is missing
if not AV_API_KEY:
    st.error("Alpha Vantage API Key (AV_API_KEY) is missing in .env file. Please add it to continue.")

def fetch_data_from_alpha_vantage(params: dict) -> dict:
    """
    Generic function to fetch data from Alpha Vantage.
    
    Parameters:
        params (dict): Dictionary of query parameters for the Alpha Vantage API.
        
    Returns:
        dict: Parsed JSON response from the Alpha Vantage API.
    """
    base_url = "https://www.alphavantage.co/query"
    params["apikey"] = AV_API_KEY
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTP errors
        data = response.json()
        if "Error Message" in data:
            st.error(f"Error from Alpha Vantage: {data['Error Message']}")
            return {}
        elif "Note" in data:
            st.warning(f"Alpha Vantage API rate limit reached. Try again later.")
            return {}
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return {}

def fetch_daily_stock_data(symbol: str, output_size: str = "compact") -> dict:
    """
    Fetch daily stock data from Alpha Vantage for a given symbol.
    
    Parameters:
        symbol (str): Stock ticker symbol (e.g., "IBM").
        output_size (str): "compact" (last 100 data points) or "full" (full-length time series).
        
    Returns:
        dict: Daily stock data with timestamps as keys and OHLC data as values.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": output_size,
    }
    data = fetch_data_from_alpha_vantage(params)
    time_series = data.get("Time Series (Daily)", {})
    if not time_series:
        st.error(f"No daily data found for symbol: {symbol}")
        return {}
    
    # Format the data for better readability
    formatted_data = {}
    for date, values in time_series.items():
        formatted_data[date] = {
            "Open": float(values["1. open"]),
            "High": float(values["2. high"]),
            "Low": float(values["3. low"]),
            "Close": float(values["4. close"]),
            "Volume": int(values["5. volume"]),
        }
    return formatted_data

def fetch_all_assets() -> list:
    """
    Fetch daily data for multiple asset types: stocks, forex, and crypto.
    
    Returns:
        list: Combined asset data for display.
    """
    assets = []
    stocks = ["AAPL", "GOOGL", "MSFT"]  # Example stock symbols

    # Fetch stock data concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        stock_futures = {executor.submit(fetch_daily_stock_data, symbol): symbol for symbol in stocks}
        for future in as_completed(stock_futures):
            symbol = stock_futures[future]
            stock_data = future.result()
            if stock_data:
                # Convert the daily data into a DataFrame for easier display
                df = pd.DataFrame.from_dict(stock_data, orient="index")
                df.index = pd.to_datetime(df.index)
                assets.append({
                    "Ticker": symbol,
                    "Data": df,
                })

    return assets