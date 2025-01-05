import os
import requests
import math
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

AV_API_KEY = os.getenv("AV_API_KEY")  # Our Alpha Vantage key

def fetch_stock_data(symbol: str) -> dict:
    """
    Fetch daily stock data from Alpha Vantage for a given symbol.
    Returns the latest close price, a daily price change, etc.
    """
    if not AV_API_KEY:
        st.error("Missing Alpha Vantage API Key in .env (AV_API_KEY).")
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return {}
        # The daily keys are typically in descending order (latest first)
        # We'll just get the first one
        latest_date = sorted(time_series.keys())[-1]
        daily_data = time_series[latest_date]

        close_price = float(daily_data["4. close"])
        open_price = float(daily_data["1. open"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": symbol,
            "Current Price": f"${close_price:.2f}",
            "Price Change (Today)": f"{price_change:.2f}%",
            # We can add more fields as needed
        }
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return {}

def fetch_forex_data(from_symbol: str, to_symbol: str) -> dict:
    """
    Fetch daily forex data from Alpha Vantage for a given currency pair.
    """
    if not AV_API_KEY:
        st.error("Missing Alpha Vantage API Key in .env (AV_API_KEY).")
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        time_series = data.get("Time Series FX (Daily)", {})
        if not time_series:
            return {}
        latest_date = sorted(time_series.keys())[-1]
        daily_data = time_series[latest_date]

        close_price = float(daily_data["4. close"])
        open_price = float(daily_data["1. open"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": f"{from_symbol}/{to_symbol}",
            "Current Price": f"${close_price:.4f}",
            "Price Change (Today)": f"{price_change:.2f}%",
        }
    except Exception as e:
        st.error(f"Error fetching forex data for {from_symbol}/{to_symbol}: {e}")
        return {}

def fetch_crypto_data(symbol: str, market="USD") -> dict:
    """
    Fetch daily crypto data from Alpha Vantage for a given symbol and market.
    """
    if not AV_API_KEY:
        st.error("Missing Alpha Vantage API Key in .env (AV_API_KEY).")
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": market,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        time_series = data.get("Time Series (Digital Currency Daily)", {})
        if not time_series:
            return {}
        latest_date = sorted(time_series.keys())[-1]
        daily_data = time_series[latest_date]

        close_price = float(daily_data["4a. close (USD)"])  # if market="USD"
        open_price = float(daily_data["1a. open (USD)"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": f"{symbol}/{market}",
            "Current Price": f"${close_price:.2f}",
            "Price Change (Today)": f"{price_change:.2f}%",
        }
    except Exception as e:
        st.error(f"Error fetching crypto data for {symbol}/{market}: {e}")
        return {}

def fetch_commodity_data() -> list:
    """
    Fetch monthly commodity data from Alpha Vantage (ALL_COMMODITIES).
    Returns a list. 
    This is just an example – you'll likely parse out a smaller subset 
    that interests your users.
    """
    if not AV_API_KEY:
        st.error("Missing Alpha Vantage API Key in .env (AV_API_KEY).")
        return []

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "ALL_COMMODITIES",
        "interval": "monthly",
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        # data might have multiple commodity series - you'll parse accordingly
        # We'll just return it raw or do some minimal handling
        return [data]  # For demonstration, returning entire response in a list
    except Exception as e:
        st.error(f"Error fetching commodity data: {e}")
        return []

def fetch_indicator_data(symbol: str, function_name: str, interval: str = "daily", time_period: int = 10, series_type: str = "close") -> dict:
    """
    Generic function to fetch a technical indicator from Alpha Vantage.
    function_name can be one of: RSI, MACD, STOCHRSI, SMA, EMA, BBANDS, etc.
    """
    if not AV_API_KEY:
        st.error("Missing Alpha Vantage API Key in .env (AV_API_KEY).")
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function_name,   # e.g. RSI, MACD, SMA, STOCHRSI, etc.
        "symbol": symbol,
        "interval": interval,       # e.g. daily, weekly
        "time_period": time_period, # for RSI, STOCHRSI, etc.
        "series_type": series_type, # open, close, etc.
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        return data
    except Exception as e:
        st.error(f"Error fetching {function_name} for {symbol}: {e}")
        return {}

def fetch_all_assets() -> list:
    """
    Example aggregator to fetch some stocks, forex pairs, cryptos, and commodities 
    from Alpha Vantage, returning a combined list of dictionaries.
    """
    # For demonstration, let's just fetch a few. 
    # You can expand or make it user-driven.
    stocks_to_fetch = ["IBM", "AAPL", "GOOGL"]
    forex_pairs = [("EUR", "USD"), ("GBP", "USD")]
    cryptos = [("BTC", "USD"), ("ETH", "USD")]

    results = []
    # 1) Stocks
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for sym in stocks_to_fetch:
            futures[executor.submit(fetch_stock_data, sym)] = sym

        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    # 2) Forex
    for (f, t) in forex_pairs:
        forex_data = fetch_forex_data(f, t)
        if forex_data:
            results.append(forex_data)

    # 3) Crypto
    for (c, market) in cryptos:
        crypto_data = fetch_crypto_data(c, market)
        if crypto_data:
            results.append(crypto_data)

    # 4) Commodities (optional)
    # commodity_data = fetch_commodity_data()
    # if commodity_data:
    #     # parse or combine them if you want to display in a table
    #     pass

    return results

def get_top_movers() -> list:
    """
    You can choose to implement a separate logic using the 
    fetch_crypto_data or some aggregator logic. 
    For simplicity, let's say we rely on your existing
    CoinMarketCap logic or skip this here.
    """
    # Return an empty list or integrate your existing code
    return []
