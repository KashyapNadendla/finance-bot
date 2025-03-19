import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# Predefined list of crypto symbols to check
CRYPTO_SYMBOLS = ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "DOGE"]

def get_top_movers():
    """
    Fetch daily crypto data for a set of cryptocurrencies from Alpha Vantage,
    calculate the 24h percent change based on the latest two valid days' closing prices,
    sort them by percent change (descending), and return the top 10 movers.
    """
    crypto_data = []
    for symbol in CRYPTO_SYMBOLS:
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": "USD",
            "apikey": API_KEY
        }
        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            if "Time Series (Digital Currency Daily)" not in data:
                st.warning(f"No daily data found for {symbol}. Response: {data}")
                continue

            time_series = data["Time Series (Digital Currency Daily)"]
            dates = sorted(time_series.keys(), reverse=True)
            valid_dates = []
            for date in dates:
                day_data = time_series[date]
                close_key = None
                if "4a. close (USD)" in day_data:
                    close_key = "4a. close (USD)"
                elif "4b. close (USD)" in day_data:
                    close_key = "4b. close (USD)"
                if close_key is not None:
                    valid_dates.append((date, close_key))
                if len(valid_dates) == 2:
                    break

            if len(valid_dates) < 2:
                st.warning(f"Not enough valid close price data for {symbol}.")
                continue

            (latest_date, close_key_latest), (prev_date, close_key_prev) = valid_dates
            close_today = float(time_series[latest_date][close_key_latest])
            close_prev = float(time_series[prev_date][close_key_prev])
            percent_change = ((close_today - close_prev) / close_prev) * 100

            crypto_data.append({
                "Name": data.get("Meta Data", {}).get("2. Digital Currency Code", symbol),
                "Symbol": symbol,
                "Price (USD)": close_today,
                "24h Change (%)": percent_change
            })

        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            st.error(f"Connection error for {symbol}: {e}")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response for {symbol}: {e}")

    if not crypto_data:
        st.error("No cryptocurrency data retrieved.")
        return []

    crypto_data = sorted(crypto_data, key=lambda x: x["24h Change (%)"], reverse=True)
    top_movers = crypto_data[:10]
    for item in top_movers:
        item["Price (USD)"] = f"${item['Price (USD)']:.2f}"
        item["24h Change (%)"] = f"{item['24h Change (%)']:.2f}%"
    return top_movers

def get_crypto_prices():
    """
    Fetch the current price for a subset of cryptocurrencies (e.g., BTC and ETH)
    from Alpha Vantage.
    """
    crypto_prices = {}
    for symbol in ["BTC", "ETH"]:
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": "USD",
            "apikey": API_KEY
        }
        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            if "Time Series (Digital Currency Daily)" in data:
                dates = sorted(data["Time Series (Digital Currency Daily)"].keys(), reverse=True)
                latest_day = dates[0]
                latest_data = data["Time Series (Digital Currency Daily)"][latest_day]
                close_key = None
                if "4a. close (USD)" in latest_data:
                    close_key = "4a. close (USD)"
                elif "4b. close (USD)" in latest_data:
                    close_key = "4b. close (USD)"
                if close_key is None:
                    crypto_prices[symbol] = "N/A"
                else:
                    close_price = float(latest_data[close_key])
                    crypto_prices[symbol] = f"${close_price:.2f}"
            else:
                crypto_prices[symbol] = "N/A"
        except Exception as e:
            st.error(f"Error fetching price for {symbol}: {e}")
            crypto_prices[symbol] = "Error"
    return crypto_prices