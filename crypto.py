# import streamlit as st
# import os
# import requests
# import json
# from dotenv import load_dotenv

# # Ensure environment variables are loaded
# load_dotenv()

# API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
# BASE_URL = "https://www.alphavantage.co/query"

# # Predefined list of crypto symbols to check
# CRYPTO_SYMBOLS = ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "DOGE"]

# def get_top_movers():
#     """
#     Fetch daily crypto data for a set of cryptocurrencies from Alpha Vantage,
#     calculate the 24h percent change based on the latest two valid days' closing prices,
#     sort them by percent change (descending), and return the top 10 movers.
#     """
#     crypto_data = []
#     for symbol in CRYPTO_SYMBOLS:
#         params = {
#             "function": "DIGITAL_CURRENCY_DAILY",
#             "symbol": symbol,
#             "market": "USD",
#             "apikey": API_KEY
#         }
#         try:
#             response = requests.get(BASE_URL, params=params)
#             data = response.json()
#             if "Time Series (Digital Currency Daily)" not in data:
#                 st.warning(f"No daily data found for {symbol}. Response: {data}")
#                 continue

#             time_series = data["Time Series (Digital Currency Daily)"]
#             dates = sorted(time_series.keys(), reverse=True)
#             valid_dates = []
#             for date in dates:
#                 day_data = time_series[date]
#                 close_key = None
#                 if "4a. close (USD)" in day_data:
#                     close_key = "4a. close (USD)"
#                 elif "4b. close (USD)" in day_data:
#                     close_key = "4b. close (USD)"
#                 if close_key is not None:
#                     valid_dates.append((date, close_key))
#                 if len(valid_dates) == 2:
#                     break

#             if len(valid_dates) < 2:
#                 st.warning(f"Not enough valid close price data for {symbol}.")
#                 continue

#             (latest_date, close_key_latest), (prev_date, close_key_prev) = valid_dates
#             close_today = float(time_series[latest_date][close_key_latest])
#             close_prev = float(time_series[prev_date][close_key_prev])
#             percent_change = ((close_today - close_prev) / close_prev) * 100

#             crypto_data.append({
#                 "Name": data.get("Meta Data", {}).get("2. Digital Currency Code", symbol),
#                 "Symbol": symbol,
#                 "Price (USD)": close_today,
#                 "24h Change (%)": percent_change
#             })

#         except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
#             st.error(f"Connection error for {symbol}: {e}")
#         except json.JSONDecodeError as e:
#             st.error(f"Error parsing JSON response for {symbol}: {e}")

#     if not crypto_data:
#         st.error("No cryptocurrency data retrieved.")
#         return []

#     crypto_data = sorted(crypto_data, key=lambda x: x["24h Change (%)"], reverse=True)
#     top_movers = crypto_data[:10]
#     for item in top_movers:
#         item["Price (USD)"] = f"${item['Price (USD)']:.2f}"
#         item["24h Change (%)"] = f"{item['24h Change (%)']:.2f}%"
#     return top_movers

# def get_crypto_prices():
#     """
#     Fetch the current price for a subset of cryptocurrencies (e.g., BTC and ETH)
#     from Alpha Vantage.
#     """
#     crypto_prices = {}
#     for symbol in ["BTC", "ETH"]:
#         params = {
#             "function": "DIGITAL_CURRENCY_DAILY",
#             "symbol": symbol,
#             "market": "USD",
#             "apikey": API_KEY
#         }
#         try:
#             response = requests.get(BASE_URL, params=params)
#             data = response.json()
#             if "Time Series (Digital Currency Daily)" in data:
#                 dates = sorted(data["Time Series (Digital Currency Daily)"].keys(), reverse=True)
#                 latest_day = dates[0]
#                 latest_data = data["Time Series (Digital Currency Daily)"][latest_day]
#                 close_key = None
#                 if "4a. close (USD)" in latest_data:
#                     close_key = "4a. close (USD)"
#                 elif "4b. close (USD)" in latest_data:
#                     close_key = "4b. close (USD)"
#                 if close_key is None:
#                     crypto_prices[symbol] = "N/A"
#                 else:
#                     close_price = float(latest_data[close_key])
#                     crypto_prices[symbol] = f"${close_price:.2f}"
#             else:
#                 crypto_prices[symbol] = "N/A"
#         except Exception as e:
#             st.error(f"Error fetching price for {symbol}: {e}")
#             crypto_prices[symbol] = "Error"
#     return crypto_prices

# def fetch_crypto_data():
#     """
#     Returns crypto data for display.
#     This can be an alias for get_top_movers() or get_crypto_prices().
#     """
#     # For example, you can return the top movers.
#     return get_top_movers()

# crypto.py
import streamlit as st
import os
import requests
from dotenv import load_dotenv
load_dotenv()

CMC_API_KEY = os.getenv("CMC_API_KEY")
BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

def get_coinmarketcap_crypto_data(limit=50):
    """
    Fetch cryptocurrency data from CoinMarketCap.
    """
    if not CMC_API_KEY:
        st.error("CMC_API_KEY not set. Cannot fetch crypto data.")
        return []
    
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_API_KEY,
    }
    params = {
        "start": "1",
        "limit": str(limit),
        "convert": "USD"
    }
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        data = response.json()
        crypto_data = []
        for crypto in data.get("data", []):
            crypto_data.append({
                "Name": crypto.get("name"),
                "Symbol": crypto.get("symbol"),
                "Price (USD)": crypto.get("quote", {}).get("USD", {}).get("price"),
                "24h Change (%)": crypto.get("quote", {}).get("USD", {}).get("percent_change_24h")
            })
        return crypto_data
    except Exception as e:
        st.error(f"Error fetching crypto data from CoinMarketCap: {e}")
        return []

def get_top_movers():
    """
    Return the top 10 movers sorted by absolute 24h change percentage.
    """
    crypto_data = get_coinmarketcap_crypto_data(limit=50)
    if not crypto_data:
        st.error("No cryptocurrency data retrieved.")
        return []
    
    # Sort by absolute 24h percentage change
    crypto_data = sorted(crypto_data, key=lambda x: abs(x["24h Change (%)"] or 0), reverse=True)
    top_movers = crypto_data[:10]
    
    # Format the numbers
    for item in top_movers:
        if item["Price (USD)"] is not None:
            item["Price (USD)"] = f"${item['Price (USD)']:.2f}"
        if item["24h Change (%)"] is not None:
            item["24h Change (%)"] = f"{item['24h Change (%)']:.2f}%"
    return top_movers

def get_crypto_prices():
    """
    Fetch the current price for a subset of cryptocurrencies (e.g., BTC and ETH)
    from CoinMarketCap.
    """
    crypto_data = get_coinmarketcap_crypto_data(limit=50)
    prices = {}
    for symbol in ["BTC", "ETH"]:
        for crypto in crypto_data:
            if crypto.get("Symbol") == symbol:
                price = crypto.get("quote", {}).get("USD", {}).get("price")
                prices[symbol] = f"${price:.2f}" if price is not None else "N/A"
                break
        else:
            prices[symbol] = "N/A"
    return prices

def fetch_crypto_data():
    """
    Returns crypto data for display (alias for get_top_movers).
    """
    return get_top_movers()
