import streamlit as st
import os
import requests
from dotenv import load_dotenv
load_dotenv()

CMC_API_KEY = os.getenv("CMC_API_KEY")
BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

def get_coinmarketcap_crypto_data(limit=100):
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
    Return the top 100 movers sorted by absolute 24h change percentage.
    """
    crypto_data = get_coinmarketcap_crypto_data(limit=100)
    if not crypto_data:
        st.error("No cryptocurrency data retrieved.")
        return []
    
    # Sort by absolute 24h percentage change
    crypto_data = sorted(crypto_data, key=lambda x: abs(x["24h Change (%)"] or 0), reverse=True)
    top_movers = crypto_data[:100]
    
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
    crypto_data = get_coinmarketcap_crypto_data(limit=100)
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
