import pandas as pd
import streamlit as st
import requests
from data_fetcher import fetch_indicator_data

def parse_technical_indicators(symbol: str) -> dict:
    """
    Fetch different technical indicators from Alpha Vantage for the given symbol
    and parse them into a dictionary. 
    This is just an example - tailor to your needs.
    """
    indicators = {}
    # Examples of function_name: "SMA", "EMA", "MACD", "RSI", "STOCHRSI", "BBANDS"
    # We'll do an RSI fetch here as a demonstration:
    rsi_data = fetch_indicator_data(symbol, function_name="RSI", interval="daily", time_period=10, series_type="close")
    # parse the JSON for RSI
    if "Technical Analysis: RSI" in rsi_data:
        ta_rsi = rsi_data["Technical Analysis: RSI"]
        # The keys here are dates. We can pick the most recent date:
        dates = sorted(ta_rsi.keys())
        if dates:
            latest_date = dates[-1]
            latest_rsi = ta_rsi[latest_date]["RSI"]
            indicators["RSI"] = float(latest_rsi)

    # We can do the same for MACD, STOCHRSI, etc.
    macd_data = fetch_indicator_data(symbol, function_name="MACD", interval="daily", series_type="close")
    if "Technical Analysis: MACD" in macd_data:
        ta_macd = macd_data["Technical Analysis: MACD"]
        dates = sorted(ta_macd.keys())
        if dates:
            latest_date = dates[-1]
            macd_val = ta_macd[latest_date]["MACD"]
            signal_val = ta_macd[latest_date]["MACD_Signal"]
            indicators["MACD"] = float(macd_val)
            indicators["MACD_Signal"] = float(signal_val)

    # Add more calls for STOCHRSI, BBANDS, etc. as needed

    return indicators
