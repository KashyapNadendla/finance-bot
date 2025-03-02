import streamlit as st
import yfinance as yf
import ta

def perform_technical_analysis(ticker, period="1y", interval="1d"):
    """
    Fetch historical data and compute technical indicators:
      - RSI (14-period)
      - 20-day Simple Moving Average (SMA20)
      - Bollinger Bands (20-day window, 2 std dev)
    Returns a summary string of the latest values.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return "No data available"
    try:
        rsi = ta.momentum.RSIIndicator(hist["Close"], window=14).rsi().iloc[-1]
    except Exception as e:
        rsi = None
    sma20 = hist["Close"].rolling(window=20).mean().iloc[-1]
    bb_indicator = ta.volatility.BollingerBands(hist["Close"], window=20, window_dev=2)
    bb_upper = bb_indicator.bollinger_hband().iloc[-1]
    bb_lower = bb_indicator.bollinger_lband().iloc[-1]
    latest_close = hist["Close"].iloc[-1]
    if rsi is not None:
        summary = (
            f"Price: ${latest_close:.2f}, "
            f"RSI: {rsi:.2f}, "
            f"SMA20: ${sma20:.2f}, "
            f"Bollinger Bands: Upper=${bb_upper:.2f}, Lower=${bb_lower:.2f}"
        )
    else:
        summary = (
            f"Price: ${latest_close:.2f}, "
            f"SMA20: ${sma20:.2f}, "
            f"Bollinger Bands: Upper=${bb_upper:.2f}, Lower=${bb_lower:.2f}"
        )
    return summary

def get_technical_analysis_summaries(asset_data):
    """
    For each asset, compute a technical analysis summary on daily and weekly timeframes.
    """
    summaries = ""
    for asset in asset_data:
        ticker = asset["Ticker"]
        daily_summary = perform_technical_analysis(ticker, period="1y", interval="1d")
        weekly_summary = perform_technical_analysis(ticker, period="2y", interval="1wk")
        summaries += (
            f"{ticker} Technical Analysis:\n"
            f"Daily: {daily_summary}\n"
            f"Weekly: {weekly_summary}\n\n"
        )
    return summaries
