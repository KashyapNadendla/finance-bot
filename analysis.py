import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from TA import TechnicalAnalysis
import requests
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def perform_technical_analysis(ticker, period="1y", interval="1d"):
    """
    Fetch historical data and compute technical indicators.
    Returns a dictionary summary and a dataframe for plotting.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            return {"error": "No data available"}, pd.DataFrame()
        ta_analyzer = TechnicalAnalysis(ticker)
        ta_df = ta_analyzer.compute_local_ta(hist)
        summary = ta_analyzer.aggregate_ta(ta_df)
        return summary, ta_df
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker):
    """
    Fetch fundamental data for a stock using Alpha Vantage.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_data = requests.get(url).json()
        url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        income_data = requests.get(url).json()
        url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        balance_data = requests.get(url).json()
        url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        cash_flow_data = requests.get(url).json()
        fundamental_data = {
            "overview": overview_data,
            "income_statement": income_data.get("annualReports", []),
            "balance_sheet": balance_data.get("annualReports", []),
            "cash_flow": cash_flow_data.get("annualReports", [])
        }
        return fundamental_data
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_macro_indicators():
    """
    Fetch macroeconomic indicators from Alpha Vantage for:
      - Real GDP
      - Real GDP per Capita
      - Treasury Yield
      - Federal Funds Rate
      - CPI
      - Inflation
      - Retail Sales
      - Durable Goods Orders
      - Unemployment Rate
      - Nonfarm Payroll
    Then, fetch DXY data from yfinance using ticker "DX-Y.NYB".
    """
    indicators = {}
    try:
        # Real GDP
        url = f"https://www.alphavantage.co/query?function=REAL_GDP&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Real GDP"] = data.get("data", [])
        
        # Real GDP per Capita
        url = f"https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Real GDP per Capita"] = data.get("data", [])
        
        # Treasury Yield (10-year, monthly)
        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Treasury Yield"] = data.get("data", [])
        
        # Federal Funds Rate
        url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Federal Funds Rate"] = data.get("data", [])
        
        # CPI (monthly)
        url = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["CPI"] = data.get("data", [])
        
        # Inflation
        url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Inflation"] = data.get("data", [])
        
        # Retail Sales
        url = f"https://www.alphavantage.co/query?function=RETAIL_SALES&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Retail Sales"] = data.get("data", [])
        
        # Durable Goods Orders
        url = f"https://www.alphavantage.co/query?function=DURABLE_GOODS_ORDERS&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Durable Goods Orders"] = data.get("data", [])
        
        # Unemployment Rate
        url = f"https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Unemployment Rate"] = data.get("data", [])
        
        # Nonfarm Payroll
        url = f"https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey={ALPHA_VANTAGE_API_KEY}"
        data = requests.get(url).json()
        indicators["Nonfarm Payroll"] = data.get("data", [])
    except Exception as e:
        indicators["error"] = str(e)
    
    # DXY from yfinance using ticker "DX-Y.NYB"
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="5d")
        if not hist.empty:
            prices = hist['Close']
            trend = "decreasing" if prices.iloc[-1] < prices.iloc[0] else "increasing"
            indicators["DXY"] = {"Trend": trend, "Latest": prices.iloc[-1]}
        else:
            indicators["DXY"] = {"Trend": "N/A"}
    except Exception as e:
        indicators["DXY"] = {"Trend": "N/A", "error": str(e)}
    
    return indicators


def generate_ta_chart(ta_df, ticker):
    """
    Generate a Plotly chart for technical analysis.
    """
    if ta_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ta_df.index,
        open=ta_df['open'],
        high=ta_df['high'],
        low=ta_df['low'],
        close=ta_df['close'],
        name='Price'
    ))
    if 'sma_20' in ta_df.columns:
        fig.add_trace(go.Scatter(
            x=ta_df.index,
            y=ta_df['sma_20'],
            name='SMA 20',
            line=dict(width=1)
        ))
    if 'sma_50' in ta_df.columns:
        fig.add_trace(go.Scatter(
            x=ta_df.index,
            y=ta_df['sma_50'],
            name='SMA 50',
            line=dict(width=1)
        ))
    if 'sma_200' in ta_df.columns:
        fig.add_trace(go.Scatter(
            x=ta_df.index,
            y=ta_df['sma_200'],
            name='SMA 200',
            line=dict(width=1)
        ))
    if 'BBU_20_2.0' in ta_df.columns and 'BBL_20_2.0' in ta_df.columns:
        fig.add_trace(go.Scatter(
            x=ta_df.index,
            y=ta_df['BBU_20_2.0'],
            name='BB Upper',
            line=dict(width=1, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=ta_df.index,
            y=ta_df['BBL_20_2.0'],
            name='BB Lower',
            line=dict(width=1, dash="dash"),
            fill='tonexty',
            fillcolor='rgba(250, 0, 0, 0.05)'
        ))
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if 'rsi' in ta_df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ta_df.index,
            y=ta_df['rsi'],
            name='RSI',
            line=dict(width=1)
        ))
        fig2.add_shape(
            type="line",
            x0=ta_df.index[0],
            y0=70,
            x1=ta_df.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash")
        )
        fig2.add_shape(
            type="line",
            x0=ta_df.index[0],
            y0=30,
            x1=ta_df.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash")
        )
        fig2.update_layout(
            title="RSI (14)",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=200,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        return fig, fig2
    return fig, None

def get_technical_analysis_summaries(asset_data):
    """
    For each asset, compute a technical analysis summary.
    """
    summaries = ""
    for asset in asset_data:
        ticker = asset["Ticker"] if "Ticker" in asset else asset
        summary, _ = perform_technical_analysis(ticker)
        if "error" in summary:
            summaries += f"{ticker}: Error - {summary['error']}\n\n"
            continue
        summaries += f"{ticker} Technical Analysis:\n"
        if "overall_signal" in summary:
            summaries += f"Overall Signal: {summary['overall_signal']}\n"
        if "rsi" in summary:
            summaries += f"RSI: {summary['rsi']['value']:.2f} - {summary['rsi']['signal']}\n"
        if "macd" in summary:
            summaries += f"MACD: {summary['macd']['interpretation']}\n"
        if "moving_averages" in summary:
            summaries += f"Trend: {summary['moving_averages']['trend']}\n"
        if "bollinger_bands" in summary:
            summaries += f"Bollinger Bands: {summary['bollinger_bands']}\n\n"
    return summaries

def get_aggregated_insights(ticker_list):
    """
    For each ticker in ticker_list, perform technical analysis and aggregate insights.
    Returns a dictionary mapping ticker to its technical analysis summary.
    """
    aggregated = {}
    for ticker in ticker_list:
        summary, _ = perform_technical_analysis(ticker)
        aggregated[ticker] = summary
    return aggregated

def format_insights(aggregated_insights):
    """
    Format the aggregated insights dictionary into a human-readable string.
    """
    output = ""
    for ticker, summary in aggregated_insights.items():
        output += f"Ticker: {ticker}\n"
        if isinstance(summary, dict):
            if "error" in summary:
                output += f"  Error: {summary['error']}\n"
            else:
                # Customize the formatting as needed.
                if "overall_signal" in summary:
                    output += f"  Overall Signal: {summary['overall_signal']}\n"
                if "rsi" in summary:
                    output += f"  RSI: {summary['rsi']['value']:.2f} ({summary['rsi']['signal']})\n"
                if "macd" in summary:
                    output += f"  MACD: {summary['macd']['interpretation']}\n"
                if "moving_averages" in summary:
                    output += f"  Trend: {summary['moving_averages']['trend']}\n"
        else:
            output += f"  {summary}\n"
        output += "\n"
    return output
