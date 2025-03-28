import streamlit as st
import numpy as np
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# --- Import local modules ---
import pdf_processing
import news
import stocks
import chat
import budgeting
import agentic
import forecasting
import crypto

# ---------------------- LOAD ENVIRONMENT VARIABLES ---------------------- #
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CMC_API_KEY = os.getenv("CMC_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ---------------------- STORE KEYS IN SESSION STATE ---------------------- #
if API_KEY and "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = API_KEY
if NEWS_API_KEY and "NEWS_API_KEY" not in st.session_state:
    st.session_state["NEWS_API_KEY"] = NEWS_API_KEY
if CMC_API_KEY and "CMC_API_KEY" not in st.session_state:
    st.session_state["CMC_API_KEY"] = CMC_API_KEY
if ALPHA_VANTAGE_API_KEY and "ALPHA_VANTAGE_API_KEY" not in st.session_state:
    st.session_state["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY

# ---------------------- STREAMLIT PAGE CONFIG ---------------------- #
st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# ---------------------- INITIALIZE SESSION STATE ---------------------- #
if "financial_data" not in st.session_state:
    st.session_state["financial_data"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "asset_data" not in st.session_state:
    st.session_state["asset_data"] = []

if "asset_data_timestamp" not in st.session_state:
    st.session_state["asset_data_timestamp"] = None

if "agentic_history" not in st.session_state:
    st.session_state["agentic_history"] = []

# ---------------------- MAIN APP ---------------------- #
def main():
    st.markdown("# Welcome to Your Personal Finance Assistant 💰")

    # On initial load, if we have no asset data, fetch from Alpha Vantage
    if not st.session_state["asset_data"]:
        with st.spinner("Loading fresh stock prices from Alpha Vantage..."):
            st.session_state["asset_data"] = stocks.get_asset_data()
            st.session_state["asset_data_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Display "last updated" info + manual update button
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write("")  # Placeholder if needed
    with col2:
        if st.session_state["asset_data_timestamp"]:
            st.write(f"**Stock prices updated as of:** {st.session_state['asset_data_timestamp']}")
        else:
            st.write("**Stock prices not loaded.**")
        if st.button("Update Stock Prices"):
            with st.spinner("Fetching fresh stock prices..."):
                st.session_state["asset_data"] = stocks.get_asset_data()
                st.session_state["asset_data_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Stock prices updated.")

    # ---------------------- MAIN TABS ---------------------- #
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Assets", "Budgeting", "Agentic Advisor", "Forecasting and News"])

    # 1) INTRODUCTION TAB (renamed from News)
    with tab1:
        st.header("Introduction & Instructions")
        st.write("""
        ### How to Use:
        1. **Assets Tab**: Displays stock and cryptocurrency data. You can check asset prices and view price charts.
        2. **Budgeting Tab**: Helps you track your monthly income, expenses, and savings.
        3. **Agentic Advisor Tab**: Provides personalized investment advice and forecasts.
        4. **Forecasting and News Tab**: Displays forecasted stock prices and recent finance news.

        ### How to Interact:
        - **Update Stock Prices**: Click the "Update Stock Prices" button in the Assets tab to get the latest stock and crypto data.
        - **Ask for Investment Advice**: Enter your query in the Agentic Advisor tab to get personalized investment advice.
        - **Forecasting**: Use the forecasting tool to predict stock prices for the next days based on historical data.
        """)
    # 2) ASSETS TAB
    with tab2:
        st.header("Stock & Crypto Data")
    
        # Fetch stock data
        if st.session_state.get("asset_data"):
            df = pd.DataFrame(st.session_state["asset_data"])
            st.dataframe(df)
            st.write(f"Data last updated: {st.session_state['asset_data_timestamp']}")
    
            # Asset Graph Feature
            tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
            selected_ticker = st.selectbox("Select an asset to view its price chart:", tickers)
            asset_type = st.radio("Select Asset Type:", options=["Stock", "Crypto"], index=0)
            tech_mode = st.checkbox("Technical Analysis Mode")
    
            if selected_ticker:
                # Use forecasting.get_asset_history to fetch data from Alpha Vantage
                hist = forecasting.get_asset_history(selected_ticker, asset_type=asset_type.lower(), period="1y")
                if hist.empty:
                    st.write("No historical data available for this asset.")
                else:
                    if tech_mode:
                        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
                        st.line_chart(hist[["Close", "SMA50"]])
                    else:
                        st.line_chart(hist["Close"])
        else:
            st.write("No stock data available.")
    
        # Display crypto data below stocks if available (optional)
        st.subheader("Cryptocurrency Data")
        crypto_data = crypto.fetch_crypto_data()
        if crypto_data:
            df_crypto = pd.DataFrame(crypto_data)
            st.dataframe(df_crypto)
        else:
            st.write("No crypto data available.")



    # 3) BUDGETING TAB
    with tab3:
        budgeting.budgeting_tool()

    # 4) AGENTIC ADVISOR TAB
    with tab4:
        st.header("Agentic Advisor")
        user_query = st.text_input("Enter your investment query:")
        deep_research = st.checkbox("Deep Research Mode")
        macro_mode = st.checkbox("Macroeconomics Mode")

        if st.button("Get Advice"):
            if macro_mode:
                macro_data = analysis.fetch_macro_indicators()
                macro_report = analysis.generate_macro_report(macro_data)
                st.subheader("Macroeconomic Report:")
                st.write(macro_report)

            advice = agentic.agentic_advisor(
                user_query,
                deep_mode=deep_research,
                tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY")
            )
            st.subheader("Investment Advice:")
            st.write(advice)

    
        # 5) FORECASTING AND NEWS TAB
    with tab5:
        st.header("Forecasting and News")
    
        # Display News
        st.subheader("Latest Finance News")
        articles = news.fetch_finance_news()
        if articles:
            for art in articles:
                st.markdown(f"[**{art['title']}**]({art['url']}) - {art['source']}")
        else:
            st.write("No news available.")
    
        # Forecasting Section
        st.header("Forecasting")
        # Create combined options for stocks and crypto
        available_stock_tickers = [asset["Ticker"] for asset in st.session_state["asset_data"] if "Ticker" in asset]
        crypto_df = pd.DataFrame(crypto.fetch_crypto_data()) if crypto.fetch_crypto_data() else pd.DataFrame()
        available_crypto_tickers = crypto_df["Symbol"].tolist() if "Symbol" in crypto_df.columns else []
    
        combined_options = []
        for t in available_stock_tickers:
            combined_options.append(f"{t}|stock")
        for t in available_crypto_tickers:
            combined_options.append(f"{t}|crypto")
    
        selected = st.multiselect("Select asset(s) for forecasting (one or two)", options=combined_options, default=combined_options[:1])
        forecast_method = st.radio("Select Forecasting Method", options=["ARIMA", "Prophet"])
        forecast_days = st.number_input("Forecast Period (days)", min_value=7, max_value=365, value=30, step=7)
        generate_ai_comp = st.checkbox("Generate AI Comparison (if two assets selected)")
    
        if st.button("Run Forecast"):
            if not selected:
                st.error("Please select at least one asset for forecasting.")
            else:
                forecast_results = {}
                for sel in selected:
                    ticker, asset_type = sel.split("|")
                    if forecast_method == "ARIMA":
                        hist, forecast_series = forecasting.forecast_arima(ticker, forecast_days=forecast_days, asset_type=asset_type)
                    else:
                        hist, forecast_series = forecasting.forecast_prophet(ticker, forecast_days=forecast_days, asset_type=asset_type)
    
                    if hist is None or forecast_series is None:
                        st.error(f"Forecasting failed for {ticker}.")
                    else:
                        fig = forecasting.plot_forecast(hist, forecast_series, title=f"{ticker} Forecast ({forecast_method})")
                        st.pyplot(fig)
                    forecast_results[ticker] = {
                        "historical": hist,
                        "forecast": forecast_series,
                        "asset_type": asset_type
                    }
    
                # If two assets are selected and AI comparison is requested, generate a brief summary
                if len(selected) == 2 and generate_ai_comp:
                    t1, a1 = selected[0].split("|")
                    t2, a2 = selected[1].split("|")
                    prompt = (
                        f"Compare the forecasted prices for {t1} ({a1}) and {t2} ({a2}) using {forecast_method} over the next {forecast_days} days. "
                        f"For {t1}, the forecast is: {forecast_results[t1]['forecast'].round(2).to_dict()}. "
                        f"For {t2}, the forecast is: {forecast_results[t2]['forecast'].round(2).to_dict()}. "
                        "Provide a brief comparison on the trends and any potential risks."
                    )
                    summary = agentic.call_openai_llm(prompt, system="You are a financial analyst specializing in forecasting comparisons.")
                    st.subheader("Forecast Comparison Summary:")
                    st.write(summary)


# ---------------------- LAUNCH ---------------------- #
if __name__ == "__main__":
    main()
