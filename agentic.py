import streamlit as st
import analysis
import crypto
import news
import stocks
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def call_openai_llm(prompt, system="", model="gpt-4o", api_key=None):
    """
    Helper function to call the OpenAI LLM with a given prompt.
    """
    if not api_key:
        api_key = st.session_state.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("OpenAI API key not set.")
        return "No recommendations because API key is missing."
    
    client = OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        response = completion.choices[0].message.content
        # (Optional debug: uncomment the next line to print each LLM response)
        # st.write("LLM response:", response)
        return response
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "Error fetching recommendations."

def get_commodities_data():
    """
    Fetch live commodities data for Gold, Crude Oil, and Silver.
    (This function still uses yfinance; you can later replace it with an Alpha Vantage endpoint if available.)
    """
    import yfinance as yf
    commodities = {}
    symbols = {"Gold": "GC=F", "Crude Oil": "CL=F", "Silver": "SI=F"}
    for name, symbol in symbols.items():
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            commodities[name] = f"${price:.2f}"
        else:
            commodities[name] = "N/A"
    return commodities

def get_macro_conditions():
    """
    Retrieve macroeconomic indicators from Alpha Vantage for:
      - 10Y Treasury Yield (using the TREASURY_YIELD endpoint)
      - Federal Funds Rate (using FEDERAL_FUNDS_RATE)
      - CPI (using CPI)
      - Inflation (using INFLATION)
    
    If any data is missing, it returns "N/A" for that indicator.
    """
    api_key = ALPHA_VANTAGE_API_KEY
    indicators = {}
    try:
        # 10Y Treasury Yield (monthly)
        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={api_key}"
        data = requests.get(url).json()
        if "data" in data and data["data"]:
            last = data["data"][-1]
            indicators["Treasury Yield"] = last.get("value", "N/A")
        else:
            indicators["Treasury Yield"] = "N/A"
    except Exception as e:
        indicators["Treasury Yield"] = "N/A"
    
    try:
        # Federal Funds Rate
        url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={api_key}"
        data = requests.get(url).json()
        if "data" in data and data["data"]:
            last = data["data"][-1]
            indicators["Federal Funds Rate"] = last.get("value", "N/A")
        else:
            indicators["Federal Funds Rate"] = "N/A"
    except Exception as e:
        indicators["Federal Funds Rate"] = "N/A"
    
    try:
        # CPI (monthly)
        url = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={api_key}"
        data = requests.get(url).json()
        if "data" in data and data["data"]:
            last = data["data"][-1]
            indicators["CPI"] = last.get("value", "N/A")
        else:
            indicators["CPI"] = "N/A"
    except Exception as e:
        indicators["CPI"] = "N/A"
    
    try:
        # Inflation
        url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={api_key}"
        data = requests.get(url).json()
        if "data" in data and data["data"]:
            last = data["data"][-1]
            indicators["Inflation"] = last.get("value", "N/A")
        else:
            indicators["Inflation"] = "N/A"
    except Exception as e:
        indicators["Inflation"] = "N/A"
    
    conditions_str = (
        f"10Y Treasury Yield: {indicators.get('Treasury Yield', 'N/A')}, "
        f"Federal Funds Rate: {indicators.get('Federal Funds Rate', 'N/A')}, "
        f"CPI: {indicators.get('CPI', 'N/A')}, "
        f"Inflation: {indicators.get('Inflation', 'N/A')}."
    )
    return conditions_str

def agentic_advisor(user_input, **kwargs):
    """
    Multi-agent advisor chain:
      1. Parse user input.
      2. Gather live market data.
      3. Generate asset recommendations.
      4. Evaluate and refine recommendations.
    """
    # Step 1: Parse user input
    analysis_prompt = (
        f"Analyze the following user input for investment advice, risk appetite, and expected returns: "
        f"'{user_input}'. Summarize the key factors and preferences."
    )
    user_intent = call_openai_llm(
        analysis_prompt,
        system="You are an analyst specializing in extracting investment preferences."
    )
    st.write("Parsed user intent:", user_intent)
    
    # Step 2: Gather market data
    live_news = news.fetch_finance_news()
    asset_data = st.session_state.get('asset_data', [])
    stock_suggestions = stocks.format_asset_suggestions(asset_data) if hasattr(stocks, "format_asset_suggestions") else str(asset_data)
    crypto_data = crypto.get_top_movers()
    commodities_data = get_commodities_data()
    macro_data = get_macro_conditions()
    live_data_str = (
        f"News: {live_news}\n"
        f"Stocks: {stock_suggestions}\n"
        f"Cryptocurrencies: {crypto_data}\n"
        f"Commodities: {commodities_data}\n"
        f"Macro Indicators: {macro_data}\n"
        f"Google Trends: N/A"
    )
    
    # Step 3: Generate initial recommendations
    suggestion_prompt = f"""
    Based on the following user analysis and live market data, suggest a few assets for investment.
    
    User Analysis:
    {user_intent}
    
    Live Market Data:
    {live_data_str}
    
    Consider the user's risk appetite and expected returns. Provide asset names, expected percentage gains, and risk levels.
    """
    recommendations = call_openai_llm(
        suggestion_prompt,
        system="You are a financial advisor specializing in asset recommendations."
    )
    st.write("Initial asset recommendations:", recommendations)
    
    # Step 4: Evaluate recommendations against macro and technical conditions
    tech_analysis = analysis.get_technical_analysis_summaries(asset_data)
    evaluation_prompt = f"""
    Evaluate the following asset suggestions against current macroeconomic conditions and technical analysis:
    
    Asset Suggestions:
    {recommendations}
    
    Macroeconomic Conditions:
    {macro_data}
    
    Technical Analysis:
    {tech_analysis}
    
    If conditions are favorable, confirm the recommendations; otherwise, adjust them to be more conservative.
    """
    validated_recommendations = call_openai_llm(
        evaluation_prompt,
        system="You are a senior financial strategist specializing in risk management."
    )
    
    iterations = 0
    # We'll refine recommendations only if "conservative" is present.
    while iterations < 3:
        if "conservative" not in validated_recommendations.lower():
            break
        re_adjust_prompt = f"""
        The current market conditions indicate a need for more conservative asset recommendations.
        Adjust the previous suggestions accordingly.
        
        Previous Suggestions:
        {recommendations}
        
        Macroeconomic Conditions:
        {macro_data}
        
        Technical Analysis:
        {tech_analysis}
        """
        new_recommendations = call_openai_llm(
            re_adjust_prompt,
            system="You are a financial advisor specializing in asset recommendations."
        )
        # If the new recommendations don't change, exit the loop.
        if new_recommendations.strip() == recommendations.strip():
            break
        recommendations = new_recommendations
        evaluation_prompt = f"""
        Evaluate the following adjusted asset suggestions:
        
        Adjusted Suggestions:
        {recommendations}
        
        Macroeconomic Conditions:
        {macro_data}
        
        Technical Analysis:
        {tech_analysis}
        """
        validated_recommendations = call_openai_llm(
            evaluation_prompt,
            system="You are a senior financial strategist specializing in risk management."
        )
        iterations += 1

    st.write("Final validated recommendations:", validated_recommendations)
    return validated_recommendations

def agentic_chat_interface():
    st.header("Agentic Advisor Chat")
    user_input = st.text_input("Enter your investment query:")
    if st.button("Submit Agentic Query"):
        if user_input:
            with st.spinner("Processing your query through multiple agents..."):
                response = agentic_advisor(user_input, tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                                           alpha_vantage_api_key=ALPHA_VANTAGE_API_KEY)
            st.session_state.setdefault('agentic_history', []).append({"user": user_input, "advisor": response})
    for entry in st.session_state.get('agentic_history', []):
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Agentic Advisor:** {entry['advisor']}")
