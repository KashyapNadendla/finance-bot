import streamlit as st
import analysis
import crypto
import news
import stocks
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import requests 

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# def call_openai_llm(prompt, system="", model="gpt-4o", api_key=None):
#     """
#     Helper function to call the OpenAI LLM with a given prompt.
#     """
#     if not api_key:
#         # Use the key from session state
#         api_key = st.session_state.get("OPENAI_API_KEY", None)
#     if not api_key:
#         st.error("OpenAI API key not set.")
#         return "No recommendations because API key is missing."
#     messages = []
#     if system:
#         messages.append({"role": "system", "content": system})
#     messages.append({"role": "user", "content": prompt})
#     try:
#         completion = client.chat.completions.create(
#             model=model,
#             messages=messages,
#         )
#         response = completion.choices[0].message.content
#         return response
#     except Exception as e:
#         st.error(f"OpenAI API error: {e}")
#         return "Error fetching recommendations."
def call_openai_llm(prompt, system="", model="gpt-4o", api_key=None):
    """
    Helper function to call the OpenAI LLM with a given prompt.
    """
    if not api_key:
        # Use the key from session state
        api_key = st.session_state.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("OpenAI API key not set.")
        return "No recommendations because API key is missing."
    
    # Create the OpenAI client instance
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
        return response
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "Error fetching recommendations."


def get_commodities_data():
    """
    Fetch live commodities data for Gold, Crude Oil, and Silver.
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
    Retrieve macroeconomic conditions.
    """
    import yfinance as yf
    conditions = {}
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        conditions["10Y Treasury Yield"] = hist['Close'].iloc[-1] if not hist.empty else "N/A"
    except:
        conditions["10Y Treasury Yield"] = "N/A"
    try:
        dxy = yf.Ticker("^DXY")
        hist = dxy.history(period="5d")
        if not hist.empty:
            prices = hist['Close']
            conditions["DXY Trend"] = "decreasing" if prices.iloc[-1] < prices.iloc[0] else "increasing"
            conditions["DXY Latest"] = prices.iloc[-1]
        else:
            conditions["DXY Trend"] = "N/A"
    except:
        conditions["DXY Trend"] = "N/A"
    yld = conditions.get("10Y Treasury Yield", "N/A")
    if isinstance(yld, (int, float)) and yld < 3.0:
        conditions["Interest Rates"] = "Low"
    else:
        conditions["Interest Rates"] = "High or N/A"
    articles = news.fetch_finance_news()
    war_flag = any("war" in article['title'].lower() or "conflict" in article['title'].lower() for article in articles)
    conditions["War Status"] = "No ongoing wars" if not war_flag else "Potential conflict detected"
    conditions_str = (
        f"10Y Treasury Yield: {conditions.get('10Y Treasury Yield', 'N/A')}, "
        f"DXY Trend: {conditions.get('DXY Trend', 'N/A')}, "
        f"Interest Rates: {conditions.get('Interest Rates', 'N/A')}, "
        f"War Status: {conditions.get('War Status', 'N/A')}."
    )
    return conditions_str

def agentic_advisor(user_input, **kwargs):
    """
    Multi-agent advisor chain:
      1. Parse user input.
      2. Fetch relevant data.
      3. Analyze data.
      4. Generate recommendations.
      5. Validate and possibly refine recommendations.
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
    
    # Step 2: Gather market data
    live_news = news.fetch_finance_news()
    asset_data = st.session_state.get('asset_data', [])
    crypto_data = crypto.get_top_movers()  # assuming function signature without CMC_API_KEY for simplicity
    commodities_data = get_commodities_data()
    live_data_str = (
        f"News: {live_news}\n"
        f"Stocks: {stocks.format_asset_suggestions(asset_data)}\n"
        f"Cryptocurrencies: {crypto_data}\n"
        f"Commodities: {commodities_data}\n"
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
    
    # Step 4: Evaluate recommendations against macro and technical conditions
    macro_conditions = get_macro_conditions()
    tech_analysis = analysis.get_technical_analysis_summaries(asset_data)
    evaluation_prompt = f"""
    Evaluate the following asset suggestions against current macroeconomic conditions and technical analysis:
    
    Asset Suggestions:
    {recommendations}
    
    Macroeconomic Conditions:
    {macro_conditions}
    
    Technical Analysis:
    {tech_analysis}
    
    If conditions are favorable, confirm the recommendations; otherwise, adjust to be more conservative.
    """

    validated_recommendations = "Your recommendations here (placeholder)"
    
    # Optionally refine recommendations up to 3 times if conservative adjustments are needed.
    iterations = 0
    while iterations < 3:
        if "conservative" not in validated_recommendations.lower():
            break
        else:
            re_adjust_prompt = f"""
            The current conditions indicate a need for more conservative asset recommendations.
            Adjust the previous suggestions accordingly.
            
            Previous Suggestions:
            {recommendations}
            
            Macroeconomic Conditions:
            {macro_conditions}
            
            Technical Analysis:
            {tech_analysis}
            """
            recommendations = call_openai_llm(
                re_adjust_prompt,
                system="You are a financial advisor specializing in asset recommendations."
            )
            evaluation_prompt = f"""
            Evaluate the following adjusted asset suggestions:
            
            Adjusted Suggestions:
            {recommendations}
            
            Macroeconomic Conditions:
            {macro_conditions}
            
            Technical Analysis:
            {tech_analysis}
            """
            validated_recommendations = call_openai_llm(
                evaluation_prompt,
                system="You are a senior financial strategist specializing in risk management."
            )
            iterations += 1

    return validated_recommendations
