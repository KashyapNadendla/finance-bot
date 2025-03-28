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
import websearch

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
        # Optional debug: uncomment the next line to print each LLM response
        # st.write("LLM response:", response)
        return response
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "Error fetching recommendations."


def get_commodities_data():
    """
    Fetch live commodities data (Gold, Crude Oil, Silver) via Alpha Vantage.
    Alpha Vantage does not provide a direct commodity price for Crude Oil,
    so we return "N/A" for that. We pull XAU -> USD and XAG -> USD as 
    approximate 'Gold' and 'Silver' prices.
    """
    api_key = st.session_state.get("ALPHA_VANTAGE_API_KEY", None)
    if not api_key:
        st.error("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY.")
        return {
            "Gold": "N/A",
            "Crude Oil": "N/A",
            "Silver": "N/A"
        }

    base_url = "https://www.alphavantage.co/query"
    
    def fetch_spot_price(from_currency, to_currency):
        """
        Fetches the real-time currency exchange rate from Alpha Vantage
        for the given currency pair (e.g., XAU -> USD for Gold).
        """
        try:
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": api_key
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            rate_info = data.get("Realtime Currency Exchange Rate", {})
            # '5. Exchange Rate' is the key for the actual float exchange rate
            return f"${float(rate_info.get('5. Exchange Rate', '0.0')):,.2f}"
        except Exception as e:
            st.error(f"Error fetching {from_currency}->{to_currency} price: {e}")
            return "N/A"

    # Gold is typically XAU (troy ounce) to USD
    gold_price = fetch_spot_price("XAU", "USD")

    # Silver is XAG to USD
    silver_price = fetch_spot_price("XAG", "USD")

    # Crude Oil not provided by Alpha Vantage, so fallback to "N/A" or use another data source
    crude_oil_price = "N/A"

    return {
        "Gold": gold_price,
        "Crude Oil": crude_oil_price,
        "Silver": silver_price
    }

def get_macro_conditions():
    """
    Retrieve macroeconomic indicators from Alpha Vantage.
    """
    api_key = ALPHA_VANTAGE_API_KEY
    indicators = {}
    endpoints = {
        "Treasury Yield": f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={api_key}",
        "Federal Funds Rate": f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={api_key}",
        "CPI": f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={api_key}",
        "Inflation": f"https://www.alphavantage.co/query?function=INFLATION&apikey={api_key}"
    }
    
    for indicator, url in endpoints.items():
        try:
            data = requests.get(url).json()
            if "data" in data and data["data"]:
                last = data["data"][-1]
                indicators[indicator] = last.get("value", "N/A")
            else:
                indicators[indicator] = "N/A"
        except Exception:
            indicators[indicator] = "N/A"
    
    conditions_str = (
        f"10Y Treasury Yield: {indicators.get('Treasury Yield', 'N/A')}, "
        f"Federal Funds Rate: {indicators.get('Federal Funds Rate', 'N/A')}, "
        f"CPI: {indicators.get('CPI', 'N/A')}, "
        f"Inflation: {indicators.get('Inflation', 'N/A')}."
    )
    return conditions_str

# def agentic_advisor(user_input, deep_mode=False, **kwargs):
#     """
#     Multi-agent advisor chain:
#       1. Parse user input.
#       2. Gather market data.
#       3. Generate recommendations.
      
#     If deep_mode is True, the prompt instructs the LLM to provide a detailed report;
#     otherwise, a concise asset suggestion.
#     """
#     # Step 1: Parse user input
#     base_prompt = f"Analyze the following user input for investment advice, risk appetite, and expected returns: '{user_input}'."
    
#     if deep_mode:
#         # Deep research mode prompt
#         prompt = base_prompt + " Provide a detailed analysis including macroeconomic context, technical indicators, and suggested investment amounts for each recommended asset."
#     else:
#         # Concise mode
#         prompt = base_prompt + " Provide a short, actionable recommendation on which assets to invest in and approximate allocations, suited for a beginner investor."
    
#     user_intent = call_openai_llm(prompt, system="You are a financial advisor specializing in market analysis.")
    
#     st.write("Parsed user intent:", user_intent)
    
#     # Step 2: Gather market data
#     live_news = news.fetch_finance_news()
#     asset_data = st.session_state.get('asset_data', [])
#     stock_suggestions = stocks.format_asset_suggestions(asset_data) if hasattr(stocks, "format_asset_suggestions") else str(asset_data)
#     crypto_data = crypto.get_top_movers()
#     commodities_data = get_commodities_data()
#     macro_data = get_macro_conditions()
#     live_data_str = (
#         f"News: {live_news}\n"
#         f"Stocks: {stock_suggestions}\n"
#         f"Cryptocurrencies: {crypto_data}\n"
#         f"Commodities: {commodities_data}\n"
#         f"Macro Indicators: {macro_data}\n"
#         f"Google Trends: N/A"
#     )
    
#     # Step 3: Generate initial recommendations
#     suggestion_prompt = f"""
#     Based on the following user analysis and live market data, suggest a few assets for investment.
    
#     User Analysis:
#     {user_intent}
    
#     Live Market Data:
#     {live_data_str}
    
#     Consider the user's risk appetite and expected returns. Provide asset names, expected percentage gains, and risk levels.
#     """
#     recommendations = call_openai_llm(
#         suggestion_prompt,
#         system="You are a financial advisor specializing in asset recommendations."
#     )
#     # st.write("Initial asset recommendations:", recommendations)
    
#     # Step 4: Evaluate recommendations against macro and technical conditions
#     tech_analysis = analysis.get_technical_analysis_summaries(asset_data)
#     evaluation_prompt = f"""
#     Evaluate the following asset suggestions against current macroeconomic conditions and technical analysis:
    
#     Asset Suggestions:
#     {recommendations}
    
#     Macroeconomic Conditions:
#     {macro_data}
    
#     Technical Analysis:
#     {tech_analysis}
    
#     If conditions are favorable, confirm the recommendations; otherwise, adjust them to be more conservative.
#     """
#     validated_recommendations = call_openai_llm(
#         evaluation_prompt,
#         system="You are a senior financial strategist specializing in risk management."
#     )
    
#     iterations = 0
#     # We'll refine recommendations only if "conservative" is present.
#     while iterations < 3:
#         if "conservative" not in validated_recommendations.lower():
#             break
#         re_adjust_prompt = f"""
#         The current market conditions indicate a need for more conservative asset recommendations.
#         Adjust the previous suggestions accordingly.
        
#         Previous Suggestions:
#         {recommendations}
        
#         Macroeconomic Conditions:
#         {macro_data}
        
#         Technical Analysis:
#         {tech_analysis}
#         """
#         new_recommendations = call_openai_llm(
#             re_adjust_prompt,
#             system="You are a financial advisor specializing in asset recommendations."
#         )
#         # If the new recommendations don't change, exit the loop.
#         if new_recommendations.strip() == recommendations.strip():
#             break
#         recommendations = new_recommendations
#         evaluation_prompt = f"""
#         Evaluate the following adjusted asset suggestions:
        
#         Adjusted Suggestions:
#         {recommendations}
        
#         Macroeconomic Conditions:
#         {macro_data}
        
#         Technical Analysis:
#         {tech_analysis}
#         """
#         validated_recommendations = call_openai_llm(
#             evaluation_prompt,
#             system="You are a senior financial strategist specializing in risk management."
#         )
#         iterations += 1

#     # st.write("Final validated recommendations:", validated_recommendations)
#     return validated_recommendations

# def agentic_chat_interface():
#     st.header("Agentic Advisor Chat")
#     user_input = st.text_input("Enter your investment query:")
    
#     if st.button("Submit Agentic Query"):
#         if user_input:
#             with st.spinner("Processing your query through multiple agents..."):
#                 response = agentic_advisor(
#                     user_input, 
#                     tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
#                     alpha_vantage_api_key=ALPHA_VANTAGE_API_KEY
#                 )
#             st.session_state.setdefault('agentic_history', []).append({"user": user_input, "advisor": response})
    
#     for entry in st.session_state.get('agentic_history', []):
#         st.markdown(f"**You:** {entry['user']}")
#         st.markdown(f"**Agentic Advisor:** {entry['advisor']}")
def agentic_advisor(user_input, deep_mode=False, **kwargs):
    """
    Multi-agent advisor chain:
      1. Parse user input.
      2. Gather market data.
      3. Generate recommendations.
      4. Evaluate and refine using macro + technical data.
      5. Use web search if response lacks data or context.
    """
    # Step 1: Parse user input
    base_prompt = f"Analyze the following user input for investment advice, risk appetite, and expected returns: '{user_input}'."
    if deep_mode:
        prompt = base_prompt + " Provide a detailed analysis including macroeconomic context, technical indicators, and suggested investment amounts for each recommended asset."
    else:
        prompt = base_prompt + " Provide a short, actionable recommendation on which assets to invest in and approximate allocations, suited for a beginner investor."
    
    user_intent = call_openai_llm(prompt, system="You are a financial advisor specializing in market analysis.")
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

    # Step 4: Fallback to web search if the model says it lacks data
    if "insufficient data" in recommendations.lower() or not recommendations.strip():
        st.warning("LLM response insufficient — fetching additional context via web search.")
        search_context = websearch.run_web_search(user_input)
        fallback_prompt = suggestion_prompt + "\nAdditional Web Search Context:\n" + str(search_context)
        recommendations = call_openai_llm(
            fallback_prompt,
            system="You are a financial advisor specializing in asset recommendations."
        )

    # Step 5: Evaluate recommendations using macro and technical data
    tech_analysis = analysis.get_technical_analysis_summaries(asset_data)
    evaluation_prompt = f"""
    Evaluate the following asset suggestions against current macroeconomic conditions and technical analysis:

    Asset Suggestions:
    {recommendations}

    Macroeconomic Conditions:
    {macro_data}

    Technical Analysis:
    {tech_analysis}

    If conditions are unfavorable, adjust recommendations to be more conservative.
    """
    validated_recommendations = call_openai_llm(
        evaluation_prompt,
        system="You are a senior financial strategist specializing in risk management."
    )

    # Optional iterative refinement
    iterations = 0
    while iterations < 3:
        if "conservative" not in validated_recommendations.lower():
            break
        re_adjust_prompt = f"""
        Adjust the previous recommendations to be more conservative based on market conditions.

        Previous Suggestions:
        {recommendations}

        Macroeconomic Conditions:
        {macro_data}

        Technical Analysis:
        {tech_analysis}
        """
        new_recommendations = call_openai_llm(
            re_adjust_prompt,
            system="You are a financial advisor specializing in cautious investing strategies."
        )

        if new_recommendations.strip() == recommendations.strip():
            break
        recommendations = new_recommendations

        validated_recommendations = call_openai_llm(
            f"""
            Re-evaluate these adjusted recommendations:

            Adjusted Suggestions:
            {recommendations}

            Macro Indicators:
            {macro_data}

            Technical Indicators:
            {tech_analysis}
            """,
            system="You are a senior strategist reviewing adjusted investment plans."
        )

        iterations += 1

    return validated_recommendations
