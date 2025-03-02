import streamlit as st
import analysis
import crypto
import news
import stocks
from datetime import datetime
from openai import OpenAI

def call_openai_llm(prompt, system="", model="gpt-4o", api_key=None):
    """
    Helper function to call the OpenAI LLM with a given prompt and system instruction.
    """
    if not api_key:
        api_key = st.session_state.get("OPENAI_API_KEY", None)

    client = OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    response = completion.choices[0].message.content
    return response

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
    Retrieve macroeconomic conditions including US 10Y Treasury yield, DXY trend, and war status.
    """
    import yfinance as yf
    conditions = {}

    # 10Y Treasury
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            ten_year_yield = hist['Close'].iloc[-1]
            conditions["10Y Treasury Yield"] = ten_year_yield
        else:
            conditions["10Y Treasury Yield"] = "N/A"
    except:
        conditions["10Y Treasury Yield"] = "N/A"

    # DXY
    try:
        dxy = yf.Ticker("^DXY")
        hist = dxy.history(period="5d")
        if not hist.empty:
            prices = hist['Close']
            trend = "decreasing" if prices.iloc[-1] < prices.iloc[0] else "increasing"
            conditions["DXY Trend"] = trend
            conditions["DXY Latest"] = prices.iloc[-1]
        else:
            conditions["DXY Trend"] = "N/A"
    except:
        conditions["DXY Trend"] = "N/A"

    # Interest Rates
    yld = conditions.get("10Y Treasury Yield", "N/A")
    if isinstance(yld, (int, float)) and yld < 3.0:
        conditions["Interest Rates"] = "Low"
    else:
        conditions["Interest Rates"] = "High or N/A"

    # War status
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

def agentic_advisor(user_input):
    """
    Multi-agent advisor chain:
      1. Analyze user input
      2. Generate asset suggestions
      3. Re-evaluate suggestions vs macro conditions, technical analysis, etc.
      4. Possibly iterate for more conservative suggestions if needed
    """

    # Step 1: Analyze user intent
    analysis_prompt = (
        f"Analyze the following user input for investment advice, risk appetite, and expected returns: "
        f"'{user_input}'. Summarize the key factors and preferences."
    )
    llm1_response = call_openai_llm(
        analysis_prompt,
        system="You are an analyst specializing in extracting investment preferences."
    )

    # Step 2: Gather market data
    live_news = news.fetch_finance_news()
    asset_data = st.session_state['asset_data']  # assume we have it loaded
    crypto_data = crypto.get_top_movers(st.session_state.get("CMC_API_KEY", None))
    commodities_data = get_commodities_data()
    trends_data = "N/A (function omitted or replaced)"  # or use google trends if needed

    live_data_str = (
        f"News: {live_news}\n"
        f"Stocks: {stocks.format_asset_suggestions(asset_data)}\n"
        f"Cryptocurrencies: {crypto_data}\n"
        f"Commodities: {commodities_data}\n"
        f"Google Trends: {trends_data}"
    )

    suggestion_prompt = f"""
    Based on the following user analysis and live market data, suggest a few assets for investment.
    
    User Analysis:
    {llm1_response}
    
    Live Market Data:
    {live_data_str}
    
    Consider the user's risk appetite and expected returns. Provide asset names, expected percentage gains, and risk levels.
    """
    llm2_response = call_openai_llm(
        suggestion_prompt,
        system="You are a financial advisor specializing in asset recommendations."
    )

    # Step 3: Evaluate suggestions vs macro conditions, technical analysis
    macro_conditions = get_macro_conditions()
    tech_analysis = analysis.get_technical_analysis_summaries(asset_data)

    evaluation_prompt = f"""
    Evaluate the following asset suggestions against current macroeconomic conditions, technical analysis, and any available trends data:
    
    Asset Suggestions:
    {llm2_response}
    
    Macroeconomic Conditions:
    {macro_conditions}
    
    Technical Analysis:
    {tech_analysis}
    
    If conditions are favorable for risk assets (e.g., low US interest rates, low 10Y yield, decreasing DXY, no ongoing wars, positive tech indicators), confirm the recommendations.
    Otherwise, adjust the suggestions to be more conservative.
    """
    llm3_response = call_openai_llm(
        evaluation_prompt,
        system="You are a senior financial strategist specialized in macroeconomic analysis, technical analysis, and risk management."
    )

    # up to 3 iterations if we see the word 'conservative'
    iterations = 0
    while iterations < 3:
        if "conservative" not in llm3_response.lower():
            break
        else:
            re_adjust_prompt = f"""
            The current macroeconomic conditions, technical analysis, etc. indicate a need for more conservative asset recommendations.
            Adjust the previous suggestions accordingly.
            
            Previous Asset Suggestions:
            {llm2_response}
            
            Macroeconomic Conditions:
            {macro_conditions}
            
            Technical Analysis:
            {tech_analysis}
            """
            llm2_response = call_openai_llm(
                re_adjust_prompt,
                system="You are a financial advisor specializing in asset recommendations."
            )

            evaluation_prompt = f"""
            Evaluate the following adjusted asset suggestions against current macroeconomic conditions, technical analysis, etc:
            
            Adjusted Asset Suggestions:
            {llm2_response}
            
            Macroeconomic Conditions:
            {macro_conditions}
            
            Technical Analysis:
            {tech_analysis}
            """
            llm3_response = call_openai_llm(
                evaluation_prompt,
                system="You are a senior financial strategist specialized in macroeconomic analysis, technical analysis, and risk management."
            )
            iterations += 1

    return llm3_response

def agentic_chat_interface():
    st.header("Agentic Advisor Chat")
    user_input = st.text_input("Enter your investment query:")
    if st.button("Submit Agentic Query"):
        if user_input:
            with st.spinner("Processing your query through multiple agents..."):
                response = agentic_advisor(user_input)
            st.session_state['agentic_history'].append({"user": user_input, "advisor": response})

    for entry in st.session_state['agentic_history']:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Agentic Advisor:** {entry['advisor']}")
