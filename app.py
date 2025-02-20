import streamlit as st
import os
import time
import json
import re
import requests
import pandas as pd
import PyPDF2
import yfinance as yf
import ta  # Technical Analysis library (install via pip install ta)
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Langchain and Search/Trends Tools ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from openai import OpenAI  # Using OpenAI's client

# Built-in AlphaVantage tool (if available in your Langchain version)
try:
    from langchain.tools.alpha_vantage import AlphaVantageTool
    alpha_vantage_tool = AlphaVantageTool(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
except ImportError:
    alpha_vantage_tool = None

# Use DuckDuckGo search utility from Langchain for free search
try:
    from langchain.utilities import DuckDuckGoSearchResults
    duckduckgo = DuckDuckGoSearchResults()
except ImportError:
    duckduckgo = None

# Google Trends via pytrends
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)

# ---------------------- INITIAL SETUP ---------------------- #

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CMC_API_KEY = os.getenv("CMC_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

client = OpenAI(api_key=API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# CSV file for caching stock data
STOCK_CSV = "stock_data.csv"
STOCK_DATA_TTL_SECONDS = 600  # 10 minutes

# Initialize session state variables
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = ''
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'asset_data' not in st.session_state:
    st.session_state['asset_data'] = []
    st.session_state['asset_data_timestamp'] = None
if 'agentic_history' not in st.session_state:
    st.session_state['agentic_history'] = []

# ---------------------- DOCUMENT PROCESSING ---------------------- #

def load_and_process_pdfs(data_folder):
    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "".join([page.extract_text() or "" for page in reader.pages])
                    pdf_texts.append(text)
                st.write(f"Processed {filename}")
            except PyPDF2.errors.PdfReadError:
                st.error(f"Warning: '{filename}' could not be processed. Skipping.")
            except Exception as e:
                st.error(f"Warning: Error processing '{filename}': {e}. Skipping.")
    return pdf_texts

def create_vector_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]
    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    vector_store = FAISS.from_texts(texts_chunks, embeddings)
    return vector_store

# ---------------------- NEWS ---------------------- #

@st.cache_data(ttl=86400)
def fetch_finance_news():
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        last_week = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        news = newsapi.get_everything(
            q="finance OR economy",
            from_param=last_week,
            to=today,
            language="en",
            sort_by="relevancy",
            page_size=3
        )
        articles = news.get('articles', [])
        return [{"title": article['title'], "url": article['url'], "source": article['source']['name']} for article in articles]
    except NewsAPIException as e:
        if 'rateLimited' in str(e):
            st.warning("News API rate limit exceeded. Please try again later.")
        else:
            st.error("An error occurred while fetching news.")
        return []

def display_finance_news():
    st.subheader("Top 3 Finance News Articles Today")
    articles = fetch_finance_news()
    if articles:
        for i, article in enumerate(articles, 1):
            st.markdown(f"[**{i}. {article['title']}**]({article['url']})")
            st.write(f"Source: {article['source']}")
    else:
        st.write("No news articles available at this time.")

# ---------------------- STOCK DATA ---------------------- #

def fetch_stock_data(ticker):
    base_url = "https://www.alphavantage.co/query"
    
    # Parameters for stock price and company overview
    price_params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "5min",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    overview_params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    try:
        price_response = requests.get(base_url, params=price_params)
        price_data = price_response.json()
        time_series_key = "Time Series (5min)"
        
        if time_series_key not in price_data:
            return None

        latest_time = max(price_data[time_series_key].keys())
        latest_data = price_data[time_series_key][latest_time]
        open_price = float(latest_data["1. open"])
        close_price = float(latest_data["4. close"])
        price_change_today = ((close_price - open_price) / open_price) * 100

        overview_response = requests.get(base_url, params=overview_params)
        overview_data = overview_response.json()
        dividend_yield = overview_data.get("DividendYield", "N/A")
        
        if dividend_yield != "N/A":
            dividend_yield = f"{float(dividend_yield) * 100:.2f}%"

        return {
            "Ticker": ticker,
            "Current Price": f"${close_price:.2f}",
            "Price Change (Today)": f"{price_change_today:.2f}%",
            "Dividend Yield": dividend_yield
        }
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None

def scout_assets():
    tickers = [
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO"
    ]
    asset_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                asset_data.append(result)
    return asset_data

def save_asset_data_to_csv(asset_data, filename=STOCK_CSV):
    df = pd.DataFrame(asset_data)
    df.to_csv(filename, index=False)

def load_asset_data_from_csv(filename=STOCK_CSV, ttl=STOCK_DATA_TTL_SECONDS):
    if os.path.exists(filename):
        if os.path.getsize(filename) == 0:
            st.warning(f"{filename} is empty. Fetching new asset data.")
            return None
        mod_time = os.path.getmtime(filename)
        if time.time() - mod_time < ttl:
            try:
                df = pd.read_csv(filename)
                if df.empty:
                    st.warning(f"{filename} contains no data. Fetching new asset data.")
                    return None
                return df.to_dict(orient="records")
            except pd.errors.EmptyDataError:
                st.warning(f"Encountered EmptyDataError when reading {filename}.")
                return None
    return None

def get_asset_data():
    data = load_asset_data_from_csv()
    if data is None:
        data = scout_assets()
        save_asset_data_to_csv(data)
    return data

# ---------------------- DISPLAY FUNCTIONS ---------------------- #

def display_assets():
    st.header("Asset Data")
    if 'preferred_assets' not in st.session_state:
        st.session_state['preferred_assets'] = []
    if st.session_state['asset_data']:
        asset_data = st.session_state['asset_data']
        df = pd.DataFrame(asset_data)
        st.dataframe(df)
        tickers = df['Ticker'].tolist()
        selected_assets = st.multiselect("Select your preferred assets", tickers, default=st.session_state['preferred_assets'])
        st.session_state['preferred_assets'] = selected_assets
        if selected_assets:
            st.write("Your preferred assets:")
            preferred_df = df[df['Ticker'].isin(selected_assets)]
            st.dataframe(preferred_df)
            check_price_alerts()
    else:
        st.info("Asset data not loaded. Click 'Update Stock Prices' to load.")

def check_price_alerts():
    if st.session_state.get('preferred_assets'):
        alert_threshold = st.slider("Set price change alert threshold (%)", min_value=0.0, max_value=10.0, value=5.0)
        for asset in st.session_state['asset_data']:
            if asset['Ticker'] in st.session_state['preferred_assets']:
                price_change = float(asset['Price Change (Today)'].strip('%'))
                if abs(price_change) >= alert_threshold:
                    st.warning(f"Alert: {asset['Ticker']} has changed by {price_change:.2f}% today!")

def display_asset_charts():
    if st.session_state['asset_data']:
        asset_data = st.session_state['asset_data']
        tickers = [asset['Ticker'] for asset in asset_data]
        selected_ticker = st.selectbox("Select a ticker to view price chart:", tickers)
        stock = yf.Ticker(selected_ticker)
        hist = stock.history(period="1mo")
        if not hist.empty:
            st.line_chart(hist['Close'])
        else:
            st.info("No chart data available for the selected ticker.")
    else:
        st.info("Asset data not loaded. Please update stock prices to view charts.")

def display_chart_for_asset(message):
    pattern = r'\b(?:price|chart)\s+(?:of\s+)?([A-Za-z0-9.\-]+)\b'
    matches = re.findall(pattern, message, re.IGNORECASE)
    if matches:
        ticker = matches[0].upper()
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period="1y")
            if not hist.empty:
                return hist['Close']
            else:
                st.write(f"No data found for ticker {ticker}")
                return None
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
            return None
    else:
        return None

def format_asset_suggestions(suggestions):
    if not suggestions:
        return "No assets currently meet the criteria for recommendation."
    suggestion_text = "Here are some asset suggestions based on recent performance:\n\n"
    for asset in suggestions:
        suggestion_text += (
            f"**{asset['Ticker']}**\n"
            f"- Price Change (Today): {asset['Price Change (Today)']}\n"
            f"- Current Price: {asset['Current Price']}\n"
        )
        if "Dividend Yield" in asset and asset["Dividend Yield"] != "N/A":
            suggestion_text += f"- Dividend Yield: {asset['Dividend Yield']}\n"
        suggestion_text += "\n"
    return suggestion_text

def generate_response(financial_data, user_message, vector_store):
    if st.session_state.get('asset_data'):
        asset_suggestions = st.session_state['asset_data']
        formatted_suggestions = format_asset_suggestions(asset_suggestions)
    else:
        formatted_suggestions = "No asset data available."
    query = financial_data + "\n" + user_message
    docs = vector_store.similarity_search(query, k=3) if vector_store else []
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Based on the user's financial data, the following asset suggestions, and the context from documents:

    Financial Data:
    {financial_data}

    Asset Suggestions:
    {formatted_suggestions}

    Context from documents:
    {context}

    User Message:
    {user_message}

    Provide a helpful and informative response as a personal finance assistant. Include prices of top movers in stocks.
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial assistant providing advice based on user data and market trends."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response

def chat_interface():
    st.header("Chat with Your Personal Finance Assistant")
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'chart_data' in message:
                st.line_chart(message['chart_data'])
    user_input = st.chat_input("You:")
    if user_input:
        financial_data = st.session_state['financial_data']
        vector_store = st.session_state['vector_store']
        response = generate_response(financial_data, user_input, vector_store)
        chart_data = display_chart_for_asset(user_input)
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        assistant_message = {"role": "assistant", "content": response}
        if chart_data is not None:
            assistant_message['chart_data'] = chart_data
        st.session_state['chat_history'].append(assistant_message)
        for message in st.session_state['chat_history'][-2:]:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                if 'chart_data' in message:
                    st.line_chart(message['chart_data'])

# ---------------------- BUDGETING & CRYPTO ---------------------- #

def budgeting_tool():
    st.header("Budgeting Tool")
    income = st.number_input("Monthly Income", min_value=0.0, step=100.0, help="Your total monthly income after taxes.")
    expenses = st.number_input("Monthly Expenses", min_value=0.0, step=100.0, help="Your total monthly expenses.")
    savings = income - expenses
    if savings >= 0:
        st.success(f"Monthly Savings: ${savings:.2f}")
    else:
        st.error(f"Monthly Deficit: ${-savings:.2f}")

def get_top_movers():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': '1',
        'limit': '500',
        'convert': 'USD',
        'sort': 'market_cap',
        'sort_dir': 'desc'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': CMC_API_KEY,
    }
    session = requests.Session()
    session.headers.update(headers)
    try:
        response = session.get(url, params=parameters)
        data = response.json()
        if 'data' not in data:
            st.error("Error: No data found in the response from CoinMarketCap.")
            return []
        crypto_data = []
        for crypto in data['data']:
            crypto_data.append({
                'Name': crypto['name'],
                'Symbol': crypto['symbol'],
                'Price (USD)': crypto['quote']['USD']['price'],
                '24h Change (%)': crypto['quote']['USD']['percent_change_24h']
            })
        crypto_data = sorted(crypto_data, key=lambda x: x['24h Change (%)'], reverse=True)
        top_movers = crypto_data[:10]
        for item in top_movers:
            item['Price (USD)'] = f"${item['Price (USD)']:.2f}"
            item['24h Change (%)'] = f"{item['24h Change (%)']:.2f}%"
        return top_movers
    except (requests.ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
        st.error("Connection error while fetching cryptocurrency data. Please try again later.")
        return []
    except json.JSONDecodeError as e:
        st.error("Error parsing JSON response. Please check the API response format.")
        return []

def get_crypto_prices():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd")
    return response.json()

# ---------------------- TECHNICAL ANALYSIS FUNCTIONS ---------------------- #

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
        summary = f"Price: ${latest_close:.2f}, RSI: {rsi:.2f}, SMA20: ${sma20:.2f}, Bollinger Bands: Upper=${bb_upper:.2f}, Lower=${bb_lower:.2f}"
    else:
        summary = f"Price: ${latest_close:.2f}, SMA20: ${sma20:.2f}, Bollinger Bands: Upper=${bb_upper:.2f}, Lower=${bb_lower:.2f}"
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
        summaries += f"{ticker} Technical Analysis:\nDaily: {daily_summary}\nWeekly: {weekly_summary}\n\n"
    return summaries

# ---------------------- GOOGLE TRENDS FUNCTION ---------------------- #

def fetch_google_trends(keyword="finance"):
    """
    Uses pytrends to fetch interest over time data for the given keyword.
    Returns a summary string of current trends.
    """
    try:
        pytrends.build_payload([keyword], cat=0, timeframe='now 7-d', geo='', gprop='')
        trends = pytrends.interest_over_time()
        if trends.empty:
            return f"No trends data available for {keyword}."
        latest = trends[keyword].iloc[-1]
        avg = trends[keyword].mean()
        summary = f"Latest interest for '{keyword}' is {latest} (average over last 7 days: {avg:.2f})."
        return summary
    except Exception as e:
        return f"Error fetching trends data: {e}"

# ---------------------- AGENTIC ADVISOR FUNCTIONS ---------------------- #

def call_openai_llm(prompt, system="", model="gpt-4o"):
    """
    Helper function to call the OpenAI LLM with a given prompt and system instruction.
    """
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
    conditions = {}
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            ten_year_yield = hist['Close'].iloc[-1]
            conditions["10Y Treasury Yield"] = ten_year_yield
        else:
            conditions["10Y Treasury Yield"] = "N/A"
    except Exception as e:
        conditions["10Y Treasury Yield"] = "N/A"
    
    # try:
    #     dxy = yf.Ticker("DX-Y.NYB")
    #     hist = dxy.history(period="5d")
    #     if not hist.empty:
    #         prices = hist['Close']
    #         trend = "decreasing" if prices.iloc[-1] < prices.iloc[0] else "increasing"
    #         conditions["DXY Trend"] = trend
    #         conditions["DXY Latest"] = prices.iloc[-1]
    #     else:
    #         conditions["DXY Trend"] = "N/A"
    # except Exception as e:
    #     conditions["DXY Trend"] = "N/A"

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
    except Exception as e:
        conditions["DXY Trend"] = "N/A"

    
    if isinstance(conditions.get("10Y Treasury Yield"), (int, float)) and conditions["10Y Treasury Yield"] < 3.0:
        conditions["Interest Rates"] = "Low"
    else:
        conditions["Interest Rates"] = "High or N/A"
    
    articles = fetch_finance_news()
    war_flag = any("war" in article['title'].lower() or "conflict" in article['title'].lower() for article in articles)
    conditions["War Status"] = "No ongoing wars" if not war_flag else "Potential conflict detected"
    conditions_str = (f"10Y Treasury Yield: {conditions.get('10Y Treasury Yield', 'N/A')}, "
                      f"DXY Trend: {conditions.get('DXY Trend', 'N/A')}, "
                      f"Interest Rates: {conditions.get('Interest Rates', 'N/A')}, "
                      f"War Status: {conditions.get('War Status', 'N/A')}.")
    return conditions_str

def agentic_advisor(user_input):
    """
    Multi-agent advisor chain:
      1. LLM1 analyzes the user input.
      2. LLM2 generates asset suggestions based on live data.
      3. LLM3 re-evaluates suggestions against macroeconomic conditions, technical analysis, and Google Trends data.
         If conditions aren’t favorable, the loop will iterate (up to 3 times) for more conservative recommendations.
    """
    # Step 1: Analyze user intent
    analysis_prompt = f"Analyze the following user input for investment advice, risk appetite, and expected returns: '{user_input}'. Summarize the key factors and preferences."
    llm1_response = call_openai_llm(analysis_prompt, system="You are an analyst specializing in extracting investment preferences.")
    
    # Step 2: Gather live market data
    live_news = fetch_finance_news()
    asset_data = get_asset_data()
    crypto_data = get_top_movers()
    commodities_data = get_commodities_data()
    trends_data = fetch_google_trends("finance")
    
    live_data_str = (
        f"News: {live_news}\n"
        f"Stocks: {format_asset_suggestions(asset_data)}\n"
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
    llm2_response = call_openai_llm(suggestion_prompt, system="You are a financial advisor specializing in asset recommendations.")
    
    # Step 3: Evaluate against macroeconomic conditions, technical analysis, and trends
    macro_conditions = get_macro_conditions()
    tech_analysis = get_technical_analysis_summaries(asset_data)
    evaluation_prompt = f"""
    Evaluate the following asset suggestions against current macroeconomic conditions, technical analysis, and Google Trends data:
    
    Asset Suggestions:
    {llm2_response}
    
    Macroeconomic Conditions:
    {macro_conditions}
    
    Technical Analysis:
    {tech_analysis}
    
    Google Trends Data:
    {trends_data}
    
    If conditions are favorable for risk assets (e.g., low US interest rates, low 10Y yield, decreasing DXY, no ongoing wars, and positive technical indicators), confirm the recommendations.
    Otherwise, adjust the suggestions to be more conservative.
    """
    llm3_response = call_openai_llm(evaluation_prompt, system="You are a senior financial strategist specialized in macroeconomic analysis, technical analysis, and risk management.")
    
    iterations = 0
    while iterations < 3:
        if "conservative" not in llm3_response.lower():
            break
        else:
            re_adjust_prompt = f"""
            The current macroeconomic conditions, technical analysis, and trends indicate a need for more conservative asset recommendations.
            Adjust the previous suggestions accordingly.
            
            Previous Asset Suggestions:
            {llm2_response}
            
            Macroeconomic Conditions:
            {macro_conditions}
            
            Technical Analysis:
            {tech_analysis}
            
            Google Trends Data:
            {trends_data}
            """
            llm2_response = call_openai_llm(re_adjust_prompt, system="You are a financial advisor specializing in asset recommendations.")
            evaluation_prompt = f"""
            Evaluate the following adjusted asset suggestions against current macroeconomic conditions, technical analysis, and trends:
            
            Adjusted Asset Suggestions:
            {llm2_response}
            
            Macroeconomic Conditions:
            {macro_conditions}
            
            Technical Analysis:
            {tech_analysis}
            
            Google Trends Data:
            {trends_data}
            """
            llm3_response = call_openai_llm(evaluation_prompt, system="You are a senior financial strategist specialized in macroeconomic analysis, technical analysis, and risk management.")
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

# ---------------------- MAIN APP ---------------------- #

st.markdown("# Welcome to Your Personal Finance Assistant 💰")

if not st.session_state['asset_data']:
    with st.spinner('Loading stock prices...'):
        st.session_state['asset_data'] = get_asset_data()
        st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

col1, col2 = st.columns([8, 2])
with col1:
    pass
with col2:
    if st.session_state['asset_data_timestamp']:
        st.write(f"**Stock prices updated as of:** {st.session_state['asset_data_timestamp']}")
    else:
        st.write("**Stock prices not loaded.**")
    if st.button("Update Stock Prices"):
        with st.spinner("Updating stock prices..."):
            st.session_state['asset_data'] = scout_assets()
            save_asset_data_to_csv(st.session_state['asset_data'])
            st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.success("Stock prices updated.")

with st.sidebar:
    st.header("User Settings")
    st.header("Process Documents for Vector Store")
    if st.button("Process Documents"):
        with st.spinner('Processing documents...'):
            data_folder = 'data'
            pdf_texts = load_and_process_pdfs(data_folder)
            if pdf_texts:
                st.session_state['vector_store'] = create_vector_store(pdf_texts)
                st.success("Documents processed and vector store created.")
            else:
                st.warning("No PDF documents found in the 'data' folder.")
    st.header("Upload Your Own Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        pdf_texts = []
        for uploaded_file in uploaded_files:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            pdf_texts.append(text)
        st.session_state['vector_store'] = create_vector_store(pdf_texts)
        st.success("Documents uploaded and processed.")
    st.header("Enter Your Financial Data")
    with st.form("financial_data_form"):
        st.write("Please provide your financial data.")
        financial_data_input = st.text_area("Financial Data", value=st.session_state['financial_data'], height=200,
                                            help="Enter any financial information you would like the assistant to consider.")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state['financial_data'] = financial_data_input
            st.success("Financial data updated.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["News", "Assets", "Chat", "Tools", "Agentic Advisor"])

with tab1:
    display_finance_news()

with tab2:
    display_assets()
    st.subheader("Asset Price Chart")
    display_asset_charts()
    st.subheader("Top Cryptocurrency Movers (24h Change)")
    crypto_data = get_top_movers()
    if crypto_data:
        df_crypto = pd.DataFrame(crypto_data)
        st.dataframe(df_crypto)
    else:
        st.write("Failed to retrieve cryptocurrency prices.")

with tab3:
    chat_interface()

with tab4:
    budgeting_tool()

with tab5:
    agentic_chat_interface()
