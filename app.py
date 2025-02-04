import streamlit as st
from openai import OpenAI
import os
import PyPDF2
import yfinance as yf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import re
from newsapi.newsapi_exception import NewsAPIException
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CMC_API_KEY = os.getenv("CMC_API_KEY")

# Set up OpenAI client instance
client = OpenAI(api_key=API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Define the CSV file for storing stock data
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
    st.session_state['asset_data_timestamp'] = None  # To store the time when data was last updated


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
                st.error(f"Warning: '{filename}' could not be processed (EOF marker not found). Skipping.")
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
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="1d", interval="1m")
        if hist.empty:
            return None
        latest_close = hist['Close'].iloc[-1]
        open_price = hist['Close'].iloc[0]
        price_change_today = ((latest_close - open_price) / open_price) * 100
        dividend_yield = stock.info.get("dividendYield", "N/A")
        return {
            "Ticker": ticker,
            "Current Price": f"${latest_close:.2f}",
            "Dividend Yield": f"{dividend_yield:.2%}" if dividend_yield != "N/A" else "N/A",
            "Price Change (Today)": f"{price_change_today:.2f}%"
        }
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None

def scout_assets():
    tickers = [
        # (List of tickers...)
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO",
        # ... add the rest as in your original code
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
        mod_time = os.path.getmtime(filename)
        if time.time() - mod_time < ttl:
            df = pd.read_csv(filename)
            return df.to_dict(orient="records")
    return None

def get_asset_data():
    # First try to load from CSV if recent
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
        selected_assets = st.multiselect(
            "Select your preferred assets",
            tickers,
            default=st.session_state['preferred_assets']
        )
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
        alert_threshold = st.slider(
            "Set price change alert threshold (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0
        )
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
            f"- Dividend Yield: {asset['Dividend Yield']}\n"
            f"- Current Price: {asset['Current Price']}\n\n"
        )
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

    Provide a helpful and informative response as a personal finance assistant. Consider the user's financial data, asset suggestions, and the context from the documents in your response. Include prices of top movers in stocks based on the data you have.
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial assistant that provides advice based on the user's data and market trends. Always ensure that your advice is appropriate for the user's financial situation."},
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
    income = st.number_input("Monthly Income", min_value=0.0, step=100.0,
                             help="Your total monthly income after taxes.")
    expenses = st.number_input("Monthly Expenses", min_value=0.0, step=100.0,
                               help="Your total monthly expenses.")
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


# ---------------------- MAIN APP ---------------------- #

st.markdown("# Welcome to Your Personal Finance Assistant 💰")

# Load asset data on initial app load (use CSV caching)
if not st.session_state['asset_data']:
    with st.spinner('Loading stock prices...'):
        st.session_state['asset_data'] = get_asset_data()
        st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Display update information and button
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
        financial_data_input = st.text_area("Financial Data",
                                            value=st.session_state['financial_data'],
                                            height=200,
                                            help="Enter any financial information you would like the assistant to consider.")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state['financial_data'] = financial_data_input
            st.success("Financial data updated.")

tab1, tab2, tab3, tab4 = st.tabs(["News", "Assets", "Chat", "Tools"])

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

# ---------------------- SUGGESTED UPGRADES ----------------------
# Future improvements might include:
# 1. **Agentic Architecture:** Integrate a LangChain Agent (or similar) that can autonomously
#    perform tasks (e.g. fetch news, analyze your portfolio, trigger alerts, etc.) using various tools.
# 2. **Modularization:** Split your code into multiple modules (e.g., data_fetch.py, chat.py, utils.py)
#    to improve maintainability.
# 3. **Real-time Updates:** Use websockets or scheduled background jobs to update stock and crypto data in real time.
# 4. **Enhanced Caching:** Use a more robust caching backend (such as Redis) for sharing cached data across sessions.
