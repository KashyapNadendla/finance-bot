import streamlit as st
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import yfinance as yf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import re  # Added for regular expression matching
from newsapi.newsapi_exception import NewsAPIException
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_ta as ta
import tweepy
import math

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")    # Set up NEWS_API_KEY in .env
CMC_API_KEY = os.getenv("CMC_API_KEY")      # Added for CoinMarketCap API key

# Set up OpenAI client instance
client = OpenAI(api_key=API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

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

# Function to load and process PDFs from the data folder with error handling
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
                st.write(f"Processed {filename}")  # Feedback to the user
            except PyPDF2.errors.PdfReadError:
                print(f"Warning: '{filename}' could not be processed (EOF marker not found). Skipping.")
            except Exception as e:
                print(f"Warning: An error occurred while processing '{filename}': {e}. Skipping.")
    return pdf_texts




def create_vector_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]

    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    # Create a FAISS vector store without the `collection_name` argument
    vector_store = FAISS.from_texts(texts_chunks, embeddings)
    
    return vector_store

# Function to fetch the top 3 finance-related news articles
@st.cache_data(ttl=86400)  # Cache for 24 hours
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
        if 'rateLimited' in str(e):  # Check for rate limit error in exception message
            st.warning("News API rate limit exceeded. Please try again later.")
        else:
            st.error("An error occurred while fetching news. Please try again later.")
        return []

# Function to display the top finance news in Streamlit
# Display the finance news
def display_finance_news():
    st.subheader("Top 3 Finance News Articles Today")
    articles = fetch_finance_news()
    if articles:
        for i, article in enumerate(articles, 1):
            st.markdown(f"[**{i}. {article['title']}**]({article['url']})")
            st.write(f"Source: {article['source']}\n")
    else:
        st.write("No news articles available at this time.")

# Function to scout assets with real-time price action from Yahoo Finance
@st.cache_data(ttl=600)
def scout_assets():
    tickers = [
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO", 
        "LLY", "WMT", "JPM", "V", "UNH", "XOM", "NVO", "ORCL", "MA", "PG", "HD", "COST", 
        "JNJ", "ABBV", "BAC", "NFLX", "KO", "CRM", "SAP", "CVX", "ASML", "MRK", "TMUS", 
        "AMD", "TM", "PEP", "LIN", "AZN", "BABA", "CSCO", "NVS", "WFC", "ACN", "ADBE", 
        "TMO", "MCD", "PM", "SHEL", "ABT", "NOW", "AXP", "MS", "TXN", "GE", "IBM", "QCOM", 
        "CAT", "ISRG", "DHR", "RY", "INTU", "VZ", "GS", "DIS", "AMGN", "PDD", "UBER", "HSBC", 
        "CMCSA", "NEE", "RTX", "ARM", "PFE", "T", "HDB", "UL", "AMAT", "SPGI", "BKNG", "LOW", 
        "TTE", "BLK", "PGR", "BHP", "UNP", "SYK", "BX", "ETN", "SNY", "HON", "SCHW", "LMT", 
        "TJX", "BUD", "ANET", "KKR", "MUFG", "BSX", "VRTX", "C", "COP", "ADP", "PANW", "MDT", 
        "MU", "UPS", "CB", "ADI", "NKE", "FI", "BA", "RIO", "DE", "SBUX", "IBN", "GILD", "MMC", 
        "SONY", "PLD", "BMY", "SHOP", "MELI", "UBS", "AMT", "REGN", "LRCX", "PLTR", "SO", "TD", 
        "ICE", "INTC", "ELV", "MDLZ", "HCA", "KLAC", "DELL", "SHW", "INFY", "ENB", "DUK", "SCCO", 
        "CI", "RELX", "EQIX", "ABNB", "WM", "WELL", "MO", "RACE", "TT", "PBR.A", "PBR", "CTAS", "SMFG", 
        "BN", "MCO", "APO", "ZTS", "GD", "APH", "SNPS", "GEV", "CEG", "CME", "PH", "AON", "CDNS", "SPOT", 
        "ITW", "PYPL", "CL", "BP", "CMG", "BTI", "USB", "MSI", "PNC", "CRWD", "NU", "TRI", "GSK", "TDG", "MAR", 
        "NOC", "SAN", "CP", "CNQ", "ECL", "MRVL", "CVS", "DEO", "APD", "MMM", "CNI", "EOG", "TGT", "BDX", 
        "EQNR", "ORLY", "FDX", "BMO", "FCX", "CARR", "CRH", "MCK", "CSX", "BNS", "WMB", "DASH", "COF", "EPD", 
        "WDAY", "NGG", "NXPI", "AJG", "EMR", "RSG", "ADSK", "AFL", "DLR", "FTNT", "TTD", "CM", "PSA", "ROP", 
        "JD", "MET", "HLT", "TFC", "APP", "NSC", "GM", "BBVA", "TRV", "SLB", "ET", "OKE", "SPG", "RCL", "ITUB", 
        "BK", "KMI", "PCAR", "DHI", "SE", "GWW", "NEM", "MFG", "URI", "ING", "SRE", "O", "MFC", "COIN", "NTES", 
        "FANG", "AEP", "MNST", "AZO", "JCI", "PAYX", "PSX", "CPRT", "MSTR", "ALL", "AMP", "TEAM", "FIS", "AIG", 
        "FICO", "D", "AMX", "MPC", "TRP", "SU", "E", "HMC", "CHTR", "CPNG", "OXY", "CCI", "LHX", "LEN", "ROST", 
        "ALC", "VALE", "TEL", "PWR", "WCN", "BCS", "CMI", "PRU", "MPLX", "SQ", "COR", "FAST", "MPWR", "KMB", 
        "KDP", "MSCI", "AEM", "PEG", "TAK", "HLN", "KVUE", "ODFL", "NDAQ", "DDOG", "PCG", "STZ", "LYG", "VST", 
        "CTVA", "TCOM", "VRT", "FLUT", "F", "EW", "HWM", "VLO", "HES", "LNG", "KHC", "MCHP", "KR", "IT", "SNOW", 
        "GEHC", "EXC", "CBRE", "NWG", "FERG", "EA", "GRMN", "IQV", "ACGL", "OTIS", "VRSK", "IR", "AME", "GLW", 
        "IMO", "DFS", "LVS", "STLA", "GIS", "A", "YUM", "DAL", "IRM", "LULU", "IDXX", "BKR", "MLM", "CTSH", 
        "TRGP", "VMC", "SYY", "ALNY"]
    period = "1mo"
    interval = "1d"
## removed for testing "","HSY", "RMD", "ED", "HPQ", "ABEV", "XEL", "CCEP", "WIT", "GOLD", "EXR", 
        # "DD", "VEEV", "DOW", "HEI", "ARES", "VICI", "NUE", "EFX", "ARGX", "AXON", "WAB", "AVB", "MTB", "DB", 
        # "HIG", "SLF", "BIDU", "EIX", "HUM", "XYL", "ON", "EL", "CNC", "FMX", "NET", "EBAY", "WPM", "CVE", 
        # "WEC", "RJF", "BRO", "ROK", "CSGP", "HEI.A", "WTW", "FITB", "WDS", "CHT", "BCE", "FER", "PPG", 
        # "TSCO", "LI", "HUBS", "CCL", "ETR", "ANSS", "TTWO", "ZS", "LYB", "ERIC", "DXCM", "EQR", "FCNCA", 
        # "RBLX", "K", "NVR", "FCNCO", "STT", "MTD", "VTR", "TW", "IOT", "BNTX", "LYV", "BEKE", "PHM", "TEF", 
        # "ADM", "TPL", "DOV", "UAL", "AWK", "HPE", "BIIB", "KEYS", "TYL", "GPN", "FNV", "CAH", "CDW", "SW",
        # "NOK", "IFF", "DECK", "BBD", "DTE", "CVNA", "KB", "VLTO", "GIB", "FTV", "DVN", "STM", "HOOD", "SBAC", 
        # "TROW", "BR", "LDOS", "CHD", "PHG", "VOD", "IX", "HAL", "NTAP", "FE", "PBA", "TECK", "CQP", "PPL", 
       # "TU", "NTR", "ERIE", "ILMN", "CCJ", "BAH", "ES", "HUBB", "AEE", "WY", "CPAY", "ZM", "WDC", "EQT", 
       # "HBAN", "GDDY", "QSR", "ROL", "WST", "BAM", "PTC""
    # Parallel processing with ThreadPoolExecutor
    # def fetch_stock_data(ticker):
    #     stock = yf.Ticker(ticker)
    #     try:
    #         hist = stock.history(period="1mo", interval="1d")
    #         if hist.empty:
    #             return None  # Skip if no data found

    #         latest_close = hist['Close'].iloc[-1]
    #         open_price = hist['Close'].iloc[0]
    #         price_change_today = ((latest_close - open_price) / open_price) * 100
    #         dividend_yield = stock.info.get("dividendYield", "N/A")

    #         return {
    #             "Ticker": ticker,
    #             "Current Price": f"${latest_close:.2f}",
    #             "Dividend Yield": f"{dividend_yield:.2%}" if dividend_yield != "N/A" else "N/A",
    #             "Price Change (Today)": f"{price_change_today:.2f}%"
    #         }
    #     except Exception as e:
    #         print(f"Error retrieving data for {ticker}: {e}")
    #         return None

    # asset_data = []
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = {executor.submit(fetch_stock_data, ticker): ticker for ticker in tickers}
    #     for future in as_completed(futures):
    #         result = future.result()
    #         if result:
    #             asset_data.append(result)

    # return asset_data
    



    def get_social_popularity(ticker: str, max_tweets: int = 50) -> float:
        """
        Fetches recent tweets mentioning the given ticker symbol and calculates a
        simple popularity score.
    
        Arguments:
        ----------
        ticker : str
            The stock ticker to search for on Twitter (e.g. "TSLA")
        max_tweets : int
            The maximum number of tweets to retrieve for scoring.
    
        Returns:
        --------
        float
            A popularity score where a higher value indicates higher popularity.
            Returns 0 if no tweets found or an error occurs.
    """
    # 1. Get the bearer token from environment variable
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        print("Error: Missing TWITTER_BEARER_TOKEN in .env file.")
        return 0.0
    
    try:
        # 2. Initialize Tweepy with OAuth2 Bearer Token
        auth = tweepy.OAuth2BearerHandler(bearer_token)
        api = tweepy.API(auth)

        # 3. Construct a query. Searching for $TSLA can help find tweets mentioning the ticker.
        #    The "recent" search might not be the same as v1.1 "search_tweets", 
        #    but for demonstration, we'll use this approach. 
        #    If you're using Twitter API v2 or another endpoint, adjustments are needed.
        query = f"${ticker} -is:retweet"

        # 4. Search tweets using the old v1.1 endpoint "search_tweets":
        #    * If you're using Twitter API v2, you'll use the new client / search_recent_tweets etc.
        tweets = api.search_tweets(q=query, count=max_tweets, tweet_mode="extended")

        if not tweets:
            return 0.0  # No tweets found => 0 popularity

        # 5. Compute a naive popularity score
        #    For instance, we can sum the tweet's favorite_count + retweet_count,
        #    or simply count how many tweets we got. 
        #    You can add weight if you want to emphasize likes more than retweets, etc.
        popularity_score = 0
        for tweet in tweets:
            # v1.1 typical fields:
            #   tweet.favorite_count
            #   tweet.retweet_count
            #   tweet.full_text
            popularity_score += tweet.favorite_count + tweet.retweet_count

        # 6. Optional: Transform the raw sum into a scale (e.g., 0 - 100)
        #    For demonstration, let's do a log scale so that a few giant tweets
        #    won't overshadow everything.
        if popularity_score > 0:
            popularity_score = math.log2(popularity_score + 1) * 10
        else:
            # If no retweets/likes, just use the tweet count.
            popularity_score = len(tweets)

        return popularity_score

    except tweepy.TweepyException as e:
        print(f"Tweepy error: {e}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 0.0
    
    def fetch_stock_data(ticker):
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                st.warning(f"No data for {ticker}.")
                return None

            latest_close = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[-1]
            price_change_today = ((latest_close - open_price) / open_price) * 100

            # Calculate TA signals
            signal, reason = calculate_technical_indicators(ticker, period, interval)
            #popularity_score = get_social_popularity(ticker)

            return {
                "Ticker": ticker,
                "Current Price": f"${latest_close:.2f}",
                "Price Change (Today)": f"{price_change_today:.2f}%",
                "TA Signal": signal or "N/A",
                "TA Reason": reason or "N/A",
                #"Social Popularity": popularity_score,
            }
        except Exception as e:
            st.error(f"Error retrieving data for {ticker}: {e}")
            return None

    asset_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_stock_data, t): t for t in tickers}
        for f in as_completed(futures):
            res = f.result()
            if res:
                asset_data.append(res)
    for asset in asset_data:
        ticker = asset["Ticker"]
        popularity_score = get_social_popularity(ticker)
        asset["Popularity Score"] = f"{popularity_score:.2f}"
    return asset_data



# Function to display assets in a table with user preferences
def display_assets():
    st.header("Asset Data")
    if 'preferred_assets' not in st.session_state:
        st.session_state['preferred_assets'] = []

    # Check if asset data is available
    if st.session_state['asset_data']:
        asset_data = st.session_state['asset_data']
        df = pd.DataFrame(asset_data)
        st.dataframe(df)

        # Allow user to select preferred assets
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

# Function to check for price alerts
def check_price_alerts():
    if 'preferred_assets' in st.session_state and st.session_state['preferred_assets']:
        alert_threshold = st.slider(
            "Set price change alert threshold (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0
        )
        if st.session_state['asset_data']:
            asset_data = st.session_state['asset_data']
            for asset in asset_data:
                if asset['Ticker'] in st.session_state['preferred_assets']:
                    price_change = float(asset['Price Change (Today)'].strip('%'))
                    if abs(price_change) >= alert_threshold:
                        st.warning(f"Alert: {asset['Ticker']} has changed by {price_change:.2f}% today!")
        else:
            st.info("Asset data not loaded. Please update stock prices to check price alerts.")

# Function to display asset charts
def display_asset_charts():
    if st.session_state['asset_data']:
        asset_data = st.session_state['asset_data']
        tickers = [asset['Ticker'] for asset in asset_data]
        selected_ticker = st.selectbox("Select a ticker to view price chart:", tickers)
        stock = yf.Ticker(selected_ticker)
        hist = stock.history(period="1mo")
        st.line_chart(hist['Close'])
    else:
        st.info("Asset data not loaded. Please update stock prices to view charts.")

# Function to format asset suggestions as text
def format_asset_suggestions(suggestions):
    if not suggestions:
        return "No assets currently meet the criteria for recommendation."
    suggestion_text = "Here are some asset suggestions based on recent performance:\n\n"
    for asset in suggestions:
        suggestion_text += (
            f"**{asset['Ticker']}**\n"
            f"- Price Change (Today): {asset['Price Change (Today)']}\n"
            f"- Dividend Yield: {asset['Dividend Yield']}\n"
            f"- Current Price: {asset['Current Price']}\n"
            f"- Popularity Score: {asset.get('Popularity Score', 'N/A')}\n\n"
        )
    return suggestion_text


# Function to generate response from OpenAI
def generate_response(financial_data, user_message, vector_store):
    if st.session_state['asset_data']:
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

    # Generate response from OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial assistant that provides advice based on the user's data and market trends. Always ensure that your advice is appropriate for the user's financial situation."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content

    return response

# Function to handle the chat interface
def chat_interface():
    st.header("Chat with Your Personal Finance Assistant")

    # Display the chat messages in order
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            # If there's a chart associated with this message, display it
            if 'chart_data' in message:
                st.line_chart(message['chart_data'])

    # Get User Input
    user_input = st.chat_input("You:")
    if user_input:
        financial_data = st.session_state['financial_data']
        vector_store = st.session_state['vector_store']

        # Generate assistant's response
        response = generate_response(financial_data, user_input, vector_store)

        # Check if the user is asking for the price of an asset
        chart_data = display_chart_for_asset(user_input)

        # Add user message to chat history
        st.session_state['chat_history'].append({"role": "user", "content": user_input})

        # Add assistant's message and chart data to chat history
        assistant_message = {"role": "assistant", "content": response}
        if chart_data is not None:
            assistant_message['chart_data'] = chart_data
        st.session_state['chat_history'].append(assistant_message)

        # Display the last two messages (user and assistant)
        for message in st.session_state['chat_history'][-2:]:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                if 'chart_data' in message:
                    st.line_chart(message['chart_data'])


def display_chart_for_asset(message):
    # Regular expression pattern to detect phrases like 'price of [ticker]' or 'chart of [ticker]'
    pattern = r'\b(?:price|chart)\s+(?:of\s+)?([A-Za-z0-9.\-]+)\b'
    matches = re.findall(pattern, message, re.IGNORECASE)
    if matches:
        ticker = matches[0].upper()
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period="1y")
            if not hist.empty:
                # Return the closing prices for plotting
                return hist['Close']
            else:
                st.write(f"No data found for ticker {ticker}")
                return None
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
            return None
    else:
        return None

        response = generate_response(financial_data, user_input, vector_store)

        # Add user message and assistant response to chat history
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

        # Display latest messages
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)


def budgeting_tool():
    st.header("Enhanced Budgeting Tool")

    # Monthly Income Input
    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        step=100.0,
        help="Your total monthly income after taxes."
    )

    # Expense Categories
    st.subheader("Enter Your Monthly Expenses by Category")
    expense_categories = {
        'Housing (Rent/Mortgage)': 0.0,
        'Utilities': 0.0,
        'Food': 0.0,
        'Transportation': 0.0,
        'Entertainment': 0.0,
        'Healthcare': 0.0,
        'Insurance': 0.0,
        'Debt Payments': 0.0,
        'Education': 0.0,
        'Savings & Investments': 0.0,
        'Miscellaneous': 0.0,
    }

    total_expenses = 0.0

    # Input expenses for each category
    for category in expense_categories.keys():
        expense = st.number_input(
            f"{category}",
            min_value=0.0,
            step=10.0,
            help=f"Enter your monthly expenses for {category.lower()}."
        )
        expense_categories[category] = expense
        total_expenses += expense

    # Calculate Savings
    savings = income - total_expenses

    # Display Results
    if savings >= 0:
        st.success(f"Monthly Savings: ${savings:.2f}")
    else:
        st.error(f"Monthly Deficit: ${-savings:.2f}")

    # Calculate the percentage of income spent on each category
    st.subheader("Expenses Breakdown")
    if income > 0:
        expense_percentages = {category: (amount / income) * 100 for category, amount in expense_categories.items()}
        df_expenses = pd.DataFrame({
            'Category': list(expense_categories.keys()),
            'Amount': list(expense_categories.values()),
            'Percentage of Income': [f"{percent:.2f}%" for percent in expense_percentages.values()]
        })

        st.dataframe(df_expenses)

        # Visualize expenses using a pie chart
        # Visualize expenses using a pie chart
        fig1, ax1 = plt.subplots(figsize=(8, 6))  # Increase figure size for better readability

        # Create the pie chart without labels
        wedges, texts, autotexts = ax1.pie(
        expense_categories.values(),
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},
        pctdistance=0.85  # Position the percentage labels closer to the center
        )

        # Draw a circle at the center to turn the pie into a donut chart (optional for aesthetics)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig1.gca().add_artist(centre_circle)

        # Add a legend outside the pie chart
        ax1.legend(
        wedges,
        expense_categories.keys(),
        title="Expense Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
        )

        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.warning("Please enter your income to see expense breakdown.")

    # Savings Goals
    st.subheader("Savings Goals")
    goal_amount = st.number_input(
        "Enter your savings goal amount:",
        min_value=0.0,
        step=100.0,
        help="Enter the total amount you aim to save."
    )
    goal_timeframe = st.number_input(
        "Enter your goal timeframe in months:",
        min_value=1,
        step=1,
        help="Enter the number of months you plan to achieve this goal."
    )

    if savings > 0 and goal_amount > 0 and goal_timeframe > 0:
        monthly_savings_needed = goal_amount / goal_timeframe
        if savings >= monthly_savings_needed:
            st.success(f"You are on track to meet your savings goal! You need to save at least ${monthly_savings_needed:.2f} per month.")
        else:
            st.warning(f"You need to save ${monthly_savings_needed:.2f} per month to meet your goal. Consider reducing expenses or increasing income.")
    elif goal_amount > 0 and goal_timeframe > 0:
        st.error("Your current budget does not allow for savings towards your goal.")

    # Budget Recommendations based on 50/30/20 rule
    st.subheader("Budget Recommendations")
    if income > 0:
        needs = income * 0.5
        wants = income * 0.3
        savings_recommended = income * 0.2

        st.write(f"Based on the 50/30/20 rule:")
        st.write(f"- **Needs (50%)**: ${needs:.2f}")
        st.write(f"- **Wants (30%)**: ${wants:.2f}")
        st.write(f"- **Savings (20%)**: ${savings_recommended:.2f}")

        # Compare actual expenses to recommendations
        total_needs = sum([expense_categories[cat] for cat in ['Housing (Rent/Mortgage)', 'Utilities', 'Food', 'Transportation', 'Healthcare', 'Insurance', 'Debt Payments']])
        total_wants = sum([expense_categories[cat] for cat in ['Entertainment', 'Education', 'Miscellaneous']])
        total_savings = savings if savings > 0 else 0

        st.write(f"Your actual spending:")
        st.write(f"- **Needs**: ${total_needs:.2f}")
        st.write(f"- **Wants**: ${total_wants:.2f}")
        st.write(f"- **Savings**: ${total_savings:.2f}")

        # Visual comparison
        labels = ['Needs', 'Wants', 'Savings']
        recommended = [needs, wants, savings_recommended]
        actual = [total_needs, total_wants, total_savings]

        x = np.arange(len(labels))  # label locations
        width = 0.35  # bar width

        fig2, ax2 = plt.subplots()
        rects1 = ax2.bar(x - width/2, recommended, width, label='Recommended')
        rects2 = ax2.bar(x + width/2, actual, width, label='Actual')

        # Add labels, title, and custom x-axis tick labels
        ax2.set_ylabel('Amount ($)')
        ax2.set_title('Recommended vs Actual Spending')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()

        # Attach a text label above each bar in *rects*, displaying its height.
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'${height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        st.pyplot(fig2)
    else:
        st.warning("Please enter your income to see budget recommendations.")

    # Future Projections
    st.subheader("Future Savings Projection")
    projection_months = st.number_input(
        "Enter the number of months for projection:",
        min_value=1,
        step=1,
        help="Enter how many months into the future you want to project your savings."
    )

    if savings > 0 and projection_months > 0:
        projected_savings = savings * projection_months
        st.write(f"In {projection_months} months, you can save approximately ${projected_savings:.2f} if your income and expenses remain the same.")
    elif projection_months > 0:
        st.error("Your current budget does not allow for savings projection.")



# Function to get top movers within the top 500 coins
def get_top_movers():
    CMC_API_KEY = os.getenv("CMC_API_KEY")
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': '1',
        'limit': '500',  # Fetch top 500 coins by market cap
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

        # Check if the 'data' key is present in the response
        if 'data' not in data:
            st.error("Error: No data found in the response from CoinMarketCap.")
            print("Response content:", data)  # Log the full response for debugging
            return []

        # Extract the relevant data
        crypto_data = []
        for crypto in data['data']:
            crypto_data.append({
                'Name': crypto['name'],
                'Symbol': crypto['symbol'],
                'Price (USD)': crypto['quote']['USD']['price'],
                '24h Change (%)': crypto['quote']['USD']['percent_change_24h']
            })

        # Sort the data by 24h Change (%) in descending order
        crypto_data = sorted(crypto_data, key=lambda x: x['24h Change (%)'], reverse=True)

        # Get top 10 movers
        top_movers = crypto_data[:10]

        # Format the prices and percentage changes
        for item in top_movers:
            item['Price (USD)'] = f"${item['Price (USD)']:.2f}"
            item['24h Change (%)'] = f"{item['24h Change (%)']:.2f}%"

        return top_movers

    except (requests.ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
        st.error("Connection error while fetching cryptocurrency data. Please try again later.")
        print(e)
        return []
    except json.JSONDecodeError as e:
        st.error("Error parsing JSON response. Please check the API response format.")
        print("JSON decode error:", e)
        return []


# Function to get cryptocurrency prices
def get_crypto_prices():
    response = requests.get(
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
    )
    data = response.json()
    return data

## Trying to integrate Technical analysis
def calculate_technical_indicators(ticker, period="3mo", interval="1d"):
    """
    Fetch daily data for the given period & interval, then compute basic TA indicators
    like RSI. Return a summary with any signals.
    """
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            return None, "No data found"

        # Example using pandas_ta
        import pandas_ta as ta
        hist["RSI"] = ta.rsi(hist["Close"])
        
        # Simple logic: If RSI < 30 => 'Bullish', RSI > 70 => 'Bearish', else 'Neutral'
        current_rsi = hist["RSI"].iloc[-1]
        if current_rsi < 30:
            signal = "Bullish"
            reason = f"RSI is {current_rsi:.2f}, indicating oversold conditions."
        elif current_rsi > 70:
            signal = "Bearish"
            reason = f"RSI is {current_rsi:.2f}, indicating overbought conditions."
        else:
            signal = "Neutral"
            reason = f"RSI is {current_rsi:.2f}, indicating neither overbought nor oversold."

        return signal, reason
    except Exception as e:
        return None, f"Error: {e}"


# Reasoning why we took a trade
def generate_trade_reason(signal, reason): #, popularity_score=None):
    # Construct a user-friendly explanation
    explanation = f"Signal: {signal}\nReason: {reason}"
    #if popularity_score is not None:
        #explanation += f"\nPopularity Score: {popularity_score}"
    return explanation



# Main App

# Add a welcome header with an emoji
st.markdown("# Welcome to Your Personal Finance Assistant 💰")

# Load asset data on initial app load
if not st.session_state['asset_data']:
    with st.spinner('Loading stock prices...'):
        st.session_state['asset_data'] = scout_assets()
        st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Display stock prices updated time and Update button at the top right
col1, col2 = st.columns([8, 2])

with col1:
    pass  # Empty column for layout alignment

with col2:
    if st.session_state['asset_data_timestamp']:
        st.write(f"**Stock prices updated as of:** {st.session_state['asset_data_timestamp']}")
    else:
        st.write("**Stock prices not loaded.**")
    if st.button("Update Stock Prices"):
        with st.spinner("Updating stock prices..."):
            st.session_state['asset_data'] = scout_assets()
            st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.success("Stock prices updated.")

# Sidebar with user inputs
with st.sidebar:
    st.header("User Settings")

    # Process Documents for Vector Store
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

    # Allow Users to Upload Their Own Documents
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

    # Financial Data Input
    st.header("Enter Your Financial Data")
    with st.form("financial_data_form"):
        st.write("Please provide your financial data.")
        financial_data_input = st.text_area(
            "Financial Data",
            value=st.session_state['financial_data'],
            height=200,
            help="Enter any financial information you would like the assistant to consider."
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state['financial_data'] = financial_data_input
            st.success("Financial data updated.")

# Create Tabs for Navigation
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
