import streamlit as st
from openai import OpenAI
import os
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

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")    
CMC_API_KEY = os.getenv("CMC_API_KEY")      

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

# Function to fetch the top 5 finance-related news articles
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
            page_size=5
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
    st.subheader("Top 5 Finance News Articles Today")
    articles = fetch_finance_news()
    if articles:
        for i, article in enumerate(articles, 1):
            st.markdown(f"[**{i}. {article['title']}**]({article['url']})")
            st.write(f"Source: {article['source']}\n")
    else:
        st.write("No news articles available at this time.")

# Function to scout assets with real-time price action from Yahoo Finance
@st.cache_data(ttl=600)
<<<<<<< Updated upstream
# def scout_assets():
#     tickers = [
#         "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO", 
#         "LLY", "WMT", "JPM", "V", "UNH", "XOM", "NVO", "ORCL", "MA", "PG", "HD", "COST", 
#         "JNJ", "ABBV", "BAC", "NFLX", "KO", "CRM", "SAP", "CVX", "ASML", "MRK", "TMUS", 
#         "AMD", "TM", "PEP", "LIN", "AZN", "BABA", "CSCO", "NVS", "WFC", "ACN", "ADBE", 
#         "TMO", "MCD", "PM", "SHEL", "ABT", "NOW", "AXP", "MS", "TXN", "GE", "IBM", "QCOM", 
#         "CAT", "ISRG", "DHR", "RY", "INTU", "VZ", "GS", "DIS", "AMGN", "PDD", "UBER", "HSBC", 
#         "CMCSA", "NEE", "RTX", "ARM", "PFE", "T", "HDB", "UL", "AMAT", "SPGI", "BKNG", "LOW", 
#         "TTE", "BLK", "PGR", "BHP", "UNP", "SYK", "BX", "ETN", "SNY", "HON", "SCHW", "LMT", 
#         "TJX", "BUD", "ANET", "KKR", "MUFG", "BSX", "VRTX", "C", "COP", "ADP", "PANW", "MDT", 
#         "MU", "UPS", "CB", "ADI", "NKE", "FI", "BA", "RIO", "DE", "SBUX", "IBN", "GILD", "MMC", 
#         "SONY", "PLD", "BMY", "SHOP", "MELI", "UBS", "AMT", "REGN", "LRCX", "PLTR", "SO", "TD", 
#         "ICE", "INTC", "ELV", "MDLZ", "HCA", "KLAC", "DELL", "SHW", "INFY", "ENB", "DUK", "SCCO", 
#         "CI", "RELX", "EQIX", "ABNB", "WM", "WELL", "MO", "RACE", "TT", "PBR.A", "PBR", "CTAS", "SMFG", 
#         "BN", "MCO", "APO", "ZTS", "GD", "APH", "SNPS", "GEV", "CEG", "CME", "PH", "AON", "CDNS", "SPOT", 
#         "ITW", "PYPL", "CL", "BP", "CMG", "BTI", "USB", "MSI", "PNC", "CRWD", "NU", "TRI", "GSK", "TDG", "MAR", 
#         "NOC", "SAN", "CP", "CNQ", "ECL", "MRVL", "CVS", "DEO", "APD", "MMM", "CNI", "EOG", "TGT", "BDX", 
#         "EQNR", "ORLY", "FDX", "BMO", "FCX", "CARR", "CRH", "MCK", "CSX", "BNS", "WMB", "DASH", "COF", "EPD", 
#         "WDAY", "NGG", "NXPI", "AJG", "EMR", "RSG", "ADSK", "AFL", "DLR", "FTNT", "TTD", "CM", "PSA", "ROP", 
#         "JD", "MET", "HLT", "TFC", "APP", "NSC", "GM", "BBVA", "TRV", "SLB", "ET", "OKE", "SPG", "RCL", "ITUB", 
#         "BK", "KMI", "PCAR", "DHI", "SE", "GWW", "NEM", "MFG", "URI", "ING", "SRE", "O", "MFC", "COIN", "NTES", 
#         "FANG", "AEP", "MNST", "AZO", "JCI", "PAYX", "PSX", "CPRT", "MSTR", "ALL", "AMP", "TEAM", "FIS", "AIG", 
#         "FICO", "D", "AMX", "MPC", "TRP", "SU", "E", "HMC", "CHTR", "CPNG", "OXY", "CCI", "LHX", "LEN", "ROST", 
#         "ALC", "VALE", "TEL", "PWR", "WCN", "BCS", "CMI", "PRU", "MPLX", "SQ", "COR", "FAST", "MPWR", "KMB", 
#         "KDP", "MSCI", "AEM", "PEG", "TAK", "HLN", "KVUE", "ODFL", "NDAQ", "DDOG", "PCG", "STZ", "LYG", "VST", 
#         "CTVA", "TCOM", "VRT", "FLUT", "F", "EW", "HWM", "VLO", "HES", "LNG", "KHC", "MCHP", "KR", "IT", "SNOW", 
#         "GEHC", "EXC", "CBRE", "NWG", "FERG", "EA", "GRMN", "IQV", "ACGL", "OTIS", "VRSK", "IR", "AME", "GLW", 
#         "IMO", "DFS", "LVS", "STLA", "GIS", "A", "YUM", "DAL", "IRM", "LULU", "IDXX", "BKR", "MLM", "CTSH", 
#         "TRGP", "VMC", "SYY", "ALNY", "HSY", "RMD", "ED", "HPQ", "ABEV", "XEL", "CCEP", "WIT", "GOLD", "EXR", 
#         "DD", "VEEV", "DOW", "HEI", "ARES", "VICI", "NUE", "EFX", "ARGX", "AXON", "WAB", "AVB", "MTB", "DB", 
#         "HIG", "SLF", "BIDU", "EIX", "HUM", "XYL", "ON", "EL", "CNC", "FMX", "NET", "EBAY", "WPM", "CVE", 
#         "WEC", "RJF", "BRO", "ROK", "CSGP", "HEI.A", "WTW", "FITB", "WDS", "CHT", "BCE", "FER", "PPG", 
#         "TSCO", "LI", "HUBS", "CCL", "ETR", "ANSS", "TTWO", "ZS", "LYB", "ERIC", "DXCM", "EQR", "FCNCA", 
#         "RBLX", "K", "NVR", "FCNCO", "STT", "MTD", "VTR", "TW", "IOT", "BNTX", "LYV", "BEKE", "PHM", "TEF", 
#         "ADM", "TPL", "DOV", "UAL", "AWK", "HPE", "BIIB", "KEYS", "TYL", "GPN", "FNV", "CAH", "CDW", "SW",
#         "NOK", "IFF", "DECK", "BBD", "DTE", "CVNA", "KB", "VLTO", "GIB", "FTV", "DVN", "STM", "HOOD", "SBAC", 
#         "TROW", "BR", "LDOS", "CHD", "PHG", "VOD", "IX", "HAL", "NTAP", "FE", "PBA", "TECK", "CQP", "PPL", 
#         "TU", "NTR", "ERIE", "ILMN", "CCJ", "BAH", "ES", "HUBB", "AEE", "WY", "CPAY", "ZM", "WDC", "EQT", 
#         "HBAN", "GDDY", "QSR", "ROL", "WST", "BAM", "PTC"]

    # # Parallel processing with ThreadPoolExecutor
    # def fetch_stock_data(ticker):
    #     stock = yf.Ticker(ticker)
    #     try:
    #         hist = stock.history(period="1d", interval="1m")
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
def scout_assets():
    tickers = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO", 
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
        "TRGP", "VMC", "SYY", "ALNY", "HSY", "RMD", "ED", "HPQ", "ABEV", "XEL", "CCEP", "WIT", "GOLD", "EXR", 
        "DD", "VEEV", "DOW", "HEI", "ARES", "VICI", "NUE", "EFX", "ARGX", "AXON", "WAB", "AVB", "MTB", "DB", 
        "HIG", "SLF", "BIDU", "EIX", "HUM", "XYL", "ON", "EL", "CNC", "FMX", "NET", "EBAY", "WPM", "CVE", 
        "WEC", "RJF", "BRO", "ROK", "CSGP", "HEI.A", "WTW", "FITB", "WDS", "CHT", "BCE", "FER", "PPG", 
        "TSCO", "LI", "HUBS", "CCL", "ETR", "ANSS", "TTWO", "ZS", "LYB", "ERIC", "DXCM", "EQR", "FCNCA", 
        "RBLX", "K", "NVR", "FCNCO", "STT", "MTD", "VTR", "TW", "IOT", "BNTX", "LYV", "BEKE", "PHM", "TEF", 
        "ADM", "TPL", "DOV", "UAL", "AWK", "HPE", "BIIB", "KEYS", "TYL", "GPN", "FNV", "CAH", "CDW", "SW",
        "NOK", "IFF", "DECK", "BBD", "DTE", "CVNA", "KB", "VLTO", "GIB", "FTV", "DVN", "STM", "HOOD", "SBAC", 
        "TROW", "BR", "LDOS", "CHD", "PHG", "VOD", "IX", "HAL", "NTAP", "FE", "PBA", "TECK", "CQP", "PPL", 
        "TU", "NTR", "ERIE", "ILMN", "CCJ", "BAH", "ES", "HUBB", "AEE", "WY", "CPAY", "ZM", "WDC", "EQT", 
        "HBAN", "GDDY", "QSR", "ROL", "WST", "BAM", "PTC"]
    period = "1mo"
    interval = "1d"

    asset_data = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                st.warning(f"No data found for ticker {ticker}.")
                continue

            latest_close = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[-1]
            price_change_today = ((latest_close - open_price) / open_price) * 100
            dividend_yield = stock.info.get("dividendYield", "N/A")

            asset_data.append({
                "Ticker": ticker,
                "Current Price": f"${latest_close:.2f}",
                "Dividend Yield": f"{dividend_yield:.2%}" if dividend_yield != "N/A" else "N/A",
                "Price Change (Today)": f"{price_change_today:.2f}%"
            })
        except Exception as e:
            st.error(f"Error retrieving data for {ticker}: {e}")
            continue

    if not asset_data:
        st.error("No asset data could be fetched. Please try again later.")
=======


# def fetch_and_save_stock_data():
#     AV_API_KEY = os.getenv("AV_API_KEY")
#     tickers =  ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO", 
#         "LLY", "WMT", "JPM", "V", "UNH", "XOM", "NVO", "ORCL", "MA", "PG", "HD", "COST", 
#         "JNJ", "ABBV", "BAC", "NFLX", "KO", "CRM", "SAP", "CVX", "ASML", "MRK", "TMUS", 
#         "AMD", "TM", "PEP", "LIN", "AZN", "BABA", "CSCO", "NVS", "WFC", "ACN", "ADBE", 
#         "TMO", "MCD", "PM", "SHEL", "ABT", "NOW", "AXP", "MS", "TXN", "GE", "IBM", "QCOM", 
#         "CAT", "ISRG", "DHR", "RY", "INTU", "VZ", "GS", "DIS", "AMGN", "PDD", "UBER", "HSBC", 
#         "CMCSA", "NEE", "RTX", "ARM", "PFE", "T", "HDB", "UL", "AMAT", "SPGI", "BKNG", "LOW", 
#         "TTE", "BLK", "PGR", "BHP", "UNP", "SYK", "BX", "ETN", "SNY", "HON", "SCHW", "LMT", 
#         "TJX", "BUD", "ANET", "KKR", "MUFG", "BSX", "VRTX", "C", "COP", "ADP", "PANW", "MDT", 
#         "MU", "UPS", "CB", "ADI", "NKE", "FI", "BA", "RIO", "DE", "SBUX", "IBN", "GILD", "MMC", 
#         "SONY", "PLD", "BMY", "SHOP", "MELI", "UBS", "AMT", "REGN", "LRCX", "PLTR", "SO", "TD", 
#         "ICE", "INTC", "ELV", "MDLZ", "HCA", "KLAC", "DELL", "SHW", "INFY", "ENB", "DUK", "SCCO", 
#         "CI", "RELX", "EQIX", "ABNB", "WM", "WELL", "MO", "RACE", "TT", "PBR.A", "PBR", "CTAS", "SMFG", 
#         "BN", "MCO", "APO", "ZTS", "GD", "APH", "SNPS", "GEV", "CEG", "CME", "PH", "AON", "CDNS", "SPOT", 
#         "ITW", "PYPL", "CL", "BP", "CMG", "BTI", "USB", "MSI", "PNC", "CRWD", "NU", "TRI", "GSK", "TDG", "MAR", 
#         "NOC", "SAN", "CP", "CNQ", "ECL", "MRVL", "CVS", "DEO", "APD", "MMM", "CNI", "EOG", "TGT", "BDX", 
#         "EQNR", "ORLY", "FDX", "BMO", "FCX", "CARR", "CRH", "MCK", "CSX", "BNS", "WMB", "DASH", "COF", "EPD", 
#         "WDAY", "NGG", "NXPI", "AJG", "EMR", "RSG", "ADSK", "AFL", "DLR", "FTNT", "TTD", "CM", "PSA", "ROP", 
#         "JD", "MET", "HLT", "TFC", "APP", "NSC", "GM", "BBVA", "TRV", "SLB", "ET", "OKE", "SPG", "RCL", "ITUB", 
#         "BK", "KMI", "PCAR", "DHI", "SE", "GWW", "NEM", "MFG", "URI", "ING", "SRE", "O", "MFC", "COIN", "NTES", 
#         "FANG", "AEP", "MNST", "AZO", "JCI", "PAYX", "PSX", "CPRT", "MSTR", "ALL", "AMP", "TEAM", "FIS", "AIG", 
#         "FICO", "D", "AMX", "MPC", "TRP", "SU", "E", "HMC", "CHTR", "CPNG", "OXY", "CCI", "LHX", "LEN", "ROST", 
#         "ALC", "VALE", "TEL", "PWR", "WCN", "BCS", "CMI", "PRU", "MPLX", "SQ", "COR", "FAST", "MPWR", "KMB", 
#         "KDP", "MSCI", "AEM", "PEG", "TAK", "HLN", "KVUE", "ODFL", "NDAQ", "DDOG", "PCG", "STZ", "LYG", "VST", 
#         "CTVA", "TCOM", "VRT", "FLUT", "F", "EW", "HWM", "VLO", "HES", "LNG", "KHC", "MCHP", "KR", "IT", "SNOW", 
#         "GEHC", "EXC", "CBRE", "NWG", "FERG", "EA", "GRMN", "IQV", "ACGL", "OTIS", "VRSK", "IR", "AME", "GLW", 
#         "IMO", "DFS", "LVS", "STLA", "GIS", "A", "YUM", "DAL", "IRM", "LULU", "IDXX", "BKR", "MLM", "CTSH", 
#         "TRGP", "VMC", "SYY", "ALNY", "HSY", "RMD", "ED", "HPQ", "ABEV", "XEL", "CCEP", "WIT", "GOLD", "EXR", 
#         "DD", "VEEV", "DOW", "HEI", "ARES", "VICI", "NUE", "EFX", "ARGX", "AXON", "WAB", "AVB", "MTB", "DB", 
#         "HIG", "SLF", "BIDU", "EIX", "HUM", "XYL", "ON", "EL", "CNC", "FMX", "NET", "EBAY", "WPM", "CVE", 
#         "WEC", "RJF", "BRO", "ROK", "CSGP", "HEI.A", "WTW", "FITB", "WDS", "CHT", "BCE", "FER", "PPG", 
#         "TSCO", "LI", "HUBS", "CCL", "ETR", "ANSS", "TTWO", "ZS", "LYB", "ERIC", "DXCM", "EQR", "FCNCA", 
#         "RBLX", "K", "NVR", "FCNCO", "STT", "MTD", "VTR", "TW", "IOT", "BNTX", "LYV", "BEKE", "PHM", "TEF", 
#         "ADM", "TPL", "DOV", "UAL", "AWK", "HPE", "BIIB", "KEYS", "TYL", "GPN", "FNV", "CAH", "CDW", "SW",
#         "NOK", "IFF", "DECK", "BBD", "DTE", "CVNA", "KB", "VLTO", "GIB", "FTV", "DVN", "STM", "HOOD", "SBAC", 
#         "TROW", "BR", "LDOS", "CHD", "PHG", "VOD", "IX", "HAL", "NTAP", "FE", "PBA", "TECK", "CQP", "PPL", 
#         "TU", "NTR", "ERIE", "ILMN", "CCJ", "BAH", "ES", "HUBB", "AEE", "WY", "CPAY", "ZM", "WDC", "EQT", 
#         "HBAN", "GDDY", "QSR", "ROL", "WST", "BAM", "PTC"]
#     asset_data = []

#     for ticker in tickers:
#         url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=5min&apikey={AV_API_KEY}"
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             time_series = data.get('Time Series (5min)', {})
#             if time_series:
#                 latest_time = sorted(time_series.keys(), reverse=True)[0]
#                 latest_data = time_series[latest_time]
#                 asset_data.append({
#                     "Ticker": ticker,
#                     "Latest Close": float(latest_data['4. close']),
#                     "Open Price": float(latest_data['1. open']),
#                     "Price Change (%)": ((float(latest_data['4. close']) - float(latest_data['1. open'])) / float(latest_data['1. open'])) * 100
#                 })
#     return asset_data

def fetch_and_save_stock_data():
    AV_API_KEY = os.getenv("AV_API_KEY")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Example tickers
    data_frames = []
>>>>>>> Stashed changes

    for ticker in tickers:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={AV_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.rename(columns=lambda x: x[3:], inplace=True)
            df['Ticker'] = ticker
            data_frames.append(df)

    if data_frames:
        full_data = pd.concat(data_frames)
        full_data.to_csv('stock_data.csv', mode='w')  # Write to CSV, overwrite existing data
        print("Data saved to CSV.")
    else:
        print("No data fetched.")

def display_asset_charts():
    if st.session_state['asset_data'] and all('Ticker' in asset for asset in st.session_state['asset_data']):
        asset_data = st.session_state['asset_data']
        tickers = [asset['Ticker'] for asset in asset_data]
        selected_ticker = st.selectbox("Select a ticker to view price chart:", tickers)
        stock = yf.Ticker(selected_ticker)
        hist = stock.history(period="1mo")
        st.line_chart(hist['Close'])
    else:
        st.error("Asset data not properly loaded or 'Ticker' key missing.")

<<<<<<< Updated upstream
# # Function to display assets in a table with user preferences
# def display_assets():
#     st.header("Asset Data")
#     if 'preferred_assets' not in st.session_state:
#         st.session_state['preferred_assets'] = []

#     # Check if asset data is available
#     if st.session_state['asset_data']:
#         asset_data = st.session_state['asset_data']
#         df = pd.DataFrame(asset_data)
#         st.dataframe(df)

#         # Allow user to select preferred assets
#         tickers = df['Ticker'].tolist()
#         selected_assets = st.multiselect(
#             "Select your preferred assets",
#             tickers,
#             default=st.session_state['preferred_assets']
#         )
#         st.session_state['preferred_assets'] = selected_assets

#         if selected_assets:
#             st.write("Your preferred assets:")
#             preferred_df = df[df['Ticker'].isin(selected_assets)]
#             st.dataframe(preferred_df)
#             check_price_alerts()
#     else:
#         st.info("Asset data not loaded. Click 'Update Stock Prices' to load.")

=======
>>>>>>> Stashed changes
def display_assets():
    st.header("Asset Data")
    if 'preferred_assets' not in st.session_state:
        st.session_state['preferred_assets'] = []

<<<<<<< Updated upstream
    # Check if asset data is available
    if st.session_state['asset_data']:
        asset_data = st.session_state['asset_data']
        df = pd.DataFrame(asset_data)

        # Convert 'Current Price' to numeric values for filtering
        df['Current Price'] = df['Current Price'].replace('[\$,]', '', regex=True).astype(float)

        # Add input fields for price range selection
        min_price = st.number_input(
            "Minimum Stock Price ($)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )
        max_price = st.number_input(
            "Maximum Stock Price ($)",
            min_value=0.0,
            value=1000.0,
            step=1.0
        )
        # After filtering the DataFrame
        df_filtered = df[(df['Current Price'] >= min_price) & (df['Current Price'] <= max_price)]

        # Store the filtered DataFrame in session state for use in other functions
        st.session_state['df_filtered'] = df_filtered

        # Filter the DataFrame based on the selected price range
        df_filtered = df[(df['Current Price'] >= min_price) & (df['Current Price'] <= max_price)]

        st.dataframe(df_filtered)

        # Allow user to select preferred assets from the filtered list
        tickers = df_filtered['Ticker'].tolist()
        selected_assets = st.multiselect(
            "Select your preferred assets",
            tickers,
            default=st.session_state['preferred_assets']
        )
        st.session_state['preferred_assets'] = selected_assets

        if selected_assets:
            st.write("Your preferred assets:")
            preferred_df = df_filtered[df_filtered['Ticker'].isin(selected_assets)]
            st.dataframe(preferred_df)
            check_price_alerts(df_filtered)
    else:
        st.info("Asset data not loaded. Click 'Update Stock Prices' to load.")


# # Function to check for price alerts
# def check_price_alerts():
#     if 'preferred_assets' in st.session_state and st.session_state['preferred_assets']:
#         alert_threshold = st.slider(
#             "Set price change alert threshold (%)",
#             min_value=0.0,
#             max_value=10.0,
#             value=5.0
#         )
#         if st.session_state['asset_data']:
#             asset_data = st.session_state['asset_data']
#             for asset in asset_data:
#                 if asset['Ticker'] in st.session_state['preferred_assets']:
#                     price_change = float(asset['Price Change (Today)'].strip('%'))
#                     if abs(price_change) >= alert_threshold:
#                         st.warning(f"Alert: {asset['Ticker']} has changed by {price_change:.2f}% today!")
#         else:
#             st.info("Asset data not loaded. Please update stock prices to check price alerts.")
def check_price_alerts(df_filtered):
=======
# Function to check for price alerts
def check_price_alerts():
>>>>>>> Stashed changes
    if 'preferred_assets' in st.session_state and st.session_state['preferred_assets']:
        alert_threshold = st.slider(
            "Set price change alert threshold (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0
        )
        for _, asset in df_filtered.iterrows():
            if asset['Ticker'] in st.session_state['preferred_assets']:
                price_change = float(asset['Price Change (Today)'].strip('%'))
                if abs(price_change) >= alert_threshold:
                    st.warning(f"Alert: {asset['Ticker']} has changed by {price_change:.2f}% today!")


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
# def format_asset_suggestions(suggestions):
#     if not suggestions:
#         return "No assets currently meet the criteria for recommendation."
#     suggestion_text = "Here are some asset suggestions based on recent performance:\n\n"
#     for asset in suggestions:
#         suggestion_text += (
#             f"**{asset['Ticker']}**\n"
#             f"- Price Change (Today): {asset['Price Change (Today)']}\n"
#             f"- Dividend Yield: {asset['Dividend Yield']}\n"
#             f"- Current Price: {asset['Current Price']}\n\n"
#         )
#     return suggestion_text
def format_asset_suggestions(suggestions):
    if not suggestions:
        return "No assets currently meet the criteria for recommendation."
    suggestion_text = "Here are some asset suggestions based on your criteria:\n\n"
    for _, asset in suggestions.iterrows():
        suggestion_text += (
            f"**{asset['Ticker']}**\n"
            f"- Price Change (Today): {asset['Price Change (Today)']}\n"
            f"- Dividend Yield: {asset['Dividend Yield']}\n"
            f"- Current Price: ${asset['Current Price']:.2f}\n\n"
        )
    return suggestion_text


# # Function to generate response from OpenAI
# def generate_response(financial_data, user_message, vector_store):
#     if st.session_state['asset_data']:
#         asset_suggestions = st.session_state['asset_data']
#         formatted_suggestions = format_asset_suggestions(asset_suggestions)
#     else:
#         formatted_suggestions = "No asset data available."

#     query = financial_data + "\n" + user_message
#     docs = vector_store.similarity_search(query, k=3) if vector_store else []
#     context = "\n".join([doc.page_content for doc in docs])

#     prompt = f"""
#     Based on the user's financial data, the following asset suggestions, and the context from documents:

#     Financial Data:
#     {financial_data}

#     Asset Suggestions:
#     {formatted_suggestions}

#     Context from documents:
#     {context}

#     User Message:
#     {user_message}

#     Provide a helpful and informative response as a personal finance assistant. Consider the user's financial data, asset suggestions, and the context from the documents in your response. Include prices of top movers in stocks based on the data you have.
#     """

#     # Generate response from OpenAI
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a financial assistant that provides advice based on the user's data and market trends. Always ensure that your advice is appropriate for the user's financial situation."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     response = completion.choices[0].message.content

#     return response

def generate_response(financial_data, user_message, vector_store, df_filtered):
    if not df_filtered.empty:
        formatted_suggestions = format_asset_suggestions(df_filtered)
    else:
        formatted_suggestions = "No asset data available within the specified price range."

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


# # Function to handle the chat interface
# def chat_interface():
#     st.header("Chat with Your Personal Finance Assistant")

#     # Display the chat messages in order
#     for message in st.session_state['chat_history']:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
#             # If there's a chart associated with this message, display it
#             if 'chart_data' in message:
#                 st.line_chart(message['chart_data'])

#     # Get User Input
#     user_input = st.chat_input("You:")
#     if user_input:
#         financial_data = st.session_state['financial_data']
#         vector_store = st.session_state['vector_store']

#         # Generate assistant's response
#         response = generate_response(financial_data, user_input, vector_store)

#         # Check if the user is asking for the price of an asset
#         chart_data = display_chart_for_asset(user_input)

#         # Add user message to chat history
#         st.session_state['chat_history'].append({"role": "user", "content": user_input})

#         # Add assistant's message and chart data to chat history
#         assistant_message = {"role": "assistant", "content": response}
#         if chart_data is not None:
#             assistant_message['chart_data'] = chart_data
#         st.session_state['chat_history'].append(assistant_message)

#         # Display the last two messages (user and assistant)
#         for message in st.session_state['chat_history'][-2:]:
#             with st.chat_message(message['role']):
#                 st.markdown(message['content'])
#                 if 'chart_data' in message:
#                     st.line_chart(message['chart_data'])
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

        # Get the filtered assets based on the price range from display_assets()
        if 'df_filtered' in st.session_state:
            df_filtered = st.session_state['df_filtered']
        else:
            df_filtered = pd.DataFrame()

        # Generate assistant's response
        response = generate_response(financial_data, user_input, vector_store, df_filtered)

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


# Function for the budgeting tool
def budgeting_tool():
    st.header("Budgeting Tool")
    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        step=100.0,
        help="Your total monthly income after taxes."
    )
    expenses = st.number_input(
        "Monthly Expenses",
        min_value=0.0,
        step=100.0,
        help="Your total monthly expenses."
    )
    savings = income - expenses
    if savings >= 0:
        st.success(f"Monthly Savings: ${savings:.2f}")
    else:
        st.error(f"Monthly Deficit: ${-savings:.2f}")


# Function to get top movers within the top 500 coins
def fetch_top_crypto_gainers_losers():
    AV_API_KEY = os.getenv("AV_API_KEY")
    cryptos = ["BTC", "ETH", "LTC"]  # Example cryptocurrencies
    crypto_data = []

<<<<<<< Updated upstream
    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)

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
        print(e)
        return []

# Function to get cryptocurrency prices
def get_crypto_prices():
    response = requests.get(
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
    )
    data = response.json()
    return data
=======
    for crypto in cryptos:
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={crypto}&market=USD&apikey={AV_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            time_series = data.get('Time Series (Digital Currency Daily)', {})
            if time_series:
                latest_time = sorted(time_series.keys(), reverse=True)[0]
                latest_data = time_series[latest_time]
                close_price = float(latest_data['4a. close (USD)'])
                open_price = float(latest_data['1a. open (USD)'])
                price_change = ((close_price - open_price) / open_price) * 100
                crypto_data.append({
                    "Crypto": crypto,
                    "Close Price": close_price,
                    "Open Price": open_price,
                    "Price Change (%)": price_change
                })

    # Sort to find top gainers and losers
    top_gainers = sorted(crypto_data, key=lambda x: x["Price Change (%)"], reverse=True)[:5]
    top_losers = sorted(crypto_data, key=lambda x: x["Price Change (%)"])[:5]

    return top_gainers, top_losers
>>>>>>> Stashed changes


# Main App

# Add a welcome header with an emoji
st.markdown("# Welcome to Your Personal Finance Assistant 💰")

# Load asset data on initial app load
if not st.session_state['asset_data']:
    with st.spinner('Loading stock prices...'):
        st.session_state['asset_data'] = fetch_and_save_stock_data()
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
        fetch_and_save_stock_data()
        with st.spinner("Updating stock prices..."):
            st.session_state['asset_data'] = fetch_top_crypto_gainers_losers()
            st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.success("Stock prices updated and saved to CSV.")


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
    crypto_data = fetch_top_crypto_gainers_losers()
    if crypto_data:
        df_crypto = pd.DataFrame(crypto_data)
        st.dataframe(df_crypto)
    else:
        st.write("Failed to retrieve cryptocurrency prices.")


with tab3:
    chat_interface()

with tab4:
    budgeting_tool()
