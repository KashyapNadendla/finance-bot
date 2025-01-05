import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# Local imports
from data_fetcher import fetch_all_assets
from news import display_finance_news
from chat import chat_interface
from budgeting import budgeting_tool
from vectorstore_utils import create_or_load_vectorstore
from data_fetcher import get_top_movers
from technical_analysis import parse_technical_indicators
#from data_fetcher import AV_API_KEY

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/Users/apple/Desktop/PG/Summer-24/Fintech/finance-bot/.env")  
AV_API_KEY = os.getenv("AV_API_KEY")

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

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

# Main Title
st.markdown("# Welcome to Your Personal Finance Assistant 💰")

# Load or create vectorstore on startup if desired (optional)
if st.session_state['vector_store'] is None:
    # Attempt to load or create from local data
    st.session_state['vector_store'] = create_or_load_vectorstore()

# Button to load all asset data from Alpha Vantage
col1, col2 = st.columns([8, 2])
with col2:
    if st.session_state['asset_data_timestamp']:
        st.write(f"**Data updated as of:** {st.session_state['asset_data_timestamp']}")
    else:
        st.write("**Data not loaded.**")
    if st.button("Update Data"):
        with st.spinner("Fetching assets from Alpha Vantage..."):
            st.session_state['asset_data'] = fetch_all_assets()
            st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.success("Asset data updated successfully!")

# Sidebar
with st.sidebar:
    st.header("User Settings")

    # Process Documents
    if st.button("Process Documents for Vector Store"):
        with st.spinner('Processing documents...'):
            st.session_state['vector_store'] = create_or_load_vectorstore(force_recreate=True)
            st.success("Vector store created/updated.")

    # Financial Data Input
    st.header("Enter Your Financial Data")
    with st.form("financial_data_form"):
        st.write("Please provide your financial data.")
        financial_data_input = st.text_area(
            "Financial Data",
            value=st.session_state['financial_data'],
            height=150,
            help="Enter any financial info you want the bot to consider."
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state['financial_data'] = financial_data_input
            st.success("Financial data updated.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["News", "Assets", "Chat", "Tools"])

# 1) News
with tab1:
    display_finance_news()

# 2) Assets
with tab2:
    st.header("Asset Data")
    if st.session_state['asset_data']:
        df = pd.DataFrame(st.session_state['asset_data'])
        st.dataframe(df)

        # Simple technical indicator parse or usage
        st.subheader("Technical Indicators Example")
        if st.button("Parse Tech Indicators for 1st Asset"):
            first_ticker = df.iloc[0]['Ticker']
            st.write(f"Parsing indicators for {first_ticker} ...")
            indicators_dict = parse_technical_indicators(first_ticker)
            st.write(indicators_dict)
    else:
        st.info("No asset data loaded. Click 'Update Data' at the top right to load data.")

    # Crypto top movers
    st.subheader("Top Cryptocurrency Movers (24h Change)")
    top_movers = get_top_movers()
    if top_movers:
        st.dataframe(pd.DataFrame(top_movers))
    else:
        st.write("Failed to retrieve cryptocurrency data.")

# 3) Chat
with tab3:
    chat_interface()

# 4) Tools
with tab4:
    budgeting_tool()

st.write("\n\n---\n\n**Alpha Vantage API Key Used:**", AV_API_KEY)
