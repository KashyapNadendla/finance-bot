import streamlit as st
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

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
import crypto
import analysis
import agentic

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
    
    # ---------------------- SIDEBAR ---------------------- #
    with st.sidebar:
        st.header("User Settings")
        # 1) Process Documents for Vector Store
        st.header("Process Documents for Vector Store")
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                data_folder = "data"  # or "temp" if you're using a temporary folder
                pdf_texts = pdf_processing.load_and_process_pdfs(data_folder)
                if pdf_texts:
                    st.session_state["vector_store"] = pdf_processing.create_vector_store(pdf_texts, API_KEY)
                    st.success("Documents processed and vector store created.")
                else:
                    st.warning("No PDF documents found in the specified folder.")
        # 2) Upload Your Own Documents
        st.header("Upload Your Own Documents")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            pdf_texts = []
            for uploaded_file in uploaded_files:
                pdf_texts.extend(pdf_processing.load_single_pdf(uploaded_file))
            if pdf_texts:
                st.session_state["vector_store"] = pdf_processing.create_vector_store(pdf_texts, API_KEY)
                st.success("Documents uploaded and processed.")
        # 3) Enter Your Financial Data
        st.header("Enter Your Financial Data")
        with st.form("financial_data_form"):
            st.write("Provide any additional financial information for the assistant.")
            financial_data_input = st.text_area(
                "Financial Data",
                value=st.session_state["financial_data"],
                height=200,
                help="Enter financial information, such as income, expenses, or personal notes."
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state["financial_data"] = financial_data_input
                st.success("Financial data updated.")
    
    # ---------------------- MAIN TABS ---------------------- #
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["News", "Assets", "Crypto", "Budgeting", "Agentic Advisor"])
    
    # 1) NEWS TAB
    with tab1:
        st.header("Latest Finance News")
        articles = news.fetch_finance_news()
        if articles:
            for art in articles:
                st.markdown(f"[**{art['title']}**]({art['url']}) - {art['source']}")
        else:
            st.write("No news available.")
    
    # 2) ASSETS TAB
    # with tab2:
    #     st.header("Stock Data")
    #     if st.session_state.get("asset_data"):
    #         df = pd.DataFrame(st.session_state["asset_data"])
    #         st.dataframe(df)
    #         st.write(f"Data last updated: {st.session_state['asset_data_timestamp']}")
    #     else:
    #         st.write("No stock data available.")
    with tab2:
        st.header("Stock Data")
        if st.session_state.get("asset_data"):
            df = pd.DataFrame(st.session_state["asset_data"])
            st.dataframe(df)
            st.write(f"Data last updated: {st.session_state['asset_data_timestamp']}")
        
        # Asset Graph Feature
            tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
            selected_ticker = st.selectbox("Select an asset to view its price chart:", tickers)
            tech_mode = st.checkbox("Technical Analysis Mode")
            if selected_ticker:
            # Fetch historical data using yfinance
                import yfinance as yf
                stock = yf.Ticker(selected_ticker)
                hist = stock.history(period="1y", interval="1d")
                if hist.empty:
                    st.write("No historical data available for this asset.")
                else:
                    if tech_mode:
                    # Simple technical analysis: plot close price and SMA (e.g., 50-day)
                        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
                        st.line_chart(hist[["Close", "SMA50"]])
                    else:
                        st.line_chart(hist["Close"])
        else:
            st.write("No stock data available.")

    # 3) CRYPTO TAB
    with tab3:
        st.header("Crypto Data")
        crypto_data = crypto.fetch_crypto_data()
        if crypto_data:
            df_crypto = pd.DataFrame(crypto_data)
            st.dataframe(df_crypto)
        else:
            st.write("No crypto data available.")
    
    # 4) BUDGETING TAB
    with tab4:
        budgeting.budgeting_tool()
    
    # 5) AGENTIC ADVISOR TAB
    # with tab5:
    #     st.header("Agentic Advisor")
    #     user_query = st.text_input("Enter your investment query:")
    #     if st.button("Get Advice"):
    #         # Here we pass along extra parameters if needed; agentic_advisor will use them as necessary.
    #         advice = agentic.agentic_advisor(user_query, tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    #                                          alpha_vantage_api_key=ALPHA_VANTAGE_API_KEY)
    #         st.subheader("Investment Advice:")
    #         st.write(advice)
    # In the Agentic Advisor tab, for example:
    with tab5:
        st.header("Agentic Advisor")
        user_query = st.text_input("Enter your investment query:")
        deep_research = st.checkbox("Deep Research Mode")
        macro_mode = st.checkbox("Macroeconomics Mode")  # New toggle for macro reports

        if st.button("Get Advice"):
        # If macro mode is on, generate and display the macro report.
            if macro_mode:
                macro_data = analysis.fetch_macro_indicators()
                macro_report = analysis.generate_macro_report(macro_data)
                st.subheader("Macroeconomic Report:")
                st.write(macro_report)
        
        # Then, get the investment advice (optionally also adjusted based on deep mode)
            advice = agentic.agentic_advisor(
                user_query,
                deep_mode=deep_research,
                tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY")
            )
            st.subheader("Investment Advice:")
            st.write(advice)


# ---------------------- LAUNCH ---------------------- #
if __name__ == "__main__":
    main()
