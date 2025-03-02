import streamlit as st
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
        pass
    with col2:
        if st.session_state["asset_data_timestamp"]:
            st.write(f"**Stock prices updated as of:** {st.session_state['asset_data_timestamp']}")
        else:
            st.write("**Stock prices not loaded.**")

        # When user clicks the button, always fetch fresh data from Alpha Vantage
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
                data_folder = "data"
                pdf_texts = pdf_processing.load_and_process_pdfs(data_folder)
                if pdf_texts:
                    st.session_state["vector_store"] = pdf_processing.create_vector_store(pdf_texts, API_KEY)
                    st.success("Documents processed and vector store created.")
                else:
                    st.warning("No PDF documents found in the 'data' folder.")

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
            st.write("Please provide your financial data.")
            financial_data_input = st.text_area(
                "Financial Data",
                value=st.session_state["financial_data"],
                height=200,
                help="Enter any financial information you would like the assistant to consider."
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state["financial_data"] = financial_data_input
                st.success("Financial data updated.")

    # ---------------------- MAIN TABS ---------------------- #
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["News", "Assets", "Chat", "Tools", "Agentic Advisor"])

    # 1) NEWS TAB
    with tab1:
        news.display_finance_news()

    # 2) ASSETS TAB
    with tab2:
        stocks.display_assets()
        st.subheader("Asset Price Chart")
        stocks.display_asset_charts()

        st.subheader("Top Cryptocurrency Movers (24h Change)")
        cmc_key = st.session_state.get("CMC_API_KEY", None)
        crypto_data = crypto.get_top_movers(cmc_key)
        if crypto_data:
            df_crypto = pd.DataFrame(crypto_data)
            st.dataframe(df_crypto)
        else:
            st.write("Failed to retrieve cryptocurrency prices.")

    # 3) CHAT TAB
    with tab3:
        chat.chat_interface(API_KEY)

    # 4) TOOLS / BUDGETING TAB
    with tab4:
        budgeting.budgeting_tool()

    # 5) AGENTIC ADVISOR TAB
    with tab5:
        agentic.agentic_chat_interface()

# ---------------------- LAUNCH ---------------------- #
if __name__ == "__main__":
    main()
