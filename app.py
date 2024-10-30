import streamlit as st
from openai import OpenAI
import os
import PyPDF2
from io import BytesIO
import yfinance as yf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# Load environment variables from the .env file
load_dotenv()

# Set up OpenAI client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize session state variables
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None  # Initially, no vector store is loaded

# Function to load and process PDFs from the data folder with error handling
def load_and_process_pdfs(data_folder):
    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page_num in range(len(reader.pages)):
                        page_text = reader.pages[page_num].extract_text()
                        if page_text:  # Ensure text extraction was successful
                            text += page_text
                    pdf_texts.append(text)
            except PyPDF2.errors.PdfReadError:
                st.warning(f"Warning: '{filename}' could not be processed (EOF marker not found). Skipping.")
            except Exception as e:
                st.warning(f"Warning: An error occurred while processing '{filename}': {e}. Skipping.")
    return pdf_texts


# Function to create a vector store from texts
def create_vector_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = []
    for text in texts:
        texts_chunks.extend(text_splitter.split_text(text))

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = Chroma.from_texts(
        texts_chunks,
        embeddings,
        collection_name="financial_assistant_collection",
        persist_directory="chroma_db"
    )
    vector_store.persist()
    return vector_store

# Function to scout assets with real-time price action from Yahoo Finance
def scout_assets():
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "JNJ", "META"]
    suggestions = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        
        # Fetch the latest data with a 1-minute interval to get the most recent price
        hist = stock.history(period="1d", interval="1m")
        
        # Only proceed if we have recent data
        if not hist.empty:
            latest_close = hist['Close'].iloc[-1]  # Get the most recent close price
            price_change = (latest_close - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
            dividend_yield = stock.info.get("dividendYield", 0)

            # Adding a recommendation criterion based on price change or dividend yield
            if price_change > 0.02 or dividend_yield > 0.02:  # Adjust threshold as needed
                suggestions.append({
                    "Ticker": ticker,
                    "Price Change (Today)": f"{price_change:.2%}",
                    "Dividend Yield": f"{dividend_yield:.2%}" if dividend_yield else "N/A",
                    "Current Price": f"${latest_close:.2f}"
                })

    return suggestions


# Function to format asset suggestions as text
def format_asset_suggestions(suggestions):
    if not suggestions:
        return "No assets currently meet the criteria for recommendation."
    suggestion_text = "Here are some asset suggestions based on recent performance:\n\n"
    for asset in suggestions:
        suggestion_text += (
            f"**{asset['Ticker']}**\n"
            f"- Price Change (1y): {asset['Price Change (1y)']}\n"
            f"- Dividend Yield: {asset['Dividend Yield']}\n"
            f"- Current Price: {asset['Current Price']}\n\n"
        )
    return suggestion_text

# Enhanced generate_response function with improved real-time data handling
def generate_response(financial_data, user_message, vector_store):
    # Detect if the user is requesting real-time data or stock recommendations
    if "fetch data" in user_message.lower() or "suggest assets" in user_message.lower() or "real-time stock prices" in user_message.lower():
        # Call scout_assets to fetch the latest stock data
        asset_suggestions = scout_assets()
        
        # Format the suggestions into a user-friendly response
        response = format_asset_suggestions(asset_suggestions)
        
        # Check if there are no suggestions based on criteria, and inform the user accordingly
        if not asset_suggestions:
            response = "Currently, no assets meet the criteria for recommendation based on recent performance and dividend yield."
        
    else:
        # Default response using financial data and context from documents
        query = financial_data + "\n" + user_message
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Based on the user's financial data and the following context:

        Financial Data:
        {financial_data}

        Context from documents:
        {context}

        User Message:
        {user_message}

        Provide a helpful and informative response as a personal finance assistant. Consider the user's financial data and the context from the documents in your response.
        """
        
        # Generate response from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable personal finance assistant. Use the user's financial data and the provided context to provide personalized advice."},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content

    return response


# UI Section for Processing Documents
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

# Section for Financial Data Input
st.header("Enter Your Financial Data")
with st.form("financial_data_form"):
    st.write("Please provide your financial data.")
    financial_data_input = st.text_area("Financial Data", value=st.session_state['financial_data'], height=200)
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state['financial_data'] = financial_data_input
        st.success("Financial data updated.")

# Chat Interface
st.header("Chat with Your Personal Finance Assistant")
if st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        with st.chat_message("assistant"):
            st.markdown(chat['bot'])
        with st.chat_message("user"):
            st.markdown(chat['user'])

# Get User Input
user_input = st.chat_input("You:")
if user_input:
    financial_data = st.session_state['financial_data']
    vector_store = st.session_state['vector_store']
    if vector_store is not None:
        response = generate_response(financial_data, user_input, vector_store)
    else:
        response = "I'm sorry, but I couldn't access the knowledge base documents."
    
    # Append chat history
    st.session_state['chat_history'].append({"user": user_input, "bot": response})

    # Display latest messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
