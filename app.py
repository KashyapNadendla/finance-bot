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
API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client instance
client = OpenAI(api_key=API_KEY)

# Initialize session state variables
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

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
            except PyPDF2.errors.PdfReadError:
                st.warning(f"Warning: '{filename}' could not be processed (EOF marker not found). Skipping.")
            except Exception as e:
                st.warning(f"Warning: An error occurred while processing '{filename}': {e}. Skipping.")
    return pdf_texts

# Function to create a vector store from texts
def create_vector_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]

    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    vector_store = Chroma.from_texts(
        texts_chunks,
        embeddings,
        collection_name="financial_assistant_collection",
        persist_directory="chroma_db"
    )
    vector_store.persist()
    return vector_store

def scout_assets():
    tickers = [
        "AAPL", "GOOGL", "GOOG", "AMZN", "META", "BRK.A", "TSLA", "AVGO", "WMT", "JPM",
        "V", "HD", "ABBV", "NFLX", "CRM", "ASML", "MRK", "TMUS", "AMD", "PEP",
        "CSCO", "ADBE", "TMO", "PM", "NOW", "CAT", "ISRG", "INTU", "VZ", "GS",
        "AMGN", "PDD", "UBER", "DIS", "NVDA", "MSFT", "UNH", "LLY", "TXN", "PG",
        "MA", "CMCSA", "ABT", "NEE", "COST", "XOM", "CVX", "PYPL", "BABA", "BAC"
    ]
    
    # List to store asset information
    asset_data = []
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        
        try:
            # Fetch the latest data with a 1-day interval for the most recent price
            hist = stock.history(period="1d", interval="1m")
            
            # Check if data was retrieved successfully
            if hist.empty:
                st.warning(f"No data found for {ticker}")
                continue
            
            # Get the most recent close price and calculate price change
            latest_close = hist['Close'].iloc[-1]
            open_price = hist['Close'].iloc[0]
            price_change_today = ((latest_close - open_price) / open_price) * 100
            
            # Fetch additional details like dividend yield
            dividend_yield = stock.info.get("dividendYield", "N/A")
            
            # Append data to the list
            asset_data.append({
                "Ticker": ticker,
                "Current Price": f"${latest_close:.2f}",
                "Dividend Yield": f"{dividend_yield:.2%}" if dividend_yield != "N/A" else "N/A",
                "Price Change (Today)": f"{price_change_today:.2f}%"
            })
        
        except Exception as e:
            st.warning(f"Error retrieving data for {ticker}: {e}")
    
    return asset_data


# Example of calling the function and displaying in Streamlit
st.header("Asset Data for All Tickers")
asset_data = scout_assets()
for asset in asset_data:
    st.write(asset)


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
            f"- Current Price: {asset['Current Price']}\n\n"
        )
    return suggestion_text

def generate_response(financial_data, user_message, vector_store):
    # Always call scout_assets to get asset suggestions
    asset_suggestions = scout_assets()
    formatted_suggestions = format_asset_suggestions(asset_suggestions)

    # Generate the query and retrieve relevant documents
    query = financial_data + "\n" + user_message
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Construct the prompt with asset suggestions, document context, and user input
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

    Provide a helpful and informative response as a personal finance assistant. Consider the user's financial data, asset suggestions, and the context from the documents in your response. Include Prices of top movers in Stocks based on the data you have.
    """

    # Generate response from OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and knowledgeable personal finance assistant. You can help people make rational Financial Decisions."},
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
    response = generate_response(financial_data, user_input, vector_store) if vector_store else "I'm sorry, but I couldn't access the knowledge base documents."
    st.session_state['chat_history'].append({"user": user_input, "bot": response})

    # Display latest messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)