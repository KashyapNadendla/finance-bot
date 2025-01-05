import streamlit as st
import re
import yfinance as yf
from openai import OpenAI
from vectorstore_utils import similarity_search_docs

def chat_interface():
    st.header("Chat with Your Personal Finance Assistant")

    # Display chat messages in session state
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'chart_data' in message:
                st.line_chart(message['chart_data'])

    user_input = st.chat_input("You:")
    if user_input:
        # The user's message
        st.session_state['chat_history'].append({"role": "user", "content": user_input})

        # Possibly fetch chart data if user requests: e.g. "price of TSLA"
        chart_data = display_chart_for_asset(user_input)

        # Generate an assistant response from OpenAI
        assistant_response = generate_assistant_response(user_input)
        assistant_message = {"role": "assistant", "content": assistant_response}

        if chart_data is not None:
            assistant_message['chart_data'] = chart_data

        st.session_state['chat_history'].append(assistant_message)

        # Display last two messages
        for msg in st.session_state['chat_history'][-2:]:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if 'chart_data' in msg:
                    st.line_chart(msg['chart_data'])

def display_chart_for_asset(message: str):
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
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
    return None

def generate_assistant_response(user_input: str) -> str:
    # Example logic:
    # 1) Do a vector store similarity search if needed
    # 2) Construct a prompt with the best context
    # 3) Call OpenAI's API for the final response
    # We'll do a mock response here:
    return "This is a placeholder response from OpenAI. (Integrate your 'generate_response' logic here.)"
