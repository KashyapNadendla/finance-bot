import streamlit as st
import re
import yfinance as yf
import stocks

def generate_response(financial_data, user_message, vector_store, openai_client):
    if st.session_state.get('asset_data'):
        asset_suggestions = st.session_state['asset_data']
        formatted_suggestions = stocks.format_asset_suggestions(asset_suggestions)
    else:
        formatted_suggestions = "No asset data available."
    query = financial_data + "\n" + user_message
    if vector_store:
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
    else:
        context = ""

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
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial assistant providing advice based on user data and market trends."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response

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

def chat_interface(openai_api_key):
    from openai import OpenAI
    openai_client = OpenAI(api_key=openai_api_key)

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
        response = generate_response(financial_data, user_input, vector_store, openai_client)

        chart_data = display_chart_for_asset(user_input)
        st.session_state['chat_history'].append({"role": "user", "content": user_input})

        assistant_message = {"role": "assistant", "content": response}
        if chart_data is not None:
            assistant_message['chart_data'] = chart_data
        st.session_state['chat_history'].append(assistant_message)

        for msg in st.session_state['chat_history'][-2:]:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if 'chart_data' in msg:
                    st.line_chart(msg['chart_data'])
