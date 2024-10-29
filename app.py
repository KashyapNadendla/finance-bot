import streamlit as st
import openai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# Initialize session state variables
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to generate response from OpenAI
def generate_response(financial_data, user_message):
    prompt = f"""
    Based on the user's financial data provided:

    Financial Data:
    {financial_data}

    User Message:
    {user_message}

    Provide a helpful and informative response as a personal finance assistant. Consider the user's financial data in your response.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and knowledgeable personal finance assistant. Use the user's financial data to provide personalized advice."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Section to input financial data
st.header("Enter Your Financial Data")

with st.form("financial_data_form"):
    st.write("Please provide your financial data.")
    financial_data_input = st.text_area("Financial Data", value=st.session_state['financial_data'], height=200)
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state['financial_data'] = financial_data_input
        st.success("Financial data updated.")

# Chat interface
st.header("Chat with Your Personal Finance Assistant")

# Display chat history
if 'chat_history' in st.session_state and st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        with st.chat_message("assistant"):
            st.markdown(chat['bot'])
        with st.chat_message("user"):
            st.markdown(chat['user'])

# Get user input
user_input = st.chat_input("You:")

if user_input:
    financial_data = st.session_state['financial_data']
    response = generate_response(financial_data, user_input)

    # Append the user input and bot response to the chat history
    st.session_state['chat_history'].append({"user": user_input, "bot": response})

    # Display the latest messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
