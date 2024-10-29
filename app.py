import streamlit as st
import openai
import os
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")

# Load environment variables from the .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state variables
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None  # Initially, no vector store is loaded

# Function to load and process PDFs from the data folder
def load_and_process_pdfs(data_folder):
    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                pdf_texts.append(text)
    return pdf_texts

# Function to create a vector store from texts
def create_vector_store(texts):
    # Split the texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = []
    for text in texts:
        texts_chunks.extend(text_splitter.split_text(text))

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Create vector store using Chroma
    vector_store = Chroma.from_texts(texts_chunks, embeddings)

    return vector_store

# Button to process documents
st.header("Process Documents for Vector Store")

if st.button("Process Documents"):
    with st.spinner('Processing documents...'):
        data_folder = 'data'  # Ensure your PDFs are in the 'data' folder
        pdf_texts = load_and_process_pdfs(data_folder)
        if pdf_texts:
            st.session_state['vector_store'] = create_vector_store(pdf_texts)
            st.success("Documents processed and vector store created.")
        else:
            st.warning("No PDF documents found in the 'data' folder.")

# Function to generate response from OpenAI with RAG
def generate_response(financial_data, user_message, vector_store):
    # Combine the financial data and user message
    query = financial_data + "\n" + user_message

    # Retrieve relevant documents from the vector store
    docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 relevant chunks

    # Combine the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])

    # Now create the prompt
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

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and knowledgeable personal finance assistant. Use the user's financial data and the provided context to provide personalized advice."},
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
if st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        with st.chat_message("assistant"):
            st.markdown(chat['bot'])
        with st.chat_message("user"):
            st.markdown(chat['user'])

# Get user input
user_input = st.chat_input("You:")

if user_input:
    financial_data = st.session_state['financial_data']
    vector_store = st.session_state['vector_store']
    if vector_store is not None:
        response = generate_response(financial_data, user_input, vector_store)
    else:
        response = "I'm sorry, but I couldn't access the knowledge base documents."

    # Append the user input and bot response to the chat history
    st.session_state['chat_history'].append({"user": user_input, "bot": response})

    # Display the latest messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
