import os
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_or_load_vectorstore(force_recreate=False):
    """
    Create or load a FAISS vector store from local PDFs or existing index.
    For demonstration, we just do an in-memory approach. 
    In production, you'd handle persistence, etc.
    """
    if not force_recreate and "vector_store" in st.session_state and st.session_state['vector_store']:
        return st.session_state['vector_store']

    # Example: process PDFs in /data
    vector_store = build_vectorstore_from_folder("data")
    return vector_store

def build_vectorstore_from_folder(data_folder: str):
    """
    A simple example of reading local PDFs from a folder,
    splitting text, embedding with OpenAI, and storing in a FAISS index in-memory.
    """
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in .env.")
        return None

    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(data_folder, filename)
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "".join(page.extract_text() or "" for page in reader.pages)
                    pdf_texts.append(text)
            except Exception as e:
                st.warning(f"Error processing {filename}: {e}")
    if not pdf_texts:
        return None

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for txt in pdf_texts:
        docs.extend(text_splitter.split_text(txt))

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store

def similarity_search_docs(vector_store, query, k=3):
    """
    Simple wrapper for similarity search if you want it.
    """
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)
