import streamlit as st
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


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
                st.write(f"Processed {filename}")
            except PyPDF2.errors.PdfReadError:
                st.error(f"Warning: '{filename}' could not be processed. Skipping.")
            except Exception as e:
                st.error(f"Warning: Error processing '{filename}': {e}. Skipping.")
    return pdf_texts

def load_single_pdf(uploaded_file):
    """Helper to process a single uploaded PDF."""
    pdf_texts = []
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        pdf_texts.append(text)
    except Exception as e:
        st.error(f"Warning: Error processing PDF: {e}")
    return pdf_texts

def create_vector_store(texts, openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = [chunk for text in texts for chunk in text_splitter.split_text(text)]
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts_chunks, embeddings)
    return vector_store
