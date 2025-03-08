import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone
from alpha_vantage.timeseries import TimeSeries

# Load API keys from .env
from dotenv import load_dotenv
load_dotenv()

import os

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(api_key=pinecone_api_key, environment="us-east-1-aws")

index_name = "daily-market-data"

# Check if index exists; if not, create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=3072,  # dimension based on multilingual-e5-large embeddings
        metric="cosine"
    )

index = pinecone.Index(index_name)

# Sample placeholder - Replace with real-time data fetched from Alpha Vantage
data = [
    {'id': 'AAPL-2025-03-06', 'text': 'Apple stock closed at 150 USD with a high of 152 USD...'},
    {'id': 'BTC-2025-03-06', 'text': 'Bitcoin closed at 45000 USD, volume increased by 10%...'},
    # Additional data from forex, commodities, etc.
]

# Generate embeddings using Pinecone inference
embeddings_response = pinecone.embed(
    model="multilingual-e5-large",
    inputs=[item['text'] for item in data],
    parameters={"input_type": "passage"}
)

vectors = [
    {
        "id": item['id'],
        "values": embedding,
        "metadata": {'text': item['text']}
    }
    for item, embedding in zip(data, embeddings['embeddings'])
]

# Upsert data into Pinecone
pinecone_index = pinecone.Index(index_name)
pinecone.upsert(vectors=vectors, namespace="market_data")
