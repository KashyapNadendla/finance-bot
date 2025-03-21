# import os
# from dotenv import load_dotenv
# import pinecone
# from pinecone import Pinecone
# from alpha_vantage.timeseries import TimeSeries

# # Load API keys from .env
# from dotenv import load_dotenv
# load_dotenv()

# import os

# # Initialize Pinecone
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# pinecone.init(api_key=pinecone_api_key, environment="us-east-1-aws")

# index_name = "daily-market-data"

# # Check if index exists; if not, create it
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         name=index_name,
#         dimension=3072,  # dimension based on multilingual-e5-large embeddings
#         metric="cosine"
#     )

# index = pinecone.Index(index_name)

# # Sample placeholder - Replace with real-time data fetched from Alpha Vantage
# data = [
#     {'id': 'AAPL-2025-03-06', 'text': 'Apple stock closed at 150 USD with a high of 152 USD...'},
#     {'id': 'BTC-2025-03-06', 'text': 'Bitcoin closed at 45000 USD, volume increased by 10%...'},
#     # Additional data from forex, commodities, etc.
# ]

# # Generate embeddings using Pinecone inference
# embeddings_response = pinecone.embed(
#     model="multilingual-e5-large",
#     inputs=[item['text'] for item in data],
#     parameters={"input_type": "passage"}
# )

# vectors = [
#     {
#         "id": item['id'],
#         "values": embedding,
#         "metadata": {'text': item['text']}
#     }
#     for item, embedding in zip(data, embeddings['embeddings'])
# ]

# # Upsert data into Pinecone
# pinecone_index = pinecone.Index(index_name)
# pinecone.upsert(vectors=vectors, namespace="market_data")
# pinecone_client.py

from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone, ServerlessSpec

# Get your Pinecone API key, region, and index name from environment variables.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENV", "us-east-1-aws")
INDEX_NAME = os.getenv("PINECONE_INDEX", "finance-index")

# Create an instance of the Pinecone client.
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists; if not, create it.
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# Retrieve the index.
index = pc.Index(INDEX_NAME)

def create_index(index_name=INDEX_NAME, dimension=768):
    """
    Returns the existing index object.
    """
    return index

def upsert_embeddings(index, ids, vectors, metadata=None):
    """
    Upsert a batch of vectors into Pinecone.
    """
    to_upsert = []
    for i, vector in enumerate(vectors):
        item = {
            "id": ids[i],
            "values": vector,
            "metadata": metadata[i] if metadata else {}
        }
        to_upsert.append(item)
    index.upsert(vectors=to_upsert)

def query_embeddings(index, query_vector, top_k=5):
    """
    Query Pinecone for the closest vectors.
    """
    result = index.query(query_vector, top_k=top_k, include_metadata=True)
    return result
