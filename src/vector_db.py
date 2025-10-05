import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, UpdateStatus
from langchain_core.documents import Document
from typing import List
import uuid

from src import config

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns the Qdrant client using a local file path.
    Ensures the storage directory exists.
    """
    # Ensure the storage directory for Qdrant exists
    os.makedirs(config.QDRANT_STORAGE_PATH, exist_ok=True)
    
    # Initialize the client with the local path
    return QdrantClient(path=config.QDRANT_STORAGE_PATH)

#def create_collection_if_not_exists(client: QdrantClient):
#    """
#    Creates the Qdrant collection if it doesn't already exist.
#    """
#    try:
#        client.get_collection(collection_name=config.QDRANT_COLLECTION_NAME)
#        print(f"Collection '{config.QDRANT_COLLECTION_NAME}' already exists.")
#    except Exception:
#        print(f"Collection '{config.QDRANT_COLLECTION_NAME}' not found. Creating...")
#        client.recreate_collection( # Use recreate_collection for simplicity in local setup
#            collection_name=config.QDRANT_COLLECTION_NAME,
#            vectors_config=models.VectorParams(
#                size=config.EMBEDDING_MODEL_DIMENSION,
#                distance=models.Distance.COSINE
#            ),
#        )
#        print("Collection created successfully.")

def create_collection_if_not_exists(client: QdrantClient):
    """
    Creates the Qdrant collection if it doesn't already exist.
    """
    # Use the collection_exists method to check
    if not client.collection_exists(collection_name=config.QDRANT_COLLECTION_NAME):
        print(f"Collection '{config.QDRANT_COLLECTION_NAME}' not found. Creating...")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_MODEL_DIMENSION,
                distance=models.Distance.COSINE
            ),
        )
        print("Collection created successfully.")
    else:
        print(f"Collection '{config.QDRANT_COLLECTION_NAME}' already exists.")

def upsert_documents(client: QdrantClient, documents: List[Document]):
    """
    Upserts documents into the Qdrant collection.
    Each document is converted into a Qdrant PointStruct.
    """
    points = []
    for doc in documents:
        # Each document must have a unique ID. We'll use UUIDs.
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=doc.metadata['embedding'], # We will store the embedding here
            payload={
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
        )
        points.append(point)

    operation_info = client.upsert(
        collection_name=config.QDRANT_COLLECTION_NAME,
        wait=True,
        points=points
    )
    
    if operation_info.status == UpdateStatus.COMPLETED:
        print(f"Successfully upserted {len(points)} documents.")
    else:
        print(f"Error upserting documents: {operation_info.status}")

def query_collection(client: QdrantClient, query_embedding: List[float], top_k: int = 3) -> List[str]:
    """
    Performs a similarity search on the Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client instance.
        query_embedding (List[float]): The embedding vector of the query.
        top_k (int): The number of top results to retrieve.

    Returns:
        List[str]: A list of page contents from the retrieved documents.
    """
    search_result = client.search(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True  # Ensure we get the payload which contains our text
    )
    
    # Extract the 'page_content' from the payload of each result
    retrieved_contents = [point.payload['page_content'] for point in search_result]
    return retrieved_contents

