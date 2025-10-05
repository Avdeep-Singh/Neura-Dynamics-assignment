import pytest
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document

from src import config
from src.vector_db import create_collection_if_not_exists, upsert_documents, query_collection

@pytest.fixture
def in_memory_client():
    """Fixture to create an in-memory Qdrant client for testing."""
    return QdrantClient(":memory:")

def test_db_operations(in_memory_client):
    """Tests the full cycle of collection creation, upsert, and query."""
    client = in_memory_client
    
    # 1. Test collection creation
    original_collection_name = config.QDRANT_COLLECTION_NAME
    config.QDRANT_COLLECTION_NAME = "test_collection"
    
    create_collection_if_not_exists(client)
    collection_info = client.get_collection(collection_name="test_collection")
    
    # The API for accessing vector params has changed in qdrant-client
    assert collection_info.config.params.vectors.size == config.EMBEDDING_MODEL_DIMENSION # <-- CHANGED

    # 2. Test upsert
    mock_docs = [
        Document(
            page_content="The Eiffel Tower is in Paris.",
            metadata={"embedding": [0.1] * config.EMBEDDING_MODEL_DIMENSION}
        )
    ]
    upsert_documents(client, mock_docs)
    assert client.count(collection_name="test_collection", exact=True).count == 1

    # 3. Test query
    query_embedding = [0.11] * config.EMBEDDING_MODEL_DIMENSION # A similar vector
    results = query_collection(client, query_embedding)
    
    assert len(results) == 1
    assert results[0] == "The Eiffel Tower is in Paris."
    
    # Cleanup: restore original config value
    config.QDRANT_COLLECTION_NAME = original_collection_name