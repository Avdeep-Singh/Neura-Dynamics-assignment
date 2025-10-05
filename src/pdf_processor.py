from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from typing import List

from src import config
from src import vector_db

def load_and_chunk_pdf(file_path: str) -> List:
    """
    Loads a PDF from the given path and splits it into chunks.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        List: A list of document chunks.
    """
    print(f"Loading and chunking PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"PDF split into {len(chunks)} chunks.")
    return chunks

def process_and_store_pdf():
    """
    The main function to run the PDF processing and storage pipeline.
    """
    print("--- Starting PDF Ingestion Pipeline ---")
    
    # 1. Initialize Qdrant Client and create collection
    qdrant_client = vector_db.get_qdrant_client()
    vector_db.create_collection_if_not_exists(qdrant_client)
    
    # 2. Load and chunk the PDF document
    doc_chunks = load_and_chunk_pdf(config.PDF_PATH)
    
    # 3. Initialize the embedding model
    embeddings_model = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
        model=config.EMBEDDING_MODEL_NAME
    )
    
    # 4. Generate embeddings for each chunk and store them in the document metadata
    print("Generating embeddings for document chunks...")
    for chunk in doc_chunks:
        # The embedding is stored in metadata to be easily passed to Qdrant
        chunk.metadata['embedding'] = embeddings_model.embed_query(chunk.page_content)
    
    # 5. Upsert the documents with their embeddings into Qdrant
    vector_db.upsert_documents(qdrant_client, doc_chunks)
    
    print("--- PDF Ingestion Pipeline Finished ---")

    ## 4. Generate embeddings for each chunk and store them in the document metadata
    #print("Generating embeddings for document chunks...")
    ## Get all page contents at once for efficiency
    #all_page_contents = [chunk.page_content for chunk in doc_chunks]
    #
    ## Embed all chunks in a single batch call
    #all_embeddings = embeddings_model.embed_documents(all_page_contents)
    #
    ## Assign the generated embeddings back to each chunk
    #for i, chunk in enumerate(doc_chunks):
    #    chunk.metadata['embedding'] = all_embeddings[i]
    #
    ## 5. Upsert the documents with their embeddings into Qdrant
    #vector_db.upsert_documents(qdrant_client, doc_chunks)
    #
    #print("--- PDF Ingestion Pipeline Finished ---")