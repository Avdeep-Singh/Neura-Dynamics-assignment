import requests
from langchain.tools import tool
#from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.utilities import OpenWeatherMapAPIWrapper

from src import config
from src import vector_db

@tool
def get_weather_info(city: str) -> str:
    """
    Fetches the current weather for a specified city using the OpenWeatherMap API via LangChain's wrapper.
    Use this tool when asked about weather, temperature, or climate in a specific location.
    """
    try:
        # Initialize the wrapper with your API key
        weather_wrapper = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=config.OPENWEATHERMAP_API_KEY
        )
        print(f"--- Fetching weather for city: '{city}' ---")
        
        # Run the query for the city (wrapper handles the API call and formatting)
        result = weather_wrapper.run(city)
        
        return result
    
    except Exception as e:
        return f"An error occurred while fetching weather data: {str(e)}"
@tool
def retrieve_pdf_context(question: str) -> str:
    """
    Retrieves relevant context from the ingested PDF document based on a user's question.
    Use this tool for any questions about the content of the document, such as inquiries about the Eiffel Tower.
    """
    print(f"--- Retrieving context for question: '{question}' ---")
    
    # Initialize the embedding model
    embeddings_model = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
        model=config.EMBEDDING_MODEL_NAME
    )
    
    # Generate embedding for the user's question
    query_embedding = embeddings_model.embed_query(question)
    
    # Get the Qdrant client
    qdrant_client = vector_db.get_qdrant_client()
    
    # Query the collection
    retrieved_docs = vector_db.query_collection(qdrant_client, query_embedding)
    
    if not retrieved_docs:
        return "No relevant information found in the document for this question."
        
    # Combine the retrieved documents into a single context string
    context = "\n\n---\n\n".join(retrieved_docs)
    #print(f"--- Retrieved Context ---\n{context}\n-------------------------")
    return context

# A list of all tools for the agent to use
all_tools = [get_weather_info, retrieve_pdf_context]