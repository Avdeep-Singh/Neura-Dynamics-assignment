from src import config
from src.agent import app


def test_config():
    """Prints the loaded configuration variables to verify them."""
    print("--- Verifying Configuration ---")
    print(f"OpenWeatherMap API Key Loaded: {bool(config.OPENWEATHERMAP_API_KEY)}")
    print(f"LangChain API Key Loaded:    {bool(config.LANGCHAIN_API_KEY)}")
    print(f"OpenAI API Key Loaded:       {bool(config.OPENAI_API_KEY)}")
    print(f"LLM Model:                   {config.LLM_MODEL_NAME}")
    print(f"Embedding Model:             {config.EMBEDDING_MODEL_NAME}")
    print(f"Qdrant URL:                  {config.QDRANT_STORAGE_PATH}")
    print(f"PDF Path:                    {config.PDF_PATH}")
    print("-----------------------------")


def run_agent():
    """
    Runs the LangGraph agent with sample queries to test both paths.
    """
    print("\n--- Testing Weather Path ---")
    weather_query = "Delhi"
    print(f"Query: {weather_query}")
    
    # The input to the graph is a dictionary with keys matching the AgentState
    final_state = app.invoke({"question": weather_query})
    
    print("\n--- Final Answer ---")
    print(final_state['answer'])
    print("----------------------\n")

    print("\n--- Testing RAG Path ---")
    rag_query = "tell me about avdeep's experience? in 2 lines"
    print(f"Query: {rag_query}")
    
    final_state = app.invoke({"question": rag_query})
    
    print("\n--- Final Answer ---")
    print(final_state['answer'])
    print("--------------------")
    

if __name__ == "__main__":
    # First, verify the configuration
    #test_config()
    
    # from src import pdf_processor
    # pdf_processor.process_and_store_pdf() # Run this once to create the DB
    
    
    # Finally, run the agent
    run_agent()