from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src import config

def get_llm():
    """Initializes and returns the ChatGroq LLM."""
    return ChatGroq(
        temperature=0.2,
        model_name=config.LLM_MODEL_NAME,
        api_key=config.GROQ_API_KEY
    )

def get_decider_chain():
    """
    Creates a chain that decides which tool to use based on the user's question.
    Returns 'weather' or 'rag'.
    """
    llm = get_llm()
    
    prompt_template = """
    Given the user's question, classify it as either 'weather' or 'rag'.
    - If the question is about weather, temperature, climate, or a specific city's forecast, respond with 'weather'.
    - If the question is about the Avdeep or Avdeep's resume like his experienc, skill set, job role, etc., respond with 'rag'

    Do not respond with any other words, just 'weather' or 'rag'.

    User question: {question}
    Classification:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    return prompt | llm | StrOutputParser()

def get_processing_chain():
    """
    Creates a chain that generates a final answer based on the context from a tool and the original question.
    """
    llm = get_llm()
    
    prompt_template = """
    You are a helpful assistant. Based on the following context, answer the user's question concisely.
    If the context is an error message, inform the user about the error.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    return prompt | llm | StrOutputParser()