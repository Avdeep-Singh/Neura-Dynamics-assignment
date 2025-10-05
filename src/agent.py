from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from src import llm_utils
from src import tool as tools


# Define the state for our graph
class AgentState(TypedDict):
    question: str
    context: str
    answer: str

# --- Node Functions ---

def decider_node(state: AgentState):
    """
    Determines the next step based on the user's question.
    This node populates the 'context' field with the classification.
    """
    print("--- Node: Decider ---")
    question = state["question"]
    decider_chain = llm_utils.get_decider_chain()
    # The result of this chain will be 'weather' or 'rag'
    result = decider_chain.invoke({"question": question})
    print(f"Decision: '{result}'")
    
    # We store the decision to be used in the conditional edge
    return {"context": result}

def weather_node(state: AgentState):
    """Calls the weather tool with the user's question."""
    print("--- Node: Weather Tool ---")
    question = state["question"]
    # For simplicity, we assume the question itself contains the city name
    weather_data = tools.get_weather_info.invoke(question)
    return {"context": weather_data}

def rag_node(state: AgentState):
    """Calls the RAG tool with the user's question."""
    print("--- Node: RAG Tool ---")
    question = state["question"]
    rag_context = tools.retrieve_pdf_context.invoke(question)
    return {"context": rag_context}

def llm_processor_node(state: AgentState):
    """Generates the final answer using the LLM."""
    print("--- Node: LLM Processor ---")
    context = state["context"]
    question = state["question"]
    
    processing_chain = llm_utils.get_processing_chain()
    final_answer = processing_chain.invoke({"context": context, "question": question})
    
    return {"answer": final_answer}

# --- Conditional Edge Logic ---

def decide_next_node(state: AgentState) -> str:
    """The conditional logic that routes to the correct tool."""
    print("--- Conditional Edge: Routing ---")
    # The decision was stored in the 'context' field by the decider_node
    if "weather" in state["context"].lower():
        print("Routing to: Weather Node")
        return "weather_node"
    else:
        print("Routing to: RAG Node")
        return "rag_node"

# --- Graph Definition and Compilation ---

# Create a new graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("decider", decider_node)
workflow.add_node("weather_node", weather_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("llm_processor_node", llm_processor_node)

# Set the entry point
workflow.set_entry_point("decider")

# Add the conditional edge from the decider
workflow.add_conditional_edges(
    "decider",
    decide_next_node,
    {
        "weather_node": "weather_node",
        "rag_node": "rag_node"
    }
)

# Add the edges from the tool nodes to the final processor node
workflow.add_edge("weather_node", "llm_processor_node")
workflow.add_edge("rag_node", "llm_processor_node")

# The final node connects to the END
workflow.add_edge("llm_processor_node", END)

# Compile the graph into a runnable app
app = workflow.compile()

print("LangGraph agent compiled successfully!")