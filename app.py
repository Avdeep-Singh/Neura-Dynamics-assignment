import streamlit as st
from src.agent import app

# --- Page Configuration ---
st.set_page_config(
    page_title="LangGraph RAG & Weather Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Page Title and Description ---
st.title("🤖 LangGraph RAG & Weather Agent")
st.info(""" This is a simple chat interface for an agent that can answer questions about a PDF document (i have used my resume) and provide real-time weather information. 
        \n Try asking 'Tell me about Avdeep in 2 lines'
        \n OR
        \n Try city name for wether report 'Delhi'
        """)

# --- Session State for Chat History ---
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
# Display existing messages from the history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
# Accept user input via the chat input box
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Get Agent's Response ---
    # Display a loading spinner while the agent is thinking
    with st.spinner("Thinking..."):
        # The input to the graph is a dictionary with keys matching the AgentState
        inputs = {"question": prompt}
        
        # Invoke the agent
        final_state = app.invoke(inputs)
        
        # Extract the final answer
        response = final_state.get("answer", "Sorry, I encountered an error.")

    # Add agent's response to session state and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)