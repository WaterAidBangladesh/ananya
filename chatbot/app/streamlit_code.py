import streamlit as st
import os
import dotenv

from langchain_groq import ChatGroq

from app import Chain

dotenv.load_dotenv()

# App title and page config
st.set_page_config(page_title="Probahini")
st.title("Probahini - Menstrual Assistant")

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
)

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# Function for clearing chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function for generating response
def generate_response(user_input):
    # Add the prompt history to construct the dialogue
    dialogue = "You are a helpful assistant."
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        dialogue += f"\n{role}: {msg['content']}"
    dialogue += f"\nUser: {user_input}\nAssistant:"
    # Call the ChatGroq model
    bot_response = llm.invoke(dialogue)
    return bot_response.content


# User input
if user_input := st.chat_input(placeholder="Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

# Generate a response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = Chain()
            response = chain.get_response(st.session_state.messages[-1]["content"], 123)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Ensure the "Clear Chat History" button only appears once at the bottom
if st.session_state.messages[-1]["role"] == "assistant":
    if st.button("Clear Chat History", key="clear_button"):
        clear_chat_history()
        st.rerun()
