import streamlit as st
import requests
import uuid

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 RAG Chatbot")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Session controls
with st.sidebar:
    st.header("Session Info")
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption(f"Total messages: {len(st.session_state.messages)}")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about policies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show spinner OUTSIDE the chat message container
    with st.spinner("Thinking..."):
        try:
            # Make POST request to FastAPI backend
            response = requests.post(
                f"{FASTAPI_URL}/chats",
                json={
                    "session_id": st.session_state.session_id,
                    "user_message": prompt
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_message = data["bot_message"]
            else:
                bot_message = f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            bot_message = "⚠️ Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000"
        except requests.exceptions.Timeout:
            bot_message = "⏱️ Request timed out. Please try again."
        except Exception as e:
            bot_message = f"❌ An error occurred: {str(e)}"
    
    # Display assistant response AFTER getting the response
    with st.chat_message("assistant"):
        st.markdown(bot_message)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_message
    })
    
    # Rerun to refresh the chat display
    st.rerun()
