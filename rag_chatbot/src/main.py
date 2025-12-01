# FastAPI RAG Chatbot

import os
import sys
import json
import logging
import yaml
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.chats import ChatRequest, ChatResponse, EndSessionRequest 
from utils.query_processor import QueryProcessor
from utils.retriever import HybridRetriever
from utils.azure_client import AzureOpenAIChatClient
from rag_chain import RAGChatbot

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_file = os.getenv("CONFIG_YAML_PATH", "config.yaml")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    
app = FastAPI()
config = load_config()
base_dir = Path(os.getcwd())
data_folder = base_dir / config['paths']['data_folder']
conversations = os.path.join(data_folder, "conversation.json")
retriever = HybridRetriever()
chat_client = AzureOpenAIChatClient()
query_processor = QueryProcessor()
chatbot = RAGChatbot()

# Ensure data folder exists
Path(data_folder).mkdir(parents=True, exist_ok=True)


def load_conversation(session_id: str) -> Dict:
    """Load conversation from JSON file or create new one."""
    logger.info(f"Loading conversation for session_id: {session_id}")
    logger.info(f"Checking conversations file at: {conversations}")
    if os.path.exists(conversations):
        with open(conversations, "r") as f:
            try: 
                data = json.load(f)
                # If session_id matches, return existing conversation
                if data.get("session_id") == session_id:
                    return data
            except json.JSONDecodeError:
                logger.warning("Conversations file is empty or corrupted. Creating new conversation.")
    # Create new conversation
    return {
        "session_id": session_id,
        "messages": []
    }


def save_conversation(conversation: Dict):
    """Save conversation to JSON file."""
    with open(conversations, "w") as f:
        json.dump(conversation, f, indent=2)



@app.post("/chats", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat requests:
    1. Create or load conversation with session_id
    2. Add user message to conversation
    3. Get bot response
    4. Save updated conversation
    5. Return response
    """
    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())
        print("session_id:", session_id)
        # Load existing conversation or create new one
        conversation = load_conversation(session_id)

        logger.info(f"Loaded conversation for session_id: {session_id}")
        logger.info(f"Conversation has {len(conversation['messages'])} messages")
        logger.info(f"User message: {request.user_message}")
        logger.info(f"conversation: {conversation}")
        # Extract messages
        conversation_history = conversation.get("messages", [])

        # Rephrase user message if needed
        rephrased_queries = query_processor.process_query(request.user_message, conversation_history)

        logger.info("Rephrased queries:", rephrased_queries)
        context = []
        # Retrieve relevant chunks for all rephrased queries
        for query in rephrased_queries:
        #     retrieved_docs = retriever.get_relevant_documents(query)
        #     context = "\n".join([doc['content'] for doc in retrieved_docs])
            embs = retriever.embedding_client.create_embeddings([query])
            # print("embs (raw):", type(embs), len(embs))
            # print("embs[0] type/len:", type(embs[0]), len(embs[0]))
            # print("first 5 values:", embs[0][:5])
            query_embedding = embs[0]
            semantic_results = retriever.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=5  # get more because we will fuse later
                            )
            # print(f"Semantic results for '{query}': {semantic_results}")
            # dummy_embedding = [0.0] * retriever.collection.dimension
            keyword_results = retriever.collection.query(
                query_embeddings=[query_embedding],
                query_texts=[query],
                n_results=1 # BM25 keyword search
            )
            # print(f"Keyword results for '{query}': {keyword_results}")
            context.append(semantic_results["documents"])
            context.append(keyword_results["documents"])

        print("Final context:", context)
        # Add user message to messages
        user_msg_dict = {"role": "user", "content": request.user_message}
        
        # Step 5: Build prompt
        messages = chatbot._build_prompt(request.user_message, context, conversation_history)

        #messages.append(user_msg_dict)
        logger.info(f"Updated messages with user message. Total messages: {len(messages)}")
        logger.info(f"Messages: {messages}")

        bot_response = chat_client.complete(
                messages=messages,
                temperature=config['generation']['temperature'],
                max_tokens=config['generation']['max_tokens'],
                top_p=config['generation']['top_p']
            )

        # Add bot response to messages
        bot_msg_dict = {"role": "assistant", "content": bot_response}
        messages.append(bot_msg_dict)
        logger.info(f"Messages: {messages}")

        # Update conversation
        conversation["messages"] = messages
        conversation["session_id"] = session_id

        # Save to file
        save_conversation(conversation)

        # Return response
        return ChatResponse(
            session_id=session_id,
            bot_message=bot_response
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Chat API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)