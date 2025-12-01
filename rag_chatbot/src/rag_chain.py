"""
RAG Chain Module

Provides helpers to build prompts and manage conversation memory

"""
import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_file = os.getenv("CONFIG_YAML_PATH", "config.yaml")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


class RAGChatbot:
    """Main RAG chatbot orchestrator with conversation memory."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize RAG chatbot.
        
        Args:
            session_id: Existing session ID to resume, or None for new session
        """
        self.config = load_config()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the assistant."""
        return """You are an AI assistant for Acme Corp's HR policies. Your role is to help employees understand company policies by answering questions based on the provided context.

Guidelines:
- Answer based ONLY on the provided context. You may refer to previous messages for clarity in case the user query is not clear.
- If information is not in the context, say "I don't have that information in the HR policy documents"
- Be concise and professional
- Use bullet points for lists
- Cite specific policy sections when relevant
- Reword the context properly so it looks more professional and less like copy-pasting

Remember: You are helpful, accurate, and only provide information from official HR documents."""
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            section = doc['metadata'].get('section_title', 'Unknown')
            text = doc['document']
            context_parts.append(f"[Source {i} - {section}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self, 
        query: str, 
        context: str, 
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Build complete prompt with system message, history, and context."""
        
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # Add conversation history (limit to last 6 messages)
        if conversation_history:
            messages.extend(conversation_history[-6:])
        if len(messages)>2:
            messages.pop(0)
        # Add current query with context
        user_message = f"""Context from HR documents:
{context}

Question: {query}"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    

