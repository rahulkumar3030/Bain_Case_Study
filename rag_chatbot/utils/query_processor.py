"""
Query Processor Module

It performs following 2 tasks
task 1: Rephrase/Contextualize vague queries using conversation history
task 2: Decompose multi-question queries into single standalone queries

Usage:
    processor = QueryProcessor()
    
    # With history
    history = [
        {"role": "user", "content": "What is leave policy?"},
        {"role": "assistant", "content": "20 days annual leave..."}
    ]
    queries = processor.process_query("How many sick days?", history)
"""

import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

from .azure_client import AzureOpenAIChatClient

load_dotenv()
logger = logging.getLogger(__name__)


class QueryProcessor:
    """Two-stage query processor: contextualize then decompose."""
    
    def __init__(self):
        """Initialize query processor."""
        self.chat_client = AzureOpenAIChatClient()
        logger.info("Initialized QueryProcessor")
    
    def process_query(
        self, 
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Process query through contextualization and decomposition.
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            
        Returns:
            List of standalone sub-queries
        """
        contextualized_query = self._contextualize_query(query, conversation_history) #if query is less than 10 words we contextualize it using previous messages for better context
        sub_queries = self._decompose_query(contextualized_query) #break down queries into multiple sub-queries if needed (to be used for multiple questions in single query)
        
        logger.info(f"Processed query into {len(sub_queries)} standalone queries")
        return sub_queries
    
    def _contextualize_query(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Rephrase query to be standalone using conversation context."""
        if not conversation_history or len(query.split()) > 10:
            return query
        
        try:
            system_prompt = """You are a query contextualizer. Rephrase the query to be standalone and complete using conversation history.

Rules:
- If query is vague, add context from history
- If query is already complete, return unchanged
- Return ONLY the rephrased query

Examples:
History: User asks about leave policy
Query: "How many sick days?"
Output: How many sick days does Acme Corp provide?"""

            messages = [{"role": "system", "content": system_prompt}]
            
            if conversation_history:
                messages.extend(conversation_history[-2:]) # Use last 2 messages for context
            
            messages.append({"role": "user", "content": query})
            
            response = self.chat_client.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=100
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Contextualization failed: {e}")
            return query
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose query into multiple sub-queries if needed."""
        try:
            system_prompt = """Split query into multiple questions if needed. Return numbered list.

Examples:
Input: "What is leave policy?"
Output:
1. What is leave policy?

Input: "What is leave policy and dress code?"
Output:
1. What is our leave policy?
2. What is our dress code?"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.chat_client.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            
            queries = []

            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    query = line.split('.', 1)[-1].strip()
                    if query:
                        queries.append(query)
            
            if not queries:
                queries = [response.strip()]
            
            return queries
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return [query]


if __name__ == "__main__":
    """Test query processor."""
    processor = QueryProcessor()
    
    # Test cases: (input_query, conversation_history, description)
    test_cases = [
        (
            "Are there any types/optional ones?",
            [
                {"role": "user", "content": "What is leave policy at ACME Corp?"},
                {"role": "assistant", "content": "we provide 20 days annual leave"}
            ]
        ),
        (
            "How many paid leaves can you take and what is our remote working policy?",
            None
        ),
        (
            "What are our employee wellness initiatives?",
            None
        )
    ]
    
    for query, history in test_cases:
        print(f"Input: {query}")
        result = processor.process_query(query, history)
        print(f"Output: {result}")
        print("-" * 50)
