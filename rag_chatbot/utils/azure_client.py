"""
Azure OpenAI Client Module

Provides wrapper classes for Azure OpenAI chat and embedding services.
Supports both synchronous and asynchronous API calls.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncAzureOpenAI


load_dotenv()
logger = logging.getLogger(__name__)


class AzureOpenAIChatClient:
    """
    Wrapper for Azure OpenAI Chat Completion API.
    Supports both sync and async calls.
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment: Optional[str] = None
    ):
        """Initialize Azure OpenAI Chat client."""
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        if not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError("Missing Azure OpenAI configuration in .env")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"Initialized AzureOpenAIChatClient (deployment: {self.deployment})")
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 500,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Synchronous chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Response content string or stream object
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return response
            
            content = response.choices[0].message.content
            logger.info(f"Chat completion successful ({len(content)} chars)")
            return content
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 500,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Asynchronous chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Response content string or stream object
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return response
            
            content = response.choices[0].message.content
            logger.info(f"Async chat completion successful ({len(content)} chars)")
            return content
            
        except Exception as e:
            logger.error(f"Async chat completion failed: {e}")
            raise


class AzureOpenAIEmbeddingClient:
    """
    Wrapper for Azure OpenAI Embeddings API.
    Supports both sync and async calls.
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment: Optional[str] = None
    ):
        """Initialize Azure OpenAI Embedding client."""
        self.endpoint = endpoint or os.getenv("AZURE_EMBEDDING_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_EMBEDDING_KEY")
        self.api_version = api_version or os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-10-21")
        self.deployment = deployment or os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        
        if not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError("Missing Azure Embedding configuration in .env")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"Initialized AzureOpenAIEmbeddingClient (deployment: {self.deployment})")
    
    def create_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Synchronous embedding creation.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                raise ValueError("texts list cannot be empty")
            
            response = self.client.embeddings.create(
                model=self.deployment,
                input=texts,
                **kwargs
            )
            
            embeddings = [item.embedding for item in response.data]
            #logger.debug(f"Created {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    async def create_embeddings_async(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Asynchronous embedding creation.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                raise ValueError("texts list cannot be empty")
            
            response = await self.async_client.embeddings.create(
                model=self.deployment,
                input=texts,
                **kwargs
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Async created {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
            return embeddings
            
        except Exception as e:
            logger.error(f"Async embedding creation failed: {e}")
            raise


if __name__ == "__main__":
    print("\n1. CHAT COMPLETION (SYNC)")
    print("-" * 70)
    try:
        chat_client = AzureOpenAIChatClient()
        messages = [
            {"role": "system", "content": "You are a helpful HR assistant."},
            {"role": "user", "content": "Explain employee onboarding process."}
        ]
        response = chat_client.complete(messages, temperature=0.3, max_tokens=500)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("-" * 70)
    print("\n2. EMBEDDINGS (SYNC)")
    print("-" * 70)
    try:
        embedding_client = AzureOpenAIEmbeddingClient()
        texts = [
            "Acme Corp has a generous leave policy",
            "Employees get 20 days of annual leave"
        ]
        embeddings = embedding_client.create_embeddings(texts)
        print(f"Created {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First 5 values: {embeddings[0][:5]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Async Usage
    # print("\n3. ASYNC USAGE EXAMPLE")
    # print("-" * 70)
#     print("""
# import asyncio
# from src.azure_client import AzureOpenAIChatClient, AzureOpenAIEmbeddingClient

# async def main():
#     # Async chat
#     chat_client = AzureOpenAIChatClient()
#     messages = [{"role": "user", "content": "Hello!"}]
#     response = await chat_client.complete_async(messages)
#     print(response)
    
#     # Async embeddings
#     embedding_client = AzureOpenAIEmbeddingClient()
#     embeddings = await embedding_client.create_embeddings_async(["text"])
#     print(len(embeddings))

# asyncio.run(main())
#     """)
    
#     print("\n" + "=" * 70)
#     print("Examples complete!")
#     print("=" * 70)
