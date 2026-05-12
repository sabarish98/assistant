"""LLM client for connecting to Ollama with error handling."""

from typing import Optional, Dict, Any
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import config
from core.logger import app_logger


class OllamaClient:
    """Robust Ollama client with connection handling and error recovery."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.ollama_model
        self.base_url = config.ollama_base_url
        self.temperature = config.ollama_temperature
        self._client: Optional[ChatOllama] = None
        
        app_logger.info(f"Initializing Ollama client with model: {self.model_name}")
    
    def _create_client(self) -> ChatOllama:
        """Create a new Ollama client instance."""
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=config.request_timeout
        )
    
    @property
    def client(self) -> ChatOllama:
        """Get or create the Ollama client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama and return status."""
        try:
            app_logger.info("Testing Ollama connection...")
            
            # Simple test message
            test_message = HumanMessage(content="Hello! Please respond with 'Connection successful'")
            
            response = await self.client.ainvoke([test_message])
            
            result = {
                "status": "success",
                "model": self.model_name,
                "base_url": self.base_url,
                "response": response.content,
                "temperature": self.temperature
            }
            
            app_logger.info(f"Connection test successful: {result}")
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "model": self.model_name,
                "base_url": self.base_url,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            app_logger.error(f"Connection test failed: {error_result}")
            return error_result
    
    def invoke(self, messages, **kwargs) -> Any:
        """Synchronous invoke with error handling."""
        try:
            return self.client.invoke(messages, **kwargs)
        except Exception as e:
            app_logger.error(f"Error invoking model: {e}")
            raise
    
    async def ainvoke(self, messages, **kwargs) -> Any:
        """Asynchronous invoke with error handling."""
        try:
            return await self.client.ainvoke(messages, **kwargs)
        except Exception as e:
            app_logger.error(f"Error in async invoke: {e}")
            raise
    
    def stream(self, messages, **kwargs):
        """Streaming response with error handling."""
        try:
            return self.client.stream(messages, **kwargs)
        except Exception as e:
            app_logger.error(f"Error in streaming: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout": config.request_timeout
        }


# Global client instance
ollama_client = OllamaClient()

__all__ = ["OllamaClient", "ollama_client"]