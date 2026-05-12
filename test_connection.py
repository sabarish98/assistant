"""Test Ollama connection and basic functionality."""

import asyncio
from core.config import config
from core.logger import app_logger
from core.llm_client import ollama_client


async def test_ollama_connection():
    """Test the Ollama connection and basic functionality."""
    
    app_logger.info("Starting Ollama connection test...")
    app_logger.info(f"Configuration: {config}")
    
    # Test connection
    connection_result = await ollama_client.test_connection()
    
    if connection_result["status"] == "success":
        print("✅ Ollama Connection Test PASSED")
        print(f"Model: {connection_result['model']}")
        print(f"Response: {connection_result['response']}")
        
        # Test model information
        model_info = ollama_client.get_model_info()
        print(f"\n📋 Model Configuration:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        return True
    else:
        print("❌ Ollama Connection Test FAILED")
        print(f"Error: {connection_result['error']}")
        print(f"Error Type: {connection_result['error_type']}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_connection())
    if success:
        print("\n🎉 All tests passed! Ready to proceed.")
    else:
        print("\n🚨 Connection failed. Please check Ollama is running and model is available.")