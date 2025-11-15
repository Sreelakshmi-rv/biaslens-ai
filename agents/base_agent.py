from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

class BaseAgent(ABC):
    """Base class for all BiasLens agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.api_key = os.getenv('GROQ_API_KEY')
        self.memory = {}
        
    @abstractmethod
    def execute(self, data_context: Dict[str, Any], user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main execution method to be implemented by each agent"""
        pass
    
    def update_memory(self, key: str, value: Any):
        """Update agent memory"""
        self.memory[key] = value
    
    def get_memory(self, key: str) -> Any:
        """Retrieve from agent memory"""
        return self.memory.get(key)
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate LLM response using Groq API directly"""
        if not self.api_key:
            return "AI insights feature requires a valid Groq API key. Please check your .env file configuration."
        
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant specialized in data analysis and fairness detection. Provide clear, concise insights."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "model": "llama-3.1-8b-instant",  # Working model
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API Error: {response.status_code} - {response.text}"
            
        except Exception as e:
            return f"AI insights temporarily unavailable. Error: {str(e)}"