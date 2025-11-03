"""
OpenAI utility functions for congressional hearing parsing.
Provides centralized OpenAI client management and API call functions.
"""

import time
import json
import logging
from openai import OpenAI
from pathlib import Path
import sys

# Add parent directory to path for config access
local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path)/ 'config.json')

# Model name constants
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GPT_5 = "gpt-5"
GPT_5_MINI = "gpt-5-mini"

# Default model
DEFAULT_MODEL = GPT_5_MINI

# Define supported models and their capabilities
SUPPORTED_MODELS = {
    GPT_4O: {"supports_temperature": True},
    GPT_4O_MINI: {"supports_temperature": True},
    GPT_5: {"supports_temperature": False},
    GPT_5_MINI: {"supports_temperature": False},
}

# Global client instance
_client = None

def get_openai_client():
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        config = load_config(add_date=False, config_path=Path(__file__).parents[1] / 'config.json')
        _client = OpenAI(api_key=config['open_ai_key'])
    return _client

def call_openai_api(messages, model=DEFAULT_MODEL, temperature=0.1, system_message=None):
    """
    Make an OpenAI API call with proper error handling and logging.
    
    Args:
        messages: List of message dictionaries for the API
        model: OpenAI model to use
        temperature: Temperature for the API call (only used if model supports it)
        system_message: Optional system message to prepend to messages
    
    Returns:
        Tuple of (success: bool, response_content: str or None, error: str or None)
    """
    client = get_openai_client()
    
    # Prepare the API call parameters
    api_params = {
        "model": model,
        "messages": messages,
    }
    
    
    # Only add temperature if the model supports it
    if SUPPORTED_MODELS[model]["supports_temperature"]:
        api_params["temperature"] = temperature
    
    # Add system message if provided
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    
    t = [len(m['content'].split()) for m in messages]
    if len(t) == 1:
        t = t[0]
    logger.info(f"prompt size(s): {t} words")
    try:
        logger.info(f"Calling {model} for API request")
        start_time = time.time()
        response = client.chat.completions.create(**api_params)
        
        # Parse the response
        response_content = response.choices[0].message.content.strip()
        end_time = time.time()
        execution_time = end_time - start_time
        
        if execution_time > 60:
            logger.error(f"Response from {model} took {int(execution_time // 60)} minutes and {int(round(execution_time % 60))} seconds")
        else:
            logger.info(f"Response from {model} took {int(round(execution_time))} seconds")
        
        return True, response_content, None
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return False, None, str(e)

def call_openai_with_json_response(messages, model=DEFAULT_MODEL, temperature=0.1, system_message=None):
    """
    Make an OpenAI API call and parse JSON response.
    
    Args:
        messages: List of message dictionaries for the API
        model: OpenAI model to use
        temperature: Temperature for the API call (only used if model supports it)
        system_message: Optional system message to prepend to messages
    
    Returns:
        Tuple of (success: bool, parsed_json: dict or None, error: str or None)
    """
    success, response_content, error = call_openai_api(messages, model, temperature, system_message)
    
    if not success:
        return False, None, error
    
    try:
        # Extract JSON from the response (in case there's extra text)
        parsed_json = json.loads(response_content)
        return True, parsed_json, None
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Raw response: {response_content}")
        return False, None, f"JSON parsing failed: {e}"
## bracket classification moved to fix_unparsed.py
