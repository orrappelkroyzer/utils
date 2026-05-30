"""
OpenAI utility functions for congressional hearing parsing.
Provides centralized OpenAI client management and API call functions.
"""

import time
import json
import logging
import hashlib
import re
from openai import OpenAI
from pathlib import Path
import sys

# Add project root to path for config access
local_python_path = str(Path(__file__).parents[2])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path)/ 'config.json')

# Model name constants
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GPT_5 = "gpt-5"
GPT_5_5 = "gpt-5.5"
GPT_5_MINI = "gpt-5-mini"
GPT_5_4 = "gpt-5.4"
GPT_5_4_MINI = "gpt-5.4-mini"

# Default model
DEFAULT_MODEL = GPT_5_MINI

# Define supported models and their capabilities
SUPPORTED_MODELS = {
    GPT_4O: {"supports_temperature": True},
    GPT_4O_MINI: {"supports_temperature": True},
    GPT_5: {"supports_temperature": False},
    GPT_5_5: {"supports_temperature": False},
    GPT_5_MINI: {"supports_temperature": False},
    GPT_5_4: {"supports_temperature": False},
    GPT_5_4_MINI: {"supports_temperature": False},
}

# Global client instance
_client = None


def is_insufficient_quota_error(error):
    """Check whether an API exception indicates quota exhaustion."""
    msg = str(error).lower()
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


def fallback_models_for(model):
    """Return model fallback chain in priority order (includes original)."""
    chains = {
        GPT_5_5: [GPT_5_5, GPT_5_4, GPT_5_MINI, GPT_5_4_MINI, GPT_4O_MINI],
        GPT_5_4: [GPT_5_4, GPT_5_MINI, GPT_5_4_MINI, GPT_4O_MINI],
        GPT_5: [GPT_5, GPT_5_MINI, GPT_5_4_MINI, GPT_4O_MINI],
    }
    return chains.get(model, [model])

def get_openai_client():
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        config = load_config(add_date=False, config_path=Path(__file__).parents[2] / 'config.json')
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
    
    # Add system message if provided
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    
    t = [len(m['content'].split()) for m in messages]
    if len(t) == 1:
        t = t[0]
    logger.info(f"prompt size(s): {t} words")
    fallback_chain = fallback_models_for(model)
    last_error = None

    for candidate_model in fallback_chain:
        api_params = {
            "model": candidate_model,
            "messages": messages,
        }
        if SUPPORTED_MODELS.get(candidate_model, {}).get("supports_temperature", True):
            api_params["temperature"] = temperature

        try:
            logger.info(f"Calling {candidate_model} for API request")
            start_time = time.time()
            response = client.chat.completions.create(**api_params)

            # Parse the response
            response_content = response.choices[0].message.content.strip()
            end_time = time.time()
            execution_time = end_time - start_time

            if execution_time > 60:
                logger.error(f"Response from {candidate_model} took {int(execution_time // 60)} minutes and {int(round(execution_time % 60))} seconds")
            else:
                logger.info(f"Response from {candidate_model} took {int(round(execution_time))} seconds")

            return True, response_content, None
        except Exception as e:
            last_error = e
            should_fallback = is_insufficient_quota_error(e) and candidate_model != fallback_chain[-1]
            if should_fallback:
                next_model = fallback_chain[fallback_chain.index(candidate_model) + 1]
                logger.warning(
                    f"Model {candidate_model} failed with insufficient quota; "
                    f"falling back to {next_model}"
                )
                continue
            logger.error(f"OpenAI API call failed on {candidate_model}: {e}")
            return False, None, str(e)

    return False, None, str(last_error) if last_error else "OpenAI API call failed"

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

_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".json": "application/json",
}


def upload_file(file_path, purpose="user_data"):
    """Upload a file to OpenAI and return the file ID."""
    client = get_openai_client()
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    logger.info(f"Uploading {file_path.name} to OpenAI")

    # Read bytes first to avoid Windows file-streaming issues (OSError 22)
    # observed when httpx iterates file handles from cloud-synced directories.
    try:
        file_bytes = file_path.read_bytes()
    except Exception as e:
        raise OSError(f"Failed reading file bytes for upload: {file_path} ({e})") from e

    if not file_bytes:
        raise ValueError(f"Cannot upload empty file: {file_path}")

    mime = _MIME_BY_EXT.get(file_path.suffix.lower(), "application/octet-stream")

    max_attempts = 2
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.files.create(
                file=(file_path.name, file_bytes, mime),
                purpose=purpose
            )
            logger.info(f"Uploaded file ID: {response.id}")
            return response.id
        except Exception as e:
            last_error = e
            logger.warning(
                f"Upload attempt {attempt}/{max_attempts} failed for {file_path}: {e}"
            )
            if attempt < max_attempts:
                time.sleep(2)

    raise RuntimeError(f"Failed uploading file after {max_attempts} attempts: {file_path}") from last_error

def hash_file(file_path: Path) -> str:
    """Return SHA256 for cache keying."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def default_upload_cache_path() -> Path:
    return Path(local_python_path) / ".openai_upload_cache.json"

def load_upload_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Upload cache is invalid JSON, recreating: %s", cache_path)
        return {}

def save_upload_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def upload_file_cached(file_path, cache_key=None, force_reload=False, purpose="user_data", cache_path=None):
    """
    Upload a file and cache OpenAI file ID by content hash.
    Returns cached file_id when possible.
    """
    file_path = Path(file_path)
    cache_path = Path(cache_path) if cache_path else default_upload_cache_path()
    cache = load_upload_cache(cache_path)
    file_hash = hash_file(file_path)
    key = cache_key or str(file_path.resolve())
    cache_entry = cache.get(key)

    if (
        not force_reload
        and cache_entry
        and cache_entry.get("sha256") == file_hash
        and cache_entry.get("file_id")
    ):
        logger.info("Using cached OpenAI file ID for %s", key)
        return cache_entry["file_id"]

    file_id = upload_file(file_path=file_path, purpose=purpose)
    cache[key] = {
        "file_id": file_id,
        "sha256": file_hash,
        "source_path": str(file_path),
    }
    save_upload_cache(cache, cache_path)
    return file_id

def call_openai_with_file(file_id, prompt, model=DEFAULT_MODEL, temperature=0.1, system_message=None):
    """
    Call OpenAI API with a file reference (for PDFs, images, etc.).

    Returns:
        Tuple of (success: bool, response_content: str or None, error: str or None)
    """
    client = get_openai_client()

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_file", "file_id": file_id},
            {"type": "input_text", "text": prompt},
        ]
    })

    fallback_chain = fallback_models_for(model)
    last_error = None

    for candidate_model in fallback_chain:
        api_params = {"model": candidate_model, "input": messages, "max_output_tokens": 16384}
        if SUPPORTED_MODELS.get(candidate_model, {}).get("supports_temperature", True):
            api_params["temperature"] = temperature

        try:
            logger.info(f"Calling {candidate_model} with file {file_id}")
            start_time = time.time()
            response = client.responses.create(**api_params)
            response_content = response.output_text.strip()
            elapsed = time.time() - start_time
            if elapsed > 60:
                logger.error(f"Response took {int(elapsed // 60)}m {int(elapsed % 60)}s")
            else:
                logger.info(f"Response took {int(round(elapsed))}s")
            if getattr(response, 'status', None) == 'incomplete':
                reason = getattr(response, 'incomplete_details', None)
                logger.warning(f"Response was truncated (incomplete). Reason: {reason}")
            return True, response_content, None
        except Exception as e:
            last_error = e
            should_fallback = is_insufficient_quota_error(e) and candidate_model != fallback_chain[-1]
            if should_fallback:
                next_model = fallback_chain[fallback_chain.index(candidate_model) + 1]
                logger.warning(
                    f"Model {candidate_model} failed with insufficient quota; "
                    f"falling back to {next_model}"
                )
                continue
            logger.error(f"OpenAI API call with file failed on {candidate_model}: {e}")
            return False, None, str(e)

    return False, None, str(last_error) if last_error else "OpenAI API call with file failed"

def call_openai_with_files(
    file_ids,
    prompt,
    model=DEFAULT_MODEL,
    temperature=0.1,
    system_message=None,
    tools=None,
):
    """
    Call OpenAI Responses API with multiple uploaded files.

    Returns:
        Tuple of (success: bool, response_content: str or None, error: str or None)
    """
    client = get_openai_client()

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    user_content = [{"type": "input_file", "file_id": fid} for fid in file_ids]
    user_content.append({"type": "input_text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    api_params = {"model": model, "input": messages, "max_output_tokens": 16384}
    if SUPPORTED_MODELS.get(model, {}).get("supports_temperature", True):
        api_params["temperature"] = temperature
    if tools:
        api_params["tools"] = tools

    try:
        logger.info(f"Calling {model} with files {file_ids}")
        start_time = time.time()
        response = client.responses.create(**api_params)
        response_content = response.output_text.strip()
        elapsed = time.time() - start_time
        if elapsed > 60:
            logger.error(f"Response took {int(elapsed // 60)}m {int(elapsed % 60)}s")
        else:
            logger.info(f"Response took {int(round(elapsed))}s")
        if getattr(response, 'status', None) == 'incomplete':
            reason = getattr(response, 'incomplete_details', None)
            logger.warning(f"Response was truncated (incomplete). Reason: {reason}")
        return True, response_content, None
    except Exception as e:
        logger.error(f"OpenAI API call with files failed: {e}")
        return False, None, str(e)

def fix_invalid_json_escapes(s):
    """Replace invalid backslash escapes that GPT OCR output may contain."""
    import re
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

def call_openai_with_file_json(file_id, prompt, model=DEFAULT_MODEL, temperature=0.1, system_message=None):
    """Call OpenAI with a file and parse JSON from the response."""
    success, content, error = call_openai_with_file(file_id, prompt, model, temperature, system_message)
    if not success:
        return False, None, error
    try:
        json_str = content
        json_start = json_str.find('[')
        json_end = json_str.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            json_str = json_str[json_start:json_end]
        return True, json.loads(json_str), None
    except json.JSONDecodeError:
        try:
            fixed = fix_invalid_json_escapes(json_str)
            return True, json.loads(fixed), None
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to parse JSON: {e2}\nRaw: {content}")
            return False, None, f"JSON parsing failed: {e2}"

def call_openai_with_files_json(
    file_ids,
    prompt,
    model=DEFAULT_MODEL,
    temperature=0.1,
    system_message=None,
    tools=None,
):
    """Call OpenAI with multiple files and parse JSON from the response."""
    success, content, error = call_openai_with_files(
        file_ids=file_ids,
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_message=system_message,
        tools=tools,
    )
    if not success:
        return False, None, error
    try:
        return True, json.loads(content), None
    except json.JSONDecodeError:
        try:
            fixed = fix_invalid_json_escapes(content)
            return True, json.loads(fixed), None
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to parse JSON: {e2}\nRaw: {content}")
            return False, None, f"JSON parsing failed: {e2}"

def call_openai_with_json_prompt_file(
    prompt_file_path,
    file_ids=None,
    prompt_parameters=None,
    model=DEFAULT_MODEL,
    temperature=0.1,
    system_message=None,
    tools=None,
):
    """
    Load prompt text from file and call OpenAI for JSON output.

    Args:
        prompt_file_path: Path to a UTF-8 text file containing the prompt.
        file_ids: Optional list of uploaded OpenAI file IDs.
        prompt_parameters: Optional dict for replacing {{param_name}} placeholders.
        model: OpenAI model to use.
        temperature: Temperature for models that support it.
        system_message: Optional system message.
    """
    prompt_file_path = Path(prompt_file_path)
    if not prompt_file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

    prompt = prompt_file_path.read_text(encoding="utf-8").strip()
    if prompt_parameters:
        for parameter_name, parameter_value in prompt_parameters.items():
            placeholder = f"{{{{{parameter_name}}}}}"
            if not isinstance(parameter_value, str):
                parameter_value = json.dumps(parameter_value, ensure_ascii=False)
            prompt = prompt.replace(placeholder, parameter_value)
        unresolved_placeholders = sorted(set(re.findall(r"\{\{[a-zA-Z0-9_]+\}\}", prompt)))
        if unresolved_placeholders:
            raise ValueError(
                f"Unresolved prompt placeholders in {prompt_file_path}: {', '.join(unresolved_placeholders)}"
            )
    if file_ids:
        return call_openai_with_files_json(
            file_ids=file_ids,
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            tools=tools,
        )
    return call_openai_with_json_response(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        system_message=system_message,
    )
 