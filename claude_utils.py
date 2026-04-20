"""
Anthropic Claude utility functions.
Provides centralized Claude client management and API call functions,
mirroring the interface of openai_utils.py.
"""

import time
import json
import re
import io
from anthropic import Anthropic, RateLimitError
from pathlib import Path
import sys

local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger

logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path) / 'config.json')

CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-6"
CLAUDE_OPUS = "claude-opus-4-6"

DEFAULT_MODEL = CLAUDE_OPUS
DEFAULT_MAX_TOKENS = 64000
FILES_BETA = "files-api-2025-04-14"

_client = None


def get_claude_client():
    """Get or create Anthropic client instance."""
    global _client
    if _client is None:
        _client = Anthropic(api_key=config['claude_key'])
    return _client


def upload_text_file(text: str, filename: str = "document.txt") -> str:
    """
    Upload a text string as a plain-text file via the Files API (beta).
    Returns the file_id for use in messages.
    """
    client = get_claude_client()
    buf = io.BytesIO(text.encode("utf-8"))
    result = client.beta.files.upload(
        file=(filename, buf, "text/plain"),
    )
    logger.info(f"Uploaded file '{filename}' -> {result.id} ({result.size_bytes} bytes)")
    return result.id


def delete_file(file_id: str):
    """Delete a previously uploaded file."""
    client = get_claude_client()
    client.beta.files.delete(file_id)
    logger.info(f"Deleted file {file_id}")


def _log_elapsed(model: str, elapsed: float):
    if elapsed > 60:
        logger.info(f"Response from {model} took {int(elapsed // 60)}m {int(round(elapsed % 60))}s")
    else:
        logger.info(f"Response from {model} took {int(round(elapsed))}s")


def call_claude_api(messages, model=DEFAULT_MODEL, temperature=0.1,
                    system_message=None, max_tokens=DEFAULT_MAX_TOKENS,
                    file_ids=None):
    """
    Make an Anthropic API call with proper error handling and logging.

    Args:
        messages: List of message dicts (role / content), same shape as OpenAI.
        model: Claude model identifier.
        temperature: Sampling temperature.
        system_message: Optional system prompt string.
        max_tokens: Maximum tokens in the response.
        file_ids: Optional list of file_ids to attach as document blocks.
                  Uses the Files API beta — files are referenced, not re-uploaded.

    Returns:
        Tuple of (success: bool, response_content: str or None, error: str or None)
    """
    client = get_claude_client()
    use_beta = file_ids is not None and len(file_ids) > 0

    if use_beta and file_ids:
        # Inject file document blocks into the first user message's content
        messages = _inject_file_blocks(messages, file_ids)

    api_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if system_message:
        api_params["system"] = system_message

    if use_beta:
        api_params["betas"] = [FILES_BETA]

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Calling {model} (streaming, attempt {attempt}/{max_retries})")
            start_time = time.time()

            collected_text = []
            stream_method = client.beta.messages.stream if use_beta else client.messages.stream
            with stream_method(**api_params) as stream:
                for text in stream.text_stream:
                    collected_text.append(text)

            response_content = "".join(collected_text).strip()
            _log_elapsed(model, time.time() - start_time)

            return True, response_content, None

        except RateLimitError as e:
            wait = 2 ** attempt * 15
            logger.warning(f"Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
            if attempt == max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries: {e}")
                return False, None, str(e)

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return False, None, str(e)

    return False, None, "Max retries exhausted"


def _inject_file_blocks(messages: list[dict], file_ids: list[str]) -> list[dict]:
    """
    Prepend document blocks for each file_id into the first user message.
    Converts simple string content into a content-block list if needed.
    """
    messages = [m.copy() for m in messages]
    first_user = next((m for m in messages if m["role"] == "user"), None)
    if first_user is None:
        return messages

    # Normalise content to list form
    if isinstance(first_user["content"], str):
        first_user["content"] = [{"type": "text", "text": first_user["content"]}]
    else:
        first_user["content"] = list(first_user["content"])

    file_blocks = [
        {"type": "document", "source": {"type": "file", "file_id": fid}}
        for fid in file_ids
    ]
    first_user["content"] = file_blocks + first_user["content"]
    return messages


def call_claude_with_json_response(messages, model=DEFAULT_MODEL, temperature=0.1,
                                   system_message=None, max_tokens=DEFAULT_MAX_TOKENS,
                                   file_ids=None):
    """
    Make a Claude API call and parse JSON response.

    Returns:
        Tuple of (success: bool, parsed_json: dict or None, error: str or None)
    """
    success, response_content, error = call_claude_api(
        messages, model, temperature, system_message, max_tokens, file_ids
    )

    if not success:
        return False, None, error

    try:
        parsed_json = json.loads(response_content)
        return True, parsed_json, None
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_content)
    if match:
        try:
            parsed_json = json.loads(match.group(1))
            return True, parsed_json, None
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse JSON response. Raw response:\n{response_content}")
    return False, None, f"JSON parsing failed for response: {response_content[:200]}"
