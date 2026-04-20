"""
Unified LLM interface wrapping both Claude and OpenAI.
Provides a single set of functions that delegate to the correct provider
based on a provider string ("claude" or "openai").
"""

from pathlib import Path
import sys

local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger

logger = get_logger(__name__)

PROVIDER_CLAUDE = "claude"
PROVIDER_OPENAI = "openai"


def upload_text_file(text: str, filename: str, provider: str) -> str:
    """Upload a text string as a file. Returns a file_id."""
    if provider == PROVIDER_CLAUDE:
        from utils.claude_utils import upload_text_file as _upload
        return _upload(text, filename)
    else:
        import tempfile, os
        from utils.openai_utils import upload_file as _upload
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        try:
            return _upload(tmp.name)
        finally:
            os.unlink(tmp.name)


def delete_file(file_id: str, provider: str):
    """Delete a previously uploaded file."""
    if provider == PROVIDER_CLAUDE:
        from utils.claude_utils import delete_file as _delete
        _delete(file_id)
    else:
        from utils.openai_utils import get_openai_client
        client = get_openai_client()
        client.files.delete(file_id)
        logger.info(f"Deleted OpenAI file {file_id}")


def call_with_json_response(messages, provider: str, system_message=None, file_ids=None):
    """
    Call the LLM and parse a JSON response.

    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}]
        provider: "claude" or "openai"
        system_message: Optional system prompt string.
        file_ids: Optional list of uploaded file_ids to attach.

    Returns:
        Tuple of (success: bool, parsed_json: dict or None, error: str or None)
    """
    if provider == PROVIDER_CLAUDE:
        from utils.claude_utils import call_claude_with_json_response, DEFAULT_MODEL
        return call_claude_with_json_response(
            messages=messages,
            model=DEFAULT_MODEL,
            system_message=system_message,
            file_ids=file_ids,
        )
    else:
        from utils.openai_utils import call_openai_with_file_json, call_openai_with_json_response, GPT_5_4
        if file_ids:
            prompt = messages[0]["content"] if messages else ""
            if system_message:
                prompt = system_message + "\n\n" + prompt
            all_ranges = []
            for fid in file_ids:
                success, parsed, error = call_openai_with_file_json(
                    file_id=fid,
                    prompt=prompt,
                    model=GPT_5_4,
                    system_message=None, 
                )
                if not success:
                    return False, None, error
                # Normalize: extract the ranges list regardless of response shape
                if isinstance(parsed, dict):
                    all_ranges.extend(parsed.get("ranges", []))
                elif isinstance(parsed, list):
                    all_ranges.extend(parsed)
            return True, {"ranges": all_ranges}, None
        else:
            return call_openai_with_json_response(
                messages=messages,
                model=DEFAULT_MODEL,
                system_message=system_message,
            )
