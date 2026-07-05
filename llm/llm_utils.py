"""
Unified LLM interface wrapping both Claude and OpenAI.
Provides a single set of functions that delegate to the correct provider
based on a provider string ("claude" or "openai").
"""

from pathlib import Path
import sys
import hashlib
import subprocess

local_python_path = str(Path(__file__).parents[2])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger

logger = get_logger(__name__)

PROVIDER_CLAUDE = "claude"
PROVIDER_OPENAI = "openai"


def get_prompt_file_version(prompt_file_path: Path) -> str:
    git_metadata = get_prompt_git_metadata(prompt_file_path)
    if git_metadata and git_metadata.get("prompt_version"):
        return git_metadata["prompt_version"]

    prompt_file_path = Path(prompt_file_path)
    prompt_text = prompt_file_path.read_text(encoding="utf-8")
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def run_git_command(repo_root: Path, args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def find_git_repo_root(start_path: Path) -> Path | None:
    start_path = Path(start_path).resolve()
    candidates = [start_path] + list(start_path.parents)
    for candidate in candidates:
        git_dir = candidate / ".git"
        if git_dir.exists():
            return candidate
    return None


def get_prompt_git_metadata(prompt_file_path: Path) -> dict | None:
    prompt_file_path = Path(prompt_file_path).resolve()
    repo_root = find_git_repo_root(prompt_file_path.parent)
    if not repo_root:
        return None

    try:
        relative_path = prompt_file_path.relative_to(repo_root).as_posix()
    except ValueError:
        return None

    tracked = run_git_command(repo_root, ["ls-files", "--error-unmatch", "--", relative_path])
    if not tracked:
        return None

    head_commit = run_git_command(repo_root, ["rev-parse", "HEAD"])
    if not head_commit:
        return None

    head_blob = run_git_command(repo_root, ["rev-parse", f"HEAD:{relative_path}"])
    if not head_blob:
        return None

    status_output = run_git_command(repo_root, ["status", "--porcelain", "--", relative_path]) or ""
    is_dirty = bool(status_output.strip())
    prompt_version = f"git:{head_commit}:{relative_path}"
    if is_dirty:
        prompt_version = f"{prompt_version}:dirty"

    return {
        "prompt_version": prompt_version,
        "prompt_git_commit": head_commit,
        "prompt_git_blob": head_blob,
        "prompt_git_path": relative_path,
        "prompt_git_is_dirty": is_dirty,
    }


def build_prompt_call_metadata(
    prompt_file_path: Path,
    model: str,
    prompt_version: str | None = None,
) -> dict:
    prompt_file_path = Path(prompt_file_path)
    metadata = {
        "prompt_file_name": prompt_file_path.name,
        "prompt_version": prompt_version or get_prompt_file_version(prompt_file_path),
        "model": model,
    }
    git_metadata = get_prompt_git_metadata(prompt_file_path)
    if git_metadata:
        metadata.update(git_metadata)
    return metadata


def wrap_response_with_metadata(
    response_payload,
    metadata: dict,
    metadata_key: str = "llm_metadata",
) -> dict:
    return {
        "response": response_payload,
        metadata_key: metadata,
    }


def split_response_and_metadata(
    wrapped_payload,
    metadata_key: str = "llm_metadata",
):
    if isinstance(wrapped_payload, dict) and "response" in wrapped_payload and metadata_key in wrapped_payload:
        return wrapped_payload["response"], wrapped_payload[metadata_key]
    return wrapped_payload, None


def upload_text_file(text: str, filename: str, provider: str) -> str:
    """Upload a text string as a file. Returns a file_id."""
    if provider == PROVIDER_CLAUDE:
        from utils.claude_utils import upload_text_file as _upload
        return _upload(text, filename)
    else:
        import tempfile, os
        from utils.llm.openai_utils import upload_file as _upload
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
        from utils.llm.openai_utils import get_openai_client
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
    elif provider == PROVIDER_OPENAI:
        from utils.llm.openai_utils import call_openai_with_file_json, call_openai_with_json_response, GPT_5_5
        if file_ids:
            prompt = messages[0]["content"] if messages else ""
            if system_message:
                prompt = system_message + "\n\n" + prompt
            all_ranges = []
            for fid in file_ids:
                success, parsed, error = call_openai_with_file_json(
                    file_id=fid,
                    prompt=prompt,
                    model=GPT_5_5,
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
                model=GPT_5_5,
                system_message=system_message,
            )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def call_with_json_prompt_file(
    prompt_file_path,
    provider: str,
    file_ids=None,
    prompt_parameters=None,
    model=None,
    temperature=0.1,
    system_message=None,
    include_response_metadata=False,
    prompt_version=None,
    metadata_key="llm_metadata",
    tools=None,
    max_tokens=None,
):
    """
    Unified prompt-file JSON call for OpenAI and Claude.

    Args:
        prompt_file_path: Path to UTF-8 prompt template file.
        provider: "openai" or "claude".
        file_ids: Optional uploaded file IDs to attach.
        prompt_parameters: Optional dict for replacing {{param_name}} placeholders.
        model: Optional explicit model override per provider.
        temperature: Sampling temperature.
        system_message: Optional system prompt string.
        include_response_metadata: When True, returns wrapped response+metadata.
        prompt_version: Optional explicit prompt version override.
        metadata_key: Metadata field name for wrapped payload.
        tools: Optional OpenAI tools payload.
        max_tokens: Optional Claude max_tokens override.
    """
    if provider == PROVIDER_OPENAI:
        from utils.llm.openai_utils import call_openai_with_json_prompt_file, DEFAULT_MODEL as OPENAI_DEFAULT_MODEL

        selected_model = model or OPENAI_DEFAULT_MODEL
        return call_openai_with_json_prompt_file(
            prompt_file_path=prompt_file_path,
            file_ids=file_ids,
            prompt_parameters=prompt_parameters,
            model=selected_model,
            temperature=temperature,
            system_message=system_message,
            tools=tools,
            include_response_metadata=include_response_metadata,
            prompt_version=prompt_version,
            metadata_key=metadata_key,
        )

    if provider == PROVIDER_CLAUDE:
        from utils.claude_utils import (
            call_claude_with_json_prompt_file,
            DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL,
            DEFAULT_MAX_TOKENS as CLAUDE_DEFAULT_MAX_TOKENS,
        )

        selected_model = model or CLAUDE_DEFAULT_MODEL
        selected_max_tokens = max_tokens or CLAUDE_DEFAULT_MAX_TOKENS
        return call_claude_with_json_prompt_file(
            prompt_file_path=prompt_file_path,
            file_ids=file_ids,
            prompt_parameters=prompt_parameters,
            model=selected_model,
            temperature=temperature,
            system_message=system_message,
            max_tokens=selected_max_tokens,
            include_response_metadata=include_response_metadata,
            prompt_version=prompt_version,
            metadata_key=metadata_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")
