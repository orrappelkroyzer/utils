import json
import os
import tempfile
from pathlib import Path

from utils.llm.openai_utils import (
    call_openai_with_json_prompt_file,
    upload_file,
    get_openai_client,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
)
from utils.utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHARS_PER_CHUNK = 12000
MULTI_ITEM_SEPARATOR = "\n\n[ITEM_BREAK]\n\n"


def split_text_to_chunks(text, max_chars=DEFAULT_MAX_CHARS_PER_CHUNK):
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n")
    chunks = []
    current = []
    current_len = 0
    for paragraph in paragraphs:
        piece = paragraph + "\n"
        if current_len + len(piece) > max_chars and current:
            chunks.append("".join(current).strip())
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def select_files_to_process(input_files, output_dir, overwrite, limit):
    if overwrite:
        candidates = list(input_files)
        skipped_existing = 0
    else:
        candidates = []
        skipped_existing = 0
        for input_path in input_files:
            output_path = output_dir / input_path.name
            if output_path.exists():
                skipped_existing += 1
                continue
            candidates.append(input_path)

    if limit is not None:
        candidates = candidates[:limit]
    return candidates, skipped_existing


def call_prompt_on_text(
    text,
    prompt_file_path,
    prompt_text_parameter_name,
    model,
    temperature,
    response_field_name,
    prompt_parameters=None,
):
    parameters = dict(prompt_parameters or {})
    parameters[prompt_text_parameter_name] = text
    success, response_content, error = call_openai_with_json_prompt_file(
        prompt_file_path=prompt_file_path,
        prompt_parameters=parameters,
        model=model,
        temperature=temperature,
    )
    if not success:
        raise RuntimeError(error or "Unknown OpenAI error")
    if not isinstance(response_content, dict):
        raise RuntimeError("Expected JSON object response.")
    if response_field_name not in response_content:
        raise RuntimeError(f"Response did not include '{response_field_name}' : {response_content.keys()}")
    return response_content


def call_prompt_on_uploaded_file(
    text,
    prompt_file_path,
    prompt_text_parameter_name,
    model,
    temperature,
    response_field_name,
    prompt_parameters=None,
):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    try:
        file_id = upload_file(tmp.name)
        response_content = call_prompt_with_uploaded_file_and_downloaded_output(
            file_id=file_id,
            prompt_file_path=prompt_file_path,
            prompt_text_parameter_name=prompt_text_parameter_name,
            response_field_name=response_field_name,
            prompt_parameters=prompt_parameters,
            model=model,
            temperature=temperature,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if not isinstance(response_content, dict):
        raise RuntimeError("Expected JSON object response.")
    if response_field_name not in response_content:
        raise RuntimeError(f"Response did not include '{response_field_name}' : {response_content.keys()}")
    return response_content


def normalize_response_field_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict) and "text" in item:
                text = normalize_response_field_text(item.get("text"))
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return MULTI_ITEM_SEPARATOR.join(parts).strip()
    if isinstance(value, dict) and "text" in value:
        return normalize_response_field_text(value.get("text"))
    return str(value).strip()


def extract_output_text_from_responses(raw_responses, response_field_name):
    outputs = []
    for response in raw_responses:
        if not isinstance(response, dict):
            continue
        if response_field_name not in response:
            continue
        output_text = normalize_response_field_text(response[response_field_name])
        if output_text:
            outputs.append(output_text)
    if not outputs:
        raise RuntimeError(f"No non-empty '{response_field_name}' field found in raw responses.")
    return "\n\n".join(outputs).strip()


def load_prompt_with_parameters(prompt_file_path, prompt_parameters):
    prompt_file_path = Path(prompt_file_path)
    if not prompt_file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    prompt = prompt_file_path.read_text(encoding="utf-8").strip()
    for parameter_name, parameter_value in (prompt_parameters or {}).items():
        placeholder = f"{{{{{parameter_name}}}}}"
        if not isinstance(parameter_value, str):
            parameter_value = json.dumps(parameter_value, ensure_ascii=False)
        prompt = prompt.replace(placeholder, parameter_value)
    return prompt


def collect_file_ids_from_object(obj):
    file_ids = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "file_id" and isinstance(value, str) and value.startswith("file-"):
                file_ids.append(value)
            if key == "id" and isinstance(value, str) and value.startswith("file-"):
                file_ids.append(value)
            file_ids.extend(collect_file_ids_from_object(value))
    elif isinstance(obj, list):
        for item in obj:
            file_ids.extend(collect_file_ids_from_object(item))
    return file_ids


def parse_json_from_text(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = None
    if "```" in text:
        import re
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj:end_obj + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr:end_arr + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def collect_text_fields_from_object(obj):
    texts = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"text", "output_text"} and isinstance(value, str):
                texts.append(value)
            texts.extend(collect_text_fields_from_object(value))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(collect_text_fields_from_object(item))
    return texts


def build_fallback_response_from_text(response_field_name, text):
    text = (text or "").strip()
    if not text:
        return None
    if response_field_name == "records":
        return {
            "full_article": 0,
            "records": [{"text": text}],
        }
    return {response_field_name: text}


def download_json_response_from_file_ids(client, file_ids):
    for file_id in file_ids:
        try:
            logger.info(f"Attempting to download OpenAI output file: {file_id}")
            file_content_response = client.files.content(file_id)
            if hasattr(file_content_response, "read"):
                file_bytes = file_content_response.read()
            elif hasattr(file_content_response, "content"):
                file_bytes = file_content_response.content
            else:
                logger.warning(f"Unknown file content response type for file {file_id}: {type(file_content_response)}")
                continue
            if isinstance(file_bytes, str):
                file_text = file_bytes
            else:
                file_text = file_bytes.decode("utf-8", errors="replace")
            logger.info(f"Downloaded file {file_id} ({len(file_text)} chars)")
            parsed = parse_json_from_text(file_text)
            if parsed is not None:
                logger.info(f"Parsed JSON from downloaded OpenAI output file: {file_id}")
                return parsed
            logger.warning(
                f"Downloaded file {file_id} but JSON parse failed. "
                f"Preview: {file_text[:300]!r}"
            )
        except Exception as exc:
            logger.warning(f"Failed downloading/parsing output file {file_id}: {exc}")
            continue
    return None


def parse_uploaded_response_payload(client, response, response_field_name):
    response_dump = response.model_dump() if hasattr(response, "model_dump") else {}
    if not response_dump:
        logger.warning("Response did not provide model_dump output; using object fallbacks only.")

    file_ids = list(dict.fromkeys(collect_file_ids_from_object(response_dump)))
    if file_ids:
        downloaded = download_json_response_from_file_ids(client, file_ids)
        if downloaded is not None:
            return downloaded, file_ids, ""

    output_text = getattr(response, "output_text", "").strip()
    parsed = parse_json_from_text(output_text)
    if parsed is not None:
        return parsed, file_ids, output_text

    text_candidates = []
    if output_text:
        text_candidates.append(output_text)
    text_candidates.extend(collect_text_fields_from_object(response_dump))

    for candidate_text in text_candidates:
        parsed = parse_json_from_text(candidate_text)
        if parsed is not None:
            return parsed, file_ids, output_text

    plain_text_candidates = [t.strip() for t in text_candidates if isinstance(t, str) and t.strip()]
    if plain_text_candidates:
        best_text = max(plain_text_candidates, key=len)
        fallback_response = build_fallback_response_from_text(response_field_name, best_text)
        if fallback_response is not None:
            logger.warning(
                "Could not parse strict JSON response; using plain-text fallback wrapper "
                f"for field '{response_field_name}'."
            )
            return fallback_response, file_ids, output_text

    return None, file_ids, output_text


def call_prompt_with_uploaded_file_and_downloaded_output(
    file_id,
    prompt_file_path,
    prompt_text_parameter_name,
    response_field_name,
    prompt_parameters,
    model,
    temperature,
):
    parameters = dict(prompt_parameters or {})
    if prompt_text_parameter_name and prompt_text_parameter_name not in parameters:
        parameters[prompt_text_parameter_name] = "Use the attached file text as the source."

    prompt = load_prompt_with_parameters(prompt_file_path, parameters)
    prompt += "\n\nIMPORTANT: Return strict JSON only, with no extra text."

    client = get_openai_client()
    api_params = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "max_output_tokens": 16384,
        "store": True,
    }
    if isinstance(model, str) and model.startswith("gpt-5"):
        api_params["reasoning"] = {"effort": "minimal"}
    if SUPPORTED_MODELS.get(model, {}).get("supports_temperature", True):
        api_params["temperature"] = temperature

    response = client.responses.create(**api_params)
    if getattr(response, "status", None) == "incomplete":
        logger.warning(
            "Uploaded-file call returned incomplete response. "
            f"details={getattr(response, 'incomplete_details', None)}"
        )

    parsed, file_ids, output_text = parse_uploaded_response_payload(
        client=client,
        response=response,
        response_field_name=response_field_name,
    )
    if parsed is not None:
        return parsed

    incomplete_details = getattr(response, "incomplete_details", None) or {}
    reason = ""
    if isinstance(incomplete_details, dict):
        reason = str(incomplete_details.get("reason", "")).strip().lower()
    if getattr(response, "status", None) == "incomplete" and reason == "max_output_tokens":
        response_id = getattr(response, "id", None)
        if response_id:
            logger.warning(
                "Retrying truncated uploaded-file response with continuation-only JSON request. "
                f"previous_response_id={response_id}"
            )
            retry_params = {
                "model": model,
                "previous_response_id": response_id,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Your previous response was cut off. "
                                    "Return ONLY the final strict JSON object now, with no explanation."
                                ),
                            }
                        ],
                    }
                ],
                "max_output_tokens": 16384,
                "store": True,
            }
            if isinstance(model, str) and model.startswith("gpt-5"):
                retry_params["reasoning"] = {"effort": "minimal"}
            if SUPPORTED_MODELS.get(model, {}).get("supports_temperature", True):
                retry_params["temperature"] = temperature

            retry_response = client.responses.create(**retry_params)
            parsed, retry_file_ids, retry_output_text = parse_uploaded_response_payload(
                client=client,
                response=retry_response,
                response_field_name=response_field_name,
            )
            if parsed is not None:
                return parsed
            file_ids = list(dict.fromkeys(file_ids + retry_file_ids))
            if not output_text and retry_output_text:
                output_text = retry_output_text

    logger.error(
        "Uploaded-file response parsing failed completely. "
        f"response_field_name={response_field_name}, "
        f"candidate_file_ids={file_ids}, "
        f"output_text_preview={output_text[:300]!r}"
    )
    raise RuntimeError("Could not obtain JSON/usable text response from downloaded output file or response text.")


def process_text_with_prompt_file(
    text,
    prompt_file_path,
    prompt_text_parameter_name,
    response_field_name,
    model=DEFAULT_MODEL,
    temperature=0.1,
    max_chars_per_chunk=DEFAULT_MAX_CHARS_PER_CHUNK,
    prompt_parameters=None,
):
    text = text.strip()
    if not text:
        return []

    # To save token overhead, use a single uploaded file call for long texts
    # instead of splitting across multiple prompts.
    if len(text) > max_chars_per_chunk:
        logger.info(
            f"Text is long ({len(text)} chars > {max_chars_per_chunk}); "
            "using uploaded-file mode instead of chunking."
        )
        raw_response = call_prompt_on_uploaded_file(
            text=text,
            prompt_file_path=prompt_file_path,
            prompt_text_parameter_name=prompt_text_parameter_name,
            model=model,
            temperature=temperature,
            response_field_name=response_field_name,
            prompt_parameters=prompt_parameters,
        )
        return [raw_response]

    raw_response = call_prompt_on_text(
        text=text,
        prompt_file_path=prompt_file_path,
        prompt_text_parameter_name=prompt_text_parameter_name,
        model=model,
        temperature=temperature,
        response_field_name=response_field_name,
        prompt_parameters=prompt_parameters,
    )
    return [raw_response]


def process_text_files_with_prompt_file(
    input_dir,
    output_dir,
    prompt_file_path,
    prompt_text_parameter_name,
    response_field_name,
    model=DEFAULT_MODEL,
    temperature=0.1,
    overwrite=False,
    limit=None,
    max_chars_per_chunk=DEFAULT_MAX_CHARS_PER_CHUNK,
    prompt_parameters=None,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    prompt_file_path = Path(prompt_file_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not prompt_file_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_file_path}")

    input_files = sorted(input_dir.glob("*.txt"))
    if not input_files:
        logger.warning(f"No .txt files found in {input_dir}")
        return {
            "cleaned_count": 0,
            "skipped_existing": 0,
            "skipped_empty": 0,
            "failed": 0,
            "selected": 0,
            "total": 0,
        }

    files_to_process, skipped_existing = select_files_to_process(
        input_files=input_files,
        output_dir=output_dir,
        overwrite=overwrite,
        limit=limit,
    )

    logger.info(
        f"Pre-check: total_files={len(input_files)}, "
        f"already_exists={skipped_existing}, "
        f"selected_for_this_run={len(files_to_process)}"
    )
    if not files_to_process:
        return {
            "cleaned_count": 0,
            "skipped_existing": skipped_existing,
            "skipped_empty": 0,
            "failed": 0,
            "selected": 0,
            "total": len(input_files),
        }

    cleaned_count = 0
    skipped_empty = 0
    failed = 0

    for i, input_path in enumerate(files_to_process, start=1):
        output_path = output_dir / input_path.name
        raw_text = input_path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw_text:
            skipped_empty += 1
            logger.warning(f"[{i}/{len(files_to_process)}] Empty input file, skipping: {input_path.name}")
            continue

        try:
            logger.info(f"[{i}/{len(files_to_process)}] Processing: {input_path.name}")
            raw_responses = process_text_with_prompt_file(
                text=raw_text,
                prompt_file_path=prompt_file_path,
                prompt_text_parameter_name=prompt_text_parameter_name,
                response_field_name=response_field_name,
                model=model,
                temperature=temperature,
                max_chars_per_chunk=max_chars_per_chunk,
                prompt_parameters=prompt_parameters,
            )
            processed_text = extract_output_text_from_responses(
                raw_responses=raw_responses,
                response_field_name=response_field_name,
            )
            output_path.write_text(processed_text, encoding="utf-8")
            metadata_path = output_dir / f"{output_path.stem}.json"
            metadata_path.write_text(
                json.dumps(raw_responses, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cleaned_count += 1
            logger.info(f"Wrote output: {output_path}")
        except Exception as exc:
            failed += 1
            logger.error(f"Failed processing {input_path.name}: {exc}")

    return {
        "cleaned_count": cleaned_count,
        "skipped_existing": skipped_existing,
        "skipped_empty": skipped_empty,
        "failed": failed,
        "selected": len(files_to_process),
        "total": len(input_files),
    }
