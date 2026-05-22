"""
PDF extraction (Ollama) and JSON-to-DataFrame conversion.

Single module: prompts, Ollama calls, nested JSON flattening.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz
import httpx
import pandas as pd

from config import (
    MAX_PAGES_PER_PDF,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OUTPUT_DIR,
    path_for_api,
    pdf_filename,
)

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH_FOR_TEXT_MODE = 80
REQUIRED_COLUMNS = ["year", "section", "table_title", "heading", "column", "parameter", "value"]

EXTRACTION_SYSTEM_PROMPT = """You are a financial statement data extraction assistant.
Extract structured numeric data from the provided PDF page.
Return valid JSON only, matching this schema:
{
  "company_name": string,
  "registration_number": string or null,
  "financial_statement_type": string,
  "financial_period": {
    "as_at_date": string or null,
    "annual_financial_statements_for_year_ended": string or null
  },
  "currency": string or null,
  "data": {
    "<YEAR>": {
      "<section_key>": {
        "<subsection_key>": {
          "<line_item_key>": <number>
        }
      }
    }
  }
}
Rules:
- Use snake_case keys for nested structure (e.g. non_current_assets, total_assets).
- Include all numeric line items and subtotals visible on the page.
- Years in "data" must be 4-digit strings (e.g. "2020", "2021").
- Use 0 for dash or empty numeric cells.
- If the page has no financial tables, return {"data": {}} with other fields null where unknown.
"""

EXTRACTION_USER_TEXT = """Extract all financial statement line items from this PDF page text.
Page number: {page_number}

--- PAGE TEXT ---
{page_text}
"""


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""


# ---------------------------------------------------------------------------
# JSON flattening (for post_processing)
# ---------------------------------------------------------------------------

def _key_to_label(key: str) -> str:
    return str(key).replace("_", " ").strip().title()


def _path_to_hierarchy(path: list[str], leaf_key: str) -> tuple[str, str, str]:
    if not path:
        return "", "", _key_to_label(leaf_key)
    section = _key_to_label(path[0])
    if len(path) == 1:
        return section, "", _key_to_label(leaf_key)
    heading = _key_to_label(path[1])
    if len(path) > 2:
        extra = " - ".join(_key_to_label(p) for p in path[2:])
        heading = f"{heading} - {extra}"
    return section, heading, _key_to_label(leaf_key)


def _walk_nested(
    node: dict,
    year: int,
    table_title: str,
    company: str,
    path: list[str],
    rows: list[dict[str, Any]],
) -> None:
    for key, value in node.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            section, heading, parameter = _path_to_hierarchy(path, key)
            if not section and company:
                section = company
            rows.append(
                {
                    "year": year,
                    "section": section,
                    "table_title": table_title,
                    "heading": heading,
                    "column": "",
                    "parameter": parameter,
                    "value": value,
                }
            )
        elif isinstance(value, dict):
            _walk_nested(value, year, table_title, company, path + [key], rows)


def nested_content_to_dataframe(extracted_content: dict, page_number: int | None = None) -> pd.DataFrame:
    """Flatten nested extracted_content into rows for post_processing."""
    rows: list[dict[str, Any]] = []
    table_title = extracted_content.get("financial_statement_type") or ""
    company = extracted_content.get("company_name") or ""
    data = extracted_content.get("data") or {}

    if not isinstance(data, dict):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for year_key, year_blob in data.items():
        try:
            year = int(str(year_key).strip())
        except (ValueError, TypeError):
            continue
        if isinstance(year_blob, dict):
            _walk_nested(year_blob, year, table_title, company, [], rows)

    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(rows)
    if page_number is not None:
        df["page_number"] = page_number
    return df


def _parse_json_from_markdown(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = match.group(1) if match else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def load_extraction_json_file(path: str) -> pd.DataFrame:
    """Load page_*_extracted_data_cleaned.json into a flat DataFrame."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    page_number = doc.get("page_number")
    frames: list[pd.DataFrame] = []

    for item in doc.get("extracted_data", []):
        page = item.get("page", page_number)
        content = item.get("extracted_content")
        if not content and item.get("data"):
            content = _parse_json_from_markdown(str(item["data"]))
        if content:
            frames.append(nested_content_to_dataframe(content, page))

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def extraction_document_to_dataframe(doc: dict) -> pd.DataFrame:
    """Convert an in-memory extraction document (API response) to a DataFrame."""
    page_number = doc.get("page_number")
    frames: list[pd.DataFrame] = []
    for item in doc.get("extracted_data", []):
        content = item.get("extracted_content")
        if content:
            frames.append(nested_content_to_dataframe(content, item.get("page", page_number)))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# PDF + Ollama extraction
# ---------------------------------------------------------------------------

def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise PDFExtractionError("Model returned invalid JSON") from None


def _check_ollama_available() -> None:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise PDFExtractionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start Ollama and ensure the service is running."
        ) from e

    models = [m.get("name", "") for m in response.json().get("models", [])]
    if not any(OLLAMA_MODEL in name or name.startswith(OLLAMA_MODEL) for name in models):
        available = ", ".join(models[:10]) or "(none)"
        raise PDFExtractionError(
            f"Model '{OLLAMA_MODEL}' not found in Ollama. "
            f"Run: ollama pull {OLLAMA_MODEL}\nAvailable: {available}"
        )


def _extract_page_text(page: fitz.Page) -> str:
    text = page.get_text("text").strip()
    if len(text) >= MIN_TEXT_LENGTH_FOR_TEXT_MODE:
        return text

    blocks = page.get_text("blocks")
    if isinstance(blocks, list):
        parts = []
        for block in blocks:
            if len(block) > 4 and isinstance(block[4], str):
                parts.append(block[4].strip())
        combined = "\n".join(p for p in parts if p).strip()
        if len(combined) >= MIN_TEXT_LENGTH_FOR_TEXT_MODE:
            return combined

    if text:
        return text

    return (
        "(minimal text extracted — page may be scanned or image-based; "
        "use a vision-capable Ollama model for better results on this page)"
    )


def _extract_page_with_ollama(page_number: int, page_text: str) -> dict:
    user_content = EXTRACTION_USER_TEXT.format(
        page_number=page_number + 1,
        page_text=page_text,
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise PDFExtractionError(f"Ollama request failed: {e}") from e

    raw = response.json().get("message", {}).get("content") or "{}"
    return _parse_llm_json(raw)


def parse_page_list(pages: str | None) -> list[int] | None:
    """
    Parse '9,10,11' into 0-based page indices [8,9,10].
    Empty/None means process all pages.
    """
    if not pages or not str(pages).strip():
        return None
    indices = []
    for part in str(pages).split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        indices.append(n - 1 if n >= 1 else n)
    return indices or None


def extract_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    page_numbers: list[int] | None = None,
) -> dict[str, Any]:
    """
    Extract financial data from each PDF page via Ollama.

    Saves page_<n>_extracted_data_cleaned.json under output/<pdf_stem>/.
    page_numbers: 0-based indices; None = all pages (up to MAX_PAGES_PER_PDF).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF not found: {pdf_path}")

    _check_ollama_available()

    stem = pdf_path.stem
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = min(len(doc), MAX_PAGES_PER_PDF)
    pages_to_process = page_numbers if page_numbers else list(range(total_pages))
    pages_to_process = [p for p in pages_to_process if 0 <= p < len(doc)]

    all_pages: list[dict[str, Any]] = []
    saved_files: list[str] = []

    for page_num in pages_to_process:
        page_text = _extract_page_text(doc[page_num])
        try:
            extracted_content = _extract_page_with_ollama(page_num, page_text)
            status = "success"
        except Exception as e:
            logger.error("Page %s extraction failed: %s", page_num + 1, e)
            extracted_content = {"data": {}}
            status = "error"

        if status == "success" and not (extracted_content.get("data") or {}):
            continue

        page_doc = {
            "source_pdf": pdf_filename(pdf_path),
            "page_number": page_num + 1,
            "processed_pages": 1,
            "model": OLLAMA_MODEL,
            "extracted_data": [
                {
                    "page": page_num + 1,
                    "status": status,
                    "extracted_content": extracted_content,
                }
            ],
        }
        filename = f"page_{page_num + 1}_extracted_data_cleaned.json"
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(page_doc, f, indent=2)
        saved_files.append(path_for_api(out_path))
        all_pages.append(page_doc)

    doc.close()

    return {
        "source_pdf": pdf_filename(pdf_path),
        "output_directory": path_for_api(out_dir),
        "folder_name": stem,
        "model": OLLAMA_MODEL,
        "pages_processed": len(all_pages),
        "saved_files": saved_files,
        "pages": all_pages,
    }
