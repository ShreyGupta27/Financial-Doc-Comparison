"""
FastAPI application for PDF extraction and financial report comparison.

Run: uvicorn app:app --reload
Swagger UI: http://127.0.0.1:8000/docs
"""
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import INPUT_DIR, OUTPUT_DIR, path_for_api
from extraction import (
    PDFExtractionError,
    extract_pdf,
    extraction_document_to_dataframe,
    parse_page_list,
)
from post_processing import CompareExtractedData

app = FastAPI(
    title="Financial Reports Comparison API",
    description="Upload PDFs, extract financial data with a local Ollama model, and compare across reporting periods.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.replace({np.nan: None})
    return out.to_dict(orient="records")


async def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF (.pdf)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / upload.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest_path


class CompareFoldersRequest(BaseModel):
    previous_folder: str = Field(..., description="Folder name under output/ for previous period")
    current_folder: str = Field(..., description="Folder name under output/ for current period")
    previous_pdf_filename: str | None = Field(None, description="Optional PDF filename hint")
    current_pdf_filename: str | None = Field(None, description="Optional PDF filename hint")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", summary="Extract financial data from a PDF")
async def extract_endpoint(
    file: UploadFile = File(..., description="Financial report PDF"),
    pages: str | None = Form(
        None,
        description="Optional comma-separated 1-based page numbers (e.g. 9,10,11). Omit to process all pages.",
    ),
) -> dict[str, Any]:
    """
    Upload a PDF, run Ollama extraction per page, save JSON under output/<pdf_name>/,
    and return extraction results in the response body.
    """
    pdf_path = await _save_upload(file, INPUT_DIR)
    try:
        result = extract_pdf(pdf_path, page_numbers=parse_page_list(pages))
    except PDFExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}") from e

    preview_rows: list[dict[str, Any]] = []
    for page_doc in result.get("pages", []):
        df = extraction_document_to_dataframe(page_doc)
        preview_rows.extend(_df_to_records(df.head(20)))

    return {
        "message": "Extraction complete",
        "folder_name": result["folder_name"],
        "output_directory": result["output_directory"],
        "pages_processed": result["pages_processed"],
        "saved_files": result["saved_files"],
        "pages": result["pages"],
        "flat_preview": preview_rows[:50],
    }


@app.post("/compare", summary="Extract two PDFs and compare them")
async def compare_pdfs_endpoint(
    previous_pdf: UploadFile = File(..., description="Previous period PDF"),
    current_pdf: UploadFile = File(..., description="Current period PDF"),
    pages: str | None = Form(
        None,
        description="Optional comma-separated 1-based page numbers to extract from both PDFs.",
    ),
) -> dict[str, Any]:
    """
    Upload previous and current PDFs, extract both, then run post_processing comparison.
    Returns extraction summaries and per-page comparison results.
    """
    prev_path = await _save_upload(previous_pdf, INPUT_DIR)
    curr_path = await _save_upload(current_pdf, INPUT_DIR)

    page_indices = parse_page_list(pages)
    try:
        prev_extract = extract_pdf(prev_path, page_numbers=page_indices)
        curr_extract = extract_pdf(curr_path, page_numbers=page_indices)
    except PDFExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}") from e

    processor = CompareExtractedData()
    try:
        results_dir, _ = processor.run_comparison(
            root_path=str(OUTPUT_DIR),
            folder_name_list=[prev_extract["folder_name"], curr_extract["folder_name"]],
            previous_pdf_filename=previous_pdf.filename,
            current_pdf_filename=current_pdf.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    page_results = _load_comparison_results(results_dir)

    return {
        "message": "Extraction and comparison complete",
        "previous": {
            "filename": previous_pdf.filename,
            "folder_name": prev_extract["folder_name"],
            "pages_processed": prev_extract["pages_processed"],
        },
        "current": {
            "filename": current_pdf.filename,
            "folder_name": curr_extract["folder_name"],
            "pages_processed": curr_extract["pages_processed"],
        },
        "comparison_results_directory": path_for_api(results_dir),
        "page_comparisons": page_results,
    }


@app.post("/compare-folders", summary="Compare previously extracted JSON folders")
async def compare_folders_endpoint(body: CompareFoldersRequest) -> dict[str, Any]:
    """
    Compare two folders under output/ that already contain page_*_extracted_data_cleaned.json files.
    """
    prev_dir = OUTPUT_DIR / body.previous_folder
    curr_dir = OUTPUT_DIR / body.current_folder
    if not prev_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {body.previous_folder}")
    if not curr_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {body.current_folder}")

    processor = CompareExtractedData()
    try:
        results_dir, current_folder = processor.run_comparison(
            root_path=str(OUTPUT_DIR),
            folder_name_list=[body.previous_folder, body.current_folder],
            previous_pdf_filename=body.previous_pdf_filename,
            current_pdf_filename=body.current_pdf_filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "message": "Comparison complete",
        "current_folder": current_folder,
        "comparison_results_directory": path_for_api(results_dir),
        "page_comparisons": _load_comparison_results(results_dir),
    }


def _load_comparison_results(results_dir: str) -> list[dict[str, Any]]:
    """Read comparison Excel outputs into JSON-serializable records."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return []

    page_results: list[dict[str, Any]] = []
    json_files = sorted(results_path.glob("comparison_result_*.json"))
    if json_files:
        for path in json_files:
            df = pd.read_json(path)
            page_results.append(
                {
                    "result_file": path.name,
                    "row_count": len(df),
                    "rows": _df_to_records(df),
                }
            )
    else:
        for path in sorted(results_path.glob("comparison_result_*.xlsx")):
            df = pd.read_excel(path)
            page_results.append(
                {
                    "result_file": path.name,
                    "row_count": len(df),
                    "rows": _df_to_records(df),
                }
            )
    return page_results
