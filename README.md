# Financial Reports Comparison

Upload financial report PDFs, extract structured data with a local Ollama model, and compare line items across reporting periods.

## Project layout (4 Python modules)

| File | Role |
|------|------|
| `config.py` | Paths (`input/`, `output/`) and Ollama settings |
| `extraction.py` | PDF → Ollama → JSON files; JSON → flat tables |
| `post_processing.py` | Fuzzy match two periods; write comparison results |
| `app.py` | FastAPI / Swagger UI |

## End-to-end flow

```
PDF upload (Swagger or input/)
        │
        ▼
  extraction.py  ──►  Ollama (one call per page)
        │
        ▼
output/<report_name>/page_<N>_extracted_data_cleaned.json   ← raw extraction per page
        │
        ▼
  post_processing.py  ──►  match same page numbers across two report folders
        │
        ▼
output/comparison_results/comparison_result_page_<N>.json   ← differences (API + disk)
output/comparison_results/comparison_result_page_<N>.xlsx   ← same data, Excel
```

### The three folders under `output/`

1. **`output/<previous_report>/`** — JSON from the **older** PDF (folder name = PDF filename without `.pdf`, e.g. `previous_report`). One file per page with financial data. These are **not** diffs; they are extracted line items for that period.

2. **`output/<current_report>/`** — JSON from the **newer** PDF (e.g. `current_report`). Same structure as above.

3. **`output/comparison_results/`** — **This is where changes are shown.** For each page number that exists in **both** report folders, the app writes:
   - `numeric_value_prev` — value from the previous report (common year column)
   - `numeric_value_curr` — value from the current report
   - Matched rows use fuzzy matching on section / parameter / heading, etc.

Pages are paired by number: `page_3` in the previous folder is compared to `page_3` in the current folder.

### Where to view results

| Location | What you see |
|----------|----------------|
| **Swagger** — `POST /compare` or `/compare-folders` response → `page_comparisons[].rows` | Side-by-side prev/curr numbers in JSON |
| **`output/comparison_results/*.json`** | Same comparison rows on disk |
| **`output/comparison_results/*.xlsx`** | Same data in Excel |

Extraction-only responses (`POST /extract`) show nested JSON in `pages` and a small `flat_preview` — not year-over-year diffs.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
ollama serve
ollama pull qwen2.5-coder:7b
```

## Run the API

```bash
python -m uvicorn app:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /extract` | One PDF → JSON under `output/<pdf_name>/` |
| `POST /compare` | Two PDFs → extract → compare → `page_comparisons` in response |
| `POST /compare-folders` | Compare existing folders (skip re-extraction) |
| `GET /health` | Health check |

Optional form field **`pages`**: comma-separated 1-based page numbers (e.g. `1,2,3,4,5`) to extract only those pages.

## Python usage (without API)

```python
from extraction import extract_pdf
from post_processing import CompareExtractedData

extract_pdf("input/previous_report.pdf")
extract_pdf("input/current_report.pdf")

processor = CompareExtractedData()
results_dir, _ = processor.run_comparison(
    root_path="./output",
    folder_name_list=["previous_report", "current_report"],
)
```

## License

Copyright (c) 2026 Shrey Gupta. All rights reserved.

Unauthorized copying, modification, or distribution of this software, via any medium, is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.