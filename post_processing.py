"""
Financial Reports Comparison - Post Processing Module

This module compares extracted data from financial reports across different periods,
matching rows based on similarity scores and handling missing data intelligently.
"""
import re
import os
import logging
import numpy as np
import pandas as pd
from recordlinkage import Index, Compare

from extraction import load_extraction_json_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

SCORE_WEIGHTS = {
    'section_score': 0.05,
    'param_score': 0.7,
    'title_score': 0.05,
    'heading_score': 0.1,
    'column_score': 0.1
}

DEFAULT_SIMILARITY_METHOD = "jarowinkler"
DEFAULT_SECTION_THRESHOLD = 0.9
DEFAULT_PARAM_THRESHOLD = 0.9

DATE_PATTERNS = [
    r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
    r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',
    r'\b\d{4}-\d{1,2}-\d{1,2}\b',
]

MISSING_VALUE_INDICATORS = ("", "-", "—", "–", "NaN", "nan", "N/A", "n/a")

PAGE_FILE_PATTERN = re.compile(r"page_(\d+)", re.IGNORECASE)
CLEANED_JSON_SUFFIX = "_extracted_data_cleaned.json"


def normalize_text(s):
    """
    Normalize text for comparison by converting to lowercase and removing special characters.
    
    Args:
        s: Text to normalize (handles NaN, None, and non-string types)
        
    Returns:
        str: Normalized text with only alphanumerics and spaces
    """
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep only alphanumerics and spaces
    s = re.sub(r"\s+", " ", s).strip()   # collapse multiple spaces
    return s

def calculate_dynamic_score(features, prev_df, curr_df):
    """
    Calculate dynamic scoring weights based on column availability.
    
    Handles all possible combinations of empty columns and redistributes weights
    among remaining non-empty columns (section, param, title, heading, column).
    
    Args:
        features: DataFrame with similarity scores for each column
        prev_df: Previous period DataFrame with normalized columns
        curr_df: Current period DataFrame with normalized columns
    
    Returns:
        Series: Calculated scores using dynamic weights
    """
    def is_column_empty(df, col_name):
        """Check if a column is effectively empty (all NaN or empty strings)"""
        if col_name not in df.columns:
            return True
        return (df[col_name].isna().all() or 
                df[col_name].astype(str).str.strip().eq("").all())
    
    adjusted_weights = SCORE_WEIGHTS.copy()
    empty_columns = []
    
    # Check standard columns for emptiness
    column_checks = [
        ('section_score', '_section_norm'),
        ('param_score', '_param_norm'),
        ('title_score', '_title_norm'),
        ('heading_score', '_heading_norm')
    ]
    
    for score_col, norm_col in column_checks:
        if is_column_empty(prev_df, norm_col) and is_column_empty(curr_df, norm_col):
            empty_columns.append(score_col)
    
    # Special check for column_score
    if (is_column_empty(prev_df, '_column_norm') and is_column_empty(curr_df, '_column_norm')):
        empty_columns.append('column_score')
    
    # Redistribute weights from empty columns
    total_removed_weight = sum(adjusted_weights.get(col, 0) for col in empty_columns)
    for col in empty_columns:
        adjusted_weights[col] = 0
    
    if total_removed_weight > 0:
        non_empty_columns = [col for col in SCORE_WEIGHTS.keys() if col not in empty_columns]
        if non_empty_columns:
            weight_per_column = total_removed_weight / len(non_empty_columns)
            for col in non_empty_columns:
                adjusted_weights[col] += weight_per_column
    
    # Calculate final score using vectorized operations
    score = (
        adjusted_weights['section_score'] * features['section_score'] +
        adjusted_weights['param_score'] * features['param_score'] +
        adjusted_weights['title_score'] * features['title_score'] +
        adjusted_weights['heading_score'] * features['heading_score'] +
        adjusted_weights['column_score'] * features['column_score']
    )
    
    return score

def compare_extracted_results(
    df_prev, df_curr,
    target_year,
    year_col="year", section_col="section", title_col="table_title",
    heading_col="heading", column_col="column",
    param_col="parameter", value_col="value",
    section_threshold=DEFAULT_SECTION_THRESHOLD, param_threshold=DEFAULT_PARAM_THRESHOLD
):
    """
    Compare extracted data from two periods and match rows based on similarity.
    
    Args:
        df_prev: Previous period DataFrame
        df_curr: Current period DataFrame
        target_year: Year to filter comparison to
        year_col: Name of year column
        section_col: Name of section column
        title_col: Name of title column
        heading_col: Name of heading column
        column_col: Name of column identifier
        param_col: Name of parameter column
        value_col: Name of value column
        section_threshold: Similarity threshold for section matching
        param_threshold: Similarity threshold for parameter matching
        
    Returns:
        DataFrame: Comparison results with matched and unmatched rows
        
    Raises:
        ValueError: If required columns are missing
    """
    # Validate input dataframes
    required_cols = {year_col, section_col, title_col, heading_col, column_col, param_col, value_col}
    missing_in_prev = required_cols - set(df_prev.columns)
    missing_in_curr = required_cols - set(df_curr.columns)
    
    if missing_in_prev:
        raise ValueError(f"Missing columns in previous DataFrame: {missing_in_prev}")
    if missing_in_curr:
        raise ValueError(f"Missing columns in current DataFrame: {missing_in_curr}")
    
    # Filter to only include rows with the target year
    prev = df_prev[df_prev[year_col] == target_year].copy().reset_index(drop=True)
    curr = df_curr[df_curr[year_col] == target_year].copy().reset_index(drop=True)
    
    if prev.empty or curr.empty:
        logger.warning(f"Empty DataFrame after filtering to year {target_year}")
        return pd.DataFrame()

    # Normalize & clean numbers
    for df in [prev, curr]:
        df["_section_norm"] = df[section_col].apply(normalize_text)
        df["_title_norm"] = df[title_col].apply(normalize_text)
        df["_heading_norm"] = df[heading_col].apply(normalize_text)
        df["_column_norm"] = df[column_col].apply(normalize_text)
        df["_param_norm"] = df[param_col].apply(normalize_text)
        df["_num"] = df[value_col].apply(lambda x: 0 if str(x).strip() == "-" else x)

    # Candidate pairs only within same year
    indexer = Index()
    indexer.block(year_col)
    candidate_links = indexer.index(prev, curr)

    compare = Compare()
    compare.string("_section_norm", "_section_norm", method=DEFAULT_SIMILARITY_METHOD, threshold=None, label="section_score")
    compare.string("_title_norm", "_title_norm", method=DEFAULT_SIMILARITY_METHOD, threshold=None, label="title_score")
    compare.string("_heading_norm", "_heading_norm", method=DEFAULT_SIMILARITY_METHOD, threshold=None, label="heading_score")
    compare.string("_column_norm", "_column_norm", method=DEFAULT_SIMILARITY_METHOD, threshold=None, label="column_score")
    compare.string("_param_norm", "_param_norm", method=DEFAULT_SIMILARITY_METHOD, threshold=None, label="param_score")
    features = compare.compute(candidate_links, prev, curr).reset_index()

    # Add similarity score with dynamic weight calculation
    features["score"] = calculate_dynamic_score(features, prev, curr)
    
    # Keep best match per prev and per curr
    matches = (
        features
        .sort_values("score", ascending=False)
        .drop_duplicates("level_0")
        .drop_duplicates("level_1")
    )

    # Merge back
    merged = (
        matches
        .merge(prev.reset_index(), left_on="level_0", right_on="index", how="left")
        .merge(curr.reset_index(), left_on="level_1", right_on="index", how="left", suffixes=("_prev","_curr"))
    )

    # Build output with outer join
    out = pd.DataFrame({
        "year": merged[f"{year_col}_curr"].where(merged[f"{year_col}_curr"].notna(), merged[f"{year_col}_prev"]).astype("Int64"),
        "section": merged[f"{section_col}_curr"].where(merged[f"{section_col}_curr"].notna(), merged[f"{section_col}_prev"]),
        "title": merged[f"{title_col}_curr"].where(merged[f"{title_col}_curr"].notna(), merged[f"{title_col}_prev"]),
        "heading": merged[f"{heading_col}_curr"].where(merged[f"{heading_col}_curr"].notna(), merged[f"{heading_col}_prev"]),
        "column": merged[f"{column_col}_curr"].where(merged[f"{column_col}_curr"].notna(), merged[f"{column_col}_prev"]),
        "parameter": merged[f"{param_col}_curr"].where(merged[f"{param_col}_curr"].notna(), merged[f"{param_col}_prev"]),
        "numeric_value_prev": merged["_num_prev"],
        "numeric_value_curr": merged["_num_curr"]
    })

    # Add rows only in prev (no match in curr)
    only_prev = prev.loc[~prev.index.isin(matches["level_0"])]
    for_add_prev = pd.DataFrame({
        "year": only_prev[year_col],
        "section": only_prev[section_col],
        "title": only_prev[title_col],
        "heading": only_prev[heading_col],
        "column": only_prev[column_col],
        "parameter": only_prev[param_col],
        "numeric_value_prev": only_prev["_num"],
        "numeric_value_curr": np.nan
    })

    # Add rows only in curr (no match in prev)
    only_curr = curr.loc[~curr.index.isin(matches["level_1"])]
    for_add_curr = pd.DataFrame({
        "year": only_curr[year_col],
        "section": only_curr[section_col],
        "title": only_curr[title_col],
        "heading": only_curr[heading_col],
        "column": only_curr[column_col],
        "parameter": only_curr[param_col],
        "numeric_value_prev": np.nan,
        "numeric_value_curr": only_curr["_num"]
    })

    # Final outer join style result
    final = pd.concat([out, for_add_prev, for_add_curr], ignore_index=True)
    final = final[final["year"] == int(target_year)].reset_index(drop=True)
    final["numeric_value_1_num"] = final["numeric_value_prev"].apply(clean_number_val)
    final["numeric_value_2_num"] = final["numeric_value_curr"].apply(clean_number_val)

    # Replace missing values with 0 and treat NaN and 0 as the same
    final["numeric_value_1_num"] = final["numeric_value_1_num"].fillna(0)
    final["numeric_value_2_num"] = final["numeric_value_2_num"].fillna(0)

    return final

def get_common_year(corrected_df1, corrected_df2):
    """
    Return the latest common year between two DataFrames.
    
    Args:
        corrected_df1: First DataFrame with 'year' column
        corrected_df2: Second DataFrame with 'year' column
        
    Returns:
        int: Latest common year, or None if no common years exist
    """
    years1 = set(corrected_df1['year'].dropna().astype(int).unique())
    years2 = set(corrected_df2['year'].dropna().astype(int).unique())
    common_years = sorted(years1.intersection(years2))
    return common_years[-1] if common_years else None


def extract_year_from_row_vectorized(df, columns):
    """
    Vectorized year extraction from multiple columns.
    Much faster than apply(axis=1) for large datasets.
    
    Args:
        df: DataFrame with potential year values
        columns: List of column names to search for years
        
    Returns:
        Series: Extracted years
    """
    if "year" not in df.columns:
        df = df.copy()
        df["year"] = pd.NA
    result = df["year"].copy()
    mask = result.isna()

    for col in columns:
        if not mask.any():
            break
        if col not in df.columns:
            continue

        years = df.loc[mask, col].astype(str).str.extract(
            f"({YEAR_PATTERN.pattern})", expand=False
        )
        if isinstance(years, pd.DataFrame):
            years = years.iloc[:, 0]
        years = pd.to_numeric(years, errors="coerce")
        
        # Fill missing values in result
        fill_mask = mask & years.notna()
        result.loc[fill_mask] = years[fill_mask]
        mask = result.isna()
    
    return result


def remove_duplicate_year_vectorized(df, columns):
    """
    Vectorized removal of duplicate years from columns.
    
    Args:
        df: DataFrame to process
        columns: Columns to remove year values from
        
    Returns:
        DataFrame: Modified copy with years removed from specified columns
    """
    df = df.copy()

    def strip_matching_year(row, col_name):
        text = str(row[col_name])
        if pd.isna(row.get("year")):
            return text.strip()
        year_s = str(int(row["year"]))
        return YEAR_PATTERN.sub(
            lambda m: "" if m.group(0) == year_s else m.group(0), text
        ).strip()

    for col in columns:
        if col in df.columns:
            df[col] = df.apply(lambda row: strip_matching_year(row, col), axis=1)

    return df

def clean_number_val(val):
    """
    Normalize numeric text for comparison.
    
    Examples:
        '20 074' → 20074.0
        '20,074' → 20074.0
        '(33 499)' → -33499.0
        'Rs. 20,074' → 20074.0
        '-', '', NaN → 0.0
    
    Args:
        val: Value to normalize (any type)
        
    Returns:
        float: Normalized numeric value, 0.0 for missing values
    """
    if pd.isna(val):
        return 0.0

    text = str(val).strip()

    # Handle empty strings or common missing indicators
    if text in MISSING_VALUE_INDICATORS:
        return 0.0

    is_negative = False
    if re.match(r'^\(.*\)$', text):
        is_negative = True
        text = text.strip("()")

    text = re.sub(r"[^0-9.\-]", "", text)

    if text == "" or text == "-":
        return 0.0

    try:
        num = float(text)
        if is_negative:
            num = -num
        return num
    except ValueError:
        return 0.0


def remove_duplicate_text_between_columns(df, col1, col2):
    """
    Vectorized removal of duplicate text between two columns.
    
    If col1 and col2 have the same text (case-insensitive, after stripping),
    clear col2 so the value appears only once.
    
    Args:
        df: DataFrame to process
        col1: Source column to compare
        col2: Target column to clear if duplicate
        
    Returns:
        DataFrame: Modified copy
    """
    df = df.copy()
    
    # Vectorized comparison
    col1_clean = df[col1].astype(str).str.strip().str.lower()
    col2_clean = df[col2].astype(str).str.strip().str.lower()
    
    mask = (col1_clean == col2_clean) & (col1_clean != "")
    df.loc[mask, col2] = ""
    
    return df

def fill_parameter_from_heading(df, parameter_col='parameter', heading_col='heading'):
    """
    Fill empty parameter column with heading data.
    
    If the parameter column is empty (NaN or blank) but heading has data,
    copy heading into parameter, then clear heading.
    
    Args:
        df: DataFrame to process
        parameter_col: Column name for parameters
        heading_col: Column name for headings
        
    Returns:
        DataFrame: Modified copy
    """
    df = df.copy()
    
    if parameter_col not in df.columns or heading_col not in df.columns:
        return df
    
    # Vectorized operations
    param_empty = df[parameter_col].isna() | (df[parameter_col].astype(str).str.strip() == "")
    heading_has = (~df[heading_col].isna()) & (df[heading_col].astype(str).str.strip() != "")
    mask = param_empty & heading_has
    
    if mask.any():
        df.loc[mask, parameter_col] = df.loc[mask, heading_col]
        df.loc[mask, heading_col] = ""
    
    return df


def process_section_col(df):
    """
    Remove date patterns from section column.
    
    Examples: '31 Dec 2020', '30 March 2021', '31/12/2020', etc.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame: Modified copy with dates removed from section column
    """
    df = df.copy()
    
    def clean_section_text(text):
        if pd.isna(text):
            return text
        
        text = str(text)
        for pattern in DATE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else None
    
    df['section'] = df['section'].apply(clean_section_text)
    return df


def _page_index_from_filename(filename: str) -> str | None:
    """Return page number string from filenames like page_9_extracted_data_cleaned.json."""
    match = PAGE_FILE_PATTERN.search(filename)
    return match.group(1) if match else None


def _index_folder_pages(folder_path: str) -> dict[str, str]:
    """
    Map page number -> file path. Prefer cleaned JSON over Excel.
    """
    json_by_page: dict[str, str] = {}
    xlsx_by_page: dict[str, str] = {}

    if not os.path.isdir(folder_path):
        return {}

    for filename in os.listdir(folder_path):
        page = _page_index_from_filename(filename)
        if not page:
            continue
        full_path = os.path.join(folder_path, filename)
        if filename.endswith(CLEANED_JSON_SUFFIX):
            json_by_page[page] = full_path
        elif filename.endswith(".xlsx") and not filename.startswith("~$"):
            xlsx_by_page[page] = full_path

    merged = dict(xlsx_by_page)
    merged.update(json_by_page)
    return merged


def load_period_dataframe(file_path: str) -> pd.DataFrame:
    """Load a period file from cleaned extraction JSON or legacy Excel."""
    if file_path.endswith(".json"):
        return load_extraction_json_file(file_path)
    return pd.read_excel(file_path)


class CompareExtractedData:
    """
    Compare extracted financial data from two reporting periods.
    
    This class handles the entire workflow of matching and comparing financial
    line items across different time periods, identifying changes and discrepancies.
    """
    
    def _extract_year_from_filename(self, filename):
        """
        Extract year from filename using regex pattern.
        
        Args:
            filename: Filename or folder name to extract year from
            
        Returns:
            int: Extracted year, or empty string if not found
        """
        match = YEAR_PATTERN.search(filename)
        if match:
            return int(match.group(1))
        return ''
    
    def run_comparison(self, root_path, folder_name_list, previous_pdf_filename=None, current_pdf_filename=None):
        """
        Run comparison on extracted data from two folders.
        
        Args:
            root_path: Root directory containing the data folders
            folder_name_list: List of two folder names to compare
            previous_pdf_filename: Optional explicit name of previous PDF
            current_pdf_filename: Optional explicit name of current PDF
            
        Returns:
            tuple: (comparison_results_directory, current_folder_name)
            
        Raises:
            ValueError: If folder_name_list doesn't contain exactly 2 folders
        """
        if len(folder_name_list) != 2:
            raise ValueError(f"Expected 2 folders, got {len(folder_name_list)}")
        
        logger.info(f"Starting comparison with folders: {folder_name_list}")
        logger.info(f"Previous PDF: {previous_pdf_filename}, Current PDF: {current_pdf_filename}")
        
        # Determine which folder is previous and which is current
        prev_file, curr_file = self._match_folders_to_periods(folder_name_list, previous_pdf_filename, current_pdf_filename)
        
        logger.info(f"Final assignment: prev_file={prev_file}, curr_file={curr_file}")
        
        # Create output directory
        comparison_results_dir = os.path.join(root_path, "comparison_results")
        os.makedirs(comparison_results_dir, exist_ok=True)
        
        prev_pages = _index_folder_pages(os.path.join(root_path, prev_file))
        curr_pages = _index_folder_pages(os.path.join(root_path, curr_file))
        common_pages = sorted(set(prev_pages) & set(curr_pages), key=int)

        if not common_pages:
            logger.warning("No matching page files found between periods")
            return comparison_results_dir, curr_file

        for page in common_pages:
            self._process_comparison_file(
                prev_pages[page],
                curr_pages[page],
                page,
                comparison_results_dir,
            )

        return comparison_results_dir, curr_file

    def run_comparison_in_memory(
        self,
        df_prev: pd.DataFrame,
        df_curr: pd.DataFrame,
        page_label: str = "combined",
    ) -> pd.DataFrame:
        """
        Run comparison on two DataFrames (e.g. from API extraction) and return results.
        """
        if df_curr.empty:
            logger.warning("Current DataFrame is empty")
            return pd.DataFrame()

        df_prev = process_section_col(df_prev.copy())
        df_curr = process_section_col(df_curr.copy())
        df_prev = fill_parameter_from_heading(df_prev)
        df_curr = fill_parameter_from_heading(df_curr)

        columns_to_search = ["table_title", "heading"]
        df_prev["year"] = extract_year_from_row_vectorized(df_prev, columns_to_search)
        df_curr["year"] = extract_year_from_row_vectorized(df_curr, columns_to_search)
        df_prev = remove_duplicate_year_vectorized(df_prev, ["column"])
        df_curr = remove_duplicate_year_vectorized(df_curr, ["column"])

        common_year = get_common_year(df_prev, df_curr)
        if common_year is None:
            logger.warning("No common year found")
            return pd.DataFrame()

        df_prev = remove_duplicate_text_between_columns(df_prev, "parameter", "heading")
        df_prev = remove_duplicate_text_between_columns(df_prev, "heading", "table_title")
        df_curr = remove_duplicate_text_between_columns(df_curr, "parameter", "heading")
        df_curr = remove_duplicate_text_between_columns(df_curr, "heading", "table_title")

        result = compare_extracted_results(df_prev, df_curr, target_year=common_year)
        if not result.empty:
            result["page_number"] = page_label
        return result
    
    def _match_folders_to_periods(self, folder_list, prev_name=None, curr_name=None):
        """
        Match folder names to previous and current periods.
        
        Args:
            folder_list: List of two folder names
            prev_name: Optional explicit previous period name
            curr_name: Optional explicit current period name
            
        Returns:
            tuple: (previous_folder, current_folder)
        """
        # Try matching by explicit filenames
        if prev_name and curr_name:
            prev_file = next((f for f in folder_list if prev_name.lower().replace('.pdf', '') in f.lower()), None)
            curr_file = next((f for f in folder_list if curr_name.lower().replace('.pdf', '') in f.lower()), None)
            
            if prev_file and curr_file:
                logger.info(f"Matched by filename: {prev_name} → {prev_file}, {curr_name} → {curr_file}")
                return prev_file, curr_file
        
        # Fall back to year-based logic
        logger.info("Using year-based logic")
        year1 = self._extract_year_from_filename(folder_list[0])
        year2 = self._extract_year_from_filename(folder_list[1])
        logger.info(f"Extracted years: {year1}, {year2}")
        
        if year1 and year2:
            if year1 < year2:
                return folder_list[0], folder_list[1]
            elif year1 > year2:
                return folder_list[1], folder_list[0]
        
        # Default to folder order
        return folder_list[0], folder_list[1]
    
    def _process_comparison_file(self, prev_file_path, curr_file_path, page_label, output_dir):
        """
        Process a single comparison file pair (JSON or Excel).
        
        Args:
            prev_file_path: Path to previous period page file
            curr_file_path: Path to current period page file
            page_label: Page identifier for output naming
            output_dir: Output directory for results
        """
        if not os.path.exists(prev_file_path):
            logger.warning(f"Previous file not found: {prev_file_path}")
            return

        if not os.path.exists(curr_file_path):
            logger.warning(f"Current file not found: {curr_file_path}")
            return

        try:
            df_prev = load_period_dataframe(prev_file_path)
            df_curr = load_period_dataframe(curr_file_path)
            
            if df_curr.empty:
                logger.warning(f"Current DataFrame is empty for page {page_label}")
                return

            result = self.run_comparison_in_memory(df_prev, df_curr, page_label=f"page_{page_label}")

            if result.empty:
                logger.warning(f"Comparison resulted in empty DataFrame for page {page_label}")
                return

            base_name = f"comparison_result_page_{page_label}"
            result.to_excel(os.path.join(output_dir, f"{base_name}.xlsx"), index=False)
            result.to_json(
                os.path.join(output_dir, f"{base_name}.json"),
                orient="records",
                indent=2,
            )
            logger.info(f"Successfully processed page {page_label}")

        except Exception as e:
            logger.error(f"Error processing page {page_label}: {str(e)}", exc_info=True)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example usage:
    # root_path = "path/to/your/data"
    # processor = CompareExtractedData()
    # result_path, current_folder = processor.run_comparison(
    #     root_path=root_path,
    #     folder_name_list=['previous_report', 'current_report'],
    #     previous_pdf_filename='previous_report.pdf',
    #     current_pdf_filename='current_report.pdf'
    # )
    # print(f"Results saved to: {result_path}")
    pass
