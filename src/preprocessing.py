"""
Data Preprocessing Module for XAI-WebGuard
Handles text cleaning, URL decoding, HTML unescaping, canonicalization,
deduplication, and label harmonization for Benign, SQLi, and XSS requests.
"""

import re
import html
import urllib.parse
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


def clean_and_normalize(text: Optional[str]) -> str:
    """
    Cleans and canonicalizes an HTTP query string, URL, or payload:
    1. Multi-pass URL decoding (handles single and double percent-encoding).
    2. HTML entity unescaping (e.g., &quot; -> ", &lt; -> <).
    3. Normalizes whitespace and removes control characters.
    4. Preserves all critical syntactic symbols used in SQLi and XSS.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Pass 1 & 2: URL Decoding (handles double-encoded payloads like %2527 -> %27 -> ')
    try:
        decoded = urllib.parse.unquote(text)
        decoded = urllib.parse.unquote(decoded)
    except Exception:
        decoded = text

    # HTML Entity Decoding (e.g., &lt;script&gt; -> <script>)
    try:
        decoded = html.unescape(decoded)
    except Exception:
        pass

    # Replace newlines, tabs, and carriage returns with spaces
    decoded = re.sub(r'[\r\n\t]+', ' ', decoded)

    # Filter out non-printable ASCII/control chars, keeping standard printable characters (32 to 126)
    decoded = re.sub(r'[^\x20-\x7E]+', ' ', decoded)

    # Collapse multiple whitespace
    decoded = re.sub(r'\s+', ' ', decoded).strip()

    return decoded


def extract_csic_requests(csic_df: pd.DataFrame, target_class: str = "Valid") -> List[str]:
    """
    Extracts HTTP request strings from the CSIC 2010 dataset.
    Combines URI, GET query parameters, and POST body data.
    """
    filtered = csic_df[csic_df['Class'] == target_class]
    samples = []
    
    for _, row in filtered.iterrows():
        uri = str(row['URI']) if pd.notna(row.get('URI')) else ''
        get_q = str(row['GET-Query']) if pd.notna(row.get('GET-Query')) else ''
        post_d = str(row['POST-Data']) if pd.notna(row.get('POST-Data')) else ''

        parts = []
        if uri and uri != 'nan':
            parts.append(uri)
        if get_q and get_q != 'nan':
            parts.append(f"?{get_q}")
        if post_d and post_d != 'nan':
            parts.append(f"body={post_d}")

        if parts:
            combined = " ".join(parts)
            cleaned = clean_and_normalize(combined)
            if len(cleaned) >= 4:
                samples.append(cleaned)

    return list(dict.fromkeys(samples))  # preserve order deduplication
