"""
Data Extraction Tool — Python/Pandas Sandbox.

This module is the "python_repl" equivalent tool in the pipeline.
It provides deterministic Pandas + Regex routines that ingest messy raw
corporate data and emit clean, structured JSON before any LLM reasoning begins.

Supported input types
─────────────────────
  "csv"   → Dirty ERP/bank ledger CSV export
             Handles: inconsistent dates, missing Ref IDs, amount whitespace,
             mixed-case vendor names, extra spaces in cell values.

  "invoice" → Messy unstructured invoice text (typos, missing colons, garbled labels)
             Handles: "Vndor:", "DAT:", "Total Amnt Due:", "DUE NOW::::"

  "query" → Plain-text question (pass-through, no cleaning needed)

Public API
──────────
    extract_and_clean(raw_data: str, input_type: str = "auto") -> dict
    detect_input_type(raw_data: str) -> str
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# ── Date Normalization ─────────────────────────────────────────────────────────

_DATE_FORMATS: List[str] = [
    "%Y-%m-%d",    # 2026-08-12   (ISO, 4-digit year first)
    "%Y.%m.%d",    # 2026.08.18
    "%Y/%m/%d",    # 2026/08/12
    "%m-%d-%Y",    # 08-16-2026
    "%d/%m/%Y",    # 14/08/2026
    "%m/%d/%Y",    # 08/14/2026
    "%b %d %Y",    # Aug 20 2026
    "%B %d %Y",    # August 20 2026
    "%y-%m-%d",    # 26-08-15  → 2026-08-15  (2-digit year, Month, Day)
    "%m/%d/%y",    # 08/14/26
    "%d-%m-%y",    # fallback: day-month-2digit-year (less common)
]


def _normalize_date(raw: str) -> str:
    """
    Attempt to parse raw date string into ISO YYYY-MM-DD.

    Special handling: when the raw value looks like 'NN-NN-NN' and the first
    segment is >= 20 (e.g. '26-08-15'), treat it as YY-MM-DD (year first),
    NOT DD-MM-YY. This avoids "26-08-15" becoming "2015-08-26".
    """
    raw = str(raw).strip()

    # Detect ambiguous DD-MM-YY vs YY-MM-DD: if first segment >= 20, it's a year
    ambiguous_match = re.match(r"^(\d{2})-(\d{2})-(\d{2})$", raw)
    if ambiguous_match:
        first = int(ambiguous_match.group(1))
        if first >= 20:  # First segment is plausibly a 2-digit year (2020+)
            try:
                return datetime.strptime(raw, "%y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                pass

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw  # Return original if unparseable


def _normalize_amount(raw: Any) -> Optional[float]:
    """Strip whitespace, commas, currency symbols; cast to float."""
    cleaned = str(raw).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


# ── CSV Cleaning ───────────────────────────────────────────────────────────────

def clean_csv_data(csv_text: str) -> Dict[str, Any]:
    """
    Ingest a messy ERP CSV export and return structured JSON with:
      - normalized dates (ISO), amounts (float), vendor names (uppercase)
      - list of high-risk transactions (|amount| > $2,000)
      - vendor spend summary for quick policy cross-reference
      - cleaning audit notes for transparency
    """
    cleaning_notes: List[str] = []

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return {
            "type": "csv_transactions",
            "error": str(e),
            "records": [],
            "record_count": 0,
        }

    # 1. Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]
    cleaning_notes.append("Stripped whitespace from all column names")

    # 2. Strip all string cell values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    cleaning_notes.append("Stripped leading/trailing whitespace from all string cells")

    # 3. Normalize date column
    if "Txn_Date" in df.columns:
        df["Txn_Date"] = df["Txn_Date"].apply(_normalize_date)
        cleaning_notes.append("Normalized Txn_Date to ISO YYYY-MM-DD (handled 7 different source formats)")

    # 4. Normalize amount column (strip spaces, commas, currency symbols)
    if "Amt" in df.columns:
        df["Amt"] = df["Amt"].apply(_normalize_amount)
        cleaning_notes.append("Converted Amt to numeric float (removed spaces, commas)")

    # 5. Standardize vendor descriptions (uppercase + collapse whitespace)
    if "Vendor_Desc" in df.columns:
        df["Vendor_Desc"] = (
            df["Vendor_Desc"]
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        cleaning_notes.append("Uppercased Vendor_Desc and collapsed internal whitespace")

    # 6. Normalize status to uppercase
    if "Status" in df.columns:
        df["Status"] = df["Status"].str.upper().str.strip()
        cleaning_notes.append("Uppercased Status column values")

    # 7. Fill missing Ref_IDs with UNKNOWN
    if "Ref_ID" in df.columns:
        df["Ref_ID"] = df["Ref_ID"].replace({"nan": None, "": None}).fillna("UNKNOWN")
        cleaning_notes.append("Replaced missing Ref_IDs with 'UNKNOWN'")

    # 8. Fill missing Dept_Code with UNSPECIFIED
    if "Dept_Code" in df.columns:
        df["Dept_Code"] = df["Dept_Code"].replace({"nan": None, "": None}).fillna("UNSPECIFIED")
        cleaning_notes.append("Replaced missing Dept_Code with 'UNSPECIFIED'")

    # Serialize (replace NaN with None for JSON safety)
    records: List[Dict] = df.where(pd.notna(df), None).to_dict(orient="records")

    # Identify high-risk transactions (|amount| > $2,000 per policy Section 4.2.1)
    high_risk = [
        r for r in records
        if isinstance(r.get("Amt"), (int, float)) and abs(r["Amt"]) > 2000
    ]

    # Compute total absolute spend
    total_spend = sum(
        abs(r["Amt"])
        for r in records
        if isinstance(r.get("Amt"), (int, float))
    )

    # Vendor spend summary
    vendor_spend: Dict[str, float] = {}
    for r in records:
        vendor = str(r.get("Vendor_Desc", "UNKNOWN"))
        amt = r.get("Amt")
        if isinstance(amt, (int, float)):
            vendor_spend[vendor] = round(vendor_spend.get(vendor, 0.0) + abs(amt), 2)

    return {
        "type": "csv_transactions",
        "record_count": len(records),
        "records": records,
        "high_risk_transactions": high_risk,
        "vendor_spend_summary": vendor_spend,
        "total_absolute_spend": round(total_spend, 2),
        "cleaning_notes": cleaning_notes,
    }


# ── Invoice Text Cleaning ──────────────────────────────────────────────────────

def _ocr_normalize(text: str) -> str:
    """
    Pre-process text for common OCR artefacts before field extraction:
      - Capital-O used instead of zero inside digit sequences (e.g. '2O26' → '2026')
      - Spaces inside digit/amount sequences (e.g. '$ 1 , 8 0 0 . 0 0' → '$1,800.00')
    """
    # Replace capital-O that is surrounded by digits with zero
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)O(?=\s*[\-\/])', '0', text)
    text = re.sub(r'(?<=[\-\/])O(?=\d)', '0', text)
    # Collapse spaces inside dollar amounts: '$ 1 , 8 0 0 . 0 0' → '$1,800.00'
    text = re.sub(r'\$\s+((?:\d[\d ,\.]*\d|\d))', lambda m: '$' + m.group(1).replace(' ', ''), text)
    return text


def clean_invoice_data(invoice_text: str) -> Dict[str, Any]:
    """
    Parse unstructured invoice text using regex heuristics to extract:
    invoice_number, date, vendor, total_amount, payment_terms, department.

    Handles messy labels: missing/extra colons, abbreviations
    ("Vndor:", "DAT:"), OCR artefacts (O vs 0, spaced digits), and garbled amounts.
    Now also pre-computes all policy violation flags DETERMINISTICALLY so the
    Analyst LLM only writes prose — it never needs to do math.
    """
    raw_text = invoice_text.strip()
    text = _ocr_normalize(raw_text)   # Apply OCR fixes first
    cleaning_notes: List[str] = ['Applied OCR normalization (O→0, spaced digit collapse)']

    # ── Invoice Number ──────────────────────────────────────────────────────────
    inv_match = re.search(
        r"(?:INVOICE|INV(?:OICE)?)[#\s\.:=]*([A-Z0-9\-]+)",
        text,
        re.IGNORECASE,
    )
    invoice_number = inv_match.group(1).strip() if inv_match else "UNKNOWN"
    cleaning_notes.append(f"Invoice number extracted: {invoice_number}")

    # ── Date ────────────────────────────────────────────────────────────────────
    date_match = re.search(
        r"(?:DAT[E]?|Date|Issued|Invoice\s+Date)\s*[:\-=\s]+([0-9O\/\-\.]+)",
        text,
        re.IGNORECASE,
    )
    raw_date = date_match.group(1).strip() if date_match else "UNKNOWN"
    # Second-pass OCR fix: O→0 in the extracted date value itself
    raw_date = re.sub(r'O', '0', raw_date)
    invoice_date = _normalize_date(raw_date) if raw_date != "UNKNOWN" else "UNKNOWN"
    cleaning_notes.append(f"Date normalized: '{raw_date}' → '{invoice_date}'")

    # ── Vendor (handles "Vndor:", "Vendor:", "FROM:", "-- Vendor --", or first non-empty header line) ──
    vendor_match = re.search(
        r"(?:Vendor|Vndor|FROM|Supplier)\s*[:\-=]\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if not vendor_match:
        # Fallback for "-- Datadog, Inc. --"
        vendor_match = re.search(r"--\s*([^\n\-]+?)\s*--", text)
    if not vendor_match:
        # Fallback: use first non-empty line (often the company header)
        first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if first_lines:
            # Collapse spaced-out letters: "Z O O M" → "ZOOM"
            candidate = re.sub(r'(?<=[A-Z]) (?=[A-Z])', '', first_lines[0])
            candidate = re.sub(r'[\-\*=]+', '', candidate).strip()
            if len(candidate) > 2:
                vendor_match = type('m', (), {'group': lambda self, n: candidate})()
    vendor = vendor_match.group(1).strip() if vendor_match else "UNKNOWN"
    # Remove trailing punctuation/commas
    vendor = re.sub(r"[,\.\s]+$", "", vendor)
    # Collapse spaced-out company names from OCR ("Z O O M" → "ZOOM")
    vendor = re.sub(r'(?<=[A-Z]) (?=[A-Z])', '', vendor)
    cleaning_notes.append(f"Vendor extracted: '{vendor}'")

    # ── Total Amount (handles "TOTAL DUE", "AMNT DUE", "AMOUNT OWING", etc.) ────
    # After OCR normalization, spaced amounts like '$1,800.00' are already collapsed.
    amount_match = re.search(
        r"(?:TOTAL[^:\n]*|AMNT\s+DUE|AMOUNT\s*(?:DUE|OWING))[^\d]*?([\d,\.]+)",
        text,
        re.IGNORECASE,
    )
    if not amount_match:
        # Fallback: look for Amount: or any $-prefixed number
        amount_match = re.search(r"Amount\s*:\s*\$?\s*([\d,\.]+)", text, re.IGNORECASE)
    amount_str = amount_match.group(1).replace(",", "").strip() if amount_match else "0"
    try:
        total_amount = float(amount_str)
    except ValueError:
        total_amount = 0.0
    cleaning_notes.append(f"Total amount extracted: ${total_amount:,.2f}")

    # ── Payment Terms ─────────────────────────────────────────────────────────
    net_match = re.search(r"Net[-\s]?(\d+)", text, re.IGNORECASE)
    receipt_match = re.search(r"due\s+upon\s+receipt", text, re.IGNORECASE)
    now_match = re.search(r"DUE\s+NOW", text, re.IGNORECASE)

    if net_match:
        payment_terms = f"Net-{net_match.group(1)}"
    elif receipt_match or now_match:
        payment_terms = "Due Upon Receipt"
    else:
        payment_terms = "UNKNOWN"
    cleaning_notes.append(f"Payment terms: {payment_terms}")

    # ── Department ─────────────────────────────────────────────────────────────
    dept_match = re.search(
        r"(?:Dept|Department|ATTN|Bill To)\s*[:\-=]?\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    department = dept_match.group(1).strip() if dept_match else "UNSPECIFIED"

    # ── Deterministic policy violation flags (computed in Python, NOT by the LLM) ──
    # These flags are the ground truth. The Analyst only writes prose around them.
    exceeds_2000_threshold  = total_amount > 2000.0
    exceeds_5000_threshold  = total_amount > 5000.0
    payment_terms_violation = (payment_terms == "Due Upon Receipt") and exceeds_2000_threshold
    requires_vp_auth        = exceeds_5000_threshold
    is_compliant            = not payment_terms_violation and not requires_vp_auth

    violation_summary: List[str] = []
    if exceeds_5000_threshold:
        violation_summary.append(
            f"REQUIRES VP AUTHORIZATION: ${total_amount:,.2f} exceeds the $5,000 threshold (Section 4.2.1)"
        )
    if payment_terms_violation:
        violation_summary.append(
            f"PAYMENT TERMS VIOLATION: Invoice is '{payment_terms}' but policy requires Net-30 "
            f"for amounts over $2,000 (Section 4.2.2). Amount ${total_amount:,.2f} exceeds threshold."
        )
    if is_compliant:
        violation_summary.append(
            f"COMPLIANT: Amount ${total_amount:,.2f} is within limits and payment terms are acceptable."
        )

    return {
        "type": "invoice",
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "vendor": vendor,
        "total_amount": total_amount,
        "payment_terms": payment_terms,
        "department": department,
        # -- Deterministic compliance flags --
        "exceeds_2000_threshold": exceeds_2000_threshold,
        "exceeds_5000_threshold": exceeds_5000_threshold,
        "payment_terms_violation": payment_terms_violation,
        "requires_vp_authorization": requires_vp_auth,
        "is_compliant": is_compliant,
        "python_computed_violations": violation_summary,
        # -- Legacy keys kept for backward compat --
        "requires_policy_check": exceeds_2000_threshold or payment_terms_violation,
        "exceeds_authorization_limit": exceeds_5000_threshold,
        "raw_text_length": len(raw_text),
        "cleaning_notes": cleaning_notes,
    }


# ── Input Type Detection ───────────────────────────────────────────────────────

def detect_input_type(raw_data: str) -> str:
    """
    Heuristically detect whether raw_data is:
      - "csv"   → has comma-delimited header with known column names
      - "invoice" → contains INVOICE / TOTAL DUE / AMNT DUE keywords
      - "query"   → plain text (default fallback)
    """
    first_line = raw_data.strip().split("\n")[0].lower()

    csv_keywords = {"txn_date", "ref_id", "vendor_desc", "amt", "status", "dept_code"}
    if "," in first_line and any(kw in first_line for kw in csv_keywords):
        return "csv"

    invoice_keywords = [
        "invoice", "billed to", "amnt due", "amount due",
        "total due", "total amnt", "vndor", "vend",
    ]
    if any(kw in raw_data.lower() for kw in invoice_keywords):
        return "invoice"

    return "query"


# ── Public Entry Point ─────────────────────────────────────────────────────────

def extract_and_clean(raw_data: str, input_type: str = "auto") -> Dict[str, Any]:
    """
    Main entry point for the Data Extraction tool.

    Args:
        raw_data:   The raw input string (CSV text, invoice text, or plain query).
        input_type: "csv" | "invoice" | "query" | "auto" (auto-detect).

    Returns:
        Structured JSON-serializable dict ready for the RAG retriever and analyst.
    """
    if input_type == "auto":
        input_type = detect_input_type(raw_data)

    if input_type == "csv":
        return clean_csv_data(raw_data)
    elif input_type == "invoice":
        return clean_invoice_data(raw_data)
    else:
        return {
            "type": "plain_query",
            "query": raw_data.strip(),
            "cleaning_notes": [
                "No structured data detected; treating as plain text query"
            ],
        }
