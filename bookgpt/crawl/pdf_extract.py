"""PDF text extraction and cleaning utilities."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a PDF file using pdfplumber with PyMuPDF fallback.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted and cleaned text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = _try_pdfplumber(pdf_path)
    if not text or len(text.strip()) < 100:
        logger.info(f"pdfplumber yielded little text, trying PyMuPDF for {pdf_path.name}")
        text = _try_pymupdf(pdf_path)

    if not text:
        logger.warning(f"No text extracted from {pdf_path.name}")
        return ""

    return clean_text(text)


def _try_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {e}")
        return ""


def _try_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz

        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text())
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {pdf_path.name}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Clean extracted text: remove artifacts, normalize whitespace.

    Preserves LaTeX notation and mathematical Unicode.
    """
    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Remove repeated header/footer patterns (lines that repeat every ~page)
    lines = text.split("\n")
    if len(lines) > 50:
        text = _remove_repeated_lines(lines)

    # Fix common PDF extraction artifacts
    # Rejoin hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse multiple blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize whitespace within lines (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove very short lines that are likely artifacts (< 3 chars, not math symbols)
    cleaned_lines = []
    for line in text.split("\n"):
        if len(line) < 3 and not re.match(r"^[=+\-*/^∑∫∏√≤≥≠±∞πθαβγδε]$", line):
            if line.strip():
                continue  # skip very short non-math artifact lines
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    return text.strip()


def _remove_repeated_lines(lines: list[str], threshold: int = 3) -> str:
    """Remove lines that appear more than `threshold` times (likely headers/footers)."""
    from collections import Counter

    line_counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in line_counts.items() if count >= threshold and len(line) < 100}

    filtered = [line for line in lines if line.strip() not in repeated]
    return "\n".join(filtered)
