"""Crawler for OpenStax free textbooks.

OpenStax textbooks are free under CC license. Since their website is
JavaScript-rendered (scraping HTML yields empty content), we download
the official PDF versions and extract text using pdfplumber/PyMuPDF.
"""

import logging
import re
import tempfile
import time
from pathlib import Path

import requests

from bookgpt.crawl.pdf_extract import extract_text_from_pdf

logger = logging.getLogger(__name__)

# OpenStax math textbooks with their known PDF download URLs.
# These are the official free PDFs provided by OpenStax under CC BY license.
# PDF URLs are fetched via the OpenStax CMS API.
OPENSTAX_BOOKS = [
    {
        "slug": "algebra-and-trigonometry-2e",
        "title": "Algebra and Trigonometry 2e",
        "subject": "mathematics",
    },
    {
        "slug": "calculus-volume-1",
        "title": "Calculus Volume 1",
        "subject": "mathematics",
    },
    {
        "slug": "calculus-volume-2",
        "title": "Calculus Volume 2",
        "subject": "mathematics",
    },
    {
        "slug": "introductory-statistics-2e",
        "title": "Introductory Statistics 2e",
        "subject": "mathematics",
    },
    {
        "slug": "prealgebra-2e",
        "title": "Prealgebra 2e",
        "subject": "mathematics",
    },
]

OPENSTAX_BASE = "https://openstax.org"


def crawl_openstax(
    output_dir: str | Path,
    max_books: int = 5,
    delay_seconds: float = 2.0,
) -> list[dict]:
    """Download math textbooks from OpenStax by fetching their free PDFs.

    Args:
        output_dir: Directory to save cleaned text files.
        max_books: Maximum number of books to download.
        delay_seconds: Delay between requests.

    Returns:
        List of metadata dicts for each downloaded book.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    books = OPENSTAX_BOOKS[:max_books]

    for book_info in books:
        try:
            meta = _download_openstax_book(book_info, output_dir, delay_seconds)
            if meta:
                results.append(meta)
                logger.info(f"Downloaded: {meta['title']} ({meta['token_count']:,} chars)")
        except Exception as e:
            logger.error(f"Failed to download {book_info['title']}: {e}")

        time.sleep(delay_seconds)

    logger.info(f"Downloaded {len(results)}/{len(books)} books from OpenStax")
    return results


def _download_openstax_book(
    book_info: dict, output_dir: Path, delay: float
) -> dict | None:
    """Download a single OpenStax book PDF and extract text."""
    slug = book_info["slug"]
    title = book_info["title"]

    logger.info(f"Fetching PDF URL for {title}...")

    # Step 1: Get the PDF download URL from the OpenStax CMS API
    pdf_url = _get_pdf_url(slug, delay)

    if not pdf_url:
        logger.warning(f"Could not find PDF URL for {slug}")
        return None

    logger.info(f"Downloading PDF for {title} from {pdf_url}...")

    # Step 2: Download the PDF
    try:
        resp = requests.get(
            pdf_url,
            timeout=120,  # PDFs can be large
            allow_redirects=True,
            headers={"User-Agent": "BookGPT/1.0 (educational research project)"},
        )
        if resp.status_code != 200:
            logger.warning(f"PDF download failed for {slug}: HTTP {resp.status_code}")
            return None

        if len(resp.content) < 10000:
            logger.warning(f"PDF too small for {slug} ({len(resp.content)} bytes)")
            return None

    except requests.RequestException as e:
        logger.warning(f"PDF download error for {slug}: {e}")
        return None

    # Step 3: Save PDF temporarily and extract text
    pdf_mb = len(resp.content) / 1024 / 1024
    logger.info(f"Extracting text from {title} PDF ({pdf_mb:.1f} MB)...")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not text or len(text) < 1000:
        logger.warning(f"Text extraction yielded too little for {slug}")
        return None

    # Step 4: Clean the text
    text = _clean_openstax_text(text)

    # Step 5: Save
    book_id = f"openstax_{slug.replace('-', '_')}"
    file_path = output_dir / f"{book_id}.txt"
    file_path.write_text(text, encoding="utf-8")

    return {
        "book_id": book_id,
        "title": title,
        "source": "openstax",
        "source_url": f"{OPENSTAX_BASE}/details/books/{slug}",
        "subject": book_info["subject"],
        "token_count": len(text),
        "file_path": str(file_path),
    }


def _get_pdf_url(slug: str, delay: float) -> str | None:
    """Get the PDF download URL for an OpenStax book using their CMS API."""

    # Approach 1: Use the OpenStax CMS API to find the book and its PDF link
    api_url = (
        f"{OPENSTAX_BASE}/apps/cms/api/v2/pages/"
        f"?type=books.Book&fields=*&slug={slug}"
    )

    try:
        resp = requests.get(api_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            if items:
                book_data = items[0]

                # Check for direct PDF URL fields
                for field in [
                    "high_resolution_pdf_url",
                    "low_resolution_pdf_url",
                ]:
                    url = book_data.get(field)
                    if url:
                        logger.debug(f"  Found PDF via API field '{field}': {url}")
                        return url

                # If not in the listing, fetch the full page detail
                page_id = book_data.get("id")
                if page_id:
                    time.sleep(delay)
                    detail_resp = requests.get(
                        f"{OPENSTAX_BASE}/apps/cms/api/v2/pages/{page_id}/",
                        timeout=15,
                    )
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()

                        for field in [
                            "high_resolution_pdf_url",
                            "low_resolution_pdf_url",
                        ]:
                            url = detail.get(field)
                            if url:
                                logger.debug(f"  Found PDF via detail field '{field}': {url}")
                                return url

                        # Check nested resource lists
                        for res_field in ["book_faculty_resources", "book_student_resources"]:
                            resources = detail.get(res_field, [])
                            if isinstance(resources, list):
                                for res in resources:
                                    link = (
                                        res.get("link_document_url")
                                        or res.get("link_external")
                                        or ""
                                    )
                                    if ".pdf" in link.lower():
                                        logger.debug(f"  Found PDF in resources: {link}")
                                        return link

    except Exception as e:
        logger.debug(f"API query failed: {e}")

    # Approach 2: Try common direct URL patterns
    time.sleep(delay)
    direct_patterns = [
        f"https://assets.openstax.org/oscms-prodcms/media/documents/{slug}.pdf",
        f"https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/{slug}.pdf",
    ]

    for url in direct_patterns:
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                logger.debug(f"  Found PDF via direct URL: {url}")
                return url
        except requests.RequestException:
            pass

    return None


def _clean_openstax_text(text: str) -> str:
    """Clean OpenStax PDF-extracted text."""
    # Remove common PDF artifacts
    text = re.sub(r"Access for free at.*?openstax\.org[^\n]*", "", text)
    text = re.sub(r"OpenStax.*?Creative Commons.*?\n", "", text)

    # Remove page headers/footers that repeat
    text = re.sub(r"(?m)^\d+\s*\|\s*Chapter\s+\d+.*$", "", text)
    text = re.sub(r"(?m)^\d+\s*\|\s*Index.*$", "", text)

    # Remove standalone page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()
