"""Crawler for Project Gutenberg math textbooks.

Most math textbooks on Gutenberg are available as TeX (LaTeX) and/or PDF,
not plain text. This crawler tries multiple formats in priority order:
1. Plain text (.txt) — best quality, but rare for math books
2. TeX/LaTeX (.tex) — preserves math notation, needs light cleaning
3. HTML (.htm/.html) — good fallback, needs HTML stripping
4. PDF — last resort, uses pdf_extract module
"""

import logging
import re
import tempfile
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from bookgpt.crawl.pdf_extract import extract_text_from_pdf

logger = logging.getLogger(__name__)

# Verified Project Gutenberg IDs for English math textbooks.
# Each entry: (id, title, author) — all verified to exist and be in English.
MATH_BOOKS = [
    # --- Original 15 books ---
    (33283, "Calculus Made Easy", "Silvanus P. Thompson"),
    (13309, "A First Book in Algebra", "Wallace C. Boyden"),
    (33063, "Plane Geometry", "George A. Wentworth"),
    (26373, "The Elements of Non-Euclidean Geometry", "Julian Lowell Coolidge"),
    (38769, "A Course of Pure Mathematics", "G. H. Hardy"),
    (16713, "Amusements in Mathematics", "Henry Ernest Dudeney"),
    (21076, "The First Six Books of the Elements of Euclid", "John Casey"),
    (41568, "An Introduction to Mathematics", "Alfred North Whitehead"),
    (36670, "The First Steps in Algebra", "G. A. Wentworth"),
    (17384, "The Foundations of Geometry", "David Hilbert"),
    (39041, "Elementary Illustrations of the Differential and Integral Calculus", "Augustus De Morgan"),
    (36640, "Lectures on Elementary Mathematics", "J. L. Lagrange"),
    (29785, "First Course in the Theory of Equations", "Leonard E. Dickson"),
    (26839, "Mathematical Recreations and Essays", "W. W. Rouse Ball"),
    (13693, "The Theory of Numbers", "R. D. Carmichael"),
    # --- Analysis & Calculus ---
    (18741, "Introduction to Infinitesimal Analysis", "Oswald Veblen and N. J. Lennes"),
    (38993, "The Integration of Functions of a Single Variable", "G. H. Hardy"),
    # --- Logic & Foundations ---
    (36884, "The Mathematical Analysis of Logic", "George Boole"),
    (15114, "An Investigation of the Laws of Thought", "George Boole"),
    (10836, "The Algebra of Logic", "Louis Couturat"),
    (28696, "Symbolic Logic", "Lewis Carroll"),
    (4763, "The Game of Logic", "Lewis Carroll"),
    (52091, "An Essay on the Foundations of Geometry", "Bertrand Russell"),
    # --- Number Theory & Algebra ---
    (21016, "Essays on the Theory of Numbers", "Richard Dedekind"),
    (37030, "Some Famous Problems of the Theory of Numbers", "G. H. Hardy"),
    (25156, "An Introduction to Nonassociative Algebras", "Richard D. Schafer"),
    (17920, "The Number-System of Algebra", "Henry B. Fine"),
    # --- Geometry & Trigonometry ---
    (19770, "Spherical Trigonometry", "I. Todhunter"),
    (32973, "Elements of Plane Trigonometry", "Hugh Blackburn"),
    (29807, "Solid Geometry with Problems and Applications", "H. E. Slaught and N. J. Lennes"),
    (17001, "An Elementary Course in Synthetic Projective Geometry", "Derrick Norman Lehmer"),
    # --- Mathematical Physics ---
    (50992, "Elementary Principles of Statistical Mechanics", "J. Willard Gibbs"),
    (37157, "Science and Hypothesis", "Henri Poincare"),
    (13609, "Vector Analysis and Quaternions", "Alexander Macfarlane"),
    (26262, "Utility of Quaternions in Physics", "Alexander McAulay"),
    (7825, "Geometrical Solutions Derived from Mechanics", "Archimedes"),
    # --- Probability & Statistics ---
    (57359, "The Logic of Chance", "John Venn"),
    # --- General Mathematics ---
    (39088, "On the Study and Difficulties of Mathematics", "Augustus De Morgan"),
    (36154, "The Evanston Colloquium Lectures on Mathematics", "Felix Klein"),
    (29788, "Four Lectures on Mathematics", "Jacques Hadamard"),
    (50977, "On Multiple Algebra", "J. Willard Gibbs"),
]

BASE_URL = "https://www.gutenberg.org"


def crawl_gutenberg(
    output_dir: str | Path,
    max_books: int = 10,
    delay_seconds: float = 2.0,
    book_ids: list[int] | None = None,
) -> list[dict]:
    """Download math books from Project Gutenberg.

    Tries multiple formats: plain text, TeX, HTML, PDF.

    Args:
        output_dir: Directory to save cleaned text files.
        max_books: Maximum number of books to download.
        delay_seconds: Delay between requests to be polite.
        book_ids: Specific Gutenberg IDs to download. If None, uses MATH_BOOKS.

    Returns:
        List of metadata dicts for each downloaded book.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if book_ids:
        books = [(bid, f"Gutenberg Book {bid}", "Unknown") for bid in book_ids[:max_books]]
    else:
        books = MATH_BOOKS[:max_books]

    results = []

    for book_id, title, author in books:
        try:
            meta = _download_book(book_id, title, author, output_dir, delay_seconds)
            if meta:
                results.append(meta)
                logger.info(
                    f"Downloaded: {meta['title']} ({meta['token_count']:,} chars) "
                    f"[format: {meta.get('format', '?')}]"
                )
            else:
                logger.warning(f"Failed to download: {title} (ID {book_id})")
        except Exception as e:
            logger.error(f"Error downloading book {book_id} ({title}): {e}")

        time.sleep(delay_seconds)

    logger.info(f"Downloaded {len(results)}/{len(books)} books from Project Gutenberg")
    return results


def _download_book(
    book_id: int, title: str, author: str, output_dir: Path, delay: float
) -> dict | None:
    """Download a single book, trying multiple formats."""

    # Strategy 1: Plain text
    text, fmt = _try_plain_text(book_id, delay)

    # Strategy 2: TeX/LaTeX
    if not text:
        text, fmt = _try_tex(book_id, delay)

    # Strategy 3: HTML
    if not text:
        text, fmt = _try_html(book_id, delay)

    # Strategy 4: PDF
    if not text:
        text, fmt = _try_pdf(book_id, delay)

    if not text or len(text.strip()) < 1000:
        return None

    # Clean the text
    text = _clean_gutenberg_text(text)

    if len(text) < 500:
        logger.warning(f"Book {book_id} too short after cleaning ({len(text)} chars)")
        return None

    # Save
    book_slug = _slugify(title)
    file_path = output_dir / f"{book_slug}.txt"
    file_path.write_text(text, encoding="utf-8")

    return {
        "book_id": book_slug,
        "title": f"{title} by {author}",
        "source": "gutenberg",
        "source_url": f"{BASE_URL}/ebooks/{book_id}",
        "subject": "mathematics",
        "token_count": len(text),
        "file_path": str(file_path),
        "format": fmt,
    }


def _try_plain_text(book_id: int, delay: float) -> tuple[str | None, str]:
    """Try to download plain text version."""
    urls = [
        f"{BASE_URL}/ebooks/{book_id}.txt.utf-8",
        f"{BASE_URL}/cache/epub/{book_id}/pg{book_id}.txt",
        f"{BASE_URL}/files/{book_id}/{book_id}-0.txt",
        f"{BASE_URL}/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        text = _fetch_text(url)
        if text and len(text) > 1000:
            logger.debug(f"  Plain text found: {url}")
            return text, "txt"
        time.sleep(delay)
    return None, ""


def _try_tex(book_id: int, delay: float) -> tuple[str | None, str]:
    """Try to download TeX/LaTeX version."""
    urls = [
        f"{BASE_URL}/files/{book_id}/{book_id}-t/{book_id}-t.tex",
        f"{BASE_URL}/files/{book_id}/{book_id}-t.tex",
    ]
    for url in urls:
        text = _fetch_text(url)
        if text and len(text) > 500:
            logger.debug(f"  TeX found: {url}")
            cleaned = _clean_tex(text)
            return cleaned, "tex"
        time.sleep(delay)
    return None, ""


def _try_html(book_id: int, delay: float) -> tuple[str | None, str]:
    """Try to download HTML version and extract text."""
    urls = [
        f"{BASE_URL}/files/{book_id}/{book_id}-h/{book_id}-h.htm",
        f"{BASE_URL}/files/{book_id}/{book_id}-h.htm",
        f"{BASE_URL}/ebooks/{book_id}.html.images",
    ]
    for url in urls:
        html = _fetch_text(url)
        if html and len(html) > 1000:
            logger.debug(f"  HTML found: {url}")
            text = _html_to_text(html)
            if text and len(text) > 500:
                return text, "html"
        time.sleep(delay)
    return None, ""


def _try_pdf(book_id: int, delay: float) -> tuple[str | None, str]:
    """Try to download PDF and extract text."""
    urls = [
        f"{BASE_URL}/files/{book_id}/{book_id}-pdf.pdf",
        f"{BASE_URL}/cache/epub/{book_id}/pg{book_id}-images.pdf",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 1000:
                logger.debug(f"  PDF found: {url}")
                # Save to temp file and extract
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                text = extract_text_from_pdf(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)
                if text and len(text) > 500:
                    return text, "pdf"
        except Exception as e:
            logger.debug(f"  PDF download failed ({url}): {e}")
        time.sleep(delay)
    return None, ""


def _fetch_text(url: str) -> str | None:
    """Fetch a URL and return text content, following redirects."""
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True)
        if resp.status_code == 200:
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
    except requests.RequestException as e:
        logger.debug(f"  Fetch failed ({url}): {e}")
    return None


def _clean_tex(text: str) -> str:
    """Clean TeX/LaTeX source into readable text while preserving math."""
    # Remove TeX comments
    text = re.sub(r"(?m)^%.*$", "", text)

    # Remove document class / preamble
    begin_doc = text.find("\\begin{document}")
    if begin_doc != -1:
        text = text[begin_doc + len("\\begin{document}"):]
    end_doc = text.find("\\end{document}")
    if end_doc != -1:
        text = text[:end_doc]

    # Convert common TeX commands to readable text
    replacements = [
        (r"\\textbf\{([^}]*)\}", r"\1"),
        (r"\\textit\{([^}]*)\}", r"\1"),
        (r"\\emph\{([^}]*)\}", r"\1"),
        (r"\\underline\{([^}]*)\}", r"\1"),
        (r"\\section\*?\{([^}]*)\}", r"\n\n\1\n"),
        (r"\\subsection\*?\{([^}]*)\}", r"\n\1\n"),
        (r"\\subsubsection\*?\{([^}]*)\}", r"\n\1\n"),
        (r"\\chapter\*?\{([^}]*)\}", r"\n\n\1\n\n"),
        (r"\\title\{([^}]*)\}", r"\n\1\n"),
        (r"\\item\s*", "- "),
        (r"\\label\{[^}]*\}", ""),
        (r"\\ref\{[^}]*\}", ""),
        (r"\\cite\{[^}]*\}", ""),
        (r"\\index\{[^}]*\}", ""),
        (r"\\footnote\{([^}]*)\}", r" (\1)"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # Remove \begin{...} and \end{...} environment markers (but keep content)
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)

    # Keep math notation: $...$ and $$...$$ are preserved
    # Remove remaining unknown TeX commands but keep their argument
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # Remove standalone TeX commands
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Clean up braces
    text = text.replace("{", "").replace("}", "")

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def _html_to_text(html: str) -> str:
    """Extract text from HTML content."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text


def _clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg headers, footers, and license text."""
    # Find start of actual content (after the header)
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "E-text prepared by",
        "Produced by",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break

    # Find end of actual content (before the footer)
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Normalize line endings and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    text = re.sub(r"-+", "-", text)
    return text[:80]
