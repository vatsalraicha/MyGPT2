"""Crawler for Project Gutenberg math textbooks."""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Well-known public domain math books on Project Gutenberg
# These are verified IDs for math-related texts
MATH_BOOK_IDS = [
    33283,  # "Calculus Made Easy" by Silvanus P. Thompson
    13309,  # "A First Book in Algebra" by Wallace C. Boyden
    33063,  # "Plane Geometry" by George A. Wentworth
    26373,  # "The Elements of Non-Euclidean Geometry" by Julian Lowell Coolidge
    38769,  # "A Course of Pure Mathematics" by G. H. Hardy
    8746,   # "History of Modern Mathematics" by David Eugene Smith
    16713,  # "Amusements in Mathematics" by Henry Ernest Dudeney
    21076,  # "The First Six Books of the Elements of Euclid" by Euclid & John Casey
    41568,  # "An Introduction to Mathematics" by Alfred North Whitehead
    36670,  # "The First Steps in Algebra" by G. A. Wentworth
    17384,  # "The Foundations of Geometry" by David Hilbert
    39088,  # "On the Study and Difficulties of Mathematics" by Augustus De Morgan
    36640,  # "Lectures on Elementary Mathematics" by J. L. Lagrange
    29785,  # "First Course in the Theory of Equations" by Leonard E. Dickson
    26839,  # "Mathematical Recreations and Essays" by W. W. Rouse Ball
]

BASE_URL = "https://www.gutenberg.org"


def crawl_gutenberg(
    output_dir: str | Path,
    max_books: int = 10,
    delay_seconds: float = 2.0,
    book_ids: list[int] | None = None,
) -> list[dict]:
    """Download math books from Project Gutenberg.

    Args:
        output_dir: Directory to save cleaned text files.
        max_books: Maximum number of books to download.
        delay_seconds: Delay between requests to be polite.
        book_ids: Specific Gutenberg IDs to download. Defaults to MATH_BOOK_IDS.

    Returns:
        List of metadata dicts for each downloaded book.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = (book_ids or MATH_BOOK_IDS)[:max_books]
    results = []

    for book_id in ids:
        try:
            meta = _download_book(book_id, output_dir, delay_seconds)
            if meta:
                results.append(meta)
                logger.info(f"Downloaded: {meta['title']} ({meta['token_count']} chars)")
        except Exception as e:
            logger.error(f"Failed to download book {book_id}: {e}")

        time.sleep(delay_seconds)

    logger.info(f"Downloaded {len(results)}/{len(ids)} books from Project Gutenberg")
    return results


def _download_book(book_id: int, output_dir: Path, delay: float) -> dict | None:
    """Download a single book by its Gutenberg ID."""
    # Try to get the plain text version
    text_url = f"{BASE_URL}/cache/epub/{book_id}/pg{book_id}.txt"
    utf8_url = f"{BASE_URL}/files/{book_id}/{book_id}-0.txt"

    text = None
    for url in [text_url, utf8_url]:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text = resp.text
                break
        except requests.RequestException:
            continue
        time.sleep(delay)

    if not text:
        logger.warning(f"Could not download text for book {book_id}")
        return None

    # Get metadata from the book page
    title = _extract_title(book_id, delay)

    # Clean the text
    text = _clean_gutenberg_text(text)

    if len(text) < 1000:
        logger.warning(f"Book {book_id} too short after cleaning ({len(text)} chars), skipping")
        return None

    # Save
    book_slug = _slugify(title or f"gutenberg_{book_id}")
    file_path = output_dir / f"{book_slug}.txt"
    file_path.write_text(text, encoding="utf-8")

    return {
        "book_id": book_slug,
        "title": title or f"Gutenberg Book {book_id}",
        "source": "gutenberg",
        "source_url": f"{BASE_URL}/ebooks/{book_id}",
        "subject": "mathematics",
        "token_count": len(text),
        "file_path": str(file_path),
    }


def _extract_title(book_id: int, delay: float) -> str | None:
    """Extract the book title from its Gutenberg page."""
    try:
        time.sleep(delay)
        resp = requests.get(f"{BASE_URL}/ebooks/{book_id}", timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            title_tag = soup.find("h1", itemprop="name")
            if title_tag:
                return title_tag.get_text(strip=True)
            # Fallback: look in meta
            meta_title = soup.find("meta", property="og:title")
            if meta_title:
                return meta_title.get("content", "").strip()
    except Exception as e:
        logger.debug(f"Could not fetch title for book {book_id}: {e}")
    return None


def _clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg headers, footers, and license text."""
    # Find start of actual content (after the header)
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "E-text prepared by",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip past this line
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1 :]
            break

    # Find end of actual content (before the footer)
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
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
    return text[:80]  # limit length
