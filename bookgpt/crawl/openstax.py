"""Crawler for OpenStax free textbooks."""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from bookgpt.crawl.pdf_extract import extract_text_from_pdf

logger = logging.getLogger(__name__)

# Known OpenStax math textbook URLs (freely available with CC license)
# These are the web-view URLs; we'll scrape the HTML content directly
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

OPENSTAX_API = "https://openstax.org/apps/archive"
OPENSTAX_BASE = "https://openstax.org"


def crawl_openstax(
    output_dir: str | Path,
    max_books: int = 5,
    delay_seconds: float = 2.0,
) -> list[dict]:
    """Download math textbooks from OpenStax.

    Scrapes the HTML content directly from the OpenStax web reader,
    which provides well-structured content.

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
                logger.info(f"Downloaded: {meta['title']} ({meta['token_count']} chars)")
        except Exception as e:
            logger.error(f"Failed to download {book_info['title']}: {e}")

        time.sleep(delay_seconds)

    logger.info(f"Downloaded {len(results)}/{len(books)} books from OpenStax")
    return results


def _download_openstax_book(
    book_info: dict, output_dir: Path, delay: float
) -> dict | None:
    """Download a single OpenStax book by scraping its table of contents and chapters."""
    slug = book_info["slug"]
    toc_url = f"{OPENSTAX_BASE}/details/books/{slug}"

    try:
        resp = requests.get(toc_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Could not fetch TOC for {slug}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to find chapter links in the book details page
    chapter_links = _extract_chapter_links(soup, slug)

    if not chapter_links:
        logger.warning(f"No chapter links found for {slug}, trying direct content scrape")
        # Fallback: try the book's content page directly
        chapter_links = [f"/books/{slug}"]

    # Scrape each chapter
    all_text = []
    for link in chapter_links:
        time.sleep(delay)
        chapter_text = _scrape_chapter(link)
        if chapter_text:
            all_text.append(chapter_text)

    if not all_text:
        logger.warning(f"No content extracted for {slug}")
        return None

    full_text = "\n\n".join(all_text)
    full_text = _clean_openstax_text(full_text)

    if len(full_text) < 1000:
        logger.warning(f"Book {slug} too short ({len(full_text)} chars), skipping")
        return None

    file_path = output_dir / f"openstax_{slug.replace('-', '_')}.txt"
    file_path.write_text(full_text, encoding="utf-8")

    return {
        "book_id": f"openstax_{slug.replace('-', '_')}",
        "title": book_info["title"],
        "source": "openstax",
        "source_url": f"{OPENSTAX_BASE}/details/books/{slug}",
        "subject": book_info["subject"],
        "token_count": len(full_text),
        "file_path": str(file_path),
    }


def _extract_chapter_links(soup: BeautifulSoup, slug: str) -> list[str]:
    """Extract chapter/section links from the book details page."""
    links = []
    # Look for links containing the book slug that point to chapter content
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if f"/books/{slug}/" in href and href not in links:
            links.append(href)

    # Deduplicate and limit
    seen = set()
    unique = []
    for link in links:
        # Normalize the link
        if link.startswith("/"):
            link = f"{OPENSTAX_BASE}{link}"
        if link not in seen:
            seen.add(link)
            unique.append(link)

    return unique[:50]  # cap at 50 chapters


def _scrape_chapter(url: str) -> str | None:
    """Scrape text content from a single chapter URL."""
    if url.startswith("/"):
        url = f"{OPENSTAX_BASE}{url}"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Look for main content area
    content = soup.find("main") or soup.find("div", class_=re.compile(r"content|chapter|book"))
    if content is None:
        content = soup.body

    if content is None:
        return None

    return content.get_text(separator="\n", strip=True)


def _clean_openstax_text(text: str) -> str:
    """Clean OpenStax scraped text."""
    # Remove navigation artifacts
    text = re.sub(r"Previous\s+Next", "", text)
    text = re.sub(r"Skip to Content", "", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()
