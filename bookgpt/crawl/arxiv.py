"""Crawler for arXiv open-access math papers.

Uses the arXiv API (Atom feed) to search and download papers.
Downloads LaTeX source when available, falls back to PDF extraction.
"""

import logging
import re
import tarfile
import tempfile
import time
from pathlib import Path
from io import BytesIO

import requests
from bs4 import BeautifulSoup

from bookgpt.crawl.pdf_extract import extract_text_from_pdf

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_BASE = "https://arxiv.org"

# arXiv math subject categories
MATH_CATEGORIES = [
    "math.AG",  # Algebraic Geometry
    "math.AP",  # Analysis of PDEs
    "math.CA",  # Classical Analysis
    "math.CO",  # Combinatorics
    "math.FA",  # Functional Analysis
    "math.GN",  # General Topology
    "math.GT",  # Geometric Topology
    "math.HO",  # History and Overview
    "math.LO",  # Logic
    "math.NA",  # Numerical Analysis
    "math.NT",  # Number Theory
    "math.PR",  # Probability
    "math.RA",  # Rings and Algebras
    "math.ST",  # Statistics Theory
]


def crawl_arxiv(
    output_dir: str | Path,
    max_papers: int = 20,
    delay_seconds: float = 3.0,
    categories: list[str] | None = None,
    search_query: str | None = None,
) -> list[dict]:
    """Download math papers from arXiv.

    Args:
        output_dir: Directory to save cleaned text files.
        max_papers: Maximum number of papers to download.
        delay_seconds: Delay between requests (arXiv requires 3s minimum).
        categories: arXiv categories to search. Defaults to MATH_CATEGORIES subset.
        search_query: Custom search query. If None, uses broad math search.

    Returns:
        List of metadata dicts for each downloaded paper.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build search query
    if search_query is None:
        # Search across a few key math categories for substantial papers
        cats = categories or ["math.CA", "math.CO", "math.NT", "math.PR", "math.NA", "math.HO"]
        cat_query = " OR ".join(f"cat:{c}" for c in cats)
        search_query = cat_query

    # Fetch paper listings from arXiv API
    papers = _search_arxiv(search_query, max_results=max_papers * 2, delay=delay_seconds)
    logger.info(f"Found {len(papers)} papers from arXiv API")

    results = []
    for paper in papers:
        if len(results) >= max_papers:
            break

        try:
            meta = _download_paper(paper, output_dir, delay_seconds)
            if meta:
                results.append(meta)
                logger.info(
                    f"Downloaded: {meta['title'][:60]}... "
                    f"({meta['token_count']:,} chars) [{meta.get('format', '?')}]"
                )
        except Exception as e:
            logger.error(f"Failed to download {paper.get('id', '?')}: {e}")

        time.sleep(delay_seconds)

    logger.info(f"Downloaded {len(results)} papers from arXiv")
    return results


def _search_arxiv(query: str, max_results: int = 50, delay: float = 3.0) -> list[dict]:
    """Search arXiv API and return paper metadata.

    Args:
        query: arXiv API query string.
        max_results: Maximum results to fetch.
        delay: Delay between paginated requests.

    Returns:
        List of paper metadata dicts with keys: id, title, authors, summary, pdf_url, etc.
    """
    papers = []
    batch_size = 25  # arXiv recommends max 25 per request
    start = 0

    while start < max_results:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(batch_size, max_results - start),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"arXiv API returned {resp.status_code}")
                break

            batch = _parse_atom_feed(resp.text)
            if not batch:
                break

            papers.extend(batch)
            start += len(batch)

            if len(batch) < batch_size:
                break  # no more results

        except requests.RequestException as e:
            logger.error(f"arXiv API request failed: {e}")
            break

        time.sleep(delay)

    return papers


def _parse_atom_feed(xml_text: str) -> list[dict]:
    """Parse arXiv Atom feed XML into paper metadata dicts."""
    soup = BeautifulSoup(xml_text, "html.parser")
    papers = []

    for entry in soup.find_all("entry"):
        paper_id_tag = entry.find("id")
        if not paper_id_tag:
            continue

        paper_url = paper_id_tag.get_text(strip=True)
        # Extract arXiv ID from URL: http://arxiv.org/abs/XXXX.XXXXX
        arxiv_id = paper_url.split("/abs/")[-1] if "/abs/" in paper_url else paper_url

        title = entry.find("title")
        title_text = title.get_text(strip=True) if title else "Untitled"
        # Clean up title (remove newlines)
        title_text = re.sub(r"\s+", " ", title_text)

        summary = entry.find("summary")
        summary_text = summary.get_text(strip=True) if summary else ""

        authors = []
        for author_tag in entry.find_all("author"):
            name = author_tag.find("name")
            if name:
                authors.append(name.get_text(strip=True))

        # Find PDF link
        pdf_url = None
        for link in entry.find_all("link"):
            if link.get("title") == "pdf" or (link.get("href", "").endswith("/pdf")):
                pdf_url = link.get("href")
                break
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        if not pdf_url:
            # Construct PDF URL from ID
            pdf_url = f"{ARXIV_BASE}/pdf/{arxiv_id}"

        # Categories
        categories = []
        for cat in entry.find_all("category"):
            term = cat.get("term")
            if term:
                categories.append(term)

        papers.append({
            "id": arxiv_id,
            "title": title_text,
            "authors": authors,
            "summary": summary_text,
            "pdf_url": pdf_url,
            "categories": categories,
            "url": paper_url,
        })

    return papers


def _download_paper(paper: dict, output_dir: Path, delay: float) -> dict | None:
    """Download a single paper. Try LaTeX source first, then PDF."""
    arxiv_id = paper["id"]
    clean_id = arxiv_id.replace("/", "_").replace(".", "_")

    # Strategy 1: Try to get LaTeX source
    text, fmt = _try_latex_source(arxiv_id, delay)

    # Strategy 2: Fall back to PDF
    if not text:
        text, fmt = _try_pdf(paper["pdf_url"], delay)

    if not text or len(text.strip()) < 2000:
        logger.debug(f"Paper {arxiv_id} yielded insufficient text")
        return None

    # Clean
    text = _clean_paper_text(text)

    if len(text) < 1000:
        return None

    # Save
    slug = f"arxiv_{clean_id}"
    file_path = output_dir / f"{slug}.txt"
    file_path.write_text(text, encoding="utf-8")

    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += " et al."

    return {
        "book_id": slug,
        "title": f"{paper['title']} ({authors_str})",
        "source": "arxiv",
        "source_url": paper["url"],
        "subject": "mathematics",
        "token_count": len(text),
        "file_path": str(file_path),
        "format": fmt,
    }


def _try_latex_source(arxiv_id: str, delay: float) -> tuple[str | None, str]:
    """Try to download and extract LaTeX source from arXiv e-print."""
    source_url = f"{ARXIV_BASE}/e-print/{arxiv_id}"

    try:
        resp = requests.get(source_url, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            return None, ""

        content = resp.content

        # arXiv source is usually a gzipped tar file
        if content[:2] == b"\x1f\x8b" or _is_tar(content):
            return _extract_tex_from_tar(content), "tex"

        # Sometimes it's just a single .tex file (gzipped)
        try:
            import gzip
            decompressed = gzip.decompress(content)
            text = decompressed.decode("utf-8", errors="ignore")
            if "\\begin{document}" in text or "\\section" in text:
                return _clean_tex_source(text), "tex"
        except Exception:
            pass

        # Maybe it's raw TeX
        try:
            text = content.decode("utf-8", errors="ignore")
            if "\\begin{document}" in text or "\\section" in text:
                return _clean_tex_source(text), "tex"
        except Exception:
            pass

    except requests.RequestException as e:
        logger.debug(f"Source download failed for {arxiv_id}: {e}")

    return None, ""


def _is_tar(content: bytes) -> bool:
    """Check if content looks like a tar file."""
    try:
        import gzip
        decompressed = gzip.decompress(content)
        return decompressed[:5] in (b"ustar", b"\x00\x00\x00\x00\x00") or len(decompressed) > 1000
    except Exception:
        return False


def _extract_tex_from_tar(content: bytes) -> str | None:
    """Extract .tex files from a gzipped tar archive."""
    import gzip

    try:
        decompressed = gzip.decompress(content)
    except Exception:
        decompressed = content

    try:
        with tarfile.open(fileobj=BytesIO(decompressed), mode="r:*") as tar:
            tex_contents = []
            for member in tar.getmembers():
                if member.name.endswith(".tex") and member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        tex_text = f.read().decode("utf-8", errors="ignore")
                        tex_contents.append(tex_text)

            if tex_contents:
                # Find the main .tex file (usually the longest one with \begin{document})
                main_tex = None
                for tex in tex_contents:
                    if "\\begin{document}" in tex:
                        if main_tex is None or len(tex) > len(main_tex):
                            main_tex = tex

                if main_tex is None:
                    main_tex = max(tex_contents, key=len)

                return _clean_tex_source(main_tex)

    except (tarfile.TarError, Exception) as e:
        logger.debug(f"Tar extraction failed: {e}")

    return None


def _try_pdf(pdf_url: str, delay: float) -> tuple[str | None, str]:
    """Download PDF and extract text."""
    try:
        resp = requests.get(pdf_url, timeout=60, allow_redirects=True)
        if resp.status_code != 200 or len(resp.content) < 5000:
            return None, ""

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            text = extract_text_from_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if text and len(text) > 1000:
            return text, "pdf"

    except Exception as e:
        logger.debug(f"PDF download failed ({pdf_url}): {e}")

    return None, ""


def _clean_tex_source(text: str) -> str:
    """Clean LaTeX source into readable text preserving math content."""
    # Remove comments
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # Extract body only
    begin_doc = text.find("\\begin{document}")
    if begin_doc != -1:
        text = text[begin_doc + len("\\begin{document}"):]
    end_doc = text.find("\\end{document}")
    if end_doc != -1:
        text = text[:end_doc]

    # Convert structural commands
    replacements = [
        (r"\\textbf\{([^}]*)\}", r"\1"),
        (r"\\textit\{([^}]*)\}", r"\1"),
        (r"\\emph\{([^}]*)\}", r"\1"),
        (r"\\section\*?\{([^}]*)\}", r"\n\n## \1\n"),
        (r"\\subsection\*?\{([^}]*)\}", r"\n### \1\n"),
        (r"\\subsubsection\*?\{([^}]*)\}", r"\n#### \1\n"),
        (r"\\title\{([^}]*)\}", r"\n# \1\n"),
        (r"\\item\s*", "- "),
        (r"\\label\{[^}]*\}", ""),
        (r"\\ref\{[^}]*\}", ""),
        (r"\\cite\{[^}]*\}", ""),
        (r"\\bibliography\{[^}]*\}", ""),
        (r"\\bibliographystyle\{[^}]*\}", ""),
        (r"\\index\{[^}]*\}", ""),
        (r"\\footnote\{([^}]*)\}", r" (\1)"),
        (r"\\author\{([^}]*)\}", r"\1"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # Remove environments but keep content
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)

    # Keep math: $...$ and $$...$$ preserved
    # Remove remaining commands but keep arguments
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Clean braces
    text = text.replace("{", "").replace("}", "")

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def _clean_paper_text(text: str) -> str:
    """Final cleaning pass for paper text."""
    # Remove references section if present
    ref_markers = ["## References", "## Bibliography", "# References"]
    for marker in ref_markers:
        idx = text.rfind(marker)
        if idx != -1 and idx > len(text) * 0.6:  # only if in the latter part
            text = text[:idx]

    # Remove very short lines that are artifacts
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if len(line.strip()) < 3 and line.strip() not in ("", "-"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    # Normalize
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
