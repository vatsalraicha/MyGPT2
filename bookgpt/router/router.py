"""Router model: selects which book-model(s) are relevant for a query.

Implements TF-IDF embedding similarity (approach 5.2a from the spec).
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BookRouter:
    """Routes queries to the most relevant book models using TF-IDF similarity."""

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.book_embeddings: np.ndarray | None = None
        self.book_ids: list[str] = []
        self.book_metadata: list[dict] = []

    def build_index(self, manifest_path: str | Path):
        """Build the routing index from the book manifest.

        For each book, computes a TF-IDF embedding from its full text.

        Args:
            manifest_path: Path to the manifest.json file.
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if not manifest:
            raise ValueError("Empty manifest — no books to index")

        texts = []
        self.book_ids = []
        self.book_metadata = []

        for entry in manifest:
            book_path = Path(entry["file_path"])
            if not book_path.exists():
                logger.warning(f"Book file not found: {book_path}, skipping")
                continue

            text = book_path.read_text(encoding="utf-8")
            texts.append(text)
            self.book_ids.append(entry["book_id"])
            self.book_metadata.append(entry)

        if not texts:
            raise ValueError("No book texts could be loaded")

        # Fit TF-IDF vectorizer on all books
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.book_embeddings = self.vectorizer.fit_transform(texts)

        logger.info(
            f"Router index built: {len(self.book_ids)} books, "
            f"vocab={len(self.vectorizer.vocabulary_)} features"
        )

    def route(self, query: str, top_k: int = 3) -> list[dict]:
        """Route a query to the most relevant book models.

        Args:
            query: User query text.
            top_k: Number of top books to return.

        Returns:
            List of dicts with 'book_id', 'score', and metadata for each selected book.
        """
        if self.vectorizer is None or self.book_embeddings is None:
            raise RuntimeError("Router index not built. Call build_index() first.")

        # Embed the query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.book_embeddings).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # only include if there's some relevance
                results.append(
                    {
                        "book_id": self.book_ids[idx],
                        "score": float(similarities[idx]),
                        **self.book_metadata[idx],
                    }
                )

        if not results:
            # Fallback: return all books with uniform score
            logger.warning("No relevant books found, returning all with uniform scores")
            for i, book_id in enumerate(self.book_ids):
                results.append(
                    {
                        "book_id": book_id,
                        "score": 1.0 / len(self.book_ids),
                        **self.book_metadata[i],
                    }
                )

        return results

    def save(self, save_dir: str | Path):
        """Save router artifacts."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "router.pkl", "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "book_embeddings": self.book_embeddings,
                    "book_ids": self.book_ids,
                    "book_metadata": self.book_metadata,
                },
                f,
            )
        logger.info(f"Router saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str | Path) -> "BookRouter":
        """Load router from saved artifacts."""
        load_dir = Path(load_dir)

        with open(load_dir / "router.pkl", "rb") as f:
            data = pickle.load(f)

        router = cls()
        router.vectorizer = data["vectorizer"]
        router.book_embeddings = data["book_embeddings"]
        router.book_ids = data["book_ids"]
        router.book_metadata = data["book_metadata"]

        logger.info(f"Router loaded from {load_dir} ({len(router.book_ids)} books)")
        return router
