"""Orchestrator: routes queries, runs inference on selected models, merges answers."""

import json
import logging
from pathlib import Path

import torch

from bookgpt.model.gpt2 import GPT2
from bookgpt.model.generate import generate_answer, generate_text
from bookgpt.router.router import BookRouter
from bookgpt.tokenizer.train_bpe import load_tokenizer
from bookgpt.utils.device import get_device, mps_empty_cache

logger = logging.getLogger(__name__)


class Orchestrator:
    """Multi-model orchestrator that routes queries and merges answers."""

    def __init__(
        self,
        manifest_path: str | Path,
        router_dir: str | Path,
        models_dir: str | Path,
        tokenizer_dir: str | Path,
        device: torch.device | None = None,
        merge_strategy: str = "confidence",
        use_finetuned: bool = True,
    ):
        """
        Args:
            manifest_path: Path to books manifest.json.
            router_dir: Path to saved router artifacts.
            models_dir: Base path containing pretrained/ or finetuned/ subdirs.
            tokenizer_dir: Path to shared tokenizer directory.
            device: Device to run inference on.
            merge_strategy: How to combine answers ("confidence", "voting", "concat").
            use_finetuned: Whether to use fine-tuned models (True) or pretrained (False).
        """
        self.device = device or get_device()
        self.merge_strategy = merge_strategy
        self.manifest_path = Path(manifest_path)
        self.models_dir = Path(models_dir)
        self.use_finetuned = use_finetuned

        # Load router
        logger.info("Loading router...")
        self.router = BookRouter.load(router_dir)

        # Load shared tokenizer (same for all models)
        logger.info("Loading shared tokenizer...")
        self.tokenizer = load_tokenizer(tokenizer_dir)

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = {entry["book_id"]: entry for entry in json.load(f)}

        # Cache for loaded models (lazy loading)
        self._model_cache: dict[str, GPT2] = {}

    def _load_model(self, book_id: str) -> GPT2:
        """Load a model for a book (with caching). Tokenizer is shared."""
        if book_id in self._model_cache:
            return self._model_cache[book_id]

        # Determine model directory
        if self.use_finetuned:
            model_dir = self.models_dir / "finetuned" / book_id / "best"
            if not model_dir.exists():
                model_dir = self.models_dir / "finetuned" / book_id / "final"
            if not model_dir.exists():
                # Fallback to pretrained
                model_dir = self.models_dir / "pretrained" / book_id / "best"
        else:
            model_dir = self.models_dir / "pretrained" / book_id / "best"

        if not model_dir.exists():
            model_dir = self.models_dir / "pretrained" / book_id / "final"

        if not model_dir.exists():
            raise FileNotFoundError(f"No model found for book {book_id}")

        # Load
        model = GPT2.from_pretrained(model_dir, device=self.device)
        model.eval()

        self._model_cache[book_id] = model
        logger.info(f"Loaded model for {book_id} from {model_dir}")

        return model

    def _unload_model(self, book_id: str):
        """Unload a model from cache to free memory."""
        if book_id in self._model_cache:
            del self._model_cache[book_id]
            mps_empty_cache()

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_answer_tokens: int = 256,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        """Process a user query through the full pipeline.

        Args:
            question: The user's question.
            top_k: Number of book models to consult.
            max_answer_tokens: Maximum tokens per answer.
            temperature: Sampling temperature.
            verbose: Whether to print intermediate results.

        Returns:
            Dict with 'answer', 'sources', 'all_answers', and routing info.
        """
        # Step 1: Route
        routes = self.router.route(question, top_k=top_k)

        if verbose:
            print(f"\n[Router] Selected {len(routes)} book(s):")
            for r in routes:
                print(f"  - {r['book_id']} (score: {r['score']:.3f})")

        # Step 2: Generate answers from each selected model
        answers = []
        for route_info in routes:
            book_id = route_info["book_id"]
            try:
                model = self._load_model(book_id)

                if self.use_finetuned:
                    # Use Q&A format
                    answer_text, log_prob = generate_answer(
                        model=model,
                        tokenizer=self.tokenizer,
                        context="",  # No specific context; the model has book knowledge
                        question=question,
                        max_new_tokens=max_answer_tokens,
                        temperature=temperature,
                        device=self.device,
                    )
                else:
                    # Use prompt completion
                    answer_text = generate_text(
                        model=model,
                        tokenizer=self.tokenizer,
                        prompt=question,
                        max_new_tokens=max_answer_tokens,
                        temperature=temperature,
                        device=self.device,
                    )
                    log_prob = 0.0  # no confidence for raw generation

                answer_entry = {
                    "book_id": book_id,
                    "title": self.manifest.get(book_id, {}).get("title", book_id),
                    "answer": answer_text,
                    "confidence": log_prob,
                    "route_score": route_info["score"],
                }
                answers.append(answer_entry)

                if verbose:
                    print(f"\n[{book_id}] {answer_text[:200]}...")

            except Exception as e:
                logger.error(f"Error generating from {book_id}: {e}")
                continue

        if not answers:
            return {
                "answer": "I could not generate an answer from any of the available models.",
                "sources": [],
                "all_answers": [],
            }

        # Step 3: Merge answers
        final_answer = self._merge_answers(answers)

        if verbose:
            print(f"\n[Final] {final_answer}")

        return {
            "answer": final_answer,
            "sources": [{"book_id": a["book_id"], "title": a["title"]} for a in answers],
            "all_answers": answers,
        }

    def _merge_answers(self, answers: list[dict]) -> str:
        """Merge answers from multiple models into a single response."""
        if len(answers) == 1:
            return answers[0]["answer"]

        if self.merge_strategy == "confidence":
            return self._merge_by_confidence(answers)
        elif self.merge_strategy == "voting":
            return self._merge_by_voting(answers)
        elif self.merge_strategy == "concat":
            return self._merge_by_concat(answers)
        else:
            return self._merge_by_confidence(answers)

    def _merge_by_confidence(self, answers: list[dict]) -> str:
        """Select the answer with highest confidence (log probability)."""
        # Combine route score and generation confidence
        scored = []
        for a in answers:
            combined_score = a["route_score"] * 0.5 + (a["confidence"] + 10) / 10 * 0.5
            scored.append((combined_score, a))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return best["answer"]

    def _merge_by_voting(self, answers: list[dict]) -> str:
        """Simple voting: pick the answer that shares the most key terms with others."""
        if not answers:
            return ""

        # Extract key terms from each answer
        answer_terms = []
        for a in answers:
            terms = set(
                w.lower()
                for w in a["answer"].split()
                if len(w) > 3 and w.isalpha()
            )
            answer_terms.append(terms)

        # Score each answer by overlap with others
        best_idx = 0
        best_overlap = 0
        for i, terms_i in enumerate(answer_terms):
            overlap = sum(
                len(terms_i & terms_j)
                for j, terms_j in enumerate(answer_terms)
                if j != i
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        return answers[best_idx]["answer"]

    def _merge_by_concat(self, answers: list[dict]) -> str:
        """Concatenate all answers with source attribution."""
        parts = []
        for a in answers:
            parts.append(f"[{a['title']}]: {a['answer']}")
        return "\n\n".join(parts)
