"""Semantic similarity scorer for benchmark responses."""

import re
import sys
from typing import Dict, List, Optional

from .constants import DEFAULT_SEMANTIC_MODEL
from .keyword_scorer import is_censored_response

try:
    from sentence_transformers import SentenceTransformer, util

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


def parse_reference_answers(filepath: str = "answers_all.txt") -> Dict[int, str]:
    """
    Parse reference answers from answers_all.txt.

    Returns:
        Dict mapping question_id -> reference answer text
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        print("   Make sure you're running the script from the project root directory")
        sys.exit(1)

    answers = {}
    pattern = r"=== Q(\d+):.*?===\s+(.*?)(?=\n=== Q\d+:|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    for q_id, answer in matches:
        answers[int(q_id)] = answer.strip()

    return answers


class SemanticScorer:
    """Semantic similarity scorer with embedding cache."""

    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL):
        """Initialize semantic scorer with specified model."""
        if not SEMANTIC_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: uv sync --extra semantic"
            )

        print(f"📦 Loading semantic model: {model_name}...")
        if "gte" in model_name.lower() or "Alibaba" in model_name:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            self.model = SentenceTransformer(model_name)
        self.reference_embeddings = {}
        print("   ✓ Model loaded")

    @staticmethod
    def _similarity_to_score(similarity: float) -> int:
        """Map cosine similarity to benchmark score buckets."""
        if similarity >= 0.75:
            return 100
        elif similarity >= 0.65:
            return 75
        elif similarity >= 0.45:
            return 50
        else:
            return 0

    def load_reference_answers(self, filepath: str = "answers_all.txt") -> None:
        """Load and embed reference answers once."""
        answers = parse_reference_answers(filepath)

        print("📦 Encoding reference answers...")
        q_ids = list(answers.keys())
        answer_texts = [answers[q_id] for q_id in q_ids]
        embeddings = self.model.encode(
            answer_texts, convert_to_tensor=True, show_progress_bar=False
        )

        for index, q_id in enumerate(q_ids):
            self.reference_embeddings[q_id] = embeddings[index]

        print(f"   ✓ Encoded {len(answers)} reference answers")

    def score_response(self, q_id: int, response: str) -> int:
        """Score response using semantic similarity."""
        if is_censored_response(response):
            return 0

        if q_id not in self.reference_embeddings:
            print(f"   ⚠️  Warning: No reference answer for Q{q_id}")
            return 50

        response_embedding = self.model.encode(
            [response], convert_to_tensor=True, show_progress_bar=False
        )[0]
        similarity = util.cos_sim(
            response_embedding, self.reference_embeddings[q_id]
        ).item()
        return self._similarity_to_score(similarity)

    def score_responses(self, items: List[Dict]) -> List[int]:
        """
        Batch-score responses.

        Each item must contain "id" and "response". Returns scores in the
        same order as input items.
        """
        scores: List[Optional[int]] = [None] * len(items)
        encode_items = []

        for index, item in enumerate(items):
            q_id = item["id"]
            response = item["response"]

            if is_censored_response(response):
                scores[index] = 0
            elif q_id not in self.reference_embeddings:
                print(f"   ⚠️  Warning: No reference answer for Q{q_id}")
                scores[index] = 50
            else:
                encode_items.append((index, q_id, response))

        if encode_items:
            response_embeddings = self.model.encode(
                [item[2] for item in encode_items],
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            for embedding_index, (item_index, q_id, _) in enumerate(encode_items):
                similarity = util.cos_sim(
                    response_embeddings[embedding_index],
                    self.reference_embeddings[q_id],
                ).item()
                scores[item_index] = self._similarity_to_score(similarity)

        return [score if score is not None else 0 for score in scores]
