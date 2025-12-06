#!/usr/bin/env python3
"""
Semantic Scoring Analysis Tool

Compares quality of semantic scoring across multiple embedding models.
Analyzes results_*.json files and re-scores responses with different models.

Usage:
    uv run analyze_semantic.py <folder_path> [options]

Example:
    uv run analyze_semantic.py ./results --output analysis.json
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

from scoring.keyword_scorer import is_censored_response

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    from tabulate import tabulate

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEP = str(e)

# Default models to compare
DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "Alibaba-NLP/gte-large-en-v1.5",
]

# Short names for display
MODEL_SHORT_NAMES = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6",
    "sentence-transformers/all-mpnet-base-v2": "MPNet-base",
    "Alibaba-NLP/gte-large-en-v1.5": "GTE-large",
}

# Scoring thresholds (same as run_benchmark.py)
SCORE_THRESHOLDS = {
    "high": 0.85,  # 100%
    "medium": 0.70,  # 75%
    "low": 0.50,  # 50%
}


def get_short_name(model_name: str) -> str:
    """Get short display name for model."""
    return MODEL_SHORT_NAMES.get(model_name, model_name.split("/")[-1])


def parse_reference_answers(filepath: str = "answers_all.txt") -> Dict[int, str]:
    """Parse reference answers from answers_all.txt."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        print("   Make sure you're running from the project root directory")
        sys.exit(1)

    answers = {}
    pattern = r"=== Q(\d+):.*?===\s+(.*?)(?=\n=== Q\d+:|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    for q_id, answer in matches:
        answers[int(q_id)] = answer.strip()

    return answers


def similarity_to_score(similarity: float) -> int:
    """Convert cosine similarity to score (0/50/75/100)."""
    if similarity >= SCORE_THRESHOLDS["high"]:
        return 100
    elif similarity >= SCORE_THRESHOLDS["medium"]:
        return 75
    elif similarity >= SCORE_THRESHOLDS["low"]:
        return 50
    else:
        return 0


class MultiModelScorer:
    """Semantic scorer supporting multiple embedding models."""

    def __init__(self, model_names: List[str]):
        """Initialize with multiple models."""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(
                f"Dependencies not installed: {MISSING_DEP}\n"
                "Install with: uv sync --extra semantic"
            )

        self.models: Dict[str, SentenceTransformer] = {}
        self.reference_embeddings: Dict[str, Dict[int, any]] = {}

        for model_name in model_names:
            short_name = get_short_name(model_name)
            print(f"Loading model: {short_name}...")
            # GTE models require trust_remote_code=True
            if "gte" in model_name.lower() or "Alibaba" in model_name:
                self.models[model_name] = SentenceTransformer(
                    model_name, trust_remote_code=True
                )
            else:
                self.models[model_name] = SentenceTransformer(model_name)
            self.reference_embeddings[model_name] = {}
            print(f"   Model loaded: {short_name}")

    def load_reference_answers(self, filepath: str = "answers_all.txt"):
        """Load and embed reference answers for all models."""
        answers = parse_reference_answers(filepath)

        print(f"\nEncoding {len(answers)} reference answers...")
        for model_name, model in self.models.items():
            short_name = get_short_name(model_name)
            print(f"   Encoding with {short_name}...")
            for q_id, answer_text in answers.items():
                self.reference_embeddings[model_name][q_id] = model.encode(
                    answer_text, convert_to_tensor=True, show_progress_bar=False
                )

        print("   Reference answers encoded\n")

    def score_response(self, q_id: int, response: str) -> Dict[str, Dict[str, float]]:
        """
        Score response with all models.

        Returns:
            {"model_name": {"score": 75, "similarity": 0.78}, ...}
        """
        results = {}

        # Check censorship first (same for all models)
        if is_censored_response(response):
            for model_name in self.models:
                results[model_name] = {"score": 0, "similarity": 0.0, "censored": True}
            return results

        for model_name, model in self.models.items():
            if q_id not in self.reference_embeddings[model_name]:
                results[model_name] = {
                    "score": 50,
                    "similarity": 0.5,
                    "censored": False,
                }
                continue

            response_embedding = model.encode(
                response, convert_to_tensor=True, show_progress_bar=False
            )

            similarity = util.cos_sim(
                response_embedding, self.reference_embeddings[model_name][q_id]
            ).item()

            results[model_name] = {
                "score": similarity_to_score(similarity),
                "similarity": round(similarity, 4),
                "censored": False,
            }

        return results


class ResultsAnalyzer:
    """Analyze benchmark results with multiple semantic models."""

    def __init__(self, scorer: MultiModelScorer):
        self.scorer = scorer
        self.results_files: List[Dict] = []
        self.all_scores: Dict[str, List[Dict]] = {model: [] for model in scorer.models}

    def load_results_folder(self, folder: str) -> int:
        """Load all results_*.json files from folder."""
        pattern = os.path.join(folder, "results_*.json")
        files = glob.glob(pattern)

        if not files:
            print(f"Error: No results_*.json files found in {folder}")
            sys.exit(1)

        for filepath in sorted(files):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["_filepath"] = filepath
                    data["_filename"] = os.path.basename(filepath)
                    self.results_files.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return len(self.results_files)

    def analyze_file(self, results_data: Dict) -> Dict:
        """Analyze single results file with all models."""
        filename = results_data.get("_filename", "unknown")
        model_name = results_data.get("model", "unknown")
        results_list = results_data.get("results", [])

        file_analysis = {
            "filename": filename,
            "llm_model": model_name,
            "responses_count": len(results_list),
            "scores": {},
        }

        # Initialize per-model tracking
        for emb_model in self.scorer.models:
            file_analysis["scores"][emb_model] = {
                "similarities": [],
                "scores": [],
                "per_question": [],
            }

        # Score each response
        for result in results_list:
            q_id = result.get("id")
            response = result.get("full_response", "")

            if not response:
                continue

            model_scores = self.scorer.score_response(q_id, response)

            for emb_model, score_data in model_scores.items():
                file_analysis["scores"][emb_model]["similarities"].append(
                    score_data["similarity"]
                )
                file_analysis["scores"][emb_model]["scores"].append(score_data["score"])
                file_analysis["scores"][emb_model]["per_question"].append(
                    {
                        "q_id": q_id,
                        "category": result.get("category", ""),
                        "similarity": score_data["similarity"],
                        "score": score_data["score"],
                        "censored": score_data.get("censored", False),
                    }
                )

                # Track for aggregated analysis
                self.all_scores[emb_model].append(
                    {
                        "q_id": q_id,
                        "similarity": score_data["similarity"],
                        "score": score_data["score"],
                        "file": filename,
                    }
                )

        # Calculate statistics per model
        for emb_model in self.scorer.models:
            scores_data = file_analysis["scores"][emb_model]
            if scores_data["similarities"]:
                scores_data["avg_similarity"] = round(
                    np.mean(scores_data["similarities"]), 4
                )
                scores_data["avg_score"] = round(np.mean(scores_data["scores"]), 2)
                scores_data["std_dev"] = round(np.std(scores_data["similarities"]), 4)
                scores_data["distribution"] = {
                    "0": scores_data["scores"].count(0),
                    "50": scores_data["scores"].count(50),
                    "75": scores_data["scores"].count(75),
                    "100": scores_data["scores"].count(100),
                }

        return file_analysis

    def analyze_all(self) -> Dict:
        """Analyze all loaded files and compute aggregated statistics."""
        per_file_results = []

        print("Analyzing files...")
        for results_data in self.results_files:
            file_analysis = self.analyze_file(results_data)
            per_file_results.append(file_analysis)
            print(f"   Analyzed: {file_analysis['filename']}")

        # Aggregated statistics
        aggregated = {
            "total_responses": sum(
                len(self.all_scores[m]) for m in list(self.scorer.models.keys())[:1]
            ),
            "scores": {},
        }

        for emb_model in self.scorer.models:
            model_data = self.all_scores[emb_model]
            if model_data:
                sims = [d["similarity"] for d in model_data]
                scores = [d["score"] for d in model_data]
                aggregated["scores"][emb_model] = {
                    "avg_similarity": round(np.mean(sims), 4),
                    "avg_score": round(np.mean(scores), 2),
                    "std_dev": round(np.std(sims), 4),
                    "distribution": {
                        "0": scores.count(0),
                        "50": scores.count(50),
                        "75": scores.count(75),
                        "100": scores.count(100),
                    },
                }

        # Correlation matrix
        correlation_matrix = self._compute_correlation()
        aggregated["correlation_matrix"] = correlation_matrix

        return {
            "per_file": per_file_results,
            "aggregated": aggregated,
        }

    def _compute_correlation(self) -> Dict[str, Dict[str, float]]:
        """Compute correlation matrix between models based on similarity scores."""
        model_names = list(self.scorer.models.keys())
        correlation = {}

        for m1 in model_names:
            correlation[m1] = {}
            sims1 = [d["similarity"] for d in self.all_scores[m1]]

            for m2 in model_names:
                sims2 = [d["similarity"] for d in self.all_scores[m2]]

                if len(sims1) == len(sims2) and len(sims1) > 1:
                    corr = np.corrcoef(sims1, sims2)[0, 1]
                    correlation[m1][m2] = round(corr, 3)
                else:
                    correlation[m1][m2] = 1.0 if m1 == m2 else 0.0

        return correlation


def print_results(analysis: Dict, folder: str):
    """Print formatted analysis results to console."""
    per_file = analysis["per_file"]
    aggregated = analysis["aggregated"]

    print("\n" + "=" * 70)
    print("SEMANTIC SCORING ANALYSIS")
    print("=" * 70)
    print(f"Folder: {folder}")
    print(f"Files analyzed: {len(per_file)}")

    # Per-file results
    print("\nPer-file results:")
    print("-" * 70)

    for file_data in per_file:
        print(f"\n{file_data['filename']} ({file_data['responses_count']} responses)")
        print(f"  LLM: {file_data['llm_model']}")

        table_data = []
        for model_name, scores in file_data["scores"].items():
            if "avg_similarity" in scores:
                table_data.append(
                    [
                        get_short_name(model_name),
                        f"{scores['avg_similarity']:.3f}",
                        f"{scores['avg_score']:.1f}%",
                        f"{scores['std_dev']:.3f}",
                    ]
                )

        if table_data:
            print(
                tabulate(
                    table_data,
                    headers=["Model", "Avg Sim", "Avg Score", "Std Dev"],
                    tablefmt="simple",
                    stralign="left",
                )
            )

    # Aggregated results
    print("\n" + "=" * 70)
    print(f"AGGREGATED RESULTS ({aggregated['total_responses']} total responses)")
    print("=" * 70)

    table_data = []
    for model_name, scores in aggregated["scores"].items():
        dist = scores["distribution"]
        table_data.append(
            [
                get_short_name(model_name),
                f"{scores['avg_similarity']:.3f}",
                f"{scores['avg_score']:.1f}%",
                f"{scores['std_dev']:.3f}",
                dist["0"],
                dist["50"],
                dist["75"],
                dist["100"],
            ]
        )

    print(
        tabulate(
            table_data,
            headers=[
                "Model",
                "Avg Sim",
                "Avg Score",
                "Std Dev",
                "0%",
                "50%",
                "75%",
                "100%",
            ],
            tablefmt="simple",
            stralign="left",
        )
    )

    # Correlation matrix
    print("\nCorrelation Matrix (similarity scores):")
    corr_matrix = aggregated["correlation_matrix"]
    model_names = list(corr_matrix.keys())
    short_names = [get_short_name(m) for m in model_names]

    corr_table = []
    for m1 in model_names:
        row = [get_short_name(m1)]
        for m2 in model_names:
            row.append(f"{corr_matrix[m1][m2]:.2f}")
        corr_table.append(row)

    print(
        tabulate(
            corr_table,
            headers=[""] + short_names,
            tablefmt="simple",
            stralign="left",
        )
    )


def save_json(analysis: Dict, folder: str, output_path: Optional[str] = None):
    """Save analysis results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        filepath = output_path
    else:
        filepath = os.path.join(folder, f"semantic_analysis_{timestamp}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "folder": folder,
        "models": list(analysis["aggregated"]["scores"].keys()),
        "per_file": {},
        "aggregated": analysis["aggregated"],
    }

    # Format per_file data for JSON
    for file_data in analysis["per_file"]:
        filename = file_data["filename"]
        output_data["per_file"][filename] = {
            "llm_model": file_data["llm_model"],
            "responses_count": file_data["responses_count"],
            "scores": {},
        }

        for model_name, scores in file_data["scores"].items():
            # Remove raw lists, keep only statistics
            output_data["per_file"][filename]["scores"][model_name] = {
                "avg_similarity": scores.get("avg_similarity"),
                "avg_score": scores.get("avg_score"),
                "std_dev": scores.get("std_dev"),
                "distribution": scores.get("distribution"),
                "per_question": scores.get("per_question", []),
            }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Analyze semantic scoring quality across embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run analyze_semantic.py ./results
  uv run analyze_semantic.py ./results --output analysis.json
  uv run analyze_semantic.py ./results --models MiniLM-L6,MPNet-base
        """,
    )

    parser.add_argument(
        "folder",
        help="Folder containing results_*.json files",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models (default: all 3)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated in folder)",
    )

    parser.add_argument(
        "--answers",
        type=str,
        default="answers_all.txt",
        help="Path to reference answers file (default: answers_all.txt)",
    )

    args = parser.parse_args()

    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print(f"Error: Dependencies not installed: {MISSING_DEP}")
        print("Install with: uv sync --extra semantic")
        sys.exit(1)

    # Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        sys.exit(1)

    # Parse models
    if args.models:
        model_mapping = {
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2",
            "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
            "gte": "Alibaba-NLP/gte-large-en-v1.5",
            "gte-large": "Alibaba-NLP/gte-large-en-v1.5",
        }

        models = []
        for m in args.models.split(","):
            m_lower = m.strip().lower()
            if m_lower in model_mapping:
                models.append(model_mapping[m_lower])
            elif "/" in m:
                models.append(m)
            else:
                print(f"Warning: Unknown model '{m}', skipping")

        if not models:
            print("Error: No valid models specified")
            sys.exit(1)
    else:
        models = DEFAULT_MODELS

    print(f"Models to compare: {len(models)}")
    for m in models:
        print(f"  - {get_short_name(m)}")
    print()

    # Initialize scorer
    scorer = MultiModelScorer(models)
    scorer.load_reference_answers(args.answers)

    # Load and analyze results
    analyzer = ResultsAnalyzer(scorer)
    num_files = analyzer.load_results_folder(args.folder)
    print(f"\nLoaded {num_files} results file(s)\n")

    # Run analysis
    analysis = analyzer.analyze_all()

    # Output results
    print_results(analysis, args.folder)
    save_json(analysis, args.folder, args.output)


if __name__ == "__main__":
    main()
