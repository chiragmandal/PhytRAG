#!/usr/bin/env python3
"""
Retrieval quality evaluation tracked in MLflow.

Metrics computed:
  hit_at_5   - fraction of questions where at least one retrieved chunk
               contains at least one expected keyword (Hit@5)
  mrr_at_5   - Mean Reciprocal Rank at 5
  avg_score  - average cosine similarity score across all retrieved chunks

Run after ingestion:
    python -m eval.retrieval_eval

Results are logged to MLflow at http://localhost:5000 under the
experiment 'phytrag-retrieval-eval'.

Why track this in MLflow?
  When the embedding model or chunk size changes, you run this eval,
  compare runs in the MLflow UI, and make data-backed decisions.
  This is the same pattern used for tracking LLM fine-tuning experiments.
"""
import logging
import sys
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.yml"


def run_eval() -> dict[str, float]:
    settings = get_settings()

    # Load eval questions
    with EVAL_QUESTIONS_PATH.open() as f:
        eval_data = yaml.safe_load(f)
    questions = eval_data["questions"]
    logger.info("Loaded %d eval questions", len(questions))

    # Init retrieval components
    embedder = Embedder(model_name=settings.embed_model)
    vector_store = VectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        vector_size=settings.vector_size,
    )

    # Check the collection is populated
    info = vector_store.collection_info()
    if info["vectors_count"] == 0:
        logger.error("Collection is empty. Run 'make ingest' first.")
        sys.exit(1)

    # Run retrieval for each question
    hits = 0
    reciprocal_ranks = []
    all_scores = []

    for item in questions:
        question = item["question"]
        keywords = [kw.lower() for kw in item["keywords"]]

        query_vec = embedder.embed(question)
        results = vector_store.search(
            query_vector=query_vec,
            top_k=5,
            score_threshold=0.0,  # include all results for eval
        )

        all_scores.extend(r["score"] for r in results)

        # Hit@5: at least one result contains at least one keyword
        hit = False
        rr = 0.0
        for rank, r in enumerate(results, start=1):
            text_lower = r["text"].lower()
            if any(kw in text_lower for kw in keywords):
                hit = True
                if rr == 0.0:
                    rr = 1.0 / rank
        if hit:
            hits += 1
        reciprocal_ranks.append(rr)

        logger.info(
            "Q: '%s...' | hit=%s | best_score=%.3f",
            question[:50],
            hit,
            results[0]["score"] if results else 0,
        )

    n = len(questions)
    metrics = {
        "hit_at_5": hits / n,
        "mrr_at_5": sum(reciprocal_ranks) / n,
        "avg_retrieval_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "questions_evaluated": float(n),
        "collection_vectors": float(info["vectors_count"]),
    }

    return metrics


def main() -> None:
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    params = {
        "embed_model": settings.embed_model,
        "retrieval_top_k": settings.retrieval_top_k,
        "collection_name": settings.collection_name,
    }

    with mlflow.start_run(run_name=f"eval-{settings.embed_model}"):
        mlflow.log_params(params)

        logger.info("Running retrieval evaluation...")
        metrics = run_eval()

        mlflow.log_metrics(metrics)

        logger.info("Results:")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

        print(f"\nHit@5:  {metrics['hit_at_5']:.2%}")
        print(f"MRR@5:  {metrics['mrr_at_5']:.4f}")
        print(f"Avg cosine score: {metrics['avg_retrieval_score']:.4f}")
        print(f"\nRun logged to MLflow: {settings.mlflow_tracking_uri}")
        print("Open http://localhost:5000 to compare runs.")


if __name__ == "__main__":
    main()
