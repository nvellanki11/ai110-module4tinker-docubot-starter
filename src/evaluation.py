"""
A lightweight evaluation harness for DocuBot.

This module helps students compare:
- naive generation over the full docs
- retrieval only answers
- RAG answers (retrieval + Gemini)

The evaluation is intentionally simple: it checks whether DocuBot retrieves
the correct files for each query and reports hit rate, precision@k,
recall@k, and mean reciprocal rank (MRR).
"""

from dataset import SAMPLE_QUERIES


# -----------------------------------------------------------
# Expected document signals for evaluation
# -----------------------------------------------------------
# This dictionary maps a query substring to the filename(s)
# that should be relevant. It does NOT need to be perfect.
# It simply gives students a way to measure improvements.
#
# Example:
#   If a query contains the phrase "auth token",
#   evaluation expects AUTH.md to appear in the retrieval results.
#
EXPECTED_SOURCES = {
    "auth token": ["AUTH.md"],
    "environment variables": ["AUTH.md"],
    "database": ["DATABASE.md"],
    "users": ["API_REFERENCE.md"],
    "projects": ["API_REFERENCE.md"],
    "refresh": ["AUTH.md"],
    "users table": ["DATABASE.md"],
}


def expected_files_for_query(query):
    """
    Returns a list of expected filenames based on simple substring matching.
    """
    query_lower = query.lower()
    matches = []
    for key, files in EXPECTED_SOURCES.items():
        if key in query_lower:
            matches.extend(files)
    return matches


# -----------------------------------------------------------
# Evaluation function
# -----------------------------------------------------------

def _precision_at_k(retrieved_files, expected):
    """
    Fraction of the retrieved list (chunks may repeat a filename) whose
    filename is an expected match. 0 when nothing was retrieved.
    """
    if not retrieved_files:
        return 0.0
    relevant = sum(1 for f in retrieved_files if f in expected)
    return relevant / len(retrieved_files)


def _recall_at_k(retrieved_files, expected):
    """
    Fraction of the distinct expected filenames that appear anywhere in
    the retrieved list. 0 when there are no expected files for the query.
    """
    if not expected:
        return 0.0
    found = {f for f in expected if f in retrieved_files}
    return len(found) / len(set(expected))


def _reciprocal_rank(retrieved_files, expected):
    """
    1 / rank of the first retrieved item whose filename is expected,
    0 if none of the retrieved items match (or there's nothing expected).
    """
    if not expected:
        return 0.0
    for rank, f in enumerate(retrieved_files, start=1):
        if f in expected:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(bot, top_k=3):
    """
    Runs DocuBot's retrieval system against SAMPLE_QUERIES.
    Returns a tuple: (metrics, detailed_results)

    metrics: dict with
      - hit_rate: fraction of queries where at least one retrieved
        snippet's filename matched an expected filename.
      - precision_at_k: average, per query, of the fraction of retrieved
        chunks whose filename was expected.
      - recall_at_k: average, per query, of the fraction of expected
        filenames that were found somewhere in the retrieved chunks.
      - mrr: mean reciprocal rank of the first relevant retrieved chunk.
    Queries with no expected files (not present in EXPECTED_SOURCES)
    contribute 0 to every metric, matching how hit_rate already treats
    them.
    detailed_results: list of dictionaries with structured info.
    """
    results = []
    hits = 0
    precision_sum = 0.0
    recall_sum = 0.0
    rr_sum = 0.0

    for query in SAMPLE_QUERIES:
        expected = expected_files_for_query(query)
        retrieved = bot.retrieve(query, top_k=top_k)

        retrieved_files = [fname for fname, _ in retrieved]

        hit = any(f in retrieved_files for f in expected) if expected else False
        precision = _precision_at_k(retrieved_files, expected)
        recall = _recall_at_k(retrieved_files, expected)
        rr = _reciprocal_rank(retrieved_files, expected)

        if hit:
            hits += 1
        precision_sum += precision
        recall_sum += recall
        rr_sum += rr

        results.append({
            "query": query,
            "expected": expected,
            "retrieved": retrieved_files,
            "hit": hit,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": rr,
        })

    num_queries = len(SAMPLE_QUERIES)
    metrics = {
        "hit_rate": hits / num_queries,
        "precision_at_k": precision_sum / num_queries,
        "recall_at_k": recall_sum / num_queries,
        "mrr": rr_sum / num_queries,
    }
    return metrics, results


# -----------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------

def print_eval_results(metrics, results):
    """
    Nicely formats evaluation results.
    """
    print("\nEvaluation Results")
    print("------------------")
    print(f"Hit rate:        {metrics['hit_rate']:.2f}")
    print(f"Precision@k:     {metrics['precision_at_k']:.2f}")
    print(f"Recall@k:        {metrics['recall_at_k']:.2f}")
    print(f"MRR:             {metrics['mrr']:.2f}\n")

    for item in results:
        print(f"Query: {item['query']}")
        print(f"  Expected:  {item['expected']}")
        print(f"  Retrieved: {item['retrieved']}")
        print(f"  Hit:       {item['hit']}")
        print(f"  Precision@k: {item['precision_at_k']:.2f}")
        print(f"  Recall@k:    {item['recall_at_k']:.2f}")
        print(f"  RR:          {item['reciprocal_rank']:.2f}")
        print()


# -----------------------------------------------------------
# Optional CLI entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from docubot import DocuBot
    from llm_client import GeminiClient

    print("Running retrieval evaluation...\n")

    try:
        llm_client = GeminiClient()
    except RuntimeError as exc:
        print(f"Cannot run evaluation: {exc}")
        print("Retrieval now relies on Gemini embeddings, so GEMINI_API_KEY must be set.")
    else:
        bot = DocuBot(llm_client=llm_client)
        metrics, results = evaluate_retrieval(bot)
        print_eval_results(metrics, results)
