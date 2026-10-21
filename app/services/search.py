
from app.data.devices import DEVICE_DOCUMENTS
from app.utils.bm25_index import bm25_rank
from app.utils.embeddings import dense_rank


# RRF Search
def hybrid_search(query: str, k: int = 3) -> list[tuple[str, float]]:
    
    # document indices ranked by each search method
    bm25_indices = bm25_rank(query)
    dense_indices = dense_rank(query)

    # Start every document with an RRF score of zero
    rrf_scores = {
        index: 0.0 for index in range(len(DEVICE_DOCUMENTS))
    }

    # each document's BM25 rank contribution
    for rank, index in enumerate(bm25_indices, start=1):
        rrf_scores[index] += 1 / (60 + rank)



    # each document's dense rank contribution
    for rank, index in enumerate(dense_indices, start=1):
        rrf_scores[index] += 1 / (60 + rank)

    # Keep the indices with the highest combined scores.
    ranked_indices = sorted(
        rrf_scores,
        key=lambda index: rrf_scores[index],
        reverse=True,
    )[:k]

    # Return each document's text and fused score.
    return [
        (DEVICE_DOCUMENTS[index], rrf_scores[index])
        for index in ranked_indices
    ]
