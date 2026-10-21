from rank_bm25 import BM25Okapi

from app.data.devices import DEVICE_DOCUMENTS

# Sparse Embedding

# tokenize func to prepare the Doc and the query in the same way
def tokenize(text: str) -> list[str]:
    """Lowercase text and split it on whitespace."""
    return text.lower().split()

# A) the Document
# Build the document index once then reuse it for every query
TOKENIZED_DOCUMENTS = [tokenize(doc) for doc in DEVICE_DOCUMENTS]  # ==> comprehensive list and the result [" "," ", " "]

BM25_INDEX = BM25Okapi(TOKENIZED_DOCUMENTS)        # ==> the model will take this [" "," ", " "] and index it 


# B) the Query
def bm25_rank(query: str) -> list[int]:
    """Return document indices from highest to lowest BM25 score."""
    
    # 1) tokenize the Query ( divide ) to words 
    query_tokens = tokenize(query)
    
    # 2) get_scores ==> returns the index of arranged tokens 
    scores = BM25_INDEX.get_scores(query_tokens)

    # 3) arrange the scores 
    return sorted(
        range(len(DEVICE_DOCUMENTS)),
        key=lambda index: scores[index],
        reverse=True, # sorted is reversed so it is Desc
    )