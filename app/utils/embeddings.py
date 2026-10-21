import numpy as np
from sentence_transformers import SentenceTransformer

from app.data.devices import DEVICE_DOCUMENTS

# Dense Embedding
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load the model and encode the catalog once
model = SentenceTransformer(MODEL_NAME)

# document_embeddings calculated only once
document_embeddings = model.encode(
    DEVICE_DOCUMENTS,
    normalize_embeddings=True,
)


def dense_rank(query: str) -> list[int]:
    """Return document indices from highest to lowest semantic similarity."""
    
    # Encode and normalize the new query
    # query_embedding calculated every for every new question
    query_embedding = model.encode( query, normalize_embeddings=True)

    # the vectors are normalized because of normalize_embeddings=True  ==> the dot product here = cosine similarity
    scores = np.dot(document_embeddings, query_embedding)

    return sorted(
        range(len(DEVICE_DOCUMENTS)),
        key=lambda index: scores[index],
        reverse=True,
    )