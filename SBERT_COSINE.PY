from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize FastAPI app
app = FastAPI()

# Load SBERT model (optimized for similarity tasks) and enable quantization for speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)  # Smaller, faster SBERT (384 dimensions)

@app.post("/sbert-similarity")
async def sbert_similarity(
    texts: list[str],          # List of sentences to compare
    query: str,                # Query sentence
    max_length: int = 128      # Max sequence length (optimized for short queries)
):
    """
    Compute cosine similarity between query and a batch of sentences using SBERT.
    Args:
        texts: List of sentences for comparison.
        query: Single query sentence.
        max_length: Maximum token length for truncation.
    Returns:
        Sorted similarities for each input text.
    """
    # Encode query and text embeddings with truncation and dynamic padding
    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device, truncation=True, max_length=max_length
    )
    text_embeddings = model.encode(
        texts, convert_to_tensor=True, device=device, truncation=True, max_length=max_length
    )

    # Compute cosine similarity between query and each text
    similarities = util.pytorch_cos_sim(query_embedding, text_embeddings)[0]

    # Sort results by similarity score in descending order
    results = [
        {"text": texts[i], "similarity": similarities[i].item()}
        for i in range(len(texts))
    ]
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    return {"results": results}
