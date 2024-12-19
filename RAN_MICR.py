# embedding_service.py
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

# Load pre-trained BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

@app.post("/embedding")
async def generate_embedding(text: str):
    """
    Generate embeddings for input text using BERT.
    """
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    embedding = torch.mean(output.last_hidden_state, dim=1).tolist()
    return {"embedding": embedding}

# retrieval_service.py
from fastapi import FastAPI
import pinecone

app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("candidate-profiles")

@app.post("/retrieve")
async def retrieve_candidates(query_embedding: list, top_k: int = 10):
    """
    Retrieve top-k candidates from Pinecone based on query embedding.
    """
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return {"candidates": response['matches']}

# ranking_service.py
from fastapi import FastAPI

app = FastAPI()

@app.post("/rank")
async def rank_candidates(candidates: list, query_embedding: list):
    """
    Rank candidates based on scores.
    """
    ranked_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    for i, candidate in enumerate(ranked_candidates):
        candidate["rank"] = i + 1
    return {"ranked_candidates": ranked_candidates}

# explainability_service.py
from fastapi import FastAPI
import shap
import numpy as np

app = FastAPI()

# Placeholder model for SHAP (use actual model in production)
class DummyModel:
    def predict(self, X):
        return np.sum(X, axis=1)

model = DummyModel()
explainer = shap.Explainer(model.predict)

@app.post("/explain")
async def explain_ranking(query_embedding: list, ranked_candidates: list):
    """
    Explain ranking decisions using SHAP.
    """
    shap_values = explainer(np.array([query_embedding]))
    explanations = {candidate["id"]: f"SHAP Value: {shap_values.values[0][i]}" for i, candidate in enumerate(ranked_candidates)}
    return {"explanations": explanations}

# rag_service.py
from fastapi import FastAPI
from llama_index import SimpleKeywordTableIndex, Document

app = FastAPI()

# Sample documents for RAG
documents = [
    "Machine Learning Engineer job requires experience in Python, PyTorch, and BERT.",
    "Data Scientist job involves expertise in NLP and cloud computing."
]
index = SimpleKeywordTableIndex([Document(text=doc) for doc in documents])

@app.post("/rag")
async def retrieve_context(query: str):
    """
    Retrieve relevant context for the query using LlamaIndex.
    """
    context = index.query(query)
    return {"context": context}

# orchestrator_service.py
from fastapi import FastAPI
import requests

app = FastAPI()

EMBEDDING_SERVICE_URL = "http://embedding-service:8000/embedding"
RETRIEVAL_SERVICE_URL = "http://retrieval-service:8000/retrieve"
RANKING_SERVICE_URL = "http://ranking-service:8000/rank"
EXPLAINABILITY_SERVICE_URL = "http://explainability-service:8000/explain"
RAG_SERVICE_URL = "http://rag-service:8000/rag"

@app.post("/pipeline")
async def run_pipeline(query: str):
    """
    Run the full pipeline for ranking candidates.
    """
    # Step 1: Get query embedding
    embedding_response = requests.post(EMBEDDING_SERVICE_URL, json={"text": query})
    query_embedding = embedding_response.json()["embedding"]

    # Step 2: Retrieve relevant context (RAG)
    rag_response = requests.post(RAG_SERVICE_URL, json={"query": query})
    context = rag_response.json()["context"]

    # Step 3: Retrieve candidates
    retrieval_response = requests.post(RETRIEVAL_SERVICE_URL, json={"query_embedding": query_embedding, "top_k": 10})
    candidates = retrieval_response.json()["candidates"]

    # Step 4: Rank candidates
    ranking_response = requests.post(RANKING_SERVICE_URL, json={"candidates": candidates, "query_embedding": query_embedding})
    ranked_candidates = ranking_response.json()["ranked_candidates"]

    # Step 5: Explain ranking
    explainability_response = requests.post(EXPLAINABILITY_SERVICE_URL, json={"query_embedding": query_embedding, "ranked_candidates": ranked_candidates})
    explanations = explainability_response.json()["explanations"]

    return {
        "ranked_candidates": ranked_candidates,
        "context": context,
        "explanations": explanations
    }

"""
How to Run These Services

1]
Save each service in a separate file (e.g., embedding_service.py, retrieval_service.py, etc.).

2]
Install dependencies:
pip install fastapi uvicorn transformers torch pinecone-client llama-index shap


3]
Run each service using Uvicorn:

uvicorn embedding_service:app --host 0.0.0.0 --port 8000
uvicorn retrieval_service:app --host 0.0.0.0 --port 8001
uvicorn ranking_service:app --host 0.0.0.0 --port 8002
uvicorn explainability_service:app --host 0.0.0.0 --port 8003
uvicorn rag_service:app --host 0.0.0.0 --port 8004
uvicorn orchestrator_service:app --host 0.0.0.0 --port 8005
