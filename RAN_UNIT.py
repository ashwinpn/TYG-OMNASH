"""
Install Pytest:

pip install pytest

Directory Structure:

Copy code
.
├── services/
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── ranking_service.py
│   ├── explainability_service.py
│   ├── rag_service.py
│   └── orchestrator_service.py
├── tests/
│   ├── test_embedding_service.py
│   ├── test_retrieval_service.py
│   ├── test_ranking_service.py
│   ├── test_explainability_service.py
│   ├── test_rag_service.py
│   └── test_orchestrator_service.py
└── requirements.txt
Unit Tests

"""




# 1. Embedding Service

# tests/test_embedding_service.py
import pytest
from embedding_service import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_generate_embedding():
    response = client.post("/embedding", json={"text": "NLP engineer with BERT experience"})
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0


# 2. Retrieval Service

# tests/test_retrieval_service.py
import pytest
from retrieval_service import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_retrieve_candidates():
    query_embedding = [0.1] * 768  # Example embedding vector
    response = client.post("/retrieve", json={"query_embedding": query_embedding, "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert isinstance(data["candidates"], list)



# Ranking Service


# tests/test_ranking_service.py
import pytest
from ranking_service import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_rank_candidates():
    candidates = [
        {"id": 1, "name": "John", "score": 0.85},
        {"id": 2, "name": "Jane", "score": 0.90}
    ]
    query_embedding = [0.1] * 768  # Example embedding vector
    response = client.post("/rank", json={"candidates": candidates, "query_embedding": query_embedding})
    assert response.status_code == 200
    data = response.json()
    assert "ranked_candidates" in data
    ranked_candidates = data["ranked_candidates"]
    assert ranked_candidates[0]["id"] == 2  # Higher score should come first


# 4. Explainability Service

# tests/test_explainability_service.py
import pytest
from explainability_service import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_explain_ranking():
    query_embedding = [0.1] * 768  # Example embedding vector
    ranked_candidates = [
        {"id": 1, "name": "John", "rank": 1},
        {"id": 2, "name": "Jane", "rank": 2}
    ]
    response = client.post("/explain", json={"query_embedding": query_embedding, "ranked_candidates": ranked_candidates})
    assert response.status_code == 200
    data = response.json()
    assert "explanations" in data
    assert isinstance(data["explanations"], dict)
    assert "1" in data["explanations"]  # Check explanation for the first candidate



# 5. RAG Service

# tests/test_rag_service.py
import pytest
from rag_service import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_retrieve_context():
    response = client.post("/rag", json={"query": "Looking for an NLP engineer"})
    assert response.status_code == 200
    data = response.json()
    assert "context" in data
    assert isinstance(data["context"], str)
    assert len(data["context"]) > 0  # Ensure context is not empty



# 6. Orchestrator Service

# tests/test_orchestrator_service.py
import pytest
from orchestrator_service import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_run_pipeline():
    query = "Looking for an NLP engineer with BERT experience"
    response = client.post("/pipeline", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "ranked_candidates" in data
    assert "context" in data
    assert "explanations" in data
    assert isinstance(data["ranked_candidates"], list)
    assert isinstance(data["context"], str)
    assert isinstance(data["explanations"], dict)

"""
Running the Tests
Run all tests:
pytest tests/


Run tests for a specific service:
pytest tests/test_embedding_service.py

"""
