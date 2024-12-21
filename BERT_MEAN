from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize FastAPI
app = FastAPI()

# Load pre-trained BERT tokenizer and model (optimized for production)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@app.post("/embedding")
async def generate_embeddings(texts: list[str], max_length: int = 128):
    """
    Generate embeddings for a batch of text inputs.
    Args:
        texts: List of input sentences.
        max_length: Maximum length for tokenization.
    Returns:
        List of embeddings (one per input).
    """
    # Tokenize input texts with dynamic padding, truncation, and batching
    tokens = tokenizer(
        texts,  # Supports a batch of text
        return_tensors='pt',
        padding=True,  # Pads dynamically to the longest sentence in the batch
        truncation=True,  # Truncates inputs longer than max_length
        max_length=max_length  # Controls maximum token length
    )

    # Move tokens to GPU if available
    tokens = {key: val.to(device) for key, val in tokens.items()}

    # Generate embeddings
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(**tokens)

    # Apply mean pooling to get sentence-level embeddings
    embeddings = torch.mean(output.last_hidden_state, dim=1)

    # Move embeddings back to CPU and convert to list for output
    return {"embeddings": embeddings.cpu().tolist()}
