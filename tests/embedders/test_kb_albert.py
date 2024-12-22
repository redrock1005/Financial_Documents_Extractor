import pytest
import numpy as np
from src.embedders.kb_albert import KBAlbertEmbedder

@pytest.fixture
def embedder():
    return KBAlbertEmbedder()

def test_embedder_initialization(embedder):
    assert embedder is not None
    assert embedder.model is not None
    assert embedder.tokenizer is not None

def test_single_embedding(embedder):
    text = "우대금리 조건"
    embedding = embedder.embed(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)
    assert not np.isnan(embedding).any()

def test_batch_embedding(embedder):
    texts = ["우대금리 조건", "금리인하 요건", "대출한도 증액"]
    embeddings = embedder.batch_embed(texts, batch_size=2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)
    assert not np.isnan(embeddings).any()
