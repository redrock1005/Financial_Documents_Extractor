from typing import List, Union, Dict
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from ..interfaces.embedder import BaseEmbedder

class KBAlbertEmbedder(BaseEmbedder):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.max_length = 512

    def embed(self, text: str) -> Union[List[float], None]:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            return embeddings.tolist()
        
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """여러 텍스트를 배치로 임베딩"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
