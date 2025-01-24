from typing import List, Union
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from interfaces.embedder import BaseEmbedder
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KBAlbertEmbedder(BaseEmbedder):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.max_length = 512
        self.model.eval()

    def embed(self, text: str) -> Union[List[float], None]:
        """단일 텍스트 임베딩"""
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
                # [CLS] 토큰의 임베딩 사용
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            return embeddings.tolist()
        except Exception as e:
            print(f"임베딩 오류: {e}")
            return None

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """배치 임베딩"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
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
                    
            except Exception as e:
                print(f"배치 임베딩 오류: {e}")
                continue
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
