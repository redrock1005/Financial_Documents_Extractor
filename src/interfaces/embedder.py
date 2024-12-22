from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union

class BaseEmbedder(ABC):
    """임베딩 생성을 위한 기본 인터페이스"""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """단일 텍스트에 대한 임베딩 생성

        Args:
            text: 임베딩할 텍스트

        Returns:
            텍스트의 임베딩 벡터
        """
        pass

    @abstractmethod
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """다수 텍스트에 대한 배치 임베딩 생성

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기

        Returns:
            텍스트들의 임베딩 벡터 배열
        """
        pass 