import logging
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict

class HybridAnalyzer:
    def __init__(self, llm, embedder):
        self.logger = logging.getLogger('PDFProcessor.HybridAnalyzer')
        self.llm = llm
        self.embedder = embedder
    
    def analyze_structure(self, chunks: Dict, chunk_embeddings: Dict) -> pd.DataFrame:
        """임베딩을 기반으로 금융 상품 구조 분석"""
        try:
            # 1. 벡터 간 유사도 행렬 계산
            vectors = np.array(list(chunk_embeddings.values()))
            n_samples = len(vectors)
            similarity_matrix = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                for j in range(n_samples):
                    similarity_matrix[i][j] = 1 - cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            
            # 2. 계층적 클러스터링
            max_depth = 5
            hierarchical_clusters = {}
            
            for depth in range(2, max_depth + 1):
                clustering = AgglomerativeClustering(
                    n_clusters=depth,
                    affinity='precomputed',
                    linkage='complete'
                )
                hierarchical_clusters[depth] = clustering.fit_predict(1 - similarity_matrix)
            
            # 3. 구조화된 데이터 생성
            structured_data = self._create_structured_data(
                chunks, hierarchical_clusters, max_depth
            )
            
            return pd.DataFrame(structured_data)
            
        except Exception as e:
            self.logger.error(f"구조 분석 중 오류 발생: {str(e)}")
            raise 