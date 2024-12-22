import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.document_loaders import PyPDFLoader
import logging
from datetime import datetime
import re
import unicodedata
from agentic_chunker import AgenticChunker
from langchain.chains import create_extraction_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, Dict, List
import json
from text_embedder import TextEmbedder
import faiss
import numpy as np
from sheets_manager import GoogleSheetsManager
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from transformers import AlbertModel

# 메인 시작 부분에 로깅 설정 추가
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class HybridAnalyzer:
    # 클래스 레벨에서 PROMPT_TEMPLATE 정의
    PROMPT_TEMPLATE = """당신은 은행의 금융 상품 분석 전문가입니다. 
의미적으로 그룹화된 대출 상품 정보를 분석하여 체계적으로 분류해주세요.

분석할 그룹화된 정보:
{}

다음 규칙을 반드시 준수해주세요:
1. 모든 속성은 명사형으로 작성
2. 최대 5단계의 위계로 구분하여 분류 (구분/대분류/중분류/소분류/상세)
3. 각 단계별 분류는 다음과 같이 구성:
   - 구분: 가장 상위 개념 (예: 대출종류, 금리유형, 상환방식 등)
   - 대분류: 구분의 주요 카테고리
   - 중분류: 대분류의 세부 카테고리
   - 소분류: 구체적인 속성이나 조건
   - 상세: 구체적인 수치나 세부 내용

응답은 반드시 다음 JSON 형식으로 작성해주세요:
{
    "classified": [
        {
            "구분": "string",
            "대분류": "string",
            "중분류": "string",
            "소분류": "string",
            "상세": "string"
        }
    ]
}
"""

    def __init__(self, llm):
        self.logger = logging.getLogger('HybridAnalyzer')
        self.llm = llm
        
        # KB-Albert 초기화
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('KB-Albert/')
            self.kb_model = AlbertModel.from_pretrained('KB-Albert/')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.kb_model.to(self.device)
            
            # FAISS 인덱스 초기화
            self.vector_dimension = self.kb_model.config.hidden_size
            self.index = faiss.IndexFlatL2(self.vector_dimension)
            
        except Exception as e:
            self.logger.error(f"초기화 중 오류 발생: {str(e)}")
            raise

    def embed_chunks(self, chunks: Dict) -> Dict[str, np.ndarray]:
        """청크 텍스트를 KB-Albert로 임베딩하고 FAISS에 저장"""
        embeddings = {}
        vectors_to_add = []  # FAISS에 추가할 벡터들
        
        for chunk_id, chunk in chunks.items():
            try:
                # 토크나이징 & 임베딩 (기존 코드와 동일)
                inputs = self.tokenizer(
                    chunk['text'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.kb_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                
                embeddings[chunk_id] = {
                    'text': chunk['text'],
                    'embedding': embedding
                }
                
                # FAISS용 벡터 추가
                vectors_to_add.append(embedding)
                
            except Exception as e:
                self.logger.error(f"청크 {chunk_id} 임베딩 실패: {str(e)}")
                continue
        
        # FAISS 인덱스에 벡터들 추가
        try:
            if vectors_to_add:
                vectors_array = np.array(vectors_to_add)
                self.index.add(vectors_array)
                
                # 벡터 저장소에 저장
                save_path = os.path.join('vector_store', 'kb_albert_vectors.index')
                os.makedirs('vector_store', exist_ok=True)
                faiss.write_index(self.index, save_path)
                
                self.logger.info(f"벡터 저장소에 {len(vectors_to_add)}개의 벡터 저장 완료")
                
        except Exception as e:
            self.logger.error(f"FAISS 저장 중 오류 발생: {str(e)}")
            # 임베딩은 계속 사용할 수 있도록 에러를 raise하지 않음
                
        return embeddings

    def load_vectors(self):
        """저장된 벡터 로드"""
        try:
            index_path = os.path.join('vector_store', 'kb_albert_vectors.index')
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                self.logger.info("벡터 저장소에서 인덱스 로드 완료")
            else:
                self.logger.warning("저장된 벡터 인덱스가 없습니다.")
                
        except Exception as e:
            self.logger.error(f"벡터 로드 중 오류 발생: {str(e)}")
            raise

    def analyze_structure(self, chunks: Dict, embeddings: Dict) -> pd.DataFrame:
        try:
            # 1. 벡터 간 유사도 행렬 계산 (이전과 동일)
            vectors = np.array([emb['embedding'] for emb in embeddings.values()])
            n_samples = len(vectors)
            similarity_matrix = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                for j in range(n_samples):
                    similarity_matrix[i][j] = 1 - cosine(vectors[i], vectors[j])
            
            # 2. 계층적 클러스터링으로 위계 구조 파악
            max_depth = 5
            hierarchical_clusters = {}
            
            for depth in range(2, max_depth + 1):
                clustering = AgglomerativeClustering(
                    n_clusters=depth,
                    affinity='precomputed',
                    linkage='complete'
                )
                hierarchical_clusters[depth] = clustering.fit_predict(1 - similarity_matrix)
            
            # 3. 클러스터 중심점(centroid)으로 대표 텍스트 선택
            structured_data = []
            chunk_texts = list(chunks.values())
            
            for idx in range(n_samples):
                row = {f'd{depth}': None for depth in range(1, max_depth + 1)}
                
                for depth in range(2, max_depth + 1):
                    cluster_id = hierarchical_clusters[depth][idx]
                    cluster_indices = [i for i in range(n_samples) 
                                     if hierarchical_clusters[depth][i] == cluster_id]
                    
                    # 클러스터 중심에 가장 가까운 텍스트를 대표값으로 선택
                    cluster_vectors = vectors[cluster_indices]
                    centroid = np.mean(cluster_vectors, axis=0)
                    
                    # 중심점과 가장 가까운 텍스트 선택
                    distances = [cosine(centroid, vectors[i]) for i in cluster_indices]
                    representative_idx = cluster_indices[np.argmin(distances)]
                    
                    row[f'd{depth}'] = chunk_texts[representative_idx]['text']
                
                row['original_text'] = chunk_texts[idx]['text']
                structured_data.append(row)
            
            return pd.DataFrame(structured_data)
            
        except Exception as e:
            self.logger.error(f"구조 분석 중 오류 발생: {str(e)}")
            raise

    def _group_similar_chunks(self, chunk_embeddings: dict, chunks: dict, similarity_threshold: float = 0.8) -> dict:
        try:
            grouped_chunks = {}
            processed_chunks = set()
            
            for chunk_id, embedding in chunk_embeddings.items():
                if chunk_id in processed_chunks:
                    continue
                    
                group = {
                    'main_chunk': ' '.join(chunks[chunk_id]['propositions']),
                    'related_chunks': []
                }
                
                # 유사한 청크 찾기
                for other_id, other_embedding in chunk_embeddings.items():
                    if other_id != chunk_id and other_id not in processed_chunks:
                        similarity = self._calculate_similarity(
                            np.array(embedding).flatten(),
                            np.array(other_embedding).flatten()
                        )
                        if similarity >= similarity_threshold:
                            group['related_chunks'].append(
                                ' '.join(chunks[other_id]['propositions']))
                            processed_chunks.add(other_id)
                
                grouped_chunks[f"group_{len(grouped_chunks)}"] = group
                processed_chunks.add(chunk_id)
            
            return grouped_chunks
            
        except Exception as e:
            self.logger.error(f"청크 그룹화 중 오류 발생: {str(e)}")
            return {}

    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """코사인 유사도 계산"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def _extract_domain_hierarchy(self, semantic_groups: dict, embeddings: dict) -> dict:
        """금융 도메인 기반 동적 계층 구조 추출"""
        try:
            # 1. 각 그룹의 중심 벡터와 대표 텍스트 준비
            group_info = {}
            for group_id, chunk_ids in semantic_groups.items():
                # 중심 벡터 계산
                group_embeddings = [embeddings[cid] for cid in chunk_ids]
                centroid = np.mean(group_embeddings, axis=0)
                
                # 대표 텍스트 선정 (중심과 가장 가까운 청크)
                distances = [cosine_similarity(centroid.reshape(1, -1), 
                                emb.reshape(1, -1))[0][0] 
                            for emb in group_embeddings]
                representative_chunk = chunk_ids[np.argmax(distances)]
                
                group_info[group_id] = {
                    'centroid': centroid,
                    'representative': representative_chunk,
                    'chunks': chunk_ids
                }
            
            # 2. 계층적 클러스터링으로 자연스러운 그룹 형성
            linkage_matrix = self._compute_hierarchical_clustering(
                [info['centroid'] for info in group_info.values()]
            )
            
            # 3. 최적의 클러스터 수 동적 결정
            optimal_clusters = self._find_optimal_clusters(linkage_matrix)
            
            # 4. 클러스터 간 관계 분석으로 계층 구조 생성
            hierarchy = self._build_dynamic_hierarchy(
                group_info, 
                linkage_matrix, 
                optimal_clusters
            )
            
            return hierarchy
        
        except Exception as e:
            self.logger.error(f"계층 구조 추출 중 오류 발생: {str(e)}")
            return {}

    def _compute_hierarchical_clustering(self, centroids: List[np.ndarray]) -> np.ndarray:
        """계층적 클러스터링 수행"""
        # Ward 방법으로 자연스러운 그룹화
        return linkage(centroids, method='ward')

    def _find_optimal_clusters(self, linkage_matrix: np.ndarray) -> int:
        """최적의 클러스터 수 결정"""
        # 1. 실루엣 점수로 클러스터 품질 평가
        silhouette_scores = []
        for n_clusters in range(2, 6):  # 2~5개 클러스터 시도
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            score = silhouette_score(linkage_matrix, labels)
            silhouette_scores.append((n_clusters, score))
        
        # 2. 엘보우 메소드로 급격한 변화 지점 찾기
        distances = linkage_matrix[:, 2]
        acceleration = np.diff(np.diff(distances))
        elbow_idx = np.argmax(acceleration) + 2
        
        # 3. 두 방법을 종합하여 최적 클러스터 수 결정
        optimal_n = max(silhouette_scores, key=lambda x: x[1])[0]
        return min(optimal_n, elbow_idx)  # 더 보수적인 값 선택

    def _build_dynamic_hierarchy(self, group_info: dict, 
                               linkage_matrix: np.ndarray, 
                               n_clusters: int) -> dict:
        """동적 계층 구조 생성"""
        # 1. 기본 클러스터 형성
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # 2. 클러스터 간 관계 분석
        cluster_relationships = self._analyze_cluster_relationships(
            group_info, 
            labels
        )
        
        # 3. 관계 그래프 생성 및 계층 구조 추출
        hierarchy = self._extract_hierarchy_from_relationships(
            cluster_relationships
        )
        
        return hierarchy

    def _analyze_cluster_relationships(self, group_info: dict, 
                                     labels: np.ndarray) -> nx.DiGraph:
        """클러스터 간 관계 분석"""
        # NetworkX 그래프로 관계 모델링
        G = nx.DiGraph()
        
        # 모든 클러스터 쌍에 대해 관계 분석
        unique_labels = np.unique(labels)
        for label1 in unique_labels:
            for label2 in unique_labels:
                if label1 != label2:
                    relationship = self._analyze_cluster_pair(
                        group_info, 
                        labels, 
                        label1, 
                        label2
                    )
                    if relationship > 0:
                        G.add_edge(label1, label2, weight=relationship)
        
        return G

    def _extract_hierarchy_from_relationships(self, G: nx.DiGraph) -> dict:
        """관계 그래프에서 계층 구조 추출"""
        # 1. 위상 정렬로 기본 계층 순서 결정
        hierarchy_order = list(nx.topological_sort(G))
        
        # 2. 각 노드의 깊이 계산
        node_depths = nx.shortest_path_length(G, source=hierarchy_order[0])
        
        # 3. 깊이를 기반으로 동적 계층 구조 생성
        hierarchy = defaultdict(list)
        for node, depth in node_depths.items():
            hierarchy[f"level_{depth}"].extend(node)
        
        return dict(hierarchy)

    def _analyze_with_vectors(self, chunk_embeddings: Dict[str, Dict]) -> List[Dict]:
        """KB-ALBERT 임베딩 벡터 기반 명사 위계 구조 분석"""
        try:
            from konlpy.tag import Okt
            okt = Okt()
            
            # 1. 데이터 준비 및 로깅
            embeddings_list = []
            self.logger.info("=== 데이터 처리 시작 ===")
            
            for chunk_id, data in chunk_embeddings.items():
                if isinstance(data, dict) and 'embedding' in data:
                    if isinstance(data['embedding'], np.ndarray) and data['embedding'].shape[0] == 768:
                        text = data.get('text', '') or ' '.join(data.get('propositions', []))
                        nouns = okt.nouns(text)
                        
                        self.logger.info(f"청크 {chunk_id}:")
                        self.logger.info(f"원본 텍스트: {text}")
                        self.logger.info(f"추출된 명사: {nouns}")
                        
                        if nouns:
                            embeddings_list.append({
                                'id': chunk_id,
                                'vector': data['embedding'],
                                'nouns': nouns,
                                'original_text': text  # 디버깅용
                            })

            # 2. 벡터 관계 분석 (기존 코드와 동일)
            n_vectors = len(embeddings_list)
            relation_matrix = np.zeros((n_vectors, n_vectors))
            
            for i in range(n_vectors):
                for j in range(n_vectors):
                    if i != j:
                        vec1 = embeddings_list[i]['vector']
                        vec2 = embeddings_list[j]['vector']
                        
                        cosine_sim = 1 - cosine(vec1, vec2)
                        magnitude_ratio = np.linalg.norm(vec1) / np.linalg.norm(vec2)
                        direction = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        
                        relation_score = (cosine_sim + direction) * magnitude_ratio
                        relation_matrix[i][j] = relation_score

            # 3. 위계 구조 생성 및 로깅
            hierarchy = []
            used_indices = set()
            
            self.logger.info("=== 위계 구조 생성 ===")
            
            for i in range(n_vectors):
                if i in used_indices:
                    continue
                    
                relations = relation_matrix[i]
                current_nouns = embeddings_list[i]['nouns']
                
                self.logger.info(f"현재 처리 중인 명사들: {current_nouns}")
                
                # 명사만 저장하도록 수정
                structure = {'d1': current_nouns[0] if current_nouns else ''}
                used_indices.add(i)
                
                related_indices = np.argsort(relations)[::-1]
                current_level = 2
                
                for idx in related_indices:
                    if idx in used_indices or relations[idx] < 0.5:
                        continue
                    
                    related_nouns = embeddings_list[idx]['nouns']
                    if related_nouns:
                        level_key = f'd{current_level}'
                        structure[level_key] = related_nouns[0]  # 첫 번째 명사만 사용
                        self.logger.info(f"{level_key}에 저장된 명사: {related_nouns[0]}")
                        used_indices.add(idx)
                        current_level += 1
                    
                    if current_level > 4:
                        break
                
                hierarchy.append(structure)
                self.logger.info(f"생성된 구조: {structure}")

            # 4. 결과 정규화 및 최종 로깅
            max_levels = max(len(struct) for struct in hierarchy)
            normalized_hierarchy = []
            
            self.logger.info("=== 최종 결과 ===")
            for struct in hierarchy:
                normalized = {}
                for i in range(1, max_levels + 1):
                    key = f'd{i}'
                    normalized[key] = struct.get(key, '')
                normalized_hierarchy.append(normalized)
                self.logger.info(f"정규화된 구조: {normalized}")

            return normalized_hierarchy

        except Exception as e:
            self.logger.error(f"벡터 구조화 중 오류 발생: {str(e)}")
            self.logger.error("오류 발생 지점:", exc_info=True)
            return []

    def _extract_nouns(self, text: str) -> List[str]:
        """텍스트에서 명사 추출"""
        try:
            from konlpy.tag import Okt
            okt = Okt()
            return okt.nouns(text)
        except Exception as e:
            self.logger.error(f"명사 추출 중 오류: {str(e)}")
            return []

class PDFPreprocessor:
    def __init__(self, pdf_path: str):
        # 로거 설정
        self.logger = logging.getLogger('PDFProcessor')
        self.logger.info("PDFPreprocessor 초기화 시작")
        
        # 기본 설정
        self.pdf_path = pdf_path
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.vector_store_dir = os.path.join(self.base_dir, "vector_store")
        
        # 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # OpenAI 설정
        load_dotenv()
        self.llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0
        )
        
        # 컴포넌트 초기화
        self.chunker = AgenticChunker()
        self.chunker.llm = self.llm  # GPT 모델 설정
        
        # KB-Albert 임베더 초기화
        kb_albert_path = os.path.join(self.base_dir, "KB-Albert")
        self.embedder = TextEmbedder(model_path=kb_albert_path)
        
        # 하이브리드 분석기 초기화
        self.analyzer = HybridAnalyzer(llm=self.llm, embedder=self.embedder)
        
        # 구글 시트 매니저 초기화
        self.sheets_manager = GoogleSheetsManager()
        
        self.logger.info("PDFPreprocessor 초기화 완료")

    def process_pdf(self):
        try:
            # 1. PDF 텍스트 추출
            self.logger.info("\n=== PDF 텍스트 추출 시작 ===")
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            text = ' '.join([page.page_content for page in pages])
            self.logger.info(f"PDF 페이지 수: {len(pages)}")
            self.logger.info(f"추출된 전체 텍스트 길이: {len(text)} 문자")
            self.logger.info("텍스트 추출 완료")
            
            # 2. 청킹
            self.logger.info("\n=== 청킹 프로세스 시작 ===")
            self.logger.info("텍스트를 청커에 입력 중...")
            self.chunker.add_text(text)
            
            self.logger.info("청크 생성 중...")
            chunks = self.chunker.get_chunks()
            self.logger.info(f"총 생성된 청크 수: {len(chunks)}개")
            
            # 청크 샘플 출력
            self.logger.info("\n=== 청크 샘플 ===")
            sample_size = min(3, len(chunks))
            for i, (chunk_id, chunk_data) in enumerate(list(chunks.items())[:sample_size]):
                self.logger.info(f"\n청크 {i+1}/{sample_size}")
                self.logger.info(f"청크 ID: {chunk_id}")
                self.logger.info(f"청크 길이: {len(chunk_data['text'])} 문자")
                self.logger.info(f"청크 내용 미리보기: {chunk_data['text'][:200]}...")
            
            self.logger.info("\n=== 청킹 프로세스 완료 ===")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"\n❌ PDF 처리 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def _get_file_hash(self):
        """PDF 일의 해시값 생성"""
        import hashlib
        with open(self.pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

def main():
    # PDF 파일 경로 지정
    pdf_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/doc/토스_가계대출_상품설명서.pdf"
    
    try:
        # PDFPreprocessor 초기화 및 실행
        preprocessor = PDFPreprocessor(pdf_path)
        preprocessor.process_pdf()
        
    except Exception as e:
        print(f"프로세스 실행 중 오류 발생: {str(e)}")
        print("프로세스를 중단합니다.")
        raise 