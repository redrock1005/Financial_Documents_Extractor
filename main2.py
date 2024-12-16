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

    def __init__(self, embedder: TextEmbedder, llm: ChatOpenAI):
        self.logger = logging.getLogger('HybridAnalyzer')
        self.embedder = embedder
        self.llm = llm

    def _extract_finance_terms(self, text: str) -> dict:
        """KB-ALBERT를 사용하여 금융 용어와 관계 추출"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.kb_model(**inputs)
            
            # KB-ALBERT의 출력을 사용하여 금융 용어 추출
            finance_terms = {
                "extracted_terms": self._process_model_outputs(outputs, text)
            }
            
            return finance_terms
            
        except Exception as e:
            logging.error(f"금융 용어 추출 중 오류 발생: {str(e)}")
            return {"extracted_terms": []}

    def _process_model_outputs(self, outputs, text: str) -> list:
        """모델 출력 처리 및 금융 용어 추출"""
        # 여기에 KB-ALBERT 출력 처리 로직 구현
        # 예: 특정 임계값 이상의 활성화를 보이는 토큰 추출
        return []

    def _structure_with_gpt(self, finance_terms: dict, text: str) -> pd.DataFrame:
        """GPT-4를 사용하여 추출된 용어를 구조화"""
        try:
            # prompt 변수 정의 추가
            prompt = f"""당신은 은행의 금융 상품 분석 전문가입니다. 
            주어진 대출 상품 설명서를 분석하여 상품의 속성들을 체계적으로 분류해주세요.

            분석할 텍스트:
            {text}

            다음 규칙을 반드시 준수해주세요:
            1. 모든 속성은 명사형으로 작성
            2. 최대 5단계의 위계로 구분하여 분류 (구분/대분류/중분류/소분류/상세)
            3. 각 단계별 분류는 다음과 같이 구성:
               - 구분: 가장 상위 개념 (예: 대출종류, 금리유형, 상환방식 등)
               - 대분류: 구분의 주요 카테고리
               - 중분류: 대분류의 세부 카테고리
               - 소분류: 구체적인 속성이나 조건
               - 상세: 구체적인 수치나 세부 내용

            4. 분류 예시:
               예시 1)
               - 구분: 금리
               - 대분류: 우대금리
               - 중분류: 조건
               - 소분류: 금리 인하 쿠폰
               - 상세: 적용금리 0.2%

               예시 2)
               - 구분: 금리
               - 대분류: 우대금리
               - 중분류: 최대 금리 제한
               - 소분류: 최대 2.0%이하
               - 상세: (없음)

            5. 금융 용어와 관계:
            {json.dumps(finance_terms, ensure_ascii=False, indent=2)}

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

            주의사항:
            1. 상세 항목이 없는 경우 "(없음)"으로 표시
            2. 모든 금리 관련 수치는 정확히 표기
            3. 조건과 한도는 구체적으로 명시
            """
            
            self.logger.info("GPT 구조화 시작...")
            response = self.llm.invoke(prompt)
            
            # JSON 파싱
            result = json.loads(response.content)
            df = pd.DataFrame(result['classified'])
            self.logger.info(f"구조화 완료. 데이터프레임 크기: {df.shape}")
            return df
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {str(e)}")
            self.logger.error(f"GPT 응답: {response.content}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"구조화 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def analyze_structure(self, chunks: dict) -> pd.DataFrame:
        """임베딩 기반 구조화 분석"""
        try:
            self.logger.info("=== 임베딩 기반 구조화 시작 ===")
            
            # 1. 입력 데이터 검증
            if not chunks:
                self.logger.error("빈 chunks 딕셔너리가 입력되었습니다")
                return pd.DataFrame()
            
            # 2. 임베딩 벡터와 텍스트 준비
            chunk_embeddings = {}
            for chunk_id, chunk_data in chunks.items():
                if isinstance(chunk_data, dict):
                    text = chunk_data.get('text', '') or ' '.join(chunk_data.get('propositions', []))
                    embedding = self.embedder.embed_text(text)
                    if embedding is not None:
                        chunk_embeddings[chunk_id] = {
                            'embedding': embedding,
                            'text': text
                        }

            # 3. 벡터 기반 분석 실행
            structured_data = self._analyze_with_vectors(chunk_embeddings)  # 여기서 호출
            
            # 4. DataFrame 변환
            if structured_data:
                df = pd.DataFrame(structured_data)
                self.logger.info(f"구조화 완료. 데이터프레임 크기: {df.shape}")
                return df
            else:
                self.logger.error("구조화된 데이터가 없습니다")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"구조화 중 오류 발생: {str(e)}")
            self.logger.error("오류 발생 지점:", exc_info=True)
            return pd.DataFrame()

    def _discover_hierarchical_structure(self, embeddings: Dict[str, np.ndarray], chunks: dict) -> List[Dict]:
        """고차원 임베딩 벡터의 의미 구조를 유지하며 계층적 구조 발견"""
        try:
            self.logger.info("계층 구조 발견 시작")
            
            vectors = list(embeddings.values())
            chunk_ids = list(embeddings.keys())
            
            # 텍스트 추출 헬퍼 함수
            def get_chunk_text(chunk_data):
                if 'text' in chunk_data:
                    return chunk_data['text']
                elif 'propositions' in chunk_data:
                    return ' '.join(chunk_data['propositions'])
                return ''
            
            # 청크 데이터 검증
            for chunk_id in chunk_ids:
                if chunk_id not in chunks:
                    raise ValueError(f"임베딩 ID {chunk_id}에 해당하는 청크를 찾을 수 없습니다")
                if not isinstance(chunks[chunk_id], dict):
                    raise ValueError(f"청크 {chunk_id}가 딕셔너리가 아닙니다")
                if 'text' not in chunks[chunk_id] and 'propositions' not in chunks[chunk_id]:
                    raise ValueError(f"청크 {chunk_id}에 텍스트 데이터가 없습니다")

            # 1. 벡터 간 유사도 행렬 계산 (원본 차원 유지)
            similarity_matrix = np.zeros((len(vectors), len(vectors)))
            for i in range(len(vectors)):
                for j in range(len(vectors)):
                    # numpy의 dot product 사용하여 코사인 유사도 계산
                    dot_product = np.dot(vectors[i].flatten(), vectors[j].flatten())
                    norm_i = np.linalg.norm(vectors[i])
                    norm_j = np.linalg.norm(vectors[j])
                    similarity = dot_product / (norm_i * norm_j)
                    similarity_matrix[i][j] = similarity
            
            # 2. 계층적 의미 구조 발견
            hierarchy = []
            used_indices = set()
            
            # 2.1 최상위 계층(d1) 구성
            d1_groups = []
            for i in range(len(vectors)):
                if i in used_indices:
                    continue
                    
                # 유사도가 높은 벡터들을 그룹화
                current_group = []
                for j in range(len(vectors)):
                    if similarity_matrix[i][j] > 0.8:  # d1 레벨 유사도 임계값
                        current_group.append(j)
                        used_indices.add(j)
                
                if current_group:
                    d1_groups.append(current_group)
            
            # 2.2 각 d1 그룹 내에서 하위 계층 구성
            for d1_group in d1_groups:
                # d1 레벨의 대표 벡터 선정
                d1_similarities = [
                    np.mean([similarity_matrix[i][j] for j in d1_group])
                    for i in d1_group
                ]
                d1_representative = d1_group[np.argmax(d1_similarities)]
                
                # 하위 계층 구성을 위한 임시 구조
                sub_hierarchy = {
                    'd1': get_chunk_text(chunks[chunk_ids[d1_representative]])[:100],
                    'members': {}
                }
                
                # 남은 멤버들의 계층 구조 구성
                remaining = [i for i in d1_group if i != d1_representative]
                current_level = 2
                
                while remaining:
                    level_key = f'd{current_level}'
                    level_groups = []
                    used_in_level = set()
                    
                    # 현재 레벨의 그룹 구성
                    for i in remaining:
                        if i in used_in_level:
                            continue
                            
                        current_group = []
                        for j in remaining:
                            if similarity_matrix[i][j] > 0.6:  # 하위 레벨 유사도 임계값
                                current_group.append(j)
                                used_in_level.add(j)
                        
                        if current_group:
                            level_groups.append(current_group)
                    
                    if not level_groups:
                        break
                    
                    # 각 그룹의 대표 텍스트 선정
                    sub_hierarchy['members'][level_key] = []
                    for group in level_groups:
                        group_similarities = [
                            np.mean([similarity_matrix[i][j] for j in group])
                            for i in group
                        ]
                        representative = group[np.argmax(group_similarities)]
                        sub_hierarchy['members'][level_key].append(
                            get_chunk_text(chunks[chunk_ids[representative]])[:100]
                        )
                    
                    remaining = [i for i in remaining if i not in used_in_level]
                    current_level += 1
                
                hierarchy.append(sub_hierarchy)
            
            # 3. 최종 데이터 구조화
            structured_data = []
            max_depth = max(
                max(len(h['members']) for h in hierarchy) if hierarchy else 0,
                1
            ) + 1
            
            for h in hierarchy:
                row = {'d1': h['d1']}
                for level in range(2, max_depth + 1):
                    level_key = f'd{level}'
                    row[level_key] = ' | '.join(h['members'].get(level_key, ['']))
                structured_data.append(row)
            
            self.logger.debug(f"발견된 구조: {len(structured_data)} 행, {max_depth} 열")
            return structured_data
            
        except Exception as e:
            self.logger.error(f"계층 구조 발견 중 오류 발생: {str(e)}")
            self.logger.error("오류 발생 지점:", exc_info=True)
            return []

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
    def __init__(self):
        # 로깅 설정
        self._setup_logging()
        
        # 환경 설정
        self.base_dir = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory"
        self.pdf_path = os.path.join(self.base_dir, "doc", "토스뱅크_가계대출_상품설명서.pdf")
        self.target_page = 1
        
        # KB-ALBERT 모델 경로
        self.model_path = os.path.join(self.base_dir, "KB-Albert")
        
        # 기본 청커 초기화
        self.chunker = AgenticChunker()
        self.chunker.print_logging = True
        self.chunker.generate_new_metadata_ind = True
        
        # KB-ALBERT 기반 임베더 초기화
        self.embedder = TextEmbedder(model_path=self.model_path)
        
        # GPT-4 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0
        )
        
        # 하이브리드 분석기 초기화 - embedder 전달
        self.analyzer = HybridAnalyzer(
            embedder=self.embedder,  # 기존 embedder 재사용
            llm=self.llm
        )
        
        # FAISS 인덱스 디렉토리 설정
        self.index_dir = os.path.join(self.base_dir, "vector_store")
        os.makedirs(self.index_dir, exist_ok=True)
        
        # 청크 ID와 인덱스 매핑을 위한 딕셔너리
        self.id_to_index = {}
        self.index_to_id = {}

    def analyze_financial_structure(self, chunks: Dict) -> pd.DataFrame:
        """금융 상품 속성 분석 및 위계 구조 생성"""
        try:
            # 모든 청크의 텍스트 결합
            all_text = " ".join([chunk_data.get('text', '') for chunk_data in chunks.values()])
            
            # 하이브리드 분석기를 사용하여 구조 분석
            return self.analyzer.analyze_structure(all_text)
                
        except Exception as e:
            self.logger.error(f"금융 상품 구조 분석 중 오류 발생: {str(e)}")
            raise

    def _custom_find_relevant_chunk(self, proposition):
        """커스텀 청킹 로직"""
        current_chunk_outline = self.chunker.get_chunk_outline()

        CUSTOM_PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                금융 상품 설명서의 텍스트를 의미 단위로 청킹하는 전문가입니다.
                다음 규칙에 따라 청크를 생성하거나 기존 청크에 추가하세요:

                1. 동일한 주제나 조건에 대한 설명은 반드시 같은 청크로 묶어야 합니다.
                2. 대출 한도, 금리, 상환 방식과 같은 핵심 조건들은 관련 내용을 모두 포함하여 하나의 청크로 만드세요.
                3. 체크박스(□) 항목이 연속되는 경우, 같은 주제의 체크박스들은 하나의 청크로 묶으세요.
                4. 제목과 그에 해당하는 상세 설명은 반드시 같은 청크에 포함해야 합니다.
                5. 날짜, 기간, 금액 정보는 반드시 관련 조건이나 설명과 함께 묶어야 합니다.
                6. 하나의 청크는 반드시 1개 이상의 문장으로 구성되어야 합니다. 문장은 마침표, 물음표, 느낌표를 기준으로
                해당 문장부호 앞에 한글 문자가 입력되어야 합니다. 예를 들어, "~하는데 6."은 마침표로 끝나지만 마침표 앞에 숫자가 입력되어 있어 문장이 아닙니다.
                7. 입력된 텍스트가 기존 청크의 주제나 조건과 관련이 있다면 해당 청크 ID를 반환하고,
                새로운 주제나 조건이라면 "No chunks"를 반환하세요.

                응답 형식:
                {
                    "chunk_id": "청크ID 또는 No chunks"
                }
                """,
            ),
            ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
            ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
        ])

        # 구조화된 출력을 위한 함수 정의
        def get_chunk_id(text: str) -> dict:
            return {
                "chunk_id": str
            }

        try:
            # 구조화된 출력 사용
            response = self.chunker.llm.with_structured_output(get_chunk_id).invoke(
                CUSTOM_PROMPT.format_messages(
                    current_chunk_outline=current_chunk_outline,
                    proposition=proposition
                )
            )

            chunk_found = response.get("chunk_id", "No chunks")

            if chunk_found == "No chunks" or len(chunk_found) != self.chunker.id_truncate_limit:
                return None

            return chunk_found

        except Exception as e:
            self.logger.error(f"청크 ID 추출 중 오류 발생: {str(e)}")
            return None

    def _clean_text(self, text: str) -> list:
        """텍스트 정제 및 초기 분할"""
        self.logger.debug("원본 텍스트:\n%s", text)
        
        # 1. 기본 정제
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 줄조 표시 보존
        reference_pattern = r'(\[[0-9]+\.[^\]]*\])'
        references = {}
        for i, match in enumerate(re.finditer(reference_pattern, text)):
            key = f'__REF_{i}__'
            references[key] = match.group(1)
            text = text.replace(match.group(1), key)
        
        # 3. 줄바꿈 처리
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'([가-힣])\s*\n\s*([가-힣])', r'\1\2', text)
        text = re.sub(r'([가-힣])\s*\n\s*([.!?※])', r'\1\2', text)
        
        # 4. 의미 단위로 분할
        segments = []
        
        # 4.1 체크박스 항목 분리
        checkbox_pattern = r'(□[^□\n]+?(?:예|아니오|동의|미동의|대상|비대상)[^□\n]*)'
        parts = re.split(checkbox_pattern, text)
        
        for part in parts:
            if re.match(checkbox_pattern, part):
                segments.append(part.strip())
            else:
                # 문장 단으로 분할 (마침표, 느낌표, 물음표로 끝나는 경우)
                sentences = re.split(r'([.!?])\s+(?=[가-힣\[A-Za-z])', part)
                current_sentence = ''
                for i in range(0, len(sentences)-1, 2):
                    if sentences[i].strip():
                        current_sentence = sentences[i].strip() + sentences[i+1]
                        if len(current_sentence) > 10:
                            segments.append(current_sentence)
        
        # 5. 참조 표시 복원
        segments = [
            re.sub(r'__REF_(\d+)__', lambda m: references[f'__REF_{m.group(1)}__'], seg)
            for seg in segments
        ]
        
        # 6. 최종 정제
        segments = [
            re.sub(r'\s+', ' ', seg).strip()
            for seg in segments 
            if len(seg.strip()) > 10
        ]
        
        self.logger.debug("분할된 세그먼트:\n%s", '\n'.join(segments))
        return segments

    def _initialize_faiss_index(self, dimension):
        """FAISS 인덱스 초기화"""
        # L2 거리를 사용하는 인덱스 생성
        self.index = faiss.IndexFlatL2(dimension)
        self.current_index = 0

    def _save_faiss_index(self):
        """FAISS 인덱스 저장"""
        index_path = os.path.join(self.index_dir, "faiss_index.index")
        mapping_path = os.path.join(self.index_dir, "id_mapping.json")
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, index_path)
        
        # ID 매핑 저장
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"FAISS 인덱스가 {index_path}에 저장되었습니다.")
        self.logger.info(f"ID 매핑이 {mapping_path}에 저장되었습니다.")

    def save_to_sheets(self, df: pd.DataFrame):
        """데이터프레임을 구글 시트에 저장"""
        try:
            # 파일명 수정
            credentials_path = os.path.join(self.base_dir, "credentials", "google_sheets_credentials.json")
            
            self.logger.info("=== 구글 시트 저장 시도 ===")
            self.logger.info(f"데이터프레임 크기: {df.shape}")
            self.logger.info(f"Credentials 경로: {credentials_path}")
            
            if not os.path.exists(credentials_path):
                self.logger.error(f"인증 파일을 찾을 수 없습니다: {credentials_path}")
                return
            
            # 시트 ID
            sheet_id = "1PfejJaIkOCz3oMGidXUTmfZrOvNZk0a3c1hj450yots"
            
            sheets_manager = GoogleSheetsManager(credentials_path=credentials_path)
            
            # 데이터프레임을 2D 리스트로 변환
            values = [df.columns.tolist()]  # 헤더
            values.extend(df.values.tolist())  # 데이터
            
            self.logger.info(f"저장할 데이터 행 수: {len(values)}")
            
            # 시트 업데이트
            sheets_manager.update_values(sheet_id, "'시트1'!A1", values)
            
            self.logger.info("구글 시트에 데이터 저장 완료!")
            self.logger.info(f"시트 URL: https://docs.google.com/spreadsheets/d/{sheet_id}")
            
        except Exception as e:
            self.logger.error(f"구글 시트 저장 중 오류 발생: {str(e)}")

    def process_pdf(self):
        try:
            cache_dir = os.path.join(self.base_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            pdf_hash = self._get_file_hash()
            cache_file = os.path.join(cache_dir, f"{pdf_hash}.json")
            
            if os.path.exists(cache_file):
                self.logger.info("=== 캐시된 청킹 결과 로드 중 ===")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 캐시 데이터 구조 확인을 위한 로깅 추가
                self.logger.info("캐시 데이터 구조:")
                self.logger.info(f"캐시 데이터 키: {cached_data.keys()}")
                
                chunks = cached_data['chunks']
                # 청크 데이터 구조 확인
                sample_chunk = next(iter(chunks.items()))
                self.logger.info(f"샘플 청크 구조: {sample_chunk}")
                
                # 청크 데이터 검증
                if not all('text' in chunk for chunk in chunks.values()):
                    self.logger.error("일부 청크에 'text' 키가 없습니다")
                    # 청크 구조 상세 로깅
                    for chunk_id, chunk_data in chunks.items():
                        self.logger.info(f"청크 {chunk_id} 키: {chunk_data.keys()}")
                
                # HybridAnalyzer를 통한 구조화 실행
                self.logger.info("=== 임베딩 기반 구조화 시작 ===")
                classified_df = self.analyzer.analyze_structure(chunks)
                
                # 구조화 결과 증
                if classified_df.empty:
                    self.logger.error("구조화 결과가 비어있습니다. 구글 시트 저장을 건너뜁니다.")
                    return None, None
                
                # 구글 시트에 저장
                self.logger.info("=== 구글 시트 저장 시작 ===")
                self.save_to_sheets(classified_df)
                
                return chunks, classified_df
                
            self.logger.info("캐시 파일이 없어 전체 프로세스를 실행합니다.")
            return self._process_pdf_internal()
            
        except Exception as e:
            self.logger.error(f"프로세스 실행 중 오류 발생: {str(e)}", exc_info=True)
            self.logger.error("프로세스를 중단합니다.")
            return None, None

    def _get_file_hash(self):
        """PDF 일의 해시값 생성"""
        import hashlib
        with open(self.pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def _process_pdf_internal(self):
        """PDF 전처리 및 청킹 실행"""
        try:
            self.logger.info(f"PDF 처리 시작: {self.pdf_path}")
            
            # PDF 로딩
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load_and_split()
            
            if self.target_page > len(pages):
                raise ValueError(f"지정한 페이지({self.target_page})가 PDF 총 페이지 수({len(pages)})를 초과합니다.")
            
            # 1페이지만 처리
            page = pages[self.target_page - 1]
            self.logger.info(f"페이지 {self.target_page} 처리 중...")
            
            # 텍스트 전처리 및 세그먼트 분할
            segments = self._clean_text(page.page_content)
            self.logger.info(f"전처리 완료. {len(segments)}개의 세그먼트 생성")
            
            # 청킹 수행
            self.logger.info("청킹 시작...")
            self.chunker.add_propositions(segments)
            
            # 청킹 결과 출력
            self.logger.info("\n=== 청킹 결과 ===")
            self.chunker.pretty_print_chunks()
            
            # 임베딩 생성 및 FAISS 인덱스에 저장
            chunks = self.chunker.get_chunks(get_type='dict')
            # 청크 구조 통일
            for chunk_id, chunk_data in chunks.items():
                if 'propositions' in chunk_data:
                    chunk_data['text'] = ' '.join(chunk_data['propositions'])
            
            embeddings = {}
            for chunk_id, chunk_data in chunks.items():
                chunk_text = chunk_data.get('text', '')
                embedding = self.embedder.embed_text(chunk_text)
                embeddings[chunk_id] = embedding
                
                # 첫 번째 임베딩으로 FAISS 인덱스 초기화
                if not hasattr(self, 'index'):
                    self._initialize_faiss_index(embedding.shape[1])
                
                # FAISS 인덱스에 임베딩 추가
                self.index.add(embedding)
                
                # ID 매핑 업데이트
                self.id_to_index[chunk_id] = self.current_index
                self.index_to_id[str(self.current_index)] = chunk_id
                self.current_index += 1
            
            # FAISS 인덱스 저장
            self._save_faiss_index()
            
            # 결과 저장
            output_dir = os.path.join(self.base_dir, "chunks")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir, 
                f"chunks_page_{self.target_page}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'source_file': self.pdf_path,
                        'page_number': self.target_page,
                        'processing_date': datetime.now().isoformat()
                    },
                    'segments': segments,
                    'chunks': chunks
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"청킹 결과 {output_file}에 저장되었습니다.")
            
            # 금융 상품 구조 분석
            classified_df = self.analyze_financial_structure(chunks)
            
            # 구글 시트에 저장
            self.save_to_sheets(classified_df)
            
            return chunks, classified_df

        except Exception as e:
            self.logger.error(f"처리 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def query_similar_chunks(self, query_text, k=5):
        """유사한 청크 검색"""
        query_embedding = self.embedder.embed_text(query_text)
        
        # FAISS로 유사한 벡터 검색
        distances, indices = self.index.search(query_embedding, k)
        
        # 결과 변환
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            chunk_id = self.index_to_id[str(idx)]
            results.append({
                'chunk_id': chunk_id,
                'distance': float(distance)
            })
        
        return results

    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger('PDFPreprocessor')
        self.logger.setLevel(logging.DEBUG)
        
        # 로그 파일 핸들러
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

def main():
    preprocessor = PDFPreprocessor()
    preprocessor.process_pdf()

if __name__ == "__main__":
    main() 