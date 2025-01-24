from typing import List, Dict, Any, Tuple
import pandas as pd
from openai import OpenAI
import json
import logging
import numpy as np
from transformers import AlbertModel, BertTokenizer
import torch
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# OpenMP 스레드 수를 1로 제한
os.environ["OMP_NUM_THREADS"] = "1"

class FinancialProductExtractor:
    def __init__(self, api_key: str):
        """금융 상품 정보 추출기 초기화"""
        self.client = OpenAI(api_key=api_key)
        
        # KB-ALBERT 모델 로컬 경로 설정
        model_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"
        
        try:
            # BertTokenizer 사용 (config.json에 명시된 대로)
            self.tokenizer = BertTokenizer.from_pretrained(
                model_path,
                do_lower_case=False,
                strip_accents=False
            )
            # 모델은 ALBERT 아키텍처 사용
            self.model = AlbertModel.from_pretrained(model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            print("모델 로드 완료:", model_path)
            
        except Exception as e:
            logging.error(f"모델 로드 중 오류 발생: {e}")
            raise
        
        self.chunk_prompt = "이 텍스트에서 중요한 엔터티들을 모두 추출해주세요."
        self.cluster_prompt = "이 엔터티 그룹을 대표하는 하나의 명사 또는 명사구를 추출해주세요."

    def _get_embeddings(self, text: str) -> np.ndarray:
        """KB-ALBERT를 사용하여 텍스트 임베딩 생성"""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, max_length=512, 
                                  padding=True).to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]

    def _extract_chunk_entities(self, chunk: str) -> List[str]:
        """청크에서 엔터티 추출"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 금융 문서에서 중요한 엔터티를 추출하는 전문가입니다.
                        다음 규칙을 반드시 따라주세요:
                        1. 금융 상품과 관련된 주요 용어만 추출
                        2. 각 엔터티는 명사 또는 명사구 형태로 추출
                        3. 추출된 엔터티는 JSON 배열 형식으로 반환
                        
                        예시 출력:
                        ["대출한도", "금리조건", "상환방식"]
                        """
                    },
                    {"role": "user", "content": f"다음 텍스트에서 중요 엔터티를 추출해주세요:\n\n{chunk}"}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # JSON 파싱 전처리
            result = result.replace("'", '"')  # 작은따옴표를 큰따옴표로 변환
            if not result.startswith('['):
                # JSON 배열이 아닌 경우, 줄바꿈으로 구분된 텍스트로 간주
                entities = [line.strip() for line in result.split('\n') if line.strip()]
                return entities
            
            try:
                entities = json.loads(result)
                if isinstance(entities, list):
                    return [str(entity).strip() for entity in entities if entity]
                else:
                    logging.warning(f"예상치 못한 응답 형식: {result}")
                    return []
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류: {e}\n응답: {result}")
                # 콤마로 구분된 텍스트로 시도
                entities = [e.strip() for e in result.strip('[]').split(',') if e.strip()]
                return entities
            
        except Exception as e:
            logging.error(f"엔터티 추출 오류: {str(e)}")
            return []

    def _get_contextual_entity_embedding(self, entity: str, chunk_embedding: np.ndarray) -> np.ndarray:
        """엔터티와 청크 임베딩을 결합하여 문맥화된 엔터티 임베딩 생성"""
        entity_embedding = self._get_embeddings(entity)
        # 청크 임베딩과 엔터티 임베딩을 결합 (가중치 조정 가능)
        return 0.7 * entity_embedding + 0.3 * chunk_embedding

    def _cluster_entities(self, entity_embeddings: List[np.ndarray], entities: List[str]) -> Dict[int, List[str]]:
        """엔터티들을 의미유사도 기반으로 클러스터링"""
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # 벡터 정규화
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(entity_embeddings)
        
        # 데이터 기반으로 최대 클러스터 수 결정
        n_samples = len(entities)
        # 엔터티 개수의 제곱근을 기준으로 최대 클러스터 수 설정
        max_clusters = max(2, int(np.sqrt(n_samples)))
        
        logging.info(f"전체 엔터티 수: {n_samples}, 시도할 최대 클러스터 수: {max_clusters}")
        
        best_score = -1
        best_n_clusters = 2
        
        # 최적의 클러스터 수 찾기
        for n_clusters in range(2, max_clusters + 1):
            if n_clusters >= n_samples:
                break
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                tol=0.0001
            )
            labels = kmeans.fit_predict(normalized_embeddings)
            
            # 실루엣 점수 계산
            if len(set(labels)) > 1:
                score = silhouette_score(normalized_embeddings, labels)
                logging.info(f"클러스터 수 {n_clusters}의 실루엣 점수: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
        
        # 최적의 클러스터 수로 최종 클러스터링
        logging.info(f"최적의 클러스터 수: {best_n_clusters} (실루엣 점수: {best_score:.4f})")
        
        final_kmeans = KMeans(
            n_clusters=best_n_clusters,
            random_state=42
        )
        cluster_labels = final_kmeans.fit_predict(normalized_embeddings)
        
        # 클러스터 중심과 각 포인트 간의 거리 계산
        distances = final_kmeans.transform(normalized_embeddings)
        
        # 클러스터링 결과 저장 (유사도가 높은 것들만 그룹화)
        clusters = {}
        similarity_threshold = 0.7  # 유사도 임계값 (조정 가능)
        
        for idx, (label, dist) in enumerate(zip(cluster_labels, distances)):
            # 해당 클러스터 중심까지의 거리를 유사도로 변환
            similarity = 1 / (1 + dist[label])
            
            if similarity >= similarity_threshold:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(entities[idx])
                logging.info(f"엔터티 '{entities[idx]}'의 클러스터 {label}에 대한 유사도: {similarity:.4f}")
        
        # 클러스터링 결과 로깅
        for label, cluster_entities in clusters.items():
            logging.info(f"클러스터 {label}: {len(cluster_entities)}개 엔터티")
            logging.info(f"클러스터 {label} 엔터티들: {cluster_entities}")
            
        return clusters

    def _extract_cluster_entity(self, entities: List[str]) -> str:
        """클러스터를 대표하는 엔터티 추출"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """당신은 금융 엔터티 그룹을 대표하는 상위 개념을 추출하는 전문가입니다.
                    입력으로 주어진 엔터티들을 포괄하는 하나의 명사 또는 명사구만을 반환해주세요.
                    예시:
                    입력: ["중도상환수수료", "대출이자", "연체이자율"]
                    출력: 대출비용
                    """
                },
                {"role": "user", "content": f"다음 엔터티들을 포괄하는 상위 개념을 추출해주세요:\n{', '.join(entities)}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def extract_factors(self, documents: List[str]) -> pd.DataFrame:
        """문서에서 금융 상품 요소 추출 및 구조화"""
        try:
            # 1. 청크 임베딩 생성 (타임아웃 추가)
            chunk_embeddings = []
            logging.info(f"총 {len(documents)}개 청크 임베딩 시작")
            for i, chunk in enumerate(documents):
                try:
                    embedding = self._get_embeddings(chunk)
                    chunk_embeddings.append(embedding)
                    logging.info(f"청크 {i+1} 임베딩 완료")
                except Exception as e:
                    logging.error(f"청크 {i+1} 임베딩 실패: {str(e)}")
                    continue

            if not chunk_embeddings:
                logging.error("임베딩된 청크가 없습니다")
                return pd.DataFrame(columns=["대분류", "중분류", "소분류", "상세"])

            # 2. 의미유사도 기반 클러스터링 (최대 반복 횟수 제한)
            try:
                clusters = self._cluster_chunks(np.array(chunk_embeddings), documents)
                logging.info(f"{len(clusters)}개의 클러스터 생성됨")
            except Exception as e:
                logging.error(f"클러스터링 실패: {str(e)}")
                return pd.DataFrame(columns=["대분류", "중분류", "소분류", "상세"])

            # 3. 클러스터별 청크 병합 및 엔터티 추출 (타임아웃 추가)
            cluster_entities = {}
            for cluster_id, chunk_group in clusters.items():
                try:
                    merged_chunk = "\n".join(chunk_group)
                    entities = self._extract_chunk_entities(merged_chunk)
                    if entities:  # 빈 리스트가 아닐 경우만 저장
                        cluster_entities[cluster_id] = entities
                    logging.info(f"클러스터 {cluster_id}: {len(entities)}개 엔터티 추출")
                except Exception as e:
                    logging.error(f"클러스터 {cluster_id} 처리 실패: {str(e)}")
                    continue

            # 4. 클러스터별 계층구조 추출 (타임아웃 추가)
            hierarchy_data = []
            for cluster_id, entities in cluster_entities.items():
                try:
                    cluster_concept = self._extract_cluster_entity(entities)
                    for entity in entities:
                        try:
                            hierarchy = self._determine_hierarchy(entity, cluster_concept)
                            hierarchy_data.append(hierarchy)
                        except Exception as e:
                            logging.error(f"엔터티 '{entity}' 계층구조 추출 실패: {str(e)}")
                            continue
                except Exception as e:
                    logging.error(f"클러스터 {cluster_id} 계층구조 추출 실패: {str(e)}")
                    continue

            # 5. DataFrame 생성
            result_df = pd.DataFrame(hierarchy_data)
            if result_df.empty:
                logging.warning("최종 DataFrame이 비어있습니다")
            else:
                logging.info(f"최종 DataFrame 생성 완료: {len(result_df)}개 행")

            return result_df

        except Exception as e:
            logging.error(f"전체 프로세스 실패: {str(e)}")
            return pd.DataFrame(columns=["대분류", "중분류", "소분류", "상세"])

    def _cluster_chunks(self, embeddings: np.ndarray, chunks: List[str]) -> Dict[int, List[str]]:
        """청크 임베딩을 기반으로 유사한 청크들을 클러스터링"""
        try:
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings)
            
            # 데이터 기반으로 최대 클러스터 수 결정 (단순화)
            n_samples = len(chunks)
            n_clusters = min(max(2, int(np.sqrt(n_samples))), n_samples - 1)
            
            # 단순화된 클러스터링
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                max_iter=300  # 최대 반복 횟수 제한
            )
            labels = kmeans.fit_predict(normalized_embeddings)
            
            # 클러스터별로 청크 그룹화
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(chunks[idx])
            
            return clusters
            
        except Exception as e:
            logging.error(f"클러스터링 중 오류 발생: {str(e)}")
            # 실패 시 각 청크를 개별 클러스터로 처리
            return {i: [chunk] for i, chunk in enumerate(chunks)}

    def _save_extraction_result(self, file_name: str, result: Dict[str, Any]) -> None:
        """추출 결과를 파일로 저장"""
        try:
            with open(f"extracted_{file_name}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"결과 저장 중 오류 발생: {e}")

    def _determine_hierarchy(self, entity: str, cluster_name: str) -> dict:
        """엔터티의 계층 구조를 동적으로 결정"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 금융 상품 용어의 계층 구조를 분석하는 전문가입니다.
                        주어진 엔터티를 적절한 계층으로 분류해주세요.
                        
                        규칙:
                        1. 대분류는 가장 포괄적인 개념 (예: 대출비용, 대출조건)
                        2. 중분류는 구체적인 항목 (예: 이자, 수수료)
                        3. 소분류는 세부 유형 (예: 중도상환수수료, 연체이자)
                        4. 상세는 부가 설명이나 구체적인 내용
                        
                        반환 형식:
                        {
                            "대분류": "가장 포괄적인 개념",
                            "중분류": "구체적 항목",
                            "소분류": "세부 유형",
                            "상세": "부가 설명"
                        }
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"다음 엔터티를 계층 구조로 분류해주세요.\n엔터티: {entity}\n현재 분류: {cluster_name}"
                    }
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            try:
                hierarchy = json.loads(result)
                return hierarchy
            except json.JSONDecodeError:
                logging.error(f"계층 구조 파싱 실패: {result}")
                return {
                    "대분류": cluster_name,
                    "중분류": entity,
                    "소분류": "",
                    "상세": ""
                }
            
        except Exception as e:
            logging.error(f"계층 구조 결정 중 오류 발생: {str(e)}")
            return {
                "대분류": cluster_name,
                "중분류": entity,
                "소분류": "",
                "상세": ""
            } 