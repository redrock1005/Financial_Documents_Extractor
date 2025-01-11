import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 명시적으로 병렬 처리 비활성화

import sys
from dotenv import load_dotenv
from processors.pdf import PDFProcessor
from processors.chunk import ChunkProcessor
from embedders.kb_albert import KBAlbertEmbedder
from sklearn.cluster import KMeans
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI
from typing import List

class DocumentProcessor:
    def __init__(self):
        try:
            # 환경변수 로드
            load_dotenv('/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/.env')
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            
            # 프로세서 초기화 전에 토크나이저 설정 확인
            if "TOKENIZERS_PARALLELISM" not in os.environ:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # 프로세서 초기화
            self.pdf_processor = PDFProcessor()
            self.chunk_processor = ChunkProcessor()
            self.embedder = KBAlbertEmbedder()
            self.client = OpenAI(api_key=self.openai_api_key)
            
            print("DocumentProcessor 초기화 완료")
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {e}")
            raise

    def extract_entities(self, text):
        """GPT를 사용하여 엔터티 추출"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "당신은 금융 상품 설명서에서 중요한 정보를 추출하는 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"다음 텍스트에서 금융 상품을 구성하는 요소를 추출해주세요:\n{text}"
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"엔터티 추출 오류: {e}")
            return ""

    def update_google_sheet(self, entities: List[str], spreadsheet_id: str) -> bool:
        """구글 시트 업데이트"""
        try:
            credentials_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/credentials/google_sheets_credentials.json"
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            service = build('sheets', 'v4', credentials=credentials)
            values = [[entity] for entity in entities]
            body = {'values': values}
            
            try:
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range='A1',
                    valueInputOption='RAW',
                    body=body
                ).execute()
                print("구글 시트 업데이트 완료")
                return True
            except Exception as e:
                print(f"구글 시트 업데이트 오류: {e}")
                return False
        except Exception as e:
            print(f"구글 시트 업데이트 중 오류 발생: {e}")
            return False

    def process_document(self, pdf_path, spreadsheet_id):
        """전체 문서 처리 파이프라인"""
        try:
            # 1. PDF 존재 확인
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            print(f"1. PDF 파일 확인 완료: {pdf_path}")

            # 2. PDF 텍스트 추출
            text = self.pdf_processor.process(pdf_path)
            if not text:
                raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")
            print("2. PDF 텍스트 추출 완료")

            # 3. 청킹
            print("3. 청킹 시작")
            chunk_results = self.chunk_processor.process(text)
            if not chunk_results["chunks"]:
                raise ValueError("청킹 결과가 없습니다.")
            chunks = [chunk["text"] for chunk in chunk_results["chunks"]]
            print(f"   청킹 완료: {len(chunks)}개의 청크 생성")

            # 4. 임베딩
            print("4. 임베딩 시작")
            embeddings = self.embedder.batch_embed(chunks)
            if embeddings.size == 0:
                raise ValueError("임베딩 생성 실패")
            print("   임베딩 완료")

            # 5. 클러스터링
            print("5. 클러스터링 시작")
            n_clusters = min(5, len(chunks))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            print(f"   클러스터링 완료: {n_clusters}개의 클러스터 생성")

            # 6. 엔터티 추출
            print("6. 엔터티 추출 시작")
            all_entities = []
            for i, chunk in enumerate(chunks):
                print(f"   청크 {i+1}/{len(chunks)} 처리 중...")
                entities = self.extract_entities(chunk)
                if entities:
                    all_entities.append(entities)
            
            if not all_entities:
                raise ValueError("엔터티 추출 결과가 없습니다.")
            print("   엔터티 추출 완료")

            # 7. 구글 시트 업데이트
            print("7. 구글 시트 업데이트 시작")
            self.update_google_sheet(all_entities, spreadsheet_id)
            print("   구글 시트 업데이트 완료")

            return {
                "chunks": chunks,
                "cluster_labels": cluster_labels.tolist(),
                "entities": all_entities
            }

        except Exception as e:
            print(f"\n처리 중단: {str(e)}")
            return None

def main():
    # PDF 파일 경로
    pdf_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/doc/하나은행_가계대출_상품설명서.pdf"
    
    # 구글 스프레드시트 ID
    spreadsheet_id = "1PfejJaIkOCz3oMGidXUTmfZrOvNZk0a3c1hj450yots"
    
    try:
        # 문서 처리기 초기화
        processor = DocumentProcessor()
        
        print("\n=== 문서 처리 시작 ===")
        results = processor.process_document(pdf_path, spreadsheet_id)
        
        if results:
            print("\n=== 처리 결과 요약 ===")
            print(f"총 청크 수: {len(results['chunks'])}")
            print(f"총 추출된 엔터티 수: {len(results['entities'])}")
            
            # 엔터티 결과 출력
            print("\n=== 추출된 엔터티 ===")
            for i, entity in enumerate(results['entities'], 1):
                print(f"\n엔터티 그룹 {i}:")
                print(entity)
                
            print("\n처리 완료!")
        else:
            print("\n처리가 실패했습니다.")
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 